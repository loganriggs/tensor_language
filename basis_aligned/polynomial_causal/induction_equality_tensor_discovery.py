#!/usr/bin/env python3
"""Sequential fixed-equality extraction/removal assay for four induction heads."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for root in (ROOT, HERE, BQ):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade
import circuit_induction_tensor as induction


ROWS_RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
SELECTION_ROWS = BQ / ".rowcache_terminal_copy_induction_v2/selection_natural.pt"
PREREGISTRATION = HERE / "INDUCTION_EQUALITY_TENSOR_DISCOVERY_PREREGISTRATION.md"
OUTPUT = HERE / "induction_equality_tensor_discovery.json"
SELECTED = {5: (5,), 7: (3,), 8: (3, 4)}
ARMS = (
    "native", "full_replay", "remove_equality", "heads_deleted",
    "extract_equality", "deranged_equality",
)
CELLS = ("positive", "matched_negative", "off_target", "all")
D = 1152
HEADS = 9
HEAD_DIM = 128
BATCH = 4
SCORING = slice(64, 256)
VOCABULARY = 50_304
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_083_011


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _linear(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(state, weight.to(dtype=state.dtype, device=state.device))


@torch.no_grad()
def replay_attention_site(
    state: torch.Tensor,
    first_value: torch.Tensor,
    attention: torch.nn.Module,
    selected_heads: tuple[int, ...],
    tokens: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Analytically replay a site and replace only registered head contractions."""

    batch, length, width = state.shape
    if width != D or first_value.shape != (batch, length, HEADS, HEAD_DIM):
        raise ValueError("attention replay interface changed")
    q = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(
        batch, length, HEADS, HEAD_DIM,
    )
    value = (1 - attention.lamb) * raw_value + attention.lamb * first_value.view_as(raw_value)
    cos, sin = attention.rotary(q)
    module = sys.modules[type(attention).__module__]
    q = module.apply_rotary_emb(F.rms_norm(q, (HEAD_DIM,)), cos, sin)
    k = module.apply_rotary_emb(F.rms_norm(k, (HEAD_DIM,)), cos, sin)
    q2 = module.apply_rotary_emb(F.rms_norm(q2, (HEAD_DIM,)), cos, sin)
    k2 = module.apply_rotary_emb(F.rms_norm(k2, (HEAD_DIM,)), cos, sin)
    score = torch.einsum("bqhd,bkhd->bhqk", q, k) / HEAD_DIM
    score2 = torch.einsum("bqhd,bkhd->bhqk", q2, k2) / HEAD_DIM
    pattern = score * score2
    causal = torch.tril(torch.ones(
        length, length, dtype=torch.bool, device=state.device,
    ))
    pattern = pattern.masked_fill(~causal, 0)
    full = torch.einsum("bhqk,bkhd->bhqd", pattern, value)

    per_arm = {arm: full.clone() for arm in ARMS[1:]}
    for head in selected_heads:
        head_score = pattern[:, head]
        head_value = value[:, :, head]
        equality = induction.contract_induction_fetch(head_score, head_value, tokens)
        deranged = induction.contract_induction_fetch(
            head_score, head_value, tokens,
            vocabulary_offset=1, vocabulary_size=VOCABULARY,
        )
        per_arm["remove_equality"][:, head] = full[:, head] - equality
        per_arm["heads_deleted"][:, head] = 0
        per_arm["extract_equality"][:, head] = equality
        per_arm["deranged_equality"][:, head] = deranged

    writes = {}
    for arm, heads in per_arm.items():
        flattened = heads.transpose(1, 2).contiguous().view(batch, length, width)
        writes[arm] = _linear(flattened, attention.c_proj.weight)
    return writes, raw_value


def empty_ledger():
    return {
        arm: {
            cell: {"loss_sum": [], "kl_sum": [], "top1_changes": [], "count": []}
            for cell in CELLS
        }
        for arm in ARMS
    }


def append_documents(ledger, arm, masks, loss, kl, top1_change):
    for cell, mask in masks.items():
        for document in range(len(mask)):
            selected = mask[document]
            count = int(selected.sum())
            ledger[arm][cell]["count"].append(count)
            ledger[arm][cell]["loss_sum"].append(float(loss[document, selected].sum()))
            ledger[arm][cell]["kl_sum"].append(float(kl[document, selected].sum()))
            ledger[arm][cell]["top1_changes"].append(float(top1_change[document, selected].sum()))


def pooled_reports(ledger):
    output = {}
    for arm in ARMS:
        output[arm] = {}
        for cell in CELLS:
            count = sum(ledger[arm][cell]["count"])
            output[arm][cell] = {
                "tokens": count,
                "ce": sum(ledger[arm][cell]["loss_sum"]) / max(count, 1),
                "native_to_arm_kl": sum(ledger[arm][cell]["kl_sum"]) / max(count, 1),
                "top1_change_fraction": sum(ledger[arm][cell]["top1_changes"]) / max(count, 1),
            }
    return output


def bootstrap_effects(ledger):
    documents = len(ledger["native"]["all"]["count"])
    arrays = {}
    for arm in ARMS:
        for cell in CELLS:
            arrays[(arm, cell)] = (
                torch.tensor(ledger[arm][cell]["loss_sum"], dtype=torch.float64),
                torch.tensor(ledger[arm][cell]["count"], dtype=torch.float64),
            )

    def ce(arm, cell, indices):
        sums, counts = arrays[(arm, cell)]
        return sums[indices].sum(-1) / counts[indices].sum(-1).clamp_min(1)

    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    names = (
        "target_damage", "specificity", "off_target_damage",
        "extraction_recovery", "deranged_recovery",
    )
    draws = {name: [] for name in names}
    for start in range(0, BOOTSTRAP_DRAWS, 500):
        count = min(500, BOOTSTRAP_DRAWS - start)
        indices = torch.randint(documents, (count, documents), generator=generator)
        target = ce("remove_equality", "positive", indices) - ce("native", "positive", indices)
        negative = ce("remove_equality", "matched_negative", indices) - ce(
            "native", "matched_negative", indices,
        )
        off = ce("remove_equality", "off_target", indices) - ce("native", "off_target", indices)
        stake = ce("heads_deleted", "positive", indices) - ce("native", "positive", indices)
        recovery = (
            ce("heads_deleted", "positive", indices) - ce("extract_equality", "positive", indices)
        ) / stake.clamp_min(1e-12)
        deranged = (
            ce("heads_deleted", "positive", indices) - ce("deranged_equality", "positive", indices)
        ) / stake.clamp_min(1e-12)
        for name, value in zip(names, (target, target - negative, off, recovery, deranged)):
            draws[name].append(value)
    output = {}
    for name, chunks in draws.items():
        values = torch.cat(chunks).sort().values
        output[name] = {
            "mean": float(values.mean()),
            "bootstrap_95_low": float(values[math.floor(0.025 * (len(values) - 1))]),
            "bootstrap_95_high": float(values[math.ceil(0.975 * (len(values) - 1))]),
        }
    return output


@torch.no_grad()
def main() -> None:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError(f"induction equality namespace already exists: {OUTPUT}")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    entry = receipt["entries"]["selection_natural"]
    if entry["path"] != str(SELECTION_ROWS.resolve()) or file_sha256(SELECTION_ROWS) != entry[
        "file_sha256"
    ]:
        raise RuntimeError("selection row binding changed")
    bundle = torch.load(SELECTION_ROWS, map_location="cpu", weights_only=True)
    rows = bundle["rows"]
    records = bundle["records"]
    cells = bundle["copy_cells"]
    masks_cpu = {
        "positive": cells["positive"],
        "matched_negative": cells["matched_negative"],
        "off_target": cells["off_target"],
        "all": torch.zeros_like(cells["positive"]),
    }
    masks_cpu["all"][:, SCORING] = True
    if rows.shape != (192, 257) or len(records) != 192:
        raise RuntimeError("selection role schema changed")
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)

    current = {"arm": None}
    counters = {
        arm: {
            f"L{layer}:{name}": 0
            for layer in SELECTED for name in ("attention", "q", "k", "q2", "k2", "v", "o")
        }
        for arm in ARMS
    }

    def hook(layer, name):
        def count(_module, _inputs, _output):
            arm = current["arm"]
            if arm is None:
                raise RuntimeError("selected attention call escaped registered arm")
            counters[arm][f"L{layer}:{name}"] += 1
        return count

    handles = []
    for layer in SELECTED:
        attention = model.transformer.h[layer].attn
        handles.append(attention.register_forward_hook(hook(layer, "attention")))
        for name, module in (
            ("q", attention.c_q), ("k", attention.c_k), ("q2", attention.c_q2),
            ("k2", attention.c_k2), ("v", attention.c_v), ("o", attention.c_proj),
        ):
            handles.append(module.register_forward_hook(hook(layer, name)))

    ledger = empty_ledger()
    replay_max_logit_error = 0.0
    replay_kl_sum = 0.0
    replay_tokens = 0
    cell_documents = {cell: set() for cell in CELLS}
    try:
        for start in range(0, len(rows), BATCH):
            batch_rows = rows[start:start + BATCH]
            tokens = batch_rows[:, :-1].to(device)
            targets = batch_rows[:, 1:].to(device)
            batch_masks = {
                cell: mask[start:start + BATCH].to(device) for cell, mask in masks_cpu.items()
            }
            for cell, mask in batch_masks.items():
                for local_document in range(len(mask)):
                    if bool(mask[local_document].any()):
                        cell_documents[cell].add(start + local_document)
            batch_logits = {}
            for arm in ARMS:
                current["arm"] = arm

                def attention_dispatch(event: facade.AttentionEvent, arm=arm):
                    if event.site not in SELECTED or arm == "native":
                        return event.block.attn(event.state, event.first_value)
                    writes, _ = replay_attention_site(
                        event.state, event.first_value, event.block.attn,
                        SELECTED[event.site], tokens,
                    )
                    return writes[arm], event.first_value

                def mlp_dispatch(event: facade.EarlyMLPEvent):
                    return event.block.mlp(event.state)

                batch_logits[arm] = facade.forward_with_dispatch(
                    model, tokens, attention_dispatch, mlp_dispatch,
                )
                current["arm"] = None
            native_logits = batch_logits["native"]
            native_log_prob = F.log_softmax(native_logits, dim=-1)
            native_prob = native_log_prob.exp()
            native_top1 = native_logits.argmax(-1)
            for arm, logits in batch_logits.items():
                loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction="none")
                log_prob = F.log_softmax(logits, dim=-1)
                kl = (native_prob * (native_log_prob - log_prob)).sum(-1)
                top1 = logits.argmax(-1) != native_top1
                append_documents(ledger, arm, batch_masks, loss, kl, top1)
                if arm == "full_replay":
                    replay_max_logit_error = max(
                        replay_max_logit_error, float((logits - native_logits).abs().max()),
                    )
                    replay_kl_sum += float(kl[:, SCORING].sum())
                    replay_tokens += kl[:, SCORING].numel()
    finally:
        current["arm"] = None
        for handle in handles:
            handle.remove()

    expected_batches = len(rows) // BATCH
    if any(value != expected_batches for value in counters["native"].values()):
        raise RuntimeError(f"native selected-site census changed: {counters['native']}")
    for arm in ARMS[1:]:
        if any(counters[arm].values()):
            raise RuntimeError(f"analytical induction arm called native selected sites: {arm}")
    replay_mean_kl = replay_kl_sum / replay_tokens
    replay_pass = replay_max_logit_error <= 1e-4 and replay_mean_kl <= 1e-8
    support = {
        cell: {
            "tokens": int(mask.sum()),
            "documents": len(cell_documents[cell]),
            "powered": int(mask.sum()) >= 200 and len(cell_documents[cell]) >= 30,
        }
        for cell, mask in masks_cpu.items()
    }
    reports = pooled_reports(ledger)
    effects = bootstrap_effects(ledger)
    target = effects["target_damage"]
    off = effects["off_target_damage"]
    extraction = effects["extraction_recovery"]
    gates = {
        "all_named_cells_powered": all(value["powered"] for value in support.values()),
        "replay": replay_pass,
        "removal_necessity": target["bootstrap_95_low"] > 0,
        "specificity": effects["specificity"]["bootstrap_95_low"] > 0,
        "extraction": extraction["mean"] >= 0.80 and extraction["bootstrap_95_low"] >= 0.60,
        "collateral": off["bootstrap_95_high"] <= 0.01
        and off["mean"] <= 0.1 * target["mean"],
        "deranged_null": effects["deranged_recovery"]["bootstrap_95_high"]
        < 0.5 * extraction["mean"],
        "zero_native_candidate_calls": True,
    }
    output = {
        "schema": "induction_equality_tensor_discovery_v1",
        "status": "discovery_complete",
        "claim_boundary": (
            "Already-opened SELECT discovery only; FINAL natural, OOD code, and synthetic "
            "outcomes unopened; no terminal, uniqueness, compression, or strict-ledger credit."
        ),
        "checkpoint": checkpoint.__dict__,
        "documents": len(rows),
        "heads": {str(layer): list(heads) for layer, heads in SELECTED.items()},
        "arms": list(ARMS),
        "cells": support,
        "reports": reports,
        "effects": effects,
        "replay": {
            "maximum_absolute_logit_error": replay_max_logit_error,
            "mean_native_to_replay_kl": replay_mean_kl,
            "passed": replay_pass,
        },
        "gates": gates,
        "eligible_for_fresh_terminal_run": all(gates.values()),
        "call_census": counters,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "runtime_seconds": time.time() - started,
        "parents": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "tensor_primitive_sha256": file_sha256(HERE / "circuit_induction_tensor.py"),
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
            "selection_rows_sha256": file_sha256(SELECTION_ROWS),
        },
    }
    with OUTPUT.open("x") as sink:
        json.dump(output, sink, indent=2, sort_keys=True)
        sink.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
