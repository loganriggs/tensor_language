#!/usr/bin/env python3
"""Discovery-only extraction/removal assay for L0H3's fixed previous-token tensor."""

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
import circuit_previous_token_tensor as previous
import run_mlp1_sparse_c512_continue_factorial_v1_fit as rows_parent


ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
PREREGISTRATION = HERE / "PREVIOUS_TOKEN_TENSOR_DISCOVERY_PREREGISTRATION.md"
OUTPUT = HERE / "previous_token_tensor_discovery.json"
HEAD = 3
D = 1152
HEADS = 9
HEAD_DIM = 128
SCORING = slice(64, 256)
BATCH = 4
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 2_026_083_003
ARMS = (
    "native", "full_replay", "remove_previous", "head_deleted",
    "extract_previous", "deranged_minus_2", "deranged_plus_2",
)
CELLS = (
    "previous_top", "previous_top_unseen_bigram", "previous_top_seen_bigram",
    "self_top", "other_top", "all",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_bigram_set(rows: torch.Tensor) -> set[tuple[int, int]]:
    if rows.ndim != 2 or rows.shape[1] != 257:
        raise ValueError("FIT row schema changed")
    tokens = rows[:, :-1]
    return set(zip(tokens[:, :-1].reshape(-1).tolist(), tokens[:, 1:].reshape(-1).tolist()))


def cell_masks(
    tokens: torch.Tensor,
    head_pattern: torch.Tensor,
    fit_bigrams: set[tuple[int, int]],
) -> dict[str, torch.Tensor]:
    """Frozen evaluation strata; no mask controls candidate execution."""

    if tokens.ndim != 2 or head_pattern.shape != (*tokens.shape, tokens.shape[1]):
        raise ValueError("previous-token cell shapes changed")
    batch, length = tokens.shape
    top_source = head_pattern.abs().argmax(-1)
    query = torch.arange(length, device=tokens.device)[None, :].expand(batch, -1)
    scored = torch.zeros_like(tokens, dtype=torch.bool)
    scored[:, SCORING] = True
    previous_top = scored & (query >= 1) & (top_source == query - 1)
    self_top = scored & (top_source == query)
    unseen = torch.zeros_like(previous_top)
    locations = previous_top.nonzero(as_tuple=False).cpu().tolist()
    token_cpu = tokens.detach().cpu()
    for document, position in locations:
        pair = (int(token_cpu[document, position - 1]), int(token_cpu[document, position]))
        unseen[document, position] = pair not in fit_bigrams
    return {
        "previous_top": previous_top,
        "previous_top_unseen_bigram": previous_top & unseen,
        "previous_top_seen_bigram": previous_top & ~unseen,
        "self_top": self_top,
        "other_top": scored & ~previous_top & ~self_top,
        "all": scored,
    }


def _linear(state: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.linear(state, weight.to(dtype=state.dtype, device=state.device))


@torch.no_grad()
def frozen_layer0_replays(
    state: torch.Tensor,
    attention: torch.nn.Module,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Recompute layer-0 attention and all fixed-mask arms without module forwards."""

    batch, length, width = state.shape
    if width != D:
        raise ValueError("layer-0 width changed")
    q = _linear(state, attention.c_q.weight).view(batch, length, HEADS, HEAD_DIM)
    k = _linear(state, attention.c_k.weight).view(batch, length, HEADS, HEAD_DIM)
    q2 = _linear(state, attention.c_q2.weight).view(batch, length, HEADS, HEAD_DIM)
    k2 = _linear(state, attention.c_k2.weight).view(batch, length, HEADS, HEAD_DIM)
    raw_value = _linear(state, attention.c_v.weight).view(
        batch, length, HEADS, HEAD_DIM,
    )
    value = (1 - attention.lamb) * raw_value + attention.lamb * raw_value
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
    head_scores = pattern[:, HEAD]
    head_values = value[:, :, HEAD]
    previous_only = previous.contract_fixed_shift(head_scores, head_values, -1)
    minus_two_only = previous.contract_fixed_shift(head_scores, head_values, -2)
    plus_two_only = previous.contract_fixed_shift(head_scores, head_values, 2)

    def project(head_write: torch.Tensor) -> torch.Tensor:
        assembled = full.clone()
        assembled[:, HEAD] = head_write
        flattened = assembled.transpose(1, 2).contiguous().view(batch, length, width)
        return _linear(flattened, attention.c_proj.weight)

    zeros = torch.zeros_like(full[:, HEAD])
    writes = {
        "full_replay": project(full[:, HEAD]),
        "remove_previous": project(full[:, HEAD] - previous_only),
        "head_deleted": project(zeros),
        "extract_previous": project(previous_only),
        "deranged_minus_2": project(minus_two_only),
        "deranged_plus_2": project(plus_two_only),
    }
    return writes, raw_value, head_scores


def empty_ledger() -> dict[str, dict[str, dict[str, list[float]]]]:
    return {
        arm: {
            cell: {"loss_sum": [], "kl_sum": [], "top1_changes": [], "count": []}
            for cell in CELLS
        }
        for arm in ARMS
    }


def append_documents(
    ledger: dict[str, dict[str, dict[str, list[float]]]],
    arm: str,
    masks: dict[str, torch.Tensor],
    loss: torch.Tensor,
    kl: torch.Tensor,
    top1_change: torch.Tensor,
) -> None:
    for cell, mask in masks.items():
        for document in range(len(mask)):
            selected = mask[document]
            count = int(selected.sum())
            ledger[arm][cell]["count"].append(count)
            ledger[arm][cell]["loss_sum"].append(float(loss[document, selected].sum()))
            ledger[arm][cell]["kl_sum"].append(float(kl[document, selected].sum()))
            ledger[arm][cell]["top1_changes"].append(
                float(top1_change[document, selected].sum())
            )


def _ratio_sum(values: torch.Tensor, counts: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    numerator = values[indices].sum(-1)
    denominator = counts[indices].sum(-1)
    return numerator / denominator.clamp_min(1)


def bootstrap_effects(
    ledger: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, object]:
    documents = len(ledger["native"]["all"]["count"])
    generator = torch.Generator().manual_seed(BOOTSTRAP_SEED)
    point_ce: dict[str, dict[str, float]] = {arm: {} for arm in ARMS}
    arrays: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    for arm in ARMS:
        for cell in CELLS:
            sums = torch.tensor(ledger[arm][cell]["loss_sum"], dtype=torch.float64)
            counts = torch.tensor(ledger[arm][cell]["count"], dtype=torch.float64)
            arrays[(arm, cell)] = sums, counts
            point_ce[arm][cell] = float(sums.sum() / counts.sum().clamp_min(1))

    draws: dict[str, list[torch.Tensor]] = {
        "removal_previous_top": [], "specificity_previous_minus_self": [],
        "extraction_recovery_previous_top": [],
        "extraction_recovery_unseen": [], "extraction_recovery_seen": [],
        "all_removal": [], "null_minus_2_recovery": [], "null_plus_2_recovery": [],
    }

    def ce(arm: str, cell: str, indices: torch.Tensor) -> torch.Tensor:
        sums, counts = arrays[(arm, cell)]
        return _ratio_sum(sums, counts, indices)

    for start in range(0, BOOTSTRAP_DRAWS, 500):
        count = min(500, BOOTSTRAP_DRAWS - start)
        indices = torch.randint(documents, (count, documents), generator=generator)
        removal_previous = ce("remove_previous", "previous_top", indices) - ce(
            "native", "previous_top", indices,
        )
        removal_self = ce("remove_previous", "self_top", indices) - ce(
            "native", "self_top", indices,
        )
        draws["removal_previous_top"].append(removal_previous)
        draws["specificity_previous_minus_self"].append(removal_previous - removal_self)
        draws["all_removal"].append(
            ce("remove_previous", "all", indices) - ce("native", "all", indices)
        )
        for name, cell in (
            ("extraction_recovery_previous_top", "previous_top"),
            ("extraction_recovery_unseen", "previous_top_unseen_bigram"),
            ("extraction_recovery_seen", "previous_top_seen_bigram"),
        ):
            deleted = ce("head_deleted", cell, indices)
            denominator = deleted - ce("native", cell, indices)
            draws[name].append(
                (deleted - ce("extract_previous", cell, indices))
                / denominator.clamp_min(1e-12)
            )
        denominator = ce("head_deleted", "previous_top", indices) - ce(
            "native", "previous_top", indices,
        )
        for name, arm in (
            ("null_minus_2_recovery", "deranged_minus_2"),
            ("null_plus_2_recovery", "deranged_plus_2"),
        ):
            draws[name].append(
                (ce("head_deleted", "previous_top", indices) - ce(arm, "previous_top", indices))
                / denominator.clamp_min(1e-12)
            )

    intervals = {}
    for name, chunks in draws.items():
        values = torch.cat(chunks).sort().values
        intervals[name] = {
            "mean": float(values.mean()),
            "bootstrap_95_low": float(values[math.floor(0.025 * (len(values) - 1))]),
            "bootstrap_95_high": float(values[math.ceil(0.975 * (len(values) - 1))]),
        }
    return {"pooled_ce": point_ce, "effects": intervals}


@torch.no_grad()
def main() -> None:
    started = time.time()
    if OUTPUT.exists():
        raise RuntimeError(f"previous-token namespace already exists: {OUTPUT}")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    fit_rows = rows_parent.load_role(receipt["entries"]["FIT"])
    select_rows = rows_parent.load_role(receipt["entries"]["SELECT"])
    bigrams = fit_bigram_set(fit_rows)
    device = torch.device("cuda")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
    attention0 = model.transformer.h[0].attn
    counters = {
        arm: {name: 0 for name in ("attention0", "q", "k", "q2", "k2", "v", "o")}
        for arm in ARMS
    }
    current = {"arm": None}

    def hook(name: str):
        def count(_module, _inputs, _output):
            arm = current["arm"]
            if arm is None:
                raise RuntimeError("layer-0 attention call escaped registered arm")
            counters[arm][name] += 1
        return count

    handles = [attention0.register_forward_hook(hook("attention0"))]
    for name, module in (
        ("q", attention0.c_q), ("k", attention0.c_k), ("q2", attention0.c_q2),
        ("k2", attention0.c_k2), ("v", attention0.c_v), ("o", attention0.c_proj),
    ):
        handles.append(module.register_forward_hook(hook(name)))

    ledger = empty_ledger()
    replay_max_absolute_logit_error = 0.0
    replay_kl_sum = 0.0
    replay_tokens = 0
    cell_documents = {cell: set() for cell in CELLS}
    try:
        for start in range(0, len(select_rows), BATCH):
            batch_rows = select_rows[start:start + BATCH]
            tokens = batch_rows[:, :-1].to(device)
            targets = batch_rows[:, 1:].to(device)
            x0 = F.rms_norm(model.transformer.wte(tokens), (D,))
            token_state = (model.transformer.h[0].lambdas[0] + model.transformer.h[0].lambdas[1]) * x0
            attention_state = F.rms_norm(token_state, (D,))
            replay_writes, replay_v1, head_pattern = frozen_layer0_replays(
                attention_state, attention0,
            )
            masks = cell_masks(tokens, head_pattern, bigrams)
            for cell, mask in masks.items():
                for local_document in range(len(mask)):
                    if bool(mask[local_document].any()):
                        cell_documents[cell].add(start + local_document)
            batch_logits: dict[str, torch.Tensor] = {}
            for arm in ARMS:
                current["arm"] = arm

                def attention_dispatch(event: facade.AttentionEvent, arm=arm):
                    if event.site == 0 and arm != "native":
                        return replay_writes[arm], replay_v1
                    return event.block.attn(event.state, event.first_value)

                def mlp_dispatch(event: facade.EarlyMLPEvent):
                    return event.block.mlp(event.state)

                logits = facade.forward_with_dispatch(
                    model, tokens, attention_dispatch, mlp_dispatch,
                )
                batch_logits[arm] = logits
                current["arm"] = None
            native_logits = batch_logits["native"]
            native_log_prob = F.log_softmax(native_logits, dim=-1)
            native_prob = native_log_prob.exp()
            native_top1 = native_logits.argmax(-1)
            for arm, logits in batch_logits.items():
                loss = F.cross_entropy(
                    logits.transpose(1, 2), targets, reduction="none",
                )
                log_prob = F.log_softmax(logits, dim=-1)
                kl = (native_prob * (native_log_prob - log_prob)).sum(-1)
                top1_change = logits.argmax(-1) != native_top1
                append_documents(ledger, arm, masks, loss, kl, top1_change)
                if arm == "full_replay":
                    replay_max_absolute_logit_error = max(
                        replay_max_absolute_logit_error,
                        float((logits - native_logits).abs().max()),
                    )
                    replay_kl_sum += float(kl[:, SCORING].sum())
                    replay_tokens += kl[:, SCORING].numel()
    finally:
        current["arm"] = None
        for handle in handles:
            handle.remove()

    expected_batches = len(select_rows) // BATCH
    expected_native = {name: expected_batches for name in counters["native"]}
    if counters["native"] != expected_native:
        raise RuntimeError(f"native L0 attention census changed: {counters['native']}")
    for arm in ARMS[1:]:
        if any(counters[arm].values()):
            raise RuntimeError(f"analytical arm called native L0 attention: {arm}")
    replay_mean_kl = replay_kl_sum / replay_tokens
    replay_pass = replay_max_absolute_logit_error <= 1e-4 and replay_mean_kl <= 1e-8
    support = {
        cell: {
            "tokens": int(sum(ledger["native"][cell]["count"])),
            "documents": len(cell_documents[cell]),
            "powered": int(sum(ledger["native"][cell]["count"])) >= 200
            and len(cell_documents[cell]) >= 30,
        }
        for cell in CELLS
    }
    analysis = bootstrap_effects(ledger)
    effects = analysis["effects"]
    target_damage = effects["removal_previous_top"]
    all_damage = effects["all_removal"]
    extraction = effects["extraction_recovery_previous_top"]
    unseen = effects["extraction_recovery_unseen"]
    seen = effects["extraction_recovery_seen"]
    gates = {
        "all_named_cells_powered": all(item["powered"] for item in support.values()),
        "replay": replay_pass,
        "removal_necessity": target_damage["bootstrap_95_low"] > 0,
        "specificity": effects["specificity_previous_minus_self"]["bootstrap_95_low"] > 0,
        "extraction": extraction["mean"] >= 0.80 and extraction["bootstrap_95_low"] >= 0.60,
        "unseen_transport": unseen["mean"] >= 0 and unseen["mean"] >= 0.5 * seen["mean"],
        "collateral": all_damage["bootstrap_95_high"] <= 0.01
        and all_damage["mean"] <= 0.1 * target_damage["mean"],
        "null_minus_2": effects["null_minus_2_recovery"]["bootstrap_95_high"]
        < 0.5 * extraction["mean"],
        "null_plus_2": effects["null_plus_2_recovery"]["bootstrap_95_high"]
        < 0.5 * extraction["mean"],
        "zero_native_candidate_calls": True,
    }
    output = {
        "schema": "previous_token_tensor_discovery_v1",
        "status": "discovery_complete",
        "claim_boundary": (
            "Already-opened FIT/SELECT discovery only; no fresh FINAL, terminal, strict-ledger, "
            "uniqueness, full-price, or whole-model circuit credit."
        ),
        "checkpoint": checkpoint.__dict__,
        "documents": {"FIT": len(fit_rows), "SELECT": len(select_rows)},
        "fit_distinct_bigrams": len(bigrams),
        "cells": support,
        "arms": list(ARMS),
        "replay": {
            "maximum_absolute_logit_error": replay_max_absolute_logit_error,
            "mean_native_to_replay_kl": replay_mean_kl,
            "passed": replay_pass,
        },
        "analysis": analysis,
        "gates": gates,
        "eligible_for_fresh_terminal_run": all(gates.values()),
        "call_census": counters,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "runtime_seconds": time.time() - started,
        "parents": {
            "runner_sha256": file_sha256(Path(__file__).resolve()),
            "preregistration_sha256": file_sha256(PREREGISTRATION),
            "tensor_primitive_sha256": file_sha256(HERE / "circuit_previous_token_tensor.py"),
            "rows_receipt_sha256": file_sha256(ROWS_RECEIPT),
        },
    }
    with OUTPUT.open("x") as sink:
        json.dump(output, sink, indent=2, sort_keys=True)
        sink.write("\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
