#!/usr/bin/env python3
"""One-shot natural FINAL and code-OOD replication of the frozen equality tensor."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
import traceback

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for root in (ROOT, HERE, BQ):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade
import induction_equality_tensor_discovery as discovery


ROWS_RECEIPT = BQ / "terminal_copy_induction_v2_rows_receipt.json"
DISCOVERY_RESULT = HERE / "induction_equality_tensor_discovery.json"
PREREGISTRATION = HERE / "INDUCTION_EQUALITY_TENSOR_FINAL_OOD_PREREGISTRATION.md"
AUDIT = HERE / "induction_equality_tensor_final_ood_independent_audit.json"
AUTHORITY = HERE / "induction_equality_tensor_final_ood_authority.json"
OUTPUT = HERE / "induction_equality_tensor_final_ood_result.json"
FAILURE = HERE / "induction_equality_tensor_final_ood_failure.json"
ROLE_PATHS = {
    "final_natural": BQ / ".rowcache_terminal_copy_induction_v2/final_natural.pt",
    "ood_code": BQ / ".rowcache_terminal_copy_induction_v2/ood_code.pt",
}
DISCOVERY_SHA256 = "0b826952d227c6f2c9e8b0fadf19aeb28edcd4153a52e4b67777a587733e184b"
ROWS_RECEIPT_SHA256 = "aea52a94c643906ef822a7c6ddb37a371b4315507a1a0a79acd539a19ae7f5c8"
ROLE_SHA256S = {
    "final_natural": "1997026ce15d0524bd16540047799a6461bc94a57fbdd2812ef41ff36e8d5e3c",
    "ood_code": "6cf514e75dfd03399f223a9ba5f6ebe5f4b1315bcb839a515e1c19e7b5474bd9",
}
SELECT_TARGET = 0.5122487687425222
SELECT_EXTRACTION = 0.9739717690344445


def validate_authority() -> dict:
    if OUTPUT.exists() or FAILURE.exists():
        raise RuntimeError("one-shot FINAL/OOD namespace is already closed")
    if not AUTHORITY.exists():
        raise RuntimeError("source-bound FINAL/OOD authority is absent")
    authority = json.loads(AUTHORITY.read_text())
    required = {
        "approved": True,
        "outcome_access": False,
        "runner_sha256": discovery.file_sha256(Path(__file__).resolve()),
        "preregistration_sha256": discovery.file_sha256(PREREGISTRATION),
        "audit_sha256": discovery.file_sha256(AUDIT),
    }
    for key, expected in required.items():
        if authority.get(key) != expected:
            raise RuntimeError(f"authority mismatch for {key}")
    return authority


def role_gates(role: str, support: dict, effects: dict, replay_pass: bool) -> dict:
    target = effects["target_damage"]
    specificity = effects["specificity"]
    off = effects["off_target_damage"]
    extraction = effects["extraction_recovery"]
    deranged = effects["deranged_recovery"]
    common = {
        "all_named_cells_powered": all(value["powered"] for value in support.values()),
        "replay": replay_pass,
        "removal_necessity": target["bootstrap_95_low"] > 0,
        "specificity": specificity["bootstrap_95_low"] > 0,
        "deranged_null": deranged["bootstrap_95_high"] < 0.5 * extraction["mean"],
        "zero_native_candidate_calls": True,
    }
    if role == "final_natural":
        common.update({
            "extraction": extraction["mean"] >= 0.80
            and extraction["bootstrap_95_low"] >= 0.60,
            "collateral": off["bootstrap_95_high"] <= 0.01
            and off["mean"] <= 0.10 * target["mean"],
        })
    elif role == "ood_code":
        common.update({
            "target_transport": target["mean"] >= 0.50 * SELECT_TARGET,
            "extraction": extraction["mean"] >= 0.60
            and extraction["bootstrap_95_low"] >= 0.40,
            "extraction_transport": extraction["mean"] >= 0.50 * SELECT_EXTRACTION,
            "collateral": off["bootstrap_95_high"] <= 0.02
            and off["mean"] <= 0.20 * target["mean"],
        })
    else:
        raise ValueError(role)
    return common


@torch.no_grad()
def score_role(model: torch.nn.Module, role: str, path: Path) -> dict:
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    rows = bundle["rows"]
    cells = bundle["copy_cells"]
    masks_cpu = {
        "positive": cells["positive"],
        "matched_negative": cells["matched_negative"],
        "off_target": cells["off_target"],
        "all": torch.zeros_like(cells["positive"]),
    }
    masks_cpu["all"][:, discovery.SCORING] = True
    if rows.ndim != 2 or rows.shape[1] != 257 or len(rows) % discovery.BATCH:
        raise RuntimeError(f"{role} row schema changed")

    current = {"arm": None}
    counters = {
        arm: {
            f"L{layer}:{name}": 0
            for layer in discovery.SELECTED
            for name in ("attention", "q", "k", "q2", "k2", "v", "o")
        }
        for arm in discovery.ARMS
    }

    def hook(layer, name):
        def count(_module, _inputs, _output):
            arm = current["arm"]
            if arm is None:
                raise RuntimeError("selected attention call escaped registered arm")
            counters[arm][f"L{layer}:{name}"] += 1
        return count

    handles = []
    for layer in discovery.SELECTED:
        attention = model.transformer.h[layer].attn
        handles.append(attention.register_forward_hook(hook(layer, "attention")))
        for name, module in (
            ("q", attention.c_q), ("k", attention.c_k),
            ("q2", attention.c_q2), ("k2", attention.c_k2),
            ("v", attention.c_v), ("o", attention.c_proj),
        ):
            handles.append(module.register_forward_hook(hook(layer, name)))

    ledger = discovery.empty_ledger()
    replay_max_logit_error = 0.0
    replay_kl_sum = 0.0
    replay_tokens = 0
    cell_documents = {cell: set() for cell in discovery.CELLS}
    device = next(model.parameters()).device
    try:
        for start in range(0, len(rows), discovery.BATCH):
            batch_rows = rows[start:start + discovery.BATCH]
            tokens = batch_rows[:, :-1].to(device)
            targets = batch_rows[:, 1:].to(device)
            batch_masks = {
                cell: mask[start:start + discovery.BATCH].to(device)
                for cell, mask in masks_cpu.items()
            }
            for cell, mask in batch_masks.items():
                for local_document in range(len(mask)):
                    if bool(mask[local_document].any()):
                        cell_documents[cell].add(start + local_document)
            batch_logits = {}
            for arm in discovery.ARMS:
                current["arm"] = arm

                def attention_dispatch(event: facade.AttentionEvent, arm=arm):
                    if event.site not in discovery.SELECTED or arm == "native":
                        return event.block.attn(event.state, event.first_value)
                    writes, _ = discovery.replay_attention_site(
                        event.state, event.first_value, event.block.attn,
                        discovery.SELECTED[event.site], tokens,
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
                discovery.append_documents(ledger, arm, batch_masks, loss, kl, top1)
                if arm == "full_replay":
                    replay_max_logit_error = max(
                        replay_max_logit_error, float((logits - native_logits).abs().max()),
                    )
                    replay_kl_sum += float(kl[:, discovery.SCORING].sum())
                    replay_tokens += kl[:, discovery.SCORING].numel()
    finally:
        current["arm"] = None
        for handle in handles:
            handle.remove()

    expected_batches = len(rows) // discovery.BATCH
    if any(value != expected_batches for value in counters["native"].values()):
        raise RuntimeError(f"{role} native census changed")
    for arm in discovery.ARMS[1:]:
        if any(counters[arm].values()):
            raise RuntimeError(f"{role} analytical arm called native selected site: {arm}")
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
    effects = discovery.bootstrap_effects(ledger)
    gates = role_gates(role, support, effects, replay_pass)
    return {
        "documents": len(rows),
        "support": support,
        "reports": discovery.pooled_reports(ledger),
        "effects": effects,
        "replay": {
            "maximum_absolute_logit_error": replay_max_logit_error,
            "mean_native_to_replay_kl": replay_mean_kl,
            "passed": replay_pass,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "call_census": counters,
    }


def execute() -> dict:
    started = time.time()
    authority = validate_authority()
    if discovery.file_sha256(DISCOVERY_RESULT) != DISCOVERY_SHA256:
        raise RuntimeError("SELECT discovery result binding changed")
    if discovery.file_sha256(ROWS_RECEIPT) != ROWS_RECEIPT_SHA256:
        raise RuntimeError("row receipt binding changed")
    receipt = json.loads(ROWS_RECEIPT.read_text())
    for role, path in ROLE_PATHS.items():
        if discovery.file_sha256(path) != ROLE_SHA256S[role]:
            raise RuntimeError(f"{role} container binding changed")
        if receipt["entries"][role]["file_sha256"] != ROLE_SHA256S[role]:
            raise RuntimeError(f"{role} receipt entry changed")
    model, checkpoint = facade.load_bilin18(
        device=torch.device("cuda"), dtype=torch.bfloat16,
    )
    roles = {role: score_role(model, role, path) for role, path in ROLE_PATHS.items()}
    return {
        "schema": "induction_equality_tensor_final_ood_v1",
        "status": "one_shot_complete",
        "claim_boundary": (
            "Fresh natural FINAL and single-repository code OOD for the fixed equality "
            "tensor; no uniqueness, global copy, synthetic, or cheaper-parameter claim."
        ),
        "checkpoint": checkpoint.__dict__,
        "heads": {str(layer): list(heads) for layer, heads in discovery.SELECTED.items()},
        "arms": list(discovery.ARMS),
        "roles": roles,
        "passed_both_roles": all(result["passed"] for result in roles.values()),
        "bootstrap": {
            "draws": discovery.BOOTSTRAP_DRAWS,
            "seed": discovery.BOOTSTRAP_SEED,
        },
        "runtime_seconds": time.time() - started,
        "parents": {
            "runner_sha256": discovery.file_sha256(Path(__file__).resolve()),
            "preregistration_sha256": discovery.file_sha256(PREREGISTRATION),
            "audit_sha256": discovery.file_sha256(AUDIT),
            "authority_sha256": discovery.file_sha256(AUTHORITY),
            "discovery_result_sha256": discovery.file_sha256(DISCOVERY_RESULT),
            "rows_receipt_sha256": discovery.file_sha256(ROWS_RECEIPT),
            "role_sha256s": ROLE_SHA256S,
            "authority": authority,
        },
    }


def main() -> None:
    try:
        result = execute()
        with OUTPUT.open("x") as sink:
            json.dump(result, sink, indent=2, sort_keys=True)
            sink.write("\n")
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as error:
        if not OUTPUT.exists() and not FAILURE.exists():
            failure = {
                "schema": "induction_equality_tensor_final_ood_failure_v1",
                "status": "closed_by_failure",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "runner_sha256": discovery.file_sha256(Path(__file__).resolve()),
                "preregistration_sha256": discovery.file_sha256(PREREGISTRATION),
            }
            with FAILURE.open("x") as sink:
                json.dump(failure, sink, indent=2, sort_keys=True)
                sink.write("\n")
        raise


if __name__ == "__main__":
    main()
