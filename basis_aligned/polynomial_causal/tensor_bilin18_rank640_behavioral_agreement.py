#!/usr/bin/env python3
"""Forward-only top-1, KL, and rare-target audit of admitted rank640."""

from __future__ import annotations

from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Mapping
import weakref

import torch
import torch.nn.functional as F

import bilin18_observed_model_facade as facade
import tensor_attention_projection_frontier as frontier
import tensor_bilin18_rank512_cross_task_validation as cross
import tensor_bilin18_shared_qk_whole_program as shared
from tensor_bilin18_program import TensorBilin18Program
from tensor_preserving_mlp import TensorMLPBank


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "tensor_bilin18_rank640_behavioral_agreement_results.json"
PREREG = HERE / "RANK640_BEHAVIORAL_AGREEMENT_PREREGISTRATION.md"
PREDICTIVE_PARENT = HERE / "tensor_bilin18_rank640_predictive_validation_results.json"
CAUSAL_PARENT = HERE / "tensor_bilin18_causal_intervention_bank_results.json"
EXPECTED_PREDICTIVE_SHA256 = "639fb8480efee790403113079333100bd63bb61426f6fd6e4dcebd89b21c337d"
EXPECTED_CAUSAL_SHA256 = "73bd18ee81067775680b7d579036e6ec8c04b41116cd3e516b8460a7e7c7ab20"
RANK = 640
EXPECTED_TOTAL = 516_707_766
PRODUCTION_LOGIT_VOCAB = 50_304
BATCH = 4
SCORE_START = 64
BUCKETS = (
    ("0", 0, 0), ("1-4", 1, 4), ("5-24", 5, 24),
    ("25-124", 25, 124), ("125+", 125, math.inf),
)
SOURCES = (
    Path(__file__).resolve(), PREREG,
    HERE / "tensor_bilin18_rank640_predictive_validation.py",
    HERE / "tensor_bilin18_rank512_cross_task_validation.py",
    HERE / "tensor_bilin18_shared_qk_whole_program.py",
    HERE / "tensor_bilin18_program.py",
    HERE / "tensor_attention_projection_frontier.py",
    HERE / "tensor_preserving_attention.py",
    HERE / "tensor_preserving_mlp.py",
    HERE / "bilin18_observed_model_facade.py",
    ROOT / "jacclust/tt_model.py",
    HERE / "test_tensor_bilin18_rank640_behavioral_agreement.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def require_committed_pushed_sources() -> str:
    relative = [str(path.resolve().relative_to(ROOT)) for path in SOURCES]
    status = subprocess.run(
        ("git", "status", "--porcelain=v1", "--", *relative), cwd=ROOT,
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if status:
        raise RuntimeError(f"rank640 agreement source closure is dirty: {status}")
    commit = subprocess.run(
        ("git", "log", "-1", "--format=%H", "--", *relative), cwd=ROOT,
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()
    if not commit or subprocess.run(
        ("git", "merge-base", "--is-ancestor", commit, "origin/main"), cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).returncode:
        raise RuntimeError("rank640 agreement source closure is not pushed")
    return commit


def publish_create_only(value: dict[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        offset = 0
        while offset < len(payload):
            advanced = os.write(descriptor, payload[offset:])
            if advanced <= 0:
                raise OSError("rank640 agreement publication made no progress")
            offset += advanced
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def empty_row() -> dict[str, float | int]:
    return {
        "n": 0, "native_ce_sum": 0.0, "program_ce_sum": 0.0,
        "kl_sum": 0.0, "agreement": 0, "native_correct": 0,
        "program_correct": 0,
    }


def add_metrics(
    accumulator: dict[str, dict[str, float | int]], native: torch.Tensor,
    program: torch.Tensor, targets: torch.Tensor, frequencies: torch.Tensor,
) -> None:
    native_logp = F.log_softmax(native.float(), dim=-1)
    program_logp = F.log_softmax(program.float(), dim=-1)
    native_prob = native_logp.exp()
    native_loss = -torch.gather(native_logp, -1, targets.unsqueeze(-1)).squeeze(-1)
    program_loss = -torch.gather(program_logp, -1, targets.unsqueeze(-1)).squeeze(-1)
    kl = (native_prob * (native_logp - program_logp)).sum(dim=-1)
    native_top = native.argmax(dim=-1)
    program_top = program.argmax(dim=-1)
    masks = {"all": torch.ones_like(targets, dtype=torch.bool)}
    for name, low, high in BUCKETS:
        masks[name] = frequencies >= low
        if high != math.inf:
            masks[name] &= frequencies <= high
    for name, mask in masks.items():
        row = accumulator[name]
        row["n"] += int(mask.sum())
        row["native_ce_sum"] += float(native_loss[mask].double().sum())
        row["program_ce_sum"] += float(program_loss[mask].double().sum())
        row["kl_sum"] += float(kl[mask].double().sum())
        row["agreement"] += int((native_top[mask] == program_top[mask]).sum())
        row["native_correct"] += int((native_top[mask] == targets[mask]).sum())
        row["program_correct"] += int((program_top[mask] == targets[mask]).sum())


def finalize(accumulator: dict[str, dict[str, float | int]]) -> dict[str, Any]:
    result = {}
    for name, row in accumulator.items():
        n = int(row["n"])
        if n == 0:
            result[name] = {"n": 0}
            continue
        native_accuracy = int(row["native_correct"]) / n
        program_accuracy = int(row["program_correct"]) / n
        result[name] = {
            "n": n,
            "native_ce": float(row["native_ce_sum"]) / n,
            "program_ce": float(row["program_ce_sum"]) / n,
            "ce_harm": (float(row["program_ce_sum"]) - float(row["native_ce_sum"])) / n,
            "kl_live_program": float(row["kl_sum"]) / n,
            "top1_agreement": int(row["agreement"]) / n,
            "native_top1_accuracy": native_accuracy,
            "program_top1_accuracy": program_accuracy,
            "accuracy_difference": program_accuracy - native_accuracy,
            "accuracy_retained_fraction": (
                program_accuracy / native_accuracy if native_accuracy > 0 else None
            ),
        }
    return result


def combine_tail(result: dict[str, Any]) -> dict[str, float | int | None]:
    n = result["0"]["n"] + result["1-4"]["n"]
    native_correct = sum(
        result[name]["native_top1_accuracy"] * result[name]["n"]
        for name in ("0", "1-4")
    )
    program_correct = sum(
        result[name]["program_top1_accuracy"] * result[name]["n"]
        for name in ("0", "1-4")
    )
    native = native_correct / n
    program = program_correct / n
    return {
        "n": n, "native_top1_accuracy": native, "program_top1_accuracy": program,
        "accuracy_difference": program - native,
        "accuracy_retained_fraction": program / native if native > 0 else None,
    }


def gates(
    role_results: dict[str, Any], cost: Mapping[str, Any], ownership: Mapping[str, Any],
    parent: Mapping[str, Any],
) -> dict[str, bool]:
    controls = []
    for role, result in role_results.items():
        parent_role = parent["roles"][role]
        controls.extend((
            abs(result["all"]["native_ce"] - parent_role["native"]["all"]["ce"]) <= 2e-6,
            abs(result["all"]["program_ce"] - parent_role["program"]["all"]["ce"]) <= 2e-6,
        ))
    return {
        "A_complete_price_ownership_and_parents": bool(
            cost["total_stored_values"] == EXPECTED_TOTAL
            and cost["native_calls_per_forward"] == 0
            and cost["fitted_lookup_table_values"] == 0
            and cost["total_input_support"] and ownership["storage_disjoint"]
            and not ownership["native_module_references"]
            and file_sha256(PREDICTIVE_PARENT) == EXPECTED_PREDICTIVE_SHA256
            and file_sha256(CAUSAL_PARENT) == EXPECTED_CAUSAL_SHA256
        ),
        "B_exact_parent_ce_replay": all(controls),
        "C_top1_agreement_at_least_0.98": all(
            result["all"]["top1_agreement"] >= 0.98 for result in role_results.values()
        ),
        "D_accuracy_loss_at_most_0.005": all(
            result["all"]["accuracy_difference"] >= -0.005
            for result in role_results.values()
        ),
        "E_kl_at_most_0.01": all(
            result["all"]["kl_live_program"] <= 0.01 for result in role_results.values()
        ),
        "F_rare_target_accuracy": all(
            result["target_frequency_0_4"]["accuracy_difference"] >= -0.005
            and result["target_frequency_0_4"]["accuracy_retained_fraction"] >= 0.97
            for result in role_results.values()
        ),
    }


@torch.no_grad()
def run() -> dict[str, Any]:
    if OUTPUT.exists():
        raise RuntimeError("rank640 behavioral agreement result is create-only")
    started = time.time()
    source_commit = require_committed_pushed_sources()
    source_snapshot = {str(path.resolve()): file_sha256(path.resolve()) for path in SOURCES}
    if file_sha256(PREDICTIVE_PARENT) != EXPECTED_PREDICTIVE_SHA256 or file_sha256(
        CAUSAL_PARENT
    ) != EXPECTED_CAUSAL_SHA256:
        raise RuntimeError("rank640 admitted parent bytes changed")
    parent = json.loads(PREDICTIVE_PARENT.read_text())
    causal = json.loads(CAUSAL_PARENT.read_text())
    if parent.get("status") != "pass" or parent.get("rank") != RANK or causal.get(
        "status"
    ) != "rank640_robust_pass":
        raise RuntimeError("rank640 parent semantics changed")
    role_receipts = {name: cross.validate_role(path) for name, path in cross.ROLE_PATHS.items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = facade.load_bilin18(device=device, dtype=torch.float32)
    fit_rows = frontier.load_rows(frontier.FIT_ROWS, 480)
    attention_bank, fit_receipt = shared.compile_shared_bank(model, fit_rows, rank=RANK)
    blocks = tuple(model.transformer.h)
    program = TensorBilin18Program(
        token_embedding=model.transformer.wte.weight.detach(),
        residual_lambdas=torch.stack([block.lambdas.detach() for block in blocks]),
        unembedding=model.lm_head.weight.detach(), attention_bank=attention_bank,
        mlp_bank=TensorMLPBank.from_model(model),
    )
    native_ptrs = shared.pointer_set(tuple(model.parameters()) + tuple(model.buffers()))
    program_ptrs = shared.pointer_set(tuple(program.parameters()) + tuple(program.buffers()))
    ownership = {
        "storage_disjoint": native_ptrs.isdisjoint(program_ptrs),
        "native_module_references": [
            f"{module.__class__.__module__}.{module.__class__.__qualname__}"
            for module in program.modules()
            if module.__class__.__module__.startswith("jacclust.tt_model")
        ],
    }
    fit_targets = fit_rows[:, 1:frontier.T + 1][:, SCORE_START:].reshape(-1)
    target_counts = torch.bincount(fit_targets.cpu(), minlength=PRODUCTION_LOGIT_VOCAB).to(device)
    role_results = {}
    for role, path in cross.ROLE_PATHS.items():
        rows = frontier.load_rows(path, 192)
        accumulator = {name: empty_row() for name in ("all", *(x[0] for x in BUCKETS))}
        for start in range(0, len(rows), BATCH):
            row_batch = rows[start:start + BATCH]
            tokens = row_batch[:, :frontier.T].to(device)

            def native_attention(event: facade.AttentionEvent):
                return event.block.attn(event.state, event.first_value)

            def native_mlp(event: facade.EarlyMLPEvent):
                return event.block.mlp(event.state)

            native_logits = facade.forward_with_dispatch(
                model, tokens, native_attention, native_mlp,
            )[:, SCORE_START:]
            program_logits = program(tokens)[:, SCORE_START:]
            targets = row_batch[:, 1:frontier.T + 1].to(device)[:, SCORE_START:]
            frequencies = target_counts[targets]
            add_metrics(accumulator, native_logits, program_logits, targets, frequencies)
        role_results[role] = finalize(accumulator)
        role_results[role]["target_frequency_0_4"] = combine_tail(role_results[role])
    cost = program.cost_receipt()
    predictions = gates(role_results, cost, ownership, parent)
    model_reference = weakref.ref(model)
    del blocks, model, fit_rows
    gc.collect()
    if model_reference() is not None:
        raise RuntimeError("checkpoint survives rank640 agreement scoring boundary")
    result = {
        "status": "pass" if all(predictions.values()) else "measured_gate_failure",
        "scope": (
            "new-instrument audit on previously opened rank640 cross-task roles; "
            "not fresh OOD promotion"
        ),
        "rank": RANK,
        "checkpoint": asdict(checkpoint),
        "roles": role_results,
        "predictions": predictions,
        "cost": cost,
        "fit": fit_receipt,
        "execution": {
            **ownership,
            "checkpoint_model_collected_after_joint_scoring": True,
            "program_native_calls": 0,
        },
        "provenance": {
            "source_commit": source_commit, "sources": source_snapshot,
            "parents": {
                "rank640_predictive": file_sha256(PREDICTIVE_PARENT),
                "rank640_causal": file_sha256(CAUSAL_PARENT),
            },
            "roles": role_receipts, "fit": file_sha256(frontier.FIT_ROWS),
        },
        "runtime_s": time.time() - started,
    }
    if {str(path.resolve()): file_sha256(path.resolve()) for path in SOURCES} != (
        source_snapshot
    ):
        raise RuntimeError("rank640 agreement source closure changed before publication")
    publish_create_only(result)
    return result


if __name__ == "__main__":
    outcome = run()
    print(json.dumps({
        "status": outcome["status"], "roles": outcome["roles"],
        "predictions": outcome["predictions"], "runtime_s": outcome["runtime_s"],
    }, indent=2, sort_keys=True), flush=True)
    print(f"wrote {OUTPUT}", flush=True)
