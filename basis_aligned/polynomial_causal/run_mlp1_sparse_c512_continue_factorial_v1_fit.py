#!/usr/bin/env python3
"""FIT/SELECT stage for the standalone sparse MLP1-Down composition candidate.

This stage never requests or deserializes FINAL.  It captures the exact native MLP1
gate and bias-free Down action on FIT/SELECT, trains the prospectively frozen seeds,
selects by SELECT output R2 only, then measures the selected program's SELECT CE gate.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
from typing import Any, Mapping

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BQ = HERE.parent / "bilinear_quotient"
for root in (ROOT, HERE, BQ):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import bilin18_observed_model_facade as facade  # noqa: E402
import mlp1_sparse_down_program_v1 as sparse  # noqa: E402
import prepare_mlp1_sparse_c512_continue_factorial_v1_rows as rows_life  # noqa: E402


PREREG = HERE / "MLP1_SPARSE_C512_CONTINUE_FACTORIAL_V1_PREREGISTRATION.md"
ROWS_RECEIPT = BQ / "mlp1_sparse_c512_continue_factorial_v1_rows_receipt.json"
AUTHORITY = HERE / "mlp1_sparse_c512_continue_factorial_v1_fit_authority.json"
BUNDLE = HERE / "mlp1_sparse_c512_continue_factorial_v1_fit_bundle.pt"
RESULT = HERE / "mlp1_sparse_c512_continue_factorial_v1_fit_result.json"
RECEIPT = HERE / "mlp1_sparse_c512_continue_factorial_v1_fit_receipt.json"
FAILURE = HERE / "mlp1_sparse_c512_continue_factorial_v1_fit_failure.json"
LOCK = Path("/workspace/runs/.mlp1_sparse_c512_continue_factorial_v1_fit.lock")

SEEDS = (0, 1, 2)
STEPS = 2400
BATCH_SIZE = 1024
LEARNING_RATE = 0.003
CURVE_EVERY = 200
DOCUMENT_BATCH = 4
SCORING = slice(64, 256)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_json(path: Path, expected: str | None = None) -> tuple[dict[str, Any], str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    middle = hashlib.sha256(raw).hexdigest()
    after = file_sha256(path)
    if before != middle or middle != after or (expected is not None and before != expected):
        raise RuntimeError(f"JSON parent raced or changed: {path}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON parent is not an object: {path}")
    return value, before


def stable_torch(path: Path, expected: str | None = None) -> tuple[Any, str]:
    before = file_sha256(path)
    raw = path.read_bytes()
    middle = hashlib.sha256(raw).hexdigest()
    after = file_sha256(path)
    if before != middle or middle != after or (expected is not None and before != expected):
        raise RuntimeError(f"tensor parent raced or changed: {path}")
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True), before


def write_json_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n"); sink.flush(); os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_torch_create_only(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def select_seed(records: list[Mapping[str, Any]]) -> Mapping[str, Any]:
    if {record.get("seed") for record in records} != set(SEEDS):
        raise RuntimeError("sparse-Down seed records changed")
    return min(records, key=lambda record: (-float(record["final_select_r2"]), int(record["seed"])))


def convergence_metrics(curve: list[Mapping[str, Any]]) -> dict[str, Any]:
    if len(curve) != STEPS // CURVE_EVERY or curve[-1]["step"] != STEPS:
        raise RuntimeError("sparse-Down training curve cadence changed")
    final = float(curve[-1]["select_r2"])
    last = [float(row["select_r2"]) for row in curve[-3:]]
    best = max(float(row["select_r2"]) for row in curve)
    return {
        "last_three_r2_range": max(last) - min(last),
        "final_minus_best_r2": final - best,
        "converged": max(last) - min(last) <= 0.01 and final >= best - 0.005,
    }


def selection_gates(seed_records: list[Mapping[str, Any]], ce_recovery: float) -> dict[str, Any]:
    selected = select_seed(seed_records)
    finals = torch.tensor([float(row["final_select_r2"]) for row in seed_records], dtype=torch.float64)
    return {
        "selected_seed": int(selected["seed"]),
        "executable_select_ce_recovery_ge_0p90": ce_recovery >= 0.90,
        "selected_curve_converged": bool(selected["convergence"]["converged"]),
        "seed_final_select_r2_std_le_0p02": float(finals.std(unbiased=False)) <= 0.02,
        "seed_final_select_r2_std": float(finals.std(unbiased=False)),
        "admitted_to_final": ce_recovery >= 0.90,
    }


def validate_row_receipt(value: Mapping[str, Any], sources: Mapping[str, str]) -> None:
    expected_roles = {
        "FIT": {
            "authorized_for_training": True,
            "authorized_for_selection": False,
            "authorized_for_final": False,
        },
        "SELECT": {
            "authorized_for_training": False,
            "authorized_for_selection": True,
            "authorized_for_final": False,
        },
        "FINAL": {
            "authorized_for_training": False,
            "authorized_for_selection": False,
            "authorized_for_final": True,
        },
    }
    if value.get("schema") != "mlp1_sparse_c512_continue_factorial_v1_rows" \
            or value.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" \
            or value.get("source_hashes") != dict(sources) \
            or value.get("selection") != {
                "start_document_index": 122000,
                "documents_per_role": 96,
                "token_length": 257,
                "scored_slice": [64, 256],
            } or value.get("roles") != expected_roles \
            or set(value.get("entries", {})) != set(expected_roles) \
            or not all(value.get("disjointness", {}).values()):
        raise RuntimeError("sparse-Down row receipt semantics changed")
    for role, entry in value["entries"].items():
        path = Path(entry["path"])
        if entry.get("shape") != [96, 257] or entry.get("dtype") != "torch.int64" \
                or not path.is_file() or file_sha256(path) != entry.get("file_sha256"):
            raise RuntimeError(f"sparse-Down {role} row entry changed")


@torch.no_grad()
def capture_gate_action(
    model: Any, role_rows: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    gates, actions = [], []
    calls = {"forwards": 0, "site1_captures": 0, "native_mlp_calls": 0}
    for start in range(0, len(role_rows), DOCUMENT_BATCH):
        tokens = role_rows[start:start + DOCUMENT_BATCH, :-1].to(device)

        def attention(event: facade.AttentionEvent):
            return event.block.attn(event.state, event.first_value)

        def mlp(event: facade.EarlyMLPEvent):
            if event.site == 1:
                gate = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
                action = event.block.mlp.Down(gate)
                gates.append(gate.detach().float().cpu().reshape(-1, sparse.GATE_DIM))
                actions.append(action.detach().float().cpu().reshape(-1, sparse.OUTPUT_DIM))
                calls["site1_captures"] += 1
                return action + event.block.mlp.Down_bias
            calls["native_mlp_calls"] += 1
            return event.block.mlp(event.state)

        facade.forward_with_dispatch(model, tokens, attention, mlp)
        calls["forwards"] += 1
    expected = len(role_rows) // DOCUMENT_BATCH
    if calls != {
        "forwards": expected,
        "site1_captures": expected,
        "native_mlp_calls": expected * 17,
    }:
        raise RuntimeError(f"sparse-Down capture call census changed: {calls}")
    return torch.cat(gates), torch.cat(actions), calls


@torch.no_grad()
def r2_chunks(
    encoder: torch.Tensor, decoder: torch.Tensor, intercept: torch.Tensor,
    gates: torch.Tensor, targets: torch.Tensor, device: torch.device,
) -> float:
    residual = 0.0
    centered = float(((targets.double() - targets.double().mean(0)).square()).sum())
    for start in range(0, len(gates), 2048):
        gate = gates[start:start + 2048].to(device)
        target = targets[start:start + 2048].to(device)
        prediction = sparse.topk_relu(gate @ encoder.T) @ decoder.T + intercept
        residual += float((target.double() - prediction.double()).square().sum())
    return 1.0 - residual / max(centered, 1e-30)


def train_seed(
    seed: int, fit_gate: torch.Tensor, fit_target: torch.Tensor,
    select_gate: torch.Tensor, select_target: torch.Tensor, device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(10_000 + seed)
    encoder = torch.randn(
        sparse.DICTIONARY_SIZE, sparse.GATE_DIM, device=device,
    ) / math.sqrt(sparse.GATE_DIM)
    encoder /= encoder.norm(dim=1, keepdim=True)
    decoder = torch.randn(
        sparse.OUTPUT_DIM, sparse.DICTIONARY_SIZE, device=device,
    ) / math.sqrt(sparse.OUTPUT_DIM)
    intercept = fit_target.mean(0).to(device)
    encoder.requires_grad_(True); decoder.requires_grad_(True); intercept.requires_grad_(True)
    optimizer = torch.optim.Adam([encoder, decoder, intercept], lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=STEPS)
    curve = []
    for step in range(1, STEPS + 1):
        indices = torch.randint(len(fit_gate), (BATCH_SIZE,), generator=generator)
        gate = fit_gate.index_select(0, indices).to(device)
        target = fit_target.index_select(0, indices).to(device)
        codes = sparse.topk_relu(gate @ encoder.T)
        prediction = codes @ decoder.T + intercept
        loss = F.mse_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); scheduler.step()
        with torch.no_grad():
            encoder /= encoder.norm(dim=1, keepdim=True).clamp_min(1e-12)
        if step % CURVE_EVERY == 0:
            with torch.no_grad():
                select_r2 = r2_chunks(
                    encoder, decoder, intercept, select_gate, select_target, device,
                )
            curve.append({
                "step": step,
                "train_mse": float(loss.detach()),
                "select_r2": select_r2,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            })
            print(f"seed={seed} step={step} mse={float(loss):.6g} select_r2={select_r2:.6f}", flush=True)
    state = {
        "encoder": encoder.detach().cpu().float().contiguous(),
        "decoder": decoder.detach().cpu().float().contiguous(),
        "intercept": intercept.detach().cpu().float().contiguous(),
    }
    sparse.validate_state(state)
    metrics = convergence_metrics(curve)
    return state, {
        "seed": seed,
        "final_select_r2": float(curve[-1]["select_r2"]),
        "curve": curve,
        "convergence": metrics,
    }


@torch.no_grad()
def score_select_ce(
    model: Any, role_rows: torch.Tensor, program: sparse.SparseDownProgram,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, Any]]:
    sums = {arm: 0.0 for arm in ("NATIVE", "ZERO", "SPARSE")}
    counts = {arm: 0 for arm in sums}
    calls = {arm: {"forwards": 0, "native_mlp1": 0, "sparse_mlp1": 0, "zero_mlp1": 0}
             for arm in sums}
    for start in range(0, len(role_rows), DOCUMENT_BATCH):
        batch = role_rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        for arm in sums:
            def attention(event: facade.AttentionEvent):
                return event.block.attn(event.state, event.first_value)

            def mlp(event: facade.EarlyMLPEvent, arm=arm):
                if event.site != 1:
                    return event.block.mlp(event.state)
                if arm == "NATIVE":
                    calls[arm]["native_mlp1"] += 1
                    return event.block.mlp(event.state)
                gate = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
                if arm == "ZERO":
                    calls[arm]["zero_mlp1"] += 1
                    return torch.zeros_like(event.state) + event.block.mlp.Down_bias
                calls[arm]["sparse_mlp1"] += 1
                return program(gate) + event.block.mlp.Down_bias

            logits = facade.forward_with_dispatch(model, tokens, attention, mlp)
            ce = F.cross_entropy(
                logits[:, SCORING].reshape(-1, logits.shape[-1]),
                targets[:, SCORING].reshape(-1), reduction="sum",
            )
            sums[arm] += float(ce); counts[arm] += targets[:, SCORING].numel()
            calls[arm]["forwards"] += 1
    result = {arm: sums[arm] / counts[arm] for arm in sums}
    denominator = result["ZERO"] - result["NATIVE"]
    result["recovery"] = (result["ZERO"] - result["SPARSE"]) / denominator
    return result, calls


def main() -> None:
    terminals = (AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE, LOCK)
    if any(path.exists() for path in terminals):
        raise RuntimeError("sparse-Down FIT namespace already exists")
    claim = rows_life.base.acquire_claim(LOCK)
    fit_select_opened = False
    final_opened = False
    started = time.time()
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        sources = rows_life.source_hashes(commit)
        audit, audit_sha = rows_life.validate_independent_audit(sources)
        row_receipt, row_receipt_sha = stable_json(ROWS_RECEIPT)
        validate_row_receipt(row_receipt, sources)
        authority = {
            "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_authority",
            "status": "frozen_before_fit_select_open",
            "source_commit": commit,
            "source_hashes": sources,
            "audit_sha256": audit_sha,
            "audit_reviewer": audit["reviewer"],
            "row_receipt_sha256": row_receipt_sha,
            "roles_opened": [],
            "final_role_forbidden": True,
            "seeds": list(SEEDS),
            "steps": STEPS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "selection_variable": "final SELECT output R2 only",
        }
        write_json_create_only(AUTHORITY, authority)
        authority_sha = file_sha256(AUTHORITY)
        entries = row_receipt["entries"]
        fit, _ = stable_torch(Path(entries["FIT"]["path"]), entries["FIT"]["file_sha256"])
        select, _ = stable_torch(
            Path(entries["SELECT"]["path"]), entries["SELECT"]["file_sha256"],
        )
        fit_select_opened = True
        if tuple(fit.shape) != (96, 257) or tuple(select.shape) != (96, 257):
            raise RuntimeError("FIT/SELECT row shape changed")
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        fit_gate, fit_target, fit_calls = capture_gate_action(model, fit, device)
        select_gate, select_target, select_calls = capture_gate_action(model, select, device)
        seed_states = []
        seed_records = []
        for seed in SEEDS:
            state, record = train_seed(
                seed, fit_gate, fit_target, select_gate, select_target, device,
            )
            seed_states.append(state); seed_records.append(record)
        selected_record = select_seed(seed_records)
        selected_index = list(SEEDS).index(int(selected_record["seed"]))
        selected_state = seed_states[selected_index]
        program = sparse.SparseDownProgram(selected_state, device).eval()
        select_ce, score_calls = score_select_ce(model, select, program, device)
        gates = selection_gates(seed_records, float(select_ce["recovery"]))
        bundle = {
            "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_bundle",
            "status": "selected_program_frozen_before_final",
            "authority_sha256": authority_sha,
            "program": selected_state,
            "selected_seed": gates["selected_seed"],
            "price": sparse.SparseDownProgram.price(),
            "final_opened": False,
        }
        write_torch_create_only(BUNDLE, bundle)
        bundle_sha = file_sha256(BUNDLE)
        replay, _ = stable_torch(BUNDLE, bundle_sha)
        sparse.validate_state(replay["program"])
        result = {
            "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_result",
            "status": "admitted_to_final" if gates["admitted_to_final"] else "selection_gate_failed",
            "runtime_seconds": time.time() - started,
            "documents": {"FIT": 96, "SELECT": 96, "FINAL_opened": 0},
            "captured_positions": {"FIT": len(fit_gate), "SELECT": len(select_gate)},
            "seed_records": seed_records,
            "select_ce": select_ce,
            "selection_gates": gates,
            "price": sparse.SparseDownProgram.price(),
            "calls": {"FIT": fit_calls, "SELECT_capture": select_calls, "SELECT_score": score_calls},
            "parents": {"authority_sha256": authority_sha, "bundle_sha256": bundle_sha},
            "checkpoint": checkpoint.__dict__,
            "claim_boundary": "FIT/SELECT candidate admission only; no FINAL or composition claim",
        }
        write_json_create_only(RESULT, result)
        result_sha = file_sha256(RESULT)
        receipt = {
            "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_receipt",
            "status": "fit_select_complete_receipt_last",
            "authority_sha256": authority_sha,
            "bundle_sha256": bundle_sha,
            "result_sha256": result_sha,
            "fit_select_opened": fit_select_opened,
            "final_opened": final_opened,
            "admitted_to_final": gates["admitted_to_final"],
        }
        write_json_create_only(RECEIPT, receipt)
        print(json.dumps(result, indent=2, sort_keys=True))
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            write_json_create_only(FAILURE, {
                "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_failure",
                "status": "terminal_failure",
                "error": repr(error),
                "fit_select_may_have_opened": fit_select_opened,
                "final_may_have_opened": final_opened,
                "artifact_hashes": {
                    path.name: file_sha256(path)
                    for path in (AUTHORITY, BUNDLE, RESULT) if path.is_file()
                },
            })
        raise
    finally:
        rows_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
