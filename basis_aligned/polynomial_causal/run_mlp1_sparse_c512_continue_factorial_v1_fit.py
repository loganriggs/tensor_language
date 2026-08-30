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
from typing import Any, Callable, Mapping

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
SCORING_POSITIONS = SCORING.stop - SCORING.start
ROLE_DOCUMENTS = 96
EXPECTED_FORWARDS = ROLE_DOCUMENTS // DOCUMENT_BATCH
ROW_RECEIPT_KEYS = {
    "schema", "status", "source_commit", "source_hashes", "independent_audit",
    "selection", "roles", "entries", "provenance", "disjointness",
    "ordered_manifest_gate", "registry_hashes", "prior_tensor_hashes",
    "waiver_proofs", "nonrow_proofs", "outcome_access",
}
DISJOINTNESS_KEYS = {
    "unique_source_documents", "unique_dataset_indices", "unique_full_rows",
    "unique_prefix32", "source_documents_disjoint_from_registry",
    "dataset_indices_disjoint_from_registry", "full_rows_disjoint_from_registry",
    "prefix32_disjoint_from_registry",
}


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


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_create_only(
    path: Path, value: Any, *, pre_link_check: Callable[[], None] | None = None,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        with temporary.open("x") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n"); sink.flush(); os.fsync(sink.fileno())
        if pre_link_check is not None:
            pre_link_check()
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def write_torch_create_only(
    path: Path, value: Any, *, pre_link_check: Callable[[], None] | None = None,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{secrets.token_hex(8)}")
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        if pre_link_check is not None:
            pre_link_check()
        os.link(temporary, path)
        fsync_directory(path.parent)
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
    if not bool(torch.isfinite(finals).all()) or not math.isfinite(float(ce_recovery)):
        raise RuntimeError("non-finite sparse-Down selection metric")
    return {
        "selected_seed": int(selected["seed"]),
        "executable_select_ce_recovery_ge_0p90": ce_recovery >= 0.90,
        "selected_curve_converged": bool(selected["convergence"]["converged"]),
        "seed_final_select_r2_std_le_0p02": float(finals.std(unbiased=False)) <= 0.02,
        "seed_final_select_r2_std": float(finals.std(unbiased=False)),
        "admitted_to_final": ce_recovery >= 0.90,
    }


def validate_row_receipt(
    value: Mapping[str, Any], sources: Mapping[str, str], audit: Mapping[str, Any],
    audit_sha: str,
) -> None:
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
    audit_entry = value.get("independent_audit", {})
    if set(value) != ROW_RECEIPT_KEYS \
            or value.get("schema") != "mlp1_sparse_c512_continue_factorial_v1_rows" \
            or value.get("status") != "fresh_roles_frozen_before_any_model_or_training_access" \
            or value.get("source_commit") != audit.get("audited_source_commit") \
            or value.get("source_hashes") != dict(sources) \
            or value.get("selection") != {
                "start_document_index": 122000,
                "documents_per_role": 96,
                "token_length": 257,
                "scored_slice": [64, 256],
            } or value.get("roles") != expected_roles \
            or value.get("outcome_access") != {"model_loaded": False, "training_run": False} \
            or set(audit_entry) != {
                "path", "file_sha256", "audited_source_commit", "reviewer", "tests_passed",
            } or audit_entry.get("file_sha256") != audit_sha \
            or audit_entry.get("audited_source_commit") != audit.get("audited_source_commit") \
            or audit_entry.get("reviewer") != audit.get("reviewer") \
            or audit_entry.get("tests_passed") != audit.get("tests_passed") \
            or set(value.get("entries", {})) != set(expected_roles) \
            or set(value.get("provenance", {})) != set(expected_roles) \
            or set(value.get("disjointness", {})) != DISJOINTNESS_KEYS \
            or not all(item is True for item in value["disjointness"].values()):
        raise RuntimeError("sparse-Down row receipt semantics changed")
    documents: set[str] = set()
    indices: set[int] = set()
    for role, entry in value["entries"].items():
        records = value["provenance"][role]
        path = Path(entry["path"])
        if set(entry) != {"path", "file_sha256", "tensor_sha256", "shape", "dtype"} \
                or entry.get("shape") != [ROLE_DOCUMENTS, 257] \
                or entry.get("dtype") != "torch.int64" \
                or not path.is_file() or file_sha256(path) != entry.get("file_sha256"):
            raise RuntimeError(f"sparse-Down {role} row entry changed")
        if not isinstance(records, list) or len(records) != ROLE_DOCUMENTS:
            raise RuntimeError(f"sparse-Down {role} provenance changed")
        role_documents, role_indices = [], []
        for record in records:
            if set(record) != {
                "document_id", "dataset_document_index", "row_index",
                "source_document_ordinal", "chunk_id", "token_start",
            } or not isinstance(record["document_id"], str) \
                    or not record["document_id"] \
                    or any(type(record[key]) is not int for key in (
                        "dataset_document_index", "row_index", "source_document_ordinal",
                        "chunk_id", "token_start",
                    )):
                raise RuntimeError(f"sparse-Down {role} provenance record changed")
            role_documents.append(record["document_id"])
            role_indices.append(record["dataset_document_index"])
        if len(set(role_documents)) != ROLE_DOCUMENTS \
                or len(set(role_indices)) != ROLE_DOCUMENTS \
                or documents.intersection(role_documents) or indices.intersection(role_indices):
            raise RuntimeError("sparse-Down role provenance overlaps")
        documents.update(role_documents); indices.update(role_indices)


def load_role(entry: Mapping[str, Any]) -> torch.Tensor:
    path = Path(entry["path"])
    value, _ = stable_torch(path, str(entry["file_sha256"]))
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int64 \
            or tuple(value.shape) != (ROLE_DOCUMENTS, 257) \
            or rows_life.base.tensor_sha256(value) != entry["tensor_sha256"]:
        raise RuntimeError(f"row tensor semantics changed: {path}")
    return value


def checkpoint_snapshot() -> dict[str, Any]:
    snapshot = Path(facade.DEFAULT_SNAPSHOT)
    config = snapshot / "config.json"
    weights = snapshot / "pytorch_model.bin"
    value = {
        "config_sha256": file_sha256(config),
        "weights_sha256": file_sha256(weights),
        "weights_bytes": weights.stat().st_size,
    }
    if value != {
        "config_sha256": facade.CONFIG_SHA256,
        "weights_sha256": facade.WEIGHTS_SHA256,
        "weights_bytes": facade.WEIGHTS_BYTES,
    }:
        raise RuntimeError("pinned bilin18 checkpoint changed")
    return value


def protected_snapshot(
    commit: str, sources: Mapping[str, str], audit_sha: str, row_receipt_sha: str,
) -> dict[str, Any]:
    if rows_life.source_hashes(commit) != dict(sources):
        raise RuntimeError("protected sparse-Down source closure changed")
    audit, current_audit_sha = rows_life.validate_independent_audit(sources)
    if current_audit_sha != audit_sha:
        raise RuntimeError("protected sparse-Down audit changed")
    receipt, current_receipt_sha = stable_json(ROWS_RECEIPT, row_receipt_sha)
    validate_row_receipt(receipt, sources, audit, audit_sha)
    row_hashes = {
        role: file_sha256(Path(receipt["entries"][role]["path"]))
        for role in ("FIT", "SELECT", "FINAL")
    }
    expected_rows = {
        role: receipt["entries"][role]["file_sha256"]
        for role in ("FIT", "SELECT", "FINAL")
    }
    if row_hashes != expected_rows:
        raise RuntimeError("protected sparse-Down row bytes changed")
    return {
        "source_commit": commit,
        "source_hashes": dict(sources),
        "audit_sha256": current_audit_sha,
        "row_receipt_sha256": current_receipt_sha,
        "row_hashes": row_hashes,
        "checkpoint": checkpoint_snapshot(),
    }


def verify_protected(
    expected: Mapping[str, Any], commit: str, sources: Mapping[str, str],
    audit_sha: str, row_receipt_sha: str, claim: Any,
) -> None:
    rows_life.base.require_claim(claim, LOCK)
    if protected_snapshot(commit, sources, audit_sha, row_receipt_sha) != dict(expected):
        raise RuntimeError("protected sparse-Down inputs changed during execution")
    rows_life.base.require_claim(claim, LOCK)


def artifact_snapshot(paths: tuple[Path, ...]) -> dict[str, str | None]:
    """Bind presence as well as bytes, so an absent artifact cannot appear unnoticed."""

    return {path.name: file_sha256(path) if path.is_file() else None for path in paths}


def failure_input_observation(
    expected: Mapping[str, Any], commit: str, sources: Mapping[str, str],
    audit_sha: str, row_receipt_sha: str,
) -> dict[str, Any]:
    """Return an exact stable-input replay or an explicit, reproducible drift record."""

    try:
        current = protected_snapshot(commit, sources, audit_sha, row_receipt_sha)
    except BaseException as error:
        return {
            "status": "protected_input_validation_failed",
            "error_type": type(error).__name__,
            "error": repr(error),
        }
    return {
        "status": "matches_initial" if current == dict(expected) else "changed",
        "snapshot": current,
    }


def finish_publication_guard(
    expected: Mapping[str, Any], commit: str, sources: Mapping[str, str],
    audit_sha: str, row_receipt_sha: str, claim: Any,
    absent_paths: tuple[Path, ...], message: str,
) -> None:
    """Finish a success guard with replay, then rival check, then claim check."""

    verify_protected(expected, commit, sources, audit_sha, row_receipt_sha, claim)
    if any(path.exists() for path in absent_paths):
        raise RuntimeError(message)
    rows_life.base.require_claim(claim, LOCK)


@torch.no_grad()
def capture_gate_action(
    model: Any, role_rows: torch.Tensor, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    gates, actions = [], []
    calls = {
        "forwards": 0, "outer_returns": 0, "attention_calls": 0,
        "attention_returns": 0, "site1_captures": 0, "native_mlp_calls": 0,
    }
    for start in range(0, len(role_rows), DOCUMENT_BATCH):
        tokens = role_rows[start:start + DOCUMENT_BATCH, :-1].to(device)

        def attention(event: facade.AttentionEvent):
            calls["attention_calls"] += 1
            value = event.block.attn(event.state, event.first_value)
            calls["attention_returns"] += 1
            return value

        def mlp(event: facade.EarlyMLPEvent):
            if event.site == 1:
                gate = event.block.mlp.Left(event.state) * event.block.mlp.Right(event.state)
                action = event.block.mlp.Down(gate)
                gates.append(
                    gate[:, SCORING].detach().float().cpu().reshape(-1, sparse.GATE_DIM)
                )
                actions.append(
                    action[:, SCORING].detach().float().cpu().reshape(-1, sparse.OUTPUT_DIM)
                )
                calls["site1_captures"] += 1
                return action + event.block.mlp.Down_bias
            calls["native_mlp_calls"] += 1
            return event.block.mlp(event.state)

        facade.forward_with_dispatch(model, tokens, attention, mlp)
        calls["forwards"] += 1
        calls["outer_returns"] += 1
    expected = len(role_rows) // DOCUMENT_BATCH
    expected_calls = {
        "forwards": expected,
        "outer_returns": expected,
        "attention_calls": expected * 18,
        "attention_returns": expected * 18,
        "site1_captures": expected,
        "native_mlp_calls": expected * 17,
    }
    if calls != expected_calls:
        raise RuntimeError(f"sparse-Down capture call census changed: {calls}")
    gate_tensor, action_tensor = torch.cat(gates), torch.cat(actions)
    expected_positions = len(role_rows) * SCORING_POSITIONS
    if tuple(gate_tensor.shape) != (expected_positions, sparse.GATE_DIM) \
            or tuple(action_tensor.shape) != (expected_positions, sparse.OUTPUT_DIM):
        raise RuntimeError("sparse-Down scored-position capture changed")
    return gate_tensor, action_tensor, calls


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
    calls = {arm: {
        "forwards": 0, "outer_returns": 0, "attention_calls": 0,
        "attention_returns": 0, "other_native_mlp": 0, "native_mlp1": 0,
        "sparse_mlp1": 0, "zero_mlp1": 0,
    }
             for arm in sums}
    for start in range(0, len(role_rows), DOCUMENT_BATCH):
        batch = role_rows[start:start + DOCUMENT_BATCH]
        tokens = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        for arm in sums:
            def attention(event: facade.AttentionEvent):
                calls[arm]["attention_calls"] += 1
                value = event.block.attn(event.state, event.first_value)
                calls[arm]["attention_returns"] += 1
                return value

            def mlp(event: facade.EarlyMLPEvent, arm=arm):
                if event.site != 1:
                    calls[arm]["other_native_mlp"] += 1
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
            calls[arm]["outer_returns"] += 1
    expected_common = {
        "forwards": EXPECTED_FORWARDS,
        "outer_returns": EXPECTED_FORWARDS,
        "attention_calls": EXPECTED_FORWARDS * 18,
        "attention_returns": EXPECTED_FORWARDS * 18,
        "other_native_mlp": EXPECTED_FORWARDS * 17,
    }
    for arm in sums:
        expected = {
            **expected_common,
            "native_mlp1": EXPECTED_FORWARDS if arm == "NATIVE" else 0,
            "sparse_mlp1": EXPECTED_FORWARDS if arm == "SPARSE" else 0,
            "zero_mlp1": EXPECTED_FORWARDS if arm == "ZERO" else 0,
        }
        if calls[arm] != expected:
            raise RuntimeError(f"sparse-Down SELECT call census changed for {arm}: {calls[arm]}")
    result = {arm: sums[arm] / counts[arm] for arm in sums}
    denominator = result["ZERO"] - result["NATIVE"]
    if not all(math.isfinite(value) for value in result.values()) \
            or not math.isfinite(denominator) or denominator <= 0:
        raise RuntimeError("sparse-Down SELECT CE baseline is non-finite or non-positive")
    result["recovery"] = (result["ZERO"] - result["SPARSE"]) / denominator
    if not math.isfinite(result["recovery"]):
        raise RuntimeError("sparse-Down SELECT CE recovery is non-finite")
    return result, calls


def assert_state_equal(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    actual_state = sparse.validate_state(actual)
    expected_state = sparse.validate_state(expected)
    if any(not torch.equal(actual_state[key], expected_state[key]) for key in actual_state):
        raise RuntimeError("serialized sparse-Down state changed")


def validate_bundle(
    value: Any, expected_state: Mapping[str, Any], authority_sha: str,
    selected_seed: int,
) -> None:
    if not isinstance(value, dict) or set(value) != {
        "schema", "status", "authority_sha256", "program", "selected_seed",
        "price", "final_opened",
    } or value.get("schema") != "mlp1_sparse_c512_continue_factorial_v1_fit_bundle" \
            or value.get("status") != "selected_program_frozen_before_final" \
            or value.get("authority_sha256") != authority_sha \
            or value.get("selected_seed") != selected_seed \
            or value.get("price") != sparse.SparseDownProgram.price() \
            or value.get("final_opened") is not False:
        raise RuntimeError("serialized sparse-Down bundle semantics changed")
    assert_state_equal(value["program"], expected_state)


def validate_result(value: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    if dict(value) != dict(expected):
        raise RuntimeError("serialized sparse-Down result changed")
    if set(value) != {
        "schema", "status", "runtime_seconds", "documents", "captured_positions",
        "seed_records", "select_ce", "selection_gates", "price", "calls", "parents",
        "checkpoint", "claim_boundary",
    } or value.get("documents") != {"FIT": 96, "SELECT": 96, "FINAL_opened": 0} \
            or value.get("captured_positions") != {
                "FIT": ROLE_DOCUMENTS * SCORING_POSITIONS,
                "SELECT": ROLE_DOCUMENTS * SCORING_POSITIONS,
            } or value.get("price") != sparse.SparseDownProgram.price():
        raise RuntimeError("sparse-Down result schema changed")
    ce = value["select_ce"]
    if set(ce) != {"NATIVE", "ZERO", "SPARSE", "recovery"} \
            or not all(math.isfinite(float(item)) for item in ce.values()) \
            or float(ce["ZERO"]) - float(ce["NATIVE"]) <= 0:
        raise RuntimeError("sparse-Down result CE semantics changed")
    recomputed_recovery = (
        (float(ce["ZERO"]) - float(ce["SPARSE"]))
        / (float(ce["ZERO"]) - float(ce["NATIVE"]))
    )
    if recomputed_recovery != float(ce["recovery"]):
        raise RuntimeError("sparse-Down result recovery replay changed")
    gates = selection_gates(value["seed_records"], recomputed_recovery)
    if gates != value["selection_gates"] or value["status"] != (
        "admitted_to_final" if gates["admitted_to_final"] else "selection_gate_failed"
    ):
        raise RuntimeError("sparse-Down result decision replay changed")


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
        validate_row_receipt(row_receipt, sources, audit, audit_sha)
        protected = protected_snapshot(commit, sources, audit_sha, row_receipt_sha)
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
            "protected_inputs": protected,
        }

        def authority_guard() -> None:
            finish_publication_guard(
                protected, commit, sources, audit_sha, row_receipt_sha, claim,
                (AUTHORITY, BUNDLE, RESULT, RECEIPT, FAILURE),
                "sparse-Down authority namespace raced publication",
            )

        write_json_create_only(AUTHORITY, authority, pre_link_check=authority_guard)
        authority_sha = file_sha256(AUTHORITY)
        authority_replay, _ = stable_json(AUTHORITY, authority_sha)
        if authority_replay != authority:
            raise RuntimeError("serialized sparse-Down authority changed")
        entries = row_receipt["entries"]
        fit = load_role(entries["FIT"])
        select = load_role(entries["SELECT"])
        fit_select_opened = True
        device = torch.device("cuda")
        model, checkpoint = facade.load_bilin18(device=device, dtype=torch.bfloat16)
        if {
            "config_sha256": checkpoint.config_sha256,
            "weights_sha256": checkpoint.weights_sha256,
            "weights_bytes": checkpoint.weights_bytes,
        } != protected["checkpoint"]:
            raise RuntimeError("loaded checkpoint differs from protected checkpoint")
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

        def bundle_guard() -> None:
            stable_json(AUTHORITY, authority_sha)
            finish_publication_guard(
                protected, commit, sources, audit_sha, row_receipt_sha, claim,
                (BUNDLE, RESULT, RECEIPT, FAILURE),
                "sparse-Down bundle namespace raced publication",
            )

        write_torch_create_only(BUNDLE, bundle, pre_link_check=bundle_guard)
        bundle_sha = file_sha256(BUNDLE)
        replay, _ = stable_torch(BUNDLE, bundle_sha)
        validate_bundle(replay, selected_state, authority_sha, gates["selected_seed"])
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

        def result_guard() -> None:
            stable_json(AUTHORITY, authority_sha)
            bundle_replay, _ = stable_torch(BUNDLE, bundle_sha)
            validate_bundle(bundle_replay, selected_state, authority_sha, gates["selected_seed"])
            finish_publication_guard(
                protected, commit, sources, audit_sha, row_receipt_sha, claim,
                (RESULT, RECEIPT, FAILURE),
                "sparse-Down result namespace raced publication",
            )

        write_json_create_only(RESULT, result, pre_link_check=result_guard)
        result_sha = file_sha256(RESULT)
        result_replay, _ = stable_json(RESULT, result_sha)
        validate_result(result_replay, result)
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
        print(json.dumps(result, indent=2, sort_keys=True))

        def receipt_guard() -> None:
            authority_again, _ = stable_json(AUTHORITY, authority_sha)
            if authority_again != authority:
                raise RuntimeError("sparse-Down authority replay changed before receipt")
            bundle_again, _ = stable_torch(BUNDLE, bundle_sha)
            validate_bundle(bundle_again, selected_state, authority_sha, gates["selected_seed"])
            result_again, _ = stable_json(RESULT, result_sha)
            validate_result(result_again, result)
            finish_publication_guard(
                protected, commit, sources, audit_sha, row_receipt_sha, claim,
                (RECEIPT, FAILURE),
                "sparse-Down terminal raced receipt publication",
            )

        write_json_create_only(RECEIPT, receipt, pre_link_check=receipt_guard)
    except BaseException as error:
        if not RECEIPT.exists() and not FAILURE.exists():
            partial_paths = (AUTHORITY, BUNDLE, RESULT)
            partial_snapshot = artifact_snapshot(partial_paths)
            has_protected = "protected" in locals()
            input_observation = failure_input_observation(
                protected, commit, sources, audit_sha, row_receipt_sha,
            ) if has_protected else {
                "status": "initial_protected_snapshot_not_constructed",
            }
            failure = {
                "schema": "mlp1_sparse_c512_continue_factorial_v1_fit_failure",
                "status": "terminal_failure",
                "error": repr(error),
                "fit_select_may_have_opened": fit_select_opened,
                "final_may_have_opened": final_opened,
                "artifact_snapshot": partial_snapshot,
                "initial_protected_inputs": protected if has_protected else None,
                "protected_inputs_at_failure": input_observation,
            }

            def failure_guard() -> None:
                rows_life.base.require_claim(claim, LOCK)
                if RECEIPT.exists() or FAILURE.exists():
                    raise RuntimeError("sparse-Down terminal raced failure publication")
                if artifact_snapshot(partial_paths) != partial_snapshot:
                    raise RuntimeError("sparse-Down partial artifact aggregate changed before failure")
                if has_protected and failure_input_observation(
                    protected, commit, sources, audit_sha, row_receipt_sha,
                ) != input_observation:
                    raise RuntimeError("sparse-Down protected failure state changed before publication")
                if RECEIPT.exists() or FAILURE.exists():
                    raise RuntimeError("sparse-Down terminal raced failure publication")
                rows_life.base.require_claim(claim, LOCK)

            write_json_create_only(FAILURE, failure, pre_link_check=failure_guard)
        raise
    finally:
        rows_life.base.release_claim(claim, LOCK)


if __name__ == "__main__":
    main()
