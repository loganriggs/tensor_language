#!/usr/bin/env python3
"""Resumable, source-closed training-only grid for factorization v1.

The production entrypoint accepts no paths, ranks, seeds, or role choices.  Each
candidate/seed cell is published atomically and create-only, so interruption cannot
erase completed work.  Validation and EVAL are not imported or represented here.
"""

from __future__ import annotations

from dataclasses import asdict
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import stat
import tempfile
import time
from typing import Callable, Sequence

import torch

from causal_response_factorization_v1 import FitResult, ResponseProgram, predict_from_codes
from causal_response_factorization_v1_accelerated import (
    fit_shared_private_program_accelerated,
    seeded_initial_mse,
)
from causal_response_factorization_v1_candidate_price_audit import (
    RANK_PAIRS,
    SEEDS,
    audit_rows,
)
from causal_response_factorization_v1_fit_adapter import FitTrainingInput
from causal_response_factorization_v1_training_snapshot import (
    load_production_training_snapshot,
)
import causal_response_factorization_v1_training_lifecycle as training_lifecycle


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "causal_response_factorization_v1_grid_results"
GRID_AUDIT = HERE / "causal_response_factorization_v1_grid_independent_audit.json"
STEPS = 2_000
LEARNING_RATE = 0.03
MINIMUM_IMPROVEMENT = 1e-4
NUMERICAL_FAILURE_MESSAGES = {
    "accelerated shared/private optimizer became nonfinite",
    "accelerated canonical replay ended nonfinite",
}
SOURCE_PATHS = tuple(dict.fromkeys((*training_lifecycle.SOURCE_PATHS, *(
    HERE / "causal_response_factorization_v1_candidate_price_audit.py",
    HERE / "causal_response_factorization_v1_candidate_price_audit.json",
    HERE / "causal_response_factorization_v1_grid_runner.py",
    HERE / "test_causal_response_factorization_v1_grid_runner.py",
    HERE / "CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_12.md",
    HERE / "CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_13.md",
))))


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _json_bytes(value: object) -> bytes:
    return (json.dumps(
        value, sort_keys=True, indent=2, allow_nan=False,
    ) + "\n").encode()


def _logical_sha256(value: object) -> str:
    return _sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode())


def _tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    header = json.dumps({
        "dtype": str(value.dtype), "shape": list(value.shape),
    }, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(header + b"\0" + value.numpy().tobytes())


def _source_closure(*, require_published: bool) -> dict[str, object]:
    hashes: dict[str, str] = {}
    paths = SOURCE_PATHS + ((GRID_AUDIT,) if require_published else ())
    for path in paths:
        relative = str(path.relative_to(ROOT))
        raw = path.read_bytes()
        hashes[relative] = _sha256(raw)
    if require_published:
        audit_raw = GRID_AUDIT.read_bytes()
        audit = json.loads(audit_raw)
        body: dict[str, object] = {
            "audited_source_commit": audit.get("audited_source_commit"),
            "independent_audit_sha256": _sha256(audit_raw),
            "paths": hashes,
        }
    else:
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        ).strip()
        body = {"commit": head, "paths": hashes}
    return {**body, "sha256": _logical_sha256(body)}


def _validate_grid_audit(source: dict[str, object]) -> None:
    raw = GRID_AUDIT.read_bytes()
    audit = json.loads(raw)
    required = {
        "schema", "status", "approved", "reviewer", "outcome_access",
        "audited_source_commit", "audited_source_hashes", "tests_passed",
        "remaining_execution_blockers",
    }
    source_hashes = dict(source["paths"])
    source_hashes.pop(str(GRID_AUDIT.relative_to(ROOT)))
    if (
        type(audit) is not dict or set(audit) != required
        or audit.get("schema") != "causal_response_factorization_v1_grid_independent_audit"
        or audit.get("status") != "GO" or audit.get("approved") is not True
        or audit.get("outcome_access") is not False
        or not isinstance(audit.get("reviewer"), str) or not audit["reviewer"]
        or not isinstance(audit.get("tests_passed"), int) or audit["tests_passed"] < 1
        or audit.get("remaining_execution_blockers") != []
        or audit.get("audited_source_hashes") != source_hashes
        or source.get("audited_source_commit") != audit.get("audited_source_commit")
        or source.get("independent_audit_sha256") != _sha256(raw)
    ):
        raise RuntimeError("factor-grid independent audit is not exact GO")
    audited_commit = audit["audited_source_commit"]
    resolved = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{audited_commit}^{{commit}}"],
        cwd=ROOT, text=True,
    ).strip()
    if resolved != audited_commit or subprocess.run(
        ["git", "merge-base", "--is-ancestor", audited_commit, "origin/main"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("factor-grid audited commit is not published")
    audit_relative = str(GRID_AUDIT.relative_to(ROOT))
    published_audit = subprocess.run(
        ["git", "show", f"origin/main:{audit_relative}"], cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    if published_audit.returncode != 0 or published_audit.stdout != raw:
        raise RuntimeError("factor-grid independent audit blob is not published")
    for relative, digest in source_hashes.items():
        committed = subprocess.run(
            ["git", "show", f"{audited_commit}:{relative}"], cwd=ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        if committed.returncode != 0 or _sha256(committed.stdout) != digest:
            raise RuntimeError(f"factor-grid audit source mismatch: {relative}")


@contextmanager
def _output_lock(path: Path):
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        opened = os.fstat(descriptor)
        named = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev, opened.st_ino
        ) != (named.st_dev, named.st_ino):
            raise RuntimeError("factor-grid lock is not one stable regular inode")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        named = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("factor-grid lock pathname changed during acquisition")
        yield
        named = path.stat(follow_symlinks=False)
        if (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino):
            raise RuntimeError("factor-grid lock pathname changed while owned")
    finally:
        os.close(descriptor)


def _input_binding(value: FitTrainingInput) -> dict[str, object]:
    body: dict[str, object] = {
        "artifact_binding": asdict(value.artifacts),
        "response_sha256": _tensor_sha256(value.response),
        "valid_sha256": _tensor_sha256(value.valid),
        "document_ids_sha256": _tensor_sha256(value.document_ids),
        "original_document_indices_sha256": _tensor_sha256(
            value.original_document_indices
        ),
        "source_groups_sha256": _tensor_sha256(value.source_groups),
        "shape": list(value.response.shape),
        "owner_components": list(value.owner_components),
        "phases": list(value.phases),
        "source_tags": list(value.source_tags),
        "target_tags": list(value.target_tags),
        "validation_values_read": False,
        "eval_values_read": False,
    }
    return {**body, "sha256": _logical_sha256(body)}


def _program_payload(result: FitResult) -> dict[str, object]:
    program = result.program
    return {
        "schema": "causal_response_factorization_v1_grid_cell",
        "status": "complete_training_only",
        "program": {
            "global_phase": program.global_phase,
            "global_source": program.global_source,
            "global_target": program.global_target,
            "private_phase": program.private_phase,
            "private_source": program.private_source,
            "private_target": program.private_target,
            "source_groups": program.source_groups,
        },
        "document_codes": result.document_codes,
        "metrics": {
            "initial_mse": result.initial_mse,
            "final_mse": result.final_mse,
            "improvement_fraction": result.improvement_fraction,
            "steps": result.steps,
            "seed": result.seed,
        },
    }


def _validated_bytes_create(
    path: Path,
    raw: bytes,
    validator: Callable[[Path], object],
    *,
    before_link: Callable[[Path], None] | None = None,
) -> object:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())
        validated = validator(temporary)
        if before_link is not None:
            before_link(temporary)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return validated
    finally:
        temporary.unlink(missing_ok=True)


def _validated_torch_create(
    path: Path, value: object, validator: Callable[[Path], object],
) -> object:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb+") as source:
            os.fsync(source.fileno())
        validated = validator(temporary)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return validated
    finally:
        temporary.unlink(missing_ok=True)


def _cell_stem(global_rank: int, private_rank: int, seed: int) -> str:
    return f"g{global_rank:02d}_p{private_rank:02d}_s{seed}"


def _load_torch_cell(path: Path) -> dict[str, object]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if type(value) is not dict or value.get("schema") != (
        "causal_response_factorization_v1_grid_cell"
    ) or value.get("status") != "complete_training_only":
        raise RuntimeError(f"grid cell schema changed: {path.name}")
    return value


def _validate_result_cell(
    path: Path,
    training: FitTrainingInput,
    *,
    source_sha256: str,
    input_sha256: str,
    global_rank: int,
    private_rank: int,
    seed: int,
    steps: int,
    learning_rate: float,
    optimizer_device: str,
) -> tuple[dict[str, object], bytes]:
    payload = _load_torch_cell(path)
    if set(payload) != {"schema", "status", "program", "document_codes", "metrics", "receipt"}:
        raise RuntimeError(f"grid result payload keys changed: {path.name}")
    program_payload = payload["program"]
    if type(program_payload) is not dict or set(program_payload) != {
        "global_phase", "global_source", "global_target", "private_phase",
        "private_source", "private_target", "source_groups",
    }:
        raise RuntimeError(f"grid result program schema changed: {path.name}")
    program = ResponseProgram(**program_payload)
    if program.shape != training.response.shape[:3] or (
        program.global_phase.shape[1] != global_rank
        or len(program.private_phase) != len(training.owner_components)
        or any(block.shape[1] != private_rank for block in program.private_phase)
    ):
        raise RuntimeError(f"grid result registered ranks changed: {path.name}")
    codes = payload["document_codes"]
    if type(codes) is not torch.Tensor or codes.dtype != torch.float64 or (
        codes.device.type != "cpu" or not codes.is_contiguous()
        or codes.shape != (training.response.shape[-1], program.code_dimension)
        or not bool(torch.isfinite(codes).all())
    ):
        raise RuntimeError(f"grid result codes changed: {path.name}")
    if not torch.equal(program.source_groups, training.source_groups):
        raise RuntimeError(f"grid result owner topology changed: {path.name}")
    metrics = payload["metrics"]
    receipt = payload["receipt"]
    if type(metrics) is not dict or set(metrics) != {
        "initial_mse", "final_mse", "improvement_fraction", "steps", "seed",
    } or type(receipt) is not dict:
        raise RuntimeError(f"grid result metrics/receipt schema changed: {path.name}")
    report_keys = {
        "training_response_rms", "normalized_training_mse", "phase_mse",
        "source_owner_mse", "target_owner_mse", "owner_pair_nrmse",
        "worst_owner_pair_nrmse",
    }
    receipt_keys = {
        "source_closure_sha256", "input_binding_sha256", "global_rank",
        "private_rank_each_owner", "seed", "steps", "learning_rate",
        "optimizer_device", "persistent_values", "per_document_values",
        "amortized_total_values", "strict_dense_matched_rank",
        "amortized_total_dense_rank_noncontrolling",
        "prediction_multiply_adds_per_document", "calibration_cells_training_stage",
        "registered_validation_calibration_arm_budgets", "initial_mse", "final_mse",
        "registered_validation_calibration_costs",
        "improvement_fraction", "healthy", "minimum_improvement", "elapsed_seconds",
        "validation_values_read", "eval_values_read", *report_keys,
    }
    if set(receipt) != receipt_keys:
        raise RuntimeError(f"grid result receipt keys changed: {path.name}")
    fixed = {
        "source_closure_sha256": source_sha256,
        "input_binding_sha256": input_sha256,
        "global_rank": global_rank,
        "private_rank_each_owner": private_rank,
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "optimizer_device": optimizer_device,
        "validation_values_read": False,
        "eval_values_read": False,
    }
    if any(receipt.get(key) != value for key, value in fixed.items()) or (
        metrics["seed"] != seed or metrics["steps"] != steps
        or metrics["initial_mse"] != receipt.get("initial_mse")
        or metrics["final_mse"] != receipt.get("final_mse")
        or metrics["improvement_fraction"] != receipt.get("improvement_fraction")
    ):
        raise RuntimeError(f"grid result fixed binding changed: {path.name}")
    replay = predict_from_codes(program.basis(), codes).reshape_as(training.response)
    replay_mse = float(((replay[training.valid] - training.response[training.valid]) ** 2).mean())
    initial_mse = seeded_initial_mse(
        training.response, training.valid, training.source_groups,
        global_rank=global_rank, private_rank=private_rank, seed=seed,
    )
    improvement = (initial_mse - replay_mse) / max(
        initial_mse, torch.finfo(torch.float64).tiny,
    )
    fitted = FitResult(
        program=program, document_codes=codes, initial_mse=initial_mse,
        final_mse=replay_mse, improvement_fraction=improvement, steps=steps, seed=seed,
    )
    expected_report = _training_error_report(training, fitted)
    code_offset = 0
    blocks = []
    if global_rank:
        blocks.append((
            (program.global_phase, program.global_source, program.global_target),
            codes[:, :global_rank],
        ))
    code_offset += global_rank
    for group in range(len(program.private_phase)):
        if private_rank:
            blocks.append((
                (
                    program.private_phase[group], program.private_source[group],
                    program.private_target[group],
                ),
                codes[:, code_offset:code_offset + private_rank],
            ))
        code_offset += private_rank

    def block_is_canonical(
        factors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        block_codes: torch.Tensor,
    ) -> bool:
        rank = block_codes.shape[1]
        keys = []
        for factor in factors:
            if not bool(torch.allclose(
                factor.norm(dim=0), torch.ones(rank, dtype=torch.float64),
                atol=1e-12, rtol=0,
            )):
                return False
            pivots = factor.abs().argmax(dim=0)
            if not bool((factor[pivots, torch.arange(rank)] > 0).all()):
                return False
        for column in range(rank):
            packed = torch.cat([factor[:, column] for factor in factors])
            keys.append(hashlib.sha256(packed.numpy().tobytes()).digest())
        return keys == sorted(keys)

    canonical = all(
        block_is_canonical(factors, block_codes) for factors, block_codes in blocks
    )
    price_row = {
        (row.global_rank, row.private_rank_each_owner): row for row in audit_rows()
    }.get((global_rank, private_rank))
    checks = {
        "final_mse": replay_mse == receipt.get("final_mse"),
        "initial_mse": initial_mse == receipt.get("initial_mse"),
        "improvement": improvement == receipt.get("improvement_fraction"),
        "error_report": all(
            receipt.get(key) == value for key, value in expected_report.items()
        ),
        "persistent_price": receipt.get("persistent_values") == program.persistent_values,
        "code_price": receipt.get("per_document_values") == program.code_dimension,
        "amortized_price": receipt.get("amortized_total_values") == (
            program.persistent_values + training.response.shape[-1] * program.code_dimension
        ),
        "prediction_cost": receipt.get("prediction_multiply_adds_per_document") == (
            training.response.shape[0] * training.response.shape[1]
            * training.response.shape[2] * program.code_dimension
        ),
        "calibration_stage": receipt.get("calibration_cells_training_stage") == 0,
        "calibration_budgets": (
            receipt.get("registered_validation_calibration_arm_budgets") == [2, 4, 8, 16]
        ),
        "calibration_costs": receipt.get("registered_validation_calibration_costs") == [
            {
                "arms": arms,
                "cells": arms * training.response.shape[2],
                "normal_equation_multiply_add_upper_bound": (
                    arms * training.response.shape[2] * program.code_dimension
                    * (program.code_dimension + 1) + program.code_dimension ** 3
                ),
            }
            for arms in (2, 4, 8, 16)
        ],
        "health_threshold": receipt.get("minimum_improvement") == MINIMUM_IMPROVEMENT,
        "elapsed": (
            isinstance(receipt.get("elapsed_seconds"), (int, float))
            and math.isfinite(receipt["elapsed_seconds"])
            and receipt["elapsed_seconds"] >= 0
        ),
        "control_prices": price_row is None or (
            receipt.get("strict_dense_matched_rank") == price_row.strict_dense_matched_rank
            and receipt.get("amortized_total_dense_rank_noncontrolling")
            == price_row.amortized_total_dense_rank
        ),
        "health": receipt.get("healthy") is (
            math.isfinite(replay_mse) and improvement >= MINIMUM_IMPROVEMENT
        ),
        "canonical_gauge": canonical,
    }
    failed = sorted(label for label, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError(
            f"grid result semantic replay changed ({','.join(failed)}): {path.name}"
        )
    raw = path.read_bytes()
    return receipt, raw


def _validate_failure_cell(
    path: Path,
    *,
    source_sha256: str,
    input_sha256: str,
    global_rank: int,
    private_rank: int,
    seed: int,
    steps: int,
    learning_rate: float,
    optimizer_device: str,
    registered_only: bool,
) -> tuple[dict[str, object], bytes]:
    value = json.loads(path.read_bytes())
    required = {
        "schema", "status", "source_closure_sha256", "input_binding_sha256",
        "global_rank", "private_rank_each_owner", "seed", "steps", "learning_rate",
        "optimizer_device", "elapsed_seconds", "error_type", "error_message",
        "validation_values_read", "eval_values_read",
    }
    fixed = {
        "schema": "causal_response_factorization_v1_grid_failure",
        "status": "failed_training_only", "source_closure_sha256": source_sha256,
        "input_binding_sha256": input_sha256, "global_rank": global_rank,
        "private_rank_each_owner": private_rank, "seed": seed, "steps": steps,
        "learning_rate": learning_rate, "optimizer_device": optimizer_device,
        "validation_values_read": False, "eval_values_read": False,
    }
    if type(value) is not dict or set(value) != required or any(
        value.get(key) != item for key, item in fixed.items()
    ) or not isinstance(value["error_type"], str) or not value["error_type"] or (
        not isinstance(value["error_message"], str)
        or not isinstance(value["elapsed_seconds"], (int, float))
        or not math.isfinite(value["elapsed_seconds"])
        or value["elapsed_seconds"] < 0
    ):
        raise RuntimeError(f"grid failure semantic replay changed: {path.name}")
    if registered_only and (
        value["error_type"] != "RuntimeError"
        or value["error_message"] not in NUMERICAL_FAILURE_MESSAGES
    ):
        raise RuntimeError(f"grid failure is not a registered numerical outcome: {path.name}")
    raw = path.read_bytes()
    return value, raw


def _mean_controls(response: torch.Tensor, valid: torch.Tensor) -> dict[str, float | int]:
    strict_rank_zero_mse = float((response[valid] ** 2).mean())
    counts = valid.sum(dim=-1)
    means = torch.zeros(response.shape[:-1], dtype=torch.float64)
    supported = counts > 0
    means[supported] = (response * valid).sum(dim=-1)[supported] / counts[supported]
    error = (response - means[..., None])[valid]
    return {
        "strict_dense_rank_zero_mse": strict_rank_zero_mse,
        "observationwise_training_mean_mse": float((error ** 2).mean()),
        "observationwise_training_mean_persistent_values": int(means.numel()),
    }


def _training_error_report(
    training: FitTrainingInput, fitted: FitResult,
) -> dict[str, object]:
    prediction = predict_from_codes(
        fitted.program.basis(), fitted.document_codes,
    ).reshape_as(training.response)
    squared = (prediction - training.response) ** 2
    response_rms = math.sqrt(float((training.response[training.valid] ** 2).mean()))
    if not math.isfinite(response_rms) or response_rms <= 0:
        raise RuntimeError("training response RMS is not positive and finite")

    def masked_mse(mask: torch.Tensor) -> float:
        selected = training.valid & mask
        if not bool(selected.any()):
            raise RuntimeError("registered training error slice has no valid cells")
        return float(squared[selected].mean())

    phase_mse = []
    for phase in range(training.response.shape[0]):
        mask = torch.zeros_like(training.valid)
        mask[phase] = True
        phase_mse.append(masked_mse(mask))
    source_owner_mse = []
    target_owner_mse = []
    owner_pair_nrmse = []
    for source_owner in range(len(training.owner_components)):
        source_mask = torch.zeros_like(training.valid)
        source_mask[:, training.source_groups == source_owner] = True
        source_owner_mse.append(masked_mse(source_mask))
        target_mask = torch.zeros_like(training.valid)
        target_mask[:, :, training.source_groups == source_owner] = True
        target_owner_mse.append(masked_mse(target_mask))
        row = []
        for target_owner in range(len(training.owner_components)):
            source_axis = (training.source_groups == source_owner)[None, :, None, None]
            target_axis = (training.source_groups == target_owner)[None, None, :, None]
            pair_mask = (source_axis & target_axis).expand_as(training.valid)
            row.append(math.sqrt(masked_mse(pair_mask)) / response_rms)
        owner_pair_nrmse.append(row)
    return {
        "training_response_rms": response_rms,
        "normalized_training_mse": fitted.final_mse / (response_rms ** 2),
        "phase_mse": phase_mse,
        "source_owner_mse": source_owner_mse,
        "target_owner_mse": target_owner_mse,
        "owner_pair_nrmse": owner_pair_nrmse,
        "worst_owner_pair_nrmse": max(max(row) for row in owner_pair_nrmse),
    }


def run_grid(
    training: FitTrainingInput,
    output: Path,
    *,
    rank_pairs: Sequence[tuple[int, int]],
    seeds: Sequence[int],
    steps: int,
    learning_rate: float,
    optimizer_device: str,
    require_published_source: bool,
    fitter: Callable[..., FitResult] = fit_shared_private_program_accelerated,
) -> dict[str, object]:
    """Synthetic-only grid surface used by source-isolated acceptance tests."""

    if require_published_source:
        raise RuntimeError("published production fitting is available only through main()")
    if output.exists() and not output.is_dir():
        raise RuntimeError("factor-grid output namespace is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".lock"
    with _output_lock(lock_path):
        return _run_grid_locked(
            training, output, rank_pairs=rank_pairs, seeds=seeds, steps=steps,
            learning_rate=learning_rate, optimizer_device=optimizer_device,
            require_published_source=False, fitter=fitter, source_override=None,
        )


def _run_grid_locked(
    training: FitTrainingInput,
    output: Path,
    *,
    rank_pairs: Sequence[tuple[int, int]], seeds: Sequence[int], steps: int,
    learning_rate: float, optimizer_device: str, require_published_source: bool,
    fitter: Callable[..., FitResult], source_override: dict[str, object] | None,
) -> dict[str, object]:
    if require_published_source and (
        output.resolve() != OUTPUT.resolve()
        or tuple(rank_pairs) != RANK_PAIRS or tuple(seeds) != SEEDS
        or steps != STEPS or learning_rate != LEARNING_RATE
        or optimizer_device != "cuda"
        or fitter is not fit_shared_private_program_accelerated
        or source_override is None
    ):
        raise RuntimeError("production factor-grid protocol changed")
    source = source_override or _source_closure(require_published=False)
    input_binding = _input_binding(training)
    price_lookup = {
        (row.global_rank, row.private_rank_each_owner): row for row in audit_rows()
    }
    cells: list[dict[str, object]] = []
    terminal_path = output / "terminal.json"
    terminal_preexists = terminal_path.exists()
    for global_rank, private_rank in rank_pairs:
        if (global_rank, private_rank) not in price_lookup and require_published_source:
            raise RuntimeError("production grid rank pair differs from the frozen audit")
        for seed in seeds:
            stem = _cell_stem(global_rank, private_rank, seed)
            result_path = output / f"{stem}.pt"
            failure_path = output / f"{stem}.failure.json"
            if result_path.exists() and failure_path.exists():
                raise RuntimeError(f"grid cell has two terminal states: {stem}")
            if result_path.exists():
                receipt, raw = _validate_result_cell(
                    result_path, training, source_sha256=source["sha256"],
                    input_sha256=input_binding["sha256"], global_rank=global_rank,
                    private_rank=private_rank, seed=seed, steps=steps,
                    learning_rate=learning_rate, optimizer_device=optimizer_device,
                )
                cells.append({**receipt, "kind": "result", "artifact": result_path.name,
                              "artifact_sha256": _sha256(raw), "bytes": len(raw)})
                continue
            if failure_path.exists():
                failure, raw = _validate_failure_cell(
                    failure_path, source_sha256=source["sha256"],
                    input_sha256=input_binding["sha256"], global_rank=global_rank,
                    private_rank=private_rank, seed=seed, steps=steps,
                    learning_rate=learning_rate, optimizer_device=optimizer_device,
                    registered_only=require_published_source,
                )
                cells.append({**failure, "kind": "failure", "artifact": failure_path.name,
                              "artifact_sha256": _sha256(raw), "bytes": len(raw)})
                continue
            if terminal_preexists:
                raise RuntimeError(f"terminal grid is missing registered cell: {stem}")
            started = time.perf_counter()
            try:
                fitted = fitter(
                    training.response, training.valid, training.source_groups,
                    global_rank=global_rank, private_rank=private_rank, seed=seed,
                    steps=steps, learning_rate=learning_rate,
                    optimizer_device=optimizer_device,
                )
            except Exception as error:
                if require_published_source and not (
                    type(error) is RuntimeError
                    and str(error) in NUMERICAL_FAILURE_MESSAGES
                ):
                    raise
                failure = {
                    "schema": "causal_response_factorization_v1_grid_failure",
                    "status": "failed_training_only",
                    "source_closure_sha256": source["sha256"],
                    "input_binding_sha256": input_binding["sha256"],
                    "global_rank": global_rank,
                    "private_rank_each_owner": private_rank,
                    "seed": seed,
                    "steps": steps,
                    "learning_rate": learning_rate,
                    "optimizer_device": optimizer_device,
                    "elapsed_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "validation_values_read": False,
                    "eval_values_read": False,
                }
                failure, raw = _validated_bytes_create(
                    failure_path, _json_bytes(failure),
                    lambda path: _validate_failure_cell(
                        path, source_sha256=source["sha256"],
                        input_sha256=input_binding["sha256"], global_rank=global_rank,
                        private_rank=private_rank, seed=seed, steps=steps,
                        learning_rate=learning_rate, optimizer_device=optimizer_device,
                        registered_only=require_published_source,
                    ),
                )
                cells.append({**failure, "kind": "failure", "artifact": failure_path.name,
                              "artifact_sha256": _sha256(raw), "bytes": len(raw)})
                continue
            elapsed = time.perf_counter() - started
            price = fitted.program.persistent_values
            code = fitted.program.code_dimension
            if require_published_source:
                expected = price_lookup[(global_rank, private_rank)]
                if (price, code) != (expected.persistent_values, expected.per_document_values):
                    raise RuntimeError("fitted program price differs from frozen audit")
            healthy = bool(
                math.isfinite(fitted.final_mse)
                and fitted.improvement_fraction >= MINIMUM_IMPROVEMENT
            )
            receipt = {
                "source_closure_sha256": source["sha256"],
                "input_binding_sha256": input_binding["sha256"],
                "global_rank": global_rank,
                "private_rank_each_owner": private_rank,
                "seed": seed,
                "steps": steps,
                "learning_rate": learning_rate,
                "optimizer_device": optimizer_device,
                "persistent_values": price,
                "per_document_values": code,
                "amortized_total_values": price + training.response.shape[-1] * code,
                "strict_dense_matched_rank": (
                    price_lookup[(global_rank, private_rank)].strict_dense_matched_rank
                    if (global_rank, private_rank) in price_lookup else None
                ),
                "amortized_total_dense_rank_noncontrolling": (
                    price_lookup[(global_rank, private_rank)].amortized_total_dense_rank
                    if (global_rank, private_rank) in price_lookup else None
                ),
                "prediction_multiply_adds_per_document": (
                    training.response.shape[0] * training.response.shape[1]
                    * training.response.shape[2] * code
                ),
                "calibration_cells_training_stage": 0,
                "registered_validation_calibration_arm_budgets": [2, 4, 8, 16],
                "registered_validation_calibration_costs": [
                    {
                        "arms": arms,
                        "cells": arms * training.response.shape[2],
                        "normal_equation_multiply_add_upper_bound": (
                            arms * training.response.shape[2] * code * (code + 1)
                            + code ** 3
                        ),
                    }
                    for arms in (2, 4, 8, 16)
                ],
                "initial_mse": fitted.initial_mse,
                "final_mse": fitted.final_mse,
                "improvement_fraction": fitted.improvement_fraction,
                "healthy": healthy,
                "minimum_improvement": MINIMUM_IMPROVEMENT,
                "elapsed_seconds": elapsed,
                "validation_values_read": False,
                "eval_values_read": False,
            }
            receipt.update(_training_error_report(training, fitted))
            payload = _program_payload(fitted)
            payload["receipt"] = receipt
            replay_receipt, raw = _validated_torch_create(
                result_path, payload,
                lambda path: _validate_result_cell(
                    path, training, source_sha256=source["sha256"],
                    input_sha256=input_binding["sha256"], global_rank=global_rank,
                    private_rank=private_rank, seed=seed, steps=steps,
                    learning_rate=learning_rate, optimizer_device=optimizer_device,
                ),
            )
            if replay_receipt != receipt:
                raise RuntimeError("new grid cell did not replay exactly")
            cells.append({**receipt, "kind": "result", "artifact": result_path.name,
                          "artifact_sha256": _sha256(raw), "bytes": len(raw)})
    final_source = _source_closure(require_published=require_published_source)
    if final_source != source:
        raise RuntimeError("factor-grid source closure changed during fitting")
    manifest_body: dict[str, object] = {
        "schema": "causal_response_factorization_v1_grid_terminal",
        "status": "complete_training_only_grid",
        "source_closure": source,
        "input_binding": input_binding,
        "rank_pairs": [list(pair) for pair in rank_pairs],
        "seeds": list(seeds),
        "steps": steps,
        "learning_rate": learning_rate,
        "optimizer_device": optimizer_device,
        "controls": _mean_controls(training.response, training.valid),
        "expected_cells": len(rank_pairs) * len(seeds),
        "result_cells": sum(cell["kind"] == "result" for cell in cells),
        "failure_cells": sum(cell["kind"] == "failure" for cell in cells),
        "healthy_cells": sum(cell.get("healthy") is True for cell in cells),
        "cells": cells,
        "validation_values_read": False,
        "eval_values_read": False,
    }
    terminal = {**manifest_body, "manifest_sha256": _logical_sha256(manifest_body)}
    raw = _json_bytes(terminal)
    if terminal_path.exists():
        if terminal_path.read_bytes() != raw:
            raise RuntimeError("factor-grid terminal namespace is already spent differently")
    else:
        expected_preterminal_names = {".lock", *(cell["artifact"] for cell in cells)}

        def validate_terminal(path: Path) -> tuple[dict[str, object], bytes]:
            staged_raw = path.read_bytes()
            staged = json.loads(staged_raw)
            if staged != terminal:
                raise RuntimeError("staged factor-grid terminal did not replay")
            return staged, staged_raw

        def validate_preterminal_census(staged_path: Path) -> None:
            expected_with_stage = expected_preterminal_names | {staged_path.name}
            if {path.name for path in output.iterdir()} != expected_with_stage:
                raise RuntimeError("factor-grid preterminal directory census changed")

        _validated_bytes_create(
            terminal_path, raw, validate_terminal, before_link=validate_preterminal_census,
        )
    return terminal


def main() -> None:
    if OUTPUT.exists() and not OUTPUT.is_dir():
        raise RuntimeError("production factor-grid namespace is not a directory")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with _output_lock(OUTPUT / ".lock"):
        source = _source_closure(require_published=True)
        _validate_grid_audit(source)
        allowed = {".lock", "terminal.json"}
        for global_rank, private_rank in RANK_PAIRS:
            for seed in SEEDS:
                stem = _cell_stem(global_rank, private_rank, seed)
                allowed.update({f"{stem}.pt", f"{stem}.failure.json"})
        extras = {path.name for path in OUTPUT.iterdir()} - allowed
        if extras:
            raise RuntimeError(f"production factor-grid namespace has extras: {sorted(extras)}")
        training = load_production_training_snapshot()
        if training.response.shape != (2, 49, 49, 229) or len(
            training.owner_components
        ) != 6:
            raise RuntimeError("production factor-grid training role changed")
        terminal = _run_grid_locked(
            training, OUTPUT, rank_pairs=RANK_PAIRS, seeds=SEEDS, steps=STEPS,
            learning_rate=LEARNING_RATE, optimizer_device="cuda",
            require_published_source=True,
            fitter=fit_shared_private_program_accelerated, source_override=source,
        )
    print(_json_bytes(terminal).decode(), end="")


if __name__ == "__main__":
    main()
