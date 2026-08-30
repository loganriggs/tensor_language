#!/usr/bin/env python3
"""Resumable, source-closed training-only grid for factorization v1.

The production entrypoint accepts no paths, ranks, seeds, or role choices.  Each
candidate/seed cell is published atomically and create-only, so interruption cannot
erase completed work.  Validation and EVAL are not imported or represented here.
"""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time
from typing import Callable, Sequence

import torch

from causal_response_factorization_v1 import FitResult
from causal_response_factorization_v1 import predict_from_codes
from causal_response_factorization_v1_accelerated import (
    fit_shared_private_program_accelerated,
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


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "causal_response_factorization_v1_grid_results"
STEPS = 2_000
LEARNING_RATE = 0.03
MINIMUM_IMPROVEMENT = 1e-4
SOURCE_PATHS = tuple(HERE / name for name in (
    "causal_response_factorization_v1.py",
    "causal_response_factorization_v1_accelerated.py",
    "causal_response_factorization_v1_candidate_price_audit.py",
    "causal_response_factorization_v1_fit_adapter.py",
    "causal_response_factorization_v1_training_snapshot.py",
    "causal_response_factorization_v1_grid_runner.py",
    "CAUSAL_RESPONSE_FACTORIZATION_V1_PREREGISTRATION.md",
    *((f"CAUSAL_RESPONSE_FACTORIZATION_V1_AMENDMENT_{index}.md") for index in range(1, 13)),
))


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
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
    ).strip()
    if require_published and subprocess.run(
        ["git", "merge-base", "--is-ancestor", head, "origin/main"], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode != 0:
        raise RuntimeError("factor-grid source commit is not published")
    hashes: dict[str, str] = {}
    for path in SOURCE_PATHS:
        relative = str(path.relative_to(ROOT))
        raw = path.read_bytes()
        if require_published:
            committed = subprocess.run(
                ["git", "show", f"{head}:{relative}"], cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            if committed.returncode != 0 or committed.stdout != raw:
                raise RuntimeError(f"factor-grid source is not exact at HEAD: {relative}")
        hashes[relative] = _sha256(raw)
    body: dict[str, object] = {"commit": head, "paths": hashes}
    return {**body, "sha256": _logical_sha256(body)}


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


def _atomic_create(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            sink.write(raw)
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_torch_create(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(value, temporary)
        with temporary.open("rb") as source:
            os.fsync(source.fileno())
        os.link(temporary, path)
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
            pair_mask = torch.zeros_like(training.valid)
            pair_mask[
                :, training.source_groups == source_owner,
                training.source_groups == target_owner,
            ] = True
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
    """Run or resume a fixed training-only grid and publish its terminal receipt."""

    if output.exists() and not output.is_dir():
        raise RuntimeError("factor-grid output namespace is not a directory")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return _run_grid_locked(
            training, output, rank_pairs=rank_pairs, seeds=seeds, steps=steps,
            learning_rate=learning_rate, optimizer_device=optimizer_device,
            require_published_source=require_published_source, fitter=fitter,
        )


def _run_grid_locked(
    training: FitTrainingInput,
    output: Path,
    *,
    rank_pairs: Sequence[tuple[int, int]], seeds: Sequence[int], steps: int,
    learning_rate: float, optimizer_device: str, require_published_source: bool,
    fitter: Callable[..., FitResult],
) -> dict[str, object]:
    source = _source_closure(require_published=require_published_source)
    input_binding = _input_binding(training)
    price_lookup = {
        (row.global_rank, row.private_rank_each_owner): row for row in audit_rows()
    }
    cells: list[dict[str, object]] = []
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
                payload = _load_torch_cell(result_path)
                receipt = payload["receipt"]
                if receipt["source_closure_sha256"] != source["sha256"] or (
                    receipt["input_binding_sha256"] != input_binding["sha256"]
                    or receipt["global_rank"] != global_rank
                    or receipt["private_rank_each_owner"] != private_rank
                    or receipt["seed"] != seed
                    or receipt["steps"] != steps
                    or receipt["learning_rate"] != learning_rate
                ):
                    raise RuntimeError(f"resumed grid cell binding changed: {stem}")
                raw = result_path.read_bytes()
                cells.append({**receipt, "kind": "result", "artifact": result_path.name,
                              "artifact_sha256": _sha256(raw), "bytes": len(raw)})
                continue
            if failure_path.exists():
                failure = json.loads(failure_path.read_bytes())
                if failure.get("source_closure_sha256") != source["sha256"] or (
                    failure.get("input_binding_sha256") != input_binding["sha256"]
                ):
                    raise RuntimeError(f"resumed failure binding changed: {stem}")
                raw = failure_path.read_bytes()
                cells.append({**failure, "kind": "failure", "artifact": failure_path.name,
                              "artifact_sha256": _sha256(raw), "bytes": len(raw)})
                continue
            started = time.perf_counter()
            try:
                fitted = fitter(
                    training.response, training.valid, training.source_groups,
                    global_rank=global_rank, private_rank=private_rank, seed=seed,
                    steps=steps, learning_rate=learning_rate,
                    optimizer_device=optimizer_device,
                )
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
                _atomic_torch_create(result_path, payload)
            except Exception as error:
                if result_path.exists():
                    raise RuntimeError(
                        f"published grid result failed post-publication checks: {stem}"
                    ) from error
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
                _atomic_create(failure_path, _json_bytes(failure))
                raw = failure_path.read_bytes()
                cells.append({**failure, "kind": "failure", "artifact": failure_path.name,
                              "artifact_sha256": _sha256(raw), "bytes": len(raw)})
                continue
            replay = _load_torch_cell(result_path)
            if replay["receipt"] != receipt:
                raise RuntimeError("new grid cell did not replay exactly")
            raw = result_path.read_bytes()
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
    terminal_path = output / "terminal.json"
    raw = _json_bytes(terminal)
    if terminal_path.exists():
        if terminal_path.read_bytes() != raw:
            raise RuntimeError("factor-grid terminal namespace is already spent differently")
    else:
        _atomic_create(terminal_path, raw)
    if json.loads(terminal_path.read_bytes()) != terminal:
        raise RuntimeError("factor-grid terminal receipt did not replay")
    expected_names = {
        ".lock", "terminal.json", *(cell["artifact"] for cell in cells),
    }
    if {path.name for path in output.iterdir()} != expected_names:
        raise RuntimeError("factor-grid terminal directory census changed")
    return terminal


def main() -> None:
    training = load_production_training_snapshot()
    terminal = run_grid(
        training, OUTPUT, rank_pairs=RANK_PAIRS, seeds=SEEDS, steps=STEPS,
        learning_rate=LEARNING_RATE, optimizer_device="cuda",
        require_published_source=True,
    )
    print(_json_bytes(terminal).decode(), end="")


if __name__ == "__main__":
    main()
