#!/usr/bin/env python3
"""FIT-only unfolding lower bounds for owner-conditioned rank-32 residuals."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch

from causal_response_factorization_v1 import ResponseProgram, predict_from_codes
from causal_response_factorization_v1_training_snapshot import load_production_training_snapshot


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
GRID = HERE / "causal_response_factorization_v1_grid_results"
OUTPUT = HERE / "causal_response_residual_unfolding_certificate_receipt.json"
SEEDS = (2026083001, 2026083002, 2026083003)
RANKS = (1, 2, 4, 8, 16, 32)
OWNER_LABELS = ("a8", "a16", "m16", "a3", "m14", "m13")
MODE_LABELS = ("phase", "source", "target", "document")
SOURCE_PATHS = (
    HERE / "causal_response_residual_unfolding_certificate.py",
    HERE / "test_causal_response_residual_unfolding_certificate.py",
    HERE / "CAUSAL_RESPONSE_RESIDUAL_UNFOLDING_CERTIFICATE_PREREGISTRATION.md",
    HERE / "CAUSAL_RESPONSE_RESIDUAL_UNFOLDING_CERTIFICATE_AMENDMENT_1.md",
)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def tensor_certificate(value: torch.Tensor) -> dict[str, object]:
    """Return exact unfolding spectra and CP approximation lower bounds."""

    if type(value) is not torch.Tensor or value.dtype != torch.float64 or value.device.type != "cpu" or value.ndim != 4 or not bool(torch.isfinite(value).all()):
        raise ValueError("certificate input must be one finite CPU float64 four-way tensor")
    total = float(torch.sum(value * value))
    if not math.isfinite(total) or total <= 0:
        raise ValueError("certificate tensor must have positive finite energy")
    modes: dict[str, object] = {}
    lower_bound = {rank: 0.0 for rank in RANKS}
    for mode, label in enumerate(MODE_LABELS):
        matrix = value.movedim(mode, 0).reshape(value.shape[mode], -1)
        singular = torch.linalg.svdvals(matrix)
        energy = singular.square()
        cumulative = torch.cumsum(energy, dim=0) / total

        def energy_rank(level: float) -> int:
            indices = torch.nonzero(cumulative >= level)
            return int(indices[0]) + 1 if indices.numel() else singular.numel()

        probabilities = energy / total
        positive = probabilities > 0
        effective = float(torch.exp(-(probabilities[positive] * torch.log(probabilities[positive])).sum()))
        tails = {}
        for rank in RANKS:
            tail = float(energy[rank:].sum() / total) if rank < energy.numel() else 0.0
            tails[str(rank)] = max(0.0, tail)
            lower_bound[rank] = max(lower_bound[rank], max(0.0, tail))
        modes[label] = {
            "matrix_shape": list(matrix.shape),
            "singular_values": singular.tolist(),
            "energy_rank_95": energy_rank(0.95),
            "energy_rank_99": energy_rank(0.99),
            "stable_rank": total / float(energy[0]),
            "effective_rank": effective,
            "tail_fraction": tails,
        }
    return {
        "shape": list(value.shape),
        "energy": total,
        "energy_per_cell": total / value.numel(),
        "modes": modes,
        "cp_rank_lower_bound_95": max(item["energy_rank_95"] for item in modes.values()),
        "cp_rank_lower_bound_99": max(item["energy_rank_99"] for item in modes.values()),
        "cp_approximation_error_lower_bound_tail_fraction": {
            str(rank): lower_bound[rank] for rank in RANKS
        },
    }


def _source_closure() -> dict[str, object]:
    hashes = {str(path.relative_to(ROOT)): sha256(path.read_bytes()) for path in SOURCE_PATHS}
    raw = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return {"paths": hashes, "sha256": sha256(raw)}


def _load_rank32_prediction(training, seed: int) -> tuple[torch.Tensor, dict[str, object]]:
    path = GRID / f"g32_p00_s{seed}.pt"
    raw = path.read_bytes()
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema") != "causal_response_factorization_v1_grid_cell" or payload.get("status") != "complete_training_only" or set(payload) != {"schema", "status", "program", "document_codes", "metrics", "receipt"}:
        raise RuntimeError(f"rank-32 cell schema changed: {path.name}")
    receipt = payload["receipt"]
    if receipt.get("global_rank") != 32 or receipt.get("private_rank_each_owner") != 0 or receipt.get("seed") != seed or receipt.get("healthy") is not True or receipt.get("validation_values_read") is not False or receipt.get("eval_values_read") is not False:
        raise RuntimeError(f"rank-32 cell identity or role changed: {path.name}")
    program = ResponseProgram(**payload["program"])
    codes = payload["document_codes"]
    prediction = predict_from_codes(program.basis(), codes).reshape_as(training.response)
    if not bool(torch.isfinite(prediction).all()):
        raise RuntimeError("rank-32 prediction is nonfinite")
    replay_mse = float(((prediction - training.response)[training.valid] ** 2).mean())
    if not math.isclose(replay_mse, receipt["final_mse"], rel_tol=1e-12, abs_tol=1e-15):
        raise RuntimeError("rank-32 cell MSE does not replay")
    return prediction, {
        "artifact": str(path.relative_to(ROOT)), "artifact_sha256": sha256(raw),
        "seed": seed, "replayed_mse": replay_mse,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {"min": min(values), "median": float(statistics.median(values)), "max": max(values)}


def common_support_rectangles(
    valid: torch.Tensor, groups: torch.Tensor,
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    if type(valid) is not torch.Tensor or valid.dtype != torch.bool or valid.ndim != 4 or type(groups) is not torch.Tensor or groups.dtype != torch.int64 or groups.shape != (valid.shape[1],):
        raise ValueError("validity geometry is malformed")
    if not bool((valid == valid[0:1, 0:1].expand_as(valid)).all()):
        raise ValueError("validity is not broadcast over phase and source")
    rectangles = []
    for owner in range(len(OWNER_LABELS)):
        targets = groups == owner
        documents = valid[0, 0, targets].all(dim=0)
        if not bool(targets.any()) or not bool(documents.any()):
            raise ValueError("target-owner rectangle is empty")
        rectangles.append((targets, documents))
    return tuple(rectangles)


def build_receipt() -> dict[str, object]:
    started = time.perf_counter()
    training = load_production_training_snapshot()
    if training.response.shape != (2, 49, 49, 229) or tuple(training.owner_components) != OWNER_LABELS:
        raise RuntimeError("FIT training topology changed")
    rectangles = common_support_rectangles(training.valid, training.source_groups)
    raw_pairs = {}
    for source_owner, source_label in enumerate(OWNER_LABELS):
        raw_pairs[source_label] = {}
        source_mask = training.source_groups == source_owner
        for target_owner, target_label in enumerate(OWNER_LABELS):
            target_mask, documents = rectangles[target_owner]
            block = training.response[:, source_mask][:, :, target_mask][:, :, :, documents]
            raw_pairs[source_label][target_label] = tensor_certificate(block.contiguous())

    seed_rows = []
    artifacts = []
    for seed in SEEDS:
        prediction, artifact = _load_rank32_prediction(training, seed)
        artifacts.append(artifact)
        residual = training.response - prediction
        pairs = {}
        for source_owner, source_label in enumerate(OWNER_LABELS):
            pairs[source_label] = {}
            source_mask = training.source_groups == source_owner
            for target_owner, target_label in enumerate(OWNER_LABELS):
                target_mask, documents = rectangles[target_owner]
                block = residual[:, source_mask][:, :, target_mask][:, :, :, documents]
                pairs[source_label][target_label] = tensor_certificate(block.contiguous())
        seed_rows.append({"seed": seed, "pairs": pairs})

    summaries = {}
    for source_label in OWNER_LABELS:
        summaries[source_label] = {}
        for target_label in OWNER_LABELS:
            pair_rows = [row["pairs"][source_label][target_label] for row in seed_rows]
            summaries[source_label][target_label] = {
                "residual_energy_per_cell": _summary([row["energy_per_cell"] for row in pair_rows]),
                "cp_rank_lower_bound_95": _summary([float(row["cp_rank_lower_bound_95"]) for row in pair_rows]),
                "cp_rank_lower_bound_99": _summary([float(row["cp_rank_lower_bound_99"]) for row in pair_rows]),
                "rank16_lower_bound_tail_fraction": _summary([
                    row["cp_approximation_error_lower_bound_tail_fraction"]["16"] for row in pair_rows
                ]),
            }

    primary = summaries["m16"]["m16"]
    m16_energy = primary["residual_energy_per_cell"]["median"]
    m16_tail = primary["rank16_lower_bound_tail_fraction"]["median"]
    other_energy = max(summaries[label]["m16"]["residual_energy_per_cell"]["median"] for label in OWNER_LABELS if label != "m16")
    other_tail = max(summaries[label]["m16"]["rank16_lower_bound_tail_fraction"]["median"] for label in OWNER_LABELS if label != "m16")
    energy_ratio = m16_energy / other_energy
    tail_ratio = m16_tail / other_tail if other_tail > 0 else None
    high_tail = m16_tail > 0 if tail_ratio is None else tail_ratio >= 1.5
    low_tail = m16_tail <= 0 if tail_ratio is None else tail_ratio <= 1.25
    if energy_ratio >= 1.5 and high_tail:
        decision = "rank_complexity_support"
    elif energy_ratio >= 1.5 and low_tail:
        decision = "amplitude_weighting_support"
    else:
        decision = "mixed_inconclusive"

    grid_terminal = GRID / "terminal.json"
    return {
        "schema": "causal_response_residual_unfolding_certificate",
        "status": "complete_fit_only",
        "source_closure": _source_closure(),
        "fit_artifact_binding": asdict(training.artifacts),
        "grid_terminal_sha256": sha256(grid_terminal.read_bytes()),
        "rank32_artifacts": artifacts,
        "missing_data_handling": {
            "imputation": False,
            "mask_broadcast_phase_source": True,
            "target_owner_rectangles": {
                label: {
                    "targets": int(rectangles[index][0].sum()),
                    "complete_documents": int(rectangles[index][1].sum()),
                }
                for index, label in enumerate(OWNER_LABELS)
            },
        },
        "raw_source_target_owner_pair_certificates": raw_pairs,
        "residual_seed_pair_certificates": seed_rows,
        "source_target_owner_pair_median_summaries": summaries,
        "registered_decision": {
            "outcome": decision,
            "m16_to_largest_other_residual_energy_ratio": energy_ratio,
            "m16_to_largest_other_rank16_tail_ratio": tail_ratio,
            "energy_threshold": 1.5, "rank_support_threshold": 1.5,
            "weighting_threshold": 1.25,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "validation_values_read": False,
        "eval_values_read": False,
        "claims": {
            "candidate_selected": False, "hierarchy_established": False,
            "semantic_atoms_established": False, "ledger_credit": False,
        },
    }


def main() -> None:
    if OUTPUT.exists():
        raise RuntimeError("residual unfolding certificate namespace is already spent")
    value = build_receipt()
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    stage = OUTPUT.with_name(f".{OUTPUT.name}.stage.{os.getpid()}")
    try:
        with stage.open("xb") as sink:
            sink.write(raw); sink.flush(); os.fsync(sink.fileno())
        os.link(stage, OUTPUT)
        directory_fd = os.open(OUTPUT.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        stage.unlink(missing_ok=True)
    print(raw.decode(), end="")


if __name__ == "__main__":
    main()
