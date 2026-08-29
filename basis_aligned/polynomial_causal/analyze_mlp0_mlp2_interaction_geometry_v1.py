#!/usr/bin/env python3
"""Frozen CPU diagnostic for document-level MLP0-C512 × MLP2 interactions."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_ledger.pt"
RECEIPT = HERE / "mlp2_trajectory_robust_r512_v3_physical_eval_receipt.json"
RESULT = HERE / "mlp0_mlp2_interaction_geometry_v1_result.json"
LEDGER_SHA = "969aa29c58ad2ee860bb0d486a44bcc20792f5c1d966cbb48ddba38f49a8ae0b"
RECEIPT_SHA = "22026cd77420e8cf739796e2283782bbe971be1852eaa1996902aaf7e0bab30e"
SEED = 2026082943
PROGRAMS = ("FULL512", "CONTINUE512", "ROBUST512")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_load() -> tuple[dict[str, Any], dict[str, Any]]:
    if file_sha256(LEDGER) != LEDGER_SHA or file_sha256(RECEIPT) != RECEIPT_SHA:
        raise RuntimeError("frozen interaction parent changed")
    ledger = torch.load(LEDGER, map_location="cpu", weights_only=False)
    receipt = json.loads(RECEIPT.read_text())
    if file_sha256(LEDGER) != LEDGER_SHA or file_sha256(RECEIPT) != RECEIPT_SHA \
            or receipt.get("ledger_sha256") != LEDGER_SHA \
            or receipt.get("status") != "result_complete_receipt_last":
        raise RuntimeError("interaction parent raced stable load")
    return ledger, receipt


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.double() - x.double().mean(); y = y.double() - y.double().mean()
    denom = torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)
    return float((x @ y) / denom) if float(denom) > 0 else 0.0


def ranks(x: torch.Tensor) -> torch.Tensor:
    # Ledger values are effectively continuous; deterministic ordinal ranks suffice.
    order = torch.argsort(x.double(), stable=True)
    out = torch.empty_like(x, dtype=torch.float64)
    out[order] = torch.arange(len(x), dtype=torch.float64)
    return out


def gini_nonnegative(x: torch.Tensor) -> float:
    x = torch.sort(x.double().abs()).values
    total = float(x.sum())
    if total == 0.0:
        return 0.0
    n = len(x); index = torch.arange(1, n + 1, dtype=torch.float64)
    return float((2 * (index @ x) / (n * x.sum())) - (n + 1) / n)


def bootstrap_mean_ci(x: torch.Tensor) -> list[float]:
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randint(len(x), (10_000, len(x)), generator=generator)
    means = x.double()[indices].mean(1)
    return [float(torch.quantile(means, q)) for q in (0.025, 0.975)]


def vector_summary(x: torch.Tensor) -> dict[str, Any]:
    x = x.double(); absolute = x.abs(); total = float(absolute.sum())
    sorted_abs = torch.sort(absolute, descending=True).values
    concentration = {}
    for fraction in (0.01, 0.05, 0.10, 0.20, 0.25):
        count = max(1, math.ceil(len(x) * fraction))
        concentration[f"top_{int(fraction * 100):02d}pct"] = (
            float(sorted_abs[:count].sum()) / total if total else 0.0
        )
    square = float(x.square().sum())
    return {
        "mean": float(x.mean()), "std": float(x.std(unbiased=True)),
        "median": float(x.median()), "minimum": float(x.min()),
        "maximum": float(x.max()), "positive_fraction": float((x > 0).double().mean()),
        "negative_fraction": float((x < 0).double().mean()),
        "mean_bootstrap_ci95": bootstrap_mean_ci(x),
        "absolute_mass_concentration": concentration,
        "absolute_gini": gini_nonnegative(x),
        "effective_participation_documents": total * total / square if square else 0.0,
    }


def per_document(arm: torch.Tensor, metric: str) -> torch.Tensor:
    count = arm[:, 8]
    if metric == "dce":
        return (arm[:, 1] - arm[:, 0]) / count
    if metric == "kl":
        return arm[:, 2] / count
    raise ValueError(metric)


def analyze(arms: dict[str, torch.Tensor]) -> dict[str, Any]:
    required = {"NATIVE", "C512", *PROGRAMS,
                *(f"C512_{program}" for program in PROGRAMS)}
    if not required.issubset(arms) or any(tuple(arms[name].shape) != (192, 9)
                                          for name in required):
        raise RuntimeError("interaction ledger shape/arms changed")
    c_dce = per_document(arms["C512"], "dce")
    c_kl = per_document(arms["C512"], "kl")
    interactions: dict[str, dict[str, torch.Tensor]] = {}
    for program in PROGRAMS:
        interactions[program] = {
            "dce": per_document(arms[f"C512_{program}"], "dce")
                   - c_dce - per_document(arms[program], "dce"),
            "kl": per_document(arms[f"C512_{program}"], "kl")
                  - c_kl - per_document(arms[program], "kl"),
        }

    summaries = {program: {metric: vector_summary(values[metric])
                           for metric in ("dce", "kl")}
                 for program, values in interactions.items()}
    matrix = torch.stack([interactions[p]["dce"] for p in PROGRAMS], 1).double()
    centered = matrix - matrix.mean(0, keepdim=True)
    raw_s = torch.linalg.svdvals(matrix); centered_s = torch.linalg.svdvals(centered)
    pairwise = {}
    for i, left in enumerate(PROGRAMS):
        for right in PROGRAMS[i + 1:]:
            key = f"{left}__{right}"
            x, y = interactions[left]["dce"], interactions[right]["dce"]
            pairwise[key] = {"pearson": pearson(x, y),
                             "spearman": pearson(ranks(x), ranks(y))}

    native_difficulty = arms["NATIVE"][:, 0] / arms["NATIVE"][:, 8]
    predictors = {"native_nll": native_difficulty, "c512_dce": c_dce}
    correlations = {}
    for program in PROGRAMS:
        program_predictors = dict(predictors)
        program_predictors["program_standalone_dce"] = per_document(arms[program], "dce")
        program_predictors["composed_dce"] = per_document(arms[f"C512_{program}"], "dce")
        correlations[program] = {
            name: {"pearson": pearson(interactions[program]["dce"], value),
                   "spearman": pearson(ranks(interactions[program]["dce"]), ranks(value))}
            for name, value in program_predictors.items()
        }

    full = interactions["FULL512"]["dce"]
    reductions = {
        "FULL_minus_CONTINUE": full - interactions["CONTINUE512"]["dce"],
        "FULL_minus_ROBUST": full - interactions["ROBUST512"]["dce"],
        "CONTINUE_minus_ROBUST": interactions["CONTINUE512"]["dce"]
                                 - interactions["ROBUST512"]["dce"],
    }
    reduction_report = {
        name: {**vector_summary(value),
               "pearson_with_abs_full_interaction": pearson(value, full.abs())}
        for name, value in reductions.items()
    }
    energy = lambda s: [float(v) for v in (s.square() / s.square().sum())]
    min_pairwise = min(value["pearson"] for value in pairwise.values())
    max_simple = max(abs(correlations[p][name]["pearson"])
                     for p in PROGRAMS for name in ("native_nll", "c512_dce"))
    rules = {
        "diffuse_all_programs": all(
            summaries[p]["dce"]["effective_participation_documents"] >= 48
            and summaries[p]["dce"]["absolute_mass_concentration"]["top_10pct"] <= 0.60
            for p in PROGRAMS),
        "shared_document_mode": min_pairwise >= 0.70
                                and energy(centered_s)[0] >= 0.70,
        "simple_difficulty_predictor": max_simple >= 0.50,
        "robust_reduction_targets_large_full_interactions": (
            reduction_report["FULL_minus_ROBUST"]["mean"] > 0
            and reduction_report["FULL_minus_ROBUST"]
                ["pearson_with_abs_full_interaction"] >= 0.50),
        "sparse_gate_candidate": any(
            summaries[p]["dce"]["absolute_mass_concentration"]["top_10pct"] >= 0.75
            for p in PROGRAMS),
    }
    return {
        "schema": "mlp0_mlp2_interaction_geometry_v1_result",
        "claim_boundary": "post_outcome_cpu_diagnostic_no_strict_ledger_move",
        "documents": 192, "seed": SEED,
        "interaction_summaries": summaries,
        "pairwise_ce_interaction_correlations": pairwise,
        "singular_energy": {"raw": energy(raw_s), "document_centered": energy(centered_s)},
        "predictor_correlations": correlations,
        "interaction_reductions": reduction_report,
        "diagnostic_rules": rules,
        "parents": {"ledger_sha256": LEDGER_SHA, "receipt_sha256": RECEIPT_SHA},
    }


def write_create_only(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(fd, encoded); os.fsync(fd)
    finally:
        os.close(fd)


def main() -> None:
    if RESULT.exists():
        raise RuntimeError("interaction geometry result namespace already exists")
    ledger, _ = stable_load()
    result = analyze(ledger["arms"])
    write_create_only(RESULT, result)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
