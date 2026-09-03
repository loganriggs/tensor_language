#!/usr/bin/env python3
"""Rung 530: source-conditioned downstream bases in the rung-424 attention0 block.

CPU-only analysis of already-open rung-480 response operators.  This is a
circuit-labelled interaction-basis screen, not rank reduction or compression.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import torch


REPO = Path("/workspace/tensor_language")
BQ = REPO / "basis_aligned/bilinear_quotient"
POLY = REPO / "basis_aligned/polynomial_causal"
PREREG = POLY / "ATTENTION0_SOURCE_CONDITIONED_DOWNSTREAM_BASIS_RUNG530_PREREGISTRATION.md"
PARENT_RESULT = BQ / "attention0_downstream_canonical_block_rung480_results.json"
PARENT_BUNDLE = BQ / "attention0_downstream_canonical_block_rung480_bundle.pt"
PARENT_SOURCE = BQ / "ops/attention0_downstream_canonical_block_rung480.py"
OUT = BQ / "attention0_source_conditioned_downstream_basis_rung530_results.json"
FROZEN_SHA256 = {
    PREREG: "4ccb1ac1674999962b0215443ab43ac542be351b24bb33fbad86abf49dc734e1",
    PARENT_RESULT: "e906cd94eb2d7a97ce6e3df59f9b9a6e270d81e027dc1251585a1d0374fbd9f8",
    PARENT_BUNDLE: "2401831045e5b269806a84d6308a941acc61a31ff4868ab9bc39904b0bea6967",
    PARENT_SOURCE: "616aa6e103011598fac8ea710b023f7c1cbaf59d96115d17cf04ec14f508b577",
}
MODES = ("score_branch_1", "score_branch_2", "payload")
MODE_DIMS = (6, 6, 32)
SOURCES = ("N", "H")
ROOTS = (0, 2, 4, 6, 8, 18)
PERMUTATION_SEEDS = tuple(range(530_300, 530_316))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite result: {path}")
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as sink:
            json.dump(value, sink, indent=2, sort_keys=True, allow_nan=False)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.link(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.double().reshape(-1)
    right = right.double().reshape(-1)
    return float(left @ right) / math.sqrt(max(float(left @ left) * float(right @ right), 1e-300))


def projector(operators: torch.Tensor, keep: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    if operators.ndim != 3 or operators.shape[-1] != operators.shape[-2]:
        raise ValueError("operators must have shape [circuits,dimension,dimension]")
    selected = operators if keep is None else operators[keep]
    if len(selected) < 2:
        raise ValueError("at least two circuit operators are required")
    gram = torch.einsum("cab,cdb->ad", selected.double(), selected.double())
    gram = (gram + gram.T) / 2
    values, vectors = torch.linalg.eigh(gram)
    vector = vectors[:, -1]
    return torch.outer(vector, vector), values


def projector_overlap(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.trace(left.double() @ right.double()))


def profile(operators: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    values = torch.einsum("ab,cba->c", basis.double(), operators.double())
    return values - values.mean()


def root_of(tag: str) -> int:
    fields = tag.split(".")
    if len(fields) < 2 or fields[0] != "r":
        raise ValueError(f"malformed circuit tag: {tag}")
    return int(fields[1])


def contrasts(sums: Sequence[torch.Tensor], counts: torch.Tensor) -> list[torch.Tensor]:
    if tuple(counts.shape) != (2, 2, 32):
        raise ValueError("response counts shape changed")
    denominator = counts[:, None, :, :, None, None].double().clamp_min(1)
    output = []
    for dimension, value in zip(MODE_DIMS, sums):
        if tuple(value.shape) != (2, 2, 2, 32, dimension, dimension):
            raise ValueError("operator tensor shape changed")
        means = value.double() / denominator
        output.append(means[:, :, 0] - means[:, :, 1])
    return output


def permutation_q95(base: torch.Tensor, test: torch.Tensor, mode: int, source: int) -> tuple[list[float], float]:
    controls = []
    for seed in PERMUTATION_SEEDS:
        generator = torch.Generator().manual_seed(seed + 100 * mode + 10 * source)
        controls.append(cosine(base, test[torch.randperm(len(test), generator=generator)]))
    return controls, float(torch.quantile(torch.tensor(controls, dtype=torch.float64), .95))


def pair_report(
    main: torch.Tensor,
    refit: torch.Tensor,
    tags: Sequence[str],
    mode: int,
    source: int,
) -> dict[str, Any]:
    fit_projector, spectrum = projector(main[0, source])
    half_projector, _ = projector(main[1, source])
    refit_projector, _ = projector(refit[0, source])
    fit_profile = profile(main[0, source], fit_projector)
    test_profile = profile(main[1, source], fit_projector)
    controls, q95 = permutation_q95(fit_profile, test_profile, mode, source)
    leave = []
    roots = torch.tensor([root_of(tag) for tag in tags])
    for root in ROOTS:
        keep = roots != root
        local_projector, _ = projector(main[0, source], keep)
        local_fit = profile(main[0, source], local_projector)
        local_test = profile(main[1, source], local_projector)
        leave.append({
            "omitted_root": root,
            "projector_overlap_with_full": projector_overlap(fit_projector, local_projector),
            "cross_half_fingerprint_cosine": cosine(local_fit, local_test),
        })
    stable_leaves = sum(
        row["projector_overlap_with_full"] >= .70
        and row["cross_half_fingerprint_cosine"] >= .60
        for row in leave)
    report = {
        "mode": MODES[mode],
        "source": SOURCES[source],
        "projector": fit_projector,
        "spectrum": spectrum.tolist(),
        "half_projector_overlap": projector_overlap(fit_projector, half_projector),
        "refit_projector_overlap": projector_overlap(fit_projector, refit_projector),
        "cross_half_fingerprint_cosine": cosine(fit_profile, test_profile),
        "fit_fingerprint_rms": float(fit_profile.square().mean().sqrt()),
        "test_fingerprint_rms": float(test_profile.square().mean().sqrt()),
        "permutation_cosines": controls,
        "permutation_q95": q95,
        "permutation_margin": cosine(fit_profile, test_profile) - q95,
        "leave_one_root": leave,
        "stable_leave_one_root_count": stable_leaves,
    }
    report["holds"] = bool(
        report["half_projector_overlap"] >= .70
        and report["refit_projector_overlap"] >= .70
        and report["cross_half_fingerprint_cosine"] >= .70
        and report["permutation_margin"] >= .15
        and report["fit_fingerprint_rms"] >= 1e-6
        and report["test_fingerprint_rms"] >= 1e-6
        and stable_leaves >= 5)
    return report


def mode_distinction(main: torch.Tensor, reports: Sequence[dict[str, Any]]) -> dict[str, Any]:
    projectors = [reports[source]["projector"] for source in range(2)]
    own_advantages = []
    own_cosines = []
    other_cosines = []
    for source in range(2):
        base = profile(main[0, source], projectors[source])
        own = cosine(base, profile(main[1, source], projectors[source]))
        other = cosine(base, profile(main[1, source], projectors[1 - source]))
        own_cosines.append(own)
        other_cosines.append(other)
        own_advantages.append(own - other)
    output = {
        "both_source_pairs_hold": bool(all(row["holds"] for row in reports)),
        "source_projector_overlap": projector_overlap(projectors[0], projectors[1]),
        "own_source_cosines": own_cosines,
        "other_source_basis_cosines": other_cosines,
        "own_source_advantages": own_advantages,
    }
    output["holds"] = bool(
        output["both_source_pairs_hold"]
        and output["source_projector_overlap"] <= .50
        and min(output["own_source_advantages"]) >= .15)
    output["strength"] = min(own_cosines) if output["holds"] else None
    return output


def leave_root_winners(main: Sequence[torch.Tensor], tags: Sequence[str]) -> list[dict[str, Any]]:
    roots = torch.tensor([root_of(tag) for tag in tags])
    rows = []
    for root in ROOTS:
        keep = roots != root
        candidates = []
        for mode, operators in enumerate(main):
            bases = [projector(operators[0, source], keep)[0] for source in range(2)]
            own_cosines = []
            advantages = []
            for source in range(2):
                fit = profile(operators[0, source], bases[source])
                own = cosine(fit, profile(operators[1, source], bases[source]))
                other = cosine(fit, profile(operators[1, source], bases[1 - source]))
                own_cosines.append(own)
                advantages.append(own - other)
            distinct = projector_overlap(bases[0], bases[1]) <= .50 and min(advantages) >= .15
            candidates.append({
                "mode": MODES[mode], "distinct": distinct,
                "minimum_own_source_cosine": min(own_cosines),
                "source_projector_overlap": projector_overlap(bases[0], bases[1]),
                "minimum_own_source_advantage": min(advantages),
            })
        winner = max(candidates, key=lambda row: (row["distinct"], row["minimum_own_source_cosine"]))
        rows.append({"omitted_root": root, "winner": winner, "candidates": candidates})
    return rows


def run() -> dict[str, Any]:
    hashes = {str(path.relative_to(REPO)): sha256(path) for path in FROZEN_SHA256}
    if any(hashes[str(path.relative_to(REPO))] != expected for path, expected in FROZEN_SHA256.items()):
        raise RuntimeError("frozen rung480 authority changed")
    parent = json.loads(PARENT_RESULT.read_text())
    bundle = torch.load(PARENT_BUNDLE, map_location="cpu", weights_only=False)
    tags = bundle["discovery_tags"]
    counts = bundle["response_counts"].double()
    main = contrasts(bundle["main_operator_sums"], counts)
    refit = contrasts(bundle["refit_operator_sums_aligned"], counts)
    symmetry_error = max(
        float((value - value.transpose(-1, -2)).abs().max())
        for family in (bundle["main_operator_sums"], bundle["refit_operator_sums_aligned"])
        for value in family)
    all_finite = all(bool(torch.isfinite(value).all()) for family in (main, refit) for value in family)
    roots = tuple(sorted({root_of(tag) for tag in tags}))
    pred_a = bool(
        parent.get("pred_a_exact_lawful_instrument") is True
        and parent.get("validation_family_outcomes_opened") is False
        and bundle.get("validation_tags_or_responses_included") is False
        and len(tags) == len(set(tags)) == 32 and roots == ROOTS
        and int(counts[:, 0].min()) >= 39 and int(counts[:, 1].min()) >= 439
        and symmetry_error <= 1e-10 and all_finite)

    reports = []
    by_mode = []
    for mode in range(3):
        local = [pair_report(main[mode], refit[mode], tags, mode, source) for source in range(2)]
        reports.extend(local)
        by_mode.append(local)
    pred_b = bool(any(row["holds"] for row in reports))
    distinctions = {
        MODES[mode]: mode_distinction(main[mode], by_mode[mode]) for mode in range(3)}
    passing_modes = [mode for mode in MODES if distinctions[mode]["holds"]]
    pred_c = bool(pred_b and passing_modes)
    winning_mode = max(
        passing_modes, key=lambda mode: float(distinctions[mode]["strength"]), default=None)
    leave = leave_root_winners(main, tags)
    same_winner_count = sum(
        row["winner"]["distinct"] and row["winner"]["mode"] == winning_mode for row in leave)
    source_report_by_key = {(row["mode"], row["source"]): row for row in reports}
    pred_d = bool(
        pred_c and same_winner_count >= 5
        and all(source_report_by_key[(winning_mode, source)]["stable_leave_one_root_count"] >= 5
                for source in SOURCES))
    strong_null = not all((pred_a, pred_b, pred_c, pred_d))
    serial_reports = []
    for row in reports:
        serial_reports.append({key: value for key, value in row.items() if key != "projector"})
    result = {
        "status": "complete",
        "rung": 530,
        "claim_level": "circuit_labelled_source_conditioned_interaction_basis_screen_not_causal_or_compression",
        "source_hashes": hashes,
        "instrument": {
            "operator_shapes": [[list(value.shape) for value in family]
                                for family in (bundle["main_operator_sums"], bundle["refit_operator_sums_aligned"])],
            "minimum_member_support": int(counts[:, 0].min()),
            "minimum_control_support": int(counts[:, 1].min()),
            "maximum_symmetry_error": symmetry_error,
            "all_finite": all_finite,
            "discovery_tags": len(tags),
            "roots": list(roots),
            "validation_responses_opened": bool(bundle["validation_tags_or_responses_included"]),
        },
        "source_conditioned_pairs": serial_reports,
        "mode_distinctions": distinctions,
        "passing_modes": passing_modes,
        "winning_mode": winning_mode,
        "leave_one_root_winners": leave,
        "same_winning_mode_count": same_winner_count,
        "pred_a_exact_cpu_instrument": pred_a,
        "pred_b_reproducible_source_conditioned_direction": pred_b,
        "pred_c_distinct_usable_downstream_bases": pred_c,
        "pred_d_not_one_root_artifact": pred_d,
        "strong_null": strong_null,
        "execution_price": {
            "model_forwards": 0, "model_backwards": 0, "fitted_model_values": 0,
            "deployed_values_added": 0, "deployed_values_removed": 0,
            "largest_eigendecomposition_dimension": 32,
        },
        "next_step": (
            "preregister_physical_source_specific_attention0_interventions_on_30_unopened_circuits"
            if not strong_null else
            "repair_CPU_reader_only" if not pred_a else
            "close_rank1_source_conditioned_basis_without_rank_or_threshold_retry"
        ),
    }
    atomic_json(OUT, result)
    print(json.dumps({
        "status": result["status"],
        "predictions": {key: value for key, value in result.items() if key.startswith("pred_")},
        "strong_null": strong_null,
        "pair_summary": [{
            "mode": row["mode"], "source": row["source"], "holds": row["holds"],
            "half_overlap": row["half_projector_overlap"],
            "refit_overlap": row["refit_projector_overlap"],
            "fingerprint_cosine": row["cross_half_fingerprint_cosine"],
            "permutation_margin": row["permutation_margin"],
            "stable_roots": row["stable_leave_one_root_count"],
        } for row in serial_reports],
        "mode_distinctions": distinctions,
        "winning_mode": winning_mode,
        "same_winning_mode_count": same_winner_count,
        "next_step": result["next_step"],
    }, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run()
