#!/usr/bin/env python3
"""Paired source-document inference for the MLP0 native-Down program family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np


MARGINS = {"kl": 0.01, "ce": 0.0075, "attn1_nrmse": 0.05, "mlp1_nrmse": 0.05}
N_BOOTSTRAP = 20_000
SEED = 20260828
MINIMUM_DOCUMENTS_PER_CELL = 60


def quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def validate_ledgers(ledgers: Mapping[str, Mapping[str, Mapping[str, object]]]) -> tuple[list[str], int, int]:
    if not ledgers:
        raise ValueError("at least one arm is required")
    arms = list(ledgers)
    n_documents = n_cells = None
    for arm, consumers in ledgers.items():
        if set(consumers) != set(MARGINS):
            raise ValueError(f"arm {arm} does not have the registered consumers")
        for consumer, ledger in consumers.items():
            sums = np.asarray(ledger["sums"], dtype=np.float64)
            counts = np.asarray(ledger["counts"], dtype=np.float64)
            if sums.ndim != 2 or counts.shape != sums.shape:
                raise ValueError("ledger arrays must share [document,cell] shape")
            if n_documents is None:
                n_documents, n_cells = sums.shape
            elif sums.shape != (n_documents, n_cells):
                raise ValueError("all ledger arrays must have identical shape")
            if not np.isfinite(sums).all() or not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError("ledger contains invalid sufficient statistics")
    assert n_documents is not None and n_cells is not None
    return arms, n_documents, n_cells


def arm_points(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    arms: list[str],
    documents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    points = np.full(len(arms), -np.inf)
    first_arm, first_consumer = arms[0], next(iter(MARGINS))
    n_cells = np.asarray(ledgers[first_arm][first_consumer]["sums"]).shape[1]
    coordinates = np.full((len(arms), len(MARGINS) * n_cells), np.inf)
    reports = {}
    for arm_index, arm in enumerate(arms):
        consumers = {}
        support_ok = True
        coordinate_offset = 0
        for consumer, margin in MARGINS.items():
            sums = np.asarray(ledgers[arm][consumer]["sums"], dtype=np.float64)[documents]
            counts = np.asarray(ledgers[arm][consumer]["counts"], dtype=np.float64)[documents]
            total_sums, total_counts = sums.sum(0), counts.sum(0)
            effects = np.divide(total_sums, total_counts, out=np.full_like(total_sums, np.inf), where=total_counts > 0)
            ratios = effects / margin
            support = (counts > 0).sum(0)
            support_ok = support_ok and bool((support >= MINIMUM_DOCUMENTS_PER_CELL).all())
            points[arm_index] = max(points[arm_index], float(np.max(ratios)))
            coordinates[arm_index, coordinate_offset:coordinate_offset + len(ratios)] = ratios
            coordinate_offset += len(ratios)
            consumers[consumer] = {
                "cell_effects": effects.tolist(),
                "cell_standardized_effects": ratios.tolist(),
                "support_documents": support.tolist(),
                "over_margin": bool(np.max(ratios) > 1),
            }
        reports[arm] = {"consumers": consumers, "support_passes": support_ok}
    return points, coordinates, reports


def bootstrap_arm_coordinates(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    arms: list[str],
    documents: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    first_consumer = next(iter(MARGINS))
    n_cells = len(ledgers[arms[0]][first_consumer]["sums"][0])
    output = np.full((n_bootstrap, len(arms), len(MARGINS) * n_cells), np.inf)
    chunk = min(256, n_bootstrap)
    for offset in range(0, n_bootstrap, chunk):
        size = min(chunk, n_bootstrap - offset)
        local = np.full((size, len(arms), len(MARGINS) * n_cells), np.inf)
        sampled = rng.integers(0, len(documents), size=(size, len(documents)))
        selected = documents[sampled]
        for arm_index, arm in enumerate(arms):
            coordinate_offset = 0
            for consumer, margin in MARGINS.items():
                sums = np.asarray(ledgers[arm][consumer]["sums"], dtype=np.float64)
                counts = np.asarray(ledgers[arm][consumer]["counts"], dtype=np.float64)
                sampled_sums = sums[selected].sum(1)
                sampled_counts = counts[selected].sum(1)
                effects = np.divide(
                    sampled_sums, sampled_counts, out=np.full_like(sampled_sums, np.inf),
                    where=sampled_counts > 0,
                )
                local[:, arm_index, coordinate_offset:coordinate_offset + n_cells] = effects / margin
                coordinate_offset += n_cells
        output[offset:offset + size] = local
    return output


def familywise_bounds(coordinates: np.ndarray, bootstrap: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Center every arm×consumer×cell before the global bootstrap maximum."""
    if bootstrap.ndim != 3 or coordinates.shape != bootstrap.shape[1:]:
        raise ValueError("coordinate bootstrap shape mismatch")
    upper_error = max(0.0, quantile(np.max(bootstrap - coordinates[None, :, :], axis=(1, 2)), 0.95))
    lower_error = max(0.0, quantile(np.max(coordinates[None, :, :] - bootstrap, axis=(1, 2)), 0.95))
    points = coordinates.max(axis=1)
    return points + upper_error, points - lower_error, upper_error, lower_error


def comparison_reports(
    points: np.ndarray,
    bootstrap_maxima: np.ndarray,
    arms: list[str],
    comparisons: Mapping[str, tuple[str, str]],
    point_reports: Mapping[str, object],
) -> dict[str, object]:
    point_differences, boot_differences = {}, {}
    for name, (baseline, candidate) in comparisons.items():
        baseline_index, candidate_index = arms.index(baseline), arms.index(candidate)
        point_differences[name] = float(points[baseline_index] - points[candidate_index])
        boot_differences[name] = bootstrap_maxima[:, baseline_index] - bootstrap_maxima[:, candidate_index]
    errors = np.stack([
        point_differences[name] - boot_differences[name] for name in comparisons
    ], axis=1)
    correction = max(0.0, quantile(np.max(errors, axis=1), 0.95))
    reports = {}
    for name, (baseline, candidate) in comparisons.items():
        pointwise = True
        for consumer in MARGINS:
            base = point_reports[baseline]["consumers"][consumer]["cell_standardized_effects"]
            cand = point_reports[candidate]["consumers"][consumer]["cell_standardized_effects"]
            pointwise = pointwise and all(c <= b for c, b in zip(cand, base))
        reports[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "point_max_reduction": point_differences[name],
            "familywise_95pct_lcb_reduction": point_differences[name] - correction,
            "candidate_pointwise_no_worse": bool(pointwise),
        }
    return {"familywise_lower_correction": correction, "comparisons": reports}


def score_scope(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    documents: np.ndarray,
    comparisons: Mapping[str, tuple[str, str]],
    *,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    arms, _, _ = validate_ledgers(ledgers)
    points, coordinates, reports = arm_points(ledgers, arms, documents)
    bootstrap_coordinates = bootstrap_arm_coordinates(
        ledgers, arms, documents, n_bootstrap=n_bootstrap, seed=seed
    )
    bootstrap_maxima = bootstrap_coordinates.max(axis=2)
    upper, lower, upper_error, lower_error = familywise_bounds(coordinates, bootstrap_coordinates)
    for index, arm in enumerate(arms):
        reports[arm].update({
            "point_max_standardized_effect": float(points[index]),
            "familywise_95pct_ucb_max_standardized_effect": float(upper[index]),
            "familywise_95pct_lcb_max_standardized_effect": float(lower[index]),
            "over_margin_consumer_families": sorted([
                consumer for consumer, report in reports[arm]["consumers"].items()
                if report["over_margin"]
            ]),
        })
    return {
        "n_documents": len(documents),
        "bootstrap": {"replicates": n_bootstrap, "seed": seed,
                      "familywise_upper_correction": upper_error,
                      "familywise_lower_correction": lower_error},
        "arms": reports,
        "superiority": comparison_reports(points, bootstrap_maxima, arms, comparisons, reports),
    }


def score_result(payload: Mapping[str, object], *, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, object]:
    ledgers = payload["sufficient_statistics"]
    arms, n_documents, _ = validate_ledgers(ledgers)
    if n_documents != 384:
        raise ValueError("registered evaluation requires exactly 384 source documents")
    construction = payload["construction"]
    ranks = construction["ranks"]
    comparisons = {}
    hierarchy_pairs = []
    for kind in ("Q", "A"):
        for rung in ("256", "512"):
            rank = int(ranks[kind][rung])
            hierarchy = f"{kind}{rank}"
            null = f"{kind}null{rank}"
            continuous = f"C{rung}"
            comparisons[f"{hierarchy}_vs_{continuous}"] = (continuous, hierarchy)
            comparisons[f"{hierarchy}_vs_{null}"] = (null, hierarchy)
            hierarchy_pairs.append((hierarchy, null, continuous))
    scopes = {
        "wave_A": score_scope(ledgers, np.arange(0, 192), comparisons,
                              n_bootstrap=n_bootstrap, seed=seed),
        "wave_B": score_scope(ledgers, np.arange(192, 384), comparisons,
                              n_bootstrap=n_bootstrap, seed=seed + 1),
        "pooled": score_scope(ledgers, np.arange(0, 384), comparisons,
                             n_bootstrap=n_bootstrap, seed=seed + 2),
    }
    coverage = payload["coverage"]
    gates = {}
    for arm in arms:
        identities = [scopes[scope]["arms"][arm]["over_margin_consumer_families"]
                      for scope in ("wave_A", "wave_B", "pooled")]
        gates[arm] = {
            "wave_A_ucb_lt_1": scopes["wave_A"]["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < 1,
            "wave_B_ucb_lt_1": scopes["wave_B"]["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < 1,
            "pooled_ucb_lt_0_8": scopes["pooled"]["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < 0.8,
            "support_each_wave": all(scopes[scope]["arms"][arm]["support_passes"] for scope in ("wave_A", "wave_B")),
            "coverage_each_wave_ge_90pct": coverage["wave_A"] >= .9 and coverage["wave_B"] >= .9,
            "over_margin_family_identity_stable": identities[0] == identities[1] == identities[2],
        }
        gates[arm]["absolute_credit"] = all(gates[arm].values())
    lexical = {}
    price_gates = payload["price_gates"]
    for hierarchy, null, continuous in hierarchy_pairs:
        at = {}
        for scope in ("wave_A", "wave_B", "pooled"):
            superiority = scopes[scope]["superiority"]["comparisons"]
            at[scope] = {
                "vs_continuous_lcb_positive": superiority[f"{hierarchy}_vs_{continuous}"]["familywise_95pct_lcb_reduction"] > 0,
                "vs_null_lcb_positive": superiority[f"{hierarchy}_vs_{null}"]["familywise_95pct_lcb_reduction"] > 0,
                "pointwise_no_worse_than_continuous": superiority[f"{hierarchy}_vs_{continuous}"]["candidate_pointwise_no_worse"],
                "pointwise_no_worse_than_null": superiority[f"{hierarchy}_vs_{null}"]["candidate_pointwise_no_worse"],
            }
        lexical[hierarchy] = {
            "absolute_credit": gates[hierarchy]["absolute_credit"],
            "physical_price_gate": bool(
                price_gates[f"{hierarchy[0]}_at_C{continuous[1:]}"]["admitted_le_ceiling"]
                and price_gates[f"{hierarchy[0]}_at_C{continuous[1:]}"]["next_rank_gt_ceiling"]
            ),
            "scopes": at,
            "priced_lexical_simplicity_credit": bool(
                gates[hierarchy]["absolute_credit"]
                and price_gates[f"{hierarchy[0]}_at_C{continuous[1:]}"]["admitted_le_ceiling"]
                and price_gates[f"{hierarchy[0]}_at_C{continuous[1:]}"]["next_rank_gt_ceiling"]
                and all(all(values.values()) for values in at.values())
            ),
        }
    return {
        "schema_version": 1,
        "inference": "paired source-document familywise centered bootstrap",
        "minimum_documents_per_cell": MINIMUM_DOCUMENTS_PER_CELL,
        "scopes": scopes,
        "absolute_gates": gates,
        "lexical_gates": lexical,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = score_result(payload)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"absolute": result["absolute_gates"], "lexical": result["lexical_gates"]}, indent=2))


if __name__ == "__main__":
    main()
