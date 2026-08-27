#!/usr/bin/env python3
"""Simultaneous source-unit inference for the C512/MLP1 physical 2x2."""

from __future__ import annotations

from typing import Mapping

import numpy as np


MARGINS = {"kl": 0.01, "ce_abs": 0.0075, "centered_logit_nrmse": 0.05}
BACKGROUNDS = ("live", "mlp2_omit")
CONTRASTS = (
    "observational_CC", "write_on_O", "write_on_C", "upstream_state",
    "interaction", "shuffle", "native_write",
)
ARMS = tuple(f"{background}/{contrast}" for background in BACKGROUNDS for contrast in CONTRASTS)
CORE_EQUIVALENCE = tuple(
    f"{background}/{contrast}"
    for background in BACKGROUNDS
    for contrast in ("observational_CC", "write_on_O", "write_on_C", "interaction")
)
N_BOOTSTRAP = 20_000
SEED = 20260830
MIN_FINEWEB_DOCUMENTS_PER_CELL = 60
MIN_CODE_FILES_PER_CELL = 12


def quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def _effect(consumer: str, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    value = np.divide(sums, counts, out=np.full_like(sums, np.inf), where=counts > 0)
    return np.abs(value) if consumer == "ce_abs" else value


def validate_ledgers(ledgers: Mapping[str, Mapping[str, Mapping[str, object]]]) -> tuple[int, int]:
    if set(ledgers) != set(ARMS):
        raise ValueError("ledger does not contain the registered background/contrast family")
    shape = None
    for arm in ARMS:
        consumers = ledgers[arm]
        if set(consumers) != set(MARGINS):
            raise ValueError(f"arm {arm} does not contain the registered consumers")
        for ledger in consumers.values():
            sums = np.asarray(ledger["sums"], dtype=np.float64)
            counts = np.asarray(ledger["counts"], dtype=np.float64)
            if sums.ndim != 2 or counts.shape != sums.shape:
                raise ValueError("ledger arrays must share [source-unit,cell] shape")
            if shape is None:
                shape = sums.shape
            elif sums.shape != shape:
                raise ValueError("all ledgers must have one source-unit/cell shape")
            if (not np.isfinite(sums).all() or not np.isfinite(counts).all()
                    or bool((counts < 0).any())):
                raise ValueError("ledger contains invalid sufficient statistics")
    assert shape is not None
    return int(shape[0]), int(shape[1])


def point_coordinates(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    units: np.ndarray,
    minimum_support: int,
) -> tuple[np.ndarray, dict[str, object]]:
    n_cells = np.asarray(ledgers[ARMS[0]][next(iter(MARGINS))]["sums"]).shape[1]
    coordinates = np.full((len(ARMS), len(MARGINS) * n_cells), np.inf)
    reports: dict[str, object] = {}
    for arm_index, arm in enumerate(ARMS):
        offset = 0
        support_passes = True
        consumers = {}
        for consumer, margin in MARGINS.items():
            sums = np.asarray(ledgers[arm][consumer]["sums"], dtype=np.float64)[units]
            counts = np.asarray(ledgers[arm][consumer]["counts"], dtype=np.float64)[units]
            effects = _effect(consumer, sums.sum(0), counts.sum(0))
            ratios = effects / margin
            support = (counts > 0).sum(0)
            support_passes = support_passes and bool((support >= minimum_support).all())
            coordinates[arm_index, offset:offset + n_cells] = ratios
            offset += n_cells
            consumers[consumer] = {
                "cell_effects": effects.tolist(),
                "cell_standardized_effects": ratios.tolist(),
                "support_source_units": support.tolist(),
                "over_margin": bool(np.max(ratios) > 1),
            }
        reports[arm] = {"consumers": consumers, "support_passes": support_passes}
    return coordinates, reports


def bootstrap_coordinates(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    units: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cells = len(ledgers[ARMS[0]][next(iter(MARGINS))]["sums"][0])
    output = np.full((n_bootstrap, len(ARMS), len(MARGINS) * n_cells), np.inf)
    for offset in range(0, n_bootstrap, min(256, n_bootstrap)):
        size = min(256, n_bootstrap - offset)
        sampled = units[rng.integers(0, len(units), size=(size, len(units)))]
        local = np.full((size, len(ARMS), len(MARGINS) * n_cells), np.inf)
        for arm_index, arm in enumerate(ARMS):
            coordinate_offset = 0
            for consumer, margin in MARGINS.items():
                sums = np.asarray(ledgers[arm][consumer]["sums"], dtype=np.float64)
                counts = np.asarray(ledgers[arm][consumer]["counts"], dtype=np.float64)
                effects = _effect(consumer, sums[sampled].sum(1), counts[sampled].sum(1))
                local[:, arm_index, coordinate_offset:coordinate_offset + n_cells] = effects / margin
                coordinate_offset += n_cells
        output[offset:offset + size] = local
    return output


def familywise_bounds(coordinates: np.ndarray, bootstrap: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Center every background/contrast/consumer/cell before one family maximum."""
    if bootstrap.ndim != 3 or coordinates.shape != bootstrap.shape[1:]:
        raise ValueError("coordinate bootstrap shape mismatch")
    upper_error = max(0.0, quantile(np.max(bootstrap - coordinates[None], axis=(1, 2)), .95))
    lower_error = max(0.0, quantile(np.max(coordinates[None] - bootstrap, axis=(1, 2)), .95))
    points = coordinates.max(1)
    return points + upper_error, points - lower_error, upper_error, lower_error


def comparison_bounds(
    coordinates: np.ndarray,
    bootstrap: np.ndarray,
    comparisons: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    """Family-wise lower bounds for reduction from baseline arm to candidate arm."""
    points = coordinates.max(1)
    bootstrap_points = bootstrap.max(2)
    point_differences = {}
    bootstrap_differences = {}
    for name, (baseline, candidate) in comparisons.items():
        bi, ci = ARMS.index(baseline), ARMS.index(candidate)
        point_differences[name] = float(points[bi] - points[ci])
        bootstrap_differences[name] = bootstrap_points[:, bi] - bootstrap_points[:, ci]
    errors = np.stack([
        point_differences[name] - bootstrap_differences[name] for name in comparisons
    ], axis=1)
    correction = max(0.0, quantile(np.max(errors, axis=1), .95))
    return {
        "familywise_lower_correction": correction,
        "comparisons": {
            name: {
                "baseline": comparisons[name][0],
                "candidate": comparisons[name][1],
                "point_max_reduction": point_differences[name],
                "familywise_95pct_lcb_reduction": point_differences[name] - correction,
            }
            for name in comparisons
        },
    }


def score_scope(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    units: np.ndarray,
    *,
    minimum_support: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    validate_ledgers(ledgers)
    coordinates, reports = point_coordinates(ledgers, units, minimum_support)
    bootstrap = bootstrap_coordinates(ledgers, units, n_bootstrap=n_bootstrap, seed=seed)
    upper, lower, upper_error, lower_error = familywise_bounds(coordinates, bootstrap)
    for arm_index, arm in enumerate(ARMS):
        reports[arm].update({
            "point_max_standardized_effect": float(coordinates[arm_index].max()),
            "familywise_95pct_ucb_max_standardized_effect": float(upper[arm_index]),
            "familywise_95pct_lcb_max_standardized_effect": float(lower[arm_index]),
            "over_margin_consumer_families": sorted(
                consumer for consumer, report in reports[arm]["consumers"].items()
                if report["over_margin"]
            ),
        })
    comparisons = {
        f"rescue_{background}": (
            f"{background}/observational_CC", f"{background}/upstream_state"
        )
        for background in BACKGROUNDS
    }
    return {
        "n_source_units": len(units),
        "bootstrap": {
            "replicates": n_bootstrap, "seed": seed,
            "familywise_upper_correction": upper_error,
            "familywise_lower_correction": lower_error,
        },
        "arms": reports,
        "rescue": comparison_bounds(coordinates, bootstrap, comparisons),
    }


def _fineweb_gates(scopes: Mapping[str, object], coverage: Mapping[str, float]) -> dict[str, object]:
    gates = {}
    for arm in ARMS:
        identities = [scopes[scope]["arms"][arm]["over_margin_consumer_families"]
                      for scope in ("wave_A", "wave_B", "pooled")]
        values = {
            "wave_A_ucb_lt_1": scopes["wave_A"]["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < 1,
            "wave_B_ucb_lt_1": scopes["wave_B"]["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < 1,
            "pooled_ucb_lt_0_8": scopes["pooled"]["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < .8,
            "support_each_wave": all(scopes[scope]["arms"][arm]["support_passes"] for scope in ("wave_A", "wave_B")),
            "coverage_each_wave_ge_90pct": coverage["wave_A"] >= .9 and coverage["wave_B"] >= .9,
            "over_margin_family_identity_stable": identities[0] == identities[1] == identities[2],
        }
        values["equivalence_credit"] = all(values.values())
        gates[arm] = values
    return gates


def score_result(payload: Mapping[str, object], *, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, object]:
    fineweb = payload["sufficient_statistics"]["fineweb"]
    code = payload["sufficient_statistics"]["code"]
    if validate_ledgers(fineweb)[0] != 384 or validate_ledgers(code)[0] != 48:
        raise ValueError("registered FineWeb/code source-unit counts changed")
    fineweb_scopes = {
        "wave_A": score_scope(fineweb, np.arange(192), minimum_support=MIN_FINEWEB_DOCUMENTS_PER_CELL,
                              n_bootstrap=n_bootstrap, seed=seed),
        "wave_B": score_scope(fineweb, np.arange(192, 384), minimum_support=MIN_FINEWEB_DOCUMENTS_PER_CELL,
                              n_bootstrap=n_bootstrap, seed=seed + 1),
        "pooled": score_scope(fineweb, np.arange(384), minimum_support=MIN_FINEWEB_DOCUMENTS_PER_CELL,
                             n_bootstrap=n_bootstrap, seed=seed + 2),
    }
    code_scope = score_scope(code, np.arange(48), minimum_support=MIN_CODE_FILES_PER_CELL,
                             n_bootstrap=n_bootstrap, seed=seed + 3)
    fineweb_gates = _fineweb_gates(fineweb_scopes, payload["coverage"]["fineweb"])
    code_gates = {
        arm: {
            "ucb_lt_1": code_scope["arms"][arm]["familywise_95pct_ucb_max_standardized_effect"] < 1,
            "support": code_scope["arms"][arm]["support_passes"],
            "coverage_ge_90pct": payload["coverage"]["code"] >= .9,
        }
        for arm in ARMS
    }
    for values in code_gates.values():
        values["equivalence_credit"] = all(values.values())

    positive_control = {
        background: fineweb_scopes["pooled"]["arms"][f"{background}/native_write"][
            "familywise_95pct_lcb_max_standardized_effect"
        ] > 1
        for background in BACKGROUNDS
    }
    code_positive_control = {
        background: code_scope["arms"][f"{background}/native_write"][
            "familywise_95pct_lcb_max_standardized_effect"
        ] > 1
        for background in BACKGROUNDS
    }
    fineweb_core = all(fineweb_gates[arm]["equivalence_credit"] for arm in CORE_EQUIVALENCE)
    code_core = all(code_gates[arm]["equivalence_credit"] for arm in CORE_EQUIVALENCE)
    rescue = {
        background: all(
            fineweb_scopes[scope]["rescue"]["comparisons"][f"rescue_{background}"][
                "familywise_95pct_lcb_reduction"
            ] > 0
            for scope in ("wave_A", "wave_B")
        )
        for background in BACKGROUNDS
    }
    observational = [f"{background}/observational_CC" for background in BACKGROUNDS]
    observational_pass = all(fineweb_gates[arm]["equivalence_credit"] for arm in observational)
    downstream_null = fineweb_core and all(positive_control.values())
    return {
        "schema_version": 1,
        "inference": "coordinatewise-centered simultaneous source-unit bootstrap",
        "margins": MARGINS,
        "minimum_support": {
            "fineweb_documents_per_cell": MIN_FINEWEB_DOCUMENTS_PER_CELL,
            "code_files_per_cell": MIN_CODE_FILES_PER_CELL,
        },
        "fineweb_scopes": fineweb_scopes,
        "code_scope": code_scope,
        "fineweb_gates": fineweb_gates,
        "code_gates": code_gates,
        "decisions": {
            "fresh_observational_equivalence": observational_pass,
            "downstream_null_on_registered_fineweb_backgrounds": downstream_null,
            "cancellation_or_interface_break": observational_pass and not fineweb_core,
            "positive_control_each_background": positive_control,
            "code_positive_control_each_background": code_positive_control,
            "conditional_alignment_null": bool(
                downstream_null and any(
                    not fineweb_gates[f"{background}/shuffle"]["equivalence_credit"]
                    for background in BACKGROUNDS
                )
            ),
            "mlp1_repair_license": rescue,
            "broad_code_register_equivalence": code_core and all(code_positive_control.values()),
        },
    }
