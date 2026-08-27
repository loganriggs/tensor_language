#!/usr/bin/env python3
"""Pure source-document inference for the C512 -> physical MLP2 factorial.

The evaluator emits signed source-document sufficient statistics.  This module
contains the complete frozen contrast family, simultaneous bootstrap, integrity
checks, and decision logic, but it never loads the model or selects rows.
"""

from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np


MARGINS = {"kl": 0.01, "ce_abs": 0.0075, "centered_logit_nrmse": 0.05}
CONTRASTS = (
    "observational",
    "prewrite_state",
    "write_on_exact_state",
    "write_on_candidate_state",
    "interaction",
    "omission_exposure",
    "alignment_null",
    "sensitivity",
)
CONTRAST_ORIENTATIONS = {
    "observational": ("OO", "CC"),
    "prewrite_state": ("OO", "CO"),
    "write_on_exact_state": ("OO", "OC"),
    "write_on_candidate_state": ("CO", "CC"),
    "interaction": ("CC", "additive_CO_plus_OC_minus_OO"),
    "omission_exposure": ("O0", "C0"),
    "alignment_null": ("OO", "CS"),
    "sensitivity": ("OO", "ON"),
}
RAW_ARMS = ("OO", "CC", "CO", "OC", "O0", "C0", "CS", "ON")
COMPARISON_DEFINITIONS = {
    "suppression": ("omission_exposure", "observational"),
    "alignment": ("alignment_null", "observational"),
}
N_BOOTSTRAP = 20_000
SEED = 20260831
MIN_DOCUMENTS_PER_CELL = 60
SCOPES = ("wave_A", "wave_B", "pooled")


def frozen_inference_contract() -> dict[str, object]:
    return {
        "raw_arms": list(RAW_ARMS),
        "contrasts": list(CONTRASTS),
        "contrast_orientations": {
            key: list(value) for key, value in CONTRAST_ORIENTATIONS.items()
        },
        "comparisons": {
            key: list(value) for key, value in COMPARISON_DEFINITIONS.items()
        },
        "margins": dict(MARGINS),
        "n_cells": 16,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "scope_seeds": {"wave_A": SEED, "wave_B": SEED + 1, "pooled": SEED + 2},
        "wave_sizes": {"wave_A": 192, "wave_B": 192},
    }


def quantile(values: np.ndarray, probability: float) -> float:
    try:
        return float(np.quantile(values, probability, method="higher"))
    except TypeError:
        return float(np.quantile(values, probability, interpolation="higher"))


def _signed_effect(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(sums, counts, out=np.full_like(sums, np.inf), where=counts > 0)


def absolute_coordinates(coordinates: np.ndarray, n_cells: int) -> np.ndarray:
    """Magnitude-transform signed CE only; KL and nRMSE are nonnegative."""
    output = coordinates.copy()
    ce_index = list(MARGINS).index("ce_abs")
    ce_slice = slice(ce_index * n_cells, (ce_index + 1) * n_cells)
    output[..., ce_slice] = np.abs(output[..., ce_slice])
    return output


def validate_ledgers(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]]
) -> tuple[int, int]:
    if set(ledgers) != set(CONTRASTS):
        raise ValueError("ledger does not contain the registered eight contrasts")
    shape = None
    reference_counts = None
    for contrast in CONTRASTS:
        consumers = ledgers[contrast]
        if set(consumers) != set(MARGINS):
            raise ValueError(f"contrast {contrast} has the wrong metric family")
        for metric, ledger in consumers.items():
            sums = np.asarray(ledger["sums"], dtype=np.float64)
            counts = np.asarray(ledger["counts"], dtype=np.float64)
            if sums.ndim != 2 or counts.shape != sums.shape:
                raise ValueError("ledger arrays must share [source-document,cell] shape")
            if shape is None:
                shape = sums.shape
            elif sums.shape != shape:
                raise ValueError("all ledgers must share one source-document/cell shape")
            if (not np.isfinite(sums).all() or not np.isfinite(counts).all()
                    or bool((counts < 0).any())
                    or not np.equal(counts, np.floor(counts)).all()):
                raise ValueError("ledger has nonfinite, negative, or noninteger statistics")
            if metric != "ce_abs" and bool((sums < 0).any()):
                raise ValueError("KL and centered-logit nRMSE sums must be nonnegative")
            if reference_counts is None:
                reference_counts = counts
            elif not np.array_equal(counts, reference_counts):
                raise ValueError("all contrasts and metrics must have identical support")
    assert shape is not None
    n_units, n_cells = int(shape[0]), int(shape[1])
    if n_cells != 16:
        raise ValueError("the registered assay requires exactly 16 cells")
    return n_units, n_cells


def point_coordinates(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    units: np.ndarray,
    minimum_support: int,
) -> tuple[np.ndarray, dict[str, object]]:
    n_cells = validate_ledgers(ledgers)[1]
    coordinates = np.full((len(CONTRASTS), len(MARGINS) * n_cells), np.inf)
    reports: dict[str, object] = {}
    for contrast_index, contrast in enumerate(CONTRASTS):
        offset = 0
        support_passes = True
        consumers = {}
        for consumer, margin in MARGINS.items():
            sums = np.asarray(ledgers[contrast][consumer]["sums"], dtype=np.float64)[units]
            counts = np.asarray(ledgers[contrast][consumer]["counts"], dtype=np.float64)[units]
            signed = _signed_effect(sums.sum(0), counts.sum(0))
            effects = np.abs(signed) if consumer == "ce_abs" else signed
            standardized = effects / margin
            support = (counts > 0).sum(0)
            support_passes = support_passes and bool((support >= minimum_support).all())
            coordinates[contrast_index, offset:offset + n_cells] = signed / margin
            offset += n_cells
            consumers[consumer] = {
                "cell_effects": effects.tolist(),
                "cell_standardized_effects": standardized.tolist(),
                "support_source_documents": support.tolist(),
                "over_margin": bool(np.max(standardized) > 1),
            }
        reports[contrast] = {"consumers": consumers, "support_passes": support_passes}
    return coordinates, reports


def bootstrap_coordinates(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    units: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_cells = validate_ledgers(ledgers)[1]
    output = np.full((n_bootstrap, len(CONTRASTS), len(MARGINS) * n_cells), np.inf)
    for offset in range(0, n_bootstrap, min(256, n_bootstrap)):
        size = min(256, n_bootstrap - offset)
        sampled = units[rng.integers(0, len(units), size=(size, len(units)))]
        local = np.full((size, len(CONTRASTS), len(MARGINS) * n_cells), np.inf)
        for contrast_index, contrast in enumerate(CONTRASTS):
            coordinate_offset = 0
            for consumer, margin in MARGINS.items():
                sums = np.asarray(ledgers[contrast][consumer]["sums"], dtype=np.float64)
                counts = np.asarray(ledgers[contrast][consumer]["counts"], dtype=np.float64)
                signed = _signed_effect(sums[sampled].sum(1), counts[sampled].sum(1))
                local[:, contrast_index,
                      coordinate_offset:coordinate_offset + n_cells] = signed / margin
                coordinate_offset += n_cells
        output[offset:offset + size] = local
    return output


def familywise_bounds(
    coordinates: np.ndarray, bootstrap: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Coordinate-centered common event; signed CE uses two-sided deviations."""
    if bootstrap.ndim != 3 or coordinates.shape != bootstrap.shape[1:]:
        raise ValueError("coordinate bootstrap shape mismatch")
    n_cells = coordinates.shape[1] // len(MARGINS)
    ce_index = list(MARGINS).index("ce_abs")
    ce_slice = slice(ce_index * n_cells, (ce_index + 1) * n_cells)
    upper_deviation = bootstrap - coordinates[None]
    lower_deviation = coordinates[None] - bootstrap
    signed_ce_deviation = np.abs(bootstrap[..., ce_slice] - coordinates[None, :, ce_slice])
    upper_deviation[..., ce_slice] = signed_ce_deviation
    lower_deviation[..., ce_slice] = signed_ce_deviation
    upper_error = max(0.0, quantile(np.max(upper_deviation, axis=(1, 2)), .95))
    lower_error = max(0.0, quantile(np.max(lower_deviation, axis=(1, 2)), .95))
    joint_error = max(0.0, quantile(np.maximum(
        np.max(upper_deviation, axis=(1, 2)),
        np.max(lower_deviation, axis=(1, 2)),
    ), .95))
    points = absolute_coordinates(coordinates, n_cells).max(1)
    return points + upper_error, points - lower_error, upper_error, lower_error, joint_error


def comparison_bounds(
    coordinates: np.ndarray,
    joint_error: float,
    comparisons: Mapping[str, tuple[str, str]],
) -> dict[str, object]:
    """Conservative baseline-minus-candidate max reduction on one common event."""
    n_cells = coordinates.shape[1] // len(MARGINS)
    absolute = absolute_coordinates(coordinates, n_cells)
    points = absolute.max(1)
    correction = float(2 * joint_error)
    output = {}
    for name, (baseline, candidate) in comparisons.items():
        bi, ci = CONTRASTS.index(baseline), CONTRASTS.index(candidate)
        difference = float(points[bi] - points[ci])
        output[name] = {
            "baseline": baseline,
            "candidate": candidate,
            "point_max_reduction": difference,
            "familywise_95pct_lcb_reduction": difference - correction,
            "candidate_pointwise_no_worse": bool(np.all(absolute[ci] <= absolute[bi])),
        }
    return {"familywise_lower_correction": correction, "comparisons": output}


def score_scope(
    ledgers: Mapping[str, Mapping[str, Mapping[str, object]]],
    units: np.ndarray,
    *,
    minimum_support: int,
    n_bootstrap: int,
    seed: int,
) -> dict[str, object]:
    coordinates, reports = point_coordinates(ledgers, units, minimum_support)
    bootstrap = bootstrap_coordinates(ledgers, units, n_bootstrap=n_bootstrap, seed=seed)
    upper, lower, upper_error, lower_error, joint_error = familywise_bounds(
        coordinates, bootstrap
    )
    absolute = absolute_coordinates(coordinates, 16)
    for index, contrast in enumerate(CONTRASTS):
        reports[contrast].update({
            "point_max_standardized_effect": float(absolute[index].max()),
            "familywise_95pct_ucb_max_standardized_effect": float(upper[index]),
            "familywise_95pct_lcb_max_standardized_effect": float(lower[index]),
            "over_margin_consumer_families": sorted(
                consumer for consumer, report in reports[contrast]["consumers"].items()
                if report["over_margin"]
            ),
        })
    comparisons = comparison_bounds(coordinates, joint_error, COMPARISON_DEFINITIONS)
    return {
        "n_source_documents": len(units),
        "bootstrap": {
            "replicates": n_bootstrap,
            "seed": seed,
            "familywise_upper_correction": upper_error,
            "familywise_lower_correction": lower_error,
            "familywise_joint_two_sided_correction": joint_error,
        },
        "contrasts": reports,
        "comparisons": comparisons,
    }


def ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def integer_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def validate_unit_identity(identity: Mapping[str, object], expected: Mapping[str, str]) -> None:
    if identity.get("unit_kind") != "source_document":
        raise ValueError("the resampling unit must be a source document")
    ordered = identity.get("ordered_ids")
    row_to_unit = np.asarray(identity.get("row_to_unit"), dtype=np.int64)
    wave_labels = identity.get("wave_labels")
    if (not isinstance(ordered, list) or len(ordered) != 384
            or len(set(ordered)) != 384
            or not all(isinstance(value, str) for value in ordered)):
        raise ValueError("ordered source-document ids are invalid")
    if (row_to_unit.ndim != 1 or bool((row_to_unit < 0).any())
            or bool((row_to_unit >= 384).any())):
        raise ValueError("row-to-document mapping is invalid")
    occupancy = np.bincount(row_to_unit, minlength=384)
    if not set(occupancy.tolist()).issubset({2, 4, 6}):
        raise ValueError("each source document must contribute one to three two-window chunks")
    registered_labels = ["A"] * 192 + ["B"] * 192
    if wave_labels != registered_labels:
        raise ValueError("ordered 192/192 wave assignment changed")
    observed = {
        "ordered_ids_sha256": ordered_ids_sha256(ordered),
        "row_to_unit_sha256": integer_array_sha256(row_to_unit),
        "occupancy_sha256": integer_array_sha256(occupancy),
        "wave_labels_sha256": ordered_ids_sha256(wave_labels),
    }
    if observed != expected:
        raise ValueError("authority-bound source-document identity changed")


def validate_coverage(coverage: Mapping[str, object]) -> None:
    if set(coverage) != set(SCOPES):
        raise ValueError("coverage must contain wave_A, wave_B, and pooled")
    if any(not isinstance(value, (int, float)) or not np.isfinite(value)
           or not 0 <= value <= 1 for value in coverage.values()):
        raise ValueError("coverage must contain finite proportions")


def validate_integrity(authority: Mapping[str, object], integrity: Mapping[str, object]) -> bool:
    if authority.get("status") != "frozen_before_any_c512_mlp2_compensation_evaluation_forward":
        return False
    contract = authority.get("integrity_contract", {})
    if authority.get("inference_contract") != frozen_inference_contract():
        return False
    expected_counts = contract.get("exact_call_counts", {})
    actual_counts = integrity.get("call_counts", {})
    required_counts = {
        "candidate_original_down_calls", "poison_canary_calls", "c512_proxy_calls",
    }
    if (set(expected_counts) != required_counts or set(actual_counts) != required_counts
            or actual_counts != expected_counts
            or any(type(value) is not int or value < 0 for value in expected_counts.values())
            or any(type(value) is not int or value < 0 for value in actual_counts.values())
            or actual_counts["candidate_original_down_calls"] != 0
            or actual_counts["poison_canary_calls"] != 1
            or actual_counts["c512_proxy_calls"] <= 0):
        return False
    expected_phases = contract.get("exact_phase_site_call_counts", {})
    actual_phases = integrity.get("phase_site_call_counts", {})
    required_phases = {
        "mlp1_teacher_capture", "mlp2_teacher_capture", "crossed_suffix_replay",
        "parent_suffix_replay", "crossed_forbidden_teacher",
    }
    if (set(expected_phases) != required_phases or actual_phases != expected_phases):
        return False
    if (any(not isinstance(values, dict) for values in actual_phases.values())
            or any(type(value) is not int or value < 0
                   for values in actual_phases.values() for value in values.values())):
        return False
    if (set(expected_phases["mlp1_teacher_capture"]) != {"1"}
            or set(expected_phases["mlp2_teacher_capture"]) != {"2"}
            or set(expected_phases["crossed_suffix_replay"]) != {str(i) for i in range(3, 18)}
            or set(expected_phases["parent_suffix_replay"]) != {str(i) for i in range(3, 18)}
            or set(expected_phases["crossed_forbidden_teacher"]) != {"1", "2"}
            or any(type(value) is not int or value <= 0
                   for phase in ("mlp1_teacher_capture", "mlp2_teacher_capture",
                                 "crossed_suffix_replay", "parent_suffix_replay")
                   for value in expected_phases[phase].values())
            or any(type(value) is not int or value != 0
                   for value in expected_phases["crossed_forbidden_teacher"].values())
            or any(type(value) is not int or value != 0
                   for value in actual_phases["crossed_forbidden_teacher"].values())
            or len(set(expected_phases["crossed_suffix_replay"].values())) != 1
            or len(set(expected_phases["parent_suffix_replay"].values())) != 1):
        return False
    hash_keys = {
        "source_closure_sha256", "row_receipt_sha256", "row_tensor_sha256",
        "c512_program_sha256", "model_checkpoint_sha256", "model_config_sha256",
        "inherited_currency_sha256", "control_realization_sha256",
    }
    bound_hashes = contract.get("bound_hashes", {})
    observed_hashes = integrity.get("observed_hashes", {})
    if (set(bound_hashes) != hash_keys or set(observed_hashes) != hash_keys
            or not all(isinstance(value, str) and len(value) == 64
                       for value in bound_hashes.values())
            or observed_hashes != bound_hashes):
        return False
    tolerance = contract.get("parent_replay_tolerances", {})
    replay = integrity.get("parent_replay", {})
    metric_keys = {"raw_logits_max_abs", "capped_logits_max_abs", "ce_abs"}
    parent_keys = {"exact_live", "candidate_live", "exact_mlp2_omit", "candidate_mlp2_omit"}
    if (set(tolerance) != metric_keys or set(replay) != parent_keys
            or any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                   for value in tolerance.values())):
        return False
    for values in replay.values():
        if set(values) != metric_keys | {"passes"} or values.get("passes") is not True:
            return False
        if any(not isinstance(values[key], (int, float)) or not np.isfinite(values[key])
               or values[key] < 0 or values[key] > tolerance[key] for key in metric_keys):
            return False
    same = integrity.get("same_realization_delta", {})
    carried = integrity.get("carried_state_identity", {})
    controls = integrity.get("control_checks", {})
    delta_tolerance = contract.get("same_realization_delta_tolerance")
    state_tolerance = contract.get("carried_state_identity_tolerance")
    if (set(same) != {"max_abs", "passes"} or same.get("passes") is not True
            or not isinstance(same.get("max_abs"), (int, float))
            or not np.isfinite(same["max_abs"]) or same["max_abs"] < 0
            or not isinstance(delta_tolerance, (int, float))
            or not np.isfinite(delta_tolerance) or delta_tolerance < 0
            or same["max_abs"] > delta_tolerance):
        return False
    if (set(carried) != {"x0_max_abs", "v1_max_abs", "passes"}
            or carried.get("passes") is not True
            or not isinstance(state_tolerance, (int, float))
            or not np.isfinite(state_tolerance) or state_tolerance < 0
            or any(not isinstance(carried[key], (int, float)) or not np.isfinite(carried[key])
                   or carried[key] < 0 or carried[key] > state_tolerance
                   for key in ("x0_max_abs", "v1_max_abs"))):
        return False
    norm_tolerance = contract.get("native_control_norm_tolerance")
    if (set(controls) != {
            "derangement_no_same_document", "derangement_wave_cell_preserving",
            "native_control_norm_max_abs", "passes",
        } or controls.get("derangement_no_same_document") is not True
            or controls.get("derangement_wave_cell_preserving") is not True
            or controls.get("passes") is not True
            or not isinstance(norm_tolerance, (int, float))
            or not np.isfinite(norm_tolerance) or norm_tolerance < 0
            or not isinstance(controls.get("native_control_norm_max_abs"), (int, float))
            or not np.isfinite(controls["native_control_norm_max_abs"])
            or controls["native_control_norm_max_abs"] < 0
            or controls["native_control_norm_max_abs"] > norm_tolerance):
        return False
    return True


def _equivalence(scope_results: Mapping[str, object], contrast: str) -> bool:
    return (
        scope_results["wave_A"]["contrasts"][contrast][
            "familywise_95pct_ucb_max_standardized_effect"
        ] < 1
        and scope_results["wave_B"]["contrasts"][contrast][
            "familywise_95pct_ucb_max_standardized_effect"
        ] < 1
        and scope_results["pooled"]["contrasts"][contrast][
            "familywise_95pct_ucb_max_standardized_effect"
        ] < .8
    )


def _powered(scope_results: Mapping[str, object], contrast: str) -> bool:
    return all(scope_results[scope]["contrasts"][contrast][
        "familywise_95pct_lcb_max_standardized_effect"
    ] > 1 for scope in SCOPES)


def score_result(
    payload: Mapping[str, object], *, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED,
    authoritative: bool = True,
) -> dict[str, object]:
    if authoritative and (n_bootstrap != N_BOOTSTRAP or seed != SEED):
        raise ValueError("authoritative inference constants are frozen")
    ledgers = payload["sufficient_statistics"]
    if validate_ledgers(ledgers)[0] != 384:
        raise ValueError("the registered assay requires 384 source documents")
    authority = payload["authority"]
    contract = authority.get("integrity_contract", {})
    validate_unit_identity(payload["unit_identity"], contract.get("unit_identity_hashes", {}))
    validate_coverage(payload["coverage"])
    integrity_passes = validate_integrity(authority, payload["integrity"])
    scopes = {
        "wave_A": score_scope(ledgers, np.arange(192), minimum_support=MIN_DOCUMENTS_PER_CELL,
                              n_bootstrap=n_bootstrap, seed=seed),
        "wave_B": score_scope(ledgers, np.arange(192, 384),
                              minimum_support=MIN_DOCUMENTS_PER_CELL,
                              n_bootstrap=n_bootstrap, seed=seed + 1),
        "pooled": score_scope(ledgers, np.arange(384), minimum_support=MIN_DOCUMENTS_PER_CELL,
                             n_bootstrap=n_bootstrap, seed=seed + 2),
    }
    common_gates = {
        "authoritative_inference": authoritative,
        "integrity": integrity_passes,
        "coverage_each_wave_ge_90pct": (
            payload["coverage"]["wave_A"] >= .9 and payload["coverage"]["wave_B"] >= .9
        ),
        "support_each_wave": all(
            scopes[scope]["contrasts"][contrast]["support_passes"]
            for scope in ("wave_A", "wave_B") for contrast in CONTRASTS
        ),
    }
    common_passes = all(common_gates.values())
    powered = {contrast: common_passes and _powered(scopes, contrast)
               for contrast in CONTRASTS}
    equivalent = {contrast: common_passes and _equivalence(scopes, contrast)
                  for contrast in CONTRASTS}
    suppression_comparisons = [
        scopes[scope]["comparisons"]["comparisons"]["suppression"] for scope in SCOPES
    ]
    alignment_comparisons = [
        scopes[scope]["comparisons"]["comparisons"]["alignment"] for scope in SCOPES
    ]
    suppression = bool(
        common_passes and powered["omission_exposure"]
        and all(value["familywise_95pct_lcb_reduction"] > 0
                and value["candidate_pointwise_no_worse"]
                for value in suppression_comparisons)
    )
    complete = bool(suppression and equivalent["observational"])
    aligned = bool(
        suppression and powered["sensitivity"]
        and all(value["familywise_95pct_lcb_reduction"] > 0
                and value["candidate_pointwise_no_worse"]
                for value in alignment_comparisons)
    )
    dependencies = {
        "prewrite_state": powered["omission_exposure"],
        "write_on_exact_state": powered["sensitivity"],
        "write_on_candidate_state": powered["sensitivity"],
        "interaction": powered["omission_exposure"] and powered["sensitivity"],
    }
    component_status = {}
    for contrast, dependency in dependencies.items():
        if equivalent[contrast] and dependency:
            component_status[contrast] = "equivalent"
        elif powered[contrast]:
            component_status[contrast] = "powered_non_null"
        else:
            component_status[contrast] = "inconclusive"
    return {
        "schema_version": 1,
        "inference": "coordinatewise-centered simultaneous source-document bootstrap",
        "margins": MARGINS,
        "minimum_documents_per_cell": MIN_DOCUMENTS_PER_CELL,
        "scopes": scopes,
        "common_gates": common_gates,
        "authoritative_inference": authoritative,
        "integrity_passes": integrity_passes,
        "contrast_powered_non_null": powered,
        "contrast_equivalence": equivalent,
        "decisions": {
            "mlp2_suppression_replicates": suppression,
            "complete_compensation": complete,
            "aligned_mlp2_write_compensates": aligned,
            "component_status": component_status,
        },
    }
