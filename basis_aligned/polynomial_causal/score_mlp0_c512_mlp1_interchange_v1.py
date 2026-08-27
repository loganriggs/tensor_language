#!/usr/bin/env python3
"""Simultaneous source-unit inference for the C512/MLP1 physical 2x2."""

from __future__ import annotations

import hashlib
import json
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


def _signed_effect(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return np.divide(sums, counts, out=np.full_like(sums, np.inf), where=counts > 0)


def absolute_coordinates(coordinates: np.ndarray, n_cells: int) -> np.ndarray:
    """Convert signed CE coordinates to the registered two-sided magnitude."""
    output = coordinates.copy()
    consumer_names = list(MARGINS)
    ce_index = consumer_names.index("ce_abs")
    output[..., ce_index * n_cells:(ce_index + 1) * n_cells] = np.abs(
        output[..., ce_index * n_cells:(ce_index + 1) * n_cells]
    )
    return output


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
    n_units, n_cells = int(shape[0]), int(shape[1])
    if n_cells != 16:
        raise ValueError("the registered assay requires exactly 16 cells")
    reference_counts = None
    for arm in ARMS:
        for consumer in MARGINS:
            counts = np.asarray(ledgers[arm][consumer]["counts"], dtype=np.float64)
            if not np.equal(counts, np.floor(counts)).all():
                raise ValueError("ledger counts must be integers")
            if reference_counts is None:
                reference_counts = counts
            elif not np.array_equal(counts, reference_counts):
                raise ValueError("all arms and consumers must have identical support counts")
    return n_units, n_cells


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
            signed_effects = _signed_effect(sums.sum(0), counts.sum(0))
            effects = np.abs(signed_effects) if consumer == "ce_abs" else signed_effects
            ratios = effects / margin
            support = (counts > 0).sum(0)
            support_passes = support_passes and bool((support >= minimum_support).all())
            coordinates[arm_index, offset:offset + n_cells] = signed_effects / margin
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
                signed_effects = _signed_effect(sums[sampled].sum(1), counts[sampled].sum(1))
                local[:, arm_index, coordinate_offset:coordinate_offset + n_cells] = signed_effects / margin
                coordinate_offset += n_cells
        output[offset:offset + size] = local
    return output


def familywise_bounds(
    coordinates: np.ndarray, bootstrap: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    """Use two-sided signed-CE and one-sided nonnegative coordinate errors."""
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
    """Conservative max reduction from simultaneous coordinate bands.

    On the common coordinate event, max(B) >= max(Bhat)-c_lower and
    max(C) <= max(Chat)+c_upper. Thus the reported difference lower bound never
    centers after an arm maximum or absolute-value kink.
    """
    n_cells = coordinates.shape[1] // len(MARGINS)
    absolute_point_coordinates = absolute_coordinates(coordinates, n_cells)
    points = absolute_point_coordinates.max(1)
    point_differences = {}
    pointwise_no_worse = {}
    for name, (baseline, candidate) in comparisons.items():
        bi, ci = ARMS.index(baseline), ARMS.index(candidate)
        point_differences[name] = float(points[bi] - points[ci])
        pointwise_no_worse[name] = bool(np.all(
            absolute_point_coordinates[ci] <= absolute_point_coordinates[bi]
        ))
    correction = float(2 * joint_error)
    return {
        "familywise_lower_correction": correction,
        "comparisons": {
            name: {
                "baseline": comparisons[name][0],
                "candidate": comparisons[name][1],
                "point_max_reduction": point_differences[name],
                "familywise_95pct_lcb_reduction": point_differences[name] - correction,
                "candidate_pointwise_no_worse": pointwise_no_worse[name],
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
    upper, lower, upper_error, lower_error, joint_error = familywise_bounds(coordinates, bootstrap)
    for arm_index, arm in enumerate(ARMS):
        reports[arm].update({
            "point_max_standardized_effect": float(
                absolute_coordinates(coordinates[arm_index:arm_index + 1],
                                     coordinates.shape[1] // len(MARGINS))[0].max()
            ),
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
            "familywise_joint_two_sided_correction": joint_error,
        },
        "arms": reports,
        "rescue": comparison_bounds(coordinates, joint_error, comparisons),
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


def ordered_ids_sha256(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def integer_array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def validate_unit_identity(identity: Mapping[str, object], expected_hashes: Mapping[str, object]) -> None:
    expected = {
        "fineweb": ("source_document", 384, 1170, 2, 6),
        "code": ("source_file", 48, 192, 4, 4),
    }
    if set(identity) != set(expected):
        raise ValueError("unit identity must contain exactly FineWeb and code")
    for domain, (kind, n_units, n_rows, minimum_rows, maximum_rows) in expected.items():
        record = identity[domain]
        if record.get("unit_kind") != kind:
            raise ValueError(f"{domain} resampling unit changed")
        ordered = record.get("ordered_ids")
        row_to_unit = np.asarray(record.get("row_to_unit"), dtype=np.int64)
        if (not isinstance(ordered, list) or len(ordered) != n_units
                or len(set(ordered)) != n_units or not all(isinstance(value, str) for value in ordered)):
            raise ValueError(f"{domain} ordered unit ids are invalid")
        if (row_to_unit.shape != (n_rows,) or bool((row_to_unit < 0).any())
                or bool((row_to_unit >= n_units).any())):
            raise ValueError(f"{domain} row-to-unit mapping is invalid")
        occupancy = np.bincount(row_to_unit, minlength=n_units)
        if int(occupancy.min()) != minimum_rows or int(occupancy.max()) != maximum_rows:
            raise ValueError(f"{domain} row-to-unit occupancy changed")
        if domain == "fineweb" and not set(occupancy.tolist()).issubset({2, 4, 6}):
            raise ValueError("FineWeb occupancy is not two windows per frozen chunk")
        observed = {
            "ordered_ids_sha256": ordered_ids_sha256(ordered),
            "row_to_unit_sha256": integer_array_sha256(row_to_unit),
            "occupancy_sha256": integer_array_sha256(occupancy),
        }
        if observed != expected_hashes.get(domain):
            raise ValueError(f"{domain} exact authority-bound unit identity changed")


def validate_integrity(
    authority: Mapping[str, object], integrity: Mapping[str, object]
) -> bool:
    if authority.get("status") != "frozen_before_any_c512_mlp1_evaluation_forward":
        return False
    contract = authority.get("integrity_contract", {})
    expected_counts = contract.get("exact_call_counts", {})
    actual_counts = integrity.get("call_counts", {})
    count_keys = {
        "candidate_original_down_calls", "poison_canary_calls",
        "mlp1_teacher_calls", "c512_proxy_calls",
    }
    if (set(expected_counts) != count_keys or set(actual_counts) != count_keys
            or not all(isinstance(value, int) and value >= 0 for value in expected_counts.values())
            or actual_counts != expected_counts
            or actual_counts.get("candidate_original_down_calls") != 0
            or actual_counts.get("poison_canary_calls") != 1
            or actual_counts.get("mlp1_teacher_calls", 0) <= 0
            or actual_counts.get("c512_proxy_calls", 0) <= 0):
        return False
    hash_keys = {
        "source_closure_sha256", "row_receipt_sha256", "row_tensor_sha256",
        "c512_program_sha256", "model_checkpoint_sha256", "code_register_sha256",
    }
    bound_hashes = contract.get("bound_hashes", {})
    observed_hashes = integrity.get("observed_hashes", {})
    if (set(bound_hashes) != hash_keys or set(observed_hashes) != hash_keys
            or not all(isinstance(value, str) and len(value) == 64 for value in bound_hashes.values())
            or observed_hashes != bound_hashes):
        return False
    replay = integrity.get("parent_replay", {})
    tolerance = contract.get("parent_replay_tolerances", {})
    metric_keys = {"raw_logits_max_abs", "capped_logits_max_abs", "ce_abs"}
    if (set(replay) != set(BACKGROUNDS) or set(tolerance) != metric_keys
            or any(not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0
                   for value in tolerance.values())):
        return False
    for background in BACKGROUNDS:
        values = replay[background]
        if set(values) != metric_keys | {"passes"} or values.get("passes") is not True:
            return False
        if any(not isinstance(values[key], (int, float)) or not np.isfinite(values[key])
               or values[key] < 0 or values[key] > tolerance[key] for key in metric_keys):
            return False
    return True


def validate_coverage(coverage: Mapping[str, object]) -> None:
    if set(coverage) != {"fineweb", "code"} or set(coverage["fineweb"]) != {"wave_A", "wave_B", "pooled"}:
        raise ValueError("coverage schema changed")
    values = [coverage["fineweb"][key] for key in ("wave_A", "wave_B", "pooled")]
    values.append(coverage["code"])
    if any(not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1
           for value in values):
        raise ValueError("coverage must contain finite proportions")


def score_result(payload: Mapping[str, object], *, n_bootstrap: int = N_BOOTSTRAP, seed: int = SEED) -> dict[str, object]:
    fineweb = payload["sufficient_statistics"]["fineweb"]
    code = payload["sufficient_statistics"]["code"]
    if validate_ledgers(fineweb)[0] != 384 or validate_ledgers(code)[0] != 48:
        raise ValueError("registered FineWeb/code source-unit counts changed")
    authority = payload["authority"]
    contract = authority.get("integrity_contract", {})
    validate_unit_identity(payload["unit_identity"], contract.get("unit_identity_hashes", {}))
    validate_coverage(payload["coverage"])
    integrity_passes = validate_integrity(authority, payload["integrity"])
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
        background: integrity_passes and all(
            fineweb_scopes[scope]["arms"][f"{background}/native_write"][
                "familywise_95pct_lcb_max_standardized_effect"
            ] > 1
            for scope in ("wave_A", "wave_B", "pooled")
        )
        for background in BACKGROUNDS
    }
    code_positive_control = {
        background: integrity_passes and code_scope["arms"][f"{background}/native_write"][
            "familywise_95pct_lcb_max_standardized_effect"
        ] > 1
        for background in BACKGROUNDS
    }
    fineweb_core = integrity_passes and all(
        fineweb_gates[arm]["equivalence_credit"] for arm in CORE_EQUIVALENCE
    )
    code_core = integrity_passes and all(
        code_gates[arm]["equivalence_credit"] for arm in CORE_EQUIVALENCE
    )
    rescue_diagnostics = {}
    for background in BACKGROUNDS:
        comparisons = [
            fineweb_scopes[scope]["rescue"]["comparisons"][f"rescue_{background}"]
            for scope in ("wave_A", "wave_B")
        ]
        rescue_diagnostics[background] = {
            "lcb_positive_each_wave": all(value["familywise_95pct_lcb_reduction"] > 0
                                          for value in comparisons),
            "pointwise_no_free_rider_each_wave": all(value["candidate_pointwise_no_worse"]
                                                      for value in comparisons),
        }
    observational = [f"{background}/observational_CC" for background in BACKGROUNDS]
    observational_pass = integrity_passes and all(
        fineweb_gates[arm]["equivalence_credit"] for arm in observational
    )
    downstream_null = fineweb_core and all(positive_control.values())
    powered_shuffle = {
        background: integrity_passes and all(
            fineweb_scopes[scope]["arms"][f"{background}/shuffle"][
                "familywise_95pct_lcb_max_standardized_effect"
            ] > 1
            for scope in ("wave_A", "wave_B", "pooled")
        )
        for background in BACKGROUNDS
    }
    causal_arms = tuple(
        f"{background}/{contrast}"
        for background in BACKGROUNDS for contrast in ("write_on_O", "write_on_C", "interaction")
    )
    powered_interface_break = integrity_passes and observational_pass and any(
        fineweb_scopes["pooled"]["arms"][arm]["familywise_95pct_lcb_max_standardized_effect"] > 1
        and fineweb_gates[arm]["support_each_wave"]
        and fineweb_gates[arm]["coverage_each_wave_ge_90pct"]
        and fineweb_gates[arm]["over_margin_family_identity_stable"]
        for arm in causal_arms
    )
    live_rescue = rescue_diagnostics["live"]
    repair_license = bool(
        observational_pass and live_rescue["lcb_positive_each_wave"]
        and live_rescue["pointwise_no_free_rider_each_wave"]
    )
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
        "integrity_passes": integrity_passes,
        "decisions": {
            "fresh_observational_equivalence": observational_pass,
            "downstream_null_on_registered_fineweb_backgrounds": downstream_null,
            "powered_cancellation_or_interface_break": powered_interface_break,
            "positive_control_each_background": positive_control,
            "code_positive_control_each_background": code_positive_control,
            "powered_shuffle_harm_each_background": powered_shuffle,
            "conditional_alignment_null": bool(downstream_null and any(powered_shuffle.values())),
            "mlp1_repair_diagnostics": rescue_diagnostics,
            "mlp1_repair_license_live": repair_license,
            "broad_code_register_equivalence": code_core and all(code_positive_control.values()),
        },
    }
