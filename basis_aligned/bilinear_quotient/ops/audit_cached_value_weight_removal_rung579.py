#!/usr/bin/env python3
"""R579: independent, model-free audit of every R576 row-level decision."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
POLY = ROOT.parent / "polynomial_causal"
RESULT = ROOT / "numbered_list_cached_value_weight_removal_rung576_results.json"
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
POSITIONS = ROOT / "numeric_factor_removal_positions_rung575.json"
R573_RESULT = ROOT / "numbered_list_factor_localization_rung573_v2_results.json"
R574_AUDIT = ROOT / "numbered_list_factor_localization_rung574_audit.json"
PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_WEIGHT_REMOVAL_RUNG576_PREREGISTRATION.md"
SCRIPT = ROOT / "ops" / "numbered_list_cached_value_weight_removal_rung576.py"
AUDIT_PREREG = POLY / "NUMBERED_LIST_CACHED_VALUE_REMOVAL_AUDIT_RUNG579.md"
OUT = ROOT / "numbered_list_cached_value_weight_removal_rung579_audit.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    POSITIONS: "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
    R573_RESULT: "052930b8b9086e8b7606e3d05929f521f468c04427be8d1182720f1772ee43ec",
    R574_AUDIT: "3d6580ee1a4f1bb77c07e4ee2b404bc23dc70f733db31425bc5da2a11a25a04e",
    PREREG: "a776ebc1df29a6f3193d3315e190ec9494c95905596e450461c002378f8f59b6",
    SCRIPT: "91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a",
    AUDIT_PREREG: "cc6e437aa345c7a6007aedb1e36d0c63e1458de9fed05f58d38ea4bb47f0bd5b",
}
RESULT_INPUTS = (ROWS, POSITIONS, R573_RESULT, R574_AUDIT, PREREG)
LIST_TARGETS = ("list_two_line_state_shift", "list_three_line_state_shift",
                "list_surface_preserved", "list_middle_index_break", "list_step_two_conflict")
COPY_CONTROLS = ("list_repeated_index_control", "sequence_digit_copy_control",
                 "sequence_word_copy_control")
SEQUENCE_TARGETS = ("sequence_digit_state_shift", "sequence_word_state_shift",
                    "sequence_cross_format_shift")
FAMILIES = LIST_TARGETS + COPY_CONTROLS + SEQUENCE_TARGETS
ENDPOINTS = ("base", "donor")
BOOTSTRAPS = 2000
SEED = 576
WEIGHTS = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    if math.isinf(float(left)) or math.isinf(float(right)):
        return float(left) == float(right)
    return abs(float(left) - float(right)) <= tolerance


def lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[indices].mean(1), .025))


def ratio(value: float, scale: float) -> float:
    return float(value / scale) if scale > 0 else float("inf")


def audit_target_report(raw: dict, saved: dict, families: tuple[str, ...], seed: int) -> tuple[list[bool], bool, int]:
    checks: list[bool] = []
    overall = True
    for family in families:
        for endpoint in ENDPOINTS:
            cells = raw[family][endpoint]
            damage = [cell["margin_damage"] for cell in cells]
            ce_values = [cell["ce_increase"] for cell in cells]
            computed = {
                "n": len(cells),
                "mean_margin_damage": float(np.mean(damage)),
                "median_margin_damage": float(np.median(damage)),
                "positive_margin_damage_fraction": float(np.mean(np.asarray(damage) > 0)),
                "bootstrap95_lower_mean_margin_damage": lower(damage, seed),
                "mean_ce_increase": float(np.mean(ce_values)),
                "bootstrap95_lower_mean_ce_increase": lower(ce_values, seed + 1),
                "median_logit_rms": float(np.median([cell["full_vocabulary_logit_rms"] for cell in cells])),
                "median_term_norm": float(np.median([cell["compiled_residual_term_norm"] for cell in cells])),
            }
            seed += 2
            computed["passed"] = bool(computed["positive_margin_damage_fraction"] >= .75
                                      and computed["bootstrap95_lower_mean_margin_damage"] > 0
                                      and computed["bootstrap95_lower_mean_ce_increase"] > 0)
            item = saved[family][endpoint]
            checks.extend(item[key] == value if isinstance(value, (bool, int)) else close(item[key], value)
                          for key, value in computed.items())
            overall &= computed["passed"]
    return checks, bool(overall), seed


def fit_scales(raw: dict) -> dict[str, float]:
    cells = [cell for family in LIST_TARGETS for endpoint in ENDPOINTS for cell in raw[family][endpoint]]
    return {
        "margin_damage": float(np.median([abs(cell["margin_damage"]) for cell in cells])),
        "logit_rms": float(np.median([cell["full_vocabulary_logit_rms"] for cell in cells])),
        "term_norm": float(np.median([cell["compiled_residual_term_norm"] for cell in cells])),
    }


def audit_controls(raw: dict, saved: dict, scales: dict[str, float]) -> tuple[list[bool], bool]:
    checks: list[bool] = []
    overall = True
    for family in COPY_CONTROLS:
        for endpoint in ENDPOINTS:
            cells = raw[family][endpoint]
            computed = {
                "n": len(cells),
                "median_term_norm_fraction_of_fit_list": ratio(
                    float(np.median([cell["compiled_residual_term_norm"] for cell in cells])), scales["term_norm"]),
                "answer_preserved_fraction": float(np.mean([cell["answer_remains_best"] for cell in cells])),
                "mean_ce_increase": float(np.mean([cell["ce_increase"] for cell in cells])),
                "median_absolute_margin_change_fraction_of_fit_list": ratio(
                    float(np.median([abs(cell["margin_damage"]) for cell in cells])), scales["margin_damage"]),
                "median_logit_rms_fraction_of_fit_list": ratio(
                    float(np.median([cell["full_vocabulary_logit_rms"] for cell in cells])), scales["logit_rms"]),
            }
            computed["passed"] = bool(
                computed["median_term_norm_fraction_of_fit_list"] >= .10
                and computed["answer_preserved_fraction"] >= .75
                and computed["mean_ce_increase"] <= .10
                and computed["median_absolute_margin_change_fraction_of_fit_list"] <= .25
                and computed["median_logit_rms_fraction_of_fit_list"] <= .25)
            item = saved[family][endpoint]
            checks.extend(item[key] == value if isinstance(value, (bool, int)) else close(item[key], value)
                          for key, value in computed.items())
            overall &= computed["passed"]
    return checks, bool(overall)


def audit_split(result: dict, split: str, expected_ids: dict) -> dict:
    raw = result[f"{split.lower()}_raw"]
    report = result[f"{split.lower()}_report"]
    row_checks = []
    for family in FAMILIES:
        for endpoint in ENDPOINTS:
            cells = raw[family][endpoint]
            ids = [cell["row_id"] for cell in cells]
            row_checks.append(len(ids) == len(set(ids)) and set(ids) == expected_ids[split][family])
    seed = SEED if split == "FIT" else SEED + 1000
    target_checks, list_pass, next_seed = audit_target_report(
        raw, report["list_necessity"], LIST_TARGETS, seed)
    scales = fit_scales(raw) if split == "FIT" else result["fit_scales"]
    control_checks, copy_pass = audit_controls(raw, report["active_copy_controls"], scales)
    sequence_checks, sequence_pass, _ = audit_target_report(
        raw, report["sequence_shared_characterization"], SEQUENCE_TARGETS, next_seed)
    return {
        "rows_exact": all(row_checks),
        "summary_equalities": all(target_checks + control_checks + sequence_checks),
        "list_pass": list_pass,
        "copy_pass": copy_pass,
        "sequence_pass": sequence_pass,
        "scales": scales,
        "report_flags_exact": (
            report["list_necessity_pass"] == list_pass
            and report["active_copy_controls_pass"] == copy_pass
            and report["all_sequence_successor_cells_pass"] == sequence_pass
            and report["required_pass"] == (list_pass and copy_pass)),
    }


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        for path, expected in HASHES.items():
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError(f"frozen authority changed: {path}")
        print(json.dumps({"status": "dryrun_passed", "rung": 579, "model_loaded": False,
                          "requires_result": str(RESULT)}, indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R579 audit namespace already exists")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen authority changed: {path}")
    result = json.loads(RESULT.read_text())
    rows = [row for row in json.loads(ROWS.read_text())["rows"]
            if row["split"] in {"FIT", "SELECT"} and row["family_id"] in FAMILIES]
    positions = json.loads(POSITIONS.read_text())
    expected_ids = {split: {family: {row["row_id"] for row in rows
                                    if row["split"] == split and row["family_id"] == family}
                            for family in FAMILIES}
                    for split in ("FIT", "SELECT")}
    position_ids = {item["row_id"] for item in positions["records"]}
    checks = {
        "result_inputs_exact": all(result["input_sha256"][str(path)] == HASHES[path]
                                   for path in RESULT_INPUTS),
        "semantic_rows_exact": len(rows) == 528 and position_ids == {row["row_id"] for row in rows},
        "forbidden_splits_closed": result["forbidden_splits_opened"] == [],
        "weight_identity_exact": result["checkpoint_weights_sha256"] == WEIGHTS,
        "forward_ceiling_exact": result["execution"]["maximum_forwards"] == 210,
        "no_training": result["model_backwards"] == 0 and result["model_weights_updated"] is False,
    }
    fit = audit_split(result, "FIT", expected_ids)
    for key in ("rows_exact", "summary_equalities", "report_flags_exact"):
        checks[f"fit_{key}"] = fit[key]
    scales_exact = all(close(result["fit_scales"][key], value) for key, value in fit["scales"].items())
    checks["fit_scales_exact"] = scales_exact
    fit_equivalence = result["fit_equivalence"]
    exact_error_keys = ("cached_bus_relative_squared_error", "projected_term_relative_squared_error",
                        "head_source_sum_relative_squared_error", "value_split_relative_squared_error",
                        "activation_vs_weight_logits_relative_squared_error")
    fit_equivalence_exact = (fit_equivalence["passed"]
                             == all(fit_equivalence["max_errors"][key] <= 1e-10 for key in exact_error_keys))
    fit_required = bool(fit_equivalence["passed"] and fit["list_pass"] and fit["copy_pass"])
    select_expected = fit_required
    checks["conditional_split_rule_exact"] = result["evaluated_splits"] == (
        ["FIT", "SELECT"] if select_expected else ["FIT"])
    checks["fit_equivalence_decision_exact"] = fit_equivalence_exact
    select = None
    if select_expected:
        select = audit_split(result, "SELECT", expected_ids)
        for key in ("rows_exact", "summary_equalities", "report_flags_exact"):
            checks[f"select_{key}"] = select[key]
        select_equivalence = result["select_equivalence"]
        checks["select_equivalence_decision_exact"] = (
            select_equivalence["passed"]
            == all(select_equivalence["max_errors"][key] <= 1e-10 for key in exact_error_keys))
    else:
        checks["select_outputs_absent"] = all(result[key] is None for key in (
            "select_equivalence", "select_report", "select_raw")) and result["execution"]["select"] is None
    fit_replay = result["execution"]["fit"]["native_replay_relative_squared_error"] <= 1e-12
    select_replay = (not select_expected or
                     result["execution"]["select"]["native_replay_relative_squared_error"] <= 1e-12)
    exact = bool(result["checkpoint_weights_sha256"] == WEIGHTS and fit_equivalence["passed"]
                 and fit_replay and select_replay)
    select_required = bool(select_expected and result["select_equivalence"]["passed"]
                           and select is not None and select["list_pass"] and select["copy_pass"])
    all_required = bool(exact and fit_required and select_required)
    expected_forwards = 210 if select_expected else 123
    checks.update({
        "forward_price_exact": result["model_forwards"] == expected_forwards
                               and result["model_forwards"] == result["execution"]["fit"]["model_forwards"]
                               + (result["execution"]["select"]["model_forwards"] if select_expected else 0)
                               + (102 if select_expected else 54),
        "pred_a_exact": result["pred_a_exact_weight_compilation"] == exact,
        "pred_b_exact": result["pred_b_list_necessity"] == bool(
            fit["list_pass"] and select is not None and select["list_pass"]),
        "pred_c_exact": result["pred_c_active_copy_preservation"] == bool(
            fit["copy_pass"] and select is not None and select["copy_pass"]),
        "pred_d_exact": result["pred_d_shared_sequence_successor"] == bool(
            fit["sequence_pass"] and select is not None and select["sequence_pass"]),
        "terminal_gate_exact": result["all_required_gates_pass"] == all_required,
        "decision_exact": result["decision"] == (
            "weight_factor_selectively_removable" if all_required else
            ("invalid_compilation" if not exact else "removal_or_selectivity_null")),
    })
    copy_cells = [cell for split in ("fit", "select") if result.get(f"{split}_raw") is not None
                  for family in COPY_CONTROLS for endpoint in ENDPOINTS
                  for cell in result[f"{split}_raw"][family][endpoint]]
    audit = {
        "rung": 579,
        "stage": "cached_value_weight_removal_independent_cpu_audit",
        "source_result_sha256": sha256(RESULT),
        "audit_preregistration_sha256": sha256(AUDIT_PREREG),
        "all_checks_pass": all(checks.values()),
        "checks": checks,
        "recomputed": {"fit": fit, "select": select},
        "copy_control_rows": len(copy_cells),
        "copy_control_nonzero_term_fraction": float(np.mean(
            [cell["compiled_residual_term_norm"] > 0 for cell in copy_cells])),
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
        "outcomes_opened": ["saved_R576_result"],
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "all_checks_pass", "copy_control_rows", "copy_control_nonzero_term_fraction",
        "model_forwards")}, indent=2))
    if not audit["all_checks_pass"]:
        raise RuntimeError("R579 audit failed")


if __name__ == "__main__":
    main()
