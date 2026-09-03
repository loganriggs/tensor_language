#!/usr/bin/env python3
"""R574: independent CPU audit of the saved R573 v2 row-level decisions."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "numbered_list_factor_localization_rung573_v2_results.json"
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
POSITIONS = ROOT / "numbered_list_semantic_positions_rung573.json"
AMENDMENT = ROOT.parent / "polynomial_causal" / "NUMBERED_LIST_FACTOR_LOCALIZATION_RUNG573_V2_IMPLEMENTATION_AMENDMENT.md"
OUT = ROOT / "numbered_list_factor_localization_rung574_audit.json"
HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    POSITIONS: "b4f4f8e58c03deb3a015141656572dcf1fd0fe0c12027c358b441459c245c16b",
    AMENDMENT: "27729a0e1405221f989ad0f6b9fef5d2f797c137fbe71a044148e9a5e3e0b4d7",
}
TARGETS = ("list_two_line_state_shift", "list_three_line_state_shift")
CONTROLS = ("list_surface_preserved", "list_middle_index_break",
            "list_repeated_index_control", "list_step_two_conflict")
DIRECTIONS = ("base_to_donor", "donor_to_base")
FIT_ARMS = ("complete_heads", "all_label_joint", "final_label_joint", "final_label_score",
            "final_label_value", "final_label_cached_value", "final_label_own_value",
            "all_label_cached_value")
ORDER = ("final_label_cached_value", "final_label_value", "all_label_cached_value",
         "final_label_joint", "all_label_joint", "final_label_score", "final_label_own_value")
EXPECTED = {"FIT": 32, "SELECT": 16}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(a: float, b: float, tolerance: float = 1e-10) -> bool:
    return abs(float(a) - float(b)) <= tolerance


def main() -> None:
    if OUT.exists():
        raise RuntimeError("R574 audit namespace already exists")
    for path, expected in HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"authority changed: {path}")
    result = json.loads(RESULT.read_text())
    rows = json.loads(ROWS.read_text())["rows"]
    positions = {item["row_id"]: item for item in json.loads(POSITIONS.read_text())["mappings"]}
    expected_rows = {split: {family: {row["row_id"] for row in rows
                                     if row["hypothesis_id"] == "numbered_list_index_successor"
                                     and row["split"] == split and row["family_id"] == family}
                             for family in TARGETS + CONTROLS}
                     for split in ("FIT", "SELECT")}
    checks = {
        "checkpoint_exact": result["checkpoint_weights_sha256"]
                            == "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3",
        "price_exact": result["model_forwards"] == result["execution"]["maximum_forwards"] == 280
                       and result["model_backwards"] == 0 and result["model_weights_updated"] is False,
        "split_envelope_exact": result["evaluated_splits"] == ["FIT", "SELECT"]
                                and result["forbidden_splits_opened"] == [],
        "instrument_exact": result["execution"]["fit"]["native_replay_relative_squared_error"] <= 1e-12
                            and result["execution"]["select"]["native_replay_relative_squared_error"] <= 1e-12
                            and max(result["execution"][split]["head_source_sum_relative_squared_error"]
                                    for split in ("fit", "select")) <= 1e-10
                            and max(result["execution"][split]["value_split_relative_squared_error"]
                                    for split in ("fit", "select")) <= 1e-10,
    }
    raw_checks, report_checks = [], []
    for split, arms in (("FIT", FIT_ARMS),
                        ("SELECT", ("complete_heads", result["fit_choice"]["selected_arm"]))):
        raw = result[f"{split.lower()}_raw"]
        for family in TARGETS + CONTROLS:
            for direction in DIRECTIONS:
                for arm in arms:
                    cells = raw[family][direction][arm]
                    ids = {cell["row_id"] for cell in cells}
                    raw_checks.append(len(cells) == EXPECTED[split]
                                      and ids == expected_rows[split][family]
                                      and len(ids) == len(cells))
                    if family in TARGETS:
                        values = np.asarray([cell["donor_direction_answer_margin_effect"] for cell in cells])
                        if arm == "complete_heads":
                            report = result[f"{split.lower()}_ceiling"][family][direction]
                            report_checks.extend([
                                close(values.mean(), report["mean_effect"]),
                                close(np.median(values), report["median_effect"]),
                                close(np.mean(values > 0), report["positive_fraction"]),
                                report["passed"] == (np.mean(values > 0) >= .75
                                                      and report["bootstrap95_lower_mean_effect"] > 0),
                            ])
                        else:
                            report = result[f"{split.lower()}_factor_reports"][arm]["targets"][family][direction]
                            complete = np.asarray([cell["donor_direction_answer_margin_effect"]
                                                   for cell in raw[family][direction]["complete_heads"]])
                            mean_recovery = values.mean() / complete.mean()
                            median_recovery = np.median(values) / np.median(complete)
                            passed = (mean_recovery >= .5 and median_recovery >= .5
                                      and np.mean(values > 0) >= .75
                                      and report["bootstrap95_lower_mean_effect"] > 0)
                            report_checks.extend([
                                close(values.mean(), report["mean_effect"]),
                                close(np.median(values), report["median_effect"]),
                                close(mean_recovery, report["mean_recovery"]),
                                close(median_recovery, report["median_recovery"]),
                                close(np.mean(values > 0), report["positive_fraction"]),
                                report["passed"] == passed,
                            ])
                    elif arm != "complete_heads":
                        report = result[f"{split.lower()}_factor_reports"][arm]["controls"][family][direction]
                        answer = np.asarray([abs(cell["registered_answer_margin_change"]) for cell in cells])
                        rms = np.asarray([cell["full_vocabulary_logit_rms"] for cell in cells])
                        preserved = np.mean([cell["registered_answer_remains_best"] for cell in cells])
                        answer_fraction = np.median(answer) / result["fit_control_scales"]["answer_margin"]
                        rms_fraction = np.median(rms) / result["fit_control_scales"]["full_vocabulary_logit_rms"]
                        passed = answer_fraction <= .25 and rms_fraction <= .25 and preserved >= .75
                        report_checks.extend([
                            close(np.median(answer), report["median_absolute_answer_margin_change"]),
                            close(np.median(rms), report["median_full_vocabulary_logit_rms"]),
                            close(answer_fraction, report["fraction_of_fit_target_answer_scale"]),
                            close(rms_fraction, report["fraction_of_fit_target_logit_rms_scale"]),
                            close(preserved, report["registered_answer_preserved_fraction"]),
                            report["passed"] == passed,
                        ])
    fit_pass = {arm: result["fit_factor_reports"][arm]["passed"] for arm in ORDER}
    eligible = [arm for arm in ORDER if fit_pass[arm]]
    checks.update({
        "every_raw_cell_has_exact_rows": all(raw_checks),
        "all_saved_decisions_recomputed": all(report_checks),
        "fit_order_recomputed": result["fit_choice"]["fixed_order"] == list(ORDER)
                                and result["fit_choice"]["eligible_arms"] == eligible
                                and result["fit_choice"]["selected_arm"] == eligible[0],
        "selected_select_recomputed": result["select_factor_reports"][eligible[0]]["passed"] is True
                                      and result["selected_factor_held"] is True,
        "terminal_predicates_consistent": result["pred_a_exact_replay_and_fit_complete_head_ceiling"] is True
                                          and result["pred_b_fit_exact_factor_selected"] is True
                                          and result["pred_c_selected_factor_holds_on_select"] is True
                                          and result["all_gates_pass"] is True,
    })
    selected = result["fit_choice"]["selected_arm"]
    selected_target = [cell["donor_direction_answer_margin_effect"]
                       for split in ("fit", "select")
                       for family in TARGETS for direction in DIRECTIONS
                       for cell in result[f"{split}_raw"][family][direction][selected]]
    selected_controls = [cell for split in ("fit", "select")
                         for family in CONTROLS for direction in DIRECTIONS
                         for cell in result[f"{split}_raw"][family][direction][selected]]
    token_semantics = []
    row_by_id = {row["row_id"]: row for row in rows}
    for row_id, mapping in positions.items():
        row = row_by_id[row_id]
        base_final = mapping["endpoints"]["base"]["final_label_position"]
        donor_final = mapping["endpoints"]["donor"]["final_label_position"]
        token_semantics.append({
            "target": row["family_id"] in TARGETS,
            "final_label_token_changes": row["base_ids"][base_final] != row["donor_ids"][donor_final],
        })
    checks["selected_target_all_positive"] = all(value > 0 for value in selected_target)
    checks["selected_control_terms_are_exact_noops"] = all(
        cell["registered_answer_margin_change"] == 0.0 and cell["full_vocabulary_logit_rms"] == 0.0
        for cell in selected_controls)
    checks["final_label_change_separates_targets_from_controls"] = all(
        item["final_label_token_changes"] == item["target"] for item in token_semantics)
    audit = {
        "rung": 574, "stage": "numbered_list_factor_post_result_cpu_audit",
        "source_result_sha256": sha256(RESULT), "all_checks_pass": all(checks.values()),
        "checks": checks, "selected_arm": selected, "eligible_fit_arms": eligible,
        "audited_row_cells": len(raw_checks), "audited_summary_equalities": len(report_checks),
        "selected_target_effect_count": len(selected_target),
        "selected_target_minimum_effect": min(selected_target),
        "selected_control_row_count": len(selected_controls),
        "interpretation": (
            "For these list-state swaps, nearly all of the causal effect of L8H7/L8H3 is carried by the "
            "layer-0 cached value at the final visible label; the score and layer-8 own-value parts are not sufficient."
        ),
        "declared_limit": (
            "Every registered answer-preserving control keeps the final label token fixed, so the selected cached-value "
            "transplant is exactly a no-op on those controls. This cleanly establishes dependence on final-label identity "
            "but is not a collateral-damage removal test on unrelated circuits. Weight compilation, removal, OOD, and "
            "cross-format transfer remain open."
        ),
        "model_loaded": False, "model_forwards": 0, "outcomes_opened": ["saved_R573_v2_result"],
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({key: audit[key] for key in ("all_checks_pass", "selected_arm",
                                                   "eligible_fit_arms", "audited_row_cells",
                                                   "selected_target_minimum_effect", "declared_limit")}, indent=2))


if __name__ == "__main__":
    main()
