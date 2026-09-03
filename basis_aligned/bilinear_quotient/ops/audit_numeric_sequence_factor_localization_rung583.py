#!/usr/bin/env python3
"""R583: CPU-only independent audit of the saved R577 result."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
ROWS_RECEIPT = ROOT / "increment_two_hypothesis_rows_rung567_receipt.json"
CAPABILITY = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
CAPABILITY_AUDIT = ROOT / "numeric_two_hypothesis_capability_rung571_audit.json"
R575 = ROOT / "numeric_factor_removal_positions_rung575.json"
POSITIONS = ROOT / "numeric_sequence_semantic_positions_rung577.json"
POSITION_BUILDER = ROOT / "ops" / "numeric_sequence_semantic_positions_rung577.py"
POSITION_TEST = ROOT / "ops" / "test_numeric_sequence_semantic_positions_rung577.py"
R577_PREREG = POLY / "NUMERIC_SEQUENCE_COMPLETE_STATE_FACTOR_LOCALIZATION_RUNG577_PREREGISTRATION.md"
R577_SCRIPT = ROOT / "ops" / "numeric_sequence_complete_state_factor_localization_rung577.py"
R577_TEST = ROOT / "ops" / "test_numeric_sequence_complete_state_factor_localization_rung577.py"
REPLAY_HELPER = ROOT / "ops" / "numbered_list_factor_localization_rung573.py"
MODEL_FACADE = POLY / "bilin18_observed_model_facade.py"
R577_RESULT = ROOT / "numeric_sequence_complete_state_factor_localization_rung577_results.json"
R577_RUNLOG = ROOT / "runlogs" / "numeric_sequence_complete_state_factor_localization_rung577.log"
COMPLETION_LEDGER = ROOT / "runlogs" / "_completed.txt"
PREREG = POLY / "NUMERIC_SEQUENCE_FACTOR_LOCALIZATION_AUDIT_RUNG583_PREREGISTRATION.md"
SCRIPT = Path(__file__)
TEST = SCRIPT.with_name("test_audit_numeric_sequence_factor_localization_rung583.py")
OUT = ROOT / "numeric_sequence_factor_localization_audit_rung583.json"
DRYRUN = ROOT / "numeric_sequence_factor_localization_audit_rung583_dryrun.json"

HASHES = {
    ROWS: "3a7fa83033ead857bf86b79b5cab2549412c9df1ffc75890e800fbc8de39f053",
    ROWS_RECEIPT: "02b2c37cc23434138accd63e920f417cda10f1c86a4c08c174537149ec2b1072",
    CAPABILITY: "7cc56f22def334673e0035fad7c6a7d1fc58ab8edd3a99744bebd9fb4e6af7e7",
    CAPABILITY_AUDIT: "c5453ddaa4aa46806cbfcb9a9b0941fe8ddbb21c61e5e22d00c1d1cea6dd74bb",
    R575: "3663ebc48e5dca1ff336cb0627fc43c6db8d7d6e1666b81d7631ab150168dd4b",
    POSITIONS: "a6a98715617cf91971655c252553f42d45b59937ecfbf46722b518333721de1d",
    POSITION_BUILDER: "d9e7249d0dcc916d738fa5f821c6beda6ad5a2661b9e38e7c657b94e3d7eb083",
    POSITION_TEST: "d9e2a207036bbfb4aeb5a7b21abb21a1d5410217fdd163aa7296a258a2dc5994",
    R577_PREREG: "a35ac6dbf4ce2ee85e4e047157f0778d33bf066dee9883b94065149ae3252c98",
    R577_SCRIPT: "1c05f14bad61a805e0d6708153d988209346ffb30ea62e0a1232362fb45c9e92",
    R577_TEST: "a8e99c39de79eccd188dbb6a4aa159c959463c0cb5989549838ea8daab0d834c",
    REPLAY_HELPER: "5723e42e2a5f72a4ddab7a20b631e18e0b6d28875ff53f3db2d37d1845d6e076",
    MODEL_FACADE: "b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c",
    R577_RESULT: "c2f6b9fc87175caff0d58e2a8692c8fc071b6c808491aa90a12694493c1fd4d8",
    R577_RUNLOG: "e40a71ec808e213c096ae38e03a578a9f8353dea7fdde03bcd2b5563962533fe",
    PREREG: "6267e209878122a8ed770cdcb295aaa48d5aa1c2817d41f3e186da086c127210",
}
R577_INPUT_HASHES = {str(path): HASHES[path] for path in
                     (ROWS, ROWS_RECEIPT, CAPABILITY, CAPABILITY_AUDIT, R575, POSITIONS, R577_PREREG)}
DRYRUN_AUTHORITY_HASHES = {path: digest for path, digest in HASHES.items()
                          if path not in {R577_RESULT, R577_RUNLOG}}
COMPLETION_RECORD = "19:06 numeric_sequence_complete_state_factor_localization_rung577 exit=0"
COMPLETION_RECORD_SHA256 = "d7513f4063e94ff32e66929495631ffd93a51c01d6303d762b421890603ba303"
CHECKPOINT_SHA256 = "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
HYPOTHESIS = "numeric_sequence_continuation"
TARGETS = ("sequence_digit_state_shift", "sequence_word_state_shift",
           "sequence_cross_format_shift")
RELATION = "sequence_middle_value_break"
CONTROLS = ("sequence_digit_surface_preserved", "sequence_word_surface_preserved",
            "sequence_digit_copy_control", "sequence_word_copy_control",
            "sequence_step_two_conflict")
FAMILIES = TARGETS + (RELATION,) + CONTROLS
DIRECTIONS = ("base_to_donor", "donor_to_base")
SITE_ARMS = ("a8_h73_complete", "a8_all_heads_complete", "post_attention8_state",
             "post_mlp8_state", "post_mlp10_state", "post_mlp12_state", "post_mlp14_state")
FACTOR_ARMS = ("semantic_final_score", "semantic_nonfinal_score", "semantic_all_score",
               "semantic_nonfinal_cached_value", "semantic_final_own_value",
               "semantic_nonfinal_own_value", "semantic_all_own_value",
               "semantic_final_joint", "semantic_nonfinal_joint", "semantic_all_joint")
BOOTSTRAPS = 2_000
SEED = 577
BATCH = 24
ABS_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_authority() -> tuple[list[dict], dict[str, dict]]:
    for path, digest in DRYRUN_AUTHORITY_HASHES.items():
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"frozen authority mismatch: {path}")
    rows_document = json.loads(ROWS.read_text())
    position_document = json.loads(POSITIONS.read_text())
    rows = [row for row in rows_document["rows"] if row["hypothesis_id"] == HYPOTHESIS
            and row["split"] in {"FIT", "SELECT"}]
    positions = {item["row_id"]: item for item in position_document["records"]}
    if len(rows) != 432 or len(positions) != 432 or set(positions) != {row["row_id"] for row in rows}:
        raise RuntimeError("R577 row/position census mismatch")
    if len({row["row_id"] for row in rows}) != 432:
        raise RuntimeError("duplicate R577 row ID")
    for row in rows:
        mapping = positions[row["row_id"]]
        if mapping["group_id"] != row["group_id"] or mapping["family_id"] != row["family_id"] \
                or mapping["split"] != row["split"]:
            raise RuntimeError("semantic position membership mismatch")
        for endpoint in ("base", "donor"):
            item = mapping["endpoints"][endpoint]
            if item["sequence_length"] != len(row[f"{endpoint}_ids"]) \
                    or item["query_position"] != len(row[f"{endpoint}_ids"]) - 1 \
                    or len(item["value_positions"]) != 3:
                raise RuntimeError("semantic endpoint mismatch")
    return rows, positions


def expected_rows(rows: Sequence[dict], split: str, family: str) -> dict[str, str]:
    return {row["row_id"]: row["group_id"] for row in rows
            if row["split"] == split and row["family_id"] == family}


def validate_raw_membership(raw: Mapping[str, object], rows: Sequence[dict], split: str,
                            expected_arms: Sequence[str]) -> dict:
    if set(raw) != set(expected_arms):
        raise RuntimeError("raw arm membership mismatch")
    census = {}
    for arm in expected_arms:
        if set(raw[arm]) != set(FAMILIES):
            raise RuntimeError(f"raw family membership mismatch: {arm}")
        census[arm] = {}
        for family in FAMILIES:
            if set(raw[arm][family]) != set(DIRECTIONS):
                raise RuntimeError(f"raw direction membership mismatch: {arm}/{family}")
            authority = expected_rows(rows, split, family)
            census[arm][family] = {}
            for direction in DIRECTIONS:
                cells = raw[arm][family][direction]
                observed = {cell["row_id"]: cell["group_id"] for cell in cells}
                if len(cells) != len(observed) or observed != authority:
                    raise RuntimeError(f"raw row/group mismatch: {arm}/{family}/{direction}")
                required = ({"natural_effect", "effect", "target_answer_best",
                             "full_vocabulary_logit_rms", "intervention_vector_norm"}
                            if family in TARGETS else
                            {"natural_effect", "effect", "full_vocabulary_logit_rms",
                             "intervention_vector_norm"} if family == RELATION else
                            {"registered_margin_change", "full_vocabulary_logit_rms",
                             "intervention_vector_norm", "preference_sign_preserved",
                             "registered_answer_best", "ce_increase"})
                for cell in cells:
                    if not required <= set(cell):
                        raise RuntimeError(f"raw measurement missing: {arm}/{family}/{direction}")
                    numeric = [value for key, value in cell.items()
                               if key not in {"row_id", "group_id", "target_answer_best",
                                              "preference_sign_preserved", "registered_answer_best"}
                               and value is not None]
                    if not all(math.isfinite(float(value)) for value in numeric):
                        raise RuntimeError(f"raw nonfinite value: {arm}/{family}/{direction}")
                census[arm][family][direction] = len(cells)
    return census


def bootstrap_lower(values: Sequence[float], seed: int, traces: dict, cell_id: str,
                    replicates: int = BOOTSTRAPS) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(array), size=(replicates, len(array)))
    means = array[indices].mean(1)
    traces[cell_id] = {
        "seed": seed, "replicates": replicates, "observations": len(array),
        "index_matrix_sha256": hashlib.sha256(indices.astype(">u4").tobytes()).hexdigest(),
        "statistic_vector_sha256": hashlib.sha256(means.astype(">f8").tobytes()).hexdigest(),
    }
    return float(np.quantile(means, .025))


def effect_report(cells: Sequence[dict], seed: int, positive_bar: float, traces: dict,
                  cell_id: str, replicates: int) -> dict:
    effects = [float(item["effect"]) for item in cells]
    natural = [float(item["natural_effect"]) for item in cells]
    mean_den, median_den = float(np.mean(natural)), float(np.median(natural))
    report = {
        "n": len(cells), "mean_effect": float(np.mean(effects)),
        "median_effect": float(np.median(effects)), "mean_natural_effect": mean_den,
        "median_natural_effect": median_den,
        "mean_recovery": float(np.mean(effects)) / mean_den if mean_den > 0 else None,
        "median_recovery": float(np.median(effects)) / median_den if median_den > 0 else None,
        "positive_fraction": float(np.mean(np.asarray(effects) > 0)),
        "bootstrap95_lower_mean_effect": bootstrap_lower(
            effects, seed, traces, cell_id, replicates),
    }
    report["passed"] = bool(report["mean_recovery"] is not None
                            and report["median_recovery"] is not None
                            and report["mean_recovery"] >= .5
                            and report["median_recovery"] >= .5
                            and report["positive_fraction"] >= positive_bar
                            and report["bootstrap95_lower_mean_effect"] > 0)
    return report


def arm_report(raw: Mapping[str, object], arm: str, seed: int, traces: dict,
               *, reference_scales: dict[str, float] | None = None,
               replicates: int = BOOTSTRAPS) -> dict:
    targets, target_pass = {}, True
    for family in TARGETS:
        targets[family] = {}
        for direction in DIRECTIONS:
            cells = raw[arm][family][direction]
            report = effect_report(cells, seed, .75, traces,
                                   f"{arm}:{family}:{direction}", replicates)
            seed += 1
            report["target_answer_best_fraction"] = float(np.mean([
                item["target_answer_best"] for item in cells]))
            report["passed"] &= report["target_answer_best_fraction"] >= .5
            targets[family][direction] = report
            target_pass &= report["passed"]
    relation, relation_pass = {}, True
    for direction in DIRECTIONS:
        report = effect_report(raw[arm][RELATION][direction], seed, .65, traces,
                               f"{arm}:{RELATION}:{direction}", replicates)
        seed += 1
        relation[direction] = report
        relation_pass &= report["passed"]
    target_cells = [cell for family in TARGETS for direction in DIRECTIONS
                    for cell in raw[arm][family][direction]]
    observed_scales = {
        "answer_effect": float(np.median([abs(cell["effect"]) for cell in target_cells])),
        "logit_rms": float(np.median([cell["full_vocabulary_logit_rms"] for cell in target_cells])),
        "intervention_norm": float(np.median([cell["intervention_vector_norm"] for cell in target_cells])),
    }
    scales = observed_scales if reference_scales is None else reference_scales
    controls, control_pass = {}, min(scales.values()) > 0
    for family in CONTROLS:
        controls[family] = {}
        for direction in DIRECTIONS:
            cells = raw[arm][family][direction]
            ratio = lambda value, scale: float(value / scale) if scale > 0 else float("inf")
            report = {
                "n": len(cells),
                "median_intervention_norm_fraction": ratio(float(np.median([
                    item["intervention_vector_norm"] for item in cells])), scales["intervention_norm"]),
                "median_absolute_margin_change_fraction": ratio(float(np.median([
                    abs(item["registered_margin_change"]) for item in cells])), scales["answer_effect"]),
                "median_logit_rms_fraction": ratio(float(np.median([
                    item["full_vocabulary_logit_rms"] for item in cells])), scales["logit_rms"]),
            }
            if family == "sequence_step_two_conflict":
                report["preference_sign_preserved_fraction"] = float(np.mean([
                    item["preference_sign_preserved"] for item in cells]))
                behavioral = report["preference_sign_preserved_fraction"] >= .75
            else:
                report["registered_answer_preserved_fraction"] = float(np.mean([
                    item["registered_answer_best"] for item in cells]))
                report["mean_ce_increase"] = float(np.mean([item["ce_increase"] for item in cells]))
                behavioral = (report["registered_answer_preserved_fraction"] >= .75
                              and report["mean_ce_increase"] <= .10)
            report["passed"] = bool(report["median_intervention_norm_fraction"] >= .10
                                    and report["median_absolute_margin_change_fraction"] <= .25
                                    and report["median_logit_rms_fraction"] <= .25 and behavioral)
            controls[family][direction] = report
            control_pass &= report["passed"]
    return {"targets": targets, "relation": relation, "controls": controls,
            "observed_target_scales": observed_scales, "control_reference_scales": scales,
            "target_pass": bool(target_pass), "relation_pass": bool(relation_pass),
            "controls_pass": bool(control_pass),
            "passed": bool(target_pass and relation_pass and control_pass)}


def choose(reports: Mapping[str, dict], order: Sequence[str]) -> dict:
    eligible = [arm for arm in order if reports.get(arm, {}).get("passed")]
    return {"fixed_order": list(order), "eligible_arms": eligible,
            "selected_arm": eligible[0] if eligible else None}


def count_length_chunks(items: Sequence[Sequence[int]]) -> int:
    counts = {}
    for item in items:
        counts[len(item)] = counts.get(len(item), 0) + 1
    return sum(math.ceil(count / BATCH) for count in counts.values())


def declared_price(rows: Sequence[dict]) -> dict:
    output = {}
    for split in ("FIT", "SELECT"):
        split_rows = [row for row in rows if row["split"] == split]
        unique = {tuple(row[f"{endpoint}_ids"]) for row in split_rows for endpoint in ("base", "donor")}
        oriented = [row["base_ids"] for row in split_rows] + [row["donor_ids"] for row in split_rows]
        output[split] = {"rows": len(split_rows),
                         "unique_endpoint_capture_chunks": count_length_chunks(list(unique)),
                         "oriented_intervention_chunks": count_length_chunks(oriented)}
    output["maximum_forwards_if_all_conditionals_open"] = (
        output["FIT"]["unique_endpoint_capture_chunks"] + 1
        + len(SITE_ARMS) * output["FIT"]["oriented_intervention_chunks"]
        + len(FACTOR_ARMS) * output["FIT"]["oriented_intervention_chunks"]
        + output["SELECT"]["unique_endpoint_capture_chunks"] + 1
        + (1 + len(FACTOR_ARMS)) * output["SELECT"]["oriented_intervention_chunks"])
    return output


def compare(expected: object, observed: object, path: str, failures: list[str]) -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            failures.append(f"{path}:keys_or_type")
            return
        for key in expected:
            compare(expected[key], observed[key], f"{path}.{key}", failures)
    elif isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            failures.append(f"{path}:length_or_type")
            return
        for index, value in enumerate(expected):
            compare(value, observed[index], f"{path}[{index}]", failures)
    elif isinstance(expected, float):
        if not isinstance(observed, (int, float)) or not math.isclose(
                expected, float(observed), rel_tol=0, abs_tol=ABS_TOLERANCE):
            failures.append(f"{path}:numeric")
    elif expected != observed:
        failures.append(f"{path}:value")


def failed_control_bars(report: dict, family: str) -> list[str]:
    failed = []
    if report["median_intervention_norm_fraction"] < .10:
        failed.append("intervention_norm_below_active_bar")
    if report["median_absolute_margin_change_fraction"] > .25:
        failed.append("registered_margin_change_above_selectivity_bar")
    if report["median_logit_rms_fraction"] > .25:
        failed.append("full_vocabulary_logit_change_above_selectivity_bar")
    if family == "sequence_step_two_conflict":
        if report["preference_sign_preserved_fraction"] < .75:
            failed.append("plus_two_preference_not_preserved")
    else:
        if report["registered_answer_preserved_fraction"] < .75:
            failed.append("registered_answer_not_preserved")
        if report["mean_ce_increase"] > .10:
            failed.append("mean_ce_increase_above_bar")
    return failed


def knowledge_packet(reports: Mapping[str, dict], raw: Mapping[str, object]) -> dict:
    sites = {}
    all_control_norms = []
    for arm in SITE_ARMS:
        report = reports[arm]
        failed = []
        for family in CONTROLS:
            for direction in DIRECTIONS:
                bars = failed_control_bars(report["controls"][family][direction], family)
                if bars:
                    failed.append({"family": family, "direction": direction, "failed_bars": bars})
                all_control_norms.extend(
                    cell["intervention_vector_norm"] for cell in raw[arm][family][direction])
        sites[arm] = {
            "target_direction_cells_passed": sum(
                report["targets"][family][direction]["passed"]
                for family in TARGETS for direction in DIRECTIONS),
            "target_direction_cells_total": 6,
            "failed_target_direction_cells": [
                {"family": family, "direction": direction}
                for family in TARGETS for direction in DIRECTIONS
                if not report["targets"][family][direction]["passed"]],
            "relation_direction_cells_passed": sum(
                report["relation"][direction]["passed"] for direction in DIRECTIONS),
            "relation_direction_cells_total": 2,
            "control_direction_cells_passed": sum(
                report["controls"][family][direction]["passed"]
                for family in CONTROLS for direction in DIRECTIONS),
            "control_direction_cells_total": 10,
            "failed_active_control_cells": failed,
        }
    return {
        "scope": "complete 1152-vector or complete head-output swap at the final query only",
        "sites": sites,
        "all_control_interventions_strictly_nonzero": bool(all(value > 0 for value in all_control_norms)),
        "minimum_saved_control_intervention_norm": float(min(all_control_norms)),
        "reusable_instruction": (
            "Do not repeat broad final-query state swaps as candidate decompositions. H7/H3 and all attention-8 "
            "heads transfer all six sequence-change cells and both relation cells but fail surface-form CE/RMS "
            "selectivity; post-MLP14 also transfers all target/relation cells while failing every control cell. "
            "Split or condition the transported variable, or test selective downstream readers instead."),
    }


def audit_payload(result: Mapping[str, object], rows: Sequence[dict], *,
                  replicates: int = BOOTSTRAPS) -> dict:
    failures, traces = [], {}
    try:
        census = validate_raw_membership(result["fit_site_raw"], rows, "FIT", SITE_ARMS)
        reports = {arm: arm_report(result["fit_site_raw"], arm, SEED + 100 * index,
                                   traces, replicates=replicates)
                   for index, arm in enumerate(SITE_ARMS)}
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        return {"audit_verdict": "failed_independent_audit",
                "audit_failures": [f"raw_reconstruction:{type(error).__name__}:{error}"],
                "independently_recomputed_scientific_decision": None}
    compare(reports, result.get("fit_site_reports"), "fit_site_reports", failures)
    site_choice = choose(reports, SITE_ARMS)
    compare(site_choice, result.get("site_choice"), "site_choice", failures)
    factor_open = reports["a8_h73_complete"]["passed"]
    select_open = site_choice["selected_arm"] is not None
    expected_closed = None
    if not factor_open:
        compare(expected_closed, result.get("fit_factor_raw"), "fit_factor_raw", failures)
        compare(expected_closed, result.get("fit_factor_reports"), "fit_factor_reports", failures)
        compare(expected_closed, result.get("factor_choice"), "factor_choice", failures)
    if not select_open:
        for key in ("select_site_raw", "select_site_report", "select_factor_raw",
                    "select_factor_report", "select_factor_reports"):
            compare(expected_closed, result.get(key), key, failures)
    exactness = result.get("exactness", {})
    exact = bool(result.get("checkpoint_weights_sha256") == CHECKPOINT_SHA256
                 and max(item.get("native_replay_relative_squared_error", 0.)
                         for item in exactness.values()) <= 1e-10
                 and max(item["head_source_sum_relative_squared_error"]
                         for item in exactness.values()) <= 1e-10
                 and max(item["value_split_relative_squared_error"]
                         for item in exactness.values()) <= 1e-10)
    predicted = {
        "pred_a_exact_replay_and_semantic_factor_algebra": exact,
        "pred_b_complete_state_site_holds_fit_and_select": False,
        "pred_c_a8_h73_shared_sequence_carrier": False,
        "pred_d_semantic_factor_holds_fit_and_select": False,
    }
    for key, value in predicted.items():
        compare(value, result.get(key), key, failures)
    price = declared_price(rows)
    compare(price, result.get("execution_price", {}).get("declared"), "declared_price", failures)
    observed_forwards = (price["FIT"]["unique_endpoint_capture_chunks"] + 1
                         + len(SITE_ARMS) * price["FIT"]["oriented_intervention_chunks"])
    envelope = {"observed_forwards": observed_forwards, "model_backwards": 0,
                "fitted_vectors": 0, "weights_updated": False}
    for key, value in envelope.items():
        compare(value, result.get("execution_price", {}).get(key), f"execution_price.{key}", failures)
    compare(["FIT"], result.get("evaluated_splits"), "evaluated_splits", failures)
    compare([], result.get("forbidden_splits_opened"), "forbidden_splits_opened", failures)
    compare(R577_INPUT_HASHES, result.get("input_sha256"), "input_sha256", failures)
    compare("complete_state_site_null", result.get("decision"), "decision", failures)
    packet = knowledge_packet(reports, result["fit_site_raw"])
    if not packet["all_control_interventions_strictly_nonzero"]:
        failures.append("active_controls:zero_intervention")
    for arm in SITE_ARMS:
        for family in CONTROLS:
            for direction in DIRECTIONS:
                if reports[arm]["controls"][family][direction]["median_intervention_norm_fraction"] < .10:
                    failures.append(f"active_control_median:{arm}:{family}:{direction}")
    ordered_traces = {key: traces[key] for key in sorted(traces)}
    return {
        "audit_verdict": "held_independent_audit" if not failures else "failed_independent_audit",
        "audit_failures": failures,
        "independently_recomputed_scientific_decision": "complete_state_site_null",
        "row_cell_census": census,
        "recomputed_site_choice": site_choice,
        "factor_stage_opened": factor_open, "select_opened": select_open,
        "recomputed_exactness_pass": exact, "recomputed_predictions": predicted,
        "recomputed_declared_price": price, "recomputed_observed_forwards": observed_forwards,
        "bootstrap_cell_count": len(traces), "bootstrap_trace_sha256": content_sha256(ordered_traces),
        "bootstrap_traces": ordered_traces, "knowledge_packet": packet,
    }


def verify_run_provenance(result: Mapping[str, object]) -> list[str]:
    failures = []
    records = [line.strip() for line in COMPLETION_LEDGER.read_text().splitlines()
               if "numeric_sequence_complete_state_factor_localization_rung577" in line]
    if records != [COMPLETION_RECORD]:
        failures.append("completion_ledger_record:mismatch")
    elif hashlib.sha256((records[0] + "\n").encode()).hexdigest() != COMPLETION_RECORD_SHA256:
        failures.append("completion_ledger_record:hash")
    try:
        logged = json.loads(R577_RUNLOG.read_text())
    except (json.JSONDecodeError, OSError):
        return failures + ["runlog:not_exact_json"]
    for key in ("pred_a_exact_replay_and_semantic_factor_algebra",
                "pred_b_complete_state_site_holds_fit_and_select",
                "pred_c_a8_h73_shared_sequence_carrier",
                "pred_d_semantic_factor_holds_fit_and_select", "site_choice", "factor_choice",
                "execution_price", "evaluated_splits", "decision"):
        compare(result.get(key), logged.get(key), f"runlog.{key}", failures)
    return failures


def fixture_raw(rows: Sequence[dict], *, control_null: bool) -> dict:
    raw = {}
    for arm in SITE_ARMS:
        raw[arm] = {}
        for family in FAMILIES:
            raw[arm][family] = {}
            for direction in DIRECTIONS:
                cells = []
                for row in rows:
                    if row["split"] != "FIT" or row["family_id"] != family:
                        continue
                    is_effect_cell = family in TARGETS or family == RELATION
                    common = {"row_id": row["row_id"], "group_id": row["group_id"],
                              "full_vocabulary_logit_rms": 1. if is_effect_cell else .1,
                              "intervention_vector_norm": 1. if is_effect_cell else .2}
                    if family in TARGETS:
                        common.update({"effect": 1., "natural_effect": 1., "target_answer_best": True})
                    elif family == RELATION:
                        common.update({"effect": 1., "natural_effect": 1.})
                    elif family == "sequence_step_two_conflict":
                        common.update({"registered_margin_change": 0., "preference_sign_preserved": True,
                                       "registered_answer_best": None, "ce_increase": None})
                    else:
                        common.update({"registered_margin_change": 0., "preference_sign_preserved": None,
                                       "registered_answer_best": True,
                                       "ce_increase": .2 if control_null else 0.})
                    cells.append(common)
                raw[arm][family][direction] = cells
    return raw


def run_dryrun() -> dict:
    rows, _ = load_authority()
    held_raw = fixture_raw(rows, control_null=False)
    null_raw = fixture_raw(rows, control_null=True)
    held_reports = {arm: arm_report(held_raw, arm, SEED + 100 * index, {}, replicates=31)
                    for index, arm in enumerate(SITE_ARMS)}
    null_reports = {arm: arm_report(null_raw, arm, SEED + 100 * index, {}, replicates=31)
                    for index, arm in enumerate(SITE_ARMS)}
    receipt = {
        "schema": "numeric_sequence_factor_localization_audit_rung583_dryrun_v1",
        "status": "dryrun_passed", "rows": len(rows),
        "held_arm_fixture_passes": all(item["passed"] for item in held_reports.values()),
        "control_null_fixture_has_no_eligible_site": choose(null_reports, SITE_ARMS)["selected_arm"] is None,
        "declared_maximum_forwards": declared_price(rows)["maximum_forwards_if_all_conditionals_open"],
        "future_audit_model_forwards": 0, "model_loaded": False,
        "r577_result_opened_by_dryrun": False, "final_or_ood_opened": False,
        "script_sha256": sha256(SCRIPT), "test_sha256": sha256(TEST) if TEST.is_file() else None,
        "preregistration_sha256": sha256(PREREG),
    }
    if not receipt["held_arm_fixture_passes"] or not receipt["control_null_fixture_has_no_eligible_site"]:
        raise RuntimeError("R583 planted fixture failed")
    DRYRUN.write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps(run_dryrun(), indent=2))
        return
    if OUT.exists():
        raise RuntimeError("R583 audit namespace already exists")
    for path in (R577_RESULT, R577_RUNLOG):
        if not path.is_file() or sha256(path) != HASHES[path]:
            raise RuntimeError(f"frozen execution artifact mismatch: {path}")
    rows, _ = load_authority()
    result = json.loads(R577_RESULT.read_text())
    audit = audit_payload(result, rows)
    provenance_failures = verify_run_provenance(result)
    audit["audit_failures"].extend(provenance_failures)
    if provenance_failures:
        audit["audit_verdict"] = "failed_independent_audit"
    audit.update({
        "schema": "numeric_sequence_factor_localization_audit_rung583_v1", "rung": 583,
        "source_result_sha256": HASHES[R577_RESULT], "source_runlog_sha256": HASHES[R577_RUNLOG],
        "completion_record": COMPLETION_RECORD,
        "completion_record_sha256": COMPLETION_RECORD_SHA256,
        "result_receipt_present": False,
        "provenance_limitation": (
            "No contemporaneous R577 result receipt exists; this audit binds result bytes, runlog, completion "
            "record, checkpoint, and current code authorities but cannot recreate atomic execution provenance."),
        "authority_sha256": {str(path): digest for path, digest in HASHES.items()},
        "model_loaded": False, "model_forwards": 0, "model_backwards": 0,
        "evaluated_splits": ["FIT"], "forbidden_splits_opened": [],
    })
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps({key: audit[key] for key in (
        "audit_verdict", "audit_failures", "independently_recomputed_scientific_decision",
        "recomputed_site_choice", "factor_stage_opened", "select_opened",
        "recomputed_observed_forwards", "bootstrap_cell_count", "result_receipt_present",
        "knowledge_packet", "model_forwards")}, indent=2))


if __name__ == "__main__":
    main()
