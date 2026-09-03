#!/usr/bin/env python3
"""Independent CPU audit of R560 saved source-factor statistics."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
RESULT = ROOT / "pending_opener_source_factor_interchange_rung560_results.json"
OUT = ROOT / "pending_opener_source_factor_interchange_rung561_audit.json"
TARGETS = ("direct_three_value_type_substitution", "completed_then_reopened_three_value_order")
CONTROLS = (
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
)
DIRECTIONS = ("base_to_donor", "donor_to_base")
ARMS = ("score", "payload", "joint")
ARM_COST = {"score": 1, "payload": 1, "joint": 2}
BOOTSTRAPS, SEED = 2000, 560


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_lower(values: list[float], seed: int) -> float:
    array = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    choices = generator.integers(0, len(array), size=(BOOTSTRAPS, len(array)))
    return float(np.quantile(array[choices].mean(1), .025))


def close(left: float, right: float, tolerance: float = 1e-11) -> None:
    if not math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(f"summary mismatch: {left} != {right}")


def recompute(raw: dict, arms: tuple[str, ...], seed: int) -> dict:
    reports = {}
    for arm in arms:
        target_reports, control_reports, wrong_reports = {}, {}, {}
        passed = True
        for family in TARGETS:
            target_reports[family], wrong_reports[family] = {}, {}
            for direction in DIRECTIONS:
                values = [cell["recovery"] for cell in raw[family][direction][arm]["semantic"]]
                assert all(value is not None for value in values)
                report = {"n": len(values), "mean": float(np.mean(values)),
                          "median": float(np.median(values)),
                          "bootstrap95_lower_mean": bootstrap_lower(values, seed),
                          "positive_fraction": float(np.mean(np.asarray(values) > 0)), "values": values}
                seed += 1
                report["passed"] = bool(report["median"] >= .50 and report["bootstrap95_lower_mean"] > 0
                                        and report["positive_fraction"] >= .75)
                wrong = [cell["recovery"] for cell in raw[family][direction][arm]["wrong"]]
                wrong_report = {"n": len(wrong), "mean_absolute_recovery": float(np.mean(np.abs(wrong)))}
                wrong_report["passed"] = wrong_report["mean_absolute_recovery"] <= .25
                target_reports[family][direction] = report
                wrong_reports[family][direction] = wrong_report
                passed &= report["passed"] and wrong_report["passed"]
        for family in CONTROLS:
            control_reports[family] = {}
            for direction in DIRECTIONS:
                cells = raw[family][direction][arm]["semantic"]
                endpoint = np.asarray([cell["closer_margin_change"] for cell in cells])
                rms = np.asarray([cell["full_vocabulary_logit_rms"] for cell in cells])
                full_endpoint = np.asarray([cell["complete_head_endpoint_change"] for cell in cells])
                full_rms = np.asarray([cell["complete_head_full_vocabulary_rms"] for cell in cells])
                mean_abs = float(np.mean(np.abs(endpoint)))
                report = {"n": len(cells), "mean_absolute_closer_margin_change": mean_abs,
                          "fraction_of_complete_head_margin_change": mean_abs / float(np.mean(np.abs(full_endpoint))),
                          "mean_full_vocabulary_logit_rms": float(np.mean(rms)),
                          "fraction_of_complete_head_full_vocabulary_rms": float(np.mean(rms)) / float(np.mean(full_rms))}
                report["passed"] = bool(mean_abs <= .10
                                        and report["fraction_of_complete_head_margin_change"] <= .25
                                        and report["fraction_of_complete_head_full_vocabulary_rms"] <= .25)
                control_reports[family][direction] = report
                passed &= report["passed"]
        reports[arm] = {"targets": target_reports, "controls": control_reports,
                        "wrong_source_controls": wrong_reports, "passed": bool(passed)}
    return reports


def compare_reports(expected: dict, observed: dict) -> None:
    if expected.keys() != observed.keys():
        raise RuntimeError("report keys differ")
    for key, value in expected.items():
        other = observed[key]
        if isinstance(value, dict):
            compare_reports(value, other)
        elif isinstance(value, list):
            if len(value) != len(other):
                raise RuntimeError("report list lengths differ")
            for left, right in zip(value, other, strict=True):
                if isinstance(left, float):
                    close(left, right)
                elif left != right:
                    raise RuntimeError("report list values differ")
        elif isinstance(value, float):
            close(value, other)
        elif value != other:
            raise RuntimeError(f"report value differs at {key}: {value} != {other}")


def choose(reports: dict) -> dict:
    passing = [arm for arm, report in reports.items() if report["passed"]]
    def worst_lower(arm: str) -> float:
        return min(cell["bootstrap95_lower_mean"]
                   for family in reports[arm]["targets"].values() for cell in family.values())
    passing.sort(key=lambda arm: (ARM_COST[arm], -worst_lower(arm), arm))
    return {"eligible_arms": passing, "selected_arm": passing[0] if passing else None,
            "selected_cost": ARM_COST[passing[0]] if passing else None}


def interactions(raw: dict) -> dict:
    output = {}
    for family in TARGETS:
        output[family] = {}
        for direction in DIRECTIONS:
            by_arm = {arm: {cell["row_id"]: cell["endpoint_change"]
                            for cell in raw[family][direction][arm]["semantic"]} for arm in ARMS}
            row_ids = sorted(by_arm["joint"])
            values = [by_arm["joint"][row] - by_arm["score"][row] - by_arm["payload"][row]
                      for row in row_ids]
            output[family][direction] = {"n": len(values), "mean_interaction_logit": float(np.mean(values)),
                                         "mean_absolute_interaction_logit": float(np.mean(np.abs(values))),
                                         "values": values}
    return output


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "audited_rung": 560,
                          "result_required_only_at_execution": True, "model_forwards": 0}, indent=2))
        return
    if not RESULT.is_file():
        raise RuntimeError("R560 result is not present")
    result = json.loads(RESULT.read_text())
    fit = recompute(result["fit_raw"], ARMS, SEED)
    compare_reports(fit, result["fit_reports"])
    choice = choose(fit)
    if choice != result["fit_choice"]:
        raise RuntimeError("FIT choice differs")
    computed_interactions = interactions(result["fit_raw"])
    compare_reports(computed_interactions, result["fit_score_payload_interactions"])
    select = None
    held = False
    if choice["selected_arm"] is not None:
        selected = choice["selected_arm"]
        select = recompute(result["select_raw"], (selected,), SEED + 100)
        compare_reports(select, result["select_reports"])
        held = select[selected]["passed"]
        assert result["evaluated_splits"] == ["FIT", "SELECT"]
    else:
        assert result["select_raw"] is None and result["select_reports"] is None
        assert result["evaluated_splits"] == ["FIT"]
    execution = result["execution"]
    exact = bool(
        result["checkpoint_weights_sha256"] == "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
        and execution["fit"]["native_replay_relative_squared_error"] <= 1e-12
        and execution["fit"]["max_source_factor_relative_squared_reconstruction_error"] <= 1e-10
        and (execution["select"] is None or (
            execution["select"]["native_replay_relative_squared_error"] <= 1e-12
            and execution["select"]["max_source_factor_relative_squared_reconstruction_error"] <= 1e-10
        ))
    )
    assert result["pred_a_exact_instrument"] is exact
    assert result["pred_b_fit_selective_source_factor_exists"] is (choice["selected_arm"] is not None)
    assert result["pred_c_selected_source_factor_holds"] is bool(exact and held)
    assert result["selected_factor_held"] is bool(exact and held)
    assert result["model_forwards"] == execution["fit"]["model_forwards"] + (
        execution["select"]["model_forwards"] if execution["select"] else 0
    )
    assert result["model_backwards"] == 0 and result["model_weights_updated"] is False
    assert result["forbidden_splits_opened"] == []
    decision = "held_source_factor" if exact and held else "source_factor_null"
    assert result["decision"] == decision
    audit = {"rung": 561, "audited_rung": 560, "status": "terminal_CPU_audit_complete",
             "result_sha256": sha256(RESULT), "fit_cells_recomputed": 42,
             "select_cells_recomputed": 14 if select else 0,
             "interaction_rows_recomputed": sum(cell["n"] for family in computed_interactions.values()
                                                for cell in family.values()),
             "fit_choice": choice, "selected_factor_held": bool(exact and held),
             "decision": decision, "execution_and_split_checks_exact": True,
             "model_forwards": 0, "model_backwards": 0}
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
