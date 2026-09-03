#!/usr/bin/env python3
"""R571: independent CPU audit of R569/R570 saved sufficient statistics."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_two_hypothesis_rows_rung567.json"
RESULT = ROOT / "numeric_two_hypothesis_capability_rung569_570_results.json"
OUT = ROOT / "numeric_two_hypothesis_capability_rung571_audit.json"
LIST = "numbered_list_index_successor"
SEQUENCE = "numeric_sequence_continuation"
LIST_TARGETS = ("list_two_line_state_shift", "list_three_line_state_shift")
LIST_INVARIANCES = ("list_surface_preserved", "list_middle_index_break", "list_repeated_index_control")
SEQUENCE_TARGETS = ("sequence_digit_state_shift", "sequence_word_state_shift", "sequence_cross_format_shift")
SEQUENCE_INVARIANCES = ("sequence_digit_surface_preserved", "sequence_word_surface_preserved",
                        "sequence_digit_copy_control", "sequence_word_copy_control")
BOOTSTRAPS = 2000
SEED = 569


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(BOOTSTRAPS, len(data)))
    return float(np.quantile(data[indices].mean(1), .025))


def report(values: list[float], seed: int) -> dict:
    answer = {"n_groups": len(values), "correct_fraction": float(np.mean(np.asarray(values) > 0)),
              "mean_margin": float(np.mean(values)), "bootstrap95_lower_mean_margin": lower(values, seed)}
    answer["passed"] = bool(answer["correct_fraction"] >= .75 and answer["bootstrap95_lower_mean_margin"] > 0)
    return answer


def same(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    return all(abs(left[key] - right[key]) <= 1e-12 if isinstance(left[key], float) else left[key] == right[key]
               for key in left)


def audit_cells(split_result: dict, families: tuple[str, ...], section: str, seed: int) -> tuple[bool, int, int]:
    stats = split_result["row_statistics"]
    by_cell = {}
    for item in stats:
        by_cell.setdefault((item["family_id"], item["endpoint"]), []).append(item["margin"])
    passed, checks = True, 0
    for family in families:
        for endpoint in ("base", "donor"):
            rebuilt = report(by_cell[(family, endpoint)], seed)
            seed += 1
            assert same(rebuilt, split_result[section][family][endpoint])
            passed &= rebuilt["passed"]
            checks += 1
    return bool(passed), checks, seed


def main() -> None:
    rows = json.loads(ROWS.read_text())
    result = json.loads(RESULT.read_text())
    assert result["forbidden_splits_opened"] == [] and result["model_backwards"] == 0
    checks, saved_stats = 0, 0
    rebuilt_flags = {}
    for hypothesis in (LIST, SEQUENCE):
        for split in result["evaluated_splits"][hypothesis]:
            split_result = result["hypothesis_results"][hypothesis][split]
            saved_stats += len(split_result["row_statistics"])
            seed = (SEED + (0 if split == "FIT" else 100)) if hypothesis == LIST else (SEED + 200 + (0 if split == "FIT" else 100))
            target_families = LIST_TARGETS if hypothesis == LIST else SEQUENCE_TARGETS
            invariance_families = LIST_INVARIANCES if hypothesis == LIST else SEQUENCE_INVARIANCES
            target, count, seed = audit_cells(split_result, target_families, "state_shifts", seed)
            checks += count
            invariance, count, seed = audit_cells(split_result, invariance_families, "invariances", seed)
            checks += count
            if hypothesis == LIST:
                conflict = split_result["step_two_conflict"]
                conflict_pass = bool(conflict["correct_fraction"] >= .75 and conflict["bootstrap95_lower_mean_margin"] > 0)
                assert conflict_pass == conflict["passed"]
                third = conflict_pass
            else:
                stats = {(item["row_id"], item["endpoint"]): item["margin"] for item in split_result["row_statistics"]}
                middle = [item for item in rows["rows"] if item["hypothesis_id"] == SEQUENCE and item["split"] == split
                          and item["family_id"] == "sequence_middle_value_break"]
                base_values = [stats[(item["row_id"], "base")] for item in middle]
                drops = [stats[(item["row_id"], "base")] - stats[(item["row_id"], "donor")] for item in middle]
                base_report = report(base_values, seed)
                stored = split_result["middle_necessity"]
                assert same(base_report, stored["coherent_base"])
                assert abs(float(np.mean(drops)) - stored["mean_margin_drop"]) <= 1e-12
                assert abs(lower(drops, seed + 1) - stored["bootstrap95_lower_mean_drop"]) <= 1e-12
                third = bool(base_report["passed"] and float(np.mean(np.asarray(drops) > 0)) >= .65
                             and lower(drops, seed + 1) > 0)
                assert third == stored["passed"]
                checks += 1
            assert split_result["all_pass"] == bool(target and invariance and third)
            rebuilt_flags[(hypothesis, split)] = split_result["all_pass"]

    for hypothesis in (LIST, SEQUENCE):
        opened = result["evaluated_splits"][hypothesis]
        assert opened in (["FIT"], ["FIT", "SELECT"])
        assert (len(opened) == 2) == bool(rebuilt_flags[(hypothesis, "FIT")])
    assert result["model_forwards"] <= 30
    for path, expected in result["input_sha256"].items():
        assert sha256(Path(path)) == expected

    audit = {"rung": 571, "stage": "numeric_two_hypothesis_capability_audit",
             "result_sha256": sha256(RESULT), "rows_sha256": sha256(ROWS),
             "saved_endpoint_statistics_recomputed": saved_stats, "decision_cells_recomputed": checks,
             "input_hashes_match": True, "conditional_split_opening_match": True,
             "terminal_decisions_reproduced": True,
             "list_conflict_audit_limitation": "threshold logic only; R569 did not save row-level conflict margins",
             "model_loaded": False, "model_forwards": 0, "model_backwards": 0}
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
