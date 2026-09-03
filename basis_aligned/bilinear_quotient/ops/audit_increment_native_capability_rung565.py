#!/usr/bin/env python3
"""Independently audit R564's saved row-level sufficient statistics; CPU only."""

# BQLANE: cpu

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ROWS = ROOT / "increment_counterfactual_authority_rung563.json"
RESULT = ROOT / "increment_native_capability_rung564_results.json"
OUT = ROOT / "increment_native_capability_rung565_audit.json"
TARGETS = ("digit_coherent_shift", "word_coherent_shift", "cross_format_coherent_shift")
CONTROLS = ("operation_preserved_surface_edit", "repeated_number_numeric_control", "step_two_numeric_control")
BOOTSTRAPS = 2000
SEED = 564


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower(values: list[float], seed: int) -> float:
    data = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(data), size=(BOOTSTRAPS, len(data)))
    return float(np.quantile(data[indices].mean(axis=1), 0.025))


def report(values: list[float], seed: int) -> dict:
    answer = {
        "n_groups": len(values),
        "correct_fraction": float(np.mean(np.asarray(values) > 0)),
        "mean_numeric_candidate_margin": float(np.mean(values)),
        "bootstrap95_lower_mean_margin": lower(values, seed),
    }
    answer["passed"] = bool(answer["correct_fraction"] >= .75 and answer["bootstrap95_lower_mean_margin"] > 0)
    return answer


def close(left: dict, right: dict) -> bool:
    if left.keys() != right.keys():
        return False
    for key in left:
        if isinstance(left[key], float):
            if abs(left[key] - right[key]) > 1e-12:
                return False
        elif left[key] != right[key]:
            return False
    return True


def main() -> None:
    rows_doc = json.loads(ROWS.read_text())
    result = json.loads(RESULT.read_text())
    assert result["evaluated_splits"] == ["FIT"]
    assert result["forbidden_splits_opened"] == []
    assert result["model_forwards"] == 20 and result["model_backwards"] == 0
    fit = result["split_results"]["FIT"]
    stats = fit["row_statistics"]
    assert len(stats) == 896
    values = {(item["row_id"], item["endpoint"]): item["numeric_candidate_margin"] for item in stats}
    fit_rows = [row for row in rows_doc["rows"] if row["split"] == "FIT"]
    assert len(fit_rows) == 448 and len(values) == 896

    seed, checks, target_pass = SEED, 0, True
    for family in TARGETS:
        family_rows = [row for row in fit_rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            rebuilt = report([values[(row["row_id"], endpoint)] for row in family_rows], seed)
            seed += 1
            assert close(rebuilt, fit["target_cells"][family][endpoint])
            target_pass &= rebuilt["passed"]
            checks += 1

    control_pass = True
    for family in CONTROLS:
        family_rows = [row for row in fit_rows if row["family_id"] == family]
        for endpoint in ("base", "donor"):
            rebuilt = report([values[(row["row_id"], endpoint)] for row in family_rows], seed)
            seed += 1
            assert close(rebuilt, fit["control_cells"][family][endpoint])
            control_pass &= rebuilt["passed"]
            checks += 1

    need_rows = [row for row in fit_rows if row["family_id"] == "incoherent_middle_number_edit"]
    base = [values[(row["row_id"], "base")] for row in need_rows]
    drops = [values[(row["row_id"], "base")] - values[(row["row_id"], "donor")] for row in need_rows]
    base_report = report(base, seed)
    stored = fit["middle_number_necessity_details"]
    assert close(base_report, stored["coherent_base"])
    necessity = bool(
        base_report["passed"] and float(np.mean(np.asarray(drops) > 0)) >= .65 and lower(drops, seed + 1) > 0
    )
    assert abs(float(np.mean(drops)) - stored["mean_margin_drop_after_middle_edit"]) <= 1e-12
    assert abs(lower(drops, seed + 1) - stored["bootstrap95_lower_mean_drop"]) <= 1e-12
    assert necessity == stored["passed"]

    terminal = bool(result["pred_0_exact_instrument"] and target_pass and control_pass and necessity)
    assert terminal is False and result["all_gates_pass"] is False
    assert result["pred_a_target_capability"] == target_pass
    assert result["pred_b_nonincrement_rule_controls"] == control_pass
    assert result["pred_c_middle_number_necessity"] == necessity
    for path_string, digest in result["input_sha256"].items():
        assert sha256(Path(path_string)) == digest

    audit = {
        "rung": 565,
        "stage": "increment_native_capability_post_result_audit",
        "result_sha256": sha256(RESULT),
        "rows_sha256": sha256(ROWS),
        "row_statistics_recomputed": len(stats),
        "endpoint_cells_recomputed": checks,
        "necessity_pairs_recomputed": len(drops),
        "input_hashes_match": True,
        "call_and_split_ledger_match": True,
        "terminal_null_reproduced": True,
        "model_loaded": False,
        "model_forwards": 0,
        "model_backwards": 0,
    }
    OUT.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
