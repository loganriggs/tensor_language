#!/usr/bin/env python3
"""CPU-only independent terminal audit of the saved R546 row-level measurements."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "pending_opener_three_value_confirmation_rung546_results.json"
ROWS = ROOT / "pending_opener_three_value_fresh_rows_rung545.json"
RECEIPT = ROOT / "pending_opener_three_value_fresh_rows_rung545_receipt.json"
OUT = ROOT / "pending_opener_three_value_confirmation_rung548_audit.json"
SPLITS = ("FIT", "SELECT")
TARGETS = (
    "direct_three_value_type_substitution",
    "completed_then_reopened_three_value_order",
)
CONTROLS = (
    "pending_type_preserved_surface_rewrite",
    "pending_type_preserved_distance_extension",
    "pending_type_preserved_nonopener_punctuation",
)
EXPECTED_PER_PAIR = {"FIT": 12, "SELECT": 6}
EXPECTED_FORWARDS = 204
BOOTSTRAPS = 2000
SEED = 546
TOL = 1e-12


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lower_mean(values: list[float], seed: int, *, absolute: bool = False) -> float:
    """Reproduce the frozen percentile bootstrap without importing the R546 code."""
    array = np.asarray(values, dtype=np.float64)
    if absolute:
        array = np.abs(array)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(BOOTSTRAPS, array.size))
    return float(np.quantile(array[indices].mean(axis=1), 0.025))


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, rtol=0.0, atol=TOL):
        raise AssertionError(f"{label}: reported={actual!r}, recomputed={expected!r}")


def row_ids(rows: list[dict]) -> set[str]:
    ids = [row["row_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate row IDs in a saved measurement cell")
    return set(ids)


def audit_capability(result: dict, authority_rows: list[dict]) -> tuple[bool, dict]:
    seed = SEED
    all_pass = True
    weakest: dict[str, dict] = {}
    for split in SPLITS:
        for family in TARGETS + CONTROLS:
            expected_rows = [row for row in authority_rows if row["split"] == split and row["family_id"] == family]
            raw = result["raw_capability"][split][family]
            if row_ids(raw) != row_ids(expected_rows):
                raise AssertionError(f"capability row coverage differs for {split}/{family}")
            report = result["capability"][split][family]
            if family in TARGETS:
                pairs = sorted({(row["base_answer_id"], row["donor_answer_id"]) for row in raw})
                if len(pairs) != 6:
                    raise AssertionError(f"expected six ordered pairs for {split}/{family}")
                for pair in pairs:
                    cell = [row for row in raw if (row["base_answer_id"], row["donor_answer_id"]) == pair]
                    if len(cell) != EXPECTED_PER_PAIR[split]:
                        raise AssertionError(f"wrong n for {split}/{family}/{pair}: {len(cell)}")
                    key = f"{pair[0]}->{pair[1]}"
                    saved = report["ordered_pairs"][key]
                    base_fraction = float(np.mean([row["base_margin"] > 0 for row in cell]))
                    donor_fraction = float(np.mean([row["donor_margin"] > 0 for row in cell]))
                    close(saved["base_correct_fraction"], base_fraction, f"{split}/{family}/{key}/base")
                    close(saved["donor_correct_fraction"], donor_fraction, f"{split}/{family}/{key}/donor")
                    passed = bool(base_fraction >= 0.75 and donor_fraction >= 0.75)
                    if saved["passed"] is not passed:
                        raise AssertionError(f"pair verdict mismatch for {split}/{family}/{key}")
                    weakest[f"{split}/{family}/{key}"] = {
                        "base_correct_fraction": base_fraction,
                        "donor_correct_fraction": donor_fraction,
                    }
                margins = [row["base_margin"] for row in raw] + [row["donor_margin"] for row in raw]
                mean_margin = float(np.mean(margins))
                lower = lower_mean(margins, seed)
                seed += 1
                close(report["mean_symmetric_margin"], mean_margin, f"{split}/{family}/mean_margin")
                close(report["bootstrap95_lower_symmetric_margin"], lower, f"{split}/{family}/lower_margin")
                passed = bool(all(cell["passed"] for cell in report["ordered_pairs"].values()) and lower > 0)
            else:
                base_fraction = float(np.mean([row["base_margin"] > 0 for row in raw]))
                donor_fraction = float(np.mean([row["donor_margin"] > 0 for row in raw]))
                close(report["base_correct_fraction"], base_fraction, f"{split}/{family}/base")
                close(report["donor_correct_fraction"], donor_fraction, f"{split}/{family}/donor")
                passed = bool(base_fraction >= 0.75 and donor_fraction >= 0.75)
            if report["passed"] is not passed:
                raise AssertionError(f"capability verdict mismatch for {split}/{family}")
            all_pass &= passed
    return bool(all_pass), weakest


def audit_site(result: dict, authority_rows: list[dict]) -> tuple[bool, dict]:
    seed = SEED + 1000
    all_pass = True
    minima = {
        "target_bootstrap_lower_mean": float("inf"),
        "target_positive_fraction": float("inf"),
        "control_bootstrap_lower_abs_change": float("inf"),
        "control_full_vocabulary_logit_rms": float("inf"),
    }
    for split in SPLITS:
        for family in TARGETS + CONTROLS:
            expected_rows = [row for row in authority_rows if row["split"] == split and row["family_id"] == family]
            family_pass = True
            for direction in ("base_to_donor", "donor_to_base"):
                raw = result["raw_site_effects"][split][family][direction]
                if row_ids(raw) != row_ids(expected_rows):
                    raise AssertionError(f"site row coverage differs for {split}/{family}/{direction}")
                report = result["site_report"][split][family][direction]
                values = [row["endpoint_change"] for row in raw]
                if family in TARGETS:
                    mean = float(np.mean(values))
                    lower = lower_mean(values, seed)
                    positive = float(np.mean(np.asarray(values) > 0))
                    seed += 1
                    close(report["mean"], mean, f"{split}/{family}/{direction}/mean")
                    close(report["bootstrap95_lower_mean"], lower, f"{split}/{family}/{direction}/lower")
                    close(report["positive_fraction"], positive, f"{split}/{family}/{direction}/positive")
                    pair_pass = True
                    for pair, saved in report["ordered_pairs"].items():
                        cell = [row["endpoint_change"] for row in raw if row["ordered_pair"] == pair]
                        if len(cell) != EXPECTED_PER_PAIR[split]:
                            raise AssertionError(f"wrong target-site n for {split}/{family}/{direction}/{pair}")
                        cell_mean = float(np.mean(cell))
                        cell_positive = float(np.mean(np.asarray(cell) > 0))
                        close(saved["mean"], cell_mean, f"{split}/{family}/{direction}/{pair}/mean")
                        close(saved["positive_fraction"], cell_positive,
                              f"{split}/{family}/{direction}/{pair}/positive")
                        this_pair = bool(cell_mean > 0 and cell_positive >= 0.5)
                        if saved["passed"] is not this_pair:
                            raise AssertionError(f"target pair verdict mismatch: {split}/{family}/{direction}/{pair}")
                        pair_pass &= this_pair
                    passed = bool(lower > 0 and positive >= 0.70 and pair_pass)
                    if report["passed"] is not passed:
                        raise AssertionError(f"target verdict mismatch for {split}/{family}/{direction}")
                    minima["target_bootstrap_lower_mean"] = min(minima["target_bootstrap_lower_mean"], lower)
                    minima["target_positive_fraction"] = min(minima["target_positive_fraction"], positive)
                else:
                    lower_abs = lower_mean(values, seed, absolute=True)
                    logit_rms = float(np.mean([row["full_logit_rms"] for row in raw]))
                    seed += 1
                    close(report["bootstrap95_lower_mean_absolute"], lower_abs,
                          f"{split}/{family}/{direction}/lower_abs")
                    close(report["mean_full_vocabulary_logit_rms"], logit_rms,
                          f"{split}/{family}/{direction}/logit_rms")
                    passed = bool(lower_abs > 0.03 and logit_rms > 0.01)
                    if report["causally_live"] is not passed:
                        raise AssertionError(f"control verdict mismatch for {split}/{family}/{direction}")
                    minima["control_bootstrap_lower_abs_change"] = min(
                        minima["control_bootstrap_lower_abs_change"], lower_abs)
                    minima["control_full_vocabulary_logit_rms"] = min(
                        minima["control_full_vocabulary_logit_rms"], logit_rms)
                family_pass &= passed
            saved_family_pass = result["site_report"][split][family][
                "passed" if family in TARGETS else "causally_live"
            ]
            if saved_family_pass is not bool(family_pass):
                raise AssertionError(f"family site verdict mismatch for {split}/{family}")
            all_pass &= family_pass
    if result["site_report"]["passed"] is not bool(all_pass):
        raise AssertionError("overall site verdict mismatch")
    return bool(all_pass), minima


def main() -> None:
    result = json.loads(RESULT.read_text())
    row_document = json.loads(ROWS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    authority_rows = [row for row in row_document["rows"] if row["split"] in SPLITS]

    if len(authority_rows) != 540 or len(row_ids(authority_rows)) != 540:
        raise AssertionError("R545 FIT/SELECT authority is not exactly 540 unique rows")
    if result["model_forwards"] != EXPECTED_FORWARDS or result["model_backwards"] != 0:
        raise AssertionError("R546 model-call budget mismatch")
    if result["model_weights_updated"] is not False:
        raise AssertionError("R546 unexpectedly reports a model-weight update")
    if result["evaluated_splits"] != list(SPLITS) or result["forbidden_splits_opened"] != []:
        raise AssertionError("R546 split authority mismatch")
    if result["input_sha256"][str(ROWS)] != sha256(ROWS):
        raise AssertionError("R546 did not bind the evaluated R545 row file")
    if result["input_sha256"][str(RECEIPT)] != sha256(RECEIPT):
        raise AssertionError("R546 did not bind the R545 receipt")
    if receipt["model_loaded"] is not False or receipt["model_forwards"] != 0:
        raise AssertionError("R545 data freeze was not outcome-blind")

    capability_pass, pair_capability = audit_capability(result, authority_rows)
    site_pass, site_minima = audit_site(result, authority_rows)
    all_pass = bool(capability_pass and site_pass)
    if result["pred_a_exact_instrument"] is not True:
        raise AssertionError("R546 exact-instrument predicate failed")
    if result["pred_b_native_three_value_capability"] is not capability_pass:
        raise AssertionError("R546 overall capability verdict mismatch")
    if result["pred_c_l13h8_target_and_control_gate"] is not site_pass:
        raise AssertionError("R546 overall site verdict mismatch")
    if result["all_gates_pass"] is not all_pass:
        raise AssertionError("R546 combined verdict mismatch")

    audit = {
        "rung": 548,
        "audited_rung": 546,
        "status": "terminal_audit_complete",
        "result_sha256": sha256(RESULT),
        "rows_sha256": sha256(ROWS),
        "receipt_sha256": sha256(RECEIPT),
        "exact_budget_and_split_authority": True,
        "complete_row_identity_and_cell_counts": True,
        "independent_summary_recomputation_exact": True,
        "native_three_value_capability_held": capability_pass,
        "l13h8_complete_state_target_and_control_gate_held": site_pass,
        "all_gates_held": all_pass,
        "capability_by_ordered_pair": pair_capability,
        "site_gate_minima": site_minima,
        "decision": (
            "The fresh capability and complete-state confirmation held; a learned multi-output interchange may be "
            "preregistered, but FINAL_TEST/OOD remain unopened."
            if all_pass else
            "At least one fresh capability or complete-state cell failed; record the scientific null and do not fit "
            "a learned subspace or open FINAL_TEST/OOD."
        ),
        "final_test_or_ood_opened": False,
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
