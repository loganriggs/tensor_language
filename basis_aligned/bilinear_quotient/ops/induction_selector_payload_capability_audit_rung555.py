#!/usr/bin/env python3
"""CPU-only independent receipt/decision audit for R554."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path("/workspace/tensor_language/basis_aligned/bilinear_quotient")
POLY = ROOT.parent / "polynomial_causal"
RESULT = ROOT / "induction_selector_payload_capability_rung554_results.json"
ROWS = ROOT / "induction_selector_payload_factorial_rows_rung552.json"
RECEIPT = ROOT / "induction_selector_payload_factorial_rows_rung552_receipt.json"
AUDIT = ROOT / "induction_selector_payload_factorial_rows_rung553_audit.json"
R554_PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_RUNG554_PREREGISTRATION.md"
PREREG = POLY / "INDUCTION_SELECTOR_PAYLOAD_CAPABILITY_AUDIT_RUNG555_PREREGISTRATION.md"
OUT = ROOT / "induction_selector_payload_capability_rung555_audit.json"
EXPECTED_HASHES = {
    ROWS: "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460",
    RECEIPT: "0d42bcaaf7f86390803033ce13bc22d7690700130cd80df74170d6b2d652081a",
    AUDIT: "9fc0376fade6fb204686e164f293f8991caf7bc45c67eedd064f330dffd5d1ea",
    R554_PREREG: "9de3b16299043b6cf96e0cf2c75eb686f2063082e34a51e594fee1b0c0c4f777",
    PREREG: "f0fc6ddc6808439f8d5d620b7e44ae2265e4b3660ccd59d3d687014deaaba5a1",
}
SPLITS = ("FIT", "SELECT")
CONDITIONS = ("s0p0", "s0p1", "s1p0", "s1p1")
VARIANTS = {
    "irrelevant_source_edit": ("s0p0", "s0p1", "s1p0", "s1p1"),
    "copy_relation_preserved_nuisance_change": ("filler_change", "lag_extension"),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accuracy_pass(cell: dict) -> bool:
    return bool(cell["correct_fraction"] >= .75 and cell["bootstrap95_lower_mean_margin"] > 0)


def main() -> None:
    for path, expected in EXPECTED_HASHES.items():
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"frozen input mismatch: {path}")
    if os.environ.get("BQLIB_DRYRUN") == "1":
        print(json.dumps({"status": "dryrun_passed", "audited_rung": 554,
                          "result_required_only_at_execution": True}, indent=2))
        return
    if not RESULT.is_file():
        raise RuntimeError("R554 result is not present")
    result = json.loads(RESULT.read_text())
    assert result["checkpoint_weights_sha256"] == "680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3"
    assert result["model_forwards"] == 27 and result["model_backwards"] == 0
    assert result["model_weights_updated"] is False and result["unique_sequences"] == 864
    assert result["evaluated_splits"] == list(SPLITS) and result["forbidden_splits_opened"] == []
    assert result["input_sha256"] == {str(path): expected for path, expected in EXPECTED_HASHES.items()
                                        if path != PREREG}

    factorial = result["factorial_cells"]
    assert set(factorial) == set(SPLITS)
    pred_a = True
    for split in SPLITS:
        assert set(factorial[split]) == set(CONDITIONS)
        expected_n = 72 if split == "FIT" else 36
        for name in CONDITIONS:
            cell = factorial[split][name]
            assert cell["n_groups"] == expected_n
            passed = accuracy_pass(cell)
            assert cell["passed"] is passed
            pred_a &= passed

    controls = result["relation_preserving_controls"]
    assert set(controls) == set(SPLITS)
    pred_b = True
    for split in SPLITS:
        assert set(controls[split]) == set(VARIANTS)
        for family, variants in VARIANTS.items():
            assert set(controls[split][family]) == set(variants)
            for variant in variants:
                assert set(controls[split][family][variant]) == {"base", "donor"}
                for endpoint in ("base", "donor"):
                    cell = controls[split][family][variant][endpoint]
                    expected_n = (18 if split == "FIT" else 9) if family == "irrelevant_source_edit" \
                        else (72 if split == "FIT" else 36)
                    assert cell["n_groups"] == expected_n
                    passed = accuracy_pass(cell)
                    assert cell["passed"] is passed
                    pred_b &= passed

    necessity = result["selected_match_necessity"]
    assert set(necessity) == set(SPLITS)
    pred_c = True
    for split in SPLITS:
        cell = necessity[split]
        assert cell["n_groups"] == (72 if split == "FIT" else 36)
        passed = bool(
            cell["selected_match_break_positive_fraction"] >= .70
            and cell["bootstrap95_lower_mean_selected_match_margin_drop"] > 0
            and cell["bootstrap95_lower_mean_selective_gap"] > 0
        )
        assert cell["passed"] is passed
        pred_c &= passed

    exact_instrument = result["pred_0_exact_instrument"] is True
    assert result["pred_a_four_cell_capability"] is bool(pred_a)
    assert result["pred_b_relation_preserving_controls"] is bool(pred_b)
    assert result["pred_c_selected_match_necessity_and_selectivity"] is bool(pred_c)
    recomputed = bool(exact_instrument and pred_a and pred_b and pred_c)
    assert result["all_gates_pass"] is recomputed
    audit = {
        "rung": 555,
        "audited_rung": 554,
        "status": "terminal_receipt_audit_complete",
        "result_sha256": sha256(RESULT),
        "checkpoint_and_input_hashes_exact": True,
        "execution_budget_exact": True,
        "required_cell_coverage_exact": True,
        "all_pass_inequalities_independently_applied": True,
        "terminal_decision_recomputed": True,
        "pred_a_four_cell_capability": bool(pred_a),
        "pred_b_relation_preserving_controls": bool(pred_b),
        "pred_c_selected_match_necessity_and_selectivity": bool(pred_c),
        "all_gates_pass": recomputed,
        "model_forwards": 0,
        "model_backwards": 0,
        "forbidden_splits_opened": [],
        "limitation": "Uses saved group-bootstrap summaries; does not independently recompute bootstrap samples.",
    }
    OUT.write_text(json.dumps(audit, indent=1) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
