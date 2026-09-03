import importlib.util
import os
from pathlib import Path


os.environ["BQLIB_NO_MODEL"] = "1"
SCRIPT = Path(__file__).with_name("pending_opener_capability_rung537.py")
spec = importlib.util.spec_from_file_location("pending_capability", SCRIPT)
capability = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(capability)


def test_validate_inputs_opens_only_fit_and_select_rows():
    main, controls = capability.validate_inputs()
    assert len(main) == capability.EXPECTED_MAIN_ROWS == 192
    assert len(controls) == capability.EXPECTED_CONTROL_ROWS == 64
    assert {row["split"] for row in main + controls} == {"FIT", "SELECT"}
    assert capability.EXPECTED_FORWARDS == 32


def test_interchange_gate_accepts_two_correct_endpoints():
    main, _ = capability.validate_inputs()
    rows = [
        row for row in main
        if row["split"] == "SELECT" and row["family_id"] == "opener_type_substitution"
    ]
    scores = {}
    for row in rows:
        scores[(row["row_id"], "base")] = {
            row["base_answer_id"]: 3.0,
            row["donor_answer_id"]: 0.0,
        }
        scores[(row["row_id"], "donor")] = {
            row["base_answer_id"]: 0.0,
            row["donor_answer_id"]: 3.0,
        }
    summary = capability.interchange_summary(
        main, scores, "SELECT", "opener_type_substitution", 0
    )
    assert summary["both_endpoints_correct_fraction"] == 1.0
    assert summary["mean_symmetric_logit_separation"] == 3.0
    assert summary["passed"] is True


def test_interchange_gate_rejects_one_sided_shortcut():
    main, _ = capability.validate_inputs()
    rows = [
        row for row in main
        if row["split"] == "SELECT" and row["family_id"] == "closed_then_reopened_type"
    ]
    scores = {}
    for row in rows:
        scores[(row["row_id"], "base")] = {
            row["base_answer_id"]: 3.0,
            row["donor_answer_id"]: 0.0,
        }
        scores[(row["row_id"], "donor")] = {
            row["base_answer_id"]: 3.0,
            row["donor_answer_id"]: 0.0,
        }
    summary = capability.interchange_summary(
        main, scores, "SELECT", "closed_then_reopened_type", 1
    )
    assert summary["both_endpoints_correct_fraction"] == 0.0
    assert summary["passed"] is False


def test_invariance_gate_requires_both_surface_variants():
    _, controls = capability.validate_inputs()
    rows = [row for row in controls if row["split"] == "SELECT"]
    scores = {}
    for index, row in enumerate(rows):
        good = index < 12
        base_margin = 2.0 if good else -2.0
        donor_margin = 2.0 if good else -2.0
        scores[(row["row_id"], "base")] = {8: base_margin, 1: 0.0, 60: -1.0, 92: -1.0}
        scores[(row["row_id"], "donor")] = {8: donor_margin, 1: 0.0, 60: -1.0, 92: -1.0}
    summary = capability.invariance_summary(
        controls, scores, "SELECT", "nonopener_punctuation_substitution"
    )
    assert summary["base_correct_fraction"] == 0.75
    assert summary["donor_correct_fraction"] == 0.75
    assert summary["passed"] is True
