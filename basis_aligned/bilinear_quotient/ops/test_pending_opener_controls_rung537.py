import hashlib
import importlib.util
import json
from pathlib import Path


BQ = Path(__file__).resolve().parents[1]
BUILDER = Path(__file__).with_name("pending_opener_controls_rung537.py")
SOURCE = BQ / "pending_opener_multifamily_rows_rung537.json"
CONTROLS = BQ / "pending_opener_controls_rung537.json"
RECEIPT = BQ / "pending_opener_controls_rung537_receipt.json"


def load_builder():
    spec = importlib.util.spec_from_file_location("pending_controls", BUILDER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_control_receipt_binds_source_and_has_no_outcomes():
    controls = json.loads(CONTROLS.read_text())
    receipt = json.loads(RECEIPT.read_text())
    assert hashlib.sha256(CONTROLS.read_bytes()).hexdigest() == receipt["controls_sha256"]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == receipt["source_rows_sha256"]
    assert controls["source_rows_sha256"] == receipt["source_rows_sha256"]
    assert controls["status"] == "controls_frozen_outcomes_unopened"
    assert receipt["outcomes_opened"] is False
    assert controls["model_forwards"] == controls["model_backwards"] == 0


def test_controls_are_single_token_preopener_edits_that_preserve_state_and_answer():
    rows = json.loads(CONTROLS.read_text())["rows"]
    assert len(rows) == 96
    assert len({row["group_id"] for row in rows}) == 96
    for row in rows:
        assert row["role"] == "invariance"
        assert row["answer"] == ")"
        assert row["proposed_variable_base"] == row["proposed_variable_donor"] == "pending_paren"
        assert all(row["construction_checks"].values())
        assert row["wrong_closer_tokens"] == ["]", "}"]


def test_controls_have_exact_shared_split_counts():
    rows = json.loads(CONTROLS.read_text())["rows"]
    counts = {split: sum(row["split"] == split for row in rows) for split in ("FIT", "SELECT", "FINAL_TEST", "OOD")}
    assert counts == {"FIT": 48, "SELECT": 16, "FINAL_TEST": 16, "OOD": 16}


def test_builder_is_byte_deterministic():
    before = (CONTROLS.read_bytes(), RECEIPT.read_bytes())
    load_builder().main()
    assert before == (CONTROLS.read_bytes(), RECEIPT.read_bytes())
