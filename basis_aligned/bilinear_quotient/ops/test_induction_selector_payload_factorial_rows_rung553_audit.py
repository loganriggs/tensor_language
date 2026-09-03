import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("induction_selector_payload_factorial_rows_rung553_audit.py")
TEXT = SCRIPT.read_text()


def test_audit_parses_and_is_model_free():
    ast.parse(TEXT)
    assert "torch" not in TEXT
    assert "load_bilin18" not in TEXT
    assert '"model_forwards": 0' in TEXT


def test_audit_checks_factorial_and_control_semantics():
    for term in (
        "verify_factorial_group", "two_valid_sources_selector_swap", "payload_swap_match_preserved",
        "selector_payload_joint_answer_preserved", "match_break_payload_preserved",
        "copy_relation_preserved_nuisance_change", "irrelevant_source_edit",
    ):
        assert term in TEXT
    assert "prompt_ids.count(structure[\"query_id\"]) != 2" in TEXT
    assert "payload_position != source_position + 1" in TEXT


def test_audit_checks_real_split_and_group_boundaries():
    assert "actual_variable_tokens[left] & actual_variable_tokens[right]" in TEXT
    assert "prompt sequence crosses semantic groups" in TEXT
    assert "exact prompt/answer pair repeats" in TEXT
    assert '"outcomes_opened": []' in TEXT
