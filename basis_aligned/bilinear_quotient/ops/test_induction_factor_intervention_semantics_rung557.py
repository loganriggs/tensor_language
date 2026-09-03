import ast
from pathlib import Path


SCRIPT = Path(__file__).with_name("induction_factor_intervention_semantics_rung557.py")
TEXT = SCRIPT.read_text()


def test_script_parses_and_binds_frozen_inputs():
    ast.parse(TEXT)
    assert "6a0a6d2c8a3891ae5d6f787527b35e71c17518548b3b1836042afe730b13c460" in TEXT
    assert "7292716ea21401830ce4fd523da01d5e2923cc16ac6d8db48c0abf1dc1207042" in TEXT


def test_factor_computation_does_not_read_registered_answer():
    tree = ast.parse(TEXT)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    factor_text = ast.unparse(functions["equality_score"]) + ast.unparse(functions["payload_values"])
    assert "answer" not in factor_text
    assert "ids[position - 1] == query" in factor_text


def test_all_required_interventions_and_zero_model_calls_are_static():
    for name in (
        "pred_b_selector_score_transplant_exact", "pred_c_payload_value_transplant_exact",
        "pred_d_joint_transplant_exact", "pred_e_match_break_score_restore_exact",
        "pred_f_irrelevant_source_score_invariant",
    ):
        assert name in TEXT
    assert '"model_forwards": 0' in TEXT
    assert '"outcomes_opened": []' in TEXT
