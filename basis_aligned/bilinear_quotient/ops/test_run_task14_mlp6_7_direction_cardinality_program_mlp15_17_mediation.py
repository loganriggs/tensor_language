import importlib
import json

import pytest


m = importlib.import_module("run_task14_mlp6_7_direction_cardinality_program_mlp15_17_mediation")


def test_plan_is_target_free_and_bounded():
    plan = m.compile_plan()
    assert plan["candidate_id"] == m.CANDIDATE_ID
    assert plan["fit_operations"] == 0
    assert plan["program_changes"] == 0
    assert plan["price"] == {
        "physical_model_forwards": 9,
        "example_evaluations": 2144,
        "causal_installations": 1536,
        "mediator_clamps": 2048,
        "backwards": 0,
        "parameter_updates": 0,
        "maximum_forward_batch": 256,
    }


def test_exact_half_mediation_scores_as_mediation_not_direct():
    parent = json.loads(m.PROGRAM_RESULT.read_text())["causal_evidence"]
    evidence = []
    for item in parent:
        q = float(item["cardinality_prototype_q"])
        evidence.append({
            "row_id": item["row_id"],
            "background": item["background"],
            "direction": item["direction"],
            "template": item["template"],
            "cardinality": item["cardinality"],
            "base_native": 0.0,
            "program_native": q,
            "base_replayed": 0.0,
            "program_clamped": 0.5 * q,
            "full_program_q": q,
            "clamped_program_q": 0.5 * q,
            "mediated_q": 0.5 * q,
        })
    exactness = {
        "role_state_closure_max_absolute_error": 0.0,
        "role_normalized_closure_max_absolute_error": 0.0,
        "downstream_state_closure_max_absolute_error": 0.0,
        "downstream_normalized_closure_max_absolute_error": 0.0,
    }
    scored = m.score(evidence, exactness)
    assert scored["terminal"] == "mediation_screen"
    assert scored["predictions"][m.PRED_KEYS[0]] is True
    assert scored["predictions"][m.PRED_KEYS[1]] is True
    assert scored["predictions"][m.PRED_KEYS[2]] is True
    assert scored["predictions"][m.PRED_KEYS[3]] is False


def test_stats_distinguish_direct_and_mediated_vectors():
    same = m._stats([1.0, -2.0], [1.0, -2.0])
    half = m._stats([1.0, -2.0], [0.5, -1.0])
    assert same["cosine"] == pytest.approx(1.0)
    assert same["relative_l2_error"] == 0.0
    assert half["predicted_to_actual_norm_ratio"] == 0.5
