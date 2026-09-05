import importlib
import json


m = importlib.import_module("run_task14_direction_mediator_gain_fourth_corpus_causal_validation")


def test_plan_is_fixed_and_bounded():
    plan = m.compile_plan()
    assert plan["fit_operations"] == 0
    assert plan["gain_vector_reader_changes"] == 0
    assert plan["price"]["physical_model_forwards"] == 17
    assert plan["price"]["example_evaluations"] == 4192


def test_exact_sealed_effects_pass_scoring():
    sealed = json.loads(m.PREDICTIONS.read_text())["evidence"]
    evidence = []
    for item in sealed:
        q = item["sealed_reader_q"]
        q15 = q - item["sealed_m15"]
        q17 = q - item["sealed_m17"]
        qboth = q - item["sealed_joint_mediation"]
        evidence.append({
            "row_id": item["row_id"], "background": item["background"],
            "direction": item["direction"], "template": item["template"],
            "cardinality": item["cardinality"],
            "base_empty": 0.0, "base_15": 0.0, "base_17": 0.0, "base_both": 0.0,
            "program_empty": q, "program_15": q15, "program_17": q17,
            "program_both": qboth, "q_empty": q, "q_15": q15, "q_17": q17,
            "q_both": qboth, "m15": item["sealed_m15"], "m17": item["sealed_m17"],
            "interaction": item["sealed_interaction"], "m_both": item["sealed_joint_mediation"],
        })
    exactness = {"role_state_closure_max_absolute_error": 0.0, "role_normalized_closure_max_absolute_error": 0.0, "downstream_state_closure_max_absolute_error": 0.0, "downstream_normalized_closure_max_absolute_error": 0.0}
    scored = m.score(evidence, exactness)
    assert scored["terminal"] == "prospective_program_screen"
    assert all(scored["predictions"].values())
