import json

import analyze_task14_mlp6_7_background_subset_composition_transfer as run


def test_compile_plan_is_zero_gpu_and_hash_bound():
    plan = run.compile_plan()
    assert plan["candidate_id"] == run.CANDIDATE_ID
    assert plan["predicted_subsets_per_target_cell"] == 14
    assert plan["price"] == {"model_forwards": 0, "example_evaluations": 0,
                             "causal_interventions": 0, "backwards": 0,
                             "parameter_updates": 0}


def test_frozen_lattices_are_complete():
    matched = run._matched_q(json.loads(run.MATCHED_RESULT.read_text()))
    ood = run._ood_q(json.loads(run.OOD_RESULT.read_text()))
    assert len(matched) == 4 and len(ood) == 2
    assert all(set(q) == set(run.SUBSETS) for q in [*matched.values(), *ood.values()])


def test_endpoint_calibration_is_exact_and_uses_four_coefficients():
    q = {s: 2.0 + 4.0 * len(s) / 4.0 for s in run.SUBSETS}
    coefficients = {factor: .25 for factor in run.gate.BACKGROUND_FACTORS}
    score = run._evaluate("target", q, coefficients, "source")
    assert score["endpoint_maximum_absolute_error"] == 0.0
    assert score["normalized_mae"] == 0.0


def test_analysis_has_registered_groups_without_refitting():
    score = run.analyze()
    assert set(score["predictions"]) == {
        "pred_a_receipts_and_endpoints_close", "pred_b_matched_template_transfer",
        "pred_c_matched_to_ood_transfer", "pred_d_ood_to_matched_transfer",
        "pred_e_cardinality_unbiased", "pred_f_nontrivial_over_uniform"}
    assert [score[x]["cell_count"] for x in (
        "matched_template_transfer", "matched_to_ood_transfer", "ood_to_matched_transfer")] == [4, 2, 4]


def test_main_dry_run_does_not_write(monkeypatch, tmp_path):
    monkeypatch.setattr(run, "OUT", tmp_path / "forbidden.json")
    run.main(["--dry-run"])
    assert not run.OUT.exists()
