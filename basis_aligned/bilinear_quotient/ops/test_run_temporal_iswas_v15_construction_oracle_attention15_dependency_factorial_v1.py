import inspect

import torch

import run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1 as runner


def test_projectors_use_opposite_training_parity():
    artifact = {"projectors": {"A1": {"0": {"x": [[0.0]]}, "1": {"x": [[1.0]]}}}}
    bases = runner.bases_for_evaluation(torch, "cpu", artifact, "A1_oracle")
    assert float(bases[0]["x"][0, 0]) == 1.0
    assert float(bases[1]["x"][0, 0]) == 0.0


def test_exact_four_cells_and_complete15_switch_are_present():
    source = inspect.getsource(runner.main)
    assert '"00": execute_arm(backend, contexts, counters, {}, False)' in source
    assert '"01": execute_arm(backend, contexts, counters, {}, True)' in source
    helper = inspect.getsource(runner.build_factorials)
    assert set(runner.dependency.CELLS) == {"00", "01", "10", "11"}
    assert "decompose_dependency_factorial" in helper


def test_v16_is_constructed_without_c_and_all_choices_are_fixed():
    source = inspect.getsource(runner.main)
    assert 'row["transform_id"] in ("A1", "A2", "P")' in source
    assert "model_updates" in runner.PRICE_MAX and runner.PRICE_MAX["model_updates"] == 0
    assert runner.PRICE_MAX["fit_parameters"] == 0


def test_registered_predictions_are_emitted_separately():
    source = inspect.getsource(runner.main)
    for name in (
        "pred_b_each_oracle_transfers_own_v15_target_with_live_attention15",
        "pred_c_live_attention_retains_most_fixed_background_effect",
        "pred_d_attention15_dependency_interaction_is_small",
        "pred_e_live_attention_does_not_worsen_controls",
        "pred_f_v16_new_intervention_retains_fixed_effect",
    ):
        assert name in source
