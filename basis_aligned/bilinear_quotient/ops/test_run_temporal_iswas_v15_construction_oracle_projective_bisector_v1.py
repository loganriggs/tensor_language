import inspect

import torch

import run_temporal_iswas_v15_construction_oracle_projective_bisector_v1 as runner


def test_dryrun_freezes_four_oracle_fits_and_four_evaluation_arms():
    assert runner.TARGETS == ("A1", "A2")
    assert runner.PRICE_MAX["model_updates"] == 32
    source = inspect.getsource(runner.main)
    assert source.index("selection_finished_utc = utc_now()") < source.index("bank16 =")
    assert '"A1_oracle", "A2_oracle", "bisector", "failed_joint_parent"' in source


def test_evaluation_uses_opposite_training_parity():
    fits = {
        "A1": {0: {"best_bases": {"x": 0}}, 1: {"best_bases": {"x": 1}}},
    }
    assert runner.bases_for_evaluation(fits, "A1") == {0: {"x": 1}, 1: {"x": 0}}


def test_panel_initialization_is_panel_and_parity_scoped():
    source = inspect.getsource(runner.panel_initialization)
    assert 'panels=(panel,)' in source
    assert "parity=training_parity" in source
    assert "_dim_task_basis" in source


def test_all_registered_predictions_are_emitted_and_v16_c_is_excluded():
    source = inspect.getsource(runner.main)
    for name in (
        "pred_a_authority_alignment_parent_replay_gradient_geometry_closure_finiteness_and_price",
        "pred_b_each_construction_has_a_stable_effective_rank1_oracle",
        "pred_c_construction_axes_are_distinct",
        "pred_d_cross_use_is_construction_specific",
        "pred_e_analytic_bisector_recovers_a_fixed_invariant",
        "pred_f_bisector_beats_failed_joint_fit",
    ):
        assert name in source
    assert 'row["transform_id"] in ("A1", "A2", "P")' in source


def test_bisector_mapping_preserves_unit_columns():
    geometry = runner.projective_minimax_center(
        torch, torch.tensor([[1.0], [0.0]]), torch.tensor([[0.0], [1.0]]))
    assert geometry["center"].shape == (2, 1)
    assert torch.allclose(geometry["center"].T @ geometry["center"], torch.ones(1, 1))
