import inspect

import pytest
import torch

import run_temporal_iswas_construction_oracle_weight_convergence_v1 as runner


def test_absolute_cosine_is_sign_gauge_invariant():
    left, right = torch.tensor([1.0, 2.0]), torch.tensor([2.0, -1.0])
    assert runner.absolute_cosine(torch, left, right) == pytest.approx(
        runner.absolute_cosine(torch, -left, right))


def test_reader_joint_score_requires_alignment_and_both_strengths():
    identity = torch.eye(2)
    row = runner.reader_row(torch, "x", identity, torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]))
    assert row["response_alignment"] == 0.0
    assert row["joint_score"] == 0.0
    aligned = runner.reader_row(torch, "x", identity, torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0]))
    assert aligned["response_alignment"] == pytest.approx(1.0)
    assert aligned["joint_score"] > 0.0


def test_plan_is_zero_forward_and_uses_exact_construction_axes():
    assert runner.PRICE["model_forwards"] == 0
    assert runner.PRICE["checkpoint_loads"] == 1
    source = inspect.getsource(runner.main)
    assert 'oracles["projectors"][panel][fold][site]' in source
    assert 'oracles["between_construction_geometry"][fold][site]' in source
    assert "fastload.load_model_fast().eval()" in source


def test_predictions_distinguish_write_reader_and_pullback_convergence():
    source = inspect.getsource(runner.main)
    for name in (
        "pred_c_construction_axes_converge_as_residual_writes",
        "pred_d_l15h5_reads_a_shared_construction_response",
        "pred_e_sources_converge_on_one_shared_reader_interface",
        "pred_f_value_pullbacks_share_an_upstream_covector",
    ):
        assert name in source
