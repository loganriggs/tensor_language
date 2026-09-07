import json
from pathlib import Path

import torch

import run_temporal_iswas_v15_head_response_target_feasible_regularized_das_v1 as runner


def test_static_authorities_and_registered_terminal_match():
    paths = {
        "prior": runner.PRIOR, "dual_result": runner.DUAL_RESULT,
        "dual_runner": runner.DUAL_RUNNER, "builder": runner.BUILDER,
        "clamp_contract": runner.CLAMP_CONTRACT,
        "projector_contract": runner.PROJECTOR_CONTRACT,
        "factor_runner": runner.FACTOR_RUNNER, "greedy_runner": runner.GREEDY_RUNNER,
    }
    assert {key: runner.sha256(path) for key, path in paths.items()} == runner.EXPECTED
    assert json.loads(runner.DUAL_RESULT.read_text())["terminal"] == "no_selective_dual_greedy_program"


def test_configuration_grid_is_nondeduplicated_and_fixed_price_is_sufficient():
    grid = runner.configuration_grid()
    assert len(grid) == len({item["name"] for item in grid}) == 30
    assert all((item["noise_sigma"] == 0.0) == (item["jacobian_weight"] == 0.0)
               for item in grid)
    assert runner.STABILITY_WEIGHT == 0.25
    assert runner.PRICE_MAX == {
        "native_capture_forwards": 10,
        "differentiable_transformer_forwards": 1800,
        "transformer_backward_forwards": 1250,
        "model_updates": 480,
        "example_evaluations": 30000,
        "fit_parameters": 2048,
    }


def test_fit_and_sealed_rows_are_balanced_and_disjoint():
    rows = runner.fresh.build_rows()
    fit = [row for row in rows if row["transform_id"] in ("A1", "P")]
    sealed = [row for row in rows if row["transform_id"] in ("A2", "C")]
    assert len(fit) == len(sealed) == 32
    assert [len(runner.row_indices(fit, parity=parity)) for parity in (0, 1)] == [16, 16]
    assert {row["row_id"] for row in fit}.isdisjoint(row["row_id"] for row in sealed)


def test_basis_builders_are_orthonormal_and_dim_led():
    generator = torch.Generator().manual_seed(7)
    matrix = torch.randn(40, 128, generator=generator)
    dim, _ = runner._dim_task_basis(torch, matrix, 4)
    factor, _ = runner._top_right_basis(torch, matrix, 4)
    assert torch.allclose(dim.T @ dim, torch.eye(4), atol=1e-5)
    assert torch.allclose(factor.T @ factor, torch.eye(4), atol=1e-5)
    assert torch.allclose(dim[:, 0].abs(), (matrix.mean(0) / matrix.mean(0).norm()).abs(), atol=1e-5)


def test_registered_report_only_drops_unregistered_control_mean():
    value = {
        "targets": {"A1": {"x": 1.0}},
        "controls": {"P": {"mean_kl": 0.1, "median_kl": 0.2, "flipped_row_ids": []}},
    }
    assert runner.registered_report(value) == {
        "targets": {"A1": {"x": 1.0}},
        "controls": {"P": {"median_kl": 0.2, "flipped_row_ids": []}},
    }


def test_sealed_bank_is_constructed_only_after_selection_is_frozen():
    source = Path(runner.__file__).read_text()
    assert source.index("selection_finished_utc = utc_now()") < source.index("sealed_rows =")
    before = source[:source.index("selection_finished_utc = utc_now()")]
    assert "sealed_bank = capture_bank" not in before
