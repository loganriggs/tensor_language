import pytest
import torch

import train_mlp2_trajectory_robust_r512_v1 as assay


def test_relative_shift_known_answer_and_identity():
    native = torch.tensor([[0.0, 0.0], [2.0, 0.0]])
    assert assay.relative_shift(native, native) == 0.0
    shifted = native + torch.tensor([1.0, 0.0])
    assert assay.relative_shift(native, shifted) == pytest.approx(1.0)


def test_relative_shift_rejects_mismatch():
    with pytest.raises(ValueError):
        assay.relative_shift(torch.ones(2, 2), torch.ones(2, 3))


def test_frozen_price_and_training_split():
    assert assay.refit.RANK == 512
    assert assay.FIT_DOCUMENTS + assay.DEV_DOCUMENTS == 192
    assert assay.FIT_DOCUMENTS * assay.TOKENS_PER_DOCUMENT == 30_720
    assert assay.STEPS == 1200
    assert assay.BATCH_PER_BACKGROUND * 2 == 1024
    assert assay.PROGRAM_FLOATS == 1_770_624


def test_capture_census_is_site_exact():
    native = assay.expected_capture_census(False)
    c512 = assay.expected_capture_census(True)
    assert native["outer_calls"] == native["outer_returns"] == 48
    assert set(native["attention_sites"].values()) == {48}
    assert set(native["native_mlp_sites"].values()) == {48}
    assert c512["native_mlp_sites"]["0"] == 0
    assert set(value for key, value in c512["native_mlp_sites"].items()
               if key != "0") == {48}
    assert c512["c512"] == c512["site2_capture"] == 48


def test_optimization_status_is_numeric_not_visual():
    fit = {"best_dev_nrmse": {"native": 0.3, "c512": 0.4}}
    decreasing = [{"worst_normalized_mse": value}
                  for value in (1.0, 0.98, 0.96, 0.94, 0.92)]
    assert assay.optimization_status(fit, decreasing) == "optimization_inconclusive"
    plateau = [{"worst_normalized_mse": value}
               for value in (1.0, 0.999, 0.998, 0.997, 0.996)]
    assert assay.optimization_status(fit, plateau) == "fit_complete"


def test_nested_equal_is_tensor_exact():
    value = {"x": [torch.tensor([1.0], dtype=torch.float32)], "y": 2}
    assert assay.nested_equal(value, {"x": [value["x"][0].clone()], "y": 2})
    assert not assay.nested_equal(value, {"x": [torch.tensor([2.0])], "y": 2})
    assert not assay.nested_equal(value, {"x": [value["x"][0].double()], "y": 2})
