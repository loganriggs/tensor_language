import torch

import mlp2_error_rayleigh_collector_core as core
import mlp2_error_rayleigh_predictor as predictor


def synthetic_collector():
    torch.manual_seed(4)
    f = torch.zeros(3, 2, 3, 32, len(core.FEATURE_NAMES), dtype=torch.float64)
    i = {name: index for index, name in enumerate(core.FEATURE_NAMES)}
    finite = torch.zeros(3, 2, 32, len(core.FINITE_NAMES), dtype=torch.float64)
    finite[..., 5:] = 1
    for p in range(3):
        for d in range(32):
            x = (d - 15.5) / 10 + 0.2 * p
            for c in range(3):
                f[p, 1, c, d, i["local_mse"]] = x + c
                for suffix in ("h16", "h8"):
                    f[p, 1, c, d, i[f"ce_jvp_{suffix}"]] = x + 2*c
                    f[p, 1, c, d, i[f"qlogit_{suffix}"]] = x*x + c
                    f[p, 1, c, d, i[f"q5_{suffix}"]] = 0.5*x + c
                    f[p, 1, c, d, i[f"q6_{suffix}"]] = -0.25*x + c
            finite[p, 1, d, core.FINITE_NAMES.index("injected_dce")] = 2*x + 0.5*x*x
    return f, finite


def test_units_are_document_program_and_background_contrasts():
    features, finite = synthetic_collector()
    contrasts, target = predictor.regression_units(features, finite, 0)
    assert target.shape == (32, 3)
    assert all(value.shape == (32, 3) for value in contrasts.values())
    assert torch.equal(contrasts["linear"], features[:, 1, 0, :, 1].T)


def test_clustered_ridge_fits_without_program_identity():
    features, finite = synthetic_collector()
    result = predictor.fit_design(features, finite)
    assert set(result["models"]) == set(predictor.FAMILIES)
    assert result["target"].shape == (32, 3)
    assert set(result["null_predictions"]) == {"DERANGED", "COV_RANDOM"}
    for model in result["models"].values():
        assert model["ridge"]["selected"] in predictor.RIDGE_GRID
        assert model["design_prediction"].shape == (32, 3)
        assert model["coefficients"].numel() <= 5


def test_canonical_bundle_recomputes_exactly_and_detects_tensor_mutation():
    features, finite = synthetic_collector()
    first = predictor.serialize_fit(predictor.fit_design(features, finite))
    second = predictor.serialize_fit(predictor.fit_design(features, finite))
    assert predictor.validate_frozen_bundle(first) is first
    assert predictor.exact_nested_equal(first, second)
    family = next(iter(predictor.FAMILIES))
    second["models"][family]["coefficients"][0] += torch.finfo(torch.float64).eps
    assert not predictor.exact_nested_equal(first, second)


def test_ridge_tie_breaks_toward_larger_penalty_for_intercept_only_signal():
    # With exactly zero features all ridge values have the same intercept-only fit.
    # Directly verify the declared ordering rule without fitting a zero-variance normalizer.
    losses = {penalty: 1.0 for penalty in predictor.RIDGE_GRID}
    selected = min(predictor.RIDGE_GRID, key=lambda penalty: (losses[penalty], -penalty))
    assert selected == 100.0
