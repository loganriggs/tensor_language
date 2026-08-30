import torch

import certify_native_mlp_polarization_depth_profile as profile


def test_spectrum_metrics_and_depth_rule_are_exact():
    singular = torch.linspace(20, 1, 1152, dtype=torch.float64)
    metrics = profile.analyze_singular_values(singular, 0.5)
    assert metrics["numerical_slice_rank_lower_bound"] == 1152
    assert metrics["sigma_513"] == singular[512]
    assert metrics["sigma_769"] == singular[768]
    sites = {str(site): {
        "best_rank768_relative_frobenius_error_lower_bound": 0.2,
    } for site in profile.SITES}
    assert profile.depth_summary(sites)["ruling"] == (
        "shipped_knee_not_explained_by_e0_coefficient_slice"
    )
    sites["10"]["best_rank768_relative_frobenius_error_lower_bound"] = 0.25
    assert profile.depth_summary(sites)["adjacent_1p20_knee"] is True
