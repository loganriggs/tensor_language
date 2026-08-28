import pytest

from joint_early_mlp_pca_composition import analyze_composition


def _analysis(upstream, joint, conditional):
    return {
        "gain_by_arm": {
            "mlp0+mlp1": upstream,
            "mlp0+mlp1+mlp2": joint,
        },
        "mlp2_conditional_marginal_after_mlp0_mlp1": conditional,
    }


def test_complete_composition_gate_uses_same_split_exact_denominators():
    exact = {
        "discovery": _analysis(0.40, 0.50, 0.10),
        "heldout": _analysis(0.50, 0.60, 0.10),
    }
    projected = {
        "discovery": _analysis(0.20, 0.25, 0.05),
        "heldout": _analysis(0.25, 0.30, 0.05),
    }
    result = analyze_composition(
        exact, projected,
        upstream_heldout_ci95=[0.15, 0.35],
        joint_heldout_ci95=[0.20, 0.40],
    )
    assert result["splits"]["discovery"][
        "projected_upstream_fraction_of_exact"
    ] == pytest.approx(0.5)
    assert result["splits"]["heldout"][
        "projected_joint_fraction_of_exact"
    ] == pytest.approx(0.5)
    assert result["decisions"]["complete_composition_gate_passes"] is True
    assert all(result["registered_predictions"].values())


def test_nonpositive_conditional_mlp2_fails_even_when_joint_fraction_passes():
    exact = {
        "discovery": _analysis(0.40, 0.50, 0.10),
        "heldout": _analysis(0.40, 0.50, 0.10),
    }
    projected = {
        "discovery": _analysis(0.30, 0.29, -0.01),
        "heldout": _analysis(0.30, 0.29, -0.01),
    }
    result = analyze_composition(
        exact, projected,
        upstream_heldout_ci95=[0.20, 0.40],
        joint_heldout_ci95=[0.20, 0.38],
    )
    assert result["decisions"][
        "exact_mlp2_conditional_after_projected_upstream_positive_both_splits"
    ] is False
    assert result["decisions"]["complete_composition_gate_passes"] is False


def test_composition_rejects_invalid_denominators_and_intervals():
    exact = {
        "discovery": _analysis(0.0, 0.50, 0.10),
        "heldout": _analysis(0.40, 0.50, 0.10),
    }
    projected = {
        "discovery": _analysis(0.30, 0.40, 0.10),
        "heldout": _analysis(0.30, 0.40, 0.10),
    }
    with pytest.raises(ValueError, match="denominators"):
        analyze_composition(
            exact, projected,
            upstream_heldout_ci95=[0.2, 0.4], joint_heldout_ci95=[0.2, 0.4],
        )
    exact["discovery"] = _analysis(0.40, 0.50, 0.10)
    with pytest.raises(ValueError, match="intervals"):
        analyze_composition(
            exact, projected,
            upstream_heldout_ci95=[0.2], joint_heldout_ci95=[0.2, 0.4],
        )
