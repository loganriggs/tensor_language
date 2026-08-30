import numpy as np

from component_geometry_causal_transport import analyze_geometry, nested_matrix


def test_nested_matrix_respects_requested_name_order() -> None:
    nested = {"b": {"b": 4, "a": 3}, "a": {"b": 2, "a": 1}}
    np.testing.assert_array_equal(nested_matrix(nested, ["a", "b"]), [[1, 2], [3, 4]])


def test_geometry_screen_detects_aligned_cross_effects() -> None:
    cosine = np.asarray(
        [
            [1.0, 0.9, 0.3, 0.1],
            [0.9, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.8],
            [0.1, 0.2, 0.8, 1.0],
        ]
    )
    causal = 2.0 + 3.0 * cosine
    result = analyze_geometry(cosine, causal, seed=3, permutation_draws=500)
    assert result["off_diagonal_spearman"] > 0.99
    assert result["top1_source_matches"] == 4


def test_geometry_screen_preserves_reversed_sign() -> None:
    cosine = np.asarray(
        [
            [1.0, 0.9, 0.3, 0.1],
            [0.9, 1.0, 0.5, 0.2],
            [0.3, 0.5, 1.0, 0.8],
            [0.1, 0.2, 0.8, 1.0],
        ]
    )
    causal = 2.0 - cosine
    result = analyze_geometry(cosine, causal, seed=4, permutation_draws=500)
    assert result["off_diagonal_spearman"] < -0.99
    assert result["top1_source_matches"] == 0
