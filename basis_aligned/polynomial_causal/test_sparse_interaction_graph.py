import numpy as np
import pytest
import torch

from basis_aligned.polynomial_causal import sparse_interaction_graph as graph
from basis_aligned.polynomial_causal.extracted_circuits.equality_l5h5_m4_explicit_interaction_graph_v1 import node as equality
from basis_aligned.polynomial_causal.extracted_circuits.subject_number_l11h3_sparse_graph_v1 import node as subject


def _corners(n_ports, dividends):
    return {mask: sum((value for term, value in dividends.items() if term & mask == term),
                      np.zeros_like(next(iter(dividends.values()))))
            for mask in range(1 << n_ports)}


def test_full_transform_and_reconstruction_are_exact_for_vectors():
    rng = np.random.default_rng(17)
    expected = {mask: rng.normal(size=(6, 3)) for mask in range(16)}
    corners = _corners(4, expected)
    actual = graph.full_dividends(corners, 4)
    for mask in range(16):
        np.testing.assert_allclose(actual[mask], expected[mask], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(graph.reconstruct(actual, mask), corners[mask], rtol=1e-12, atol=1e-12)


def test_selected_subject_graph_uses_minimal_ten_corners():
    assert graph.required_corners(subject.MASKS, len(subject.PORTS)) == subject.REQUIRED_CORNERS
    rng = np.random.default_rng(19)
    expected = {mask: rng.normal(size=11) for mask in subject.MASKS}
    baseline = rng.normal(size=11)
    corners = {0: baseline}
    for corner in subject.REQUIRED_CORNERS[1:]:
        corners[corner] = baseline - sum((value for mask, value in expected.items() if mask & corner == mask),
                                         np.zeros_like(baseline))
    actual = graph.damage_atoms(corners, subject.MASKS, len(subject.PORTS))
    for mask in subject.MASKS:
        np.testing.assert_allclose(actual[mask], expected[mask], rtol=1e-12, atol=1e-12)


def test_frozen_greedy_selection_and_disjoint_ood_metrics():
    atoms = {
        1: np.array([3., 0., 2., 0., 4., 0.]),
        2: np.array([0., 2., 0., 3., 0., 4.]),
        3: np.array([.1, -.1, .1, -.1, .1, -.1]),
    }
    target = atoms[1] + atoms[2]
    selected, curve = graph.greedy_select(target, atoms, [0, 1, 2, 3], 2)
    assert selected == (1, 2)
    assert curve[-1] == pytest.approx(0.)
    reports = graph.evaluate_frozen(target, atoms, selected,
                                    {"discovery": [0, 1, 2, 3], "fresh_ood": [4, 5]})
    assert reports["discovery"]["relative_l2"] == pytest.approx(0.)
    assert reports["fresh_ood"]["relative_l2"] == pytest.approx(0.)
    with pytest.raises(ValueError, match="overlaps discovery"):
        graph.evaluate_frozen(target, atoms, selected,
                              {"discovery": [0, 1], "bad_ood": [1, 2]})


def test_generic_arithmetic_matches_two_independent_exported_packages():
    torch.manual_seed(23)
    baseline = torch.randn(3, 4)
    child_effect = torch.randn_like(baseline)
    remainder_effect = torch.randn_like(baseline)
    interaction = torch.randn_like(baseline)
    corners = {0: baseline, 1: baseline + child_effect, 2: baseline + remainder_effect,
               3: baseline + child_effect + remainder_effect + interaction}
    generic = graph.full_dividends(corners, 2)
    packaged = equality.decompose(corners[0], corners[1], corners[2], corners[3])
    torch.testing.assert_close(generic[1], packaged["child_effect"])
    torch.testing.assert_close(generic[2], packaged["remainder_effect"])
    torch.testing.assert_close(generic[3], packaged["interaction"])

    subject_corners = {mask: torch.randn(7) for mask in subject.REQUIRED_CORNERS}
    generic_subject = graph.damage_atoms(subject_corners, subject.MASKS, len(subject.PORTS))
    packaged_subject = subject.decompose(subject_corners)
    for mask, name in zip(subject.MASKS, subject.TERMS):
        torch.testing.assert_close(generic_subject[mask], packaged_subject[name])


def test_missing_corner_shape_and_mask_errors_are_rejected():
    with pytest.raises(ValueError, match="missing corner"):
        graph.selected_dividends({0: np.zeros(2), 1: np.ones(2)}, [3], 2)
    with pytest.raises(ValueError, match="matching shapes"):
        graph.selected_dividends({0: np.zeros(2), 1: np.ones(3)}, [1], 1)
    with pytest.raises(ValueError, match="invalid mask"):
        graph.required_corners([4], 2)
