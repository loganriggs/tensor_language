import pytest
import torch

import response_tree_rank_diagnostics as subject


def test_planted_hierarchy_is_the_unique_storage_winner() -> None:
    tensor = subject.planted_tree_tensor()
    result = subject.rank_trees(tensor)
    assert result["winner"] == [0, 1]
    assert result["winner_unique_by_storage"] is True
    best, second, third = result["ranked_trees"]
    assert best["edge_ranks"]["internal_bond"] == 2
    assert second["edge_ranks"]["internal_bond"] >= 4
    assert third["edge_ranks"]["internal_bond"] >= 4
    assert best["literal_minimal_ht_storage"] < second["literal_minimal_ht_storage"]


def test_cut_rank_is_invariant_under_invertible_mode_coordinates() -> None:
    tensor = subject.planted_tree_tensor(seed=17)
    generator = torch.Generator().manual_seed(18)
    transformed = tensor
    for mode, size in enumerate(tensor.shape):
        matrix = torch.randn((size, size), generator=generator, dtype=torch.float64)
        matrix = matrix + 3.0 * torch.eye(size, dtype=torch.float64)
        transformed = torch.tensordot(matrix, transformed, dims=([1], [mode]))
        transformed = transformed.movedim(0, mode)
    original = subject.rank_trees(tensor)
    changed = subject.rank_trees(transformed)
    assert [row["edge_ranks"] for row in original["ranked_trees"]] == [
        row["edge_ranks"] for row in changed["ranked_trees"]
    ]
    assert original["winner"] == changed["winner"] == [0, 1]


def test_matricization_tail_matches_best_rank_error() -> None:
    tensor = subject.planted_tree_tensor(seed=19)
    report = subject.cut_spectrum(tensor, (0, 2), energy_fraction=0.90)
    matrix = subject.matricize(tensor, (0, 2))
    rank = report["energy_rank"]
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    approximation = (u[:, :rank] * singular[:rank]) @ vh[:rank]
    observed = float((matrix - approximation).square().sum())
    assert observed == pytest.approx(report["energy_tail_squared_frobenius"], rel=1e-10, abs=1e-10)


def test_incomplete_or_malformed_inputs_fail_closed() -> None:
    tensor = subject.planted_tree_tensor()
    with pytest.raises(ValueError, match="dense four-mode"):
        subject.rank_trees(tensor[0])
    corrupted = tensor.clone()
    corrupted[0, 0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="dense four-mode"):
        subject.rank_trees(corrupted)
    with pytest.raises(ValueError, match="proper subset"):
        subject.matricize(tensor, ())
    with pytest.raises(ValueError, match="unknown"):
        subject.analyze_tree(tensor, (1, 2))
