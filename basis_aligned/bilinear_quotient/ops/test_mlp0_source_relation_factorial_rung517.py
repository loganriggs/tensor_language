import importlib.util
from pathlib import Path

import torch


PATH = Path(__file__).with_name("mlp0_source_relation_factorial_rung517.py")
SPEC = importlib.util.spec_from_file_location("r517", PATH)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def test_source_groups_partition_every_causal_edge_once():
    tokens = torch.tensor([[1, 2, 1, 4, 5, 6, 7, 8, 1], [9, 9, 8, 7, 6, 5, 4, 3, 9]])
    masks = R.source_group_masks(tokens)
    length = tokens.shape[1]
    causal = torch.tril(torch.ones(length, length, dtype=torch.int8))
    assert torch.equal(masks.to(torch.int8).sum(0), causal.expand(tokens.shape[0], -1, -1))


def test_subset_context_keeps_numerical_remainder_and_full_is_native():
    split = {
        "native_write": torch.tensor([[[7.0]]]),
        "group_writes": torch.arange(1, 6, dtype=torch.float32).view(5, 1, 1, 1),
        "numerical_remainder": torch.tensor([[[-8.0]]]),
    }
    assert torch.equal(R.subset_context(split, 0), torch.tensor([[[-8.0]]]))
    assert torch.equal(R.subset_context(split, 1 | 4), torch.tensor([[[-4.0]]]))
    assert torch.equal(R.subset_context(split, R.N_ARMS - 1), split["native_write"])


def test_group_boundary_definitions():
    tokens = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10, 3]])
    masks = R.source_group_masks(tokens)
    assert masks[R.GROUPS.index("SELF"), 0, 8, 8]
    assert masks[R.GROUPS.index("PREVIOUS"), 0, 8, 7]
    assert masks[R.GROUPS.index("NEAR"), 0, 8, 1]
    assert masks[R.GROUPS.index("DISTANT_SAME"), 0, 8, 0]


def test_mobius_round_trip():
    coefficients = torch.randn(R.N_ARMS, 11, dtype=torch.float64)
    values = R.subset_values_from_mobius(coefficients)
    recovered = R.mobius_from_subset_values(values)
    assert torch.allclose(recovered, coefficients, atol=1e-12, rtol=0)


def test_shapley_efficiency():
    coefficients = torch.randn(R.N_ARMS, 3, dtype=torch.float64)
    values = R.subset_values_from_mobius(coefficients)
    shapley = R.shapley_from_mobius(coefficients)
    assert torch.allclose(shapley.sum(0), values[-1] - values[0], atol=1e-12, rtol=0)


def test_all_eight_planted_problems_recover():
    result = R.planted_suite()
    assert len(result["cases"]) == 8
    assert result["all_eight_exact"]


def test_scientific_path_fails_closed():
    try:
        R.main()
    except RuntimeError as error:
        assert "fail-closed" in str(error)
    else:
        raise AssertionError("scientific path unexpectedly opened")
