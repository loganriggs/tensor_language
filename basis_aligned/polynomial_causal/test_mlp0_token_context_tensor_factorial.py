import torch

import mlp0_token_context_tensor_factorial as subject


def test_four_branch_bilinear_expansion_is_exact_and_indefinite():
    torch.manual_seed(2)
    batch, dimension, hidden, output = 17, 7, 11, 5
    token = torch.randn(batch, dimension)
    context = torch.randn(batch, dimension)
    left = torch.randn(hidden, dimension)
    right = torch.randn(hidden, dimension)
    down = torch.randn(output, hidden)
    branches = subject.quadratic_tensor_branches(token, context, left, right, down)
    exact = subject.full_float_quadratic(token + context, left, right, down)
    assert torch.allclose(sum(branches.values()), exact, atol=3e-5, rtol=2e-5)


def test_changing_token_context_split_preserves_full_tensor_sum():
    torch.manual_seed(3)
    batch, dimension, hidden, output = 9, 6, 8, 4
    token = torch.randn(batch, dimension)
    context = torch.randn(batch, dimension)
    shift = torch.randn(batch, dimension)
    left = torch.randn(hidden, dimension)
    right = torch.randn(hidden, dimension)
    down = torch.randn(output, hidden)
    first = sum(subject.quadratic_tensor_branches(token, context, left, right, down).values())
    second = sum(subject.quadratic_tensor_branches(
        token + shift, context - shift, left, right, down,
    ).values())
    assert torch.allclose(first, second, atol=5e-5, rtol=3e-5)


def test_observed_state_split_reconstructs_to_float32_roundoff():
    torch.manual_seed(5)
    token = torch.randn(13, subject.D)
    context = torch.randn(13, subject.D)
    raw = token + context
    normalized = raw / raw.square().mean(-1, keepdim=True).sqrt()
    token_part, context_part, error = subject.split_normalized_state(
        normalized, token, context,
    )
    relative_mse = (
        (token_part + context_part - normalized.float()).square().sum()
        / normalized.float().square().sum()
    )
    assert float(relative_mse) < 1e-14
    assert float(error.max()) < 1e-12


def test_mobius_and_shapley_exactly_account_for_factorial_value():
    weights = {"TT": 1.2, "X": -0.3, "CC": 0.7}
    pair = {frozenset(("TT", "X")): 0.4, frozenset(("X", "CC")): -0.2}
    triple = 0.5
    performance = {}
    for subset in subject.ARMS:
        value = 2.0 + sum(weights[item] for item in subset)
        value += sum(v for key, v in pair.items() if key.issubset(subset))
        if len(subset) == 3:
            value += triple
        performance[subset] = value
    dividends = subject.mobius_dividends(performance)
    shapley = subject.shapley_values(performance)
    full_gain = performance[frozenset(subject.BRANCHES)] - performance[frozenset()]
    assert abs(sum(shapley.values()) - full_gain) < 1e-12
    assert abs(sum(value for key, value in dividends.items() if key != "EMPTY") - full_gain) < 1e-12
    assert abs(dividends["TT+X"] - 0.4) < 1e-12
    assert abs(dividends["TT+X+CC"] - triple) < 1e-12


def test_overlapping_lexical_dag_plus_private_residual_reconstructs_table():
    torch.manual_seed(7)
    table = torch.randn(10, 6)
    incidence = torch.tensor([
        [1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1],
        [1, 1, 0], [1, 0, 0], [1, 1, 1], [1, 0, 1], [1, 1, 0],
    ], dtype=torch.float32)
    weights = torch.arange(1, 11, dtype=torch.float32)
    mean, atoms, shared, private = subject.lexical_dag_decomposition(
        table, incidence, weights,
    )
    assert atoms.shape == (3, 6)
    assert torch.allclose(mean + shared + private, table, atol=2e-6, rtol=2e-6)
    assert int((incidence.sum(1) > 1).sum()) > 0
