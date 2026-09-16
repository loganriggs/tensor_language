import torch

from . import node


def test_input_port_matches_explicit_product_boundary():
    torch.manual_seed(41)
    batch, seq, hidden, products, storage_rank = 2, 4, 64, 96, 24
    state = torch.randn(batch, seq, hidden, dtype=torch.bfloat16)
    left = torch.randn(products, hidden)
    right = torch.randn(products, hidden)
    writes = [torch.randn(batch, seq, hidden) for _ in range(4)]
    bias = torch.randn(hidden)
    basis = torch.randn(storage_rank, products)
    writers = torch.randn(hidden, storage_rank)
    qk = [torch.randn(hidden, hidden, dtype=torch.bfloat16) for _ in range(4)]
    cos = torch.randn(batch, seq, 16, dtype=torch.bfloat16)
    sin = torch.randn(batch, seq, 16, dtype=torch.bfloat16)
    graph, product, residuals = node.decompose(
        state, *writes, left, right, bias, basis, writers, .75,
        torch.bfloat16, qk, cos, sin, 1, 32, child_rank=8,
        selected_rank=16, return_intermediates=True,
    )
    expected_product = torch.nn.functional.linear(state, left.to(state.dtype)) * torch.nn.functional.linear(state, right.to(state.dtype))
    assert torch.equal(product, expected_product)
    expected_residuals = node.product_node.construct_residuals(
        expected_product, *writes, bias, basis, writers, .75,
        torch.bfloat16, child_rank=8, selected_rank=16,
    )
    assert all(torch.equal(residuals[name], expected_residuals[name]) for name in residuals)
    assert node.compose(graph).shape == (batch, seq, seq)
