import torch

from . import node


def test_product_port_constructor_shapes_and_closure():
    torch.manual_seed(37)
    batch, seq, hidden, products, storage_rank = 2, 5, 64, 96, 24
    product = torch.randn(batch, seq, products, dtype=torch.bfloat16)
    writes = [torch.randn(batch, seq, hidden) for _ in range(4)]
    bias = torch.randn(hidden)
    basis = torch.randn(storage_rank, products)
    writers = torch.randn(hidden, storage_rank)
    qk = [torch.randn(hidden, hidden, dtype=torch.bfloat16) for _ in range(4)]
    cos = torch.randn(batch, seq, 16, dtype=torch.bfloat16)
    sin = torch.randn(batch, seq, 16, dtype=torch.bfloat16)
    graph, residuals = node.decompose(product, *writes, bias, basis, writers,
                                      .75, torch.bfloat16, qk, cos, sin,
                                      head_index=1, head_width=32,
                                      child_rank=8, selected_rank=16,
                                      return_residuals=True)
    assert set(residuals) == {"baseline", "child", "remainder", "joint"}
    assert all(value.shape == (batch, seq, hidden) for value in residuals.values())
    assert node.compose(graph).shape == (batch, seq, seq)
    sliced = node.construct_residuals(product, *writes, bias, basis[:16],
                                      writers[:, :16], .75, torch.bfloat16,
                                      child_rank=8, selected_rank=16)
    assert all(torch.equal(residuals[name], sliced[name]) for name in residuals)
