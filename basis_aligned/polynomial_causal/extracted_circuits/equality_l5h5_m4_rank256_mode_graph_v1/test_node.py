import torch

from . import node


def test_compile_and_execute_shapes_and_bridge():
    torch.manual_seed(4)
    hidden, products, reader, rank = 12, 24, 4, 16
    qk = [torch.randn(reader, hidden) for _ in range(4)]
    down = torch.randn(hidden, products)
    program = node.compile_program(qk, down, rank=rank)
    assert program["basis"].shape == (rank, products)
    assert program["writers"].shape == (hidden, rank)
    assert program["contracted_reconstruction_relative_l2"] < 1e-10
    state = torch.randn(2, 7, hidden, dtype=torch.bfloat16)
    left = torch.randn(products, hidden)
    right = torch.randn(products, hidden)
    bias = torch.randn(hidden)
    write = node.execute_m4(state, left, right, bias, program, residual_scale=.75)
    assert write.shape == state.shape
    assert write.dtype == torch.float32
