import torch

import subspace_weight_atlas as subject


class Linear:
    def __init__(self, weight):
        self.weight = torch.nn.Parameter(weight)


def random_basis(dimension, rank, generator):
    value = torch.randn(dimension, rank, generator=generator)
    return torch.linalg.qr(value).Q


def test_attention_factors_contract_exact_ov_and_are_gauge_invariant():
    g = torch.Generator().manual_seed(4)
    attention = type("Attention", (), {"n_head": 2, "head_dim": 3})()
    for name in ("c_q", "c_k", "c_q2", "c_k2", "c_v", "c_proj"):
        setattr(attention, name, Linear(torch.randn(6, 6, generator=g)))
    source, target = random_basis(6, 2, g), random_basis(6, 2, g)
    factors = subject.attention_subspace_factors(attention, source, target)
    head = 1
    start = head * 3
    expected = (target.T @ attention.c_proj.weight[:, start:start + 3]
                @ attention.c_v.weight[start:start + 3] @ source)
    assert torch.allclose(factors[head]["ov"], expected)
    rotation = random_basis(2, 2, g)
    rotated = subject.attention_subspace_factors(attention, source @ rotation, target)
    assert abs(factors[head]["scores"]["ov"] - rotated[head]["scores"]["ov"]) < 1e-5


def test_bilinear_tensor_replays_exact_restricted_polynomial_and_norm_is_gauge_invariant():
    g = torch.Generator().manual_seed(7)
    mlp = type("MLP", (), {})()
    mlp.Left = Linear(torch.randn(8, 5, generator=g))
    mlp.Right = Linear(torch.randn(8, 5, generator=g))
    mlp.Down = Linear(torch.randn(5, 8, generator=g))
    source, target = random_basis(5, 2, g), random_basis(5, 3, g)
    result = subject.mlp_subspace_tensor(mlp, source, target)
    coordinate = torch.randn(2, generator=g)
    x = source @ coordinate
    expected = target.T @ (mlp.Down.weight @ ((mlp.Left.weight @ x) * (mlp.Right.weight @ x)))
    actual = torch.einsum("aij,i,j->a", result["tensor"], coordinate, coordinate)
    assert torch.allclose(actual, expected, atol=1e-5)
    rotation = random_basis(2, 2, g)
    rotated = subject.mlp_subspace_tensor(mlp, source @ rotation, target)
    assert abs(result["scores"]["tensor"] - rotated["scores"]["tensor"]) < 1e-5


def test_head_bank_read_write_contractions_are_exact_and_gauge_invariant():
    g = torch.Generator().manual_seed(12)
    attention = type("Attention", (), {"n_head": 3, "head_dim": 2})()
    attention.c_v = Linear(torch.randn(6, 6, generator=g))
    attention.c_proj = Linear(torch.randn(6, 6, generator=g))
    basis = random_basis(4, 2, g)
    heads = (0, 2)
    value_rows = torch.cat((attention.c_v.weight[0:2], attention.c_v.weight[4:6]))
    output_columns = torch.cat((attention.c_proj.weight[:, 0:2],
                                attention.c_proj.weight[:, 4:6]), dim=1)
    read = subject.head_bank_value_read_map(attention, heads, basis)
    assert torch.allclose(read, basis.T @ value_rows)
    mapped, singular = subject.map_head_bank_subspace_to_residual(attention, heads, basis)
    expected_singular = torch.linalg.svdvals(output_columns @ basis)
    assert torch.allclose(singular, expected_singular, atol=1e-5)
    rotation = random_basis(2, 2, g)
    rotated_read = subject.head_bank_value_read_map(attention, heads, basis @ rotation)
    assert abs(torch.linalg.matrix_norm(read) - torch.linalg.matrix_norm(rotated_read)) < 1e-5
    rotated_mapped, _ = subject.map_head_bank_subspace_to_residual(
        attention, heads, basis @ rotation)
    assert torch.allclose(mapped @ mapped.T, rotated_mapped @ rotated_mapped.T, atol=1e-5)


def test_writer_contractions_match_explicit_maps():
    g = torch.Generator().manual_seed(15)
    attention = type("Attention", (), {"n_head": 2, "head_dim": 3})()
    attention.c_proj = Linear(torch.randn(6, 6, generator=g))
    mlp = type("MLP", (), {})()
    mlp.Down = Linear(torch.randn(6, 8, generator=g))
    read = torch.randn(2, 6, generator=g)
    head = subject.attention_writer_to_read_map(attention, 1, read)
    assert torch.allclose(head["contraction"], read @ attention.c_proj.weight[:, 3:6])
    assert abs(head["score"] - torch.linalg.matrix_norm(head["contraction"])) < 1e-6
    mlp_result = subject.mlp_writer_to_read_map(mlp, read)
    assert torch.allclose(mlp_result["contraction"], read @ mlp.Down.weight)
