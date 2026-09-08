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


def test_complete_mlp_writer_tensor_replays_readout_and_is_gauge_invariant():
    g = torch.Generator().manual_seed(21)
    mlp = type("MLP", (), {})()
    mlp.Left = Linear(torch.randn(7, 5, generator=g))
    mlp.Right = Linear(torch.randn(7, 5, generator=g))
    mlp.Down = Linear(torch.randn(5, 7, generator=g))
    read = torch.randn(3, 5, generator=g)
    result = subject.mlp_writer_to_read_tensor(mlp, read)
    x = torch.randn(5, generator=g)
    expected = read @ mlp.Down.weight @ (
        (mlp.Left.weight @ x) * (mlp.Right.weight @ x))
    actual = torch.einsum("aij,i,j->a", result["tensor"], x, x)
    assert torch.allclose(actual, expected, atol=1e-5)
    rotation = random_basis(3, 3, g)
    rotated = subject.mlp_writer_to_read_tensor(mlp, rotation @ read)
    assert abs(result["score"] - rotated["score"]) < 1e-4
    assert abs(result["normalized_score"] - rotated["normalized_score"]) < 1e-7


def test_activation_conditioned_mlp_write_is_exact_and_read_gauge_invariant():
    g = torch.Generator().manual_seed(23)
    mlp = type("MLP", (), {})()
    mlp.Left = Linear(torch.randn(7, 5, generator=g))
    mlp.Right = Linear(torch.randn(7, 5, generator=g))
    mlp.Down = Linear(torch.randn(5, 7, generator=g))
    read = torch.randn(3, 5, generator=g)
    base = torch.randn(2, 4, 5, generator=g)
    donor = torch.randn(2, 4, 5, generator=g)
    result = subject.activation_conditioned_mlp_write(mlp, read, base, donor)
    direct = torch.einsum("ad,btd->bta", read,
        torch.einsum("dh,bth->btd", mlp.Down.weight,
            (donor @ mlp.Left.weight.T) * (donor @ mlp.Right.weight.T)
            - (base @ mlp.Left.weight.T) * (base @ mlp.Right.weight.T)))
    assert torch.allclose(result["response"], direct, atol=1e-5)
    rotation = random_basis(3, 3, g)
    rotated = subject.activation_conditioned_mlp_write(mlp, rotation @ read, base, donor)
    assert torch.allclose(torch.linalg.vector_norm(result["response"], dim=-1),
                          torch.linalg.vector_norm(rotated["response"], dim=-1), atol=1e-5)


def test_tensor_unfolding_spectra_are_mode_gauge_invariant():
    g = torch.Generator().manual_seed(29)
    tensor = torch.randn(3, 4, 5, generator=g)
    baseline = subject.tensor_unfolding_spectra(tensor)
    rotations = [random_basis(size, size, g) for size in tensor.shape]
    rotated = torch.einsum("ai,bj,ck,ijk->abc", *rotations, tensor)
    transformed = subject.tensor_unfolding_spectra(rotated)
    for original, changed in zip(baseline, transformed):
        assert torch.allclose(original["singular_values"], changed["singular_values"], atol=1e-5)
        assert abs(original["stable_rank"] - changed["stable_rank"]) < 1e-5


def test_literal_atoms_reconstruct_tensor_and_scores_are_subspace_gauge_invariant():
    g = torch.Generator().manual_seed(31)
    mlp = type("MLP", (), {})()
    mlp.Left = Linear(torch.randn(9, 6, generator=g))
    mlp.Right = Linear(torch.randn(9, 6, generator=g))
    mlp.Down = Linear(torch.randn(6, 9, generator=g))
    source, target = random_basis(6, 3, g), random_basis(6, 2, g)
    restricted = subject.mlp_subspace_tensor(mlp, source, target)
    atoms = subject.mlp_subspace_literal_atoms(restricted)
    assert atoms["relative_error"] < 1e-6
    assert torch.allclose(atoms["reconstruction"], restricted["tensor"], atol=1e-5)
    source_rotation, target_rotation = random_basis(3, 3, g), random_basis(2, 2, g)
    rotated = subject.mlp_subspace_tensor(
        mlp, source @ source_rotation, target @ target_rotation)
    rotated_atoms = subject.mlp_subspace_literal_atoms(rotated)
    assert torch.allclose(atoms["scores"], rotated_atoms["scores"], atol=1e-5)
    assert torch.equal(atoms["order"], rotated_atoms["order"])


def test_bilinear_writer_capability_exactly_replays_finite_substitution_and_is_gauge_invariant():
    g = torch.Generator().manual_seed(37)
    mlp = type("MLP", (), {})()
    mlp.Left = Linear(torch.randn(8, 6, generator=g))
    mlp.Right = Linear(torch.randn(8, 6, generator=g))
    mlp.Down = Linear(torch.randn(6, 8, generator=g))
    writer = random_basis(6, 3, g)
    read = random_basis(6, 2, g).T
    result = subject.bilinear_mlp_writer_capability(mlp, writer, read)
    x = torch.randn(6, generator=g)
    z = torch.randn(3, generator=g)

    def output(value):
        return read @ mlp.Down.weight @ (
            (mlp.Left.weight @ value) * (mlp.Right.weight @ value))

    expected = output(x + writer @ z) - output(x)
    actual = (torch.einsum("aik,i,k->a", result["cross"], x, z)
              + torch.einsum("akl,k,l->a", result["self"], z, z))
    assert torch.allclose(actual, expected, atol=2e-4)

    writer_rotation, read_rotation = random_basis(3, 3, g), random_basis(2, 2, g)
    rotated = subject.bilinear_mlp_writer_capability(
        mlp, writer @ writer_rotation, read_rotation.T @ read)
    assert abs(result["scores"]["cross"] - rotated["scores"]["cross"]) < 1e-4
    assert abs(result["scores"]["self"] - rotated["scores"]["self"]) < 1e-4
    scaled = subject.bilinear_mlp_writer_capability(mlp, 3 * writer, 2 * read)
    assert abs(result["scores"]["cross_normalized"]
               - scaled["scores"]["cross_normalized"]) < 1e-6
    assert abs(result["scores"]["self_normalized"]
               - scaled["scores"]["self_normalized"]) < 1e-6
