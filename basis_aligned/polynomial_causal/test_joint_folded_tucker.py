import itertools

import pytest
import torch

from joint_folded_tucker import (
    execute, executable_price, project_core, projection_error_squared,
    require_orthonormal, sparse_core, tensor_inner, mode_grams,
)


def factors(seed=5):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(*shape, generator=g, dtype=torch.float64)
            for shape in [(7, 6), (6, 5), (6, 5)]]


def dense(C, A, B):
    raw = torch.einsum('vk,ki,kj->vij', C, A, B)
    return (raw + raw.transpose(1, 2)) / 2


def test_inner_matches_materialized_tensor_and_factor_gauge():
    C, A, B = factors()
    D, L, R = factors(17)
    expected = (dense(C, A, B) * dense(D, L, R)).sum()
    for chunk in (1, 4, 20):
        torch.testing.assert_close(tensor_inner(C, A, B, D, L, R, chunk), expected)
    scales = torch.tensor([.01, -3, 12, -.2, 5, 100], dtype=C.dtype)
    torch.testing.assert_close(
        tensor_inner(C / scales, A * scales[:, None], B, D, R, L), expected)


def test_projection_and_sparse_error_match_dense():
    C, A, B = factors()
    W = torch.linalg.qr(C[:, :3]).Q
    P = torch.linalg.qr(A.T[:, :4]).Q
    require_orthonormal(W)
    require_orthonormal(P)
    core = project_core(C, A, B, W, P)
    target = dense(C, A, B)
    torch.testing.assert_close(core, torch.einsum('vij,va,ip,jq->apq', target, W, P, P))
    energy = tensor_inner(C, A, B, C, A, B)
    for budget in (0, 1, 12, 30):
        kept = sparse_core(core, budget)
        approx = torch.einsum('va,apq,ip,jq->vij', W, kept, P, P)
        torch.testing.assert_close(projection_error_squared(energy, core, kept),
                                   (target - approx).square().sum())


def test_mode_grams_against_dense_unfoldings():
    C, A, B = factors()
    target = dense(C, A, B)
    out, inp = mode_grams(C, A, B)
    torch.testing.assert_close(out, torch.einsum('vij,wij->vw', target, target))
    torch.testing.assert_close(inp, torch.einsum('vij,vkj->ik', target, target))


def test_sparsity_counts_unordered_pairs_and_energy():
    core = torch.tensor([[[1.2, 1.0], [1.0, .5]]], dtype=torch.float64)
    kept = sparse_core(core, 1)
    assert kept[0, 0, 0] == 0  # 1**2 * 2 beats 1.2**2
    assert kept[0, 0, 1] == kept[0, 1, 0] == 1
    price = executable_price(torch.ones(4, 1), torch.eye(2), kept)
    assert price['core_values'] == price['unique_pair_products'] == 1
    assert price['stored_values'] == 9


def test_native_rmsnorm_and_radial_invariance():
    C, A, B = factors()
    z = torch.randn(13, 5, dtype=torch.float64)
    W, P = torch.eye(7, dtype=z.dtype), torch.eye(5, dtype=z.dtype)
    core = project_core(C, A, B, W, P)
    eps = 1e-5
    normalized = torch.nn.functional.rms_norm(z, (5,), eps=eps)
    expected = ((normalized @ A.T) * (normalized @ B.T)) @ C.T
    torch.testing.assert_close(execute(z, W, P, core, h=z, rms_eps=eps), expected)
    torch.testing.assert_close(execute(z, W, P, core, h=z),
                               execute(z * 9, W, P, core, h=z * 9))
    assert not torch.allclose(execute(z, W, P, core), expected)


def test_nonorthogonal_frames_rejected():
    with pytest.raises(ValueError, match='orthonormal'):
        require_orthonormal(torch.eye(3) * 2)


def test_shared_pair_reuse_counted_once():
    core = torch.zeros(3, 2, 2)
    core[:, 0, 1] = core[:, 1, 0] = 1
    price = executable_price(torch.eye(3), torch.eye(2), core)
    assert price['core_values'] == 3
    assert price['unique_pair_products'] == 1


def test_two_layer_cancellation_requires_full_symmetrization():
    # User's h=(x0^2+x1^2, x0^2-x1^2), y=h0^2-h1^2.
    S = torch.tensor([[[1., 0], [0, 1]], [[1., 0], [0, -1]]], dtype=torch.float64)
    outer = torch.diag(torch.tensor([1., -1.], dtype=S.dtype))
    H = torch.einsum('ab,aij,bkl->ijkl', outer, S, S)
    sym = sum(H.permute(order) for order in itertools.permutations(range(4))) / 24
    assert torch.count_nonzero(sym) == 6
    torch.testing.assert_close(sym[sym != 0], torch.full((6,), 2/3, dtype=S.dtype))
    x = torch.randn(17, 2, dtype=S.dtype)
    hidden = torch.einsum('ni,aij,nj->na', x, S, x)
    factored = hidden[:, 0].square() - hidden[:, 1].square()
    torch.testing.assert_close(factored, 4*x[:, 0].square()*x[:, 1].square())
    torch.testing.assert_close(torch.einsum('ijkl,ni,nj,nk,nl->n', sym, x, x, x, x), factored)
