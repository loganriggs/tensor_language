import itertools

import torch

from quartic_bilinear_quotient import (
    execute, product_inner, program_inner, sparse_program_price, symmetric,
)


def sym4(H):
    return sum(H.permute(p) for p in itertools.permutations(range(4))) / 24


def test_inner_and_gradient_match_dense_symmetrization():
    g = torch.Generator().manual_seed(20)
    forms = [torch.randn(4, 4, generator=g, dtype=torch.float64, requires_grad=True)
             for _ in range(4)]
    Q, R, S, T = map(symmetric, forms)
    H = sym4(torch.einsum('ij,kl->ijkl', Q, R))
    J = sym4(torch.einsum('ij,kl->ijkl', S, T))
    dense = (H*J).sum()
    implicit = product_inner(*forms)
    torch.testing.assert_close(implicit, dense)
    a = torch.autograd.grad(implicit, forms, retain_graph=True)
    b = torch.autograd.grad(dense, forms)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right)


def test_rank_inflation_for_identical_polynomial():
    for d in (2, 3, 5):
        eye = torch.eye(d, dtype=torch.float64)
        H = torch.einsum('ij,kl->ijkl', eye, eye)
        S = sym4(H)
        assert torch.linalg.matrix_rank(H.reshape(d*d, d*d)) == 1
        assert torch.linalg.matrix_rank(S.reshape(d*d, d*d)) == d*(d+1)//2
        x = torch.randn(9, d, dtype=eye.dtype)
        direct = x.square().sum(-1).square()
        actual = torch.einsum('ijkl,ni,nj,nk,nl->n', S, x, x, x, x)
        torch.testing.assert_close(actual, direct)
        torch.testing.assert_close(product_inner(eye, eye, eye, eye), S.square().sum())


def test_slot_grouping_changes_multilinear_rank():
    A = torch.diag(torch.tensor([1., 2., 0.], dtype=torch.float64))
    B = torch.diag(torch.tensor([3., 4., 5.], dtype=torch.float64))
    H = torch.einsum('ij,kl->ijkl', A, B)
    assert torch.linalg.matrix_rank(H.reshape(9, 9)) == 1
    assert torch.linalg.matrix_rank(H.permute(0, 2, 1, 3).reshape(9, 9)) == 6


def test_signed_cancellation_equivalence_in_polynomial_quotient():
    # (x0²+x1²)²-(x0²-x1²)² = (2*x0*x1)².
    original = torch.stack([torch.eye(2), torch.diag(torch.tensor([1., -1.]))]).double()
    compact = torch.tensor([[[0., 1.], [1., 0.]]], dtype=torch.float64)
    e1 = torch.tensor([[0, 0], [1, 1]])
    e2 = torch.tensor([[0, 0]])
    w1 = torch.tensor([[1., -1.]], dtype=torch.float64)
    w2 = torch.ones(1, 1, dtype=torch.float64)
    aa = program_inner(original, e1, w1, original, e1, w1)
    bb = program_inner(compact, e2, w2, compact, e2, w2)
    ab = program_inner(original, e1, w1, compact, e2, w2)
    torch.testing.assert_close(aa+bb-2*ab, torch.zeros((), dtype=aa.dtype))
    x = torch.randn(17, 2, dtype=torch.float64)
    torch.testing.assert_close(execute(original, e1, w1, x), execute(compact, e2, w2, x))


def test_shared_feature_execution_removal_and_price():
    bank = torch.stack([torch.eye(4), torch.diag(torch.tensor([1., 1., 0., 0.]))]).double()
    edges = torch.tensor([[0, 0], [0, 1]])
    writers = torch.eye(2, dtype=torch.float64)
    x = torch.randn(12, 4, dtype=torch.float64)
    out = execute(bank, edges, writers, x)
    q, r = x.square().sum(-1), x[:, :2].square().sum(-1)
    torch.testing.assert_close(out, torch.stack([q*q, q*r], -1))
    assert execute(bank, edges, writers, x, zero_features=(0,)).count_nonzero() == 0
    price = sparse_program_price(bank, edges, writers)
    assert price['features'] == 2
    assert price['shared_input_products'] == 4
    assert price['quadratic_coefficients'] == 6
    assert price['root_products'] == 2
