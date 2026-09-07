import torch

import quadratic_reader_metric as metric


def explicit_forms(left, right, coefficients):
    return torch.stack([.5 * (left.T @ torch.diag(coefficients[:, j]) @ right +
                               right.T @ torch.diag(coefficients[:, j]) @ left)
                        for j in range(coefficients.shape[1])], dim=-1)


def test_implicit_quadratic_gram_matches_materialized_forms():
    generator = torch.Generator().manual_seed(1729)
    left = torch.randn(7, 5, dtype=torch.float64, generator=generator)
    right = torch.randn(7, 5, dtype=torch.float64, generator=generator)
    a = torch.randn(7, 3, dtype=torch.float64, generator=generator)
    b = torch.randn(7, 2, dtype=torch.float64, generator=generator)
    fa, fb = explicit_forms(left, right, a), explicit_forms(left, right, b)
    explicit = torch.einsum("abj,abk->jk", fa, fb)
    implicit = metric.quadratic_gram(torch, left, right, a, b, block_size=3)
    torch.testing.assert_close(implicit, explicit, rtol=1e-12, atol=1e-12)


def test_metric_principal_cosines_match_explicit_vectorization():
    generator = torch.Generator().manual_seed(2718)
    left = torch.randn(9, 6, dtype=torch.float64, generator=generator)
    right = torch.randn(9, 6, dtype=torch.float64, generator=generator)
    a = torch.randn(9, 4, dtype=torch.float64, generator=generator)
    b = torch.randn(9, 3, dtype=torch.float64, generator=generator)
    gaa = metric.quadratic_gram(torch, left, right, a, block_size=4)
    gbb = metric.quadratic_gram(torch, left, right, b, block_size=4)
    gab = metric.quadratic_gram(torch, left, right, a, b, block_size=4)
    implicit = metric.principal_cosines(torch, gaa, gbb, gab)
    fa = explicit_forms(left, right, a).reshape(36, 4)
    fb = explicit_forms(left, right, b).reshape(36, 3)
    qa = torch.linalg.qr(fa, mode="reduced").Q; qb = torch.linalg.qr(fb, mode="reduced").Q
    explicit = torch.linalg.svdvals(qa.T @ qb)
    torch.testing.assert_close(implicit, explicit, rtol=1e-11, atol=1e-11)


def test_quadratic_metric_is_invariant_to_reader_gauge():
    generator = torch.Generator().manual_seed(3141)
    left = torch.randn(8, 5, dtype=torch.float64, generator=generator)
    right = torch.randn(8, 5, dtype=torch.float64, generator=generator)
    a = torch.randn(8, 3, dtype=torch.float64, generator=generator)
    rotation = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64, generator=generator)).Q
    gaa = metric.quadratic_gram(torch, left, right, a)
    rotated = metric.quadratic_gram(torch, left, right, a @ rotation)
    torch.testing.assert_close(rotated, rotation.T @ gaa @ rotation, rtol=1e-12, atol=1e-12)
    cosines = metric.principal_cosines(torch, gaa, rotated,
                                       metric.quadratic_gram(torch, left, right, a, a @ rotation))
    torch.testing.assert_close(cosines, torch.ones_like(cosines), rtol=1e-11, atol=1e-11)


def test_materialized_hidden_metric_matches_blocked_gram():
    generator = torch.Generator().manual_seed(5772)
    left = torch.randn(11, 7, dtype=torch.float64, generator=generator)
    right = torch.randn(11, 7, dtype=torch.float64, generator=generator)
    a = torch.randn(11, 4, dtype=torch.float64, generator=generator)
    b = torch.randn(11, 3, dtype=torch.float64, generator=generator)
    hidden = metric.hidden_metric(left, right)
    direct = metric.gram_from_hidden_metric(a, hidden, b)
    blocked = metric.quadratic_gram(torch, left, right, a, b, block_size=5)
    torch.testing.assert_close(direct, blocked, rtol=1e-12, atol=1e-12)
