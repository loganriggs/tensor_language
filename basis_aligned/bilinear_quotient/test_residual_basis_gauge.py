import torch

from . import residual_basis_gauge as gauge


def orthogonal(width, seed):
    generator = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(
        width, width, generator=generator, dtype=torch.double))
    return q*torch.sign(torch.diag(r))


def test_canonical_frame_is_invariant_to_global_orthogonal_gauge():
    generator = torch.Generator().manual_seed(1)
    anchor = torch.randn(13, 5, generator=generator, dtype=torch.double) \
        @ torch.diag(torch.tensor([5., 4., 3., 2., 1.], dtype=torch.double))
    rotation = orthogonal(5, 2)
    first = gauge.canonical_frame(anchor)
    second = gauge.canonical_frame(anchor@rotation)
    assert torch.allclose(first["canonical_anchor"], second["canonical_anchor"],
                          atol=1e-10, rtol=1e-10)
    assert torch.allclose(second["frame"], rotation.T@first["frame"],
                          atol=1e-10, rtol=1e-10)
    assert bool((first["canonical_anchor"][first["pivot_rows"],
                                            torch.arange(5)] > 0).all())


def test_repeated_and_rank_deficient_strata_fail_closed():
    for anchor in (torch.eye(4), torch.diag(torch.tensor([4., 3., 2., 0.]))):
        try:
            gauge.canonical_frame(anchor)
        except ValueError:
            pass
        else:
            raise AssertionError("non-identifiable anchor stratum was accepted")


def test_tiny_rms_attention_bilinear_network_is_gauge_equivalent():
    generator = torch.Generator().manual_seed(3)
    vocab, width, hidden, batch = 11, 5, 7, 9
    rand = lambda *shape: torch.randn(*shape, generator=generator, dtype=torch.double)
    E, U = rand(vocab, width), rand(vocab, width)
    Wq, Wk, Wv, Wo = rand(width, width), rand(width, width), \
        rand(width, width), rand(width, width)
    left, right, down = rand(hidden, width), rand(hidden, width), rand(width, hidden)
    bias = rand(1, width)
    token_ids = torch.randint(vocab, (batch,), generator=generator)

    def forward(parts):
        E_, U_, Wq_, Wk_, Wv_, Wo_, left_, right_, down_, bias_ = parts
        x = E_[token_ids]
        z = width**.5*x/x.norm(dim=-1, keepdim=True)
        routed = (z@Wq_.T)*(z@Wk_.T)*(z@Wv_.T)
        attention = routed@Wo_.T
        mlp = ((z@left_.T)*(z@right_.T))@down_.T+bias_
        state = .7*x+attention+mlp
        return state, state@U_.T

    parts = (E, U, Wq, Wk, Wv, Wo, left, right, down, bias)
    state, logits = forward(parts)
    q = orthogonal(width, 4)
    rotated = (
        gauge.transform_residual_rows(E, q), gauge.transform_residual_rows(U, q),
        gauge.transform_read_weight(Wq, q), gauge.transform_read_weight(Wk, q),
        gauge.transform_read_weight(Wv, q), gauge.transform_write_weight(Wo, q),
        gauge.transform_read_weight(left, q), gauge.transform_read_weight(right, q),
        gauge.transform_write_weight(down, q), gauge.transform_residual_rows(bias, q))
    rotated_state, rotated_logits = forward(rotated)
    assert torch.allclose(rotated_state, state@q, atol=1e-10, rtol=1e-10)
    assert torch.allclose(rotated_logits, logits, atol=1e-10, rtol=1e-10)


def test_quadratic_factor_rule_matches_rotated_execution():
    generator = torch.Generator().manual_seed(5)
    A = torch.randn(5, 7, generator=generator, dtype=torch.double)
    B = torch.randn(5, 7, generator=generator, dtype=torch.double)
    C = torch.randn(7, 5, generator=generator, dtype=torch.double)
    z = torch.randn(8, 5, generator=generator, dtype=torch.double)
    q = orthogonal(5, 6)
    Aq, Bq, Cq = gauge.transform_quadratic_factors(A, B, C, q)
    native = ((z@A)*(z@B))@C
    rotated = (((z@q)@Aq)*((z@q)@Bq))@Cq
    assert torch.allclose(rotated, native@q, atol=1e-10, rtol=1e-10)
