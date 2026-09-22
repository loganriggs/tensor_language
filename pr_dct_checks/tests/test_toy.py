"""Sanity tests on the toy bilinear stack. Run: python -m pytest tests -q
(or `python tests/test_toy.py`). All CPU, a few seconds.

These establish that the instruments work *before* they're pointed at the
real model: exact weight-space formulas, freezing semantics, and that the
completeness check catches the PR loophole it's designed to catch.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from circuit_checks.core import FreezeSpec, mixed_hessian, participation_ratio
from circuit_checks.checks import (
    all_units, factor_alignment_to_units, effective_unit_directions,
    hidden_response, match_factors, output_score, pathway_completeness,
    random_match_null, random_units, top_units, ablation_effect,
)
from circuit_checks.toy import (
    BilinearStack, expected_mixed_hessian_2layer, plant_circuit,
    single_layer_mixed_hessian,
)

torch.set_default_dtype(torch.float64)
D, DM = 16, 8


def unit(v):
    return v / v.norm()


def basis_dirs(seed=1):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D, D, generator=g, dtype=torch.float64))
    return q[:, 0], q[:, 1], q[:, 2]   # l, r, u orthonormal


def sparse_stack(background=0.01, seed=0):
    s = BilinearStack.random(D, DM, 3, scale=background, seed=seed)
    s.A = [a.double() for a in s.A]; s.B = [b.double() for b in s.B]; s.D = [d.double() for d in s.D]
    return s


def test_single_layer_is_pure_weights():
    s = BilinearStack.random(D, DM, 1, scale=0.5, seed=3)
    s.A = [a.double() for a in s.A]; s.B = [b.double() for b in s.B]; s.D = [d.double() for d in s.D]
    l, r, _ = basis_dirs()
    closed = single_layer_mixed_hessian(s.A[0], s.B[0], s.D[0], l, r)
    for seed in range(3):
        x = torch.randn(D, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)
        H = mixed_hessian(lambda t: s.span_fn(x)(t, None)[0], l, r)
        assert torch.allclose(H, closed, atol=1e-10), "1-layer mixed Hessian should not depend on x"


def test_two_layer_expectation_uses_only_first_two_moments():
    s = BilinearStack.random(D, DM, 2, scale=0.8, seed=4)
    s.A = [a.double() for a in s.A]; s.B = [b.double() for b in s.B]; s.D = [d.double() for d in s.D]
    l, r, _ = basis_dirs()
    X = torch.randn(64, D, generator=torch.Generator().manual_seed(0), dtype=torch.float64) + 0.3
    avg = torch.stack([mixed_hessian(lambda t, x=x: s.span_fn(x)(t, None)[0], l, r) for x in X]).mean(0)
    mu = X.mean(0)
    Sigma = (X - mu).T @ (X - mu) / X.shape[0]          # population covariance
    closed = expected_mixed_hessian_2layer(s, mu, Sigma, l, r)
    assert torch.allclose(avg, closed, atol=1e-9), (avg - closed).abs().max()


def test_freezing_everything_kills_interaction():
    s = BilinearStack.random(D, DM, 3, scale=0.5, seed=5)
    s.A = [a.double() for a in s.A]; s.B = [b.double() for b in s.B]; s.D = [d.double() for d in s.D]
    l, r, u = basis_dirs()
    span = s.span_fn(torch.randn(D, dtype=torch.float64))
    spec = all_units({k: DM for k in range(3)}, range(3))
    assert abs(output_score(span, u, l, r, spec).item()) < 1e-12


def _loophole_and_genuine():
    l, r, u = basis_dirs()
    loop = sparse_stack()
    plant_circuit(loop, 0, 0, l, r, u, gain=1.0)     # real work in the source block (excluded from PR)
    plant_circuit(loop, 1, 0, l, r, u, gain=0.05)    # small remainder in one intermediate unit
    gen = sparse_stack()
    plant_circuit(gen, 1, 0, l, r, u, gain=1.0)      # the work genuinely happens in the intermediate unit
    return loop, gen, (l, r, u)


def test_completeness_catches_pr_loophole():
    loop, gen, (l, r, u) = _loophole_and_genuine()
    ctxs = [torch.randn(D, generator=torch.Generator().manual_seed(i), dtype=torch.float64) * 0.5 for i in range(4)]
    pr_layers = [1, 2]                                  # AJ-style exclusive range for source=0, target=3
    for stack, expect_high in [(loop, False), (gen, True)]:
        span = stack.span_fn(ctxs[0])
        resp = hidden_response(span, l, r, pr_layers)
        pr = participation_ratio(torch.cat([resp[k] for k in pr_layers])).item()
        assert pr < 1.5, f"both cases should look sparse by PR alone, got {pr}"
        res = pathway_completeness(
            stack.span_fn, ctxs, u, l, r,
            {"top1_intermediate": lambda sp: top_units(hidden_response(sp, l, r, pr_layers), 1),
             "source_block_mlp": lambda sp: all_units({0: DM}, [0])},
        )
        top1 = res["top1_intermediate"]["mean"]
        if expect_high:
            assert top1 > 0.9, top1
        else:
            assert top1 < 0.2, top1
            assert res["source_block_mlp"]["mean"] > 0.8


def test_finite_ablation_agrees_with_freezing():
    _, gen, (l, r, u) = _loophole_and_genuine()
    ctxs = [torch.randn(D, generator=torch.Generator().manual_seed(i), dtype=torch.float64) * 0.5 for i in range(3)]
    build = lambda sp: top_units(hidden_response(sp, l, r, [1, 2]), 1)
    abl = ablation_effect(gen.span_fn, ctxs, u, l, r, alpha=0.1, beta=0.1, build_spec=build)
    assert abl["mean"] > 0.9, abl
    g = torch.Generator().manual_seed(0)
    rnd = lambda sp: random_units({1: DM, 2: DM}, 1, g, exclude=build(sp))
    abl_rand = ablation_effect(gen.span_fn, ctxs, u, l, r, 0.1, 0.1, rnd)
    assert abl_rand["mean"] < 0.2, abl_rand


def test_neuron_readoff_alignment():
    _, gen, (l, r, u) = _loophole_and_genuine()
    x = torch.randn(D, dtype=torch.float64) * 0.5
    lefts, rights = effective_unit_directions(gen.mlp_input_fn(x, 1), gen.A[1], gen.B[1], list(range(DM)), D)
    idx, score = factor_alignment_to_units(l, r, lefts, rights)
    assert idx == 0 and score > 0.95, (idx, score)
    # swapped l/r should be recognized too
    idx2, score2 = factor_alignment_to_units(r, l, lefts, rights)
    assert idx2 == 0 and score2 > 0.95


def test_matching():
    g = torch.Generator().manual_seed(0)
    U, L, R = (torch.randn(D, 5, generator=g, dtype=torch.float64) for _ in range(3))
    perm = torch.tensor([2, 0, 4, 1, 3])
    same = match_factors((U, L, R), (U[:, perm], -R[:, perm], L[:, perm]))  # permuted, sign-flipped, L/R-swapped
    assert (same > 0.999).all(), same
    null = random_match_null(D, D, 5, n_draws=5)
    assert null.mean() < 0.15, null.mean()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("PASS", name)
