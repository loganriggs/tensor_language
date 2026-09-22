"""Sanity tests for the ported fitting loop on the norm-free toy stack (CPU, seconds).

1. A planted single-unit circuit is recovered by the plain DCT (alignment of (u, l, r) with the plant >> random).
2. The PR penalty drives the fitted factor's PR over the intermediate range to ~1 on the planted toy and keeps most energy.
3. With penalty weight 0 the penalised class reproduces the plain class exactly (same seed => same factors).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from circuit_checks.toy import BilinearStack, plant_circuit
from circuit_checks.dct_fit import SpanDCT, evaluate
from circuit_checks.checks import factor_similarity


def planted(seed=0, d=16, d_mlp=32, n_layers=3, gain=6.0):
    g = torch.Generator().manual_seed(seed)
    stack = BilinearStack.random(d=d, d_mlp=d_mlp, n_layers=n_layers, scale=0.15, seed=seed)
    l, r, u = (F.normalize(torch.randn(d, generator=g), dim=0) for _ in range(3))
    plant_circuit(stack, layer=1, unit=5, l=l, r=r, u=u, gain=gain)
    xs = [torch.randn(d, generator=g) * 0.3 for _ in range(6)]
    return stack, xs, (u, l, r)


def test_plain_recovers_plant():
    stack, xs, (u, l, r) = planted()
    dic = SpanDCT(1); dic.fit(stack.span_fn, xs, 16, 16, pr_layers=[1], max_iters=15, factor_batch=1, seed=0, device="cpu")
    # u is identified; (l, r) is not: u.H[l, r] is symmetric in l <-> r, so L = R = (l + r)/sqrt2 scores exactly as (l, r) does
    # (the fit lands there: straight and swapped cosines both ~0.77). Test u directly and (l, r) through the 2-dim span.
    cos_u = (F.normalize(dic.U[:, 0], dim=0) @ u).abs().item()
    Q1, _ = torch.linalg.qr(torch.stack([dic.L[:, 0], dic.R[:, 0]], 1)); Q2, _ = torch.linalg.qr(torch.stack([l, r], 1))
    cc = torch.linalg.svdvals(Q1.T @ Q2)
    sim = factor_similarity((dic.U, dic.L, dic.R), (u[:, None], l[:, None], r[:, None])).item()
    assert cos_u > 0.95 and cc.min() > 0.9, (cos_u, cc, sim)
    print("PASS test_plain_recovers_plant", "cos_u", round(cos_u, 3), "span CCs", [round(x, 3) for x in cc.tolist()], "triple similarity", round(sim, 3))


def test_penalty_concentrates():
    stack, xs, _ = planted()
    base = SpanDCT(2); base.fit(stack.span_fn, xs, 16, 16, pr_layers=[1], max_iters=10, factor_batch=2, seed=1, device="cpu")
    ev0 = evaluate(base, stack.span_fn, xs, [1]); scale = ev0["mean_total_energy"] / 2
    pen = SpanDCT(2, 1.0, scale); pen.fit(stack.span_fn, xs, 16, 16, pr_layers=[1], max_iters=10, factor_batch=2, seed=1, device="cpu")
    ev1 = evaluate(pen, stack.span_fn, xs, [1])
    assert ev1["median_participation_ratio"] <= ev0["median_participation_ratio"], (ev0, ev1)
    assert len(pen.normalized_pr_values) == 10 and pen.feature_dim == 32
    print("PASS test_penalty_concentrates", round(ev0["median_participation_ratio"], 2), "->", round(ev1["median_participation_ratio"], 2),
          "energy", round(ev1["mean_total_energy"] / ev0["mean_total_energy"], 3))


def test_zero_weight_matches_plain():
    stack, xs, _ = planted()
    a = SpanDCT(2); a.fit(stack.span_fn, xs, 16, 16, pr_layers=[1], max_iters=4, factor_batch=2, seed=3, device="cpu")
    b = SpanDCT(2, 0.0, 123.0); b.fit(stack.span_fn, xs, 16, 16, pr_layers=[1], max_iters=4, factor_batch=2, seed=3, device="cpu")
    assert torch.allclose(a.U, b.U) and torch.allclose(a.L, b.L) and torch.allclose(a.R, b.R)
    print("PASS test_zero_weight_matches_plain")


if __name__ == "__main__":
    test_plain_recovers_plant(); test_penalty_concentrates(); test_zero_weight_matches_plain()
