"""The four checks from the plan (E1-E4) plus helpers.

All functions take `make_span(context) -> span(theta, freeze) -> (out, hidden)`
and plain tensors, so they work for both the toy stack and the real model.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch.func import vjp

from .core import FreezeSpec, finite_mixed_difference, mixed_hessian, participation_ratio

Span = Callable[[torch.Tensor, Optional[FreezeSpec]], Tuple[torch.Tensor, Dict[int, torch.Tensor]]]


# ---------------------------------------------------------------- responses

def output_score(span: Span, u, l, r, freeze: Optional[FreezeSpec] = None) -> torch.Tensor:
    """u . H_out[l, r] (the DCT 'causal importance' before squaring)."""
    H = mixed_hessian(lambda t: span(t, freeze)[0], l, r)
    return (u.float() @ H.float())


def hidden_response(span: Span, l, r, layers: Sequence[int]) -> Dict[int, torch.Tensor]:
    """Mixed Hessian of every recorded MLP hidden unit, per layer (AJ's c_{l,i})."""
    out = {}
    for k in layers:
        out[k] = mixed_hessian(lambda t, k=k: span(t, None)[1][k], l, r)
    return out


def top_units(response: Dict[int, torch.Tensor], k: int) -> FreezeSpec:
    """FreezeSpec freezing the global top-k units by |mixed Hessian|."""
    layers = sorted(response)
    flat = torch.cat([response[L].abs().flatten() for L in layers])
    idx = torch.topk(flat, k).indices
    masks, offset = {}, 0
    for L in layers:
        n = response[L].numel()
        m = torch.zeros(n)
        sel = idx[(idx >= offset) & (idx < offset + n)] - offset
        m[sel] = 1
        masks[L] = m
        offset += n
    return FreezeSpec(mlp_masks=masks)


def random_units(widths: Dict[int, int], k: int, generator: torch.Generator,
                 exclude: Optional[FreezeSpec] = None) -> FreezeSpec:
    """Random k units across the given layers (optionally avoiding `exclude`)."""
    layers = sorted(widths)
    total = sum(widths[L] for L in layers)
    allowed = torch.ones(total, dtype=torch.bool)
    if exclude is not None:
        off = 0
        for L in layers:
            if L in exclude.mlp_masks:
                allowed[off:off + widths[L]] &= exclude.mlp_masks[L] == 0
            off += widths[L]
    pool = allowed.nonzero().flatten()
    idx = pool[torch.randperm(len(pool), generator=generator)[:k]]
    masks, off = {}, 0
    for L in layers:
        m = torch.zeros(widths[L])
        sel = idx[(idx >= off) & (idx < off + widths[L])] - off
        m[sel] = 1
        masks[L] = m
        off += widths[L]
    return FreezeSpec(mlp_masks=masks)


def all_units(widths: Dict[int, int], layers: Iterable[int]) -> FreezeSpec:
    return FreezeSpec(mlp_masks={L: torch.ones(widths[L]) for L in layers})


# ---------------------------------------------------------------- E1: completeness

def pathway_completeness(make_span: Callable[[object], Span], contexts: Sequence,
                         u, l, r, groups: Dict[str, Callable[[Span], FreezeSpec]]
                         ) -> Dict[str, Dict[str, float]]:
    """Fraction of u.H_out[l,r] that disappears when each group is frozen.

    groups: name -> fn(span) -> FreezeSpec (a function so that e.g. top-k can be
    chosen per context from that context's hidden response).

    1.0  = the whole interaction flows through the group.
    ~0   = the group is irrelevant to the interaction.
    Report this next to PR. PR ~ 1 with top-1 completeness ~ 0 is the loophole.
    """
    fracs = {name: [] for name in groups}
    full_scores = []
    for c in contexts:
        span = make_span(c)
        full = output_score(span, u, l, r).item()
        full_scores.append(full)
        for name, build in groups.items():
            frozen = output_score(span, u, l, r, build(span)).item()
            fracs[name].append(1.0 - frozen / full if abs(full) > 1e-12 else float('nan'))
    summary = {name: {"mean": float(np.nanmean(v)), "median": float(np.nanmedian(v)),
                      "per_context": v} for name, v in fracs.items()}
    summary["_full_score"] = {"mean_abs": float(np.mean(np.abs(full_scores))),
                              "per_context": full_scores}
    return summary


# ---------------------------------------------------------------- E2: neuron readoff

def effective_unit_directions(mlp_input_fn: Callable[[torch.Tensor], torch.Tensor],
                              W_left: torch.Tensor, W_right: torch.Tensor,
                              units: Sequence[int], d_source: int):
    """Pull a unit's read vectors back to the source layer: J^T a_h, J^T b_h.

    mlp_input_fn: theta -> the (normalized) vector the MLP at this layer reads.
    For a bilinear unit (a.n)(b.n), to first order in theta the unit sees
    (J^T a . theta)(J^T b . theta); these are its effective (l, r) at the source.
    """
    zero = torch.zeros(d_source, device=W_left.device)
    _, pull = vjp(mlp_input_fn, zero)
    idx = torch.as_tensor(list(units), device=W_left.device)
    try:  # batched pullback; falls back to a loop if vmap can't trace the model
        from torch.func import vmap
        lefts = vmap(lambda a: pull(a)[0], chunk_size=256)(W_left[idx].float())      # chunked: 4608 units x a 4-block pullback
        rights = vmap(lambda b: pull(b)[0], chunk_size=256)(W_right[idx].float())
    except Exception:
        lefts = torch.stack([pull(W_left[h].float())[0] for h in idx])
        rights = torch.stack([pull(W_right[h].float())[0] for h in idx])
    return lefts, rights


def neuron_readoff_candidates(span: Span, lefts, rights, units, layer_of_units,
                              pr_layers: Sequence[int]):
    """Score each neuron *as if it were a DCT factor*, with no optimization.

    For each unit: l, r = normalized effective read directions; u = the
    normalized output mixed Hessian (the best u for that (l, r)).
    Returns energy (u.H)^2 and PR over pr_layers for each.
    """
    rows = []
    for i, h in enumerate(units):
        l = torch.nn.functional.normalize(lefts[i], dim=0)
        r = torch.nn.functional.normalize(rights[i], dim=0)
        H = mixed_hessian(lambda t: span(t, None)[0], l, r).float()
        u = torch.nn.functional.normalize(H, dim=0)
        resp = hidden_response(span, l, r, pr_layers)
        c = torch.cat([resp[L] for L in pr_layers])
        rows.append({"layer": layer_of_units, "unit": int(h), "energy": float((u @ H) ** 2),
                     "pr": float(participation_ratio(c)), "u": u, "l": l, "r": r})
    return rows


def factor_alignment_to_units(l, r, lefts, rights) -> Tuple[int, float]:
    """Best unit and its alignment: max over units of the L/R-swap-symmetric
    product |cos(l,a)||cos(r,b)| vs |cos(l,b)||cos(r,a)|. Random-direction
    baseline is ~ 1/d for d-dim source space."""
    cos = lambda X, v: (torch.nn.functional.normalize(X, dim=1) @ torch.nn.functional.normalize(v, dim=0)).abs()
    s1 = cos(lefts, l) * cos(rights, r)
    s2 = cos(rights, l) * cos(lefts, r)
    s = torch.maximum(s1, s2)
    i = int(s.argmax())
    return i, float(s[i])


# ---------------------------------------------------------------- E3: ablation

def clean_cache(span: Span, d_source: int):
    """Clean activations for patching. Spans whose recorded `hidden` is
    position-averaged (the real model) must expose `span.clean_cache()` that
    returns {"hidden": {layer: full activations}, "attn": {layer: ...}}."""
    if hasattr(span, "clean_cache"):
        return span.clean_cache()
    zero = torch.zeros(d_source)
    with torch.no_grad():
        _, hidden = span(zero, None)
    return {"hidden": {k: v.detach() for k, v in hidden.items()}, "attn": {}}


def ablation_effect(make_span, contexts, u, l, r, alpha: float, beta: float,
                    build_spec: Callable[[Span], FreezeSpec]) -> Dict[str, float]:
    """Finite-size interaction (u-projected) with and without patching units to clean.

    Patching to clean is the finite analogue of freezing. Use held-out contexts.
    Returns mean fraction of the interaction removed.
    """
    fracs = []
    for c in contexts:
        span = make_span(c)
        d = l.shape[0]
        spec = build_spec(span)
        cache = clean_cache(span, d)
        spec.clean_hidden, spec.clean_attn = cache["hidden"], cache["attn"]
        with torch.no_grad():
            base = u @ finite_mixed_difference(lambda t: span(t, None)[0], l, r, alpha, beta)
            abl = u @ finite_mixed_difference(lambda t: span(t, spec)[0], l, r, alpha, beta)
        fracs.append(float(1 - abl / base) if abs(base) > 1e-12 else float('nan'))
    return {"mean": float(np.nanmean(fracs)), "median": float(np.nanmedian(fracs)),
            "per_context": fracs}


# ---------------------------------------------------------------- E4: stability

def _abscos(X, Y):
    X = torch.nn.functional.normalize(X, dim=0)
    Y = torch.nn.functional.normalize(Y, dim=0)
    return (X.T @ Y).abs()


def factor_similarity(A: Tuple[torch.Tensor, ...], B: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Pairwise similarity of factor triples (U, L, R), columns = factors.

    |cos u| * max(|cos l||cos r|, |cos l_A r_B||cos r_A l_B|)  (L/R swap-symmetric).
    """
    UA, LA, RA = A
    UB, LB, RB = B
    su = _abscos(UA, UB)
    straight = _abscos(LA, LB) * _abscos(RA, RB)
    swapped = _abscos(LA, RB) * _abscos(RA, LB)
    return su * torch.maximum(straight, swapped)


def match_factors(A, B) -> np.ndarray:
    """Hungarian-matched similarities between two dictionaries."""
    S = factor_similarity(A, B).cpu().numpy()
    rows, cols = linear_sum_assignment(-S)
    return S[rows, cols]


def random_match_null(d_source, d_target, n_factors, n_draws=20, seed=0) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda d: torch.randn(d, n_factors, generator=g)
    vals = [match_factors((rnd(d_target), rnd(d_source), rnd(d_source)),
                          (rnd(d_target), rnd(d_source), rnd(d_source))) for _ in range(n_draws)]
    return np.concatenate(vals)


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


# ---------------------------------------------------------------- span-based measures (added after the toy fits showed (l, r) is not identified)
# u.H[l, r] is symmetric in l <-> r (a mixed second derivative), so the score is a symmetric bilinear form in (l, r) and its maximisers
# over unit vectors form a continuum: for a single bilinear unit (a.n)(b.n), (l, r) = (a, b) and l = r = (a + b)/sqrt2 score identically.
# Pairwise cosines between (l, r) of two fits therefore understate agreement; these measures compare the 2-dim spans instead.

def _span_basis(l, r, tol=1e-6):
    """Orthonormal basis of span{l, r} (1 or 2 columns)."""
    Q, R = torch.linalg.qr(torch.stack([l.float(), r.float()], 1))
    k = int((R.diagonal().abs() > tol * R.diagonal().abs().max()).sum())
    return Q[:, :max(k, 1)]


def span_alignment(l, r, lefts, rights) -> Tuple[int, float, torch.Tensor]:
    """Mean squared canonical correlation between span{l, r} and each unit's span{a_h, b_h}; 1 = the factor's read span lies
    inside the unit's read span. Returns (best unit, its score, all scores)."""
    Q = _span_basis(l, r).to(lefts.device)                                          # [d, k]
    e1 = torch.nn.functional.normalize(lefts.float(), dim=1)
    b = rights.float(); e2 = b - (b * e1).sum(1, keepdim=True) * e1; e2 = torch.nn.functional.normalize(e2, dim=1)
    s = ((e1 @ Q).square() + (e2 @ Q).square()).sum(1) / Q.shape[1]
    i = int(s.argmax())
    return i, float(s[i]), s


def span_similarity(A, B) -> torch.Tensor:
    """|cos u| x mean squared canonical correlation between the (l, r) spans, for all factor pairs (columns = factors)."""
    UA, LA, RA = A; UB, LB, RB = B
    su = _abscos(UA, UB); n, m = LA.shape[1], LB.shape[1]
    S = torch.zeros(n, m)
    QA = [_span_basis(LA[:, i], RA[:, i]) for i in range(n)]; QB = [_span_basis(LB[:, j], RB[:, j]) for j in range(m)]
    for i in range(n):
        for j in range(m):
            k = min(QA[i].shape[1], QB[j].shape[1])
            S[i, j] = torch.linalg.svdvals(QA[i].T @ QB[j]).square()[:k].mean()
    return su.cpu() * S


def match_factors_span(A, B) -> np.ndarray:
    S = span_similarity(A, B).numpy()
    rows, cols = linear_sum_assignment(-S)
    return S[rows, cols]


def random_match_null_span(d_source, d_target, n_factors, n_draws=10, seed=0) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    rnd = lambda d: torch.randn(d, n_factors, generator=g)
    return np.concatenate([match_factors_span((rnd(d_target), rnd(d_source), rnd(d_source)), (rnd(d_target), rnd(d_source), rnd(d_source))) for _ in range(n_draws)])
