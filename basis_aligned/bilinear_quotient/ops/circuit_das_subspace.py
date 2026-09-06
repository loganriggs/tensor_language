#!/usr/bin/env python3
"""Distributed Alignment Search on a localized circuit's selected site.

Interchange localizes a behaviour to a whole residual-stream site: at `resid:18` the variable is
recovered at 1.000 while the best attention block reaches ~0.3 and the best MLP ~0.2. That says
where the variable is readable and nothing about what carries it. This searches for the SUBSPACE.

Method. Capture the site's activations on the base and donor sides, then learn an orthonormal
R (n_embd x k, k fixed in advance) and patch only the projection:

    x_patched = x_base + R R^T (x_donor - x_base)

optimizing R to move the prediction toward the donor's answer. A subspace that carries the
variable will transfer it; one that does not will leave the answer where it was.

Why this is exact and cheap. `resid:18` is the FINAL residual site of an 18-layer model, so the
map from it to the logits is only the final norm and unembedding —

    logits = 30 * tanh(lm_head(rms_norm(x)) / 30)

— which is three lines, fully differentiable, and needs no transformer forward inside the
optimization loop. `verify_head()` reproduces the producer's own native answer/foil values from
captured activations before any fitting happens; if that check fails, nothing else here is
trustworthy (standing lesson 6: check the instrument against a known-good case).

Scope. This is causal localization, evaluated by interchange on held-out rows. It is not
activation reconstruction and there is no reconstruction objective anywhere in it.

Protocol (ops/README.md, "DAS follow-up on localized circuits"): rank is fixed in advance and
registered with the prediction; a null is NOT permission to raise the rank; the subspace must
pass all four hypotheses, not only A1.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Sequence

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer

SITE_PREFIX = "resid:"


@dataclass
class Captured:
    """Base/donor activations at the site, plus the answer and foil ids, per family."""
    family: str
    base: object          # (rows, n_embd)
    donor: object
    answer_ids: object    # donor-side answer, the direction a patch should move toward
    base_answer_ids: object
    foil_ids: object


def _final_positions(rows, torch, device):
    return torch.tensor([len(r) - 1 for r in rows], dtype=torch.long, device=device)


def head_logits(backend, x):
    """The model's exact final head, differentiable. Mirrors producer._forward's tail."""
    torch, F, model = backend.torch, backend.F, backend.model
    return 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (model.config.n_embd,))) / 30.0)


def verify_head(backend, rows, capture_site, tolerance=1e-3):
    """Control: reproduce the producer's native answer/foil values from captured activations.

    Returns (ok, max_abs_difference). Run this before trusting any fitted subspace.
    """
    torch = backend.torch
    batch = _batch(backend, rows, side="base")
    out = backend.native(batch, capture=True)
    # producer._save keys on (row_id, site_id) and stores the value AT the semantic position,
    # so each entry is already a single (n_embd,) vector.
    acts = torch.stack([torch.as_tensor(out.captured[(r["row_id"], capture_site)]) for r in rows])
    logits = head_logits(backend, acts.to(backend.device).float())
    worst = 0.0
    for i, r in enumerate(rows):
        a = float(logits[i, r["base_answer_id"]])
        f = float(logits[i, r["base_foil_id"]])
        worst = max(worst, abs(a - out.answer_foil[i][0]), abs(f - out.answer_foil[i][1]))
    return worst <= tolerance, worst


def _batch(backend, rows, *, side):
    ids = "base_ids" if side == "base" else "donor_ids"
    ans = "base_answer_id" if side == "base" else "donor_answer_id"
    foil = "base_foil_id" if side == "base" else "donor_foil_id"
    pos = "base_semantic_position" if side == "base" else "donor_semantic_position"
    return producer.ModelBatch(
        row_ids=tuple(r["row_id"] for r in rows),
        side=side,
        token_rows=tuple(tuple(r[ids]) for r in rows),
        answer_ids=tuple(r[ans] for r in rows),
        foil_ids=tuple(r[foil] for r in rows),
        semantic_positions=tuple(r[pos] for r in rows),
    )


def capture_site(backend, rows, site_id):
    """Capture base and donor activations at `site_id`, at each row's final input token."""
    torch = backend.torch
    site = kernel.SiteRef(site_id=site_id, evidence_kind="residual")
    out = {}
    for side in ("base", "donor"):
        batch = _batch(backend, rows, side=side)
        result = backend.native(batch, capture=True)
        out[side] = torch.stack(
            [torch.as_tensor(result.captured[(r["row_id"], site_id)]) for r in rows]
        ).to(backend.device).float()
    return out["base"], out["donor"], site


def fit_subspace(backend, base, donor, answer_ids, foil_ids, *, rank, steps=300, lr=0.05, seed=0):
    """Learn an orthonormal R so that patching only R R^T moves the answer toward the donor's.

    Rank is a required argument with no default: it must be chosen and registered before fitting.
    """
    torch = backend.torch
    torch.manual_seed(seed)
    n_embd = base.shape[1]
    raw = torch.randn(n_embd, rank, device=backend.device, dtype=base.dtype) * 0.02
    raw.requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=lr)
    delta = (donor - base).detach()
    a_idx = torch.as_tensor(answer_ids, device=backend.device, dtype=torch.long)
    f_idx = torch.as_tensor(foil_ids, device=backend.device, dtype=torch.long)
    n = base.shape[0]
    arange = torch.arange(n, device=backend.device)
    # Target the donor's margin rather than maximising the margin. Maximising is wrong twice
    # over: it overshoots (a first run reached recovery 2.208, i.e. past the donor), and the
    # model's head is logit-soft-capped -- `30*tanh(logits/30)`, from tt_model.py:260 -- so
    # climbing toward the cap flattens the gradient and buys progressively less real signal
    # while still moving the direction. Matching a point inside the cap fixes both.
    with torch.no_grad():
        donor_logits = head_logits(backend, donor)
        target_margin = (donor_logits[arange, a_idx] - donor_logits[arange, f_idx]).detach()
    for _ in range(steps):
        optimizer.zero_grad()
        q, _ = torch.linalg.qr(raw)                       # orthonormal basis, differentiable
        patched = base + (delta @ q) @ q.T
        logits = head_logits(backend, patched)
        margin = logits[arange, a_idx] - logits[arange, f_idx]
        ((margin - target_margin) ** 2).mean().backward()
        optimizer.step()
    with torch.no_grad():
        q, _ = torch.linalg.qr(raw)
    return q.detach()


def target_scale(backend, base, donor, answer_ids, foil_ids):
    """The target families' median native separation, as `producer` computes it (line 554).

    This is the denominator the kernel uses for same-answer families, and using anything else
    makes P and C incomparable with A1.
    """
    import statistics
    torch = backend.torch
    with torch.no_grad():
        a_idx = torch.as_tensor(answer_ids, device=backend.device, dtype=torch.long)
        f_idx = torch.as_tensor(foil_ids, device=backend.device, dtype=torch.long)
        arange = torch.arange(base.shape[0], device=backend.device)
        def margin(x):
            lg = head_logits(backend, x)
            return lg[arange, a_idx] - lg[arange, f_idx]
        return float(statistics.median([abs(v) for v in (margin(donor) - margin(base)).tolist()]))


def subspace_same_answer_effect(backend, base, donor, q, answer_ids, foil_ids, scale):
    """Disturbance measure for families whose two sides SHARE an answer (P, and same-answer C).

    A first run divided these by `(m_donor - m_base)`, which for such a family is legitimately
    near zero, and reported P at 24.678. The kernel handles this with
    `normalized_same_answer_effect` = |intervened - base| / a registered scale; this mirrors it.
    """
    torch = backend.torch
    with torch.no_grad():
        a_idx = torch.as_tensor(answer_ids, device=backend.device, dtype=torch.long)
        f_idx = torch.as_tensor(foil_ids, device=backend.device, dtype=torch.long)
        arange = torch.arange(base.shape[0], device=backend.device)
        def margin(x):
            lg = head_logits(backend, x)
            return lg[arange, a_idx] - lg[arange, f_idx]
        patched = base + ((donor - base) @ q) @ q.T
        effect = (margin(patched) - margin(base)).abs() / scale
        return float(effect.mean()), int(effect.numel())


def subspace_recovery(backend, base, donor, q, answer_ids, foil_ids):
    """Interchange recovery through the subspace alone, for ANSWER-CHANGING families only."""
    torch = backend.torch
    with torch.no_grad():
        a_idx = torch.as_tensor(answer_ids, device=backend.device, dtype=torch.long)
        f_idx = torch.as_tensor(foil_ids, device=backend.device, dtype=torch.long)
        arange = torch.arange(base.shape[0], device=backend.device)
        def margin(x):
            lg = head_logits(backend, x)
            return lg[arange, a_idx] - lg[arange, f_idx]
        m_base, m_donor = margin(base), margin(donor)
        patched = base + ((donor - base) @ q) @ q.T
        m_patched = margin(patched)
        denominator = m_donor - m_base
        keep = denominator.abs() > 1e-6
        recovery = (m_patched - m_base)[keep] / denominator[keep]
        return float(recovery.mean()), float(recovery.abs().mean()), int(keep.sum())
