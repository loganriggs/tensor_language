#!/usr/bin/env python3
"""Component-level circuit protocol: module sweep -> unit sweep -> greedy minimal set -> joint DAS.

Why this exists. `resid:18` is the final residual site of the model, so recovery there is
`(w_answer - w_foil) . rms_norm(x)` -- patching it copies the logits, and DAS on it re-derives the
unembedding. Fifty of fifty screened behaviours score exactly 1.000 there. That number says
nothing about the circuit. The informative data are the COMPONENT effects: whole attention
modules, whole MLP modules, and individual heads, patched inside a real forward.

The protocol here, in the order the user set it:

  1. Whole-module interchange: every `attn:LL` and `mlp:LL` (36 forwards).
  2. Unit interchange: every head `attn:LL:head:HH` (162 forwards) plus the 18 MLPs as units.
  3. If no single unit carries the effect, GREEDY forward selection over a registered pool of
     the top units: at each step add the unit that most raises the JOINT recovery of the set,
     stop when the joint reaches a registered target, when the best gain drops below a
     registered floor, or at a registered size cap. The result is the smallest set found that
     reproduces the distributed effect, together with the whole curve.
  4. Joint DAS on that set: an orthonormal R over the CONCATENATED unit space (heads are 128-d
     pre-`c_proj` slices, MLPs are 1152-d outputs) is fitted so that patching only R R^T of the
     donor-minus-base difference matches the donor's margin -- through the real forward, since
     no closed form exists below the final residual. Rank is fixed and registered before fitting.

All recoveries are `kernel.signed_pairwise_donor_recovery` on the donor-oriented axis, and the
same-answer families (P, C) use `kernel.normalized_same_answer_effect` with the target
families' median native separation as scale, exactly as the producer does.

Instrument control (standing lesson: control the NEW code path). `forward_units` reimplements
the producer's forward so gradients can flow; `verify_against_producer()` checks (a) the unpatched
path reproduces `backend.native`, (b) a single-layer head set reproduces `backend.patched_heads`,
and (c) an MLP unit reproduces `backend.patched` -- all to 1e-4 -- before anything is measured.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import circuit_fast_screen_kernel as kernel
import circuit_fast_screen_producer as producer

N_LAYERS, N_HEADS, N_EMBD = 18, 9, 1152
HEAD_DIM = N_EMBD // N_HEADS


# ----------------------------------------------------------------------------- rows / batches

def rows_of(module, family):
    """Rows of one family; spec-authored modules key it `family`, older ones `transform_id`."""
    rows = module.build_rows()
    key = "family" if "family" in rows[0] else "transform_id"
    return [r for r in rows if r[key] == family]


def batch_of(rows, side):
    ids = "base_ids" if side == "base" else "donor_ids"
    ans = "base_answer_id" if side == "base" else "donor_answer_id"
    foil = "base_foil_id" if side == "base" else "donor_foil_id"
    pos = "base_semantic_position" if side == "base" else "donor_semantic_position"
    return producer.ModelBatch(
        row_ids=tuple(r["row_id"] for r in rows), side=side,
        token_rows=tuple(tuple(r[ids]) for r in rows),
        answer_ids=tuple(r[ans] for r in rows), foil_ids=tuple(r[foil] for r in rows),
        semantic_positions=tuple(r[pos] for r in rows))


# ----------------------------------------------------------------------------- units

def unit_dim(unit):
    return HEAD_DIM if ":head:" in unit else N_EMBD


def unit_layer(unit):
    return int(unit.split(":")[1])


def all_head_units():
    return [f"attn:{l:02d}:head:{h:02d}" for l in range(N_LAYERS) for h in range(N_HEADS)]


def all_mlp_units():
    return [f"mlp:{l:02d}" for l in range(N_LAYERS)]


# ----------------------------------------------------------------------------- forward

def forward_units(backend, batch, *, units=(), donor_cache=None, base_cache=None, q=None,
                  grad=False):
    """The producer's exact forward with unit interventions at each row's semantic position.

    units        unit ids to intervene on, in a fixed order (the order defines the concatenation)
    q            None  -> EXACT replacement of each unit by its donor value (producer semantics)
                 (D,r) -> SUBSPACE patch: live + slice_u( q q^T (donor - base) ), where the
                          donor-minus-base difference is the cached native difference over the
                          concatenated unit space, D = sum of unit dims. With one layer and q
                          spanning everything this equals exact replacement.
    Returns (answer_foil pairs as a (n,2) tensor, logits-free). Gradients flow to q iff grad.
    """
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    n = len(batch.row_ids)
    positions = list(batch.semantic_positions)
    by_layer: dict[int, list[str]] = {}
    for u in units:
        by_layer.setdefault(unit_layer(u), []).append(u)

    projected = None
    if q is not None:
        offsets, off = {}, 0
        for u in units:
            offsets[u] = (off, off + unit_dim(u)); off += unit_dim(u)
        delta = torch.stack([
            torch.cat([torch.as_tensor(donor_cache[(rid, u)]).float()
                       - torch.as_tensor(base_cache[(rid, u)]).float() for u in units])
            for rid in batch.row_ids]).to(backend.device)                   # (n, D)
        projected = (delta @ q) @ q.T                                         # (n, D)

    def donor_value(u):
        return torch.stack([torch.as_tensor(donor_cache[(rid, u)]) for rid in batch.row_ids]
                           ).to(backend.device)

    def apply(value, layer, kind):
        """kind 'heads' -> value is the c_proj input (n,T,1152); 'mlp' -> the MLP output."""
        here = [u for u in by_layer.get(layer, []) if ((":head:" in u) == (kind == "heads"))]
        if not here:
            return value
        changed = value.clone()
        idx = torch.arange(n, device=value.device)
        pos = torch.tensor(positions, device=value.device)
        for u in here:
            if kind == "heads":
                h = int(u.rsplit(":", 1)[1]); s, e = h * HEAD_DIM, (h + 1) * HEAD_DIM
            else:
                s, e = 0, N_EMBD
            if q is None:
                changed[idx, pos, s:e] = donor_value(u).to(value.dtype)
            else:
                o0, o1 = offsets[u]
                changed[idx, pos, s:e] = value[idx, pos, s:e] + projected[:, o0:o1].to(value.dtype)
        return changed

    with torch.set_grad_enabled(grad):
        x = F.rms_norm(model.transformer.wte(tokens), (N_EMBD,))
        x0, v1 = x, None
        for layer, block in enumerate(model.transformer.h):
            live = block.lambdas[0] * x + block.lambdas[1] * x0

            def c_proj_pre(_module, arguments, layer=layer):
                return (apply(arguments[0], layer, "heads"),) + tuple(arguments[1:])

            handle = block.attn.c_proj.register_forward_pre_hook(c_proj_pre)
            try:
                attention, v1 = block.attn(F.rms_norm(live, (N_EMBD,)), v1)
            finally:
                handle.remove()
            x = live + attention
            mlp = block.mlp(F.rms_norm(x, (N_EMBD,)))
            mlp = apply(mlp, layer, "mlp")
            x = x + mlp
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (N_EMBD,))) / 30.0)
        idx = torch.arange(n, device=logits.device)
        last = torch.tensor([l - 1 for l in lengths], device=logits.device)
        a = logits[idx, last, torch.tensor(batch.answer_ids, device=logits.device)].float()
        f = logits[idx, last, torch.tensor(batch.foil_ids, device=logits.device)].float()
        return torch.stack([a, f], dim=1)


def verify_against_producer(backend, rows, *, layer, heads, mlp_layer, tolerance=1e-4):
    """Instrument control: the new forward must reproduce the producer on three known paths."""
    torch = backend.torch
    base_batch, donor_batch = batch_of(rows, "base"), batch_of(rows, "donor")
    donor_cache = backend.native(donor_batch, capture=True).captured
    native = torch.tensor(backend.native(base_batch, capture=False).answer_foil)
    mine = forward_units(backend, base_batch).cpu()
    e_native = float((mine - native).abs().max())
    ref = torch.tensor(backend.patched_heads(base_batch, layer=layer, heads=tuple(heads),
                                             donor_cache=donor_cache).answer_foil)
    mine = forward_units(backend, base_batch,
                         units=[f"attn:{layer:02d}:head:{h:02d}" for h in heads],
                         donor_cache=donor_cache).cpu()
    e_heads = float((mine - ref).abs().max())
    site = kernel.SiteRef(site_id=f"mlp:{mlp_layer:02d}", evidence_kind="residual")
    ref = torch.tensor(backend.patched(base_batch, site=site, donor_cache=donor_cache).answer_foil)
    mine = forward_units(backend, base_batch, units=[f"mlp:{mlp_layer:02d}"],
                         donor_cache=donor_cache).cpu()
    e_mlp = float((mine - ref).abs().max())
    ok = max(e_native, e_heads, e_mlp) <= tolerance
    return {"native_max_abs_error": e_native, "heads_max_abs_error": e_heads,
            "mlp_max_abs_error": e_mlp, "tolerance": tolerance, "passed": ok}


# ----------------------------------------------------------------------------- measures

@dataclass
class Prepared:
    rows: list
    base_batch: object
    donor_batch: object
    base_cache: Mapping
    donor_cache: Mapping
    base_axis: list        # base value on the donor-oriented axis  (= -margin of base run)
    donor_axis: list       # donor value on that axis
    answer_changes: bool


def prepare(backend, rows):
    base_batch, donor_batch = batch_of(rows, "base"), batch_of(rows, "donor")
    base_out = backend.native(base_batch, capture=True)
    donor_out = backend.native(donor_batch, capture=True)
    changes = bool(rows[0].get("answer_changes", rows[0]["base_answer_id"] != rows[0]["donor_answer_id"]))
    return Prepared(rows, base_batch, donor_batch, base_out.captured, donor_out.captured,
                    [-(a - f) for a, f in base_out.answer_foil],
                    [a - f for a, f in donor_out.answer_foil], changes)


def patched_axis(backend, prep, units, q=None):
    out = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                        base_cache=prep.base_cache, q=q)
    return [-(float(a) - float(f)) for a, f in out.tolist()]


def recovery(prep, patched):
    """Mean signed recovery toward the donor (answer-changing families only)."""
    vals = [kernel.signed_pairwise_donor_recovery(b, d, p)
            for b, d, p in zip(prep.base_axis, prep.donor_axis, patched)]
    return sum(vals) / len(vals)


def same_answer_effect(prep, patched, scale):
    """Mean |movement| in units of the target families' native separation (P and C)."""
    vals = [kernel.normalized_same_answer_effect(b, p, scale)
            for b, p in zip(prep.base_axis, patched)]
    return sum(vals) / len(vals)


def target_scale(prep):
    return float(statistics.median(abs(d - b) for b, d in zip(prep.base_axis, prep.donor_axis)))


def module_sweep(backend, prep):
    """Whole-module effects for every attn and mlp block (producer path, not the new forward)."""
    out = {}
    for layer in range(N_LAYERS):
        for kind in ("attn", "mlp"):
            site = kernel.SiteRef(site_id=f"{kind}:{layer:02d}", evidence_kind="residual")
            res = backend.patched(prep.base_batch, site=site, donor_cache=prep.donor_cache)
            out[site.site_id] = recovery(prep, [-(a - f) for a, f in res.answer_foil])
    return out


def unit_sweep(backend, prep, units):
    return {u: recovery(prep, patched_axis(backend, prep, [u])) for u in units}


# ----------------------------------------------------------------------------- greedy

def greedy_select(evaluate: Callable[[Sequence[str]], float], pool: Sequence[str], *,
                  target: float, min_gain: float, max_units: int):
    """Forward selection maximising the joint effect of the chosen set.

    Every step records the joint score of EVERY candidate so the curve is auditable. A step
    whose best gain is below `min_gain` is recorded as rejected and ends the search.
    """
    chosen, steps, current, remaining = [], [], 0.0, list(pool)
    while remaining and len(chosen) < max_units:
        scores = {u: evaluate(chosen + [u]) for u in remaining}
        best, score = max(scores.items(), key=lambda kv: kv[1])
        gain = score - current
        accepted = gain >= min_gain or not chosen
        steps.append({"candidate": best, "joint": score, "gain": gain, "accepted": accepted,
                      "all_candidates": scores})
        if not accepted:
            break
        chosen.append(best); current = score; remaining.remove(best)
        if score >= target:
            break
    return {"chosen": chosen, "joint": current, "reached_target": current >= target,
            "steps": steps}


# ----------------------------------------------------------------------------- joint DAS

def fit_joint_subspace(backend, prep, units, *, rank, steps=200, lr=0.05, seed=0):
    """Orthonormal R over the concatenated unit space, fitted through the real forward.

    Objective: match the donor's margin on the donor axis (not maximise it -- the head is
    soft-capped, and maximising overshot to 2.2 in an earlier resid:18 run).
    """
    torch = backend.torch
    for p in backend.model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    dim = sum(unit_dim(u) for u in units)
    raw = (torch.randn(dim, rank, device=backend.device) * 0.02).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    target = torch.tensor(prep.donor_axis, device=backend.device)
    history = []
    for step in range(steps):
        opt.zero_grad()
        q, _ = torch.linalg.qr(raw)
        out = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                            base_cache=prep.base_cache, q=q, grad=True)
        axis = -(out[:, 0] - out[:, 1])
        loss = ((axis - target) ** 2).mean()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, float(loss)))
    with torch.no_grad():
        q, _ = torch.linalg.qr(raw)
    return q.detach(), history
