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

MLP_HIDDEN = 4 * N_EMBD


def block_key(unit):
    """(layer, kind) -- the hook at which a unit is intervened; units sharing it can share a q."""
    return (unit_layer(unit), "heads" if ":head:" in unit else "neurons" if ":neuron:" in unit else "mlp")


def blocks_of(units):
    """Ordered {block_key: [units in the given order]}."""
    out = {}
    for u in units:
        out.setdefault(block_key(u), []).append(u)
    return out


def unit_dim(unit):
    if ":head:" in unit:
        return HEAD_DIM
    if ":neuron:" in unit:
        return 1
    return N_EMBD


def hidden_key(layer):
    """Cache key for an MLP's hidden vector at the semantic position.

    bilin18's MLP is `Bilinear`: hidden = Left(x) * Right(x) (4608-d, no gate, no nonlinearity),
    output = Down(hidden) + bias. A "neuron" here is one bilinear product term l_j * r_j -- the
    model's own basis for the module, no rotation involved."""
    return f"mlp:{layer:02d}:hidden"


def unit_layer(unit):
    return int(unit.split(":")[1])


def all_head_units():
    return [f"attn:{l:02d}:head:{h:02d}" for l in range(N_LAYERS) for h in range(N_HEADS)]


def all_mlp_units():
    return [f"mlp:{l:02d}" for l in range(N_LAYERS)]


# ----------------------------------------------------------------------------- forward

def forward_units(backend, batch, *, units=(), donor_cache=None, base_cache=None, q=None,
                  grad=False, complement=False, capture_hidden=None, neuron_per_row=None,
                  return_logits=False, capture_resid=None, resid_add=None):
    """The producer's exact forward with unit interventions at each row's semantic position.

    units        unit ids to intervene on, in a fixed order (the order defines the concatenation)
    q            None  -> EXACT replacement of each unit by its donor value (producer semantics)
                 (D,r) -> JOINT-CACHED subspace patch: live + slice_u( q q^T (donor - base) ), the
                          cached native difference over the concatenated unit space, D = sum of
                          unit dims. With one layer and q spanning everything this equals exact
                          replacement. ACROSS layers it does not: the later layer's live value
                          already carries the earlier patch, and the cached offset is then a
                          donor-derived steering vector, not a swap of the live state (red-team
                          finding 2026-09-06; the discrepancy at full rank was ~2% on the head
                          sets, but the semantics are those of activation addition).
                 dict  -> BLOCK-LIVE subspace patch, the standard multi-site interchange: one
                          orthonormal (D_blk, r_blk) per (layer, kind) block, keyed as
                          block_key(unit); at each hook, live + q q^T (donor - LIVE) over that
                          block's units. At full rank it equals exact replacement exactly, in
                          every configuration. Total rank = sum of the block ranks.
    complement    with q: patch (I - q q^T)(donor - base) instead -- swap everything EXCEPT the
                  learned axes (the dormant-direction test of Makelov, Lange & Nanda 2023).
    capture_hidden a dict to receive the MLP hidden (Left*Right product) vector at the semantic
                  position for every layer, keyed (row_id, hidden_key(layer)).
    neuron_per_row (layer, [neuron index per row]) -- a DIFFERENT single neuron per row, for the
                  replicated-batch neuron sweep; needs donor_cache[(rid, hidden_key(layer))].
    Units of the form mlp:LL:neuron:J swap single hidden units of that MLP (exact, on-distribution).
    Returns answer/foil values as an (n,2) tensor. Gradients flow to q iff grad.
    return_logits=True also returns the full final-position logits (n, V) as a second value (Tier 2
    characterization: competitor tokens, log-prob shifts, off-target KL).
    capture_resid a dict to receive the PRE-rms_norm residual entering each MLP at the row's semantic
                  position, keyed (row_id, layer) -- v34 rms_norm second-order expansion.
    resid_add     {layer: (n, N_EMBD) tensor} ADDED to the residual at the row's semantic position right
                  after that layer's attention (i.e. at the point capture_resid reads) -- v35 replay of a
                  measured residual delta; it propagates to every later layer like any residual write.
    """
    torch, F, model = backend.torch, backend.F, backend.model
    tokens, lengths = backend._tensor_batch(batch)
    n = len(batch.row_ids)
    positions = list(batch.semantic_positions)
    by_layer: dict[int, list[str]] = {}
    for u in units:
        by_layer.setdefault(unit_layer(u), []).append(u)

    projected = None
    block_q = q if isinstance(q, dict) else None
    if block_q is not None:
        q = None
    if q is not None:
        offsets, off = {}, 0
        for u in units:
            offsets[u] = (off, off + unit_dim(u)); off += unit_dim(u)
        def cached(cache, rid, u):
            if ":neuron:" in u:
                j = int(u.rsplit(":", 1)[1])
                return torch.as_tensor(cache[(rid, hidden_key(unit_layer(u)))])[j:j + 1]
            return torch.as_tensor(cache[(rid, u)])
        delta = torch.stack([
            torch.cat([cached(donor_cache, rid, u).float() - cached(base_cache, rid, u).float()
                       for u in units])
            for rid in batch.row_ids]).to(backend.device)                   # (n, D)
        projected = (delta @ q) @ q.T                                         # (n, D)
        if complement:
            projected = delta - projected

    def donor_value(u):
        return torch.stack([torch.as_tensor(donor_cache[(rid, u)]) for rid in batch.row_ids]
                           ).to(backend.device)

    def kind_of(u):
        return "heads" if ":head:" in u else "neurons" if ":neuron:" in u else "mlp"

    def apply(value, layer, kind):
        """'heads' -> attn c_proj input (n,T,1152); 'neurons' -> mlp Down input (n,T,4608);
        'mlp' -> the MLP output (n,T,1152)."""
        here = [u for u in by_layer.get(layer, []) if kind_of(u) == kind]
        idx = torch.arange(n, device=value.device)
        pos = torch.tensor(positions, device=value.device)
        if kind == "neurons":
            if capture_hidden is not None:
                for i, rid in enumerate(batch.row_ids):
                    capture_hidden[(rid, hidden_key(layer))] = value[i, positions[i]].detach().clone()
            if neuron_per_row is not None and neuron_per_row[0] == layer:
                changed = value.clone()
                j = torch.tensor(neuron_per_row[1], device=value.device)
                donor_hidden = torch.stack([torch.as_tensor(donor_cache[(rid, hidden_key(layer))])
                                            for rid in batch.row_ids]).to(value.device)
                changed[idx, pos, j] = donor_hidden[idx, j].to(value.dtype)
                return changed
        if not here:
            return value
        changed = value.clone()

        def span(u):
            if kind == "heads":
                h = int(u.rsplit(":", 1)[1]); return h * HEAD_DIM, (h + 1) * HEAD_DIM
            if kind == "neurons":
                j = int(u.rsplit(":", 1)[1]); return j, j + 1
            return 0, N_EMBD

        if block_q is not None:
            qb = block_q[(layer, kind)]
            spans = [span(u) for u in here]
            live_blk = torch.cat([value[idx, pos, s:e] for s, e in spans], dim=1).float()   # (n, D_blk)
            if kind == "neurons":
                donor_blk = torch.cat([torch.stack([torch.as_tensor(donor_cache[(rid, hidden_key(layer))])[s:e]
                                                    for rid in batch.row_ids]) for s, e in spans], dim=1)
            else:
                donor_blk = torch.cat([donor_value(u) for u in here], dim=1)
            d_live = donor_blk.to(value.device).float() - live_blk
            proj = (d_live @ qb) @ qb.T
            if complement:
                proj = d_live - proj
            o = 0
            for (s, e) in spans:
                changed[idx, pos, s:e] = value[idx, pos, s:e] + proj[:, o:o + (e - s)].to(value.dtype)
                o += e - s
            return changed

        for u in here:
            s, e = span(u)
            if q is None:
                if kind == "neurons":
                    changed[idx, pos, s:e] = torch.stack(
                        [torch.as_tensor(donor_cache[(rid, hidden_key(layer))])[s:e]
                         for rid in batch.row_ids]).to(value.device, value.dtype)
                else:
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

            def mlp_proj_pre(_module, arguments, layer=layer):
                return (apply(arguments[0], layer, "neurons"),) + tuple(arguments[1:])

            if resid_add is not None and layer in resid_add:
                x = x.clone()
                x[torch.arange(n, device=x.device), torch.tensor(positions, device=x.device)] += \
                    resid_add[layer].to(x.device, x.dtype)
            if capture_resid is not None:
                for i, rid in enumerate(batch.row_ids):
                    capture_resid[(rid, layer)] = x[i, positions[i]].detach().float().clone()
            handle = block.mlp.Down.register_forward_pre_hook(mlp_proj_pre)
            try:
                mlp = block.mlp(F.rms_norm(x, (N_EMBD,)))
            finally:
                handle.remove()
            mlp = apply(mlp, layer, "mlp")
            x = x + mlp
        logits = 30.0 * torch.tanh(model.lm_head(F.rms_norm(x, (N_EMBD,))) / 30.0)
        idx = torch.arange(n, device=logits.device)
        last = torch.tensor([l - 1 for l in lengths], device=logits.device)
        a = logits[idx, last, torch.tensor(batch.answer_ids, device=logits.device)].float()
        f = logits[idx, last, torch.tensor(batch.foil_ids, device=logits.device)].float()
        af = torch.stack([a, f], dim=1)
        return (af, logits[idx, last].float()) if return_logits else af


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
    dropped: int = 0


def prepare(backend, rows, *, hidden=False, valid_only=False):
    """Native runs on both sides. hidden=True also caches every MLP's 4608-d hidden vector.

    valid_only=True drops rows whose donor does not beat the base on the donor's answer axis
    (denominator <= 1e-6): the kernel refuses them, and a design with a capability failure on the
    donor side (possessive animate_attractor) has such rows. `prep.dropped` records how many.
    """
    base_batch, donor_batch = batch_of(rows, "base"), batch_of(rows, "donor")
    base_out = backend.native(base_batch, capture=True)
    donor_out = backend.native(donor_batch, capture=True)
    dropped = 0
    if valid_only:
        keep = [i for i, ((ba, bf), (da, df)) in enumerate(zip(base_out.answer_foil, donor_out.answer_foil))
                if (da - df) - (-(ba - bf)) > 1e-6]
        dropped = len(rows) - len(keep)
        if dropped:
            prep = prepare(backend, [rows[i] for i in keep], hidden=hidden)
            prep.dropped = dropped
            return prep
    base_cache, donor_cache = dict(base_out.captured), dict(donor_out.captured)
    if hidden:
        forward_units(backend, base_batch, capture_hidden=base_cache)
        forward_units(backend, donor_batch, capture_hidden=donor_cache)
    changes = bool(rows[0].get("answer_changes", rows[0]["base_answer_id"] != rows[0]["donor_answer_id"]))
    prep = Prepared(rows, base_batch, donor_batch, base_cache, donor_cache,
                    [-(a - f) for a, f in base_out.answer_foil],
                    [a - f for a, f in donor_out.answer_foil], changes)
    prep.dropped = dropped
    return prep


def patched_axis(backend, prep, units, q=None, complement=False):
    out = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                        base_cache=prep.base_cache, q=q, complement=complement)
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

def fit_joint_subspace(backend, prep, units, *, rank, steps=200, lr=0.05, seed=0,
                       target="exact_set", complement_weight=0.0):
    """Orthonormal R over the concatenated unit space, fitted through the real forward.

    target "exact_set" (default): match the margin the EXACT interchange of `units` produces.
        The subspace is then asked to reproduce what the component set itself does, so its
        recovery is bounded by the set's, and a fraction_of_exact near 1 means the set's effect
        lives in that subspace.
    target "donor": match the donor's native margin. WRONG for a partial set -- v1 of the
        protocol used it, and because a 2-4 unit set only reaches ~0.55 of the donor, the
        optimizer was forced past the set's own effect into a steering direction: held-out
        recoveries of 1.5-4.2 and P/C effects of 0.4-0.6 (`unit_greedy_protocol_v1`). Kept only
        so that failure can be reproduced.
    Neither maximises the margin: the head is soft-capped and maximising overshot to 2.2 in an
    earlier resid:18 run.
    complement_weight > 0 adds the Makelov-style constraint as a loss term: swapping everything
        EXCEPT the learned axes must leave the base margin where it was. A dormant axis cannot
        satisfy both terms -- if it steers when swapped, the real effect is still in the complement.
        The complement term at the end of training is reported in the history; a term that will not
        go to zero at rank k says the set's effect needs more than k directions.
    """
    torch = backend.torch
    for p in backend.model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    dim = sum(unit_dim(u) for u in units)
    raw = (torch.randn(dim, rank, device=backend.device) * 0.02).requires_grad_(True)
    opt = torch.optim.Adam([raw], lr=lr)
    if target == "exact_set":
        target_axis = patched_axis(backend, prep, units)
    elif target == "donor":
        target_axis = prep.donor_axis
    else:
        raise ValueError(target)
    target = torch.tensor(target_axis, device=backend.device)
    base_target = torch.tensor(prep.base_axis, device=backend.device)
    history = []
    for step in range(steps):
        opt.zero_grad()
        q, _ = torch.linalg.qr(raw)
        out = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                            base_cache=prep.base_cache, q=q, grad=True)
        axis = -(out[:, 0] - out[:, 1])
        match = ((axis - target) ** 2).mean()
        loss = match
        inert = None
        if complement_weight > 0:
            comp = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                                 base_cache=prep.base_cache, q=q, grad=True, complement=True)
            inert = ((-(comp[:, 0] - comp[:, 1]) - base_target) ** 2).mean()
            loss = match + complement_weight * inert
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, float(match.detach()),
                            None if inert is None else float(inert.detach())))
    with torch.no_grad():
        q, _ = torch.linalg.qr(raw)
    return q.detach(), history


def random_subspace(backend, units, *, rank, seed=1):
    """A random orthonormal R of the same shape: the baseline any fitted direction must beat."""
    torch = backend.torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    dim = sum(unit_dim(u) for u in units)
    q, _ = torch.linalg.qr(torch.randn(dim, rank, generator=gen))
    return q.to(backend.device)


def diff_in_means_direction(backend, prep, units):
    """The unit-norm mean of (donor - base) over the concatenated unit space: a rank-1 direction
    with NO search freedom, so it cannot go looking for dormant axes. The primary estimate; a
    learned direction that beats it by a wide margin is exploiting something it should not."""
    torch = backend.torch

    def cached(cache, rid, u):
        if ":neuron:" in u:
            j = int(u.rsplit(":", 1)[1])
            return torch.as_tensor(cache[(rid, hidden_key(unit_layer(u)))])[j:j + 1]
        return torch.as_tensor(cache[(rid, u)])
    delta = torch.stack([
        torch.cat([cached(prep.donor_cache, rid, u).float() - cached(prep.base_cache, rid, u).float()
                   for u in units]) for rid in prep.base_batch.row_ids]).to(backend.device)
    d = (delta * _orientation(delta)[:, None]).mean(0)
    if float(d.norm()) < 1e-3:
        raise ValueError("diff-in-means cancelled: rows are not direction-pure and carry no direction_id")
    return (d / d.norm()).unsqueeze(1)


def neuron_sweep(backend, prep, layer, *, replicas=16):
    """Exact single-neuron interchange for all 4608 hidden units of one MLP.

    Rows are replicated `replicas` times per forward and each replica group swaps a different
    neuron, so the sweep costs 4608/replicas forwards of (rows x replicas) sequences.
    """
    rows = prep.rows
    n = len(rows)
    rep_rows = [r for _ in range(replicas) for r in rows]
    rep_batch = batch_of(rep_rows, "base")
    rep_base = [v for _ in range(replicas) for v in prep.base_axis]
    rep_donor = [v for _ in range(replicas) for v in prep.donor_axis]
    out = {}
    for start in range(0, MLP_HIDDEN, replicas):
        neurons = list(range(start, min(start + replicas, MLP_HIDDEN)))
        k = len(neurons)
        per_row = [neurons[g] for g in range(k) for _ in range(n)]
        batch = rep_batch if k == replicas else batch_of(rep_rows[:k * n], "base")
        res = forward_units(backend, batch, donor_cache=prep.donor_cache,
                            neuron_per_row=(layer, per_row))
        axis = [-(float(a) - float(f)) for a, f in res.tolist()]
        for g, j in enumerate(neurons):
            sl = slice(g * n, (g + 1) * n)
            vals = [kernel.signed_pairwise_donor_recovery(b, d, p)
                    for b, d, p in zip(rep_base[sl], rep_donor[sl], axis[sl])]
            out[f"mlp:{layer:02d}:neuron:{j:04d}"] = sum(vals) / len(vals)
    return out


# ----------------------------------------------------------------------------- the standard battery
# Shared by every runner from v5 on. Timings (bilin18, one H100-class GPU, 32 rows per family):
#   prepare (two native forwards)            ~0.1 s      module_sweep (36 producer patches) ~1 s
#   head_sweep (162 heads)                   ~4 s        greedy (pool 12, <= 6 steps)       ~2 s
#   direction_battery (4 forwards)           ~0.1 s      fit_joint_subspace (200 steps)     ~15 s
#   neuron_sweep (4608 terms, 16 replicas)   ~30 s       whole v4 battery, 7 sets + 2 MLPs  167 s

def greedy_heads(backend, prep, *, pool=12, target=0.50, min_gain=0.02, max_units=6, units=None):
    """162-head sweep, then forward selection over the top `pool`. Returns (singles, ranked, greedy)."""
    units = list(units) if units is not None else all_head_units()
    singles = unit_sweep(backend, prep, units)
    ranked = sorted(singles, key=singles.get, reverse=True)
    greedy = greedy_select(lambda s: recovery(prep, patched_axis(backend, prep, s)),
                           ranked[:pool], target=target, min_gain=min_gain, max_units=max_units)
    return singles, ranked, greedy


def direction_battery(backend, prep, units, q, q_rand=None):
    """Exact set, rank-k subspace `q`, its complement, and a random subspace of the same shape --
    each as a recovery and as a fraction of the exact-set effect on this prep's rows."""
    frac = lambda e, v: (v / e) if abs(e) > 1e-6 else None
    exact = recovery(prep, patched_axis(backend, prep, units))
    sub = recovery(prep, patched_axis(backend, prep, units, q=q))
    comp = recovery(prep, patched_axis(backend, prep, units, q=q, complement=True))
    out = {"exact_set": exact, "subspace": sub, "subspace_fraction": frac(exact, sub),
           "complement": comp, "complement_fraction": frac(exact, comp)}
    if q_rand is not None:
        rand = recovery(prep, patched_axis(backend, prep, units, q=q_rand))
        out.update({"random": rand, "random_fraction": frac(exact, rand)})
    return out


def pc_effects(backend, module, units, scale, q=None):
    """P and C same-answer effects of the (sub)space patch, on the module's P and C families."""
    out = {}
    for fam in ("P", "C"):
        fp = prepare(backend, rows_of(module, fam))
        out[fam] = same_answer_effect(fp, patched_axis(backend, fp, units, q=q), scale)
    return out


def set_battery(backend, module, units, *, rank=1, seed=1):
    """The whole standard follow-up for one unit set on one behaviour: exact-set A1/A2/P/C,
    diff-in-means direction (fit on even A1 rows) with complement and random on held-out A1 and A2."""
    a1 = rows_of(module, "A1")
    fit, held = prepare(backend, a1[0::2]), prepare(backend, a1[1::2])
    a2 = prepare(backend, rows_of(module, "A2"))
    q = diff_in_means_direction(backend, fit, units)
    q_rand = random_subspace(backend, units, rank=rank, seed=seed)
    scale = target_scale(fit)
    return {"units": list(units),
            "exact_set": {"a1_fit": recovery(fit, patched_axis(backend, fit, units)),
                          "a1_heldout": recovery(held, patched_axis(backend, held, units)),
                          "a2": recovery(a2, patched_axis(backend, a2, units)),
                          **{k.lower() + "_effect": v for k, v in pc_effects(backend, module, units, scale).items()}},
            "diff_in_means": {"a1_heldout": direction_battery(backend, held, units, q, q_rand),
                              "a2": direction_battery(backend, a2, units, q, q_rand),
                              **{k.lower() + "_effect": v for k, v in pc_effects(backend, module, units, scale, q=q).items()}}}


# ----------------------------------------------------------------------------- block-live variants
# The same three direction sources in the standard multi-site interchange semantics (q is a dict
# of per-(layer, kind) orthonormal matrices; the patch uses donor - LIVE at each hook).

def _orientation(delta):
    """Sign that aligns each row's delta with row 0's (geometric sign alignment). The screens
    alternate directions row by row, so a mean over MIXED rows cancels unless the deltas are
    oriented; v4-v6 fitted on even rows, which are direction-pure for the older candidates.
    Labels are not used: the spec-authored list candidate labels duplicate rows with opposite
    `direction_id`, so a label-based sign would cancel EXACT duplicates."""
    sign = (delta @ delta[0]).sign()
    sign[sign == 0] = 1.0
    return sign


def _cached_delta(backend, prep, units):
    torch = backend.torch

    def cached(cache, rid, u):
        if ":neuron:" in u:
            j = int(u.rsplit(":", 1)[1])
            return torch.as_tensor(cache[(rid, hidden_key(unit_layer(u)))])[j:j + 1]
        return torch.as_tensor(cache[(rid, u)])
    return torch.stack([
        torch.cat([cached(prep.donor_cache, rid, u).float() - cached(prep.base_cache, rid, u).float()
                   for u in units]) for rid in prep.base_batch.row_ids]).to(backend.device)


def block_diff_in_means(backend, prep, units):
    """Per-block unit-norm mean of the native (donor - base): rank 1 in every block."""
    out = {}
    for key, us in blocks_of(units).items():
        delta = _cached_delta(backend, prep, us)
        d = (delta * _orientation(delta)[:, None]).mean(0)
        if float(d.norm()) < 1e-3:
            raise ValueError("diff-in-means cancelled: rows are not direction-pure and carry no direction_id")
        out[key] = (d / d.norm()).unsqueeze(1)
    return out


def block_random_subspace(backend, units, *, rank=1, seed=1):
    torch = backend.torch
    gen = torch.Generator(device="cpu").manual_seed(seed)
    out = {}
    for key, us in blocks_of(units).items():
        q, _ = torch.linalg.qr(torch.randn(sum(unit_dim(u) for u in us), rank, generator=gen))
        out[key] = q.to(backend.device)
    return out


def block_identity(backend, units):
    """Full rank in every block: must reproduce the exact set to float precision (the control)."""
    torch = backend.torch
    return {key: torch.eye(sum(unit_dim(u) for u in us), device=backend.device)
            for key, us in blocks_of(units).items()}


def fit_block_subspace(backend, prep, units, *, rank=1, steps=200, lr=0.05, seed=0,
                       complement_weight=0.0):
    """DAS with one rank-`rank` subspace per block, all fitted jointly on the final margin against
    the exact-set target; optional complement-inertness term as in fit_joint_subspace."""
    torch = backend.torch
    for p in backend.model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    blocks = blocks_of(units)
    raws = {key: (torch.randn(sum(unit_dim(u) for u in us), rank, device=backend.device) * 0.02
                  ).requires_grad_(True) for key, us in blocks.items()}
    opt = torch.optim.Adam(list(raws.values()), lr=lr)
    target = torch.tensor(patched_axis(backend, prep, units), device=backend.device)
    base_target = torch.tensor(prep.base_axis, device=backend.device)
    history = []

    def qs():
        return {key: torch.linalg.qr(raw)[0] for key, raw in raws.items()}
    for step in range(steps):
        opt.zero_grad()
        q = qs()
        out = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                            base_cache=prep.base_cache, q=q, grad=True)
        match = ((-(out[:, 0] - out[:, 1]) - target) ** 2).mean()
        loss, inert = match, None
        if complement_weight > 0:
            comp = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                                 base_cache=prep.base_cache, q=q, grad=True, complement=True)
            inert = ((-(comp[:, 0] - comp[:, 1]) - base_target) ** 2).mean()
            loss = match + complement_weight * inert
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, float(match.detach()), None if inert is None else float(inert.detach())))
    with torch.no_grad():
        q = {key: v.detach() for key, v in qs().items()}
    return q, history


def fit_block_subspace_constrained(backend, prep, units, *, rank=1, steps=200, lr=0.05, seed=0,
                                   complement_weight=0.0, controls=(), control_weight=0.0, mu=None, init=None):
    """fit_block_subspace plus a REMOVAL-inertness penalty on control families (v74).

    init            optional {block_key: (dim, rank) tensor} (e.g. block_diff_in_means output) used as the
                    starting point instead of the seeded random raws (v82). None reproduces earlier runs.

    controls        preps (e.g. the behaviour's own C family, even rows) on which projecting the units'
                    activations onto `mu` along the fitted subspace must not change the answer log-prob.
                    The penalty is the mean squared answer-CE change over both sides' sentences, i.e.
                    exactly the v51.removal statistic made differentiable (mu is passed as the "donor").
    mu              {unit: background vector} used by the removal; required when controls are given.
    control_weight  0 -> identical objective to fit_block_subspace (same seed gives the same q; the
                    reproduction control every runner using this must register).
    Returns (q, history) with history rows (step, match, inert, control_penalty)."""
    torch = backend.torch
    F = torch.nn.functional
    for p in backend.model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    blocks = blocks_of(units)
    raws = {key: (torch.randn(sum(unit_dim(u) for u in us), rank, device=backend.device) * 0.02
                  ).requires_grad_(True) for key, us in blocks.items()}
    if init is not None:
        raws = {key: init[key].detach().clone().float().to(backend.device).requires_grad_(True) for key in blocks}
    opt = torch.optim.Adam(list(raws.values()), lr=lr)
    target = torch.tensor(patched_axis(backend, prep, units), device=backend.device)
    base_target = torch.tensor(prep.base_axis, device=backend.device)
    ctrl = []
    if control_weight > 0:
        if mu is None:
            raise ValueError("mu is required when controls are given")
        for cp in controls:
            for side in ("base", "donor"):
                batch = cp.base_batch if side == "base" else cp.donor_batch
                cache = cp.base_cache if side == "base" else cp.donor_cache
                bg = dict(cache)
                for rid in batch.row_ids:
                    for u in units:
                        bg[(rid, u)] = mu[u]
                ans = torch.tensor(batch.answer_ids, device=backend.device)
                with torch.no_grad():
                    _, nat = forward_units(backend, batch, units=[], return_logits=True)
                    lp_nat = F.log_softmax(nat.float(), -1)[torch.arange(len(batch.row_ids)), ans]
                ctrl.append((batch, cache, bg, ans, lp_nat))
    history = []

    def qs():
        return {key: torch.linalg.qr(raw)[0] for key, raw in raws.items()}
    for step in range(steps):
        opt.zero_grad()
        q = qs()
        out = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                            base_cache=prep.base_cache, q=q, grad=True)
        match = ((-(out[:, 0] - out[:, 1]) - target) ** 2).mean()
        loss, inert, pen = match, None, None
        if complement_weight > 0:
            comp = forward_units(backend, prep.base_batch, units=units, donor_cache=prep.donor_cache,
                                 base_cache=prep.base_cache, q=q, grad=True, complement=True)
            inert = ((-(comp[:, 0] - comp[:, 1]) - base_target) ** 2).mean()
            loss = loss + complement_weight * inert
        if ctrl:
            terms = []
            for batch, cache, bg, ans, lp_nat in ctrl:
                _, rem = forward_units(backend, batch, units=units, donor_cache=bg, base_cache=cache,
                                       q=q, grad=True, return_logits=True)
                lp_rem = F.log_softmax(rem.float(), -1)[torch.arange(len(batch.row_ids)), ans]
                terms.append(((lp_nat - lp_rem) ** 2).mean())
            pen = torch.stack(terms).mean()
            loss = loss + control_weight * pen
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == steps - 1:
            history.append((step, float(match.detach()), None if inert is None else float(inert.detach()),
                            None if pen is None else float(pen.detach())))
    with torch.no_grad():
        q = {key: v.detach() for key, v in qs().items()}
    return q, history


def block_cosines(qa, qb):
    """|cos| between two block-rank-1 direction sets, per block."""
    return {f"{k[0]:02d}:{k[1]}": float((qa[k][:, 0] @ qb[k][:, 0]).abs()) for k in qa}


def norm_shares(q, units):
    """How much of a joint direction's norm sits in each unit (which head 'owns' the direction)."""
    out, off = {}, 0
    for u in units:
        d = unit_dim(u)
        out[u] = float((q[off:off + d, 0] ** 2).sum())
        off += d
    return out


def block_direction_battery(backend, prep, units, q, q_rand=None):
    """`direction_battery` plus the linearity sum S + C (subspace + complement fractions; ~1 for
    a linearly carried variable, > 1 means a nonlinear route). `q` is a per-block dict."""
    b = direction_battery(backend, prep, units, q, q_rand=q_rand)
    s, c = b["subspace_fraction"], b["complement_fraction"]
    b["linearity_sum"] = None if s is None or c is None else s + c
    return b


def block_semantics(backend, prep, units):
    """The v7 control, mandatory on every multi-layer set: exact interchange vs the cached-joint
    full-rank patch (biased across layers) vs the block-live full-rank patch (must equal exact)."""
    torch = backend.torch
    exact = recovery(prep, patched_axis(backend, prep, units))
    eye = torch.eye(sum(unit_dim(u) for u in units), device=backend.device)
    cached = recovery(prep, patched_axis(backend, prep, units, q=eye))
    block = recovery(prep, patched_axis(backend, prep, units, q=block_identity(backend, units)))
    return {"exact": exact, "cached_full_rank": cached, "block_full_rank": block,
            "cached_bias_fraction": (cached - exact) / exact if abs(exact) > 1e-6 else None,
            "block_error": abs(block - exact)}


def block_battery(backend, module, units, *, seed=1, greedy=None):
    """`set_battery` in BLOCK-LIVE semantics (the object since v7): exact-set A1 fit / held-out /
    A2 / P / C, the v7 semantics control on held-out rows, and the block diff-in-means direction
    (fit on even A1 rows) with complement, linearity sum and a rank-matched random direction on
    held-out A1 and A2 rows, plus its P / C. `greedy` (optional) is stored verbatim."""
    a1 = rows_of(module, "A1")
    fit, held = prepare(backend, a1[0::2]), prepare(backend, a1[1::2])
    a2 = prepare(backend, rows_of(module, "A2"))
    scale = target_scale(fit)
    q = block_diff_in_means(backend, fit, units)
    q_rand = block_random_subspace(backend, units, rank=1, seed=seed)
    pc_exact = pc_effects(backend, module, units, scale)
    pc_dim = pc_effects(backend, module, units, scale, q=q)
    out = {"units": list(units),
           "exact_set": {"a1_fit": recovery(fit, patched_axis(backend, fit, units)),
                         "a1_heldout": recovery(held, patched_axis(backend, held, units)),
                         "a2": recovery(a2, patched_axis(backend, a2, units)),
                         "p_effect": pc_exact["P"], "c_effect": pc_exact["C"]},
           "semantics_heldout": block_semantics(backend, held, units),
           "diff_in_means": {"a1_heldout": block_direction_battery(backend, held, units, q, q_rand),
                             "a2": block_direction_battery(backend, a2, units, q, q_rand),
                             "p_effect": pc_dim["P"], "c_effect": pc_dim["C"]}}
    if greedy is not None:
        out["greedy"] = greedy
    return out


def block_union(*qs):
    """Per-block orthonormal basis of the span of several block subspaces (rank adds per block).
    Registered-rank use: the union of an A1-fit and an A2-fit direction is rank 2 per block."""
    import torch
    return {key: torch.linalg.qr(torch.cat([q[key] for q in qs], dim=1))[0] for key in qs[0]}


def lexical_variant(rows, mapping, *, encoding=None):
    """Rows with whole-word substitutions in base/donor text, re-tokenized; answers and foils kept.

    `mapping` is {old_word: new_word}; every row must change on both sides. The final token must
    survive (the prediction slot is unchanged) and the joint answer/foil tokenization must hold;
    positions are re-derived from the new length, so multi-token substitutes are allowed.
    """
    import copy
    import re
    if encoding is None:
        import circuit_fast_screen_candidate_sentence_terminal_context_control as builder
        encoding = builder.ENCODING
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in mapping) + r")\b")
    out = []
    for r in rows:
        n = copy.deepcopy(r)
        for side in ("base", "donor"):
            text = pattern.sub(lambda m: mapping[m.group(1)], n[f"{side}_text"])
            assert text != n[f"{side}_text"], (side, text)
            ids = encoding.encode(text)
            assert ids[-1] == n[f"{side}_ids"][-1], (side, text)
            for tok, key in ((n[f"{side}_answer"], "answer"), (n[f"{side}_foil"], "foil")):
                assert encoding.encode(text + tok) == ids + [n[f"{side}_{key}_id"]], (side, text, tok)
            assert n[f"{side}_semantic_position"] == len(n[f"{side}_ids"]) - 1, side
            n[f"{side}_text"], n[f"{side}_ids"] = text, ids
            n[f"{side}_semantic_position"] = n[f"{side}_prediction_position"] = len(ids) - 1
        n["row_id"] = f"{n['row_id']}:" + "_".join(mapping.values())
        out.append(n)
    return out


def swap_answer_foil(rows):
    """Rows with answer and foil exchanged on both sides (for building a sibling whose answer is the
    original foil, e.g. a whether-only control from a that-only one after a verb substitution)."""
    import copy
    out = []
    for r in rows:
        n = copy.deepcopy(r)
        for side in ("base", "donor"):
            for a, b in ((f"{side}_answer", f"{side}_foil"), (f"{side}_answer_id", f"{side}_foil_id")):
                n[a], n[b] = n[b], n[a]
        n["row_id"] = f"{n['row_id']}:swap"
        out.append(n)
    return out


def swap_base_donor(rows):
    """Rows with the base and donor sides exchanged (the reverse interchange on the same sentences)."""
    import copy
    out = []
    for r in rows:
        n = copy.deepcopy(r)
        for k in list(r):
            if k.startswith("base_"):
                d = "donor_" + k[len("base_"):]
                n[k], n[d] = r[d], r[k]
        n["row_id"] = f"{n['row_id']}:rev"
        out.append(n)
    return out
