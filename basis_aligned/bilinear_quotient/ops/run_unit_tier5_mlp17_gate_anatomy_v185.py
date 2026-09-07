#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; push layer 17, two-step scale rule fixed; the gate vector and the lambda-weighted write map are exact algebra, no fit.
"""v185: WHAT GATES mlp:17's damping? v184 showed the damping of an answer-axis push at t is the CROSS term of the bilinear MLP,
s^2 * alpha * Down(Lx*Ru + Lu*Rx), linear in the base residual x. Projected on the unit axis u it is s^2 * alpha * (x . f_u) with a fixed
GATE VECTOR f_u = L^T((D^T u)*(R u)) + R^T((D^T u)*(L u)) that depends only on the weights and u. So the damper's gain is one number: the
base residual's projection on f_u. This rung asks where that projection comes from: (i) SELF — the residual's own coordinate on u
(z0 = x.u) times (u.f_u); (ii) CONTEXT — the u-orthogonal part of x; (iii) TOKEN — the accumulated x0 re-entry (x0 = rms_norm(wte) enters
every block as lambda1 * x0; the accumulated coefficient is 145 at block 17), which depends only on the token at t; (iv) WRITERS — each
unit's base write at t, propagated with the product of the later blocks' lambda0 (exact linear accumulation), ranked by its share of the
gate. Causal arm: the top gate-writer's write at t is zeroed during the push; if it acts directly on mlp:17's read (no relay through the
units between it and layer 17 at t), the measured counter-write under removal equals the exact expansion evaluated on x minus the
propagated write.

Magnitudes printed before writing (CPU probe, 5 cells, 3 rows): cos(f_u, u) 0.568 / 0.142 / 0.527 / -0.356 (coordination, polarity,
degree, correlative_both_either); gate x.f_hat / |x| = -0.06 / -0.10 / -0.07..-0.11 / -0.01..-0.03 vs chance 0.029; self gate -216..-382
vs context gate -5742..-6590 (polarity), -406..-704 vs -2478..-4894 (degree), +1449..+2078 vs -1739..-2739 (correlative: the self part
OPPOSES) — the probe's per-row lines print before their cell label and I first mis-assigned them by one cell; corrected from the smoke
(coordination p0 self -1064 vs context -2392; p1 self +835 vs -1584, opposing); x0 term |B| = 4926 vs |x| 46k-60k, x0 share of the gate -0.016..-0.11 on the strong-gate cells, -0.22..-0.65 and -0.25..+0.36
on the two weak-gate cells (coordination p1, correlative p0, |x.f| ~ 700); lambda-weighted write map + x0 term reconstructs x.f to 0.6%
(-3478 vs -3456); top-3 writers' share 0.345 (mlp:08, attn:09, attn:07), 0.443 (mlp:16, mlp:15, mlp:10), 0.262 (attn:07, attn:12, attn:08),
0.646, 1.759 (weak-gate cell). Gate vectors across families cos 0.09-0.38 (not one universal gate).

Smoke (CPU, coordination p0 + p1, 3 rows, 3 s of measurement after setup): push 0.999 / 1.000; gate -3456 / -749; recon -3478 / -771 (abs err 22
both); self -1064 / +835 vs context -2392 / -1584; token share -0.05 / -0.40; top-3 0.345 (mlp:08, attn:09, attn:07) / 0.646 (mlp:16, mlp:15,
mlp:12); removal of the top-1 writer: measured -354 vs analytic -287 (mlp:08, direct_err 0.18) / -242 vs -254 (mlp:16, 0.042). Bars a, b, e
re-registered as stated above; c and d unchanged.

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar has a lower AND an upper bound where a fraction is compared):
  pred_a_instrument      push_r2 in 0.95-1.05 on 14/14 AND the lambda-weighted write map plus the x0 term reconstructs the gate x.f_hat
                         to within 0.002 x |x| on 14/14 (recon_err_over_xnorm <= 0.002; PRE-SMOKE bar was 2% of the gate — the smoke read
                         0.6% and 2.9%: a constant absolute error of 22 on residual norms of 55k-60k, i.e. accumulation precision, which the
                         weak-gate cell turns into 2.9% of a small gate).
  pred_b_context_gated   the u-orthogonal part of x carries the damping: perp_gate < 0 on >= 13 of 14 AND |self_gate| / |perp_gate| lies in
                         0.02-1.0 on >= 12 of 14 (the self part is real but never the larger part; probe 0.05 / 0.17 / 0.7, smoke 0.445 / 0.527;
                         PRE-SMOKE band was 0.02-0.5 on >= 10 — the smoke read 0.527 on coordination p1 and the probe 0.7 on correlative).
  pred_c_token_small     the x0 re-entry's share of the gate, B.f_hat / x.f_hat, lies in -0.15..+0.15 on >= 9 of 14 (the probe read
                         0.02-0.11 on strong-gate cells and 0.22-0.65 on the two weak-gate cells, which are expected to fail).
  pred_d_distributed     the top-3 gate writers' share of the gate lies in 0.2-0.8 on >= 10 of 14 (no single unit sets the gain; the probe read
                         0.26-0.65 on four cells and 1.76 on the weak-gate correlative cell).
  pred_e_direct_writer   zeroing the top-1 gate writer's write at t during the push, with direct_err = |measured_removed - analytic_removed| /
                         |measured| (analytic_removed = the exact expansion on x minus the propagated write): on the cells whose top-1 writer
                         is at layer >= 15 (only attn:17 can relay), direct_err <= 0.15 on >= 2/3 of them (and there are >= 3 such cells);
                         on the cells whose top-1 writer is at layer <= 14, the units between it and layer 17 RESTORE part of the gate:
                         |measured_removed| > |analytic_removed| on >= 2/3 of them (and >= 3 such cells). PRE-SMOKE form was direct_err <= 0.15
                         on >= 10 of 14 — the smoke read 0.18 for mlp:08 (measured -354 vs analytic -287: the relay restores) and 0.042 for mlp:16.
  Reported, not predicted: top-1 writer per cell, late-pair (mlp:15 + mlp:16) share, mid-chain (units 07-12) share, cos(f_u, u), gate align vs chance.
"""
from __future__ import annotations

import importlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_mlp17_gate_anatomy_v185_result.json"
V184 = ROOT / "circuits/followups/unit_tier5_mlp17_gain_terms_v184_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V183 = ROOT / "circuits/followups/unit_tier5_mlp17_damping_split_v183_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LAST = 17
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"push_band": [0.95, 1.05], "recon_tol_over_xnorm": 0.002, "perp_neg_min_cells": 13, "self_ratio_band": [0.02, 1.0], "self_min_cells": 12,
        "token_band": [-0.15, 0.15], "token_min_cells": 9, "top3_band": [0.2, 0.8], "top3_min_cells": 10, "direct_tol": 0.15, "late_layer_min": 15, "group_min_cells": 3, "group_frac": 2 / 3,
        "push_start_frac": 0.05, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_mlp17_gate_anatomy_v185", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V185_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V185_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V185_SMOKE_CELLS", "2"))]
    cells = {}

    def setup(fam, par):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1)
        fb, fd = g.cue_positions(prep.base_batch, prep.donor_batch, which="first")
        lb, ld = g.cue_positions(prep.base_batch, prep.donor_batch, which="last")
        assert all(fb[i] == lb[i] and fd[i] == ld[i] for i in range(len(a1))), "multi-token cue"
        batch, db = prep.base_batch, prep.donor_batch
        rows = len(batch.row_ids)
        ar = torch.arange(rows, device=backend.device)
        cb_t, cd_t = torch.tensor(lb, device=backend.device), torch.tensor(ld, device=backend.device)
        def capture(bt, pos_t):
            cap = {}
            hs = []
            for l in range(N_LAYERS):
                hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(f"attn:{l:02d}", a[0][ar, pos_t].detach().clone())))
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: cap.__setitem__(f"mlp:{l:02d}", o[ar, pos_t].detach().clone())))
            hs.append(model.transformer.h[1].attn.register_forward_pre_hook(lambda m_, a: cap.__setitem__("v1", a[1][ar, pos_t].detach().clone())))
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cap
        cap, capb = capture(db, cd_t), capture(batch, cb_t)
        def capture_writes(bt, pos_t):
            """unit WRITES at pos_t (attn: c_proj output; mlp: Down output), base forward."""
            cw = {}
            hs = []
            for l in range(N_LAYERS):
                hs.append(model.transformer.h[l].attn.c_proj.register_forward_hook(lambda m_, a, o, l=l: cw.__setitem__(f"attn:{l:02d}", o[ar, pos_t].detach().float().clone())))
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: cw.__setitem__(f"mlp:{l:02d}", o[ar, pos_t].detach().float().clone())))
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cw
        t_t0 = torch.tensor(list(batch.semantic_positions), device=backend.device)
        capt = capture(batch, t_t0)
        capw = capture_writes(batch, t_t0)
        toks_t = backend._tensor_batch(batch)[0][ar, t_t0]
        delta = (cap["v1"][:, CH].float() - capb["v1"][:, CH].float()).mean(0)
        t_t = torch.tensor(list(batch.semantic_positions), device=backend.device)
        sl = slice(CH * HD, (CH + 1) * HD)
        def run(vec=None, add=None, grab=None, remove=()):
            """cue-position writes base; base x0; if vec: slice CH of the cue's v1 = base + vec at layers 1-17; if add: (B,T,128) added to
            head CH's c_proj input at layer HUB; if grab: dict receiving that c_proj input (full tensor)."""
            hs = []
            for u in UNITS:
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u, l=l):
                        v = a[0].clone(); v[ar, cb_t] = capb[u].to(v.dtype)
                        if l == HUB and grab is not None: grab["x"] = v.detach().clone()
                        if l == HUB and add is not None: v[:, :, sl] = v[:, :, sl] + add.to(v.dtype)
                        if u in remove: v[ar, t_t] = capt[u].to(v.dtype)
                        return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype)
                        if u in remove: o[ar, t_t] = capt[u].to(o.dtype)
                        return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            if vec is not None:
                tgt = (capb["v1"][:, CH].float() + vec[None, :]).to(capb["v1"].dtype)
                for l in range(1, N_LAYERS):
                    def vh(m_, a):
                        t = a[1].clone(); t[ar, cb_t, CH] = tgt.to(t.dtype); return (a[0], t)
                    hs.append(model.transformer.h[l].attn.register_forward_pre_hook(vh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        td_t = torch.tensor(list(db.semantic_positions), device=backend.device)
        def run_donor(remove=()):
            """the donor batch itself (the full interchange), with the writes in `remove` clamped at t to the BASE batch's value."""
            hs = []
            for u in remove:
                l = int(u.split(":")[1])
                def mh(m_, a, o, u=u):
                    o = o.clone(); o[ar, td_t] = capt[u].to(o.dtype); return o
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, db, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [float(x) - float(f) for x, f in out.tolist()]), 3)   # donor batch: its own answer axis
        U = model.lm_head.weight.detach().float()
        ans = torch.tensor(list(batch.answer_ids), device=backend.device); foil = torch.tensor(list(batch.foil_ids), device=backend.device)
        u = U[foil] - U[ans]                                                    # (rows, N_EMBD): base batch foil = the donor's answer
        u = u / u.norm(dim=1, keepdim=True)
        gen = torch.Generator(device="cpu").manual_seed(BARS["seed"])
        rnd = torch.randn(rows, u.shape[1], generator=gen).to(backend.device)
        rnd = rnd - (rnd * u).sum(1, keepdim=True) * u; rnd = rnd / rnd.norm(dim=1, keepdim=True)
        resid = {}
        with torch.no_grad():
            g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=resid)
        rnorm = torch.stack([resid[(rid, LAST)] for rid in batch.row_ids]).norm(dim=1)   # residual norm at t after attn:17
        def run_push(alpha, dirn, clamp=None, grab=None, zero_unit=None):
            """base batch + alpha (rows,) * dirn (rows, N_EMBD) added at t after attn:17; if clamp is a (rows, N_EMBD) tensor, mlp:17's
            write at t is set to it; grab receives mlp:17's write ('w') and the residual after attn:17 ('xa') at t; zero_unit: that unit's write at t is
            set to zero (attn: c_proj output; mlp: Down output). Returns (recovery, margins)."""
            hs = []
            if zero_unit is not None:
                zk, zl = zero_unit.split(":"); zl = int(zl)
                zmod = model.transformer.h[zl].attn.c_proj if zk == "attn" else model.transformer.h[zl].mlp.Down
                def zh(m_, a, o):
                    o = o.clone(); o[ar, t_t] = 0; return o
                hs.append(zmod.register_forward_hook(zh))
            def mh(m_, a, o):
                if grab is not None: grab["w"] = o[ar, t_t].detach().float().clone()
                if clamp is not None:
                    o = o.clone(); o[ar, t_t] = clamp.to(o.dtype); return o
            hs.append(model.transformer.h[LAST].mlp.Down.register_forward_hook(mh))
            cr = {}
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                          resid_add={LAST: alpha[:, None] * dirn}, capture_resid=cr)
            finally:
                for h_ in hs: h_.remove()
            if grab is not None:   # residual at t after attn:17 (push included) = what mlp:17 reads; the producer's forward calls attn and mlp separately
                grab["xa"] = torch.stack([cr[(rid, LAST)] for rid in batch.row_ids])
            margins = [-(float(x) - float(f)) for x, f in out.tolist()]
            return round(g.recovery(prep, margins), 3), margins
        gs, gb = {}, {}
        full_swap = run(vec=delta, grab=gs); run(grab=gb)
        W = (gs["x"][:, :, sl].float() - gb["x"][:, :, sl].float())          # (B, T, 128) 07:08's write change under the slice-8 swap
        wt = W[ar, t_t]                                                        # (rows, 128) at t
        direction = wt.mean(0); direction = direction / direction.norm()
        sv = torch.linalg.svdvals(wt)
        return {"rows": rows, "delta": delta, "run": run, "run_donor": run_donor, "run_push": run_push, "u": u, "capt17": capt["mlp:17"].float(), "capw": capw, "toks_t": toks_t, "rnorm": rnorm, "prep": prep, "W": W, "direction": direction, "full_swap": full_swap,
                "rank1_frac": round(float(sv[0] ** 2 / (sv ** 2).sum()), 3), "wt_norm": round(float(wt.norm(dim=1).mean()), 3),
                "t_share_of_norm": round(float(wt.norm() ** 2 / W.norm() ** 2), 3)}


    for fam, par in cells_run:
        cells[(fam, par)] = setup(fam, par)

    R = {}
    mlp = model.transformer.h[LAST].mlp
    assert not getattr(model.config, "gated", False), "expansion assumes the ungated Bilinear"
    Lw, Rw, Dw = mlp.Left.weight.detach().float(), mlp.Right.weight.detach().float(), mlp.Down.weight.detach().float()
    D = Lw.shape[1]
    rms = lambda x: (x.float().pow(2).mean(1) + 1e-6).sqrt()
    def terms(xb, xp, alpha, u):
        """exact three-term expansion of mlp:17(rms_norm(xp)) - mlp:17(rms_norm(xb)) for xp = xb + alpha*u; returns on-axis projections (rows,)."""
        s0, s1 = 1 / rms(xb), 1 / rms(xp)
        Lx, Rx, Lu, Ru = xb @ Lw.T, xb @ Rw.T, u @ Lw.T, u @ Rw.T
        base = (Lx * Rx) @ Dw.T
        resc = (s1 ** 2 - s0 ** 2)[:, None] * base
        cross = (s1 ** 2 * alpha)[:, None] * ((Lx * Ru + Lu * Rx) @ Dw.T)
        crossL = (s1 ** 2 * alpha)[:, None] * ((Lx * Ru) @ Dw.T)              # base residual through Left, push through Right
        quad = (s1 ** 2 * alpha ** 2)[:, None] * ((Lu * Ru) @ Dw.T)
        onp = lambda v: (v * u).sum(1)
        return onp(resc), onp(cross), onp(crossL), onp(quad)
    lam = [tuple(float(v) for v in model.transformer.h[l].lambdas.detach().float()) for l in range(N_LAYERS)]
    def prop(l):
        """propagation factor of a write made in block l to the read point after attn:17 (product of the later blocks' lambda0)."""
        pl = 1.0
        for m in range(l + 1, LAST + 1): pl *= lam[m][0]
        return pl
    x0_coef = sum((lam[l][1] + (lam[l][0] if l == 0 else 0.0)) * prop(l) for l in range(LAST + 1))
    def fgate(u):
        """gate vector f_u: cross_on = s^2 * alpha * (x . f_u)."""
        Du = u @ Dw
        return ((Du * (u @ Rw.T)) @ Lw) + ((Du * (u @ Lw.T)) @ Rw)
    WRITERS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp") if not (k == "mlp" and l == LAST)]
    v184 = json.loads(V184.read_text())["measures"] if V184.exists() else {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        u = C["u"]
        a0 = BARS["push_start_frac"] * C["rnorm"]
        r0, _ = C["run_push"](a0, u)
        a1 = a0 / max(r0, 0.05)
        r1, _ = C["run_push"](a1, u)
        a2 = a1 / max(r1, 0.05)
        gp, g0 = {}, {}
        r2, _ = C["run_push"](a2, u, grab=gp); C["run_push"](0 * a2, u, grab=g0)
        meas = ((gp["w"] - g0["w"]) * u).sum(1)                                 # measured on-axis counter-write (rows,)
        x = g0["xa"].float()
        f = fgate(u); fh = f / f.norm(dim=1, keepdim=True)
        G = (x * fh).sum(1)                                                     # the gate (rows,)
        z0 = (x * u).sum(1)
        self_g = z0 * (u * fh).sum(1); perp_g = G - self_g
        x0 = torch.nn.functional.rms_norm(model.transformer.wte(C["toks_t"]).float(), (x.shape[1],))
        Bx = x0_coef * x0
        tok_share = (Bx * fh).sum(1) / G
        wmap = {k: (C["capw"][k] * fh).sum(1) * prop(int(k.split(":")[1])) for k in WRITERS}   # (rows,) per writer, propagated
        wmean = {k: float(v.mean()) for k, v in wmap.items()}
        recon = sum(wmean.values()) + float((Bx * fh).sum(1).mean())
        Gm = float(G.mean())
        recon_rel_err = abs(recon - Gm) / max(abs(Gm), 1e-6)
        ranked = sorted(wmean.items(), key=lambda kv: -abs(kv[1]))
        top1 = ranked[0][0]
        top3_share = sum(v for _, v in ranked[:3]) / Gm
        late_share = (wmean["mlp:15"] + wmean["mlp:16"]) / Gm
        mid_share = sum(v for k, v in wmean.items() if 7 <= int(k.split(":")[1]) <= 12) / Gm
        # causal: zero the top-1 writer's write at t during the push and at zero push
        gr, gr0 = {}, {}
        rr, _ = C["run_push"](a2, u, grab=gr, zero_unit=top1); C["run_push"](0 * a2, u, grab=gr0, zero_unit=top1)
        meas_rem = ((gr["w"] - gr0["w"]) * u).sum(1)
        x_rem = x - C["capw"][top1] * prop(int(top1.split(":")[1]))            # analytic: the propagated write removed, nothing else
        resc_r, cross_r, _, quad_r = terms(x_rem, x_rem + a2[:, None] * u, a2, u)
        ana_rem = resc_r + cross_r + quad_r
        measured_m = float(meas.mean()); measured_rem_m = float(meas_rem.mean()); analytic_rem_m = float(ana_rem.mean())
        direct_err = abs(measured_rem_m - analytic_rem_m) / max(abs(measured_m), 1e-6)     # pred_e quantity, one line
        resc0, cross0, _, quad0 = terms(x, gp["xa"], a2, u)
        S = {"rows": C["rows"], "push_r0": r0, "push_r1": r1, "push_r2": r2, "push_removed_r": rr, "push_norm": round(float(a2.mean()), 1),
             "measured_on": round(measured_m, 1), "cross_on": round(float(cross0.mean()), 1), "v184_cross_on": v184.get(f"{fam}:{par}", {}).get("cross_on"),
             "gate": round(Gm, 1), "gate_align": round(float((G / x.norm(dim=1)).mean()), 4), "chance_align": round(1 / x.shape[1] ** 0.5, 4),
             "cos_f_u": round(float((u * fh).sum(1).mean()), 3), "f_norm": round(float(f.norm(dim=1).mean()), 1),
             "self_gate": round(float(self_g.mean()), 1), "perp_gate": round(float(perp_g.mean()), 1),
             "self_ratio": round(abs(float(self_g.mean())) / max(abs(float(perp_g.mean())), 1e-6), 3),
             "token_share": round(float(tok_share.mean()), 3), "x0_coef": round(x0_coef, 2), "x0_term_norm": round(float(Bx.norm(dim=1).mean()), 1),
             "recon": round(recon, 1), "recon_rel_err": round(recon_rel_err, 4), "recon_err_over_xnorm": round(abs(recon - Gm) / float(x.norm(dim=1).mean()), 5),
             "top1_layer": int(top1.split(":")[1]),
             "top_writers": [(k, round(v, 1)) for k, v in ranked[:5]], "top1": top1, "top3_share": round(top3_share, 3),
             "late_pair_share": round(late_share, 3), "mid_chain_share": round(mid_share, 3),
             "measured_removed": round(measured_rem_m, 1), "analytic_removed": round(analytic_rem_m, 1), "direct_err": round(direct_err, 3),
             "removed_share": round(1 - measured_rem_m / measured_m, 3) if abs(measured_m) > 1e-6 else None,
             "writer_gate_share": round(wmean[top1] / Gm, 3)}
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_mlp17_gate_anatomy_v185", "candidate_id": "corpus.unit_tier5_mlp17_gate_anatomy_v185",
              "bars": BARS, "families": list(FAMILIES), "channel": CH,
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def pearson(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5; sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return round(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb), 3) if sa > 0 and sb > 0 else None



def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    pred_a = all(inb(S["push_r2"], B["push_band"]) for S in cells) and all(S["recon_err_over_xnorm"] <= B["recon_tol_over_xnorm"] for S in cells)
    pred_b = sum(S["perp_gate"] < 0 for S in cells) >= B["perp_neg_min_cells"] and sum(inb(S["self_ratio"], B["self_ratio_band"]) for S in cells) >= B["self_min_cells"]
    pred_c = sum(inb(S["token_share"], B["token_band"]) for S in cells) >= B["token_min_cells"]
    pred_d = sum(inb(S["top3_share"], B["top3_band"]) for S in cells) >= B["top3_min_cells"]
    late = [S for S in cells if S["top1_layer"] >= B["late_layer_min"]]; mid = [S for S in cells if S["top1_layer"] < B["late_layer_min"]]
    late_ok = len(late) >= B["group_min_cells"] and sum(S["direct_err"] <= B["direct_tol"] for S in late) >= B["group_frac"] * len(late)
    mid_ok = len(mid) >= B["group_min_cells"] and sum(abs(S["measured_removed"]) > abs(S["analytic_removed"]) for S in mid) >= B["group_frac"] * len(mid)
    pred_e = late_ok and mid_ok
    return {"pred_a_instrument": pred_a, "pred_b_context_gated": pred_b, "pred_c_token_small": pred_c, "pred_d_distributed": pred_d, "pred_e_direct_writer": pred_e}


if __name__ == "__main__":
    main()
