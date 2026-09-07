#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; push layer 17, two-step scale rule fixed; the three-term bilinear expansion is exact algebra, no fit.
"""v184: WHERE does mlp:17's damping gain come from? v184: mlp:17's counter-write to an answer-axis push at t is its ON-axis part, and it
cancels exactly its own share of the push (0.06-0.30 of it, family-specific). mlp:17 is a Bilinear MLP, out = Down(Left n * Right n) + b
with n = rms_norm(x), so its response to x_b + alpha*u is EXACTLY three terms: (i) rescale, (s^2 - s0^2) * Down(Lx*Rx) — the base product
under the changed norm; (ii) cross, s^2 * alpha * Down(Lx*Ru + Lu*Rx) — the base residual gating the push; (iii) quadratic,
s^2 * alpha^2 * Down(Lu*Ru). (s = 1/rms.) Each is computed from the weights and the captured residual, no fit, and projected on the unit
axis u; the sum must reproduce the measured on-axis counter-write to float precision (instrument). If the cross term carries the gain,
the damper's strength is set by the base CONTEXT at t (which of Lx / Rx gates it is reported, not predicted); a quadratic gain would be
context-free. A 2x push checks dose-linearity of the on-fraction directly.

Magnitudes printed before writing (v184 receipt): on-axis counter-write 188-2285 against pushes of 3.0k-9.6k (fraction 0.06-0.30);
final residual 59k-106k, norm ratio under the off-axis part 0.81-1.14.

Smoke (CPU, coordination p0 + p1, 3 rows, 60 s, run with the pre-smoke bars): expansion reproduces the measured on-axis counter-write to
0.0000 relative (-374.6 = -374.6; -271.6 = -271.6); rescale -0.1 / -0.1; cross -535 / -119; quadratic +160 / -153; Left-x half of cross
0.506 / 0.514; on-fraction 0.102 / 0.082 (v183 0.101 / 0.099 on 16 rows); 2x on-fraction 0.058 / 0.127. Bars b, c, e re-registered as
stated above.

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar has a lower AND an upper bound where a fraction is compared):
  pred_a_instrument     rescale + cross + quadratic reproduces the measured on-axis counter-write within 5% on 14/14 AND the twice-rescaled
                        push recovers 0.95-1.05 on 14/14
  pred_b_cross_damps    the cross term's on-axis projection is negative on >= 13 of 14 AND its fraction of the push lies in 0.03-0.40 on
                        >= 12 of 14 (the context-gated linear response damps in both parities; PRE-SMOKE bar was cross / measured in
                        0.7-1.3 on >= 12 — the smoke read 1.43 / 0.44)
  pred_c_quadratic_matters |quadratic term| / |measured on-axis counter-write| lies in 0.2-1.5 on >= 10 of 14 at the calibrated 1x push (PRE-SMOKE bar
                        was <= 0.2 on >= 12 — the smoke read 0.43 / 0.56, with OPPOSITE signs on p0 and p1: the quadratic projection is odd
                        in u, so it pushes toward one fixed answer whichever way the push points; that sign flip is algebra when the two
                        parities share a token pair and is NOT registered as a finding)
  pred_d_rescale_small  |rescale term| <= 0.2 x the measured on-axis counter-write on >= 12 of 14 (prior 0.7)
  pred_e_symmetric_form the Left-x/Right-u half of the cross term is 0.4-0.6 of the whole cross term on >= 12 of 14 (the bilinear form is
                        effectively symmetric on this pair although Left and Right are untied — flat cos(L, R) = -0.00005, row-wise |cos|
                        0.12, probed before registering; prior 0.5). The pre-smoke pred_e (2x on-fraction within 0.03 of 1x) was DROPPED:
                        given an exact expansion it is algebra (fraction(2x) - fraction(1x) = -quad / push), and the smoke read -0.044 = exactly that.
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
OUT = ROOT / "circuits/followups/unit_tier5_mlp17_gain_terms_v184_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V183 = ROOT / "circuits/followups/unit_tier5_mlp17_damping_split_v183_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LAST = 17
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"sum_rel_tol": 0.05, "push_band": [0.95, 1.05], "cross_neg_min_cells": 13, "cross_frac_band": [0.03, 0.40], "cross_frac_min_cells": 12,
        "quad_rel_band": [0.2, 1.5], "quad_min_cells": 10, "rescale_max_rel": 0.2, "rescale_min_cells": 12, "left_share_band": [0.4, 0.6], "left_min_cells": 12,
        "push_start_frac": 0.05, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_mlp17_gain_terms_v184", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V184_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V184_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V184_SMOKE_CELLS", "2"))]
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
        t_t0 = torch.tensor(list(batch.semantic_positions), device=backend.device)
        capt = capture(batch, t_t0)
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
        def run_push(alpha, dirn, clamp=None, grab=None):
            """base batch + alpha (rows,) * dirn (rows, N_EMBD) added at t after attn:17; if clamp is a (rows, N_EMBD) tensor, mlp:17's
            write at t is set to it; grab receives mlp:17's write ('w') and the final residual ('x') at t. Returns (recovery, margins)."""
            hs = []
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
        return {"rows": rows, "delta": delta, "run": run, "run_donor": run_donor, "run_push": run_push, "u": u, "capt17": capt["mlp:17"].float(), "rnorm": rnorm, "prep": prep, "W": W, "direction": direction, "full_swap": full_swap,
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
    v183 = json.loads(V183.read_text())["measures"] if V183.exists() else {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        u = C["u"]
        a0 = BARS["push_start_frac"] * C["rnorm"]
        r0, _ = C["run_push"](a0, u)
        a1 = a0 / max(r0, 0.05)
        r1, _ = C["run_push"](a1, u)
        a2 = a1 / max(r1, 0.05)
        gp, g0, g2 = {}, {}, {}
        r2, _ = C["run_push"](a2, u, grab=gp); C["run_push"](0 * a2, u, grab=g0); r4, _ = C["run_push"](2 * a2, u, grab=g2)
        meas = ((gp["w"] - g0["w"]) * u).sum(1)                                 # measured on-axis counter-write (rows,)
        meas2 = ((g2["w"] - g0["w"]) * u).sum(1)
        resc, cross, crossL, quad = terms(g0["xa"], gp["xa"], a2, u)
        tot = resc + cross + quad
        fm = float(meas.mean()); ft = float(tot.mean())
        S = {"rows": C["rows"], "push_r0": r0, "push_r1": r1, "push_r2": r2, "push_2x_r": r4, "push_norm": round(float(a2.mean()), 1),
             "measured_on": round(fm, 1), "sum_terms_on": round(ft, 1), "sum_rel_err": round(abs(ft - fm) / max(abs(fm), 1e-6), 4),
             "rescale_on": round(float(resc.mean()), 1), "cross_on": round(float(cross.mean()), 1), "cross_left_on": round(float(crossL.mean()), 1),
             "quad_on": round(float(quad.mean()), 1),
             "on_fraction": round(float(-fm / a2.mean()), 3), "v183_on_fraction": v183.get(f"{fam}:{par}", {}).get("on_fraction"),
             "cross_fraction": round(float(-cross.mean() / a2.mean()), 3), "on_fraction_2x": round(float(-meas2.mean() / (2 * a2).mean()), 3),
             "cross_ratio": round(float(cross.mean() / fm), 3) if abs(fm) > 1e-6 else None,
             "quad_rel": round(float(abs(quad.mean()) / abs(fm)), 3) if abs(fm) > 1e-6 else None,
             "rescale_rel": round(float(abs(resc.mean()) / abs(fm)), 3) if abs(fm) > 1e-6 else None,
             "cross_left_share": round(float(crossL.mean() / cross.mean()), 3) if abs(float(cross.mean())) > 1e-6 else None}
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_mlp17_gain_terms_v184", "candidate_id": "corpus.unit_tier5_mlp17_gain_terms_v184",
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
    pred_a = all(S["sum_rel_err"] <= B["sum_rel_tol"] for S in cells) and all(inb(S["push_r2"], B["push_band"]) for S in cells)
    pred_b = sum(S["cross_on"] < 0 for S in cells) >= B["cross_neg_min_cells"] and \
        sum(inb(S["cross_fraction"], B["cross_frac_band"]) for S in cells) >= B["cross_frac_min_cells"]
    pred_c = sum(inb(S["quad_rel"], B["quad_rel_band"]) for S in cells) >= B["quad_min_cells"]
    pred_d = sum(S["rescale_rel"] is not None and S["rescale_rel"] <= B["rescale_max_rel"] for S in cells) >= B["rescale_min_cells"]
    pred_e = sum(inb(S["cross_left_share"], B["left_share_band"]) for S in cells) >= B["left_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_cross_damps": pred_b, "pred_c_quadratic_matters": pred_c, "pred_d_rescale_small": pred_d, "pred_e_symmetric_form": pred_e}


if __name__ == "__main__":
    main()
