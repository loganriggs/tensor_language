#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; push layer 17, two-step scale rule fixed; gate vectors are exact algebra on the weights, no fit; token-class controls seeded.
"""v186: mlp:17's damper is PER-TOKEN SELF-SATURATION of frequent function-word tokens. v185: the damping of an answer-axis push at t is
s^2 * alpha * (x . f_u) with the gate vector f_u = L^T((D^T u)*(R u)) + R^T((D^T u)*(L u)), and the gate is carried by the u-orthogonal
CONTEXT written by the behaviour's own units. This rung asks what the context part of f_u is, in weight space: (i) its alignment with the
SLOT direction s = U[ans] + U[foil] (both candidates up); (ii) the single-token gate f(U[a]) for each answer token alone — is it
anti-aligned with U[a] itself (a token that is already promoted has a further push on it damped: saturation); (iii) the same for token
classes (random / frequent id<1000 / rare id>20000) as controls; (iv) causal: a push at t along U[ans] alone, along U[foil] alone, and along
a rare token's unembedding, each at the pair-calibrated alpha — the on-axis counter-fraction of each; (v) the pair gate decomposed exactly
as f(a-b) = f(a) + f(b) - X(a,b) (X the symmetric cross-token form): how much of the pair damping is the two self-terms vs the pair-tuned
cross term.

Magnitudes printed before writing (CPU probes, weights + 3-row cells): cos(f_perp, slot) = -0.249 / -0.431 / -0.606 / -0.463 / -0.528 /
-0.553 / -0.208 on the seven task pairs; both answer tokens rank in the bottom 10 of the vocabulary under U f_perp on 7/7; the slot
component of the context carries 0.46-0.84 (coordination), 0.68-1.1 (degree), 0.17-0.32 (polarity), 0.71-0.94 (correlative_both_neither) of
the gate; 48 CROSS-family pairs of the same 11 tokens: cos -0.28..-0.65 on 48/48 (so it is the tokens, not the pairs); random pairs (200):
mean +0.036, 4% below -0.15; single tokens: cos(f(U[a]), U[a]) = -0.37..-0.75 on 11/11 task tokens with |f| 41-206, vs random tokens mean
+0.092, |f| 32, rank(a) in the bottom 0.1% on 6.7% (random), 32% (id<1000), 27% (1000-5000), 9% (5k-20k), 1.7% (>20k); unembedding norm
does not explain it (bottom-0.1% fraction 6-10% across |U| quantiles). Causal (alpha = 0.06|x|, 3 rows): counter-fraction on a push along
U[ans] 0.31-0.37 / 0.14-0.17 / 0.20-0.27 (coordination p0 / polarity p0 / degree p1), along U[foil] 0.19-0.29 / 0.16-0.21 / 0.00-0.04,
along a rare token 0.05-0.10 / 0.06-0.15 / 0.02-0.05, the pair axis 0.17-0.23 / 0.06-0.08 / 0.01-0.06. Decomposition: the two self-terms
alone are 2.6 / 3.7 / 0.9 x the pair gate (the cross-token term cancels 60-70% of them on two cells, adds on degree p1).

Smoke (CPU, coordination p0 + p1, 3 rows): push_r2 0.999 / 1.000, identity_err 2e-6; slot_cos -0.249 / -0.249, slot_share 0.595 / 0.703; both
tokens rank 0.9999 under U f_perp; self_cos -0.606 / -0.599 (|f| 167 / 176); class controls random 0.067 (mean cos +0.09), frequent 0.323
(-0.02), rare 0.023 (+0.13); at the pair-calibrated alpha (3650 / 3310): ans_frac 0.344 / 0.255, foil 0.227 / 0.138, rare 0.041 / 0.069,
ans/rare 8.4 / 3.7, pair 0.104 / 0.081; self_terms_ratio 2.64 / 5.91. All within the registered bands except the pred_e ratio on p1 (see pred_e).

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar has a lower AND an upper bound where a fraction is compared):
  pred_a_instrument      pair push_r2 in 0.95-1.05 on 14/14 AND the decomposition identity f(a-b) = f(a) + f(b) - X reproduces the gate
                         within 1e-3 relative on 14/14 (algebra check of the code path).
  pred_b_slot_gate       cos(f_perp, slot_perp) <= -0.15 on >= 12 of 14 AND the slot component of the context carries 0.3-1.2 of the gate on
                         >= 10 of 14 (polarity's two cells are expected to fail the share bar at 0.17-0.32).
  pred_c_self_saturation both answer tokens of the cell have cos(f(U[a]), U[a]) <= -0.3 on >= 12 of 14; token-class controls (300 tokens each,
                         seed 0; quantity = fraction whose own token ranks in the bottom 0.1% under U f(U[a])): random in 0.02-0.15,
                         id<1000 in 0.15-0.50, id>20000 in 0.00-0.05 (a saturation of frequent tokens, absent for rare ones).
  pred_d_causal_single   counter-fraction of a push along U[ans] alone, at the pair-calibrated alpha, in 0.10-0.50 on >= 12 of 14; along a
                         seeded rare token (id>20000) in -0.05..0.15 on >= 12 of 14; ans/rare ratio in 1.5-15 on >= 10 of 14.
  pred_e_self_terms_over the two self-terms alone over-damp: |x.(f(a)+f(b))| / |x.f(a-b)| in 1.2-10.0 on >= 10 of 14 (the pair-tuned cross
                         term X cancels part of the per-token saturation; degree p1 read 0.9 in the probe and is expected to fail).
                         Pre-smoke band was 1.2-5.0; the smoke read 5.9 on coordination p1 (pair gate -98k vs self terms -579k: the ratio is
                         ill-conditioned where the pair gate is small), so the upper bound is raised to 10 BEFORE the run and disclosed here.
  Reported, not predicted: foil-push counter-fraction, the cross-token share, |f| per token, gate align, rank of both tokens under U f_perp.
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
OUT = ROOT / "circuits/followups/unit_tier5_mlp17_self_saturation_v186_result.json"
V184 = ROOT / "circuits/followups/unit_tier5_mlp17_gain_terms_v184_result.json"
V185 = ROOT / "circuits/followups/unit_tier5_mlp17_gate_anatomy_v185_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V183 = ROOT / "circuits/followups/unit_tier5_mlp17_damping_split_v183_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LAST = 17
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"push_band": [0.95, 1.05], "identity_tol": 1e-3, "slot_cos_max": -0.15, "slot_cos_min_cells": 12, "slot_share_band": [0.3, 1.2], "slot_share_min_cells": 10,
        "self_cos_max": -0.3, "self_min_cells": 12, "class_n": 300, "class_random_band": [0.02, 0.15], "class_frequent_band": [0.15, 0.50], "class_rare_band": [0.0, 0.05],
        "frequent_max_id": 1000, "rare_min_id": 20000, "bottom_quantile": 0.999,
        "ans_frac_band": [0.10, 0.50], "ans_min_cells": 12, "rare_frac_band": [-0.05, 0.15], "rare_min_cells": 12, "ratio_band": [1.5, 15.0], "ratio_min_cells": 10,
        "self_terms_band": [1.2, 10.0], "self_terms_min_cells": 10, "push_start_frac": 0.05, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_mlp17_self_saturation_v186", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V186_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V186_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V186_SMOKE_CELLS", "2"))]
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
    U = model.lm_head.weight.detach().float(); V = U.shape[0]
    nz = lambda v: v / v.norm(dim=-1, keepdim=True)
    def fgate(u):
        """gate vector f_u: cross_on = s^2 * alpha * (x . f_u); quadratic (even) in u."""
        Du = u @ Dw
        return ((Du * (u @ Rw.T)) @ Lw) + ((Du * (u @ Lw.T)) @ Rw)
    def xform(a, b):
        """symmetric cross-token form: f(a - b) = f(a) + f(b) - xform(a, b)."""
        Da, Db = a @ Dw, b @ Dw
        return ((Da * (b @ Rw.T)) @ Lw) + ((Da * (b @ Lw.T)) @ Rw) + ((Db * (a @ Rw.T)) @ Lw) + ((Db * (a @ Lw.T)) @ Rw)
    def bottom_frac(ids):
        """fraction of tokens whose own unembedding ranks in the bottom (1 - bottom_quantile) of the vocabulary under U f(U[a])."""
        u = nz(U[ids]); lg = nz(fgate(u)) @ U.T
        r = torch.stack([(lg[i] > lg[i, ids[i]]).sum() for i in range(len(ids))]).float() / V
        return round(float((r > BARS["bottom_quantile"]).float().mean()), 3), round(float((nz(fgate(u)) * u).sum(1).mean()), 3)
    gen = torch.Generator(device="cpu").manual_seed(BARS["seed"])
    n_cls = BARS["class_n"]
    cls_random = bottom_frac(torch.randint(0, V, (n_cls,), generator=gen).to(backend.device))
    cls_frequent = bottom_frac(torch.randint(0, BARS["frequent_max_id"], (n_cls,), generator=gen).to(backend.device))
    cls_rare = bottom_frac(torch.randint(BARS["rare_min_id"], V, (n_cls,), generator=gen).to(backend.device))
    CLS = {"random": cls_random, "frequent": cls_frequent, "rare": cls_rare}
    print("token-class controls (bottom-0.1% fraction, mean cos(f(U[a]), U[a])):", CLS, flush=True)
    v185 = json.loads(V185.read_text())["measures"] if V185.exists() else {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        u = C["u"]; batch = C["prep"].base_batch
        ans = torch.tensor(list(batch.answer_ids), device=backend.device); foil = torch.tensor(list(batch.foil_ids), device=backend.device)
        a0 = BARS["push_start_frac"] * C["rnorm"]
        r0, _ = C["run_push"](a0, u)
        a1 = a0 / max(r0, 0.05)
        r1, _ = C["run_push"](a1, u)
        a2 = a1 / max(r1, 0.05)
        gp, g0 = {}, {}
        r2, _ = C["run_push"](a2, u, grab=gp); C["run_push"](0 * a2, u, grab=g0)
        x = g0["xa"].float()
        A, Bv = U[foil], U[ans]                                                 # u = (U[foil] - U[ans]) / n
        n = (A - Bv).norm(dim=1, keepdim=True)
        fu = fgate(u); fa, fb, X = fgate(A / n), fgate(Bv / n), xform(A / n, Bv / n)
        identity_err = float(((fu - (fa + fb - X)).norm(dim=1) / fu.norm(dim=1)).max())
        fp = fu - (fu * u).sum(1, keepdim=True) * u; fph = nz(fp)
        xp = x - (x * u).sum(1, keepdim=True) * u
        G = (xp * fph).sum(1)                                                   # context gate (rows,)
        s = A + Bv; s = s - (s * u).sum(1, keepdim=True) * u; sh = nz(s)
        slot_cos = float((fph * sh).sum(1).mean())
        slot_share = float((((xp * sh).sum(1) * (sh * fph).sum(1)) / G).mean())
        lg = fph @ U.T
        rank_ans = float((torch.stack([(lg[i] > lg[i, ans[i]]).sum() for i in range(len(ans))]).float() / V).mean())
        rank_foil = float((torch.stack([(lg[i] > lg[i, foil[i]]).sum() for i in range(len(ans))]).float() / V).mean())
        ua, uf = nz(U[ans]), nz(U[foil])
        self_cos_ans = float((nz(fgate(ua)) * ua).sum(1).mean()); self_cos_foil = float((nz(fgate(uf)) * uf).sum(1).mean())
        f_norm_ans = float(fgate(ua).norm(dim=1).mean()); f_norm_foil = float(fgate(uf).norm(dim=1).mean())
        # causal single-token pushes at the pair-calibrated alpha; counter-fraction = -(counter-write . d) / alpha
        rare_ids = torch.randint(BARS["rare_min_id"], V, (C["rows"],), generator=gen).to(backend.device)
        fr = {}
        for lab, d in (("ans", ua), ("foil", uf), ("rare", nz(U[rare_ids]))):
            gd = {}
            C["run_push"](a2, d, grab=gd)
            fr[lab] = float((-((gd["w"] - g0["w"]) * d).sum(1) / a2).mean())
        pair_frac = float((-((gp["w"] - g0["w"]) * u).sum(1) / a2).mean())
        self_terms = float((x * (fa + fb)).sum(1).mean()); pair_gate = float((x * fu).sum(1).mean()); cross_term = float((x * X).sum(1).mean())
        self_terms_ratio = abs(self_terms) / max(abs(pair_gate), 1e-6)
        S = {"rows": C["rows"], "push_r0": r0, "push_r1": r1, "push_r2": r2, "push_norm": round(float(a2.mean()), 1), "identity_err": round(identity_err, 6),
             "gate": round(float(G.mean()), 1), "v185_gate": v185.get(f"{fam}:{par}", {}).get("gate"),
             "slot_cos": round(slot_cos, 3), "slot_share": round(slot_share, 3), "rank_ans_under_fperp": round(rank_ans, 5), "rank_foil_under_fperp": round(rank_foil, 5),
             "self_cos_ans": round(self_cos_ans, 3), "self_cos_foil": round(self_cos_foil, 3), "f_norm_ans": round(f_norm_ans, 1), "f_norm_foil": round(f_norm_foil, 1),
             "ans_id": int(ans[0]), "foil_id": int(foil[0]), "rare_ids": [int(i) for i in rare_ids[:3]],
             "pair_frac": round(pair_frac, 3), "ans_frac": round(fr["ans"], 3), "foil_frac": round(fr["foil"], 3), "rare_frac": round(fr["rare"], 3),
             "ans_over_rare": round(fr["ans"] / fr["rare"], 3) if abs(fr["rare"]) > 1e-6 else None,
             "self_terms": round(self_terms, 1), "cross_term": round(cross_term, 1), "pair_gate": round(pair_gate, 1), "self_terms_ratio": round(self_terms_ratio, 3)}
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    R["_controls"] = CLS
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_mlp17_self_saturation_v186", "candidate_id": "corpus.unit_tier5_mlp17_self_saturation_v186",
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
    CLS = R["_controls"]
    cells = [S for k, S in R.items() if not k.startswith("_")]
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    pred_a = all(inb(S["push_r2"], B["push_band"]) for S in cells) and all(S["identity_err"] <= B["identity_tol"] for S in cells)
    pred_b = sum(S["slot_cos"] <= B["slot_cos_max"] for S in cells) >= B["slot_cos_min_cells"] and sum(inb(S["slot_share"], B["slot_share_band"]) for S in cells) >= B["slot_share_min_cells"]
    pred_c = sum(S["self_cos_ans"] <= B["self_cos_max"] and S["self_cos_foil"] <= B["self_cos_max"] for S in cells) >= B["self_min_cells"] and \
        inb(CLS["random"][0], B["class_random_band"]) and inb(CLS["frequent"][0], B["class_frequent_band"]) and inb(CLS["rare"][0], B["class_rare_band"])
    pred_d = sum(inb(S["ans_frac"], B["ans_frac_band"]) for S in cells) >= B["ans_min_cells"] and sum(inb(S["rare_frac"], B["rare_frac_band"]) for S in cells) >= B["rare_min_cells"] and \
        sum(inb(S["ans_over_rare"], B["ratio_band"]) for S in cells) >= B["ratio_min_cells"]
    pred_e = sum(inb(S["self_terms_ratio"], B["self_terms_band"]) for S in cells) >= B["self_terms_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_slot_gate": pred_b, "pred_c_self_saturation": pred_c, "pred_d_causal_single": pred_d, "pred_e_self_terms_over": pred_e}


if __name__ == "__main__":
    main()
