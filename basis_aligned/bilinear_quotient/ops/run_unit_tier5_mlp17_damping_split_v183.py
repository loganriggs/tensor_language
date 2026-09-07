#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; push layer 17, two-step scale rule and seed fixed; split = projection on the unit answer axis.
"""v183: HOW does mlp:17 damp? v183: under a cue-free answer-axis push at t (after attn:17), clamping mlp:17's write to base costs
-0.05..-0.44 on 14/14, mlp:17's write change d17 projects on the axis at -6..-30% of the push, and its OFF-axis part is 4-10x larger
(up to 20k against a 50k residual). mlp:17 is the last unit before the final rms_norm, so its write can only act two ways: the ON-axis
counter-write (subtracts from the margin directly) and the OFF-axis part (inflates the residual norm, and rms_norm then scales every
logit — the margin included — by |x| / |x + off|). This rung splits d17 = on + off (on = its projection on the unit axis u) and clamps
mlp:17 at t to base + on, base + off, or base, under the same calibrated push (two rescale steps this time, so the push band can hold).
The off-axis effect has an ANALYTIC prediction with no free parameter: the keep-on arm's per-row margin times |x_on| / |x_on + off|,
where x_on is the final residual at t with on kept (captured), converted to recovery on the row's own axis.

Magnitudes printed before writing (v183 receipt): on-axis fraction -(d17.u)/push 0.06-0.30; off-axis norm / residual norm 0.025-0.40
(both/neither p1 0.40, both/either p1 0.17, both/neither p0 0.29, either/neither p1 0.15; the other ten <= 0.145); the shortfall
push cost + on-axis fraction is -0.15 on both/neither p1 and finiteness p1, |<= 0.07| elsewhere.

Smoke (CPU, coordination p0 + p1, 3 rows, 55 s): push after two rescales 0.999 / 1.000; base-clamp cost -0.128 / -0.077 (v182 -0.124 /
-0.101 on 3 rows); remove-on -0.118 / -0.089 vs on-fraction 0.102 / 0.082; remove-off -0.008 / +0.011 vs analytic 0.001 / 0.000
(off-axis 2.9k / 2.5k against a final residual of 106k / 90k — the norm ratio is 0.98 / 1.02); sum of the two = base clamp within 0.002.
Two code fixes before this smoke (the producer's forward calls attn and mlp separately, so the block hook never fired; prep passed
through the cell dict). No bar contradicted; the count bars cannot pass on 2 cells.

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar has a lower AND an upper bound where a fraction is compared):
  pred_a_instrument  the twice-rescaled push recovers 0.95-1.05 on 14/14 AND the base-clamp cost equals v183's within 0.05 on 14/14
  pred_b_on_axis     keeping only the off-axis part (removing on) costs <= -0.05 on >= 12 of 14 (the on-axis counter-write damps; prior 0.8)
  pred_c_off_by_norm keeping only the on-axis part (removing off) has a cost within +-0.05 of the analytic rms_norm prediction on >= 10 of 14
                     (the off-axis part acts only through the final norm; prior 0.5)
  pred_d_additive    cost(remove on) + cost(remove off) equals cost(remove both) within +-0.05 on >= 12 of 14 (prior 0.7)
  pred_e_on_fraction cost(remove on) is within +-0.05 of -(on-axis fraction of the push) on >= 10 of 14 (the on-axis counter-write cancels
                     exactly its share of the push, no gain; prior 0.5)
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
OUT = ROOT / "circuits/followups/unit_tier5_mlp17_damping_split_v183_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V182 = ROOT / "circuits/followups/unit_tier5_mlp17_reads_answer_axis_v182_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LAST = 17
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.05, "push_band": [0.95, 1.05], "on_max": -0.05, "on_min_cells": 12, "off_tol": 0.05, "off_min_cells": 10,
        "add_tol": 0.05, "add_min_cells": 12, "frac_tol": 0.05, "frac_min_cells": 10, "push_start_frac": 0.05, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_mlp17_damping_split_v183", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V183_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V183_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V183_SMOKE_CELLS", "2"))]
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
            if grab is not None:   # final residual at t = residual after attn:17 (push included) + mlp:17's (clamped) write; the producer's forward calls attn and mlp separately
                grab["x"] = torch.stack([cr[(rid, LAST)] for rid in batch.row_ids]) + (clamp.float() if clamp is not None else grab["w"])
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
    v182 = json.loads(V182.read_text())["measures"] if V182.exists() else {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        u, base_w = C["u"], C["capt17"]
        a0 = BARS["push_start_frac"] * C["rnorm"]
        r0, _ = C["run_push"](a0, u)
        a1 = a0 / max(r0, 0.05)
        r1, _ = C["run_push"](a1, u)
        a2 = a1 / max(r1, 0.05)                                                  # second rescale step
        gp, g0 = {}, {}
        r2, m_both = C["run_push"](a2, u, grab=gp); C["run_push"](0 * a2, u, grab=g0)
        d17 = gp["w"] - g0["w"]
        on = (d17 * u).sum(1, keepdim=True) * u; off = d17 - on
        frac = float(-((d17 * u).sum(1)).mean() / a2.mean())
        r_none, _ = C["run_push"](a2, u, clamp=base_w)
        gon = {}
        r_on, m_on = C["run_push"](a2, u, clamp=base_w + on, grab=gon)          # off removed
        r_off, _ = C["run_push"](a2, u, clamp=base_w + off)                      # on removed
        x_on = gon["x"]
        scale = (x_on.norm(dim=1) / (x_on + off).norm(dim=1)).tolist()          # final rms_norm: margin(x_on + off) = margin(x_on) * scale
        r_pred_both = g.recovery(C["prep"], [m * sc for m, sc in zip(m_on, scale)])
        cost = lambda r: round(1 - r / r2, 3)
        S = {"rows": C["rows"], "push_r0": r0, "push_r1": r1, "push_r2": r2, "push_alpha_frac": round(float((a2 / C["rnorm"]).mean()), 4),
             "cost_none": cost(r_none), "v182_push_cost17": v182.get(f"{fam}:{par}", {}).get("push_cost17"),
             "cost_remove_on": cost(r_off), "cost_remove_off": cost(r_on), "on_fraction": round(frac, 3),
             "off_pred_cost": round(1 - r_pred_both / r2, 3) if abs(r2) > 1e-6 else None,
             "off_pred_cost_note": "cost(remove off) predicted = 1 - r(keep both, predicted from keep-on margins x norm ratio) / r2; compare to cost_remove_off",
             "norm_ratio": round(float(sum(scale) / len(scale)), 4), "on_norm": round(float(on.norm(dim=1).mean()), 1),
             "off_norm": round(float(off.norm(dim=1).mean()), 1), "push_norm": round(float(a2.mean()), 1), "final_norm": round(float(x_on.norm(dim=1).mean()), 1)}
        S["off_measured_minus_pred"] = round(S["cost_remove_off"] - (1 - r_pred_both / r2), 3) if abs(r2) > 1e-6 else None
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_mlp17_damping_split_v183", "candidate_id": "corpus.unit_tier5_mlp17_damping_split_v183",
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
    pred_a = all(inb(S["push_r2"], B["push_band"]) for S in cells) and \
        all(S["v182_push_cost17"] is not None and abs(S["cost_none"] - S["v182_push_cost17"]) <= B["reproduce_tol"] for S in cells)
    pred_b = sum(S["cost_remove_on"] <= B["on_max"] for S in cells) >= B["on_min_cells"]
    pred_c = sum(S["off_measured_minus_pred"] is not None and abs(S["off_measured_minus_pred"]) <= B["off_tol"] for S in cells) >= B["off_min_cells"]
    pred_d = sum(abs(S["cost_remove_on"] + S["cost_remove_off"] - S["cost_none"]) <= B["add_tol"] for S in cells) >= B["add_min_cells"]
    pred_e = sum(abs(S["cost_remove_on"] + S["on_fraction"]) <= B["frac_tol"] for S in cells) >= B["frac_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_on_axis": pred_b, "pred_c_off_by_norm": pred_c, "pred_d_additive": pred_d, "pred_e_on_fraction": pred_e}


if __name__ == "__main__":
    main()
