#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; push layer 17, scale rule and seed fixed; removal = clamp mlp:17's write at t to base.
"""v182: WHAT does mlp:17 read? v182: clamping mlp:17's write at the answer position t to base under the FULL cue interchange costs
-0.05..-0.34 on 12/14 cells, and the opposition is dose-linear in 07:08's write — a late damper. Two readings: (i) mlp:17 reads the
answer-axis component of the residual at t and writes against it (a margin damper, blind to the cue), or (ii) it reads cue-derived
features that happen to co-vary with the answer. Test: push the BASE batch's residual at t, right after attn:17 (`resid_add` at layer 17,
so only mlp:17 and the readout see it), along the per-row unembedding answer axis u = U[donor answer] - U[base answer], with the scale
calibrated by two forwards so the push alone recovers ~1.0 of the interchange; then clamp mlp:17 at t to base under the push. A
same-norm push along a seeded random direction orthogonal to u is the control (it should move the margin ~0). v182's full-interchange
arm is re-run as the instrument bar.

Magnitudes printed before writing (v182 receipt): full-interchange mlp:17 cost -0.02..-0.34 (median -0.11); v157: residual norm at
layer 11 ~25,000 — the push starts at 5% of the row's residual norm at t and is rescaled once by 1 / recovery.

Smoke (CPU, coordination p0 + p1, 3 rows, 50 s): calibrated push recovers 1.008 / 0.995 (push norm 3.7k / 3.3k = 6% of the residual norm
60k / 55k at t after attn:17); push mlp:17 cost -0.13 / -0.08 (full-interchange -0.015 / -0.057 on 3 rows); orthogonal push -0.005 / 0.000;
mlp:17's write change projected on the axis -376 / -269 (about 10% of the push, sign against it; its off-axis norm 2.9k). No bar
contradicted; pred_a's v181 comparison fails only on the 3-row subset.

REGISTERED BEFORE THE RUN (bars in BARS; every fraction bar has a lower AND an upper bound where a fraction is compared):
  pred_a_instrument   full-interchange mlp:17 cost equals v182's within 0.02 on 14/14 AND the calibrated push recovers 0.9-1.1 on 14/14
  pred_b_margin_keyed under the answer-axis push, clamping mlp:17 at t to base costs <= -0.05 on >= 9 of 14 (mlp:17 damps the answer
                      axis itself, with no cue in sight; prior 0.5)
  pred_c_same_size    |push cost - full-interchange cost| <= 0.15 on >= 8 of 14 (the same damper whatever the source; prior 0.35)
  pred_d_orthogonal   the orthogonal same-norm push moves the margin by |recovery| <= 0.1 on >= 12 of 14 (the push is axis-specific; a
                      control that can fail; prior 0.7)
  pred_e_write_sign   mlp:17's write change at t under the push, projected on the unit answer axis and averaged over rows, is < 0 on
                      >= 12 of 14 (the opposition is on the answer axis directly; prior 0.8)
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
OUT = ROOT / "circuits/followups/unit_tier5_mlp17_reads_answer_axis_v182_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V181 = ROOT / "circuits/followups/unit_tier5_mlp17_damper_v181_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
LAST = 17
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "push_band": [0.9, 1.1], "damper_max": -0.05, "damper_min_cells": 9, "same_tol": 0.15, "same_min_cells": 8,
        "orth_max_abs": 0.1, "orth_min_cells": 12, "sign_min_cells": 12, "push_start_frac": 0.05, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_mlp17_reads_answer_axis_v182", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V182_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V182_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = [(FAMILIES[0], "p0"), (FAMILIES[0], "p1")][:int(os.environ.get("V182_SMOKE_CELLS", "2"))]
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
        def run_push(alpha, dirn, remove=(), grab=None):
            """base batch + alpha (rows,) * dirn (rows, N_EMBD) added at t after attn:17; mlp:17 clamped to base if in remove."""
            hs = []
            if "mlp:17" in remove or grab is not None:
                def mh(m_, a, o):
                    if grab is not None: grab["w"] = o[ar, t_t].detach().float().clone()
                    if "mlp:17" in remove:
                        o = o.clone(); o[ar, t_t] = capt["mlp:17"].to(o.dtype); return o
                hs.append(model.transformer.h[LAST].mlp.Down.register_forward_hook(mh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache,
                                          resid_add={LAST: alpha[:, None] * dirn})
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        gs, gb = {}, {}
        full_swap = run(vec=delta, grab=gs); run(grab=gb)
        W = (gs["x"][:, :, sl].float() - gb["x"][:, :, sl].float())          # (B, T, 128) 07:08's write change under the slice-8 swap
        wt = W[ar, t_t]                                                        # (rows, 128) at t
        direction = wt.mean(0); direction = direction / direction.norm()
        sv = torch.linalg.svdvals(wt)
        return {"rows": rows, "delta": delta, "run": run, "run_donor": run_donor, "run_push": run_push, "u": u, "rnd": rnd, "rnorm": rnorm, "W": W, "direction": direction, "full_swap": full_swap,
                "rank1_frac": round(float(sv[0] ** 2 / (sv ** 2).sum()), 3), "wt_norm": round(float(wt.norm(dim=1).mean()), 3),
                "t_share_of_norm": round(float(wt.norm() ** 2 / W.norm() ** 2), 3)}


    for fam, par in cells_run:
        cells[(fam, par)] = setup(fam, par)

    R = {}
    v181 = json.loads(V181.read_text())["measures"] if V181.exists() else {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        full17 = round(1 - C["run_donor"](remove=("mlp:17",)), 3)
        a0 = BARS["push_start_frac"] * C["rnorm"]
        r0 = C["run_push"](a0, C["u"])
        a1 = a0 / max(r0, 0.05)
        r1 = C["run_push"](a1, C["u"])
        rm = C["run_push"](a1, C["u"], remove=("mlp:17",))
        push17 = round(1 - rm / r1, 3) if abs(r1) > 1e-6 else None
        orth = C["run_push"](a1, C["rnd"])
        gp, g0 = {}, {}
        C["run_push"](a1, C["u"], grab=gp); C["run_push"](0 * a1, C["u"], grab=g0)
        d17 = gp["w"] - g0["w"]
        proj = round(float(((d17 * C["u"]).sum(1)).mean()), 3)
        S = {"rows": C["rows"], "full_cost17": full17, "v181_full_cost17": v181.get(f"{fam}:{par}", {}).get("full_cost", {}).get("mlp:17"),
             "push_r0": r0, "push_r1": r1, "push_alpha_frac": round(float((a1 / C["rnorm"]).mean()), 4), "push_removed": rm, "push_cost17": push17,
             "orth_recovery": orth, "d17_on_axis": proj, "d17_norm": round(float(d17.norm(dim=1).mean()), 2),
             "push_norm": round(float(a1.mean()), 1), "resid_norm": round(float(C["rnorm"].mean()), 1)}
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_mlp17_reads_answer_axis_v182", "candidate_id": "corpus.unit_tier5_mlp17_reads_answer_axis_v182",
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
    pred_a = all(S["v181_full_cost17"] is not None and abs(S["full_cost17"] - S["v181_full_cost17"]) <= B["reproduce_tol"] for S in cells) \
        and all(inb(S["push_r1"], B["push_band"]) for S in cells)
    pred_b = sum(S["push_cost17"] is not None and S["push_cost17"] <= B["damper_max"] for S in cells) >= B["damper_min_cells"]
    pred_c = sum(S["push_cost17"] is not None and abs(S["push_cost17"] - S["full_cost17"]) <= B["same_tol"] for S in cells) >= B["same_min_cells"]
    pred_d = sum(abs(S["orth_recovery"]) <= B["orth_max_abs"] for S in cells) >= B["orth_min_cells"]
    pred_e = sum(S["d17_on_axis"] < 0 for S in cells) >= B["sign_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_margin_keyed": pred_b, "pred_c_same_size": pred_c, "pred_d_orthogonal": pred_d, "pred_e_write_sign": pred_e}


if __name__ == "__main__":
    main()
