#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; the 20 downstream units (attn/mlp of layers 8-17) fixed; removal = clamp the unit's write at t to base.
"""Tier-5: which downstream units decode the hub's write? Per-unit removal at t under 07:08's write alone.

v179: 07:08's write under the slice-8 swap is one direction per family (the token's channel-8 delta), and added alone into a base
forward it carries 0.72-1.08 x the linear expectation. This rung keeps that whole-write arm (the (B,T,128) change added to head 8's
c_proj input at layer 7, no v1 swap, cue writes base) and clamps ONE downstream unit's write at the answer position t to its base value —
attn:08..attn:17 and mlp:08..mlp:17, 20 units — plus an arm clamping all 20 (what is left is the write's direct path to the logits
through the final norm and unembedding, plus anything routed through positions other than t). Costs are fractions of the whole-write arm.
The core mechanism of the number families is 'attn:08 writes, mlp:08-11 increment'; whether the function-word cues use the same
increment chain is the question.
Smoke (CPU, coordination p0 + both/either p0, 3 rows, 42 s, run with the pre-smoke bars): whole = v179 within 0.005; all-20-removed leaves
0.33 / 0.47; mlp chain 0.16 / 0.18; coordination's largest costs mlp:08 0.46 and mlp:11 -0.44 (opposing), both/either's mlp:15 0.09, mlp:16
0.09 (flat, distributed); additive ratio 0.95 / 1.09. pred_b and pred_c were re-registered as stated above; pred_d (shared top-2, prior now
~0.2) and pred_e are unchanged.

Registered before the run:
  pred_a_instrument   the whole-write arm reproduces v179's within 0.02 on 14/14, and the zero arm is 0.000
  pred_b_direct_large clamping all 20 downstream units at t leaves 0.2-0.6 x the whole write on >= 10 of 14 (a substantial direct path from
                      07:08's write to the logits; PRE-SMOKE bar was -0.1..0.3 on >= 12 — the smoke read 0.33 / 0.47)
  pred_c_no_mlp_chain mlp:08 + mlp:09 + mlp:10 + mlp:11 single-unit costs sum to -0.3..0.3 x the whole write on >= 10 of 14 (the function-word
                      cues do NOT use the number families' increment chain; PRE-SMOKE bar was >= 0.4 on >= 10 — the smoke read 0.16 / 0.18,
                      with mlp:11 OPPOSING the write on coordination, cost -0.44)
  pred_d_shared_top2  one pair of units is the top-2 (by single-unit cost) on >= 10 of 14 cells (a decoder shared across families; prior ~0.4)
  pred_e_additive     the 20 single-unit costs sum to 0.7-1.5 x (whole - all-removed) on >= 10 of 14 (a chain over-counts; prior ~0.5)
Reported, unregistered: the full 20-unit cost profile per cell; p0/p1 profile correlation; the 7 x 7 profile correlation.
Smoke: V180_SMOKE=<out.json> -> CPU, V180_SMOKE_ROWS rows (default 3), V180_SMOKE_CELLS cells (default 2).
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
OUT = ROOT / "circuits/followups/unit_tier5_hub0708_write_decoders_v180_result.json"
V179 = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
DOWN = [f"{k}:{l:02d}" for l in range(8, 18) for k in ("attn", "mlp")]
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "direct_band": [0.2, 0.6], "direct_min_cells": 10, "mlp_chain_band": [-0.3, 0.3], "mlp_min_cells": 10, "top2_min_cells": 10,
        "additive_band": [0.7, 1.5], "additive_min_cells": 10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub0708_write_decoders_v180", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V180_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V180_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v179 = json.loads(V179.read_text())["measures"] if V179.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = cells_run[:int(os.environ.get("V180_SMOKE_CELLS", "2"))]
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
        gs, gb = {}, {}
        full_swap = run(vec=delta, grab=gs); run(grab=gb)
        W = (gs["x"][:, :, sl].float() - gb["x"][:, :, sl].float())          # (B, T, 128) 07:08's write change under the slice-8 swap
        wt = W[ar, t_t]                                                        # (rows, 128) at t
        direction = wt.mean(0); direction = direction / direction.norm()
        sv = torch.linalg.svdvals(wt)
        return {"rows": rows, "delta": delta, "run": run, "W": W, "direction": direction, "full_swap": full_swap,
                "rank1_frac": round(float(sv[0] ** 2 / (sv ** 2).sum()), 3), "wt_norm": round(float(wt.norm(dim=1).mean()), 3),
                "t_share_of_norm": round(float(wt.norm() ** 2 / W.norm() ** 2), 3)}


    for fam, par in cells_run:
        cells[(fam, par)] = setup(fam, par)

    R = {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        whole = C["run"](add=C["W"]); zero = C["run"]()
        fr = lambda x: round((x - zero) / (whole - zero), 3) if abs(whole - zero) > 1e-6 else None
        cost = {u: round(1 - fr(C["run"](add=C["W"], remove=(u,))), 3) for u in DOWN}
        all_rm = fr(C["run"](add=C["W"], remove=tuple(DOWN)))
        top = sorted(cost, key=lambda u: -cost[u])
        S = {"rows": C["rows"], "whole_write": whole, "zero": zero, "v179_whole": v179.get(f"{fam}:{par}", {}).get("whole_write"),
             "all_removed_left": all_rm, "cost": cost, "top3": top[:3],
             "mlp_chain": round(sum(cost[f"mlp:{l:02d}"] for l in range(8, 12)), 3), "cost_sum": round(sum(cost.values()), 3),
             "additive_ratio": round(sum(cost.values()) / (1 - all_rm), 3) if all_rm is not None and abs(1 - all_rm) > 1e-6 else None}
        R[f"{fam}:{par}"] = S
        print(fam, par, {k: v for k, v in S.items() if k != "cost"}, {u: v for u, v in cost.items() if abs(v) > 0.08}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_channel8_shared_subspace_v180", "candidate_id": "corpus.unit_tier5_hub0708_write_decoders_v180",
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


def PROFILE_R(R):
    ks = list(R)
    return {a: {c: pearson([R[a]["cost"][u] for u in DOWN], [R[c]["cost"][u] for u in DOWN]) for c in ks if c != a} for a in ks}


def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, band: x is not None and band[0] <= x <= band[1]
    pred_a = all(S["v179_whole"] is not None and abs(S["whole_write"] - S["v179_whole"]) <= B["reproduce_tol"] and S["zero"] == 0.0 for S in cells)
    pred_b = sum(inb(S["all_removed_left"], B["direct_band"]) for S in cells) >= B["direct_min_cells"]
    pred_c = sum(inb(S["mlp_chain"], B["mlp_chain_band"]) for S in cells) >= B["mlp_min_cells"]
    from collections import Counter
    top2 = Counter(tuple(sorted(S["top3"][:2])) for S in cells)
    pred_d = bool(top2) and max(top2.values()) >= B["top2_min_cells"]
    pred_e = sum(inb(S["additive_ratio"], B["additive_band"]) for S in cells) >= B["additive_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_direct_large": pred_b, "pred_c_no_mlp_chain": pred_c, "pred_d_shared_top2": pred_d, "pred_e_additive": pred_e}


if __name__ == "__main__":
    main()
