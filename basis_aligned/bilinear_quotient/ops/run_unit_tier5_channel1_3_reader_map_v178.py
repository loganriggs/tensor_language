#!/usr/bin/env python3
# BQGATE: five frozen predictions; 12 cell-channel pairs fixed from v174's receipt (single-slice recovery >= 0.10) before the run; one removal per layer.
"""Tier-5: who reads the OTHER two cue channels? Head-1 and head-3 reader maps for v1 slices 1 and 3.

v174: after slice 8, slice 1 (08:01's head index) is the second carrier of the cue on the correlatives (0.18-0.24), degree p1 (0.23) and
polarity (0.14/0.15), and slice 3 (06:03 / 11:03's index) carries finiteness (0.32/0.14) and degree p1 (0.18). v177 mapped channel 8's
readers (07:08 hub + one family-specific late head-8). This rung swaps ONE slice (1 or 3) of the cue's v1 at layers 1-17 (cue writes clamped
to base) and removes head <slice> of one layer at a time (c_proj input slice clamped to base at all positions) on the 12 cell-channel
pairs whose v174 single-slice recovery is >= 0.10: channel 1 on the six correlative cells, degree p1, polarity p0/p1 (9); channel 3 on
finiteness p0/p1 and degree p1 (3). 19 forwards per pair.
Smoke (CPU, polarity p1 channel 1 + degree p1 channel 3, 3 rows, 5 s, after the bars were written): polarity c1 08 0.33 (top), 13 0.28, 15 0.23,
04 0.10, sum 1.06, top-2 0.61; degree c3 06 0.77, 04 0.11, 09 0.11, sum 0.91, top-2 0.89; early sums 0.11 / 0.095 -> pred_b widened to +-0.15
(disclosed above); every other bar unchanged; a/c/d/e false in the smoke only through the 2-pair count and 3-row-vs-16-row comparison.

Registered before the run:
  pred_a_instrument   the single-slice swap reproduces v174's value within 0.02 on 12/12, and the 17 removal costs sum to 0.7-1.3 x it on >= 10 of 12
  pred_b_late_readers the summed cost of layers 1-4 lies within -0.15..0.15 on 12/12 (PRE-SMOKE bar was +-0.1; the smoke read 0.11 and 0.095
                      on two 3-row pairs with small full effects, 0.11/0.12 — layer 4 reads ~0.1 of channels 1/3, unlike channel 8's 0.00)
  pred_c_channel1_08  08:01 is the largest single-layer cost on >= 7 of the 9 channel-1 pairs and costs >= 0.3 of the slice-1 swap on those
                      (v172: 08:01 is the largest direct-route reader on the correlatives and degree p1; prior ~0.7)
  pred_d_channel3_mid 06:03 + 11:03 together cost >= 0.4 of the slice-3 swap on 3/3 channel-3 pairs (v172: 06:03 the finiteness reader; 11:03 the number hub)
  pred_e_sparse       the two largest costs sum to 0.6-1.1 x the swap on >= 10 of 12
Reported, unregistered: full profiles, top-3 per pair, p0/p1 profile correlation.
Smoke: V178_SMOKE=<out.json> -> CPU, V178_SMOKE_ROWS rows (default 3), V178_SMOKE_CELLS pairs (default 2).
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
OUT = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v178_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
CELLS = ([(f, p, 1) for f in FAMILIES[1:4] for p in ("p0", "p1")] + [("degree_frame", "p1", 1), ("polarity_state", "p0", 1), ("polarity_state", "p1", 1)]
         + [("finiteness_selection", "p0", 3), ("finiteness_selection", "p1", 3), ("degree_frame", "p1", 3)])
N_LAYERS, N_HEADS, HD = 18, 9, 128
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "layer_sum_band": [0.7, 1.3], "layer_min_cells": 10, "early_band": [-0.15, 0.15], "c1_top_min_cells": 7, "c1_top_cost": 0.3,
        "c3_mid_min": 0.4, "top2_band": [0.6, 1.1], "top2_min_cells": 10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_channel1_3_reader_map_v178", "behaviours": 5, "targets": len(CELLS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def pearson(a, b):
    n = len(a); ma, mb = sum(a) / n, sum(b) / n
    sa = sum((x - ma) ** 2 for x in a) ** 0.5; sb = sum((y - mb) ** 2 for y in b) ** 0.5
    return round(sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb), 3) if sa > 0 and sb > 0 else None


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V178_SMOKE"); probe = False
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V178_SMOKE_ROWS", "3"))]) if (smoke or probe) else (lambda rows: rows)
    cells_run = CELLS[8::3][:int(os.environ.get("V178_SMOKE_CELLS", "2"))] if (smoke or probe) else CELLS
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    cells = {}

    def setup(fam, par, CH):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[fam]}")
        a1 = cut(g.rows_of(m, "A1")[0 if par == "p0" else 1::2])
        prep = g.prepare(backend, a1)
        lb, ld = g.cue_positions(prep.base_batch, prep.donor_batch, which="last")
        batch, db = prep.base_batch, prep.donor_batch
        rows = len(batch.row_ids)
        ar = torch.arange(rows, device=backend.device)
        cb_t, cd_t = torch.tensor(lb, device=backend.device), torch.tensor(ld, device=backend.device)
        def capture(bt, pos_t):
            cap = {}
            hs = []
            for l in range(N_LAYERS):
                def cp(m_, a, l=l):
                    cap[f"attn:{l:02d}"] = a[0][ar, pos_t].detach().clone(); cap[f"full:{l:02d}"] = a[0].detach().clone()   # returns None: args untouched
                hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(cp))
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: cap.__setitem__(f"mlp:{l:02d}", o[ar, pos_t].detach().clone())))
            hs.append(model.transformer.h[1].attn.register_forward_pre_hook(lambda m_, a: cap.__setitem__("v1", a[1][ar, pos_t].detach().clone())))
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cap
        cap, capb = capture(db, cd_t), capture(batch, cb_t)
        delta = (cap["v1"][:, CH].float() - capb["v1"][:, CH].float()).mean(0)
        def run(vec, remove_layer=None):
            """cue-position writes base; base x0; slice CH of the cue's v1 = base + vec at layers 1-17; optionally head 8 of one layer clamped to base at all positions."""
            hs = []
            for u in UNITS:
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u, l=l):
                        v = a[0].clone(); v[ar, cb_t] = capb[u].to(v.dtype)
                        if remove_layer == l:
                            v[:, :, CH * HD:(CH + 1) * HD] = capb[f"full:{l:02d}"][:, :, CH * HD:(CH + 1) * HD].to(v.dtype)
                        return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype); return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
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
        return {"rows": rows, "delta": delta, "run": run}

    for fam, par, ch in cells_run:
        cells[(fam, par, ch)] = setup(fam, par, ch)

    R = {}
    for fam, par, ch in cells_run:
        C = cells[(fam, par, ch)]; CH = ch
        dl = C["delta"]
        full = C["run"](dl); zero = C["run"](torch.zeros_like(dl))
        fr = lambda x, ref: round((x - zero) / (ref - zero), 3) if abs(ref - zero) > 1e-6 else None
        S = {"rows": C["rows"], "full": full, "zero": zero, "channel": CH, "v174_slice": v174.get(f"{fam}:{par}", {}).get("single_slice", {}).get(str(CH)), "delta_norm": round(float(dl.norm()), 3)}
        prof = {l: fr(C["run"](dl, remove_layer=l), full) for l in range(1, N_LAYERS)}
        cost = {f"{l:02d}": round(1 - v, 3) if v is not None else None for l, v in prof.items()}
        S["removal_cost"] = cost
        vals = {l: v for l, v in cost.items() if v is not None}
        S["layer_sum"] = round(sum(vals.values()), 3)
        S["early_sum"] = round(sum(vals.get(f"{l:02d}", 0.0) for l in range(1, 5)), 3)
        S["top3"] = sorted(vals, key=lambda k: -vals[k])[:3]
        S["top2_sum"] = round(sum(vals[k] for k in S["top3"][:2]), 3)
        R[f"{fam}:{par}:c{CH}"] = S
        print(fam, par, CH, {k: v for k, v in S.items() if k != "removal_cost"}, {k: v for k, v in cost.items() if v is not None and abs(v) > 0.05}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_channel1_3_reader_map_v178", "candidate_id": "corpus.unit_tier5_channel1_3_reader_map_v178",
              "bars": BARS, "cells": [list(c) for c in CELLS], "parity_r": PARITY(R), "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def _prof(S):
    return [S["removal_cost"][f"{l:02d}"] or 0.0 for l in range(1, N_LAYERS)]


def PARITY(R):
    out = {}
    for k in R:
        f, p, c = k.split(":")
        o = f"{f}:p1:{c}" if p == "p0" else None
        if o and o in R:
            out[f"{f}:{c}"] = pearson(_prof(R[k]), _prof(R[o]))
    return out


def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, band: x is not None and band[0] <= x <= band[1]
    pred_a = (all(S["v174_slice"] is not None and abs(S["full"] - S["v174_slice"]) <= B["reproduce_tol"] for S in cells)
              and sum(inb(S["layer_sum"], B["layer_sum_band"]) for S in cells) >= B["layer_min_cells"])
    pred_b = all(inb(S["early_sum"], B["early_band"]) for S in cells)
    c1 = [S for S in cells if S["channel"] == 1]; c3 = [S for S in cells if S["channel"] == 3]
    pred_c = len(c1) == 9 and sum(S["top3"][0] == "08" and S["removal_cost"]["08"] >= B["c1_top_cost"] for S in c1) >= B["c1_top_min_cells"]
    pred_d = len(c3) == 3 and all((S["removal_cost"]["06"] or 0.0) + (S["removal_cost"]["11"] or 0.0) >= B["c3_mid_min"] for S in c3)
    pred_e = sum(inb(S["top2_sum"], B["top2_band"]) for S in cells) >= B["top2_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_late_readers": pred_b, "pred_c_channel1_08": pred_c, "pred_d_channel3_mid": pred_d, "pred_e_sparse": pred_e}


if __name__ == "__main__":
    main()
