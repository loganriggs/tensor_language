#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed (v170-v176's seven function-word families x 2 parities); one removal per layer; bars written before the run.
"""Tier-5: which head-8 reads each family's channel-8 cue component? A 14-cell reader map.

v172-v176 established that a function-word cue's identity rides block 0's value residual, channel 8 (07:08/14:08/16:08's head index)
being the largest carrier on 13/14 cells (v174), and that v1 channel 8 can be read only by head 8 of layers 1-17 (architecture). v176
gave the removal profile of three families: polarity 07 0.74/0.53 + 16 0.31/0.38; both/neither 07 0.35/0.34 + 14 0.15/0.30 + 16 0.44/0.26;
either/neither 14 0.40/0.44 + 16 0.35/0.30 + 07 0.13/0.09. This rung swaps the full slice-8 delta at the cue (layers 1-17 clamped, cue
writes base, v174's instrument) and removes head 8 of ONE layer at a time (c_proj input slice clamped to base at all positions) on all 14
cells: 17 + 2 forwards per cell. Parity swaps base and donor, so the p1 delta is exactly -(p0 delta): the reader set should be a property
of the family, not of the interchange direction.
Smoke (CPU, 2 cells coordination p0 + either/neither p1, 3 rows, 6 s, after the bars were written): layer sums 1.06 / 1.03; early 0.00 / -0.01;
coordination 07 0.72 + 16 0.21 (+06 0.10); either/neither 14 0.41 + 16 0.30 (+13 0.14, 07 0.10); top-2 0.93 / 0.71; 16:08 at 0.21 on
coordination sits on the pred_c bar. a/c/e false in the smoke only through the 2-cell count and the 3-row-vs-16-row comparison. Bars unchanged.

Registered before the run:
  pred_a_instrument     the full swap reproduces v174's slice-8 value within 0.02 on 14/14, and the 17 single-layer removal costs sum to
                        0.7-1.3 x the full swap on >= 12 of 14 (cross-layer additivity, v176 6/6)
  pred_b_late_readers   the summed cost of layers 1-4 lies within -0.1..0.1 on 14/14 (v174: the index-8 channel is read only at layers >= 5)
  pred_c_common_16      16:08 removal costs >= 0.2 of the full swap on >= 12 of 14 cells (the one head-8 shared by all three v176 families;
                        prior ~0.5 for the four untested families)
  pred_d_sparse         the two largest costs sum to 0.6-1.1 x the full swap on 14/14 (two readers per family)
  pred_e_parity         the 17-layer profile has Pearson r >= 0.8 between p0 and p1 on >= 6 of 7 families
Reported, unregistered: the 7 x 7 cross-family profile correlation per parity; top-3 readers per cell; the layer-11 head 8 cost.
Smoke: V177_SMOKE=<out.json> -> CPU, V177_SMOKE_ROWS rows (default 3), V177_SMOKE_CELLS cells (default 2).
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
OUT = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
CELLS = [(f, p) for f in FAMILIES for p in ("p0", "p1")]
N_LAYERS, N_HEADS, CH, HD = 18, 9, 8, 128
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "layer_sum_band": [0.7, 1.3], "layer_min_cells": 12, "early_band": [-0.1, 0.1], "l16_min": 0.2, "l16_min_cells": 12,
        "top2_band": [0.6, 1.1], "parity_r_min": 0.8, "parity_min_fams": 6}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_channel8_reader_map_v177", "behaviours": len(FAMILIES), "targets": len(CELLS),
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
    smoke = os.environ.get("V177_SMOKE"); probe = False
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V177_SMOKE_ROWS", "3"))]) if (smoke or probe) else (lambda rows: rows)
    cells_run = CELLS[::7][:int(os.environ.get("V177_SMOKE_CELLS", "2"))] if (smoke or probe) else CELLS
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    cells = {}

    def setup(fam, par):
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

    for fam, par in cells_run:
        cells[(fam, par)] = setup(fam, par)

    R = {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        dl = C["delta"]
        full = C["run"](dl); zero = C["run"](torch.zeros_like(dl))
        fr = lambda x, ref: round((x - zero) / (ref - zero), 3) if abs(ref - zero) > 1e-6 else None
        S = {"rows": C["rows"], "full": full, "zero": zero, "v174_slice8": v174.get(f"{fam}:{par}", {}).get("single_slice", {}).get(str(CH)), "delta_norm": round(float(dl.norm()), 3)}
        prof = {l: fr(C["run"](dl, remove_layer=l), full) for l in range(1, N_LAYERS)}
        cost = {f"{l:02d}": round(1 - v, 3) if v is not None else None for l, v in prof.items()}
        S["removal_cost"] = cost
        vals = {l: v for l, v in cost.items() if v is not None}
        S["layer_sum"] = round(sum(vals.values()), 3)
        S["early_sum"] = round(sum(vals.get(f"{l:02d}", 0.0) for l in range(1, 5)), 3)
        S["top3"] = sorted(vals, key=lambda k: -vals[k])[:3]
        S["top2_sum"] = round(sum(vals[k] for k in S["top3"][:2]), 3)
        R[f"{fam}:{par}"] = S
        print(fam, par, {k: v for k, v in S.items() if k != "removal_cost"}, {k: v for k, v in cost.items() if v is not None and abs(v) > 0.05}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_channel8_reader_map_v177", "candidate_id": "corpus.unit_tier5_channel8_reader_map_v177",
              "bars": BARS, "families": list(FAMILIES), "channel": CH, "cross_family_r": CROSS(R), "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def CROSS(R):
    out = {}
    for p in ("p0", "p1"):
        ks = [k for k in R if k.endswith(p)]
        out[p] = {a: {c: pearson([R[a]["removal_cost"][f"{l:02d}"] or 0.0 for l in range(1, N_LAYERS)], [R[c]["removal_cost"][f"{l:02d}"] or 0.0 for l in range(1, N_LAYERS)]) for c in ks if c != a} for a in ks}
    return out


def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, band: x is not None and band[0] <= x <= band[1]
    pred_a = (all(S["v174_slice8"] is not None and abs(S["full"] - S["v174_slice8"]) <= B["reproduce_tol"] for S in cells)
              and sum(inb(S["layer_sum"], B["layer_sum_band"]) for S in cells) >= B["layer_min_cells"])
    pred_b = all(inb(S["early_sum"], B["early_band"]) for S in cells)
    pred_c = sum(S["removal_cost"]["16"] is not None and S["removal_cost"]["16"] >= B["l16_min"] for S in cells) >= B["l16_min_cells"]
    pred_d = all(inb(S["top2_sum"], B["top2_band"]) for S in cells)
    fams_ok = 0
    for f in FAMILIES:
        a, c = R.get(f"{f}:p0"), R.get(f"{f}:p1")
        if a and c:
            r = pearson([a["removal_cost"][f"{l:02d}"] or 0.0 for l in range(1, N_LAYERS)], [c["removal_cost"][f"{l:02d}"] or 0.0 for l in range(1, N_LAYERS)])
            fams_ok += r is not None and r >= B["parity_r_min"]
    pred_e = fams_ok >= B["parity_min_fams"]
    return {"pred_a_instrument": pred_a, "pred_b_late_readers": pred_b, "pred_c_common_16": pred_c, "pred_d_sparse": pred_d, "pred_e_parity": pred_e}


if __name__ == "__main__":
    main()
