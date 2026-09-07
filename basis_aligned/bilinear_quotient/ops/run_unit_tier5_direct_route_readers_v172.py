#!/usr/bin/env python3
# BQGATE: five frozen predictions; families fixed (v170/v171's seven function-word cells); all 162 heads scanned, none chosen after the fact.
"""Tier-5: WHO reads the function-word direct route -- heads by removal, and the block-0 value residual (v1) as a carrier.

v171: with all 36 cue-position writes clamped to base, the donor cue's x0 = rms_norm(wte) re-entered at all blocks recovers
0.32-0.86 (= v170's direct route); block 0 alone carries 0.30-0.52 of it (degree p0 0.15), blocks <= 4 cumulatively 0.74-0.94,
re-entry after block 4 <= 0.1. I read that as "picked up by attention at blocks 0-4". One carrier I had NOT accounted for:
bilin18's attention keeps the BLOCK-0 VALUE RESIDUAL -- v = (1 - lamb) * c_v(x) + lamb * v1 at every block, with v1 = block 0's
c_v(rms_norm(12.19 * x0)) -- so the cue position's block-0 value, a function of the donor x0 only, is readable by a head at ANY
layer (a hub head at 7/8/11 attending to the cue position sees donor v1 there even when the residual is base). That would make
"block 0 alone 0.30-0.52" a value-residual read by late hubs (the v164 'value read'), not an early read. This rung separates the two.

Arms per cell (7 families x 2 parities, 16 rows; all cue-position writes base; donor x0 at the cue position at all blocks = route D):
  v1 cut         v1 at the cue position clamped to base for layers 1-17 (forward pre-hook on block.attn args) -> D_nov1
  block-0 only   x0 donor at block 0 only (v171's arm), with and without the v1 cut
  head removal   each of the 162 heads (layer x 9) clamped to its BASE output at ALL positions (the cue position already is) ->
                 drop = D - D_without; a head that reads the cue's x0/v1 OR relays a reader's write is in the pathway
  layer removal  all 9 heads of one layer clamped to base (18 arms); layers 0-4 together; the largest 0-4 head at t only (unregistered)
Smoke (CPU, 4 rows per parity): 3 rows per parity, disclosed, all 14 cells: v1 cut removes 0.71-0.99 x D (D 0.34-0.85); block-0-only with v1 cut
  is -0.06 to 0.11 x block-0-only; removing all 45 heads of layers 0-4 leaves 0.66-0.98 x D; the largest single-head drop is 07:08 (coordination,
  polarity), 08:01 (both/either, both/neither, either/neither, degree) or 06:03 (finiteness), 0.20-0.43 x D, layer >= 5 on 14/14; the largest
  early (0-4) head drops <= 0.05 and at t only <= 0.04; 11:03 <= 0.07; layer additivity 0.98-1.02 on 14/14; pred_a false only because 3-row vs
  16-row (as v171's smoke). PRE-SMOKE bars (written before the smoke, with the v1 hypothesis already stated): v1 cut >= 0.15 x D on >= 10,
  block-0 ratio <= 0.5 on >= 10, late on >= 10, early-removed >= 0.35 on >= 10; tightened below to what I now believe.

Registered before the run:
  pred_a_reproduces        D equals v171's all_blocks within 0.02 on every cell (same path, both 16-row)
  pred_b_v1_carries        cutting v1 at the cue position removes 0.5-1.1 x D on >= 12 of 14 cells, and the block-0-only arm with the v1 cut is
                           <= 0.3 x the block-0-only arm on >= 12 of 14 (block 0's share is the value-residual carry)
  pred_c_readers_late      the largest single-head drop lies at layer >= 5 on >= 12 of 14 cells and is one of the hub heads 07:08 / 08:01 on >= 10 of 14
                           (the direct route is read by late hub heads through v1, not by early heads)
  pred_d_early_dispensable removing all 45 heads of layers 0-4 leaves >= 0.5 x D on >= 12 of 14 (the early segment is not the reader)
  pred_e_layer_additive    on the layer with the largest layer-removal drop, sum of its 9 head drops / layer drop lies in 0.6-1.4 on >= 12 of 14
Reported, unregistered: the 162-head drop map, per-layer drops, hub-head (07:08, 08:01, 11:03) drops, largest head at t only.
Smoke: V172_SMOKE=<out.json> -> CPU, V172_SMOKE_ROWS rows per parity (default 4).
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
OUT = ROOT / "circuits/followups/unit_tier5_direct_route_readers_v172_result.json"
V171 = ROOT / "circuits/followups/unit_tier5_x0_reentry_map_v171_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
N_LAYERS, N_HEADS, HEAD_DIM = 18, 9, 128
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
HUBS = ("07:08", "08:01", "11:03")
BARS = {"reproduce_tol": 0.02, "v1_cut_band": [0.5, 1.1], "v1_min_cells": 12, "block0_v1_ratio_max": 0.3, "block0_min_cells": 12,
        "late_layer_min": 5, "late_min_cells": 12, "hub_heads": ["07:08", "08:01"], "hub_min_cells": 10, "early_removed_min_frac": 0.5, "early_min_cells": 12,
        "additive_band": [0.6, 1.4], "additive_min_cells": 12}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 3200, 60000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_direct_route_readers_v172", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V172_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    F = torch.nn.functional
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V172_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    v171 = json.loads(V171.read_text())["measures"] if V171.exists() else {}
    R = {}

    def cell(fam, par):
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
        t_t = torch.tensor(list(batch.semantic_positions), device=backend.device)
        def capture(bt, pos_t, full):
            cap = {}
            hs = [model.transformer.wte.register_forward_hook(lambda m_, a, o: cap.__setitem__("embed", o[ar, pos_t].detach().clone()))]
            for l in range(N_LAYERS):
                if full:
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(f"attn:{l:02d}", a[0].detach().clone())))
                else:
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(lambda m_, a, l=l: cap.__setitem__(f"attn:{l:02d}", a[0][ar, pos_t].detach().clone())))
                hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(lambda m_, a, o, l=l: cap.__setitem__(f"mlp:{l:02d}", o[ar, pos_t].detach().clone())))
            hs.append(model.transformer.h[1].attn.register_forward_pre_hook(lambda m_, a: cap.__setitem__("v1", a[1][ar, pos_t].detach().clone())))
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cap
        cap, capb = capture(db, cd_t, False), capture(batch, cb_t, True)     # capb["attn:l"] is the FULL base c_proj input (rows, T, 1152)
        x0_donor = F.rms_norm(cap["embed"].float(), (cap["embed"].shape[-1],))
        def run(blocks, remove=(), v1_cut=False, remove_at_t=False):
            """cue-position writes base; x0 donor at the cue position at `blocks`; heads in `remove` (layer, head) clamped to base
            at all positions (or at t only); v1_cut clamps the block-0 value residual at the cue position to base for layers >= 1."""
            hs = []
            rem = {}
            for (l, h) in remove: rem.setdefault(l, []).append(h)
            for u in UNITS:
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u, l=l):
                        v = a[0].clone(); v[ar, cb_t] = capb[u][ar, cb_t].to(v.dtype)
                        for h in rem.get(l, ()):
                            sl = slice(h * HEAD_DIM, (h + 1) * HEAD_DIM)
                            if remove_at_t: v[ar, t_t, sl] = capb[u][ar, t_t, sl].to(v.dtype)
                            else: v[:, :, sl] = capb[u][:, :, sl].to(v.dtype)
                        return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype); return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            if v1_cut:
                for l in range(1, N_LAYERS):
                    def vh(m_, a):
                        v1 = a[1].clone(); v1[ar, cb_t] = capb["v1"].to(v1.dtype); return (a[0], v1)
                    hs.append(model.transformer.h[l].attn.register_forward_pre_hook(vh))
            blocks = set(blocks)
            def x0_override(layer, x0):
                if layer not in blocks: return x0
                x0 = x0.clone(); x0[ar, cb_t] = x0_donor.to(x0.dtype); return x0
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache, x0_override=x0_override)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        ALL = list(range(-1, N_LAYERS))
        D = run(ALL)
        D_nov1 = run(ALL, v1_cut=True)
        b0, b0_nov1 = run([-1, 0]), run([-1, 0], v1_cut=True)
        head_drop = {f"{l:02d}:{h:02d}": round(D - run(ALL, remove=[(l, h)]), 3) for l in range(N_LAYERS) for h in range(N_HEADS)}
        layer_drop = {f"{l:02d}": round(D - run(ALL, remove=[(l, h) for h in range(N_HEADS)]), 3) for l in range(N_LAYERS)}
        early_removed = run(ALL, remove=[(l, h) for l in range(5) for h in range(N_HEADS)])
        largest = max(head_drop, key=head_drop.get)
        largest_early = max((k for k in head_drop if int(k[:2]) <= 4), key=head_drop.get)
        le_l, le_h = int(largest_early[:2]), int(largest_early[3:])
        largest_early_at_t = round(D - run(ALL, remove=[(le_l, le_h)], remove_at_t=True), 3)
        best_layer = max(layer_drop, key=layer_drop.get)
        head_sum_best = round(sum(head_drop[f"{best_layer}:{h:02d}"] for h in range(N_HEADS)), 3)
        ref = v171.get(f"{fam}:{par}", {}).get("all_blocks")
        fr = lambda x: round(x / D, 3) if abs(D) > 1e-6 else None
        return {"rows": rows, "D": D, "v171_all_blocks": ref, "D_v1_cut": D_nov1, "v1_cut_frac": fr(D - D_nov1),
                "block0_only": b0, "block0_only_v1_cut": b0_nov1, "block0_v1_ratio": round(b0_nov1 / b0, 3) if abs(b0) > 1e-6 else None,
                "head_drop": head_drop, "layer_drop": layer_drop, "early_removed": early_removed, "early_removed_frac": fr(early_removed),
                "largest_head": largest, "largest_head_drop": head_drop[largest], "largest_head_frac": fr(head_drop[largest]),
                "largest_head_layer": int(largest[:2]), "largest_early_head": largest_early, "largest_early_head_drop": head_drop[largest_early],
                "largest_early_head_at_t_drop": largest_early_at_t, "hub_drops": {k: head_drop[k] for k in HUBS},
                "best_layer": best_layer, "best_layer_drop": layer_drop[best_layer], "head_sum_best_layer": head_sum_best,
                "additivity_best_layer": round(head_sum_best / layer_drop[best_layer], 3) if abs(layer_drop[best_layer]) > 1e-6 else None,
                "top5_heads": sorted(head_drop.items(), key=lambda kv: -kv[1])[:5]}

    for fam in FAMILIES:
        for par in ("p0", "p1"):
            S = cell(fam, par)
            R[f"{fam}:{par}"] = S
            print(fam, par, {k: v for k, v in S.items() if k not in ("head_drop", "layer_drop")}, flush=True)
            print("   layers", S["layer_drop"], flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_direct_route_readers_v172", "candidate_id": "corpus.unit_tier5_direct_route_readers_v172",
              "bars": BARS, "families": list(FAMILIES), "hubs": list(HUBS),
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    pred_a = all(S["v171_all_blocks"] is not None and abs(S["D"] - S["v171_all_blocks"]) <= B["reproduce_tol"] for S in cells)
    pred_b = (sum(S["v1_cut_frac"] is not None and B["v1_cut_band"][0] <= S["v1_cut_frac"] <= B["v1_cut_band"][1] for S in cells) >= B["v1_min_cells"]
              and sum(S["block0_v1_ratio"] is not None and S["block0_v1_ratio"] <= B["block0_v1_ratio_max"] for S in cells) >= B["block0_min_cells"])
    pred_c = (sum(S["largest_head_layer"] >= B["late_layer_min"] for S in cells) >= B["late_min_cells"]
              and sum(S["largest_head"] in B["hub_heads"] for S in cells) >= B["hub_min_cells"])
    pred_d = sum(S["early_removed_frac"] is not None and S["early_removed_frac"] >= B["early_removed_min_frac"] for S in cells) >= B["early_min_cells"]
    pred_e = sum(S["additivity_best_layer"] is not None and B["additive_band"][0] <= S["additivity_best_layer"] <= B["additive_band"][1] for S in cells) >= B["additive_min_cells"]
    return {"pred_a_reproduces": pred_a, "pred_b_v1_carries": pred_b, "pred_c_readers_late": pred_c,
            "pred_d_early_dispensable": pred_d, "pred_e_layer_additive": pred_e}


if __name__ == "__main__":
    main()
