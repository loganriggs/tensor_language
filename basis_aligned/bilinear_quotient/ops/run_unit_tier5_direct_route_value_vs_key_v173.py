#!/usr/bin/env python3
# BQGATE: five frozen predictions; families fixed (v170-v172's seven function-word cells); arms and block lists fixed before the run.
"""Tier-5: the function-word direct route split into its VALUE side (block-0 value residual v1) and its KEY side (x0 re-entries).

v172: with all 36 cue-position writes clamped to base, donor x0 re-entered at every block recovers D = 0.32-0.86; clamping the block-0
value residual v1 at the cue position (v = (1 - lamb) c_v(x) + lamb v1 at every layer) removes 0.70-1.01 of D; the readers are the hub
heads 07:08 / 08:01 (06:03 on finiteness) at t; the full block-0 swap alone carries 0.60-0.94 of D. So the route is a head at t reading
the cue's block-0 VALUE. A head term is pattern x value: the pattern at t over the cue position comes from the cue position's KEY, i.e.
from its residual (base writes + the x0 re-entries), the value from v1. This rung gives each side alone and the two together:
  value_only   base x0 everywhere; v1 at the cue clamped to the DONOR's v1 for layers 1-17 (dose: the whole value side, nothing else)
  key_only     donor x0 at all blocks with v1 cut (= v172's D_v1_cut, reproduced)
  both         donor x0 at all blocks (= D)
  conditional  with donor v1 held, donor x0 at ONE block 1..17 at a time (base at block 0), plus cumulative 1-4 and 5-11 -- which block's
               re-entry supplies the key side; v171's unconditional singles at 1-17 were <= 0.03
14 cells (7 families x 2 parities, 16 rows), whole-model recovery. Smoke (CPU, 3 rows per parity): 3 rows per parity, disclosed, 14 cells: value_only 0.62-0.93 x D and equal to the full block-0
  swap within 0.05 on 14/14; key_only 0.00-0.16; (value + key) / D 0.78-1.08 -- the two sides are ADDITIVE, my pre-smoke 'gating' prior (the key side
  acts only in the presence of the donor value: (D - value_only) - key_only >= 0.03 on >= 10/14) was WRONG -- the excess is +0.04..+0.13 on p0 cells and
  -0.06..+0.01 on p1 cells (8/14 >= 0.03); conditional increments over value_only are small (<= 0.06 per block), largest at block 7 on 12/14
  (6 and 1 once each), cumulative 5-11 (0.05-0.16) > cumulative 1-4 (-0.01-0.04) on 14/14; sum of the 17 increments / (D - value_only) 0.67-1.18.
  pred_a false only for 3-row vs 16-row. Registered below: what I believe after the smoke; pred_c is re-registered as additivity.

Registered before the run:
  pred_a_reproduces        both = v172's D within 0.02 and key_only = v172's D_v1_cut within 0.02 on every cell (both 16-row)
  pred_b_value_carries     value_only lies within 0.4-1.0 x D on >= 12 of 14 cells and equals the full block-0 x0 swap within 0.05 on >= 12 of 14
                           (the value side alone is most of the route, and block 0's whole contribution IS the value residual)
  pred_c_sides_additive    (value_only + key_only) / D lies within 0.7-1.2 on >= 12 of 14 cells and |(D - value_only) - key_only| <= 0.15 on
                           >= 12 of 14 (the value side and the key side of the direct route are separable and add; pre-smoke I registered the
                           opposite -- a gating excess >= 0.03 on >= 10/14 -- and the smoke refuted it)
  pred_d_key_block_late    with donor v1 held, the largest single re-entry increment among blocks 1-17 lies in 5-8 on >= 10 of 14 and the cumulative
                           5-11 increment exceeds the cumulative 1-4 increment on >= 12 of 14 (the key side enters at the hub's own layers)
  pred_e_conditional_adds  sum of the 17 conditional singles / (D - value_only) lies within 0.5-1.5 on >= 10 of 14
Reported, unregistered: the conditional 17-block map (each an INCREMENT over value_only), cumulative 1-4 and 5-11 given v1, block-0-only (v172) for comparison.
Smoke: V173_SMOKE=<out.json> -> CPU, V173_SMOKE_ROWS rows per parity (default 3).
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
OUT = ROOT / "circuits/followups/unit_tier5_direct_route_value_vs_key_v173_result.json"
V172 = ROOT / "circuits/followups/unit_tier5_direct_route_readers_v172_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
N_LAYERS = 18
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "value_frac_band": [0.4, 1.0], "value_min_cells": 12, "value_eq_block0_tol": 0.05, "gating_abs_max": 0.15, "gating_min_cells": 12,
        "sum_frac_band": [0.7, 1.2], "sum_min_cells": 12, "cum_min_cells": 12, "late_blocks": [5, 8], "late_min_cells": 10, "additive_band": [0.5, 1.5], "additive_min_cells": 10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 500, 10000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_direct_route_value_vs_key_v173", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V173_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    F = torch.nn.functional
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V173_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v172 = json.loads(V172.read_text())["measures"] if V172.exists() else {}
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
        def capture(bt, pos_t):
            cap = {}
            hs = [model.transformer.wte.register_forward_hook(lambda m_, a, o: cap.__setitem__("embed", o[ar, pos_t].detach().clone()))]
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
        x0_donor = F.rms_norm(cap["embed"].float(), (cap["embed"].shape[-1],))
        def run(blocks, v1=None):
            """cue-position writes base; x0 donor at the cue position at `blocks`; v1 at the cue clamped to base/donor for layers >= 1 (None = live)."""
            hs = []
            for u in UNITS:
                kind, l = u.split(":"); l = int(l)
                if kind == "attn":
                    def ph(m_, a, u=u):
                        v = a[0].clone(); v[ar, cb_t] = capb[u].to(v.dtype); return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype); return o
                    hs.append(model.transformer.h[l].mlp.Down.register_forward_hook(mh))
            if v1 is not None:
                src = capb["v1"] if v1 == "base" else cap["v1"]
                for l in range(1, N_LAYERS):
                    def vh(m_, a, src=src):
                        t = a[1].clone(); t[ar, cb_t] = src.to(t.dtype); return (a[0], t)
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
        key_only = run(ALL, v1="base")
        value_only = run([], v1="donor")
        block0_only = run([-1, 0])
        given = round(D - value_only, 3)
        # conditional INCREMENTS over value_only (the smoke's first cell showed the raw arms sit at value_only + 0.00-0.02)
        cond = {str(b): round(run([b], v1="donor") - value_only, 3) for b in range(1, N_LAYERS)}
        cond_1_4 = round(run([1, 2, 3, 4], v1="donor") - value_only, 3); cond_5_11 = round(run(list(range(5, 12)), v1="donor") - value_only, 3)
        ref = v172.get(f"{fam}:{par}", {})
        fr = lambda x: round(x / D, 3) if abs(D) > 1e-6 else None
        best = max(cond, key=cond.get)
        return {"rows": rows, "D": D, "v172_D": ref.get("D"), "key_only": key_only, "v172_D_v1_cut": ref.get("D_v1_cut"),
                "value_only": value_only, "value_frac": fr(value_only), "block0_only": block0_only, "block0_frac": fr(block0_only),
                "given_value_increment": given, "gating_excess": round(given - key_only, 3), "sum_frac": fr(value_only + key_only),
                "conditional_increment": cond, "cond_1_4": cond_1_4, "cond_5_11": cond_5_11,
                "largest_cond_block": int(best), "largest_cond_value": cond[best],
                "cond_singles_sum_over_increment": round(sum(cond.values()) / given, 3) if abs(given) > 1e-6 else None}

    for fam in FAMILIES:
        for par in ("p0", "p1"):
            S = cell(fam, par)
            R[f"{fam}:{par}"] = S
            print(fam, par, {k: v for k, v in S.items() if k != "conditional_increment"}, flush=True)
            print("   cond", S["conditional_increment"], flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_direct_route_value_vs_key_v173", "candidate_id": "corpus.unit_tier5_direct_route_value_vs_key_v173",
              "bars": BARS, "families": list(FAMILIES),
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    pred_a = all(S["v172_D"] is not None and abs(S["D"] - S["v172_D"]) <= B["reproduce_tol"]
                 and S["v172_D_v1_cut"] is not None and abs(S["key_only"] - S["v172_D_v1_cut"]) <= B["reproduce_tol"] for S in cells)
    pred_b = (sum(S["value_frac"] is not None and B["value_frac_band"][0] <= S["value_frac"] <= B["value_frac_band"][1] for S in cells) >= B["value_min_cells"]
              and sum(abs(S["value_only"] - S["block0_only"]) <= B["value_eq_block0_tol"] for S in cells) >= B["value_min_cells"])
    pred_c = (sum(S["sum_frac"] is not None and B["sum_frac_band"][0] <= S["sum_frac"] <= B["sum_frac_band"][1] for S in cells) >= B["sum_min_cells"]
              and sum(abs(S["gating_excess"]) <= B["gating_abs_max"] for S in cells) >= B["gating_min_cells"])
    pred_d = (sum(B["late_blocks"][0] <= S["largest_cond_block"] <= B["late_blocks"][1] for S in cells) >= B["late_min_cells"]
              and sum(S["cond_5_11"] > S["cond_1_4"] for S in cells) >= B["cum_min_cells"])
    pred_e = sum(S["cond_singles_sum_over_increment"] is not None and B["additive_band"][0] <= S["cond_singles_sum_over_increment"] <= B["additive_band"][1]
                 for S in cells) >= B["additive_min_cells"]
    return {"pred_a_reproduces": pred_a, "pred_b_value_carries": pred_b, "pred_c_sides_additive": pred_c,
            "pred_d_key_block_late": pred_d, "pred_e_conditional_adds": pred_e}


if __name__ == "__main__":
    main()
