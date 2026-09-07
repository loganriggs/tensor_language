#!/usr/bin/env python3
# BQGATE: five frozen predictions; families fixed (v170's seven function-word cells); block list fixed; no offsets.
"""Tier-5: the function-word DIRECT route -- which block's x0 re-entry carries the cue identity to the readout?

v170: on seven function-word hub behaviours the donor cue EMBEDDING alone, with all 36 cue-position writes clamped to base,
recovers 0.32-0.86 whole-model (coordination 0.32/0.52, both/either 0.71/0.78, both/neither 0.67/0.76, either/neither
0.78/0.86, degree 0.47/0.81, finiteness 0.77/0.50, polarity 0.50/0.60). bilin18 re-enters x0 = rms_norm(wte) at EVERY block:
live = lambda0 * x + lambda1 * x0, with lambdas [6.09, 6.09] at block 0, [0.013, 8.0] at block 1, [0.064, 5.06] at block 5 and
~[1, 8] elsewhere -- the running stream is nearly wiped at blocks 1 and 5 and the token identity re-enters at weight ~8
everywhere. With the cue-position writes clamped to base, the donor x0 at the cue position can act ONLY through attention
heads at other positions reading the cue position's residual. This rung swaps the cue position's x0 to donor at ONE block at
a time (block -1 = the initial stream, then blocks 0..17; g.forward_units(x0_override=...), identity override reproduces the
old forward to 0.0), plus cumulative blocks <= 4 (before the block-5 rescale) and <= 11, and all blocks (= v170's direct
route through the new code path), whole-model recovery, 14 cells (7 families x 2 parities, 16 rows each).
Smoke (CPU, 4 rows per parity, disclosed): pre-smoke bars were initial <= 0.05, cumulative <= 4 at <= 0.30 x all, largest block in
5-11; the smoke gave initial (= block 0) 0.07-0.40 = 0.15-0.51 of all on all 14 cells, cumulative <= 4 at 0.75-0.94 x all, blocks 1-4
singles <= 0.025, blocks 5-11 singles peaking at block 7 (0.008-0.036) on 12/14, blocks 12-17 ~0; singles-sum / all 0.30 (degree p0,
all 0.49) to 1.15, 13/14 within 0.6-1.3; all-blocks vs the v170 4-row smoke direct: 0.674 vs 0.674, 0.849 vs 0.849 (identical path).

Registered before the run (bars encode what I believe AFTER the 4-row smoke -- my structural priors were wrong before it: I
expected the initial stream to be inert (block 1's lambda0 = 0.013) and the pre-block-5 re-entries to be small (block 5's
lambda0 = 0.064); the smoke shows the opposite, and the pre-smoke bars are recorded in the smoke line):
  pred_a_new_path_reproduces  all-blocks + initial donor x0 (writes base) equals v170's direct_embed_only within 0.02 on every cell
                              (same quantity through the new x0_override path; both 16-row)
  pred_b_initial_stream_reads the initial stream alone (block -1) recovers 0.10-0.60 of the all-blocks value on every cell (smoke min 0.147, degree p0), and equals
                              the block-0-only arm within 0.005 (identity by construction: x = x0 and lambda0 = lambda1 at block 0)
  pred_c_pre_reset_carries    cumulative re-entry at blocks <= 4 recovers 0.65-1.00 x the all-blocks value on >= 12 of 14 cells (the
                              cue identity is picked up by readers BEFORE the block-5 rescale; the early segment's writes are large
                              enough to survive the 0.064)
  pred_d_blocks_additive      the sum of the 19 single-block recoveries lies within 0.6-1.3 x the all-blocks value on >= 12 of 14
  pred_e_block0_then_7        the largest single block is block 0 (= -1) on >= 12 of 14 cells, every single block in 1-4 is <= 0.03,
                              and the largest block among 5-17 lies in 5-8 on >= 12 of 14
Reported, unregistered: the 19-block map per cell, cumulative <= 11, the largest block's share.
Smoke: V171_SMOKE=<out.json> -> CPU, V171_SMOKE_ROWS rows per parity (default 4).
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
OUT = ROOT / "circuits/followups/unit_tier5_x0_reentry_map_v171_result.json"
V170 = ROOT / "circuits/followups/unit_tier5_cue_source_map_eight_v170_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
N_LAYERS = 18
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BLOCKS = list(range(-1, N_LAYERS))
BARS = {"reproduce_tol": 0.02, "initial_frac_band": [0.10, 0.60], "initial_eq_block0_tol": 0.005, "pre_reset_frac_band": [0.65, 1.00], "pre_reset_min_cells": 12,
        "additive_band": [0.6, 1.3], "additive_min_cells": 12, "block0_min_cells": 12, "early_single_max": 0.03, "late_blocks": [5, 8], "late_min_cells": 12}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 12000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_x0_reentry_map_v171", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V171_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    F = torch.nn.functional
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V171_SMOKE_ROWS", "4"))]) if smoke else (lambda rows: rows)
    v170 = json.loads(V170.read_text())["measures"] if V170.exists() else {}
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
            try:
                with torch.no_grad():
                    g.forward_units(backend, bt, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return cap
        cap, capb = capture(db, cd_t), capture(batch, cb_t)
        x0_donor = F.rms_norm(cap["embed"].float(), (cap["embed"].shape[-1],))          # x0 = rms_norm(wte) at the donor cue position
        def run(blocks):
            """all cue-position writes clamped to base; x0 at the cue position set to donor at the listed blocks (-1 = initial stream)."""
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
        single = {str(b): run([b]) for b in BLOCKS}
        allb = run(BLOCKS)
        pre = run([b for b in BLOCKS if b <= 4]); upto11 = run([b for b in BLOCKS if b <= 11])
        best = max((b for b in BLOCKS if b >= 0), key=lambda b: single[str(b)])
        ref = v170.get(f"{fam}:{par}", {}).get("direct_embed_only")
        return {"rows": rows, "cue_positions_base": lb, "single_block": single, "all_blocks": allb, "v170_direct": ref,
                "initial_stream_only": single["-1"], "cumulative_le_4": pre, "cumulative_le_11": upto11,
                "pre_reset_frac": round(pre / allb, 3) if abs(allb) > 1e-6 else None,
                "singles_sum_over_all": round(sum(single.values()) / allb, 3) if abs(allb) > 1e-6 else None,
                "initial_frac": round(single["-1"] / allb, 3) if abs(allb) > 1e-6 else None,
                "largest_block": best, "largest_block_value": single[str(best)], "largest_block_share": round(single[str(best)] / allb, 3) if abs(allb) > 1e-6 else None,
                "largest_block_ge5": max((b for b in BLOCKS if b >= 5), key=lambda b: single[str(b)])}

    for fam in FAMILIES:
        for par in ("p0", "p1"):
            S = cell(fam, par)
            R[f"{fam}:{par}"] = S
            print(fam, par, {k: v for k, v in S.items() if k not in ("single_block", "cue_positions_base")}, flush=True)
            print("   ", S["single_block"], flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_x0_reentry_map_v171", "candidate_id": "corpus.unit_tier5_x0_reentry_map_v171",
              "bars": BARS, "families": list(FAMILIES), "blocks": BLOCKS, "lambdas": [[round(float(x), 3) for x in blk.lambdas] for blk in model.transformer.h],
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    pred_a = all(S["v170_direct"] is not None and abs(S["all_blocks"] - S["v170_direct"]) <= B["reproduce_tol"] for S in cells)
    pred_b = all(S["initial_frac"] is not None and B["initial_frac_band"][0] <= S["initial_frac"] <= B["initial_frac_band"][1]
                 and abs(S["single_block"]["-1"] - S["single_block"]["0"]) <= B["initial_eq_block0_tol"] for S in cells)
    pred_c = sum(S["pre_reset_frac"] is not None and B["pre_reset_frac_band"][0] <= S["pre_reset_frac"] <= B["pre_reset_frac_band"][1] for S in cells) >= B["pre_reset_min_cells"]
    pred_d = sum(S["singles_sum_over_all"] is not None and B["additive_band"][0] <= S["singles_sum_over_all"] <= B["additive_band"][1] for S in cells) >= B["additive_min_cells"]
    pred_e = (sum(S["largest_block"] == 0 for S in cells) >= B["block0_min_cells"]
              and all(S["single_block"][str(b)] <= B["early_single_max"] for S in cells for b in (1, 2, 3, 4))
              and sum(B["late_blocks"][0] <= S["largest_block_ge5"] <= B["late_blocks"][1] for S in cells) >= B["late_min_cells"])
    return {"pred_a_new_path_reproduces": pred_a, "pred_b_initial_stream_reads": pred_b, "pred_c_pre_reset_carries": pred_c,
            "pred_d_blocks_additive": pred_d, "pred_e_block0_then_7": pred_e}


if __name__ == "__main__":
    main()
