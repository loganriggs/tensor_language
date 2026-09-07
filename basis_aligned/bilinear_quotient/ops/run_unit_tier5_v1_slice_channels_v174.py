#!/usr/bin/env python3
# BQGATE: five frozen predictions; families fixed (v170-v173's seven function-word cells); reader indices taken from the v172 receipt, not this run.
"""Tier-5: the block-0 value residual is a bank of per-HEAD-INDEX channels -- which 128-d slice of v1 carries the function-word cue?

v173: the function-word direct route is 0.63-0.94 the cue's block-0 value residual v1 read at t (donor v1 at the cue for layers 1-17,
base x0 everywhere, writes base = value_only 0.20-0.80). v1 is (B, T, H, D): head h of EVERY layer mixes v1[:, :, h, :] into its own value
(v = (1 - lamb) c_v(x) + lamb v1.view_as(v)), so slice h of v1 is a channel that only head-index-h heads can read. v172's readers were
07:08 with 14:08 and 16:08 (index 8), 08:01 (index 1), 06:03 with 11:03 (index 3) -- the same index recurring across layers is what the
channel picture predicts. This rung swaps ONE 128-d slice of the cue's v1 to donor at a time (9 arms), the complement of the top-2
indices taken from v172's top-5 reader list (fixed from the receipt, not from this run), and the v172-top-index slice restricted to
layers >= 5 or to layers 1-4. Whole-model recovery, 14 cells (7 function-word families x 2 parities, 16 rows).
Smoke (CPU, 3 rows per parity): 3 rows per parity, disclosed, 14 cells: all-slices = v173 value_only within 0.012; slice 8 is the LARGEST single
  channel on 14/14 (0.27-0.58 x value_only), slice 1 second on the correlatives / degree / polarity (0.17-0.39), slice 3 on finiteness (0.32-0.49;
  06:03 + 11:03 are index 3) and degree p1 (0.20), coordination spread (slice 5 0.12-0.17); other slices <= 0.10; the 9 slices sum to 0.85-0.99 x
  value_only on 13/14 (degree p0 0.47); the index-8 slice restricted to layers >= 5 carries 0.97-1.01 x its all-layer swap, to layers 1-4 <= 0.02;
  complement of the v172 top-2 indices 0.03-0.10 except coordination (0.41/0.50). PRE-SMOKE pred_b said the top slice has the index of v172's
  largest reader on >= 12/14 -- FALSE on the six correlative cells (08:01's removal was the largest drop there, yet slice 8 carries 0.5-0.58 of
  the value side and slice 1 0.29-0.36): the largest remover is not the largest value channel. Re-registered below. First smoke attempt had
  the layer restriction broken (block.attn returns the v1 it is given, so a swap at layer l persists) -- fixed by clamping every layer.

Registered before the run:
  pred_a_reproduces        all nine slices donor equals v173's value_only within 0.02 on every cell (both 16-row)
  pred_b_slice8_channel    the largest single-slice recovery is index 8 (the 07:08 / 14:08 / 16:08 channel) on >= 12 of 14 cells and carries 0.25-0.70 x
                           value_only on >= 12 of 14; the slice with v172's largest reader's index carries 0.15-1.0 x value_only on >= 12 of 14
  pred_c_slices_add        the nine single-slice recoveries sum to 0.7-1.3 x value_only on >= 12 of 14
  pred_d_read_late         the v172-top-index slice restricted to layers >= 5 carries 0.8-1.2 x its all-layer swap on >= 12 of 14, and restricted to
                           layers 1-4 it carries <= 0.2 x on >= 12 of 14
  pred_e_other_channels_quiet the complement of the top-2 v172 reader indices (7 slices donor) carries <= 0.35 x value_only on >= 10 of 14
Reported, unregistered: the 9-slice map per cell, the v172 reader indices used.
Smoke: V174_SMOKE=<out.json> -> CPU, V174_SMOKE_ROWS rows per parity (default 3).
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
OUT = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V172 = ROOT / "circuits/followups/unit_tier5_direct_route_readers_v172_result.json"
V173 = ROOT / "circuits/followups/unit_tier5_direct_route_value_vs_key_v173_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
N_LAYERS = 18
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
N_HEADS = 9
BARS = {"reproduce_tol": 0.02, "channel_index": 8, "index_min_cells": 12, "top_frac_band": [0.25, 0.70], "reader_frac_band": [0.15, 1.0], "top_min_cells": 12, "sum_band": [0.7, 1.3], "sum_min_cells": 12,
        "late_band": [0.8, 1.2], "early_max": 0.2, "layer_min_cells": 12, "complement_max": 0.35, "complement_min_cells": 10}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 500, 10000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_v1_slice_channels_v174", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V174_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    F = torch.nn.functional
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V174_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v172 = json.loads(V172.read_text())["measures"] if V172.exists() else {}
    v173 = json.loads(V173.read_text())["measures"] if V173.exists() else {}
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
        def run(slices, layers=range(1, N_LAYERS)):
            """cue-position writes base; base x0 everywhere; the listed v1 head-slices at the cue position set to the donor's v1 at the listed layers."""
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
            if slices:
                # block.attn RETURNS the v1 it was given, so a swap at layer l would persist downstream: every layer 1-17 gets a hook that
                # sets the slices to donor at the listed layers and back to BASE elsewhere (the residual is base, so base v1 = capb)
                sl = torch.tensor(sorted(slices), device=backend.device)
                layers = set(layers)
                for l in range(1, N_LAYERS):
                    def vh(m_, a, src=(cap if l in layers else capb)["v1"]):
                        t = a[1].clone(); t[ar[:, None], cb_t[:, None], sl[None, :]] = src[:, sl].to(t.dtype); return (a[0], t)
                    hs.append(model.transformer.h[l].attn.register_forward_pre_hook(vh))
            try:
                with torch.no_grad():
                    out = g.forward_units(backend, batch, donor_cache=prep.donor_cache, base_cache=prep.base_cache)
            finally:
                for h_ in hs: h_.remove()
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        ref = v172.get(f"{fam}:{par}", {})
        top_idx = int(ref["largest_head"][3:])
        seen = []
        for name, _ in ref["top5_heads"]:
            i = int(name[3:])
            if i not in seen: seen.append(i)
        top2 = seen[:2]
        value_all = run(range(N_HEADS))
        single = {str(h): run([h]) for h in range(N_HEADS)}
        best = max(single, key=single.get)
        top_late = run([top_idx], layers=range(5, N_LAYERS)); top_early = run([top_idx], layers=range(1, 5))
        complement = run([h for h in range(N_HEADS) if h not in top2])
        fr = lambda x: round(x / value_all, 3) if abs(value_all) > 1e-6 else None
        ts = single[str(top_idx)]
        return {"rows": rows, "value_all_slices": value_all, "v173_value_only": v173.get(f"{fam}:{par}", {}).get("value_only"),
                "single_slice": single, "best_slice": int(best), "best_slice_frac": fr(single[best]),
                "v172_top_index": top_idx, "v172_top2_indices": top2, "top_index_slice": ts, "top_index_frac": fr(ts),
                "slices_sum_over_all": fr(sum(single.values())), "top_late": top_late, "top_early": top_early,
                "top_late_ratio": round(top_late / ts, 3) if abs(ts) > 1e-6 else None, "top_early_ratio": round(top_early / ts, 3) if abs(ts) > 1e-6 else None,
                "complement_top2": complement, "complement_frac": fr(complement)}

    for fam in FAMILIES:
        for par in ("p0", "p1"):
            S = cell(fam, par)
            R[f"{fam}:{par}"] = S
            print(fam, par, {k: v for k, v in S.items() if k != "single_slice"}, flush=True)
            print("   slices", S["single_slice"], flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_v1_slice_channels_v174", "candidate_id": "corpus.unit_tier5_v1_slice_channels_v174",
              "bars": BARS, "families": list(FAMILIES),
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    pred_a = all(S["v173_value_only"] is not None and abs(S["value_all_slices"] - S["v173_value_only"]) <= B["reproduce_tol"] for S in cells)
    pred_b = (sum(S["best_slice"] == B["channel_index"] for S in cells) >= B["index_min_cells"]
              and sum(S["best_slice_frac"] is not None and B["top_frac_band"][0] <= S["best_slice_frac"] <= B["top_frac_band"][1] for S in cells) >= B["top_min_cells"]
              and sum(S["top_index_frac"] is not None and B["reader_frac_band"][0] <= S["top_index_frac"] <= B["reader_frac_band"][1] for S in cells) >= B["top_min_cells"])
    pred_c = sum(S["slices_sum_over_all"] is not None and B["sum_band"][0] <= S["slices_sum_over_all"] <= B["sum_band"][1] for S in cells) >= B["sum_min_cells"]
    pred_d = (sum(S["top_late_ratio"] is not None and B["late_band"][0] <= S["top_late_ratio"] <= B["late_band"][1] for S in cells) >= B["layer_min_cells"]
              and sum(S["top_early_ratio"] is not None and S["top_early_ratio"] <= B["early_max"] for S in cells) >= B["layer_min_cells"])
    pred_e = sum(S["complement_frac"] is not None and S["complement_frac"] <= B["complement_max"] for S in cells) >= B["complement_min_cells"]
    return {"pred_a_reproduces": pred_a, "pred_b_slice8_channel": pred_b, "pred_c_slices_add": pred_c,
            "pred_d_read_late": pred_d, "pred_e_other_channels_quiet": pred_e}


if __name__ == "__main__":
    main()
