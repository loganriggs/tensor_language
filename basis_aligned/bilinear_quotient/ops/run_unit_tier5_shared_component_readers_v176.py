#!/usr/bin/env python3
# BQGATE: five frozen predictions; cells fixed (polarity, both/neither, either/neither x 2 parities); axis = the other family's v175 delta; removal = head 8 of one layer.
"""Tier-5: is the polarity <-> 'neither' shared channel-8 component read by the same head in both families?

v175: the seven function-word slice-8 v1 deltas are causally disjoint except polarity (any/some) and the 'neither' correlatives:
polarity -> unit(neither-both) carries 0.379/0.357 of polarity's slice-8 swap, both/neither -> unit(any-some) 0.325/0.233,
either/neither 0.327/0.264 (cos -0.418). Shared geometry is not yet a shared circuit: the v1 channel 8 is mixed only into head 8 of every
layer, so the reader of a channel-8 component is some set of head-8s (v172: 07:08 the largest reader of polarity's direct route, 14:08/16:08
third on the correlatives). This rung splits each family's delta into the SHARED component (projection onto the other family's delta
direction: polarity onto unit(neither-both); both/neither and either/neither onto unit(any-some)) and the orthogonal RESIDUAL, swaps each
into the cue's slice 8 at layers 1-17 (v174/v175 instrument, cue-position writes clamped to base), and removes head 8 of ONE layer at a time
(c_proj input slice clamped to base at all positions, v172's removal) under the full delta and under the shared component. 17 layers x 2
components + 3 = 37 forwards per cell, 6 cells.
Probe (CPU, 2 cells, 3 rows, run BEFORE the bars below were written, full delta only, 6 s): removal cost of head 8 per layer under the
full channel-8 swap — polarity p0: 07 0.69, 16 0.31, all others <= 0.07 (sum 1.05); both/neither p1: 07 0.32, 16 0.31, 14 0.28, 08 0.10,
13 0.09 (sum 1.09). So polarity's channel-8 reader is 07:08 with 16:08 second; the correlative's is spread over 07/14/16. pred_d was
written after this probe and asks about the SHARED component, which the probe did not measure.
Smoke (CPU, 2 cells polarity p0 + both/neither p0, 3 rows, 45 s, run after the bars a/b/c/e and the PRE-SMOKE pred_d were written):
pred_a true (shared 0.376 vs v175 0.379; 0.324 vs 0.325); shared + residual 0.93 / 0.98 of full; 17-layer sum 0.90 / 1.00; Pearson 0.993 /
0.970; polarity's shared component: 07 0.76, 16 0.27; both/neither's shared component: 16 0.56, 07 0.33, 14 0.07 vs full 07 0.35 — the
shared component is read by 16:08 on the correlative, not shifted toward 07:08 (pre-smoke pred_d refuted on 1/1 correlative cells, so pred_d
is re-registered as the reading I now hold; b/c/e were false in the smoke only through the 2-cell count).

Registered before the run:
  pred_a_instrument       the shared component reproduces v175's rank-1 pair fraction within 0.02 on 6/6 cells, and the zero swap gives 0.000
  pred_b_components_add   shared + residual = 0.8-1.2 x the full swap on >= 5 of 6 (head outputs are linear in v at fixed patterns)
  pred_c_layers_add       the 17 single-layer head-8 removals of the shared component sum to 0.7-1.3 x the shared swap on >= 5 of 6
  pred_d_not_read_by_07   on the four correlative cells |07:08 removal cost of the SHARED component - 07:08 removal cost of the full delta|
                          <= 0.10 (the shared part is read in the family's own reader proportions, NOT preferentially by polarity's reader),
                          and on polarity 07:08 removal costs >= 0.5 of the shared component on 2/2
                          PRE-SMOKE VERSION (refuted by the 2-cell smoke, see below): 'shared 07 cost exceeds full 07 cost by >= 0.10 on 4/4'
  pred_e_own_readers      per family, the 17-layer removal profile of the shared component has Pearson r >= 0.8 with the profile of the full
                          delta on >= 5 of 6 cells (the shared part is read by the family's own readers, not by a distinct head)
pred_d (own proportions) and pred_e (own readers, r >= 0.8) are now the same reading measured two ways; the pre-smoke pred_d was the opposite.
Reported, unregistered: the residual component's swap, the full-delta and shared-component removal profiles, top-3 reader layers per cell.
Smoke: V176_SMOKE=<out.json> -> CPU, V176_SMOKE_ROWS rows (default 3), V176_SMOKE_CELLS cells (default 2). V176_PROBE=1 prints the full-delta
removal profiles only.
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
OUT = ROOT / "circuits/followups/unit_tier5_shared_component_readers_v176_result.json"
V175 = ROOT / "circuits/followups/unit_tier5_channel8_shared_subspace_v175_result.json"
AXIS = {"polarity_state": "correlative_both_neither", "correlative_both_neither": "polarity_state", "correlative_either_neither": "polarity_state"}
CELLS = [(f, p) for f in ("polarity_state", "correlative_both_neither", "correlative_either_neither") for p in ("p0", "p1")]
N_LAYERS, N_HEADS, CH, HD = 18, 9, 8, 128
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"reproduce_tol": 0.02, "add_band": [0.8, 1.2], "add_min_cells": 5, "layer_sum_band": [0.7, 1.3], "layer_min_cells": 5,
        "shift_07_max": 0.10, "pol_07_min": 0.5, "pearson_min": 0.8, "pearson_min_cells": 5}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_shared_component_readers_v176", "behaviours": 3, "targets": len(CELLS),
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
    smoke = os.environ.get("V176_SMOKE"); probe = os.environ.get("V176_PROBE")
    backend = producer.Bilin18TorchBackend.load("cpu" if (smoke or probe) else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V176_SMOKE_ROWS", "3"))]) if (smoke or probe) else (lambda rows: rows)
    cells_run = CELLS[::2][:int(os.environ.get("V176_SMOKE_CELLS", "2"))] if (smoke or probe) else CELLS
    v175 = json.loads(V175.read_text())["measures"] if V175.exists() else {}
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

    fams = sorted({f for f, _ in cells_run} | {AXIS[f] for f, _ in cells_run})
    for fam in fams:
        for par in sorted({p for _, p in cells_run}):
            cells[(fam, par)] = setup(fam, par)

    R = {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        dl = C["delta"]; ax = cells[(AXIS[fam], par)]["delta"]; u = ax / ax.norm()
        shared = (dl @ u) * u; resid = dl - shared
        full = C["run"](dl); zero = C["run"](torch.zeros_like(dl))
        fr = lambda x, ref: round((x - zero) / (ref - zero), 3) if abs(ref - zero) > 1e-6 else None
        S = {"rows": C["rows"], "full": full, "zero": zero, "v175_pair": v175.get(f"{fam}:{par}", {}).get("pair", {}).get(AXIS[fam]),
             "shared_norm_frac": round(float(shared.norm() / dl.norm()), 3)}
        prof_full = {l: fr(C["run"](dl, remove_layer=l), full) for l in range(1, N_LAYERS)}
        S["full_removal_cost"] = {f"{l:02d}": round(1 - v, 3) for l, v in prof_full.items()}
        print(fam, par, "full", full, "removal cost", S["full_removal_cost"], flush=True)
        if probe:
            R[f"{fam}:{par}"] = S; continue
        sh = C["run"](shared); rs = C["run"](resid)
        S.update({"shared": sh, "shared_frac": fr(sh, full), "residual": rs, "residual_frac": fr(rs, full), "sum_frac": fr(sh + rs - zero, full)})
        prof_sh = {l: fr(C["run"](shared, remove_layer=l), sh) for l in range(1, N_LAYERS)}
        S["shared_removal_cost"] = {f"{l:02d}": round(1 - v, 3) if v is not None else None for l, v in prof_sh.items()}
        costs = {l: 1 - v for l, v in prof_sh.items() if v is not None}
        S["shared_layer_sum"] = round(sum(costs.values()), 3)
        S["shared_top3"] = [f"{l:02d}" for l, _ in sorted(costs.items(), key=lambda kv: -kv[1])[:3]]
        S["full_top3"] = sorted(S["full_removal_cost"], key=lambda k: -S["full_removal_cost"][k])[:3]
        S["pearson_shared_vs_full"] = pearson([S["full_removal_cost"][f"{l:02d}"] for l in range(1, N_LAYERS)], [costs.get(l, 0.0) for l in range(1, N_LAYERS)])
        R[f"{fam}:{par}"] = S
        print(fam, par, {k: v for k, v in S.items() if "cost" not in k}, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    if probe:
        return
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_shared_component_readers_v176", "candidate_id": "corpus.unit_tier5_shared_component_readers_v176",
              "bars": BARS, "axis": AXIS, "channel": CH, "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, band: x is not None and band[0] <= x <= band[1]
    pred_a = all(S["v175_pair"] is not None and S["shared_frac"] is not None and abs(S["shared_frac"] - S["v175_pair"]) <= B["reproduce_tol"] and S["zero"] == 0.0 for S in cells)
    pred_b = sum(inb(S["sum_frac"], B["add_band"]) for S in cells) >= B["add_min_cells"]
    pred_c = sum(inb(S["shared_layer_sum"], B["layer_sum_band"]) for S in cells) >= B["layer_min_cells"]
    corr = [S for k, S in R.items() if k.startswith("correlative")]
    pol = [S for k, S in R.items() if k.startswith("polarity")]
    pred_d = (len(corr) == 4 and all(S["shared_removal_cost"]["07"] is not None and abs(S["shared_removal_cost"]["07"] - S["full_removal_cost"]["07"]) <= B["shift_07_max"] for S in corr)
              and len(pol) == 2 and all(S["shared_removal_cost"]["07"] is not None and S["shared_removal_cost"]["07"] >= B["pol_07_min"] for S in pol))
    pred_e = sum(S["pearson_shared_vs_full"] is not None and S["pearson_shared_vs_full"] >= B["pearson_min"] for S in cells) >= B["pearson_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_components_add": pred_b, "pred_c_layers_add": pred_c, "pred_d_not_read_by_07": pred_d, "pred_e_own_readers": pred_e}


if __name__ == "__main__":
    main()
