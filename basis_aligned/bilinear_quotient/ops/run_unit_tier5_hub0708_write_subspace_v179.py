#!/usr/bin/env python3
# BQGATE: five frozen predictions; 14 cells fixed; the three candidate shared pairs named from v175/v177 receipts before the run; seeded random control.
"""Tier-5: does the hub reader 07:08 write the seven function-word cues into a shared or family-specific output subspace?

v177: 07:08 reads channel 8 of the cue's value residual for six of the seven families (removal cost 0.27-0.74; either/neither 0.13/0.09).
This rung looks at what 07:08 WRITES: under the full slice-8 swap (v174/v175 instrument, cue writes clamped to base) the head-8 slice of
layer 7's c_proj input changes at every position after the cue; that change (B, T, 128) is captured, its per-row direction at t is
averaged into one 128-d write direction per cell, and the change is added back — whole, projected onto the cell's own direction (rank 1),
projected onto each other family's direction, or onto a seeded random direction — into an otherwise BASE forward (no v1 swap), so each
arm is '07:08's write alone'. v177's removal cost x the full slice-8 swap is the linear expectation for the whole-write arm.
Candidate shared pairs, named before the run from v175 (shared channel geometry) and v177 (reader-profile r >= 0.96): polarity <->
coordination, both/either <-> both/neither, polarity <-> both/neither.
Smoke (CPU, coordination p0 + both/either p0, 3 rows, 2 s, before pred_b's cosine clause and the pred_c rewrite): whole write / linear
expectation 1.05 / 1.02; own rank-1 0.98 / 1.00; rank-1 fraction of the write across rows 0.999; norm share at t 0.64 / 0.61 (the write is
also read from positions between cue and t); cross pair 0.079 / 0.008 at cosine 0.206; random -0.03 / 0.00.

Registered before the run:
  pred_a_instrument   the whole-write arm lies within 0.6-1.4 x (v177 cost_07 x v174 slice-8 swap) on >= 12 of 14 cells, and the zero arm is 0.000
  pred_b_rank1        the own-direction rank-1 arm carries 0.7-1.2 x the whole-write arm on >= 12 of 14, and |cos(write direction, slice-8
                      delta direction)| >= 0.99 on 14/14: v1 enters the head linearly and does not touch the pattern, so 07:08's write change
                      at every position is P[pos, cue] x lambda_7 x delta — one direction, and it is the channel delta itself
  pred_c_reader_not_direction
                      polarity <-> coordination (reader-profile r 0.96 in v177, channel cosine -0.013 in v175) carries -0.1..0.1 x the whole
                      write on 4/4 projections (same reader, different direction), while polarity <-> both/neither and both/either <-> both/neither
                      carry >= 0.2 on 4/4 each (shared channel geometry survives at the 07:08 route)
                      PRE-SMOKE VERSION: 'the three candidate pairs carry >= 0.3 in both directions at both parities (12/12; prior ~0.4)' — replaced
                      after the 2-cell smoke showed own_norm_frac 0.999 / rank-1 0.999 and pair cosine 0.206 = v175's channel cosine, i.e. the write
                      direction is fixed by the token delta, not by the reader (the pre-smoke hypothesis conflated reader-profile similarity with
                      write-direction similarity)
  pred_d_other_pairs  the remaining cross-family rank-1 projections lie within -0.3..0.3 on >= 80% of them
  pred_e_random       the seeded random direction carries -0.15..0.15 x the whole write on >= 12 of 14
Reported, unregistered: the 7 x 7 cosine matrix of write directions per parity, the norm fraction of every projection, the rank-1 fraction
of the write across rows (SVD).
Smoke: V179_SMOKE=<out.json> -> CPU, V179_SMOKE_ROWS rows (default 3), V179_SMOKE_CELLS cells (default 2).
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
OUT = ROOT / "circuits/followups/unit_tier5_hub0708_write_subspace_v179_result.json"
V174 = ROOT / "circuits/followups/unit_tier5_v1_slice_channels_v174_result.json"
V177 = ROOT / "circuits/followups/unit_tier5_channel8_reader_map_v177_result.json"
FAMILIES = ("coordination_agreement", "correlative_both_either", "correlative_both_neither", "correlative_either_neither",
            "degree_frame", "finiteness_selection", "polarity_state")
PAIRS = (("polarity_state", "coordination_agreement"), ("correlative_both_either", "correlative_both_neither"), ("polarity_state", "correlative_both_neither"))
N_LAYERS, N_HEADS, CH, HD, HUB = 18, 9, 8, 128, 7
UNITS = [f"{k}:{l:02d}" for l in range(N_LAYERS) for k in ("attn", "mlp")]
BARS = {"linear_band": [0.6, 1.4], "linear_min_cells": 12, "rank1_band": [0.7, 1.2], "rank1_min_cells": 12, "cos_min": 0.99, "same_reader_band": [-0.1, 0.1], "shared_min": 0.2, "pair_band": [-0.3, 0.3],
        "pair_min_frac": 0.8, "random_band": [-0.15, 0.15], "random_min_cells": 12, "seed": 0}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_hub0708_write_subspace_v179", "behaviours": len(FAMILIES), "targets": 2 * len(FAMILIES),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V179_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    model = backend.model
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:int(os.environ.get("V179_SMOKE_ROWS", "3"))]) if smoke else (lambda rows: rows)
    v174 = json.loads(V174.read_text())["measures"] if V174.exists() else {}
    v177 = json.loads(V177.read_text())["measures"] if V177.exists() else {}
    cells_run = [(f, p) for p in ("p0", "p1") for f in FAMILIES]
    if smoke: cells_run = cells_run[:int(os.environ.get("V179_SMOKE_CELLS", "2"))]
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
        delta = (cap["v1"][:, CH].float() - capb["v1"][:, CH].float()).mean(0)
        t_t = torch.tensor(list(batch.semantic_positions), device=backend.device)
        sl = slice(CH * HD, (CH + 1) * HD)
        def run(vec=None, add=None, grab=None):
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
                        return (v,) + tuple(a[1:])
                    hs.append(model.transformer.h[l].attn.c_proj.register_forward_pre_hook(ph))
                else:
                    def mh(m_, a, o, u=u):
                        o = o.clone(); o[ar, cb_t] = capb[u].to(o.dtype); return o
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

    def along(W, d):
        """component of the (B,T,128) write along unit direction d, at every position."""
        return (W @ d)[..., None] * d[None, None, :]

    gen = torch.Generator(device="cpu").manual_seed(BARS["seed"])
    R = {}
    for fam, par in cells_run:
        C = cells[(fam, par)]
        whole = C["run"](add=C["W"]); zero = C["run"]()
        fr = lambda x: round((x - zero) / (whole - zero), 3) if abs(whole - zero) > 1e-6 else None
        own = fr(C["run"](add=along(C["W"], C["direction"])))
        cost = v177.get(f"{fam}:{par}", {}).get("removal_cost", {}).get(f"{HUB:02d}")
        ref = v174.get(f"{fam}:{par}", {}).get("single_slice", {}).get(str(CH))
        pair, cos, pnorm = {}, {}, {}
        for other in FAMILIES:
            if other == fam or (other, par) not in cells: continue
            d = cells[(other, par)]["direction"]
            pair[other] = fr(C["run"](add=along(C["W"], d))); cos[other] = round(float(C["direction"] @ d), 3)
            pnorm[other] = round(float(along(C["W"], d).norm() / C["W"].norm()), 3)
        r1 = torch.randn(HD, generator=gen).to(C["direction"].device, C["direction"].dtype); r1 = r1 / r1.norm()
        rnd = fr(C["run"](add=along(C["W"], r1)))
        S = {"rows": C["rows"], "whole_write": whole, "zero": zero, "full_slice8_swap": C["full_swap"], "v174_slice8": ref, "v177_cost_07": cost,
             "linear_expectation": round(cost * ref, 3) if cost is not None and ref is not None else None,
             "whole_over_linear": round(whole / (cost * ref), 3) if cost and ref else None,
             "own_rank1": own, "rank1_frac_rows": C["rank1_frac"], "wt_norm": C["wt_norm"], "t_share_of_norm": C["t_share_of_norm"],
             "own_norm_frac": round(float(along(C["W"], C["direction"]).norm() / C["W"].norm()), 3),
             "dir_vs_delta_cos": round(float(abs(C["direction"] @ (C["delta"] / C["delta"].norm()))), 4),
             "pair": pair, "pair_cos": cos, "pair_norm_frac": pnorm, "random1": rnd}
        R[f"{fam}:{par}"] = S
        print(fam, par, S, flush=True)
    print(round(time.perf_counter() - t0), "s", flush=True)
    predictions = PREDS(R)
    result = {"predictions": predictions, "schema": "unit_tier5_channel8_shared_subspace_v179", "candidate_id": "corpus.unit_tier5_hub0708_write_subspace_v179",
              "bars": BARS, "families": list(FAMILIES), "channel": CH,
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


def PREDS(R):
    B = BARS
    cells = list(R.values())
    inb = lambda x, band: x is not None and band[0] <= x <= band[1]
    pred_a = sum(inb(S["whole_over_linear"], B["linear_band"]) for S in cells) >= B["linear_min_cells"] and all(S["zero"] == 0.0 for S in cells)
    pred_b = (sum(inb(S["own_rank1"], B["rank1_band"]) for S in cells) >= B["rank1_min_cells"]
              and all(S["dir_vs_delta_cos"] >= B["cos_min"] for S in cells))
    def projs(a, c):
        return [R[f"{x}:{p}"]["pair"].get(y) for x, y in ((a, c), (c, a)) for p in ("p0", "p1") if f"{x}:{p}" in R]
    same = projs("polarity_state", "coordination_agreement")
    sh1 = projs("polarity_state", "correlative_both_neither"); sh2 = projs("correlative_both_either", "correlative_both_neither")
    pred_c = (len(same) == 4 and all(inb(v, B["same_reader_band"]) for v in same)
              and len(sh1) == 4 and all(v is not None and v >= B["shared_min"] for v in sh1)
              and len(sh2) == 4 and all(v is not None and v >= B["shared_min"] for v in sh2))
    cand = set()
    for a, c in PAIRS:
        cand.add((a, c)); cand.add((c, a))
    others = [v for k, S in R.items() for o, v in S["pair"].items() if (k.split(":")[0], o) not in cand]
    pred_d = bool(others) and sum(inb(v, B["pair_band"]) for v in others) / len(others) >= B["pair_min_frac"]
    pred_e = sum(inb(S["random1"], B["random_band"]) for S in cells) >= B["random_min_cells"]
    return {"pred_a_instrument": pred_a, "pred_b_rank1": pred_b, "pred_c_reader_not_direction": pred_c, "pred_d_other_pairs": pred_d, "pred_e_random": pred_e}


if __name__ == "__main__":
    main()
