#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: quantifier's 'brake' is the exact patch clamping 11:03's upstream-induced response.

v155/v156: with quantifier's other six blocks exact-patched, 11:03 rank-1 (block-live) recovers 0.901 vs exact 0.844,
proportional to the total patched margin, with no single partner. The block-live rank-1 patch is live + q q^T (donor -
live): it keeps the LIVE complement, which under upstream patches already contains 11:03's response to the patched
MLP state; the exact patch clamps the whole slice to the donor's value and discards that response. Hypothesis: in
quantifier_number 11:03 READS the mlp7-10 chain -- patching the six alone moves 11:03's output a good part of the way
to the donor -- and the off-q part of that induced shift favours the answer; lexical's 11:03 reads the cue itself, so
nothing is induced and nothing is clamped. Measured on held-out parity-1 rows: 11:03's c_proj-input slice under {six
exact} (live), base and donor caches; shift = live - base; its q part and complement; W_O images added by resid_add at
layer 11. A same-norm random add and the norm ratio (complement image / residual) are reported as the refuted
norm-inflation alternative (4-row smoke: residual norm ~26,000 vs image ~170; random adds 0.000).
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 'rows 16 ... {1: 16}'; quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_closure      {six exact + 11:03 rank-1} reproduces 0.901 within 0.02, and adding the image of (donor_comp - live_comp)
                      on top returns to exact 0.844 within 0.02 (the accounting closes)
  pred_b_induced      quantifier: |q part of shift| / |q part of (donor - base)| >= 0.3 (11:03 reads the chain);
                      lexical: < 0.15 (11:03 reads the cue)
  pred_c_axis         the oriented mean complement-shift image has cos >= 0.3 with quantifier's layer-11 residual number axis
                      (donor - base residual diff-in-means at layer 11, v147)
  pred_d_lex_inert    on lexical {six exact + 11:03 exact} adding the complement-shift image changes recovery by < 0.02
  pred_e_dose         on quantifier {six exact + 11:03 exact} adding k x complement-shift image, k = 0.5 / 1 / 2, is monotone
                      and k = 1 lands at 0.901 within 0.03
Reported, unregistered: same-norm random adds (five seeds), norm ratios, the shift's per-row norms.
Prior: a should hold by algebra (if it fails the semantics differ from my reading of apply()); b/c are the mechanism; d is
the lexical control; e is linearity of the readout at this dose.
Smoke: V157_SMOKE=<out.json> -> CPU, 4 rows. 4-row smoke: closure exact, dose 0.024/0.047/0.090, lexical inert, axis cos 0.40,
induced q ratio 0.139 (the shift is small, 27 vs 102, and 85% off-q) -- bar b kept as registered.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_batch_v112 as v112
import run_unit_tier5_carrier_relay_v120 as v120
import run_unit_tier5_near_carrier_heads_v123 as v123
import run_unit_tier5_near_value_source_v131 as v131
import run_unit_tier5_near_value_mid_remainder_v132 as v132

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier5_quantifier_brake_clamp_v157_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
REPRO_TOL, INDUCED_MIN, INDUCED_LEX_MAX, AXIS_COS_MIN, LEX_TOL, DOSE_TOL, SEEDS = 0.02, 0.3, 0.15, 0.3, 0.02, 0.03, 5
V155_SIX, V155_EXACT, V155_GAIN, HUB, HUB_H, LAYER = 0.901, 0.844, 0.057, "attn:11:head:03", 3, 11
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_quantifier_brake_clamp_v157", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V157_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    P = {}
    for n in ("lexical_number_pp", "quantifier_number"):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        P[n] = {"p0": g.prepare(backend, cut(a1[0::2])), "p1": g.prepare(backend, cut(a1[1::2]))}
    R = {}
    W = backend.model.transformer.h[LAYER].attn.c_proj.weight
    Wh = W[:, HUB_H * g.HEAD_DIM:(HUB_H + 1) * g.HEAD_DIM]
    for sname, (n, units, _) in SETS.items():
        if sname == "shared_four":
            continue
        units = list(units)
        six = [u for u in units if u != HUB]
        hub_q = g.block_diff_in_means(backend, P[n]["p0"], [HUB])
        qv = hub_q[(LAYER, "heads")][:, 0]
        prep = P[n]["p1"]
        positions = list(prep.base_batch.semantic_positions)
        got = {}
        def grab(_m, args):
            v = args[0]
            got["live"] = torch.stack([v[i, positions[i], HUB_H * g.HEAD_DIM:(HUB_H + 1) * g.HEAD_DIM] for i in range(len(positions))]).detach().float().clone()
        h = backend.model.transformer.h[LAYER].attn.c_proj.register_forward_pre_hook(grab)
        try:
            resid = {}
            g.forward_units(backend, prep.base_batch, units=six, donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=resid)
        finally:
            h.remove()
        with torch.no_grad():
            live = got["live"].to(backend.device)
            base = torch.stack([torch.as_tensor(prep.base_cache[(rid, HUB)]) for rid in prep.base_batch.row_ids]).float().to(backend.device)
            donor = torch.stack([torch.as_tensor(prep.donor_cache[(rid, HUB)]) for rid in prep.base_batch.row_ids]).float().to(backend.device)
            delta, shift = donor - base, live - base
            qpart = lambda t: (t @ qv)[:, None] * qv[None, :]
            shift_comp, delta_comp, live_comp, donor_comp = shift - qpart(shift), delta - qpart(delta), live - qpart(live), donor - qpart(donor)
            img = lambda t: (t.to(Wh.dtype) @ Wh.T).float()
            close_img, shift_img = img(donor_comp - live_comp), img(shift_comp)
            induced = float((shift @ qv).abs().mean() / (delta @ qv).abs().mean())
            # layer-11 residual number axis: donor - base residual at t, oriented, unit norm
            dres = {}
            g.forward_units(backend, prep.donor_batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=dres)
            bres = {}
            g.forward_units(backend, prep.base_batch, units=(), donor_cache=prep.donor_cache, base_cache=prep.base_cache, capture_resid=bres)
            rdelta = torch.stack([dres[(rid, LAYER)] for rid in prep.donor_batch.row_ids]) - torch.stack([bres[(rid, LAYER)] for rid in prep.base_batch.row_ids])
            rdelta = rdelta.to(backend.device)
            sgn = g._orientation(rdelta)
            axis = (rdelta * sgn[:, None]).mean(0); axis = axis / axis.norm()
            sh = (shift_img * sgn[:, None]).mean(0)
            axis_cos = float(sh @ axis / sh.norm())
            rnorm = torch.stack([bres[(rid, LAYER)] for rid in prep.base_batch.row_ids]).norm(dim=1).to(backend.device)
            comp_norm = img(delta_comp).norm(dim=1)
        def rec(us, q=None, add=None):
            out = g.forward_units(backend, prep.base_batch, units=us, donor_cache=prep.donor_cache, base_cache=prep.base_cache, q=q,
                                  resid_add=None if add is None else {LAYER: add})
            return round(g.recovery(prep, [-(float(x) - float(f)) for x, f in out.tolist()]), 3)
        def rand_add(seed):
            gen = torch.Generator(device="cpu").manual_seed(seed)
            v = torch.randn(close_img.shape, generator=gen).to(close_img.device)
            return v / v.norm(dim=1, keepdim=True) * comp_norm[:, None]
        exact, r1 = rec(units), rec(units, hub_q)
        S = {"exact": exact, "hub_rank1": r1, "rank1_plus_close_image": rec(units, hub_q, close_img),
             "exact_plus_shift_comp": {k: rec(units, None, k * shift_img) for k in (0.5, 1.0, 2.0)},
             "induced_q_ratio": round(induced, 3), "shift_norm_mean": round(float(shift.norm(dim=1).mean()), 2),
             "delta_norm_mean": round(float(delta.norm(dim=1).mean()), 2), "shift_comp_norm_mean": round(float(shift_comp.norm(dim=1).mean()), 2),
             "axis_cos": round(axis_cos, 3), "random_same_norm": [rec(units, hub_q, rand_add(sd)) for sd in range(SEEDS)],
             "norm_ratio": round(float((comp_norm / rnorm).mean()), 4), "resid_norm_mean": round(float(rnorm.mean()), 1),
             "rows": len(prep.base_batch.row_ids)}
        S["dose_effect"] = {str(k): round(v - exact, 3) for k, v in S["exact_plus_shift_comp"].items()}
        R[sname] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    Qs, Lx = R["quant_seven"], R["lex_seven"]
    pred_a = abs(Qs["hub_rank1"] - V155_SIX) <= REPRO_TOL and abs(Qs["rank1_plus_close_image"] - V155_EXACT) <= REPRO_TOL
    pred_b = Qs["induced_q_ratio"] >= INDUCED_MIN and Lx["induced_q_ratio"] < INDUCED_LEX_MAX
    pred_c = Qs["axis_cos"] >= AXIS_COS_MIN
    pred_d = abs(Lx["dose_effect"]["1.0"]) < LEX_TOL
    de_ = Qs["dose_effect"]
    pred_e = de_["2.0"] > de_["1.0"] > de_["0.5"] and abs(Qs["exact_plus_shift_comp"][1.0] - V155_SIX) <= DOSE_TOL
    predictions = {"pred_a_closure": pred_a, "pred_b_induced": pred_b, "pred_c_axis": pred_c, "pred_d_lex_inert": pred_d, "pred_e_dose": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_quantifier_brake_clamp_v157", "candidate_id": "corpus.unit_tier5_quantifier_brake_clamp_v157",
              "bars": {"repro_tol": REPRO_TOL, "induced_min": INDUCED_MIN, "induced_lex_max": INDUCED_LEX_MAX, "axis_cos_min": AXIS_COS_MIN, "lex_tol": LEX_TOL, "dose_tol": DOSE_TOL, "v155": [V155_SIX, V155_EXACT, V155_GAIN]},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items() if k != "shared_four"}, "measures": {k: {kk: ({str(a): b for a, b in vv.items()} if isinstance(vv, dict) else vv) for kk, vv in v.items()} for k, v in R.items()},
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
