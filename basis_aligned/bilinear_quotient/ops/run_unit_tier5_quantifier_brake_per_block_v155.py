#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: which block carries quantifier's margin-opposing component (the 'brake')?

v152/v153/v154: on quantifier_number every rank-1 diff-in-means intervention beats the exact patch (seven-set dim/exact
1.15; shared-four own 1.09, bisector 1.26, pooled 1.32) with a NEGATIVE complement (-0.09), while lexical_number_pp is
linear (seven-set dim 0.840 / exact 0.850, S+C 0.93). So quantifier's per-unit deltas carry a component orthogonal to
their diff-in-means that opposes the answer margin. Locate it per block: for each block k of a set, patch block k at
rank 1 (its own diff-in-means from parity-0 rows) and every other block at full rank (== exact); the overshoot of block k
is that recovery minus exact. Single-block arms (only block k's units patched: exact / rank-1 / complement) give the
block's own sign. Sets: quantifier's own seven (v152 order), the shared four (v153), lexical's seven (v151) as the linear
control. q from parity 0, evaluated on held-out parity 1.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 'rows 16 ... {1: 16}'; quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_instrument   full-rank identity q reproduces exact within 0.01 on all three sets; all-block dim reproduces v152 / v153 / v151
                      within 0.03 (quantifier seven 0.969, shared four 0.673, lexical seven 0.840)
  pred_b_mlp_share    on BOTH quantifier sets the MLP blocks carry >= 0.7 of the summed per-block overshoot (sum over blocks of
                      [rank-1-block-k-others-exact minus exact], MLP part / all, all > 0)
  pred_c_mlp_negative on quantifier seven, every MLP block whose single-block exact >= 0.05 has single-block complement <= 0.00,
                      and at least two such blocks exist
  pred_d_lexical_flat on lexical seven every per-block overshoot lies within +-0.05 and all-block dim minus exact within +-0.05
  pred_e_additive     on quantifier seven the per-block overshoots sum to the total overshoot (all-block dim minus exact) within +-0.05
Reported, unregistered: per-block single-block exact / rank-1 / complement on every set.
Prior: b likely (quantifier's MLP deltas are 2.5-4x lexical's and the Bilinear MLP writes 1000-norm deltas); c open; d is the
v151 control; e open (per-block overshoots may interact through the later blocks).
Smoke: V155_SMOKE=<out.json> -> CPU, 4 rows. The 4-row smoke put the overshoot on 11:heads (0.06 of 0.06) with the MLP blocks
flat -- bars kept as registered; a b-fail then reads 'the brake is 11:03's own write, not the MLPs'.
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
OUT = ROOT / "circuits/followups/unit_tier5_quantifier_brake_per_block_v155_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
IDENT_TOL, REPRO_TOL, MLP_SHARE_MIN, SINGLE_MIN, FLAT_TOL, ADD_TOL = 0.01, 0.03, 0.7, 0.05, 0.05, 0.05
SETS = {"quant_seven": ("quantifier_number", ("mlp:10", "attn:11:head:03", "mlp:09", "mlp:08", "attn:07:head:08", "attn:08:head:01", "mlp:07"), 0.969),
        "shared_four": ("quantifier_number", ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10"), 0.673),
        "lex_seven": ("lexical_number_pp", ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08"), 0.840)}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_quantifier_brake_per_block_v155", "behaviours": 2, "targets": 3,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V155_SMOKE")
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
    for sname, (n, units, prior_dim) in SETS.items():
        units = list(units)
        dim = g.block_diff_in_means(backend, P[n]["p0"], units)
        prep = P[n]["p1"]
        def rec(us, q=None, complement=False):
            return round(g.recovery(prep, g.patched_axis(backend, prep, us, q=q, complement=complement)), 3)
        exact = rec(units)
        S = {"exact": exact, "identity": rec(units, g.block_identity(backend, units)), "dim_all": rec(units, dim), "comp_all": rec(units, dim, True), "prior_dim": prior_dim, "blocks": {}}
        for key, us in g.blocks_of(units).items():
            kk = f"{key[0]:02d}:{key[1]}"
            one = {key: dim[key]}
            S["blocks"][kk] = {"units": list(us), "rank1_others_exact": rec(units, one), "overshoot": round(rec(units, one) - exact, 3),
                               "single_exact": rec(list(us)), "single_dim": rec(list(us), one), "single_comp": rec(list(us), one, True)}
        over = {kk: v["overshoot"] for kk, v in S["blocks"].items()}
        S["overshoot_sum"] = round(sum(over.values()), 3)
        S["overshoot_total"] = round(S["dim_all"] - exact, 3)
        mlp = sum(v for kk, v in over.items() if kk.endswith("mlp"))
        S["mlp_share"] = round(mlp / S["overshoot_sum"], 3) if S["overshoot_sum"] > 0 else None
        S["rows"] = len(prep.base_batch.row_ids)
        R[sname] = S
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = all(abs(S["identity"] - S["exact"]) <= IDENT_TOL and abs(S["dim_all"] - S["prior_dim"]) <= REPRO_TOL for S in R.values())
    pred_b = all(R[s]["overshoot_sum"] > 0 and R[s]["mlp_share"] is not None and R[s]["mlp_share"] >= MLP_SHARE_MIN for s in ("quant_seven", "shared_four"))
    qm = [v for kk, v in R["quant_seven"]["blocks"].items() if kk.endswith("mlp") and v["single_exact"] >= SINGLE_MIN]
    pred_c = len(qm) >= 2 and all(v["single_comp"] <= 0.0 for v in qm)
    lx = R["lex_seven"]
    pred_d = all(abs(v["overshoot"]) <= FLAT_TOL for v in lx["blocks"].values()) and abs(lx["overshoot_total"]) <= FLAT_TOL
    pred_e = abs(R["quant_seven"]["overshoot_sum"] - R["quant_seven"]["overshoot_total"]) <= ADD_TOL
    predictions = {"pred_a_instrument": pred_a, "pred_b_mlp_share": pred_b, "pred_c_mlp_negative": pred_c, "pred_d_lexical_flat": pred_d, "pred_e_additive": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_quantifier_brake_per_block_v155", "candidate_id": "corpus.unit_tier5_quantifier_brake_per_block_v155",
              "bars": {"ident_tol": IDENT_TOL, "repro_tol": REPRO_TOL, "mlp_share_min": MLP_SHARE_MIN, "single_min": SINGLE_MIN, "flat_tol": FLAT_TOL, "add_tol": ADD_TOL},
              "sets": {k: {"family": v[0], "units": list(v[1])} for k, v in SETS.items()}, "measures": R,
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
