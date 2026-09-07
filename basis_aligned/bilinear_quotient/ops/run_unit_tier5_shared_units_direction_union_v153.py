#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the four units shared by the lexical and quantifier number sets -- same units, same per-block directions?

v151/v152: lexical's set and quantifier's set share attn:11:03, mlp:08, mlp:09, mlp:10. The residual number axis at layer
11 is shared (v147: lexical dim on quantifier rows 0.918), yet lexical's per-block directions on its seven units recover
only 0.323 on quantifier rows (complement 0.361). Test on the SHARED FOUR: per-block diff-in-means q_lex (lexical A1
parity 0) and q_q (quantifier A1 parity 0); per-block cosines; a rank-2 union per block (QR of [q_lex, q_q]); all
evaluated block-live on the held-out parity-1 rows of each set (16 + 16). Random = block_random_subspace seed 1.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 'rows 16 ... {1: 16}'; quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_instrument   own q on own held-out p1 >= 0.8 x exact(four) on both sets; |random| <= 0.05 on both
  pred_b_head_shared  cos(q_lex, q_q) on block 11:heads >= 0.7   (11:03's write direction is the same number direction for both cue classes)
  pred_c_mlp_specific cos(q_lex, q_q) < 0.5 on each of 08:mlp, 09:mlp, 10:mlp   (the MLP blocks carry cue-specific directions)
  pred_d_union        rank-2 union recovers >= 0.8 x exact(four) on BOTH held-out sets
  pred_e_union_comp   complement of the union <= 0.20 on both
Reported, unregistered: cross q (lexical q on quantifier rows and the reverse), exact(four) on each set, per-block cross
recoveries when only one block is swapped to the other set's direction.
Prior: b open (v145: the CUE-column vectors had cos -0.156, but the whole write at t may align), c open, d likely if b/c
describe the geometry, e likely.
Smoke: V153_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_shared_units_direction_union_v153_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
OWN_FRAC, RAND_MAX, HEAD_COS_MIN, MLP_COS_MAX, UNION_FRAC, COMP_MAX = 0.8, 0.05, 0.7, 0.5, 0.8, 0.20
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_shared_units_direction_union_v153", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V153_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    units = list(FOUR)
    P, Q = {}, {}
    for n in ("lexical_number_pp", "quantifier_number"):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        a1 = g.rows_of(m, "A1")
        P[n] = {"p0": g.prepare(backend, cut(a1[0::2])), "p1": g.prepare(backend, cut(a1[1::2]))}
        Q[n] = g.block_diff_in_means(backend, P[n]["p0"], units)
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
    union = {}
    for key in Q["lexical_number_pp"]:
        stacked = torch.cat([Q["lexical_number_pp"][key], Q["quantifier_number"][key].to(Q["lexical_number_pp"][key].device)], dim=1)
        qq, _ = torch.linalg.qr(stacked)
        union[key] = qq
    def rec(n, q=None, complement=False):
        prep = P[n]["p1"]
        return round(g.recovery(prep, g.patched_axis(backend, prep, units, q=q, complement=complement)), 3)
    L, N = "lexical_number_pp", "quantifier_number"
    def swapped(own, other, key):
        q = dict(Q[own]); q[key] = Q[other][key]; return q
    R = {"cos_blocks": {k: round(float(v), 3) for k, v in g.block_cosines(Q[L], Q[N]).items()},
         "exact": {L: rec(L), N: rec(N)}, "own": {L: rec(L, Q[L]), N: rec(N, Q[N])}, "own_comp": {L: rec(L, Q[L], True), N: rec(N, Q[N], True)},
         "cross": {L: rec(L, Q[N]), N: rec(N, Q[L])}, "cross_comp": {L: rec(L, Q[N], True), N: rec(N, Q[L], True)},
         "union": {L: rec(L, union), N: rec(N, union)}, "union_comp": {L: rec(L, union, True), N: rec(N, union, True)},
         "random": {L: rec(L, q_rand), N: rec(N, q_rand)},
         "one_block_swapped": {L: {f"{k[0]:02d}:{k[1]}": rec(L, swapped(L, N, k)) for k in Q[L]}, N: {f"{k[0]:02d}:{k[1]}": rec(N, swapped(N, L, k)) for k in Q[N]}},
         "rows": {n: len(P[n]["p1"].base_batch.row_ids) for n in P}}
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = all(R["own"][n] >= OWN_FRAC * R["exact"][n] and abs(R["random"][n]) <= RAND_MAX for n in (L, N))
    pred_b = R["cos_blocks"]["11:heads"] >= HEAD_COS_MIN
    pred_c = all(R["cos_blocks"][k] < MLP_COS_MAX for k in ("08:mlp", "09:mlp", "10:mlp"))
    pred_d = all(R["union"][n] >= UNION_FRAC * R["exact"][n] for n in (L, N))
    pred_e = all(R["union_comp"][n] <= COMP_MAX for n in (L, N))
    predictions = {"pred_a_instrument": pred_a, "pred_b_head_shared": pred_b, "pred_c_mlp_specific": pred_c, "pred_d_union": pred_d, "pred_e_union_comp": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_shared_units_direction_union_v153", "candidate_id": "corpus.unit_tier5_shared_units_direction_union_v153",
              "bars": {"own_frac": OWN_FRAC, "rand_max": RAND_MAX, "head_cos_min": HEAD_COS_MIN, "mlp_cos_max": MLP_COS_MAX, "union_frac": UNION_FRAC, "comp_max": COMP_MAX},
              "units": units, "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
