#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: on the four shared number units, is the unit-level object rank-2 per block, or does one pooled direction do?

v153: per-block |cos| between lexical and quantifier diff-in-means on 11:03 / mlp:08 / mlp:09 / mlp:10 is 0.49 / 0.17 /
0.24 / 0.17, and the rank-2 union (QR of both) recovers 0.785 / 0.626 vs exact 0.798 / 0.616 with complement 0.075 / -0.007.
Test the cheaper alternative: ONE pooled rank-1 direction per block -- (i) pooled = sign-aligned sum of the two sets'
oriented mean deltas (norm-weighted), (ii) bisector = sum of the two unit directions -- evaluated block-live on the
held-out parity-1 rows of both sets. If the writes are cue-specific, a bisector of cos-0.49 / cos-0.2 directions keeps
about (1+cos)/2 = 0.6-0.75 of each delta's norm along it and the pooled direction lands in 0.5-0.8 x exact.
Row population (ops/row_population.py): lexical A1 p0 'rows 16 (unequal/misaligned 1); cue columns -> distinct pairs {1: 15}',
p1 'rows 16 ... {1: 16}'; quantifier A1 p0 'rows 16 (unequal/misaligned 0); cue columns -> distinct pairs {0: 1}', p1 same.

Registered before the run:
  pred_a_instrument   rank-2 union reproduces v153 within 0.03 on both sets (0.785 lexical, 0.626 quantifier); |random| <= 0.05 on both
  pred_b_pooled       norm-weighted pooled rank-1 recovers 0.5-0.8 x exact(four) on BOTH held-out sets
  pred_c_bisector     equal-weight bisector rank-1 recovers 0.5-0.8 x exact(four) on BOTH held-out sets
  pred_d_linear       pooled subspace + pooled complement within 0.85-1.15 on both sets (the missing part sits in the complement)
  pred_e_norm_order   per-block delta-norm ratio lexical/quantifier > 1 on 11:heads and < 1 on 08:mlp (v149/v152: 11:03 carries
                      0.618 of lexical's axis projection but 0.124 of quantifier's; quantifier leans on the mlp7-10 chain)
Reported, unregistered: per-block norms, cos(pooled, q_lex) and cos(pooled, q_q), exact(four) on each set, own q on each set.
Prior: b/c expected to hold if v153's cosines describe the geometry (pooled is norm-weighted so it may lean to one set and
exceed 0.8 on that set -- that would be an informative fail); d likely; e is the v149/v152 accounting re-read per block.
Smoke: V154_SMOKE=<out.json> -> CPU, 4 rows. The 4-row smoke hinted pooled ~ own on both sets (0.62/0.68 lexical,
0.72/0.62 quantifier; MLP norms 2.8x larger on quantifier so pooled leans to it). Bars b/c kept as registered: a fail then
reads 'the cue-specific parts of the per-unit directions are margin-inert', which is the informative outcome.
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
OUT = ROOT / "circuits/followups/unit_tier5_shared_units_pooled_rank1_v154_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
REPRO_TOL, RAND_MAX, POOL_LO, POOL_HI, LIN_LO, LIN_HI = 0.03, 0.05, 0.5, 0.8, 0.85, 1.15
V153_UNION = {"lexical_number_pp": 0.785, "quantifier_number": 0.626}
FOUR = ("attn:11:head:03", "mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_shared_units_pooled_rank1_v154", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V154_SMOKE")
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
    L, N = "lexical_number_pp", "quantifier_number"
    union, pooled, bisect, norms, pcos = {}, {}, {}, {}, {}
    for key, us in g.blocks_of(units).items():
        means = {}
        for n in (L, N):
            delta = g._cached_delta(backend, P[n]["p0"], us)
            means[n] = (delta * g._orientation(delta)[:, None]).mean(0)
        sgn = 1.0 if float(means[L] @ means[N]) >= 0 else -1.0
        m = means[L] + sgn * means[N]
        pooled[key] = (m / m.norm()).unsqueeze(1)
        bl = Q[L][key][:, 0] + sgn * Q[N][key][:, 0]
        bisect[key] = (bl / bl.norm()).unsqueeze(1)
        qq, _ = torch.linalg.qr(torch.cat([Q[L][key], Q[N][key].to(Q[L][key].device)], dim=1))
        union[key] = qq
        kk = f"{key[0]:02d}:{key[1]}"
        norms[kk] = {L: round(float(means[L].norm()), 3), N: round(float(means[N].norm()), 3), "ratio_lex_over_q": round(float(means[L].norm() / means[N].norm()), 3)}
        pcos[kk] = {"pooled_vs_lex": round(abs(float(pooled[key][:, 0] @ Q[L][key][:, 0])), 3), "pooled_vs_q": round(abs(float(pooled[key][:, 0] @ Q[N][key][:, 0])), 3),
                    "lex_vs_q": round(abs(float(Q[L][key][:, 0] @ Q[N][key][:, 0])), 3)}
    def rec(n, q=None, complement=False):
        prep = P[n]["p1"]
        return round(g.recovery(prep, g.patched_axis(backend, prep, units, q=q, complement=complement)), 3)
    R = {"norms": norms, "cos": pcos,
         "exact": {L: rec(L), N: rec(N)}, "own": {L: rec(L, Q[L]), N: rec(N, Q[N])},
         "union": {L: rec(L, union), N: rec(N, union)}, "union_comp": {L: rec(L, union, True), N: rec(N, union, True)},
         "pooled": {L: rec(L, pooled), N: rec(N, pooled)}, "pooled_comp": {L: rec(L, pooled, True), N: rec(N, pooled, True)},
         "bisector": {L: rec(L, bisect), N: rec(N, bisect)}, "bisector_comp": {L: rec(L, bisect, True), N: rec(N, bisect, True)},
         "random": {L: rec(L, q_rand), N: rec(N, q_rand)},
         "rows": {n: len(P[n]["p1"].base_batch.row_ids) for n in P}}
    R["pooled_frac"] = {n: round(R["pooled"][n] / R["exact"][n], 3) for n in (L, N)}
    R["bisector_frac"] = {n: round(R["bisector"][n] / R["exact"][n], 3) for n in (L, N)}
    R["pooled_linear"] = {n: round(R["pooled"][n] + R["pooled_comp"][n], 3) for n in (L, N)}
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = all(abs(R["union"][n] - V153_UNION[n]) <= REPRO_TOL and abs(R["random"][n]) <= RAND_MAX for n in (L, N))
    pred_b = all(POOL_LO <= R["pooled_frac"][n] <= POOL_HI for n in (L, N))
    pred_c = all(POOL_LO <= R["bisector_frac"][n] <= POOL_HI for n in (L, N))
    pred_d = all(LIN_LO <= R["pooled_linear"][n] <= LIN_HI for n in (L, N))
    pred_e = norms["11:heads"]["ratio_lex_over_q"] > 1 and norms["08:mlp"]["ratio_lex_over_q"] < 1
    predictions = {"pred_a_instrument": pred_a, "pred_b_pooled": pred_b, "pred_c_bisector": pred_c, "pred_d_linear": pred_d, "pred_e_norm_order": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_shared_units_pooled_rank1_v154", "candidate_id": "corpus.unit_tier5_shared_units_pooled_rank1_v154",
              "bars": {"repro_tol": REPRO_TOL, "rand_max": RAND_MAX, "pool_band": [POOL_LO, POOL_HI], "linear_band": [LIN_LO, LIN_HI], "v153_union": V153_UNION},
              "units": units, "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
