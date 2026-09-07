#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier lift: the seven-unit number set (11:03 + the six remainder writers) through the block-live rank-1 rubric.

v150: exact interchange of {attn:11:03, mlp:09, attn:09:07, mlp:10, attn:10:05, attn:11:02, mlp:08} on lexical parity-1
rows = 0.850 of the donor margin. Standing recipe (v79/v85, diff-in-means primary): per-block unit-norm diff-in-means
q = g.block_diff_in_means fitted on A1 parity-0 rows (16), evaluated block-live via g.patched_axis(q=...) on held-out rows;
complement = the same patch on the orthogonal complement (complement=True); random = g.block_random_subspace(seed=1).
Recovery = g.recovery (signed fraction of the donor margin). Split-swap (fit parity 1, eval parity 0) reported.
Evaluation rows: A1 parity 1 (16, was -> were: the REVERSE interchange direction of the fit rows, so every held-out number
is also a reverse-direction number), A2 parity 1 (16, held-out frame, cue column 4), C (32 rows, night -> night,
g.same_answer_effect in units of A1's median |d - b|).
Row population (ops/row_population.py): A1 p0 16 (1 misaligned row is kept: block-live patching does not need aligned cues),
A1 p1 16, A2 p1 16, C 32.

Registered before the run:
  pred_a_instrument   exact set on A1 parity 1 = 0.850 +- 0.03 (v150); |random| <= 0.05 on A1 parity 1
  pred_b_heldout      dim on A1 parity 1 >= 0.60
  pred_c_linearity    |S + C - exact| <= 0.15 on A1 parity 1  (S = dim, C = complement)
  pred_d_frame        dim on A2 parity 1 >= 0.50
  pred_e_C_inert      same-answer effect of dim on C <= 0.15
Reported, unregistered: split-swap S/C, per-block cosines between the two fits, complement on A2.
Prior: b likely (v147: the residual direction is rank-1 across nouns; the per-block deltas may be less so), c open (MLP
blocks are Bilinear: S + C > 1 would mean a nonlinear route), d open, e likely.
Smoke: V151_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_number_seven_set_rubric_v151_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
V150_EXACT, EXACT_TOL, RAND_MAX, HELDOUT_MIN, LIN_TOL, FRAME_MIN, C_MAX = 0.850, 0.03, 0.05, 0.60, 0.15, 0.50, 0.15
SEVEN = ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_seven_set_rubric_v151", "behaviours": 1, "targets": 4,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V151_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    m = importlib.import_module(f"circuit_fast_screen_candidate_{names['lexical_number_pp']}")
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    a1 = g.rows_of(m, "A1")
    P = {"A1p0": g.prepare(backend, cut(a1[0::2])), "A1p1": g.prepare(backend, cut(a1[1::2])),
         "A2p1": g.prepare(backend, cut(g.rows_of(m, "A2")[1::2])), "C": g.prepare(backend, cut(g.rows_of(m, "C")))}
    units = list(SEVEN)
    q_p0 = g.block_diff_in_means(backend, P["A1p0"], units)
    q_p1 = g.block_diff_in_means(backend, P["A1p1"], units)
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
    scale = g.target_scale(P["A1p1"])

    def rec(prep, q=None, complement=False, exact=False):
        pa = g.patched_axis(backend, prep, units, q=None if exact else q, complement=complement)
        return round(g.recovery(prep, pa), 3)

    def same(prep, q, complement=False):
        return round(g.same_answer_effect(prep, g.patched_axis(backend, prep, units, q=q, complement=complement), scale), 3)

    R = {"exact_A1p1": rec(P["A1p1"], exact=True), "exact_A2p1": rec(P["A2p1"], exact=True), "exact_A1p0": rec(P["A1p0"], exact=True),
         "dim_A1p1": rec(P["A1p1"], q_p0), "comp_A1p1": rec(P["A1p1"], q_p0, complement=True), "rand_A1p1": rec(P["A1p1"], q_rand),
         "dim_A2p1": rec(P["A2p1"], q_p0), "comp_A2p1": rec(P["A2p1"], q_p0, complement=True), "rand_A2p1": rec(P["A2p1"], q_rand),
         "swap_dim_A1p0": rec(P["A1p0"], q_p1), "swap_comp_A1p0": rec(P["A1p0"], q_p1, complement=True),
         "C_same_dim": same(P["C"], q_p0), "C_same_exact": round(g.same_answer_effect(P["C"], g.patched_axis(backend, P["C"], units), scale), 3),
         "C_same_rand": same(P["C"], q_rand), "block_cos_p0_p1": {k: round(float(v), 3) for k, v in g.block_cosines(q_p0, q_p1).items()},
         "scale_logits": round(scale, 2), "rows": {k: len(v.base_batch.row_ids) for k, v in P.items()}}
    R["S_plus_C_A1p1"] = round(R["dim_A1p1"] + R["comp_A1p1"], 3)
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = (smoke or abs(R["exact_A1p1"] - V150_EXACT) <= EXACT_TOL) and abs(R["rand_A1p1"]) <= RAND_MAX
    pred_b = R["dim_A1p1"] >= HELDOUT_MIN
    pred_c = abs(R["S_plus_C_A1p1"] - R["exact_A1p1"]) <= LIN_TOL
    pred_d = R["dim_A2p1"] >= FRAME_MIN
    pred_e = R["C_same_dim"] <= C_MAX
    predictions = {"pred_a_instrument": pred_a, "pred_b_heldout": pred_b, "pred_c_linearity": pred_c, "pred_d_frame": pred_d, "pred_e_C_inert": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_seven_set_rubric_v151", "candidate_id": "corpus.unit_tier5_number_seven_set_rubric_v151",
              "bars": {"v150_exact": V150_EXACT, "exact_tol": EXACT_TOL, "rand_max": RAND_MAX, "heldout_min": HELDOUT_MIN, "lin_tol": LIN_TOL, "frame_min": FRAME_MIN, "c_max": C_MAX},
              "units": units, "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
