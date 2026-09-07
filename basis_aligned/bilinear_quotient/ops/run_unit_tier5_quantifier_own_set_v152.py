#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier lift: quantifier_number's OWN unit set, named by lambda0-scaled accounting, through the block-live rubric.

v147: the residual number direction at layer 11 is shared across cue classes (lexical dim_11 on quantifier rows 0.918).
v145: quantifier's 11:03 cue vector is NOT lexical's (cos -0.156). So which units write quantifier's dim_11? Procedure,
fixed in advance: (1) on quantifier A1 parity-0 rows (16, Each -> All, was -> were), capture dim_11 (donor - base
post-attention residual at t) and every unit's lambda0-scaled (donor - base) output delta at t for layers 0-11 (MLP 11
excluded; v149 recipe); (2) the SET = the seven units with the largest projection on dim_hat = dim_11 / |dim_11|;
(3) per-block diff-in-means q fitted on parity 0, evaluated block-live on held-out parity 1 (16, All -> Each, were -> was),
A2 parity 1 (16, other frame, cue column 3) and C (32, other frame, same answer); random block direction seed 1.
Also reported: lexical's seven-unit set (v151) patched on quantifier rows, with q fitted on lexical parity 0 -- does the
shared residual axis ride on shared units?
Row population (ops/row_population.py): quantifier A1 p0 16 / p1 16 (single cue pair), A2 p0/p1 16 (cue col 3), C 32.

Registered before the run:
  pred_a_instrument    closure cos(sum scaled delta, dim_11) >= 0.98 and norm ratio in [0.95, 1.05]; |random| <= 0.05 on held-out p1
  pred_b_not_1103      attn:11:03's scaled delta . dim_hat < 0.4 |dim_11|   (quantifier's residual number direction is written mostly by other units)
  pred_c_set_exact     exact interchange of the seven on held-out p1 >= 0.60
  pred_d_dim           dim on held-out p1 >= 0.8 x exact (same rows)
  pred_e_specific      complement on held-out p1 <= 0.20 AND C same-answer effect of dim <= 0.15
Prior: b open (v145 spoke about the CUE column only; 11:03's whole write at t may still be on the axis), c open (a
single-pair set can concentrate in fewer units), d likely, e likely.
Smoke: V152_SMOKE=<out.json> -> CPU, 4 rows.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
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
OUT = ROOT / "circuits/followups/unit_tier5_quantifier_own_set_v152_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
CLOSURE_MIN, NORM_LO, NORM_HI, RAND_MAX, NOT1103_MAX, SET_MIN, DIM_FRAC, COMP_MAX, C_MAX, K = 0.98, 0.95, 1.05, 0.05, 0.4, 0.60, 0.8, 0.20, 0.15, 7
LEX_SEVEN = ("attn:11:head:03", "mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_quantifier_own_set_v152", "behaviours": 2, "targets": 4,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V152_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    layer, h = g.unit_layer(READER), int(READER.rsplit(":", 1)[1])
    cut = (lambda rows: rows[:4]) if smoke else (lambda rows: rows)
    mq = importlib.import_module(f"circuit_fast_screen_candidate_{names['quantifier_number']}")
    ml = importlib.import_module(f"circuit_fast_screen_candidate_{names['lexical_number_pp']}")
    qa1 = g.rows_of(mq, "A1")
    P = {"p0": g.prepare(backend, cut(qa1[0::2])), "p1": g.prepare(backend, cut(qa1[1::2])),
         "A2p1": g.prepare(backend, cut(g.rows_of(mq, "A2")[1::2])), "C": g.prepare(backend, cut(g.rows_of(mq, "C")))}
    LEXp0 = g.prepare(backend, cut(g.rows_of(ml, "A1")[0::2]))

    # (1) accounting on parity 0
    D, B = P["p0"].donor_batch, P["p0"].base_batch
    capD, capB = {}, {}
    g.forward_units(backend, D, capture_resid=capD)
    g.forward_units(backend, B, capture_resid=capB)
    dim = torch.stack([capD[(rid, layer)] - capB[(rid, layer)] for rid in D.row_ids]).float().mean(0)
    dim_hat = dim / dim.norm()
    layers = list(range(0, layer + 1))
    hD, hB = v120.head_cache(backend, D, layers), v120.head_cache(backend, B, layers)
    mD, mB = v120.mlp_cache(backend, D, list(range(0, layer))), v120.mlp_cache(backend, B, list(range(0, layer)))
    lam0 = [float(backend.model.transformer.h[l].lambdas[0]) for l in range(layer + 1)]
    def scale(u):
        l = int(u.split(":")[1]); f = 1.0
        for l2 in range(l + 1, layer + 1):
            f *= lam0[l2]
        return f
    deltas = {}
    with torch.no_grad():
        for l in layers:
            Wl = backend.model.transformer.h[l].attn.c_proj.weight
            for hh in range(g.N_HEADS):
                u = f"attn:{l:02d}:head:{hh:02d}"
                sl = torch.stack([hD[(rid, u)] - hB[(rid, u)] for rid in D.row_ids]).mean(0).to(Wl.device, Wl.dtype)
                deltas[u] = (sl @ Wl[:, hh * g.HEAD_DIM:(hh + 1) * g.HEAD_DIM].T).float().to(dim.device) * scale(u)
            if l < layer:
                u = f"mlp:{l:02d}"
                deltas[u] = torch.stack([mD[(rid, u)] - mB[(rid, u)] for rid in D.row_ids]).mean(0).float().to(dim.device) * scale(u)
    total = sum(deltas.values())
    along = {u: round(float(d @ dim_hat) / float(dim.norm()), 3) for u, d in deltas.items()}
    ranked = sorted(along.items(), key=lambda kv: -kv[1])
    units = [u for u, _ in ranked[:K]]
    # (2)-(3) rubric on the named set
    q_p0 = g.block_diff_in_means(backend, P["p0"], units)
    q_rand = g.block_random_subspace(backend, units, rank=1, seed=1)
    scl = g.target_scale(P["p1"])
    def rec(prep, us, q=None, complement=False):
        return round(g.recovery(prep, g.patched_axis(backend, prep, list(us), q=q, complement=complement)), 3)
    def same(prep, us, q=None, complement=False):
        return round(g.same_answer_effect(prep, g.patched_axis(backend, prep, list(us), q=q, complement=complement), scl), 3)
    lex_q = g.block_diff_in_means(backend, LEXp0, list(LEX_SEVEN))
    R = {"closure_cos": round(float(torch.nn.functional.cosine_similarity(total, dim, dim=0)), 3), "norm_ratio": round(float(total.norm() / dim.norm()), 3),
         "norm_dim": round(float(dim.norm()), 1), "lambda0": [round(x, 3) for x in lam0],
         "along_top12": ranked[:12], "frac_1103": along[READER], "set": units, "set_frac_sum": round(sum(along[u] for u in units), 3),
         "exact_p1": rec(P["p1"], units), "dim_p1": rec(P["p1"], units, q_p0), "comp_p1": rec(P["p1"], units, q_p0, complement=True), "rand_p1": rec(P["p1"], units, q_rand),
         "exact_A2p1": rec(P["A2p1"], units), "dim_A2p1": rec(P["A2p1"], units, q_p0), "comp_A2p1": rec(P["A2p1"], units, q_p0, complement=True),
         "C_same_dim": same(P["C"], units, q_p0), "C_same_exact": same(P["C"], units), "C_same_rand": same(P["C"], units, q_rand),
         "lex_seven_exact_p1": rec(P["p1"], LEX_SEVEN), "lex_seven_lexq_p1": rec(P["p1"], LEX_SEVEN, lex_q), "lex_seven_lexq_comp_p1": rec(P["p1"], LEX_SEVEN, lex_q, complement=True),
         "overlap_with_lex_seven": sorted(set(units) & set(LEX_SEVEN)), "scale_logits": round(scl, 2),
         "rows": {k: len(v.base_batch.row_ids) for k, v in P.items()}}
    R["S_plus_C_p1"] = round(R["dim_p1"] + R["comp_p1"], 3)
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = R["closure_cos"] >= CLOSURE_MIN and NORM_LO <= R["norm_ratio"] <= NORM_HI and abs(R["rand_p1"]) <= RAND_MAX
    pred_b = R["frac_1103"] < NOT1103_MAX
    pred_c = R["exact_p1"] >= SET_MIN
    pred_d = R["dim_p1"] >= DIM_FRAC * R["exact_p1"]
    pred_e = R["comp_p1"] <= COMP_MAX and R["C_same_dim"] <= C_MAX
    predictions = {"pred_a_instrument": pred_a, "pred_b_not_1103": pred_b, "pred_c_set_exact": pred_c, "pred_d_dim": pred_d, "pred_e_specific": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_quantifier_own_set_v152", "candidate_id": "corpus.unit_tier5_quantifier_own_set_v152",
              "bars": {"closure_min": CLOSURE_MIN, "norm_band": [NORM_LO, NORM_HI], "rand_max": RAND_MAX, "not1103_max": NOT1103_MAX, "set_min": SET_MIN, "dim_frac": DIM_FRAC, "comp_max": COMP_MAX, "c_max": C_MAX, "k": K},
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "set": units}, indent=2))


if __name__ == "__main__":
    main()
