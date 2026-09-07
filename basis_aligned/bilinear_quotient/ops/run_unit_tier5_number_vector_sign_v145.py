#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the 11:03 number vector, signed by MEANING -- plural push on both interchange directions and across the cue class.

v144: lexical's 11:03 vector (plural push, from base singular -> donor plural rows) added to quantifier's base rows
("All of the ... directors", plural verb) gave donor share -0.189: on quantifier parity 1 the donor is "Each" (singular),
so a plural push is NEGATIVE there -- the sign a shared number axis predicts, but my prediction had assumed a common
base->donor orientation. And quantifier's own vector (m_w 0.031) was tested against a 0.6-of-target bar it could not
reach by size. This rung registers by meaning and magnitude-matches the source.
Orientation read off the rows: lexical parity 1 base singular -> donor plural (was -> were); lexical parity 0 the reverse;
quantifier parity 0 Each -> All (donor plural, was -> were); quantifier parity 1 All -> Each (donor singular).
Vectors (v144 pass): w_lex = mean 11:03 writes-route vector over lexical parity-1 rows (plural push);
w_q = the same over quantifier parity-1 rows (singular push); w_q_scaled = w_q * |w_lex| / |w_q|. All adds at t on the
target's BASE batch into 11:03's c_proj slice; share = (-raw - b) / (d - b) on the target's donor axis.

Registered before the run:
  pred_a_instrument   m_none = 0 and |m_rand| <= 0.03 on every target; lexical parity-1 LOO w_bar reproduces v143 loo_w
                      (0.493) within 0.02; quantifier parity-1 + w_lex reproduces v144 (-0.189) within 0.02
  pred_b_quant_both_directions  quantifier parity 0 (donor plural) + w_lex: share >= +0.15; parity 1 (donor singular): share <= -0.15
  pred_c_lex_reverse_direction  lexical parity 0 (donor singular) + (-w_lex): share >= +0.25   (the vector is a value axis, not a one-way push)
  pred_d_quant_source_matched   cos(w_q, w_lex) <= -0.5 and lexical parity 1 + w_q_scaled: share <= -0.3 (a singular push of matched size)
  pred_e_lex_wrong_sign_control lexical parity 0 + (+w_lex): share <= -0.25 (pushing plural on already-plural base rows moves AWAY from the singular donor)
Prior: b is the open one -- v95 found quantifier 8x one-sided (was 1.30 / were 0.16), so the plural (parity-0) push may be
weak; c likely (v144 removal was 0.60-0.88 of own share); d is the axis-identity test; e is the sign control.
Smoke: V145_SMOKE=<out.json> -> CPU, 4 rows per (set, parity).
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
OUT = ROOT / "circuits/followups/unit_tier5_number_vector_sign_v145_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
RAND_MAX, INSTR_TOL, V143_LOO, V144_Q, Q_MIN, LEX_REV_MIN, COS_MAX, QSRC_MAX, WRONG_MAX = 0.03, 0.02, 0.493, -0.189, 0.15, 0.25, -0.5, -0.3, -0.25
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_vector_sign_v145", "behaviours": 2, "targets": 4,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V145_SMOKE")
    backend = producer.Bilin18TorchBackend.load("cpu" if smoke else "cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    names = {**v112.BATCH, **v112.INSTRUMENT}
    layer, h = g.unit_layer(READER), int(READER.rsplit(":", 1)[1])
    units = v132.units_of(range(0, layer))
    below_all = list(range(0, layer))

    def target(n, parity, vectors):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows, geo = [], []
        for r in v123.rows_of(m, parity, smoke):
            t = r["donor_semantic_position"]
            diff = [p for p in range(t) if r["base_ids"][p] != r["donor_ids"][p]]
            if len(diff) != 1:
                continue
            rows.append(r); geo.append({"t": t, "cue": diff[0]})
        prep = g.prepare(backend, rows)
        D, B = prep.donor_batch, prep.base_batch
        E = {"B": B, "D": D, "d_ax": prep.donor_axis, "b_ax": prep.base_axis, "tpos": [ge["t"] for ge in geo], "layer": layer, "h": h,
             "rows": len(rows), "donor_answer": rows[0]["donor_answer"], "base_answer": rows[0]["base_answer"]}
        if vectors:
            cuepos = tuple(ge["cue"] for ge in geo)
            Bc, Dc = dataclasses.replace(B, semantic_positions=cuepos), dataclasses.replace(D, semantic_positions=cuepos)
            hd, md = v120.head_cache(backend, Dc, below_all), v120.mlp_cache(backend, Dc, below_all)
            PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)
            hi = [(u, 0, hd) for u in units if u.startswith("attn")]
            mi = [(u, 0, md) for u in units if u.startswith("mlp")]
            P, V = v131.capture_with_clamp(backend, Bc, hi, mi, layer)
            w = [P[i, h, ge["t"], ge["cue"]] * V[i, ge["cue"], h, :] - PB[i, h, ge["t"], ge["cue"]] * VB[i, ge["cue"], h, :] for i, ge in enumerate(geo)]
            E["w"] = w; E["wbar"] = sum(w) / len(w); E["loo"] = [(sum(w) - v) / (len(w) - 1) for v in w]
        return E

    def add_margin(E, vecs):
        def pre(_m, args):
            v = args[0].clone()
            for i, vec in enumerate(vecs):
                v[i, E["tpos"][i], h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM] += vec.to(v.device, v.dtype)
            return (v,) + tuple(args[1:])
        hdl = backend.model.transformer.h[layer].attn.c_proj.register_forward_pre_hook(pre)
        try:
            out = g.forward_units(backend, E["B"])
        finally:
            hdl.remove()
        vals = [float(a) - float(f) for a, f in out.tolist()]
        per = [(-p - bb) / (dd - bb) for dd, bb, p in zip(E["d_ax"], E["b_ax"], vals) if abs(dd - bb) > 1e-6]
        return round(sum(per) / len(per), 3)

    T = {("lexical_number_pp", 1): target("lexical_number_pp", 1, True), ("quantifier_number", 1): target("quantifier_number", 1, True),
         ("lexical_number_pp", 0): target("lexical_number_pp", 0, False), ("quantifier_number", 0): target("quantifier_number", 0, False)}
    w_lex, w_q = T[("lexical_number_pp", 1)]["wbar"], T[("quantifier_number", 1)]["wbar"]
    cos_q_lex = round(float(torch.nn.functional.cosine_similarity(w_q, w_lex, dim=0)), 3)
    w_q_scaled = w_q * (w_lex.norm() / (w_q.norm() + 1e-8))
    print("vectors", round(time.perf_counter() - t0), "s", "cos", cos_q_lex, "norms", round(float(w_lex.norm()), 2), round(float(w_q.norm()), 2), flush=True)

    R = {}
    for (n, par), E in T.items():
        k = E["rows"]
        gen = torch.Generator().manual_seed(0)
        rand = [torch.randn(w_lex.shape, generator=gen).to(w_lex.device) * (w_lex.norm() / g.HEAD_DIM ** 0.5) for _ in range(k)]
        A = {"rows": k, "base_answer": E["base_answer"], "donor_answer": E["donor_answer"],
             "m_none": add_margin(E, [torch.zeros_like(w_lex)] * k), "m_rand": add_margin(E, rand),
             "plus_w_lex": add_margin(E, [w_lex] * k), "minus_w_lex": add_margin(E, [-w_lex] * k),
             "plus_w_q_scaled": add_margin(E, [w_q_scaled] * k)}
        if "loo" in E:
            A["loo_own"] = add_margin(E, E["loo"])
        R[f"{n}|p{par}"] = A
        print(n, par, A, round(time.perf_counter() - t0), "s", flush=True)

    L1, L0, Q1, Q0 = R["lexical_number_pp|p1"], R["lexical_number_pp|p0"], R["quantifier_number|p1"], R["quantifier_number|p0"]
    pred_a = all(A["m_none"] == 0 and abs(A["m_rand"]) <= RAND_MAX for A in R.values()) and \
        (smoke or (abs(L1["loo_own"] - V143_LOO) <= INSTR_TOL and abs(Q1["plus_w_lex"] - V144_Q) <= INSTR_TOL))
    pred_b = Q0["plus_w_lex"] >= Q_MIN and Q1["plus_w_lex"] <= -Q_MIN
    pred_c = L0["minus_w_lex"] >= LEX_REV_MIN
    pred_d = cos_q_lex <= COS_MAX and L1["plus_w_q_scaled"] <= QSRC_MAX
    pred_e = L0["plus_w_lex"] <= WRONG_MAX
    predictions = {"pred_a_instrument": pred_a, "pred_b_quant_both_directions": pred_b, "pred_c_lex_reverse_direction": pred_c,
                   "pred_d_quant_source_matched": pred_d, "pred_e_lex_wrong_sign_control": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_vector_sign_v145", "candidate_id": "corpus.unit_tier5_number_vector_sign_v145",
              "bars": {"rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "v143_loo": V143_LOO, "v144_q": V144_Q, "q_min": Q_MIN, "lex_rev_min": LEX_REV_MIN,
                       "cos_max": COS_MAX, "qsrc_max": QSRC_MAX, "wrong_max": WRONG_MAX},
              "cos_q_lex": cos_q_lex, "norm_w_lex": round(float(w_lex.norm()), 3), "norm_w_q": round(float(w_q.norm()), 3),
              "targets": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "cos": cos_q_lex, "targets": R}, indent=2))


if __name__ == "__main__":
    main()
