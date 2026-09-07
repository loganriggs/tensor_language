#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5 -> rubric: the layer-11 residual number direction dim_11 on held-out frames, the reverse interchange, P and C.

v147: dim_11 (diff-in-means of donor - base post-attention residual at t, layer 11, lexical A1 parity 1, 16 nouns) added
at t on base rows gives 0.936 of the exact margin. Lift it through the standing rubric (A1/A2/P/C, held-out, both
directions), signs by MEANING (+dim_11 = push toward plural):
  A2 parity 1  held-out frame "In the report the <noun> beside the <place>" (cue column 4, 16 rows, was -> were): +dim_11 on base
  A1 parity 0  reverse interchange (leaders -> leader, were -> was, 15 aligned rows): -dim_11 on base
  P parity 1   noun rewrite keeping number (director -> producer, was -> was, 16 rows): +dim_11 on base; P KEEPS the number slot,
               so a value axis is expected to move it (memory: slot-keeping siblings are not spared) -- registered as a positive
  C both       different frame, answer night -> night (32 rows, different lengths): +dim_11 at base t; off-slot, expected inert
Delta arms (P, C) are raw margin changes on the row's own answer axis, sign-aligned by the A1 parity-1 push, in units of the
A1 parity-1 exact margin  |d - b|  (mean 30.7 logits at v147's rows).
Row population (ops/row_population.py): A1 p0 16 rows (1 misaligned, dropped), A1 p1 16, A2 p1 16 (cue col 4), P p1 16
(answers was -> was), C 16 + 16 (all misaligned lengths; no cue check, positions from the rows).

Registered before the run:
  pred_a_instrument  m_none = 0, |m_rand| <= 0.03, +dim_11 on A1 parity-1 base = 0.936 +- 0.02 (v147)
  pred_b_heldout_A2  share on A2 parity-1 in [0.75, 1.25]   (upper bound: the A2 frame has its own, smaller exact margin; overshoot means the A1-sized vector is too large there)
  pred_c_reverse     share of -dim_11 on A1 parity-0 in [0.75, 1.25]
  pred_d_P_moves     sign-aligned mean delta on P parity-1 / exact_A1 >= 0.5   (the direction is a number value, not noun identity)
  pred_e_C_inert     mean |delta| on C / exact_A1 <= 0.15
Prior: b likely (A2 differs only in frame), c likely (v96: rank-1 directions recover both interchange directions), d likely
(v101 pattern), e open -- the direction has norm 3431 in a stream where rms_norm rescales; a large off-axis add may still
perturb night vs its foil.
Smoke: V148_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_number_residual_rubric_v148_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
RAND_MAX, INSTR_TOL, V147_DIM, A2_MIN, REV_MIN, P_MIN, C_MAX, UPPER = 0.03, 0.02, 0.936, 0.75, 0.75, 0.5, 0.15, 1.25
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_residual_rubric_v148", "behaviours": 1, "targets": 5,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V148_SMOKE")
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
        E = {"B": B, "D": D, "d_ax": prep.donor_axis, "b_ax": prep.base_axis, "tpos": [ge["t"] for ge in geo], "rows": len(rows),
             "positions_ok": list(B.semantic_positions) == [ge["t"] for ge in geo]}
        capD, capB = {}, {}
        g.forward_units(backend, D, capture_resid=capD)
        g.forward_units(backend, B, capture_resid=capB)
        E["delta_rows"] = torch.stack([capD[(rid, layer)] - capB[(rid, layer)] for rid in D.row_ids]).float()
        E["dim"] = E["delta_rows"].mean(0)
        if vectors:
            cuepos = tuple(ge["cue"] for ge in geo)
            Bc, Dc = dataclasses.replace(B, semantic_positions=cuepos), dataclasses.replace(D, semantic_positions=cuepos)
            hd, md = v120.head_cache(backend, Dc, below_all), v120.mlp_cache(backend, Dc, below_all)
            PB, VB = v131.capture_with_clamp(backend, B, [], [], layer)
            hi = [(u, 0, hd) for u in units if u.startswith("attn")]
            mi = [(u, 0, md) for u in units if u.startswith("mlp")]
            P, V = v131.capture_with_clamp(backend, Bc, hi, mi, layer)
            w = [P[i, h, ge["t"], ge["cue"]] * V[i, ge["cue"], h, :] - PB[i, h, ge["t"], ge["cue"]] * VB[i, ge["cue"], h, :] for i, ge in enumerate(geo)]
            E["wbar"] = sum(w) / len(w)
        return E

    def share(E, vals):
        per = [(-p - bb) / (dd - bb) for dd, bb, p in zip(E["d_ax"], E["b_ax"], vals) if abs(dd - bb) > 1e-6]
        return round(sum(per) / len(per), 3)

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
        return share(E, [float(a) - float(f) for a, f in out.tolist()])

    def resid_margin(E, image):
        if image.dim() == 1:
            image = image.unsqueeze(0).expand(E["rows"], -1)
        out = g.forward_units(backend, E["B"], resid_add={layer: image})
        return share(E, [float(a) - float(f) for a, f in out.tolist()])

    def plain(n, fam, parity):
        m = importlib.import_module(f"circuit_fast_screen_candidate_{names[n]}")
        rows = m.build_rows()
        key = "family" if "family" in rows[0] else "transform_id"
        rows = [r for r in rows if r[key] == fam]
        if parity is not None:
            rows = rows[parity::2]
        if smoke:
            rows = rows[:4]
        prep = g.prepare(backend, rows)
        return {"B": prep.base_batch, "D": prep.donor_batch, "d_ax": prep.donor_axis, "b_ax": prep.base_axis, "rows": len(rows),
                "tpos": list(prep.base_batch.semantic_positions)}

    def raw(E, image=None):
        kw = {}
        if image is not None:
            if image.dim() == 1:
                image = image.unsqueeze(0).expand(E["rows"], -1)
            kw["resid_add"] = {layer: image}
        out = g.forward_units(backend, E["B"], **kw)
        return [float(a) - float(f) for a, f in out.tolist()]

    LEX = target("lexical_number_pp", 1, True)
    A2, REV, P, C = plain("lexical_number_pp", "A2", 1), target("lexical_number_pp", 0, False), plain("lexical_number_pp", "P", 1), plain("lexical_number_pp", "C", None)
    w_lex = LEX["wbar"]
    k = LEX["rows"]
    gen = torch.Generator().manual_seed(0)
    rand = [torch.randn(w_lex.shape, generator=gen).to(w_lex.device) * (w_lex.norm() / g.HEAD_DIM ** 0.5) for _ in range(k)]
    dim = LEX["dim"]
    exact = sum(abs(dd - bb) for dd, bb in zip(LEX["d_ax"], LEX["b_ax"])) / k
    r0, r1 = raw(LEX), raw(LEX, dim)
    sign = 1.0 if sum(b_ - a_ for a_, b_ in zip(r0, r1)) > 0 else -1.0
    def delta(E, image):
        base, add = raw(E), raw(E, image)
        d = [(y - x) * sign / exact for x, y in zip(base, add)]
        return round(sum(d) / len(d), 3), round(sum(abs(v) for v in d) / len(d), 3)
    dP, dC = delta(P, dim), delta(C, dim)
    R = {"m_none": add_margin(LEX, [torch.zeros_like(w_lex)] * k), "m_rand": add_margin(LEX, rand),
         "a1_dim": resid_margin(LEX, dim), "a2_dim": resid_margin(A2, dim), "rev_minus_dim": resid_margin(REV, -dim),
         "rev_plus_dim_wrong_sign": resid_margin(REV, dim), "a2_rowwise_own": resid_margin(A2, A2["delta_rows"]) if "delta_rows" in A2 else None,
         "p_delta_signed": dP[0], "p_delta_abs": dP[1], "c_delta_signed": dC[0], "c_delta_abs": dC[1],
         "exact_a1_logits": round(exact, 2), "sign_convention": sign, "rows": {"A1": k, "A2": A2["rows"], "REV": REV["rows"], "P": P["rows"], "C": C["rows"]},
         "positions_ok": LEX["positions_ok"], "norm_dim": round(float(dim.norm()), 1)}
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = R["m_none"] == 0 and abs(R["m_rand"]) <= RAND_MAX and (smoke or abs(R["a1_dim"] - V147_DIM) <= INSTR_TOL) and R["positions_ok"]
    pred_b = A2_MIN <= R["a2_dim"] <= UPPER
    pred_c = REV_MIN <= R["rev_minus_dim"] <= UPPER
    pred_d = R["p_delta_signed"] >= P_MIN
    pred_e = R["c_delta_abs"] <= C_MAX
    predictions = {"pred_a_instrument": pred_a, "pred_b_heldout_A2": pred_b, "pred_c_reverse": pred_c, "pred_d_P_moves": pred_d, "pred_e_C_inert": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_residual_rubric_v148", "candidate_id": "corpus.unit_tier5_number_residual_rubric_v148",
              "bars": {"rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "v147_dim": V147_DIM, "a2_min": A2_MIN, "rev_min": REV_MIN, "p_min": P_MIN, "c_max": C_MAX, "upper": UPPER},
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
