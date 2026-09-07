#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: does the 11:03 number vector's residual image lie along the residual-level number direction, and how does the margin respond to its dose?

v143-v145: one 128-d vector w_lex in 11:03's c_proj slice is a noun-invariant, bidirectional number value axis on the
verb margin of lexical_number_pp (0.49 of exact) and quantifier_number (0.38 / 0.45), and same-side pushes saturate.
Two questions the unit-level result leaves open: (1) is its residual image  img = W_O[:, slot] w_lex  the number
direction the residual carries at t (diff-in-means of donor - base post-attention residual at layer 11, and at layer 17,
the readout point), i.e. does the tier-5 vector meet the tier-4 rank-1 directions of v79-v85 in the stream; (2) is the
margin linear in the dose below 1x and saturating above.
Row population (ops/row_population.py): lexical A1 parity 1: 16 rows, 16 cue pairs, was -> were; quantifier A1 parity 0:
16 rows, 1 pair, was -> were (Each -> All). All rows equal length, t aligned.
Residuals via g.forward_units(capture_resid=...) on D and B (post-attention, pre-MLP, at t). dim_L = mean_i (resid_D - resid_B)
at layer L. Dose arms: k * w_lex for k in (0.25, 0.5, 1, 2, 4) on lexical parity-1 base rows.

Registered before the run:
  pred_a_instrument    m_none = 0, |m_rand| <= 0.03; resid_add of img at layer 11 (at t, base batch) reproduces the 1x slot-add
                       within 0.02 (v140 showed image add = slot add); 1x slot-add reproduces v145 (0.492) within 0.02
  pred_b_dim11_lex     cos(img, dim_11 on lexical rows) >= 0.5
  pred_c_dim11_quant   cos(img, dim_11 on quantifier parity-0 rows) >= 0.3   (the residual number direction is shared across the two cue classes)
  pred_d_dose_shape    share(0.5x) / share(1x) in [0.35, 0.65] and share(2x) / share(1x) <= 1.25
  pred_e_dim17_lex     cos(img, dim_17 on lexical rows) >= 0.3   (the image already points along the readout-time number direction)
Prior: b likely (11:03 is ~half the margin); c open (v145 d: quantifier does not WRITE the vector, but its residual may still
carry the axis via other writers); d open -- v145 e showed saturation, the shape is the measurement; e unsure (the direct
share was 0.39, v140).
Smoke: V146_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_number_vector_residual_tie_v146_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
RAND_MAX, INSTR_TOL, V145_1X, COS_B, COS_C, HALF_LO, HALF_HI, TWO_MAX, COS_E, LATE = 0.03, 0.02, 0.492, 0.5, 0.3, 0.35, 0.65, 1.25, 0.3, 17
DOSES = (0.25, 0.5, 1.0, 2.0, 4.0)
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_vector_residual_tie_v146", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V146_SMOKE")
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
        E["dim"] = {L: sum(capD[(rid, L)] - capB[(rid, L)] for rid in D.row_ids) / len(rows) for L in (layer, LATE)}
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
        out = g.forward_units(backend, E["B"], resid_add={layer: image.unsqueeze(0).expand(E["rows"], -1)})
        return share(E, [float(a) - float(f) for a, f in out.tolist()])

    LEX, Q = target("lexical_number_pp", 1, True), target("quantifier_number", 0, False)
    w_lex = LEX["wbar"]
    W = backend.model.transformer.h[layer].attn.c_proj.weight[:, h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM]
    img = (w_lex.to(W.dtype) @ W.T).float()
    cos = lambda a, b: round(float(torch.nn.functional.cosine_similarity(a.float().to(b.device), b.float(), dim=0)), 3)
    k = LEX["rows"]
    gen = torch.Generator().manual_seed(0)
    rand = [torch.randn(w_lex.shape, generator=gen).to(w_lex.device) * (w_lex.norm() / g.HEAD_DIM ** 0.5) for _ in range(k)]
    R = {"m_none": add_margin(LEX, [torch.zeros_like(w_lex)] * k), "m_rand": add_margin(LEX, rand),
         "dose": {str(d): add_margin(LEX, [d * w_lex] * k) for d in DOSES},
         "img_resid_add": resid_margin(LEX, img), "positions_ok": [LEX["positions_ok"], Q["positions_ok"]],
         "cos_img_dim11_lex": cos(img, LEX["dim"][layer]), "cos_img_dim17_lex": cos(img, LEX["dim"][LATE]),
         "cos_img_dim11_quant": cos(img, Q["dim"][layer]), "cos_img_dim17_quant": cos(img, Q["dim"][LATE]),
         "cos_dim11_lex_quant": cos(LEX["dim"][layer], Q["dim"][layer]), "cos_dim17_lex_quant": cos(LEX["dim"][LATE], Q["dim"][LATE]),
         "norm_img": round(float(img.norm()), 2), "norm_dim11_lex": round(float(LEX["dim"][layer].norm()), 2), "norm_dim17_lex": round(float(LEX["dim"][LATE].norm()), 2),
         "img_share_of_dim11": round(float(img @ LEX["dim"][layer].to(img.device) / (LEX["dim"][layer].norm() ** 2)), 3)}
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    one = R["dose"]["1.0"]
    pred_a = R["m_none"] == 0 and abs(R["m_rand"]) <= RAND_MAX and abs(R["img_resid_add"] - one) <= INSTR_TOL and \
        (smoke or abs(one - V145_1X) <= INSTR_TOL) and all(R["positions_ok"])
    pred_b = R["cos_img_dim11_lex"] >= COS_B
    pred_c = R["cos_img_dim11_quant"] >= COS_C
    half, two = R["dose"]["0.5"] / one, R["dose"]["2.0"] / one
    pred_d = HALF_LO <= half <= HALF_HI and two <= TWO_MAX
    pred_e = R["cos_img_dim17_lex"] >= COS_E
    predictions = {"pred_a_instrument": pred_a, "pred_b_dim11_lex": pred_b, "pred_c_dim11_quant": pred_c, "pred_d_dose_shape": pred_d, "pred_e_dim17_lex": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_vector_residual_tie_v146", "candidate_id": "corpus.unit_tier5_number_vector_residual_tie_v146",
              "bars": {"rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "v145_1x": V145_1X, "cos_b": COS_B, "cos_c": COS_C, "half": [HALF_LO, HALF_HI], "two_max": TWO_MAX, "cos_e": COS_E},
              "dose_ratios": {"half": round(half, 3), "two": round(two, 3)}, "measures": R, "rows": [LEX["rows"], Q["rows"]],
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "dose_ratios": result["dose_ratios"]}, indent=2))


if __name__ == "__main__":
    main()
