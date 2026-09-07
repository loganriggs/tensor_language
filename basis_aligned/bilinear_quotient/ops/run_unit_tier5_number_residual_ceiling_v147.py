#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the 11:03 number axis alone reaches 0.84 of the exact margin (v146, plateau at 4x). Where is the missing sixth?

v146: img = W_O[:, slot] w_lex has cos 0.875 with dim_11 (diff-in-means of donor - base post-attention residual at t, layer 11)
and the dose curve is linear to 2x then plateaus at 0.84. dim_11 decomposes as  par = (dim_11 . img_hat) img_hat  (2.3x img)
plus a remainder  rem = dim_11 - par  (cos 0 with img, ~0.48 of dim_11's length). The residual at t after layer 11 is the
one channel through which everything the cue caused at layers <= 11 reaches the verb, UNLESS later heads read the cue
position directly. Arms (all resid_add at layer 11, at t, on lexical parity-1 BASE rows, share of exact):
  rowwise    per-row (resid_D_i - resid_B_i)         the exact residual patch at t
  dim        dim_11 (one mean vector on all rows)     rank-1 version of rowwise
  par, rem   the two components of dim_11
  img        1x W_O w_lex (v146 0.492, instrument)
  quant_dim  dim_11 of LEXICAL rows added to quantifier parity-0 base rows
Row population (ops/row_population.py): lexical A1 parity 1: 16 rows, 16 cue pairs, was -> were; quantifier A1 parity 0:
16 rows, 1 pair, was -> were. All rows equal length, t aligned.

Registered before the run:
  pred_a_instrument   m_none = 0, |m_rand| <= 0.03, img add = 0.492 +- 0.02 (v146)
  pred_b_channel      rowwise >= 0.85   (the residual at t after layer 11 carries the margin; later layers do not read the cue directly)
  pred_c_rank1        |dim - rowwise| <= 0.10   (the residual displacement is one direction across the 16 nouns)
  pred_d_second_axis  rem >= 0.15   (the remainder carries at least the missing sixth: a second axis, not a nonlinearity of the first)
  pred_e_quant_xfer   quant_dim >= 0.4   (lexical's residual number direction drives quantifier verb agreement, as the unit vector did at 0.38-0.45 in v145)
Reported, unregistered: par, par + rem vs dim (additivity), dose 2x/4x of par.
Prior: b open (v142: verb cue columns of other sets read the embedding route, so a direct late read is possible here too);
c likely (head deltas were 94-99% rank-1); d open; e likely.
Smoke: V147_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_number_residual_ceiling_v147_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
RAND_MAX, INSTR_TOL, V146_IMG, ROWWISE_MIN, RANK1_TOL, REM_MIN, QUANT_MIN = 0.03, 0.02, 0.492, 0.85, 0.10, 0.15, 0.4
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_residual_ceiling_v147", "behaviours": 2, "targets": 2,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V147_SMOKE")
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

    LEX, Q = target("lexical_number_pp", 1, True), target("quantifier_number", 0, False)
    w_lex = LEX["wbar"]
    W = backend.model.transformer.h[layer].attn.c_proj.weight[:, h * g.HEAD_DIM:(h + 1) * g.HEAD_DIM]
    img = (w_lex.to(W.dtype) @ W.T).float()
    cos = lambda a, b: round(float(torch.nn.functional.cosine_similarity(a.float().to(b.device), b.float(), dim=0)), 3)
    k = LEX["rows"]
    gen = torch.Generator().manual_seed(0)
    rand = [torch.randn(w_lex.shape, generator=gen).to(w_lex.device) * (w_lex.norm() / g.HEAD_DIM ** 0.5) for _ in range(k)]
    dim = LEX["dim"].to(img.device)
    img_hat = img / img.norm()
    par = (dim @ img_hat) * img_hat
    rem = dim - par
    R = {"m_none": add_margin(LEX, [torch.zeros_like(w_lex)] * k), "m_rand": add_margin(LEX, rand),
         "img": resid_margin(LEX, img), "rowwise": resid_margin(LEX, LEX["delta_rows"]), "dim": resid_margin(LEX, dim),
         "par": resid_margin(LEX, par), "rem": resid_margin(LEX, rem), "par_2x": resid_margin(LEX, 2 * par), "par_4x": resid_margin(LEX, 4 * par),
         "quant_dim": resid_margin(Q, dim), "quant_rowwise_own": resid_margin(Q, Q["delta_rows"]), "quant_own_dim": resid_margin(Q, Q["dim"]),
         "positions_ok": [LEX["positions_ok"], Q["positions_ok"]],
         "cos_img_dim": cos(img, dim), "cos_rem_img": cos(rem, img), "par_over_img": round(float((dim @ img_hat) / img.norm()), 3),
         "norm_dim": round(float(dim.norm()), 1), "norm_par": round(float(par.norm()), 1), "norm_rem": round(float(rem.norm()), 1),
         "rowwise_cos_to_dim": [round(float(torch.nn.functional.cosine_similarity(d, dim, dim=0)), 3) for d in LEX["delta_rows"]]}
    R["additivity_gap"] = round(R["par"] + R["rem"] - R["dim"], 3)
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = R["m_none"] == 0 and abs(R["m_rand"]) <= RAND_MAX and (smoke or abs(R["img"] - V146_IMG) <= INSTR_TOL) and all(R["positions_ok"])
    pred_b = R["rowwise"] >= ROWWISE_MIN
    pred_c = abs(R["dim"] - R["rowwise"]) <= RANK1_TOL
    pred_d = R["rem"] >= REM_MIN
    pred_e = R["quant_dim"] >= QUANT_MIN
    predictions = {"pred_a_instrument": pred_a, "pred_b_channel": pred_b, "pred_c_rank1": pred_c, "pred_d_second_axis": pred_d, "pred_e_quant_xfer": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_residual_ceiling_v147", "candidate_id": "corpus.unit_tier5_number_residual_ceiling_v147",
              "bars": {"rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "v146_img": V146_IMG, "rowwise_min": ROWWISE_MIN, "rank1_tol": RANK1_TOL, "rem_min": REM_MIN, "quant_min": QUANT_MIN},
              "measures": R, "rows": [LEX["rows"], Q["rows"]],
              "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
