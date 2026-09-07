#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: which units write the orthogonal remainder of the layer-11 residual number direction?

v147: dim_11 = par (2.31x the 11:03 image, 0.859 of exact) + rem (norm 1663, cos 0 with the image, 0.160 of exact).
Account for dim_11 by unit: for every unit at layers 0-11 (heads: c_proj-input slice at t mapped through its W_O slot;
MLPs 0-10: Down output at t; mlp:11 acts after the capture point and is excluded), delta_u = mean over lexical parity-1 rows
of (donor - base) output at t. The stream mixes  live = l0 x + l1 x0  at every layer, so a unit at layer l reaches the layer-11 capture scaled by
prod_{l' = l+1..11} l0[l'] (x0 is identical at t on both sides; the MLP responses to changed inputs are already inside the
measured deltas). With that scaling the accounting should close exactly -- registered as such; the unscaled sum is reported. rem_hat = rem / |rem|;  along(u) = delta_u . rem_hat.
Row population (ops/row_population.py): lexical A1 parity 1: 16 rows, 16 cue pairs, was -> were; aligned.

Registered before the run:
  pred_a_instrument   m_none = 0, |m_rand| <= 0.03, rem add = 0.160 +- 0.02 and dim add = 0.936 +- 0.02 (v147)
  pred_b_closure      cos(sum_u scaled delta_u, dim_11) >= 0.98 and |sum| / |dim_11| in [0.95, 1.05]
  pred_c_own_axis     delta_{11:03} . img_hat >= 0.6 |par|   (the parallel component is mostly 11:03's own write)
  pred_d_mlp_rem      sum over mlp:08-10 of along(u) >= 0.5 |rem|   (the second axis is the mlp8-11 increment chain of the core mechanism)
  pred_e_mlp_causal   resid_add at layer 11 of  (sum_{mlp:08-10} delta_u) projected on rem_hat  gives share >= 0.08 (half of rem's 0.16)
Reported, unregistered: along(u) for every unit ranked, the top non-11:03 head, layers 0-7 total.
Prior: b open (lambda mixing), c likely, d open -- the core-mechanism memory says mlp8-11 increment the count, but that was
for list behaviours; e follows d if the residual response is linear at this scale (v147: par+rem-dim = +0.08).
Smoke: V149_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_number_remainder_units_v149_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
RAND_MAX, INSTR_TOL, V147_REM, V147_DIM, CLOSURE_MIN, OWN_MIN, MLP_REM_MIN, MLP_CAUSAL_MIN, NORM_LO, NORM_HI = 0.03, 0.02, 0.160, 0.936, 0.98, 0.6, 0.5, 0.08, 0.95, 1.05
MLP_UNITS = ("mlp:08", "mlp:09", "mlp:10")
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_remainder_units_v149", "behaviours": 1, "targets": 1,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V149_SMOKE")
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

    LEX = target("lexical_number_pp", 1, True)
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
    rem_hat = rem / rem.norm()
    # per-unit (donor - base) output deltas at t, in residual space
    layers = list(range(0, layer + 1))
    hD, hB = v120.head_cache(backend, LEX["D"], layers), v120.head_cache(backend, LEX["B"], layers)
    mD, mB = v120.mlp_cache(backend, LEX["D"], list(range(0, layer))), v120.mlp_cache(backend, LEX["B"], list(range(0, layer)))
    deltas = {}
    with torch.no_grad():
        for l in layers:
            Wl = backend.model.transformer.h[l].attn.c_proj.weight
            for hh in range(g.N_HEADS):
                u = f"attn:{l:02d}:head:{hh:02d}"
                sl = torch.stack([hD[(rid, u)] - hB[(rid, u)] for rid in LEX["D"].row_ids]).mean(0).to(Wl.device, Wl.dtype)
                deltas[u] = (sl @ Wl[:, hh * g.HEAD_DIM:(hh + 1) * g.HEAD_DIM].T).float().to(img.device)
            if l < layer:
                u = f"mlp:{l:02d}"
                deltas[u] = torch.stack([mD[(rid, u)] - mB[(rid, u)] for rid in LEX["D"].row_ids]).mean(0).float().to(img.device)
    raw_total = sum(deltas.values())
    lam0 = [float(backend.model.transformer.h[l].lambdas[0]) for l in range(layer + 1)]
    def scale(u):
        l = int(u.split(":")[1]); f = 1.0
        for l2 in range(l + 1, layer + 1):
            f *= lam0[l2]
        return f
    deltas = {u: d * scale(u) for u, d in deltas.items()}
    total = sum(deltas.values())
    along = {u: round(float(d @ rem_hat), 1) for u, d in deltas.items()}
    par_along = {u: round(float(d @ img_hat), 1) for u, d in deltas.items()}
    mlp_sum = sum(deltas[u] for u in MLP_UNITS)
    mlp_rem_vec = (mlp_sum @ rem_hat) * rem_hat
    ranked = sorted(along.items(), key=lambda kv: -abs(kv[1]))[:12]
    R = {"m_none": add_margin(LEX, [torch.zeros_like(w_lex)] * k), "m_rand": add_margin(LEX, rand),
         "rem_add": resid_margin(LEX, rem), "dim_add": resid_margin(LEX, dim), "mlp_rem_add": resid_margin(LEX, mlp_rem_vec),
         "mlp_sum_add": resid_margin(LEX, mlp_sum), "total_add": resid_margin(LEX, total),
         "cos_total_dim": cos(total, dim), "norm_total": round(float(total.norm()), 1), "norm_ratio": round(float(total.norm() / dim.norm()), 3),
         "cos_raw_total_dim": cos(raw_total, dim), "norm_raw_total": round(float(raw_total.norm()), 1), "lambda0": [round(x, 3) for x in lam0], "norm_dim": round(float(dim.norm()), 1),
         "norm_par": round(float(par.norm()), 1), "norm_rem": round(float(rem.norm()), 1),
         "own_par_frac": round(float(deltas[READER] @ img_hat) / float(par.norm()), 3),
         "mlp_rem_frac": round(float(mlp_sum @ rem_hat) / float(rem.norm()), 3),
         "along_rem_ranked": ranked, "along_par_ranked": sorted(par_along.items(), key=lambda kv: -abs(kv[1]))[:8],
         "layers_0_7_rem_frac": round(sum(float(deltas[u] @ rem_hat) for u in deltas if int(u.split(":")[1]) < 8) / float(rem.norm()), 3),
         "heads_8_11_rem_frac": round(sum(float(deltas[u] @ rem_hat) for u in deltas if u.startswith("attn") and int(u.split(":")[1]) >= 8 and u != READER) / float(rem.norm()), 3),
         "reader_rem_frac": round(float(deltas[READER] @ rem_hat) / float(rem.norm()), 3),
         "positions_ok": LEX["positions_ok"], "rows": k}
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = R["m_none"] == 0 and abs(R["m_rand"]) <= RAND_MAX and R["positions_ok"] and \
        (smoke or (abs(R["rem_add"] - V147_REM) <= INSTR_TOL and abs(R["dim_add"] - V147_DIM) <= INSTR_TOL))
    pred_b = R["cos_total_dim"] >= CLOSURE_MIN and NORM_LO <= R["norm_ratio"] <= NORM_HI
    pred_c = R["own_par_frac"] >= OWN_MIN
    pred_d = R["mlp_rem_frac"] >= MLP_REM_MIN
    pred_e = R["mlp_rem_add"] >= MLP_CAUSAL_MIN
    predictions = {"pred_a_instrument": pred_a, "pred_b_closure": pred_b, "pred_c_own_axis": pred_c, "pred_d_mlp_rem": pred_d, "pred_e_mlp_causal": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_remainder_units_v149", "candidate_id": "corpus.unit_tier5_number_remainder_units_v149",
              "bars": {"rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "v147_rem": V147_REM, "v147_dim": V147_DIM, "closure_min": CLOSURE_MIN, "norm_band": [NORM_LO, NORM_HI], "own_min": OWN_MIN, "mlp_rem_min": MLP_REM_MIN, "mlp_causal_min": MLP_CAUSAL_MIN},
              "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
