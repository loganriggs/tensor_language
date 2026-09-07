#!/usr/bin/env python3
# BQGATE: five frozen predictions; reader-set pairs fixed from the v129 receipt; random control seeded; one-cue rows by rule.
"""Tier-5: the distributed remainder of the number direction, tested as a SET of six units.

v149: dim_11 = par (11:03's axis, 0.859 of exact) + rem (0.160); rem is written by no single unit (largest 12% of its
length) -- the six largest are mlp:09 (0.122), attn:09:07 (0.111), mlp:10 (0.111), attn:10:05 (0.073), attn:11:02 (0.070),
mlp:08 (0.058) = 0.545 of |rem| together. Test them as a set, three ways on lexical parity-1 rows (share of exact):
  set_rem     resid_add at layer 11 of the lambda0-scaled sum of the six deltas PROJECTED on rem_hat    (their share of the second axis)
  set_full    resid_add at layer 11 of the six scaled deltas unprojected                                 (their whole write, incl. the par part)
  set_exact   exact block-live interchange of the six units (g.patched_axis / g.recovery)               (what the set does when patched)
  six random-unit sets of the same composition (3 heads at layers 8-11 excluding 11:03, 3 MLPs at layers 0-10) for set_exact
Row population (ops/row_population.py): lexical A1 parity 1: 16 rows, 16 cue pairs, was -> were; aligned.

Registered before the run:
  pred_a_instrument   m_none = 0, |m_rand| <= 0.03, rem add = 0.160 +- 0.02, dim add = 0.936 +- 0.02 (v147/v149)
  pred_b_set_rem      set_rem >= 0.08   (half of rem's 0.16; the six carry 0.545 of |rem| geometrically)
  pred_c_additivity   |set_full - set_exact| <= 0.10   (adding the six's writes at layer 11 = patching the six; the set has no downstream interaction)
  pred_d_set_exact    set_exact >= 0.30 of the donor margin
  pred_e_selective    set_exact exceeds the max of the six random same-composition sets by >= 0.10
Prior: b likely if the residual is linear at this scale; c open (v103: adding a multi-unit set's deltas over-reproduces the
exact patch by 2-20% when downstream patched units get the upstream push AND the delta -- here the six span layers 8-11, so
some interaction is expected; 0.10 is the bar); d open; e likely.
Smoke: V150_SMOKE=<out.json> -> CPU, 4 rows.
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
OUT = ROOT / "circuits/followups/unit_tier5_number_remainder_set_v150_result.json"
V129_RECEIPT = ROOT / "circuits/followups/unit_tier5_head_read_map_v129_result.json"
RAND_MAX, INSTR_TOL, V147_REM, V147_DIM, SET_REM_MIN, ADD_TOL, SET_EXACT_MIN, SELECT_MARGIN = 0.03, 0.02, 0.160, 0.936, 0.08, 0.10, 0.30, 0.10
SIX = ("mlp:09", "attn:09:head:07", "mlp:10", "attn:10:head:05", "attn:11:head:02", "mlp:08")
N_RANDOM_SETS = 6
READER = "attn:11:head:03"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 400, 8000


def _plan():
    return {"candidate_id": "corpus.unit_tier5_number_remainder_set_v150", "behaviours": 1, "targets": 1,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    smoke = os.environ.get("V150_SMOKE")
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
        E = {"prep": prep, "B": B, "D": D, "d_ax": prep.donor_axis, "b_ax": prep.base_axis, "tpos": [ge["t"] for ge in geo], "rows": len(rows),
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
    six_sum = sum(deltas[u] for u in SIX)
    six_rem_vec = (six_sum @ rem_hat) * rem_hat
    prep = LEX["prep"]
    def exact(units):
        return round(g.recovery(prep, g.patched_axis(backend, prep, list(units))), 3)
    rng = random.Random(0)
    head_pool = [f"attn:{l:02d}:head:{hh:02d}" for l in range(8, layer + 1) for hh in range(g.N_HEADS) if not (l == layer and hh == h)]
    mlp_pool = [f"mlp:{l:02d}" for l in range(0, layer)]
    random_sets = [tuple(rng.sample(head_pool, 3) + rng.sample(mlp_pool, 3)) for _ in range(N_RANDOM_SETS)]
    R = {"m_none": add_margin(LEX, [torch.zeros_like(w_lex)] * k), "m_rand": add_margin(LEX, rand),
         "rem_add": resid_margin(LEX, rem), "dim_add": resid_margin(LEX, dim),
         "set_rem": resid_margin(LEX, six_rem_vec), "set_full": resid_margin(LEX, six_sum), "set_exact": exact(SIX),
         "six_rem_frac": round(float(six_sum @ rem_hat) / float(rem.norm()), 3), "six_par_frac": round(float(six_sum @ img_hat) / float(par.norm()), 3),
         "per_unit_exact": {u: exact([u]) for u in SIX},
         "random_sets": [{"units": list(us), "exact": exact(us)} for us in random_sets],
         "closure_cos": cos(total, dim), "reader_exact": exact([READER]), "reader_plus_six_exact": exact([READER] + list(SIX)),
         "positions_ok": LEX["positions_ok"], "rows": k}
    R["random_max"] = max(r_["exact"] for r_ in R["random_sets"])
    print(R, round(time.perf_counter() - t0), "s", flush=True)
    pred_a = R["m_none"] == 0 and abs(R["m_rand"]) <= RAND_MAX and R["positions_ok"] and \
        (smoke or (abs(R["rem_add"] - V147_REM) <= INSTR_TOL and abs(R["dim_add"] - V147_DIM) <= INSTR_TOL))
    pred_b = R["set_rem"] >= SET_REM_MIN
    pred_c = abs(R["set_full"] - R["set_exact"]) <= ADD_TOL
    pred_d = R["set_exact"] >= SET_EXACT_MIN
    pred_e = R["set_exact"] - R["random_max"] >= SELECT_MARGIN
    predictions = {"pred_a_instrument": pred_a, "pred_b_set_rem": pred_b, "pred_c_additivity": pred_c, "pred_d_set_exact": pred_d, "pred_e_selective": pred_e}
    result = {"predictions": predictions, "schema": "unit_tier5_number_remainder_set_v150", "candidate_id": "corpus.unit_tier5_number_remainder_set_v150",
              "bars": {"rand_max": RAND_MAX, "instr_tol": INSTR_TOL, "v147_rem": V147_REM, "v147_dim": V147_DIM, "set_rem_min": SET_REM_MIN, "add_tol": ADD_TOL, "set_exact_min": SET_EXACT_MIN, "select_margin": SELECT_MARGIN},
              "six": list(SIX), "measures": R, "seconds": round(time.perf_counter() - t0, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    out = Path(smoke) if smoke else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions}, indent=2))


if __name__ == "__main__":
    main()
