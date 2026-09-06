#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grids, seeds and the four controls fixed before the run.
"""v40: is the write-gated readout a bilinear product of THIS write with THIS converted vector, or a generic gain?

v39: on polarity the margin response to the fixed vector v = cross_first(1) added at mlp:08 is a + b*alpha in the
residual write alpha*Delta (a 0.020, b 0.025), and downstream MLPs 12-17 carry 79% of b. Four controls, iso design
(other stack MLPs base-frozen) as v39:
  bilinear  rec(alpha Delta + beta v) - rec(alpha Delta) on a 3x3 grid, fitted c1 beta + c2 alpha beta + c3 beta^2:
            a product formed by bilinear layers is exactly bilinear in the two amplitudes (beta^2 = v's own self term).
  randwrite the write replaced by a random vector of the same norm (gamma Delta_r, 2 seeds), v fixed: b_r.
  randvec   v replaced by a random vector of the same norm (2 seeds), real write: a_r, b_r (margin-inert control).
  selfgate  the direct path rec(alpha Delta) - rec(0), all stack base-frozen, fitted a alpha + b alpha^2: does the
            write gate ITSELF? (v35 LIN direct slope 1.03 says barely.)
  fullstack the real stack live (mlp 08-11 for polarity) with downstream MLPs frozen to the write-only run: conversion
            slope over {0.25, 0.5, 0.75, 1} -- is the alpha^1.53 of the whole conversion the downstream product?

REGISTERED BEFORE THE RUN
    pred_a_bilinear_form   R^2 >= 0.995 and beta^2 share (c3 / (c1 + c2 + c3) at (1,1)) <= 0.15 on all three.
                           Worked: 0.999 / 0.05 True; 0.98 False.
    pred_b_write_specific  |b_r| <= 0.25 |b| for both random writes on polarity. Worked: 0.003 vs 0.025 True; 0.015 False.
    pred_c_vector_specific |a_r| + |b_r| <= 0.25 (|a| + |b|) for both random vectors on polarity. Worked: 0.005 vs 0.045
                           True; 0.02 False.
    pred_d_no_self_gating  self-gate share b / (a + b) <= 0.25 on all three. Worked: 0.10 True; 0.4 False.
    pred_e_fullstack_flat  with downstream MLPs frozen, polarity's full-stack conversion slope <= 1.3. Worked: 1.53 ->
                           1.15 True; -> 1.45 False.
    Reading rule. a-c True: the gate is a bilinear AND of the specific write with its specific conversion, formed
    downstream; with d True the write does not multiply itself -- the product needs the converted copy. e True: the
    whole alpha^2 law of the early sets is this product. b False: the gate is a generic gain set by any large residual
    write (then it is a norm effect of the downstream normalizations after all, at the DOWNSTREAM layers -- test with
    Delta_r of varied norm). c False: the downstream product amplifies any vector -- a context gain, not an AND.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_write_gated_readout_v39 as v39

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_gate_specificity_v40_result.json"
SETS, STACK = v35.SETS, v35.STACK
GRID3, GRID4, GRID5 = (0.25, 0.5, 1.0), (0.25, 0.5, 0.75, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0)
SEEDS = (11, 12)
R2_BAR, BETA2_MAX, WRITE_RATIO, VEC_RATIO, SELF_MAX, FULL_MAX, N_LAYERS = 0.995, 0.15, 0.25, 0.25, 0.25, 1.3, 18
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 300, 10000


def _plan():
    return {"candidate_id": "corpus.unit_gate_specificity_v40", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grids": [list(GRID3), list(GRID4), list(GRID5)], "seeds": list(SEEDS), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def _lstsq(torch, cols, y):
    A = torch.tensor(cols, dtype=torch.float64).T
    yt = torch.tensor(y, dtype=torch.float64)
    c = torch.linalg.lstsq(A, yt.unsqueeze(1)).solution.squeeze(1)
    pred = A @ c
    sst = float(((yt - yt.mean()) ** 2).sum()); sse = float(((yt - pred) ** 2).sum())
    return [float(x) for x in c], (1 - sse / sst) if sst else None


def _quad_origin(xs, ys):
    # y = a x + b x^2
    s11 = sum(x * x for x in xs); s12 = sum(x ** 3 for x in xs); s22 = sum(x ** 4 for x in xs)
    t1 = sum(x * y for x, y in zip(xs, ys)); t2 = sum(x * x * y for x, y in zip(xs, ys))
    det = s11 * s22 - s12 * s12
    return (s22 * t1 - s12 * t2) / det, (s11 * t2 - s12 * t1) / det


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first, last = layers[0], layers[-1]
        down_mlps = list(range(last + 1, N_LAYERS))
        mlp = model.transformer.h[first].mlp
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        u_b = ins_b[first]
        _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
        v = v38._cross(mlp, u_b, ins_p1[first] - u_b)
        fz = v39._Freezer(torch, model, list(prep.base_batch.semantic_positions), backend.device)

        def run(write, vec, *, live_stack=False):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
            kw = {"units": [] if live_stack else stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
            if write is not None:
                kw["resid_add"] = {first: write}
            return v30._capture(backend, prep, layers, **kw)[0]

        # bilinear surface
        surf, cols_b, cols_ab, cols_bb, ys = {}, [], [], [], []
        for a in GRID3:
            r0 = run(a * delta1, None)
            for b in GRID3:
                y = run(a * delta1, b * v) - r0
                surf[f"{a},{b}"] = y; cols_b.append(b); cols_ab.append(a * b); cols_bb.append(b * b); ys.append(y)
        coef, r2 = _lstsq(torch, [cols_b, cols_ab, cols_bb], ys)
        tot = sum(coef)
        bilinear = {"surface": surf, "c_beta": coef[0], "c_alphabeta": coef[1], "c_beta2": coef[2], "r2": r2,
                    "beta2_share": coef[2] / tot if tot else None, "product_share": coef[1] / tot if tot else None}
        # real readout line (v39 replicate)
        line = {a: run(a * delta1 if a else None, v) - run(a * delta1 if a else None, None) for a in GRID5}
        a0, b0 = v39._fit_ab(GRID5, [line[a] for a in GRID5])
        gen = torch.Generator(device="cpu")
        randwrite, randvec = {}, {}
        for seed in SEEDS:
            gen.manual_seed(seed)
            dr = torch.randn(delta1.shape, generator=gen).to(delta1.device, delta1.dtype)
            dr = dr / dr.norm(dim=1, keepdim=True) * delta1.norm(dim=1, keepdim=True)
            ln = {a: run(a * dr if a else None, v) - run(a * dr if a else None, None) for a in GRID5}
            ar, br = v39._fit_ab(GRID5, [ln[a] for a in GRID5])
            randwrite[seed] = {"line": {str(a): ln[a] for a in GRID5}, "a": ar, "b": br}
            gen.manual_seed(seed + 100)
            vr = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
            vr = vr / vr.norm(dim=1, keepdim=True) * v.norm(dim=1, keepdim=True)
            ln = {a: run(a * delta1 if a else None, vr) - run(a * delta1 if a else None, None) for a in GRID5}
            ar, br = v39._fit_ab(GRID5, [ln[a] for a in GRID5])
            randvec[seed] = {"line": {str(a): ln[a] for a in GRID5}, "a": ar, "b": br}
        # self gating of the direct path
        r00 = run(None, None)
        direct = {a: run(a * delta1, None) - r00 for a in GRID4}
        sa, sb = _quad_origin(GRID4, [direct[a] for a in GRID4])
        # full stack with downstream MLPs frozen to the write-only run
        full = {}
        for a in GRID4:
            with fz.hooks("capture", down_mlps, []):
                base = run(a * delta1, None)
            with fz.hooks("replace", down_mlps, []):
                live = run(a * delta1, None, live_stack=True)
            full[a] = live - base
        full_free = {a: run(a * delta1, None, live_stack=True) - run(a * delta1, None) for a in GRID4}
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "downstream_mlps": down_mlps,
                        "bilinear": bilinear, "readout_line": {str(a): line[a] for a in GRID5}, "a": a0, "b": b0,
                        "randwrite": randwrite, "randvec": randvec,
                        "selfgate": {"direct": {str(a): direct[a] for a in GRID4}, "a": sa, "b": sb, "share": sb / (sa + sb) if (sa + sb) else None},
                        "fullstack": {"conv_down_frozen": {str(a): full[a] for a in GRID4}, "slope_down_frozen": v35._slope(GRID4, [full[a] for a in GRID4]),
                                      "conv_free": {str(a): full_free[a] for a in GRID4}, "slope_free": v35._slope(GRID4, [full_free[a] for a in GRID4])}}
        print(name, "R2 %.4f prod %.2f beta2 %.2f | a %.4f b %.4f | randwrite b %s | randvec a+b %s | selfgate %.2f | full %.2f -> %.2f" % (
            r2, bilinear["product_share"], bilinear["beta2_share"], a0, b0, [round(randwrite[s]["b"], 4) for s in SEEDS],
            [round(abs(randvec[s]["a"]) + abs(randvec[s]["b"]), 4) for s in SEEDS], report[name]["selfgate"]["share"],
            report[name]["fullstack"]["slope_free"] or -1, report[name]["fullstack"]["slope_down_frozen"] or -1), flush=True)

    pol = report["polarity_licensing"]
    predictions = {
        'pred_a_bilinear_form': all(report[n]["bilinear"]["r2"] is not None and report[n]["bilinear"]["r2"] >= R2_BAR and
                                    report[n]["bilinear"]["beta2_share"] is not None and abs(report[n]["bilinear"]["beta2_share"]) <= BETA2_MAX for n in SETS),
        'pred_b_write_specific': all(abs(pol["randwrite"][s]["b"]) <= WRITE_RATIO * abs(pol["b"]) for s in SEEDS),
        'pred_c_vector_specific': all(abs(pol["randvec"][s]["a"]) + abs(pol["randvec"][s]["b"]) <= VEC_RATIO * (abs(pol["a"]) + abs(pol["b"])) for s in SEEDS),
        'pred_d_no_self_gating': all(report[n]["selfgate"]["share"] is not None and report[n]["selfgate"]["share"] <= SELF_MAX for n in SETS),
        'pred_e_fullstack_flat': pol["fullstack"]["slope_down_frozen"] is not None and pol["fullstack"]["slope_down_frozen"] <= FULL_MAX,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_gate_specificity_result_v1",
              "candidate_id": "corpus.unit_gate_specificity_v40", "seeds": list(SEEDS),
              "bars": {"r2": R2_BAR, "beta2_max": BETA2_MAX, "write_ratio": WRITE_RATIO, "vec_ratio": VEC_RATIO, "self_max": SELF_MAX, "full_max": FULL_MAX},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
