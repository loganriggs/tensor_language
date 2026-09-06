#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grids, families and the freeze arms fixed before the run.
"""v41: does the bilinear AND hold on the A2 construction and under the REAL (scaled set) write?

v39/v40 established on A1 rows under a linear replayed write: the margin response to the converted vector v is
a + b*alpha in the write, exactly bilinear, write- and vector-specific, and formed in downstream MLPs (polarity
mlp 12-17, 79%; quantifier mlp:15/16). Standing rule: a mechanism has to hold on all four hypotheses, not A1
alone, and the linear write is a replay -- the circuit's own write is the scaled set (v27 semantics).
  A2      the v39 readout line (5 alphas, iso design) and the random-vector control on A2 rows; freeze MLP-all.
  REAL    A1, the scaled set write sqrt(alpha) I per set block, full stack live: conversion slope with and without
          the downstream MLPs frozen to the write-only run (v40 pred_e for the real write, whose STD slope is
          2.16 / 1.55 / 2.04).

REGISTERED BEFORE THE RUN
    pred_a_a2_polarity_gated   A2 polarity modulation share >= 0.40. Worked: 0.50 True; 0.2 False.
    pred_b_a2_vector_specific  A2 polarity random vector |a_r| + |b_r| <= 0.25 (|a| + |b|), both seeds. Worked: 0.005 vs
                               0.04 True.
    pred_c_a2_mlps_multiply    A2 polarity MLP-all removes >= 60% of b. Worked: 0.75 True; 0.4 False.
    pred_d_real_write_drop     REAL polarity conversion slope drops by >= 0.20 when mlp 12-17 are frozen. Worked: 2.16 ->
                               1.85 True; -> 2.05 False.
    pred_e_a2_quantifier_local A2 quantifier: mlp:15 alone removes >= 50% of b. Worked: 0.8 True; 0.3 False.
    Reading rule. a-c True: the AND is a property of the behaviour, not of the A1 stimulus family. d True: the
    product is part of the circuit's own computation, not an artefact of the linear replay. d False: under the real
    write the downstream product is masked by the upstream (pre-stack) superlinearity -- then it is a second-order
    contributor and the dossier says so. a False: the A1 gating is family-specific; report and stop this thread.
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
import run_unit_damper_law_v27 as v27
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_write_gated_readout_v39 as v39

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_gate_a2_and_real_write_v41_result.json"
SETS, STACK = v35.SETS, v35.STACK
GRID4, GRID5, SEEDS, N_LAYERS = (0.25, 0.5, 0.75, 1.0), (0.0, 0.25, 0.5, 0.75, 1.0), (11, 12), 18
A2_GATED, VEC_RATIO, MLP_REMOVE, REAL_DROP, LOCAL_REMOVE = 0.40, 0.25, 0.6, 0.20, 0.5
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 240, 8000


def _plan():
    return {"candidate_id": "corpus.unit_gate_a2_and_real_write_v41", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "families": ["A2", "A1"], "grids": [list(GRID4), list(GRID5)], "seeds": list(SEEDS), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first, last = layers[0], layers[-1]
        down_mlps = list(range(last + 1, N_LAYERS))
        mlp = model.transformer.h[first].mlp
        rep = {"units": list(units), "stack_mlps": stack_mlps, "downstream_mlps": down_mlps}

        # ---- A2: readout line, random vector, freezes
        prep = g.prepare(backend, g.rows_of(module, "A2"), valid_only=True)
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        rec_live1, _, _ = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        u_b = ins_b[first]
        _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
        v = v38._cross(mlp, u_b, ins_p1[first] - u_b)
        fz = v39._Freezer(torch, model, list(prep.base_batch.semantic_positions), backend.device)

        def run(write, vec):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
            kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
            if write is not None:
                kw["resid_add"] = {first: write}
            return v30._capture(backend, prep, layers, **kw)[0]

        arms = {"none": [], "MLP-all": down_mlps}
        arms.update({f"mlp:{l:02d}": [l] for l in down_mlps})
        lines = {arm: {} for arm in arms}
        for a in GRID5:
            w = a * delta1 if a else None
            for arm, ms in arms.items():
                with fz.hooks("capture", ms, []):
                    r0 = run(w, None)
                with fz.hooks("replace", ms, []):
                    r1 = run(w, v)
                lines[arm][a] = r1 - r0
        fits = {arm: v39._fit_ab(GRID5, [lines[arm][a] for a in GRID5]) for arm in arms}
        a0, b0 = fits["none"]
        gen = torch.Generator(device="cpu")
        randvec = {}
        for seed in SEEDS:
            gen.manual_seed(seed + 100)
            vr = torch.randn(v.shape, generator=gen).to(v.device, v.dtype)
            vr = vr / vr.norm(dim=1, keepdim=True) * v.norm(dim=1, keepdim=True)
            ln = {a: run(a * delta1 if a else None, vr) - run(a * delta1 if a else None, None) for a in GRID5}
            ar, br = v39._fit_ab(GRID5, [ln[a] for a in GRID5])
            randvec[seed] = {"a": ar, "b": br}
        rep["A2"] = {"rows": len(prep.rows), "rec_live_1": rec_live1, "readout": {arm: {str(a): lines[arm][a] for a in GRID5} for arm in arms},
                     "fit": {arm: {"a": fits[arm][0], "b": fits[arm][1]} for arm in arms},
                     "modulation_share": b0 / (a0 + b0) if (a0 + b0) else None,
                     "b_removed": {arm: (b0 - fits[arm][1]) / b0 if b0 else None for arm in arms if arm != "none"}, "randvec": randvec}
        print(name, "A2 rec_live %.3f a %.4f b %.4f share %.3f" % (rec_live1, a0, b0, rep["A2"]["modulation_share"]),
              {arm: round(x, 2) for arm, x in rep["A2"]["b_removed"].items()}, "randvec", {s: round(abs(x["a"]) + abs(x["b"]), 4) for s, x in randvec.items()}, flush=True)

        # ---- REAL write on A1: scaled set, full stack live, downstream MLPs frozen or free
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        _, _, outs_b = v30._capture(backend, prep, layers)
        frozen = dict(cache)
        for l, m in zip(layers, stack_mlps):
            for i, rid in enumerate(rids):
                frozen[(rid, m)] = outs_b[l][i]
        fz = v39._Freezer(torch, model, list(prep.base_batch.semantic_positions), backend.device)

        def real(alpha, stack_frozen):
            q = v27._scaled_q(backend, units, alpha)
            us = list(units)
            if stack_frozen:
                for key in g.blocks_of(stack_mlps):
                    q[key] = torch.eye(g.N_EMBD, device=backend.device)
                us = us + stack_mlps
            return v30._capture(backend, prep, layers, units=us, donor_cache=frozen, base_cache=prep.base_cache, q=q)[0]

        free, down = {}, {}
        for a in GRID4:
            free[a] = real(a, False) - real(a, True)
            with fz.hooks("capture", down_mlps, []):
                base = real(a, True)
            with fz.hooks("replace", down_mlps, []):
                live = real(a, False)
            down[a] = live - base
        rep["REAL"] = {"rows": len(prep.rows), "conv_free": {str(a): free[a] for a in GRID4}, "conv_down_frozen": {str(a): down[a] for a in GRID4},
                       "slope_free": v35._slope(GRID4, [free[a] for a in GRID4]), "slope_down_frozen": v35._slope(GRID4, [down[a] for a in GRID4])}
        rep["REAL"]["slope_drop"] = (rep["REAL"]["slope_free"] - rep["REAL"]["slope_down_frozen"]) if None not in (rep["REAL"]["slope_free"], rep["REAL"]["slope_down_frozen"]) else None
        print(name, "REAL free", {a: round(free[a], 3) for a in GRID4}, "slope", round(rep["REAL"]["slope_free"] or -1, 2), "down-frozen",
              {a: round(down[a], 3) for a in GRID4}, "slope", round(rep["REAL"]["slope_down_frozen"] or -1, 2), flush=True)
        report[name] = rep

    pol, qua = report["polarity_licensing"], report["quantifier_number"]
    predictions = {
        'pred_a_a2_polarity_gated': pol["A2"]["modulation_share"] is not None and pol["A2"]["modulation_share"] >= A2_GATED,
        'pred_b_a2_vector_specific': all(abs(x["a"]) + abs(x["b"]) <= VEC_RATIO * (abs(pol["A2"]["fit"]["none"]["a"]) + abs(pol["A2"]["fit"]["none"]["b"]))
                                         for x in pol["A2"]["randvec"].values()),
        'pred_c_a2_mlps_multiply': pol["A2"]["b_removed"]["MLP-all"] is not None and pol["A2"]["b_removed"]["MLP-all"] >= MLP_REMOVE,
        'pred_d_real_write_drop': pol["REAL"]["slope_drop"] is not None and pol["REAL"]["slope_drop"] >= REAL_DROP,
        'pred_e_a2_quantifier_local': qua["A2"]["b_removed"].get("mlp:15") is not None and qua["A2"]["b_removed"]["mlp:15"] >= LOCAL_REMOVE,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_gate_a2_and_real_write_result_v1",
              "candidate_id": "corpus.unit_gate_a2_and_real_write_v41", "seeds": list(SEEDS),
              "bars": {"a2_gated": A2_GATED, "vec_ratio": VEC_RATIO, "mlp_remove": MLP_REMOVE, "real_drop": REAL_DROP, "local_remove": LOCAL_REMOVE},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
