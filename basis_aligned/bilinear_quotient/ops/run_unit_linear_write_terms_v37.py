#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grid and the replayed term decomposition fixed before the run.
"""v37: a lone stack layer reading an exactly LINEAR write converts it as alpha^1.3-1.7 (v36 pred_a False: mlp:08 1.69,
mlp:11 1.28, mlp:07 1.36). The layer has two terms: cross Down[L(u_b)R(w)+L(w)R(u_b)] (linear in w) and self
Down[L(w)R(w)] (quadratic), w = rms_norm(x_b + alpha Delta) - rms_norm(x_b). Either the self term's share of a lone
layer's conversion is ~50% (v30's 24-29% was the whole stack under the real write), or the cross term's
margin-relevant component is not linear in alpha, i.e. the normalization's nonlinearity matters after all (against
v34). Exact replayed expansion under the linear write (LIN of v35), per alpha: capture every stack layer's normalized
input and output on the all-live run, form cross_l / self_l, and replay static stack outputs
  iso-first cross   first = base + cross_first, rest base       iso-first self, iso-first both (identity check)
  all cross         every layer base + cross_l                  all self, all both (identity check)
conv = rec(replay) - rec(all base), same alpha; slopes log-log over {0.25, 0.5, 0.75, 1}.

REGISTERED BEFORE THE RUN
    pred_a_cross_linear      iso-first cross-only conv slope in [0.9, 1.2] on all three. Worked: 1.05 True; 1.5 False.
    pred_b_self_quadratic    iso-first self-only conv slope in [1.8, 2.2] on all three. Worked: 2.0 True; 1.4 False.
    pred_c_self_half         self share of the iso-first conversion at alpha = 1 >= 0.4 on polarity AND voice. Worked:
                             0.55 True; 0.20 False.
    pred_d_identity          |iso-first both - v36 iso conv| <= 0.01 and |all both - conv_total| <= 0.01 at alpha = 1 on
                             all three (the static replay reproduces the live run). Worked: 0.002 True.
    pred_e_cross_stack_linear all-layers cross-only conv slope <= 1.25 on all three. Worked: 1.10 True; 1.5 False.
    Reading rule. a, b, e True & c True: the alpha^1.5 of a linear write is the bilinear self term at half share -- no
    hidden nonlinearity; the layer is literally a quadratic form in the write, and the whole-stack law follows from
    per-layer shares. a False: the cross term of a lone layer is superlinear in alpha although linear in w -- then
    w(alpha)'s margin-relevant component is nonlinear in alpha and v34's inertness claim fails under a linear write;
    next is <g, w(alpha)> directly. e False & a True: the cross terms compose nonlinearly across layers.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_linear_write_terms_v37_result.json"
V36 = ROOT / "circuits/followups/unit_stack_composition_v36_result.json"
SETS, STACK, EARLY, GRID = v35.SETS, v35.STACK, v35.EARLY, v35.GRID
CROSS_BAND, SELF_BAND, SELF_SHARE_BAR, ID_TOL, STACK_CROSS_MAX = (0.9, 1.2), (1.8, 2.2), 0.4, 0.01, 1.25
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 4000


def _plan():
    return {"candidate_id": "corpus.unit_linear_write_terms_v37", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    v36 = json.loads(V36.read_text())["behaviours"]
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first, m_first = layers[0], stack_mlps[0]
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])

        def replay(alpha, values):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = values.get(l, outs_b[l])[i]
            return v30._capture(backend, prep, layers, units=stack_mlps, donor_cache=c, base_cache=prep.base_cache,
                                resid_add={first: alpha * delta1})[0]

        per = {}
        for a in GRID:
            rec_live, ins_p, outs_p = v30._capture(backend, prep, layers, resid_add={first: a * delta1})
            terms = {l: v30._terms(model.transformer.h[l].mlp, ins_b[l], ins_p[l]) for l in layers}
            base = replay(a, {})
            r = {"live": rec_live, "all_base": base,
                 "iso_cross": replay(a, {first: outs_b[first] + terms[first][0]}),
                 "iso_self": replay(a, {first: outs_b[first] + terms[first][1]}),
                 "iso_both": replay(a, {first: outs_b[first] + terms[first][0] + terms[first][1]}),
                 "all_cross": replay(a, {l: outs_b[l] + terms[l][0] for l in layers}),
                 "all_self": replay(a, {l: outs_b[l] + terms[l][1] for l in layers}),
                 "all_both": replay(a, {l: outs_b[l] + terms[l][0] + terms[l][1] for l in layers})}
            conv = {k: v - base for k, v in r.items() if k not in ("all_base",)}
            per[a] = {"rec": r, "conv": conv,
                      "self_norm_over_cross_first": float((terms[first][1].norm(dim=1) / terms[first][0].norm(dim=1).clamp_min(1e-12)).mean()),
                      "write_norm_first": float((ins_p[first] - ins_b[first]).norm(dim=1).mean())}
            print(name, a, {k: round(v, 3) for k, v in conv.items()}, flush=True)
        slopes = {k: v35._slope(GRID, [per[a]["conv"][k] for a in GRID]) for k in per[GRID[0]]["conv"]}
        c1 = per[1.0]["conv"]
        v36b = v36[name]["per_alpha"]["1.0"]
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "per_alpha": per, "slopes": slopes,
                        "self_share_iso_first_1": c1["iso_self"] / c1["iso_both"] if c1["iso_both"] else None,
                        "cross_share_iso_first_1": c1["iso_cross"] / c1["iso_both"] if c1["iso_both"] else None,
                        "self_share_all_1": c1["all_self"] / c1["all_both"] if c1["all_both"] else None,
                        "identity_iso_1": c1["iso_both"] - v36b["conv_iso"][m_first], "identity_all_1": c1["all_both"] - v36b["conv_total"],
                        "write_norm_slope": v35._slope(GRID, [per[a]["write_norm_first"] for a in GRID])}
        print(name, "slopes", {k: round(s, 2) if s is not None else s for k, s in slopes.items()}, "self_share_iso", round(report[name]["self_share_iso_first_1"], 3),
              "self_share_all", round(report[name]["self_share_all_1"], 3), "ident", round(report[name]["identity_iso_1"], 4), round(report[name]["identity_all_1"], 4), flush=True)

    def sl(n, k):
        return report[n]["slopes"][k]
    def inband(v, band):
        return v is not None and band[0] <= v <= band[1]
    predictions = {
        'pred_a_cross_linear': all(inband(sl(n, "iso_cross"), CROSS_BAND) for n in SETS),
        'pred_b_self_quadratic': all(inband(sl(n, "iso_self"), SELF_BAND) for n in SETS),
        'pred_c_self_half': all(report[n]["self_share_iso_first_1"] is not None and report[n]["self_share_iso_first_1"] >= SELF_SHARE_BAR for n in EARLY),
        'pred_d_identity': all(abs(report[n]["identity_iso_1"]) <= ID_TOL and abs(report[n]["identity_all_1"]) <= ID_TOL for n in SETS),
        'pred_e_cross_stack_linear': all(sl(n, "all_cross") is not None and sl(n, "all_cross") <= STACK_CROSS_MAX for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_linear_write_terms_result_v1",
              "candidate_id": "corpus.unit_linear_write_terms_v37", "grid": list(GRID),
              "bars": {"cross_band": list(CROSS_BAND), "self_band": list(SELF_BAND), "self_share": SELF_SHARE_BAR, "id_tol": ID_TOL, "stack_cross_max": STACK_CROSS_MAX},
              "slopes": {n: report[n]["slopes"] for n in SETS}, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "slopes": result["slopes"], "seconds": round(result["seconds"], 1)}, indent=2, default=str))


if __name__ == "__main__":
    main()
