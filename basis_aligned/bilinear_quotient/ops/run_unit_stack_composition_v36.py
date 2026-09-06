#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grid and the isolated/cumulative decomposition fixed before the run.
"""v36: under a LINEAR residual write, where in the stack does the alpha^1.5 come from?

v35: replaying alpha * Delta_first(1) (the measured pre-norm residual delta at the first stack layer) on the base run
is exact, base attention patterns keep 99/100/88% of the effect, and the stack's conversion of that linear write
scales as alpha^1.53 / 1.51 / 1.55 on polarity / quantifier / voice -- the same law for all three, pattern-independent.
Every single-layer piece is accounted for: the first stack layer's cross term Down[L(u_b)R(w) + L(w)R(u_b)] is linear
in the normalized write, whose first order is linear in alpha (v34: second order inert), the self term is 24-29% and
quadratic (v30), the readout is linear (v31). A 25% quadratic admixture gives a fitted slope of ~1.15, not 1.5. What is
untested is the stack layers' composition with EACH OTHER: v32 tested write x stack, never layer x layer.
Decomposition under the linear write (LIN of v35), stack MLPs replaced by base outputs except the live ones:
  iso_l   only stack layer l live            cum_k   the first k stack layers live      total   all live
conv = rec - rec(all frozen), same alpha. Slopes log-log over {0.25, 0.5, 0.75, 1}.

REGISTERED BEFORE THE RUN
    pred_a_first_layer_linear  iso slope of the FIRST stack layer in [0.9, 1.25] on all three -- the control capable of
                               failing: a lone layer reading a linear write has only its self term to be nonlinear with.
                               Worked: 1.10 True; 1.5 False (then the single-layer algebra above is wrong).
    pred_b_every_layer_linear  every iso slope <= 1.3 on all three. Worked: max 1.2 True.
    pred_c_superadditive       interaction share at alpha=1, (conv_total - sum_l conv_iso_l) / conv_total >= 0.25 on all
                               three. Worked: 0.35 True; 0.10 False.
    pred_d_exponent_builds     cumulative slope non-decreasing in k (tolerance 0.03) for polarity AND voice. Worked:
                               1.10, 1.30, 1.45, 1.53 True; 1.4, 1.2, ... False.
    pred_e_quadratic_mixture   conv_total(alpha) fits c1 alpha + c2 alpha^2 with R^2 >= 0.995 and c2/(c1+c2) in [0.3, 0.6]
                               on all three. Worked: slope 1.5 <-> share ~0.45 True.
    Reading rule. a,b True & c True: the exponent is compositional -- layer l+1 converts the write AND layer l's
    conversion of it (product of two linear responses = quadratic), so the stack is a multiplicative cascade; next: the
    pairwise (l, l+1) product terms and whether the cascade is the same for the three behaviours. a False: re-derive the
    single-layer algebra with the actual third-order rms_norm terms. c False & b True: the layers are near-linear alone
    and additive together -- then the exponent is in something the freeze-to-base shares with the live run (the stack's
    attention values), expand those.
"""
from __future__ import annotations

import json
import math
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
OUT = ROOT / "circuits/followups/unit_stack_composition_v36_result.json"
SETS, STACK, EARLY, GRID = v35.SETS, v35.STACK, v35.EARLY, v35.GRID
FIRST_BAND, ISO_MAX, INTERACTION_BAR, MONO_TOL, R2_BAR, SHARE_BAND = (0.9, 1.25), 1.3, 0.25, 0.03, 0.995, (0.3, 0.6)
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 160, 6000


def _plan():
    return {"candidate_id": "corpus.unit_stack_composition_v36", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _quad_fit(xs, ys):
    # least squares y = c1 x + c2 x^2 (no intercept), R^2 about the mean
    s11 = sum(x * x for x in xs); s12 = sum(x ** 3 for x in xs); s22 = sum(x ** 4 for x in xs)
    t1 = sum(x * y for x, y in zip(xs, ys)); t2 = sum(x * x * y for x, y in zip(xs, ys))
    det = s11 * s22 - s12 * s12
    c1, c2 = (s22 * t1 - s12 * t2) / det, (s11 * t2 - s12 * t1) / det
    my = sum(ys) / len(ys)
    sse = sum((y - c1 * x - c2 * x * x) ** 2 for x, y in zip(xs, ys)); sst = sum((y - my) ** 2 for y in ys)
    return c1, c2, 1 - sse / sst if sst else None


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch = backend.torch
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first = layers[0]
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, _, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        rec_live1, _, _ = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        frozen = dict(cache)
        for l, m in zip(layers, stack_mlps):
            for i, rid in enumerate(rids):
                frozen[(rid, m)] = outs_b[l][i]

        def rec(alpha, live_mlps):
            fro = [m for m in stack_mlps if m not in live_mlps]
            return v30._capture(backend, prep, layers, units=fro, donor_cache=frozen, base_cache=prep.base_cache,
                                resid_add={first: alpha * delta1})[0]

        per = {}
        for a in GRID:
            base = rec(a, [])
            total = rec(a, stack_mlps) - base
            iso = {m: rec(a, [m]) - base for m in stack_mlps}
            cum = {k: rec(a, stack_mlps[:k]) - base for k in range(1, len(stack_mlps) + 1)}
            per[a] = {"rec_all_frozen": base, "conv_total": total, "conv_iso": iso, "conv_cum": cum, "sum_iso": sum(iso.values()),
                      "interaction_share": (total - sum(iso.values())) / total if total else None}
            print(name, a, "total", round(total, 3), "iso", [round(v, 3) for v in iso.values()], "cum", [round(v, 3) for v in cum.values()], flush=True)
        iso_slopes = {m: v35._slope(GRID, [per[a]["conv_iso"][m] for a in GRID]) for m in stack_mlps}
        cum_slopes = {k: v35._slope(GRID, [per[a]["conv_cum"][k] for a in GRID]) for k in range(1, len(stack_mlps) + 1)}
        c1, c2, r2 = _quad_fit(GRID, [per[a]["conv_total"] for a in GRID])
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "rec_live_1": rec_live1, "per_alpha": per,
                        "total_slope": v35._slope(GRID, [per[a]["conv_total"] for a in GRID]),
                        "sum_iso_slope": v35._slope(GRID, [per[a]["sum_iso"] for a in GRID]),
                        "iso_slopes": iso_slopes, "cum_slopes": cum_slopes, "interaction_share_1": per[1.0]["interaction_share"],
                        "quad_fit": {"c1": c1, "c2": c2, "r2": r2, "quadratic_share": c2 / (c1 + c2) if (c1 + c2) else None}}
        print(name, "total", round(report[name]["total_slope"], 2), "iso", {m: round(s, 2) if s else s for m, s in iso_slopes.items()},
              "cum", {k: round(s, 2) if s else s for k, s in cum_slopes.items()}, "int", round(per[1.0]["interaction_share"], 3), flush=True)

    def mono(name):
        s = [report[name]["cum_slopes"][k] for k in sorted(report[name]["cum_slopes"])]
        return all(x is not None for x in s) and all(b >= a - MONO_TOL for a, b in zip(s, s[1:]))
    predictions = {
        'pred_a_first_layer_linear': all(report[n]["iso_slopes"][report[n]["stack_mlps"][0]] is not None and
                                         FIRST_BAND[0] <= report[n]["iso_slopes"][report[n]["stack_mlps"][0]] <= FIRST_BAND[1] for n in SETS),
        'pred_b_every_layer_linear': all(all(s is not None and s <= ISO_MAX for s in report[n]["iso_slopes"].values()) for n in SETS),
        'pred_c_superadditive': all(report[n]["interaction_share_1"] is not None and report[n]["interaction_share_1"] >= INTERACTION_BAR for n in SETS),
        'pred_d_exponent_builds': all(mono(n) for n in EARLY),
        'pred_e_quadratic_mixture': all(report[n]["quad_fit"]["r2"] is not None and report[n]["quad_fit"]["r2"] >= R2_BAR and
                                        report[n]["quad_fit"]["quadratic_share"] is not None and
                                        SHARE_BAND[0] <= report[n]["quad_fit"]["quadratic_share"] <= SHARE_BAND[1] for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_stack_composition_result_v1",
              "candidate_id": "corpus.unit_stack_composition_v36", "grid": list(GRID),
              "bars": {"first_band": list(FIRST_BAND), "iso_max": ISO_MAX, "interaction": INTERACTION_BAR, "mono_tol": MONO_TOL,
                       "r2": R2_BAR, "share_band": list(SHARE_BAND)},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1),
                      "summary": {n: {"total": report[n]["total_slope"], "iso": report[n]["iso_slopes"], "cum": report[n]["cum_slopes"],
                                      "interaction_1": report[n]["interaction_share_1"], "quad": report[n]["quad_fit"]} for n in SETS}}, indent=2, default=str))


if __name__ == "__main__":
    main()
