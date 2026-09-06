#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grid and the per-alpha baselines fixed before the run (repairs v31 arm X).
"""v33: which CHANNEL of the stack's response is superlinear in the write, and does the write rotate?

Standing facts. v29: stack conversion ~ alpha^2 (polarity 2.08, voice 2.04). v30: at alpha = 1 the response is 72-79%
cross-term (linear in w by construction), 24-29% self-term. v31: the margin reads the stack output linearly; |w_first|
is linear (1.08 / 1.00; voice 1.30). v32: the stack's alpha = 1 output delta is sufficient ON ITS OWN (91-102% of
conversion on the base run; no interaction with the write, R^2 >= 0.995). So the alpha^2 lives inside DeltaM(alpha)
itself: either the cross channel's margin-relevant component is superlinear because w(alpha) ROTATES (rms_norm second
order, or intermediate-layer responses), or the self channel carries more of the low-alpha structure than its 24%
at alpha = 1 suggests. This run measures each channel with the baseline at the SAME alpha (v31's error).

Per alpha in {0.25, 0.5, 0.75, 1}: write at alpha; capture w_l, DeltaM_l; exact cross_l/self_l; four stack replays on
top of the alpha-write -- base stack (frozen(alpha)), +cross only, +self only, +DeltaM (= live, instrument).
conv_X(alpha) = rec(X, alpha) - rec(frozen, alpha). Cosines of w_l(alpha), cross_l(alpha), DeltaM_l(alpha) to their
alpha = 1 counterparts, per row then averaged. Slopes are log-log least squares over the grid.

REGISTERED BEFORE THE RUN
    pred_a_instrument       rec(+DeltaM, alpha) = rec_live(alpha) within 0.005 at every alpha and set; conv_total(1) matches
                            v32's conversion (0.318 / 0.193 / 0.513) within 0.005.
    pred_b_total_quadratic  conv_total slope in [1.6, 2.4] for polarity AND voice (v29 reproduced with per-alpha baselines).
    pred_c_cross_linear     conv_cross slope in [0.8, 1.2] for polarity AND voice: the cross channel's margin effect scales
                            with its norm. Worked: 1.05 True; 1.7 False (then w rotates or the margin reads a rotating part).
    pred_d_write_no_rotation  mean cos(w_first(alpha), w_first(1)) >= 0.95 at every alpha for polarity AND quantifier.
                            Worked: 0.99 True; 0.90 False.
    pred_e_response_rotates mean cos(DeltaM_first(alpha), DeltaM_first(1)) at alpha = 0.25 <= 0.90 for polarity AND voice:
                            the first stack layer's response changes direction as the write grows (self/cross balance).
                            Worked: 0.82 True; 0.97 False.
    Reading rule. c True & b True: the self channel (or the cross/self interplay in the margin) supplies the quadratic even at
    24% of the alpha=1 norm -- then Tier 4 is a two-term closed form, cross linear + self quadratic, to be checked by
    predicting conv(alpha) = c1 alpha + c2 alpha^2 from the alpha = 1 channels alone. c False & d True: the cross channel is
    superlinear without rotation of w at the first layer -- the rotation is downstream in the stack (cascade); expand
    layer by layer. d False: the write rotates before the stack (rms_norm / intermediate layers): expand rms_norm to
    second order next.
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
import run_unit_damper_law_v27 as v27
import run_unit_selfterm_sufficiency_v30 as v30

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_channel_scaling_v33_result.json"
SETS, STACK, EARLY = v30.SETS, v30.STACK, v30.EARLY
GRID = (0.25, 0.5, 0.75, 1.0)
V32_CONV = {"polarity_licensing": 0.318, "quantifier_number": 0.193, "voice_frame": 0.513}
INSTR_TOL, QUAD, LIN, COS_KEEP, COS_ROT = 0.005, (1.6, 2.4), (0.8, 1.2), 0.95, 0.90
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 80, 3000


def _plan():
    return {"candidate_id": "corpus.unit_channel_scaling_v33", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _slope(xs, ys):
    if any(y <= 0 for y in ys):
        return None
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def _cos(torch, x, y):
    return float(torch.nn.functional.cosine_similarity(x, y, dim=1).mean())


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
        first = layers[0]
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        _, ins_b, outs_b = v30._capture(backend, prep, layers)
        all_units = list(units) + stack_mlps
        captured, per_alpha = {}, {}
        for a in GRID:
            q_set = v27._scaled_q(backend, units, a)
            rec_live, ins_p, outs_p = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, q=q_set)
            terms = {l: v30._terms(model.transformer.h[l].mlp, ins_b[l], ins_p[l]) for l in layers}
            captured[a] = {"w": {l: ins_p[l] - ins_b[l] for l in layers}, "dM": {l: outs_p[l] - outs_b[l] for l in layers},
                           "cross": {l: terms[l][0] for l in layers}, "self": {l: terms[l][1] for l in layers}}
            q = dict(q_set)
            for key in g.blocks_of(stack_mlps):
                q[key] = torch.eye(g.N_EMBD, device=backend.device)
            recs = {}
            for arm, add in (("frozen", None), ("cross", "cross"), ("self", "self"), ("total", "dM")):
                c = dict(cache)
                for l, m in zip(layers, stack_mlps):
                    for i, rid in enumerate(rids):
                        c[(rid, m)] = outs_b[l][i] + (captured[a][add][l][i] if add else 0.0)
                recs[arm] = v30._capture(backend, prep, layers, units=all_units, donor_cache=c, base_cache=prep.base_cache, q=q)[0]
            per_alpha[a] = {"rec_live": rec_live, "rec": recs,
                            "conv": {k: recs[k] - recs["frozen"] for k in ("cross", "self", "total")},
                            "norm_first": {k: float(captured[a][k][first].norm(dim=1).mean()) for k in ("w", "cross", "self", "dM")}}
            print(name, a, "live %.3f" % rec_live, {k: round(v, 3) for k, v in recs.items()}, "conv", {k: round(v, 3) for k, v in per_alpha[a]["conv"].items()}, flush=True)
        for a in GRID:
            per_alpha[a]["cos_to_1_first"] = {k: _cos(torch, captured[a][k][first], captured[1.0][k][first]) for k in ("w", "cross", "self", "dM")}
            per_alpha[a]["cos_to_1_last"] = {k: _cos(torch, captured[a][k][layers[-1]], captured[1.0][k][layers[-1]]) for k in ("w", "cross", "self", "dM")}
        slopes = {k: _slope(GRID, [per_alpha[a]["conv"][k] for a in GRID]) for k in ("cross", "self", "total")}
        slopes.update({f"norm_first_{k}": _slope(GRID, [per_alpha[a]["norm_first"][k] for a in GRID]) for k in ("w", "cross", "self", "dM")})
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "per_alpha": per_alpha, "slopes": slopes,
                        "instrument_max": max(abs(per_alpha[a]["rec"]["total"] - per_alpha[a]["rec_live"]) for a in GRID),
                        "conv_total_1_vs_v32": per_alpha[1.0]["conv"]["total"] - V32_CONV[name]}
        print(name, "slopes", {k: (round(v, 2) if v is not None else None) for k, v in slopes.items()},
              "cos_first", {a: {k: round(v, 3) for k, v in per_alpha[a]["cos_to_1_first"].items()} for a in GRID[:-1]}, flush=True)

    def sl(n, k):
        return report[n]["slopes"][k]
    def inband(v, band):
        return v is not None and band[0] <= v <= band[1]
    predictions = {
        'pred_a_instrument': all(report[n]["instrument_max"] <= INSTR_TOL and abs(report[n]["conv_total_1_vs_v32"]) <= INSTR_TOL for n in SETS),
        'pred_b_total_quadratic': all(inband(sl(n, "total"), QUAD) for n in EARLY),
        'pred_c_cross_linear': all(inband(sl(n, "cross"), LIN) for n in EARLY),
        'pred_d_write_no_rotation': all(report[n]["per_alpha"][a]["cos_to_1_first"]["w"] >= COS_KEEP for n in ("polarity_licensing", "quantifier_number") for a in GRID),
        'pred_e_response_rotates': all(report[n]["per_alpha"][0.25]["cos_to_1_first"]["dM"] <= COS_ROT for n in EARLY),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_channel_scaling_result_v1",
              "candidate_id": "corpus.unit_channel_scaling_v33", "grid": list(GRID),
              "bars": {"instrument": INSTR_TOL, "quadratic": list(QUAD), "linear": list(LIN), "cos_keep": COS_KEEP, "cos_rotate": COS_ROT},
              "slopes": {n: report[n]["slopes"] for n in SETS}, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "slopes": result["slopes"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
