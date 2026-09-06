#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grid and the w1 / rest decomposition fixed before the run.
"""v38: the cross term of a lone stack layer is exactly linear in the normalized write w, yet under a LINEAR pre-norm
write its conversion scales as alpha^1.60 on polarity mlp:08 (v37; quantifier 1.15, voice 1.18 are linear), while
||w|| is linear (slope 1.016). Then the DIRECTION of w(alpha) = rms_norm(x_b + alpha Delta) - rms_norm(x_b) must
rotate with alpha. Its first order w1 = sqrt(D)/||x|| (Delta - xhat (xhat.Delta)) is linear; the second order carries
xhat (3 (xhat.Delta)^2 - ||Delta||^2) / (2 ||x||): the normalization SHRINKS the base input u_b by
alpha^2 ||Delta_perp||^2 / (2 ||x||^2) -- a multiplicative gain control on the layer's whole base computation
(bilinear: output scales as (1 - s)^2), margin-relevant wherever the base MLP output carries the base-side margin.
v34 judged w2 inert (2-6%) -- under the REAL write, on the pull-back at alpha=1, pooled over the stack. This is the
direct test on the lone layer under the linear write: w(alpha) exact, rest(alpha) = w(alpha) - alpha w1(1), replays at
the first stack layer (others base): beta cross(w(1)) for the readout, cross(alpha w1), cross(rest), cross(w2 closed
form), self(w). conv = rec(replay) - rec(all base) at the same alpha; slopes log-log over {0.25, 0.5, 0.75, 1}.

REGISTERED BEFORE THE RUN
    pred_a_readout_linear   beta-replay slope of cross_first(1) in [0.95, 1.05] on all three. Worked: 1.01 True.
    pred_b_rest_carries     share of cross(rest) in cross(w) at alpha = 1 >= 0.30 on polarity, and its slope >= 1.7.
                            Worked: 0.55 / 1.95 True; 0.10 False (then rest is not where the 1.60 lives).
    pred_c_second_order     |cross(w2) - cross(rest)| <= 0.25 |cross(rest)| at alpha = 1 on all three (the closed-form
                            second order IS the rest). Worked: rest 0.030, w2 0.027 True; w2 0.010 False.
    pred_d_shrink_direction cos(rest(1), -u_b) >= 0.9 on all three, row mean. Worked: 0.95 True; 0.5 False.
    pred_e_polarity_special the rest share on polarity is >= 2x the rest share on quantifier AND on voice (why only
                            polarity's lone layer is superlinear: its base mlp:08 output is margin-relevant). Worked:
                            0.55 vs 0.15 / 0.12 True; 0.55 vs 0.40 False.
    Reading rule. a-d True: the alpha^2 of the linear-write conversion is gain control -- the write damps the base MLP
    computation through the shared rms_norm, quadratically in the write; e says whether that is a per-behaviour
    property of what the base layer computes. c False & b True: rest is higher than second order (large-write regime,
    ||Delta|| ~ ||x||); fit the exact shrink factor instead. b False: the rotation of w is not in the normalization --
    check the captured input against the replayed residual (instrument).
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
import run_unit_rmsnorm_shrinkage_v34 as v34
import run_unit_pattern_freeze_v35 as v35

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_norm_gain_control_v38_result.json"
SETS, STACK, EARLY, GRID = v35.SETS, v35.STACK, v35.EARLY, v35.GRID
READ_BAND, REST_SHARE_BAR, REST_SLOPE_BAR, W2_TOL, COS_BAR, SPECIAL_RATIO = (0.95, 1.05), 0.30, 1.7, 0.25, 0.9, 2.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 4000


def _plan():
    return {"candidate_id": "corpus.unit_norm_gain_control_v38", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _cross(mlp, u_b, w):
    WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()
    return ((u_b @ WL.T) * (w @ WR.T) + (w @ WL.T) * (u_b @ WR.T)) @ WD.T


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
        first, m_first = layers[0], stack_mlps[0]
        mlp = model.transformer.h[first].mlp
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        x_b = torch.stack([resid_b[(rid, first)] for rid in rids])
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        u_b = ins_b[first]
        w1_full, _ = v34._expansion(torch, x_b, delta1)
        # instrument: the captured normalized input equals rms_norm of the captured pre-norm residual
        instr = float(((torch.nn.functional.rms_norm(x_b, (x_b.shape[1],)) - u_b).norm(dim=1) / u_b.norm(dim=1)).max())

        def replay(alpha, first_value):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (first_value if l == first else outs_b[l])[i]
            return v30._capture(backend, prep, layers, units=stack_mlps, donor_cache=c, base_cache=prep.base_cache,
                                resid_add={first: alpha * delta1})[0]

        # readout: beta replay of the alpha = 1 cross term
        _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
        cross1 = _cross(mlp, u_b, ins_p1[first] - u_b)
        per = {}
        for a in GRID:
            _, ins_p, _ = v30._capture(backend, prep, layers, resid_add={first: a * delta1})
            w = ins_p[first] - u_b
            rest = w - a * w1_full
            _, w2 = v34._expansion(torch, x_b, a * delta1)
            base = replay(a, outs_b[first])
            r = {"cross_w": replay(a, outs_b[first] + _cross(mlp, u_b, w)),
                 "cross_w1": replay(a, outs_b[first] + _cross(mlp, u_b, a * w1_full)),
                 "cross_rest": replay(a, outs_b[first] + _cross(mlp, u_b, rest)),
                 "cross_w2": replay(a, outs_b[first] + _cross(mlp, u_b, w2)),
                 "self_w": replay(a, outs_b[first] + v30._terms(mlp, u_b, ins_p[first])[1]),
                 "beta_cross1": replay(a, outs_b[first] + a * cross1) if a != 1.0 else replay(a, outs_b[first] + cross1)}
            # beta replay must run on the alpha=1 residual to isolate the readout; redo with the write fixed at 1
            r["beta_cross1"] = None
            conv = {k: v - base for k, v in r.items() if v is not None}
            per[a] = {"conv": conv, "rest_norm_over_w": float((rest.norm(dim=1) / w.norm(dim=1)).mean()),
                      "cos_rest_minus_ub": float(torch.nn.functional.cosine_similarity(rest, -u_b, dim=1).mean()),
                      "cos_w2_rest": float(torch.nn.functional.cosine_similarity(w2, rest, dim=1).mean()),
                      "w2_norm_over_rest": float((w2.norm(dim=1) / rest.norm(dim=1).clamp_min(1e-12)).mean()),
                      "shrink_factor_pred": float((a * a * (delta1 - x_b * ((x_b * delta1).sum(1, keepdim=True) / (x_b * x_b).sum(1, keepdim=True))).pow(2).sum(1) / (2 * (x_b * x_b).sum(1))).mean())}
            print(name, a, {k: round(v, 4) for k, v in conv.items()}, "cos(rest,-u_b)", round(per[a]["cos_rest_minus_ub"], 3), flush=True)
        # readout beta-replay: write fixed at alpha = 1, first layer output = base + beta cross1
        base1 = replay(1.0, outs_b[first])
        beta = {b: replay(1.0, outs_b[first] + b * cross1) - base1 for b in GRID}
        slopes = {k: v35._slope(GRID, [per[a]["conv"][k] for a in GRID]) for k in per[GRID[0]]["conv"]}
        slopes["beta_readout"] = v35._slope(GRID, [beta[b] for b in GRID])
        c1 = per[1.0]["conv"]
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "first": m_first, "rows": len(prep.rows), "instrument_rel_err": instr,
                        "per_alpha": per, "beta_readout": beta, "slopes": slopes,
                        "rest_share_1": c1["cross_rest"] / c1["cross_w"] if c1["cross_w"] else None,
                        "w1_share_1": c1["cross_w1"] / c1["cross_w"] if c1["cross_w"] else None,
                        "w2_vs_rest_1": abs(c1["cross_w2"] - c1["cross_rest"]) / abs(c1["cross_rest"]) if c1["cross_rest"] else None,
                        "cos_rest_minus_ub_1": per[1.0]["cos_rest_minus_ub"], "x_norm_mean": float(x_b.norm(dim=1).mean()),
                        "delta_norm_mean": float(delta1.norm(dim=1).mean())}
        print(name, "slopes", {k: round(s, 2) if s is not None else s for k, s in slopes.items()}, "rest_share", round(report[name]["rest_share_1"], 3),
              "w2_vs_rest", round(report[name]["w2_vs_rest_1"], 3), "instr", "%.1e" % instr, flush=True)

    def inband(v, band):
        return v is not None and band[0] <= v <= band[1]
    rs = {n: report[n]["rest_share_1"] for n in SETS}
    predictions = {
        'pred_a_readout_linear': all(inband(report[n]["slopes"]["beta_readout"], READ_BAND) for n in SETS),
        'pred_b_rest_carries': rs["polarity_licensing"] is not None and rs["polarity_licensing"] >= REST_SHARE_BAR
                               and report["polarity_licensing"]["slopes"]["cross_rest"] is not None
                               and report["polarity_licensing"]["slopes"]["cross_rest"] >= REST_SLOPE_BAR,
        'pred_c_second_order': all(report[n]["w2_vs_rest_1"] is not None and report[n]["w2_vs_rest_1"] <= W2_TOL for n in SETS),
        'pred_d_shrink_direction': all(report[n]["cos_rest_minus_ub_1"] >= COS_BAR for n in SETS),
        'pred_e_polarity_special': all(v is not None for v in rs.values()) and
                                   all(rs["polarity_licensing"] >= SPECIAL_RATIO * rs[n] for n in ("quantifier_number", "voice_frame")),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_norm_gain_control_result_v1",
              "candidate_id": "corpus.unit_norm_gain_control_v38", "grid": list(GRID),
              "bars": {"read_band": list(READ_BAND), "rest_share": REST_SHARE_BAR, "rest_slope": REST_SLOPE_BAR, "w2_tol": W2_TOL, "cos": COS_BAR, "special_ratio": SPECIAL_RATIO},
              "slopes": {n: report[n]["slopes"] for n in SETS}, "rest_share_1": rs, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "slopes": result["slopes"], "rest_share_1": rs, "seconds": round(result["seconds"], 1)}, indent=2, default=str))


if __name__ == "__main__":
    main()
