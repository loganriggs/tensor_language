#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the readout gradient point and the rms_norm expansion fixed before the run.
# Re-run note: first run's gradient was taken of the MEAN recovery (1/n too small, pred_a failed by exactly n=32); fixed to per-row sum.
"""v34 (Tier 4 candidate): the quadratic conversion is rms_norm SHRINKAGE of the stack MLPs' own base output.

Chain. v29 conversion ~ alpha^2. v30 the stack's response is 72-79% cross-term (linear in the normalized-input delta
w). v31 the margin reads the stack output linearly; |w| linear. v32 no cross-layer product. v33 the cross channel's
MARGIN effect is superlinear (2.31 / 1.68 / 1.95) while its norm is linear, and for quantifier w does not even rotate
(cos 0.993) -- so the margin reads a component of w that is second order in the write and small in norm.
Candidate: rms_norm. With x the pre-norm residual entering a stack MLP and Delta the write's residual delta there,
    rms_norm(x + a Delta) - rms_norm(x) = a w1 + a^2 w2 + O(a^3),
    w1 = sqrt(D)/|x| (Delta - xhat (xhat.Delta)),  w2 = sqrt(D)/|x| [ -(xhat.Delta)/|x| Delta + xhat (3 (xhat.Delta)^2 - |Delta|^2) / (2|x|) ].
w2 is (for Delta roughly orthogonal to x) -|Delta|^2/(2|x|^2) u_b: a SHRINKAGE of the base's own normalized state.
The cross channel on -eps u_b is exactly -2 eps (M(u_b) - bias): the write turns the stack MLP's own base output
down by the write's relative energy |Delta|^2/|x|^2. Quadratic, closed-form, weight-explicit.
Instruments: r_l = d rec / d(stack output_l) by autograd at (write = 1, stack = base); g_l = J_cross^T r_l pulls the
readout back to the MLP input through the exact cross-term Jacobian W_D[diag(L u_b) W_R + diag(R u_b) W_L].

REGISTERED BEFORE THE RUN
    pred_a_readout_linear   |sum_l mean_i <r_l, DeltaM_l(alpha)> - conv_total(alpha)| <= 0.02 at every alpha in
                            {0.25, 0.5, 0.75, 1} on all three sets (conv_total from v33). Worked: 0.318 vs 0.31 True.
    pred_b_expansion        |w(1) - w1 - w2| / |w(1)| <= 0.10 at the first stack layer on all three. Worked: 0.05 True.
    pred_c_linear_inert     |sum_l mean <g_l, w1_l>| <= 0.35 |sum_l mean <g_l, w_l(1)>| on all three: the first-order
                            (linear) part of the write is margin-inert through the cross channel. Worked: 0.2 True; 0.6 False.
    pred_d_shrinkage        mean_i cos(w_l(1) - w1_l, u_b) <= -0.70 at the first stack layer on all three (the measured
                            remainder is anti-aligned with the base state). Worked: -0.85 True; -0.4 False.
    pred_e_closed_form      sum_l mean <g_l, w2_l> / conv_cross(1) in [0.6, 1.4] for polarity AND quantifier (voice's
                            multi-layer cascade, w cos 0.74, is reported not registered). Worked: 0.9 True; 0.4 False.
    Reading rule. all True: Tier 4 for the converter -- conversion = -(|Delta_l|^2/|x_l|^2) <r_l, M_l(u_b) - bias>
    summed over the stack, an exact weight statement given the write; remaining: the self channel (24-29%) and the
    write's own algebra inside the set's heads. c False: the linear part matters and the superlinearity needs the
    intermediate layers (cascade), expand layer-by-layer.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_rmsnorm_shrinkage_v34_result.json"
V33 = ROOT / "circuits/followups/unit_channel_scaling_v33_result.json"
SETS, STACK = v30.SETS, v30.STACK
GRID = (0.25, 0.5, 0.75, 1.0)
READ_TOL, EXP_TOL, INERT_BAR, SHRINK_BAR, CF_BAND = 0.02, 0.10, 0.35, -0.70, (0.6, 1.4)
CF_SETS = ("polarity_licensing", "quantifier_number")
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 1500


def _plan():
    return {"candidate_id": "corpus.unit_rmsnorm_shrinkage_v34", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 3, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _readout_grad(backend, prep, units, stack_mlps, layers, outs_b, cache):
    """r_l = d rec / d(stack MLP output_l at the row position), at write = 1 (exact set), stack = base."""
    torch = backend.torch
    rids = list(prep.base_batch.row_ids)
    Z = {l: torch.zeros_like(outs_b[l], requires_grad=True) for l in layers}
    c = dict(cache)
    for l, m in zip(layers, stack_mlps):
        for i, rid in enumerate(rids):
            c[(rid, m)] = outs_b[l][i] + Z[l][i]
    out = g.forward_units(backend, prep.base_batch, units=list(units) + stack_mlps, donor_cache=c, base_cache=prep.base_cache, grad=True)
    p = -(out[:, 0] - out[:, 1])
    b = torch.tensor(prep.base_axis, device=p.device, dtype=p.dtype)
    d = torch.tensor(prep.donor_axis, device=p.device, dtype=p.dtype)
    per_row = (p - b) / (d - b)
    per_row.sum().backward()          # per-row gradient; the mean over rows is taken by the caller's mean_i <r, .>
    return {l: Z[l].grad.detach().clone() for l in layers}, float(per_row.mean())


def _pullback(mlp, u_b, r):
    """g = J_cross^T r with J_cross = W_D [diag(L u_b) W_R + diag(R u_b) W_L]."""
    WL, WR, WD = mlp.Left.weight.float(), mlp.Right.weight.float(), mlp.Down.weight.float()
    Lb, Rb = u_b @ WL.T, u_b @ WR.T
    t = r @ WD                                   # (n, 4608) = W_D^T r per row
    return (Lb * t) @ WR + (Rb * t) @ WL          # (n, D)


def _expansion(torch, x, delta):
    D = x.shape[1]
    xn = x.norm(dim=1, keepdim=True)
    xhat = x / xn
    xd = (xhat * delta).sum(dim=1, keepdim=True)
    s = D ** 0.5 / xn
    w1 = s * (delta - xhat * xd)
    w2 = s * (-(xd / xn) * delta + xhat * (3 * xd ** 2 - (delta ** 2).sum(dim=1, keepdim=True)) / (2 * xn))
    return w1, w2


def _dot_mean(a, b):
    return float((a * b).sum(dim=1).mean())


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    v33 = json.loads(V33.read_text())["behaviours"]
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
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        r, rec_at = _readout_grad(backend, prep, units, stack_mlps, layers, outs_b, cache)
        x_b = {l: torch.stack([resid_b[(rid, l)] for rid in rids]) for l in layers}
        u_b = {l: ins_b[l] for l in layers}
        gl = {l: _pullback(model.transformer.h[l].mlp, u_b[l], r[l]) for l in layers}
        base_out = {l: outs_b[l] - model.transformer.h[l].mlp.Down_bias.float() for l in layers}
        per_alpha, w1, w2, delta1 = {}, {}, {}, {}
        for a in GRID:
            rp = {}
            q = v27._scaled_q(backend, units, a)
            _, ins_p, outs_p = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, q=q, capture_resid=rp)
            x_p = {l: torch.stack([rp[(rid, l)] for rid in rids]) for l in layers}
            w = {l: ins_p[l] - u_b[l] for l in layers}
            dM = {l: outs_p[l] - outs_b[l] for l in layers}
            terms = {l: v30._terms(model.transformer.h[l].mlp, u_b[l], ins_p[l]) for l in layers}
            pred_total = sum(_dot_mean(r[l], dM[l]) for l in layers)
            pred_cross = sum(_dot_mean(r[l], terms[l][0]) for l in layers)
            pull_cross = sum(_dot_mean(gl[l], w[l]) for l in layers)
            assert abs(pull_cross - pred_cross) <= 1e-3 * max(1.0, abs(pred_cross)), (pull_cross, pred_cross)
            if a == 1.0:
                delta1 = {l: x_p[l] - x_b[l] for l in layers}
                for l in layers:
                    w1[l], w2[l] = _expansion(torch, x_b[l], delta1[l])
                w_1 = w
            per_alpha[a] = {"_per_layer": {l: _dot_mean(gl[l], w[l]) for l in layers}, "pred_total": pred_total, "pred_cross": pred_cross, "pred_self": sum(_dot_mean(r[l], terms[l][1]) for l in layers),
                            "v33_conv_total": v33[name]["per_alpha"][str(a)]["conv"]["total"], "v33_conv_cross": v33[name]["per_alpha"][str(a)]["conv"]["cross"],
                            "delta_first_cos_to_1": None, "delta_first_norm_ratio": None}
            if a < 1.0:
                per_alpha[a]["_dx"] = x_p[first] - x_b[first]
        for a in GRID[:-1]:
            dx = per_alpha[a].pop("_dx")
            per_alpha[a]["delta_first_cos_to_1"] = float(torch.nn.functional.cosine_similarity(dx, delta1[first], dim=1).mean())
            per_alpha[a]["delta_first_norm_ratio"] = float((dx.norm(dim=1) / (a * delta1[first].norm(dim=1))).mean())
        rem = {l: w_1[l] - w1[l] for l in layers}
        exp_err = {l: float(((w_1[l] - w1[l] - w2[l]).norm(dim=1) / w_1[l].norm(dim=1)).mean()) for l in layers}
        lin_part = sum(_dot_mean(gl[l], w1[l]) for l in layers)
        quad_part = sum(_dot_mean(gl[l], w2[l]) for l in layers)
        full = sum(_dot_mean(gl[l], w_1[l]) for l in layers)
        rel_energy = {l: float(((delta1[l].norm(dim=1) ** 2) / (x_b[l].norm(dim=1) ** 2)).mean()) for l in layers}
        shrink_reading = sum(-_dot_mean(r[l], base_out[l] * ((delta1[l].norm(dim=1) ** 2) / (x_b[l].norm(dim=1) ** 2))[:, None]) for l in layers)
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "rec_at_grad_point": rec_at,
                        "per_alpha": per_alpha, "expansion_rel_err": {f"mlp:{l:02d}": e for l, e in exp_err.items()},
                        "pullback": {"linear_w1": lin_part, "quadratic_w2": quad_part, "full_w1": full,
                                     "linear_share": lin_part / full if full else None, "quadratic_share": quad_part / full if full else None,
                                     "closed_form_over_conv_cross": quad_part / per_alpha[1.0]["v33_conv_cross"]},
                        "remainder_cos_u_b": {f"mlp:{l:02d}": float(torch.nn.functional.cosine_similarity(rem[l], u_b[l], dim=1).mean()) for l in layers},
                        "w2_cos_u_b": {f"mlp:{l:02d}": float(torch.nn.functional.cosine_similarity(w2[l], u_b[l], dim=1).mean()) for l in layers},
                        "relative_write_energy": {f"mlp:{l:02d}": e for l, e in rel_energy.items()},
                        "shrinkage_reading_sum": shrink_reading,
                        "per_layer_quadratic": {f"mlp:{l:02d}": _dot_mean(gl[l], w2[l]) for l in layers},
                        "per_layer_linear": {f"mlp:{l:02d}": _dot_mean(gl[l], w1[l]) for l in layers},
                        "per_layer_full_by_alpha": {str(a): {f"mlp:{l:02d}": v for l, v in per_alpha[a].pop("_per_layer").items()} for a in GRID}}
        print(name, "rec_at %.3f" % rec_at, "pullback", {k: (round(v, 3) if v is not None else None) for k, v in report[name]["pullback"].items()},
              "exp_err", {k: round(v, 3) for k, v in report[name]["expansion_rel_err"].items()},
              "rem_cos", {k: round(v, 3) for k, v in report[name]["remainder_cos_u_b"].items()}, "shrink %.3f" % shrink_reading, flush=True)
        for a in GRID:
            print("   ", a, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in per_alpha[a].items()}, flush=True)

    fk = {n: f"mlp:{g.unit_layer([m for m in STACK[n] if m.startswith('mlp:')][0]):02d}" for n in SETS}
    predictions = {
        'pred_a_readout_linear': all(abs(report[n]["per_alpha"][a]["pred_total"] - report[n]["per_alpha"][a]["v33_conv_total"]) <= READ_TOL for n in SETS for a in GRID),
        'pred_b_expansion': all(report[n]["expansion_rel_err"][fk[n]] <= EXP_TOL for n in SETS),
        'pred_c_linear_inert': all(report[n]["pullback"]["linear_share"] is not None and abs(report[n]["pullback"]["linear_share"]) <= INERT_BAR for n in SETS),
        'pred_d_shrinkage': all(report[n]["remainder_cos_u_b"][fk[n]] <= SHRINK_BAR for n in SETS),
        'pred_e_closed_form': all(CF_BAND[0] <= report[n]["pullback"]["closed_form_over_conv_cross"] <= CF_BAND[1] for n in CF_SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_rmsnorm_shrinkage_result_v1",
              "candidate_id": "corpus.unit_rmsnorm_shrinkage_v34", "grid": list(GRID),
              "bars": {"readout": READ_TOL, "expansion": EXP_TOL, "inert": INERT_BAR, "shrink": SHRINK_BAR, "closed_form": list(CF_BAND)},
              "summary": {n: report[n]["pullback"] for n in SETS}, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "summary": result["summary"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
