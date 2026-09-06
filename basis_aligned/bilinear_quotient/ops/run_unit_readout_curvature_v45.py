#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the fixed vector, the floor arm and the readout decomposition fixed before the run.
"""v45: the gate that survives MLP linearisation AND attention-pattern freezing -- is it the final readout's curvature?

v43/v44: with every downstream MLP first-order and every downstream attention pattern frozen to base, 0.22 (polarity)
/ 0.10 (quantifier) / 0.59 (voice) of the gated readout b remains, and the attention patterns carry none of it. The
nonlinearities left on the path are rms_norm's scale (at every block input and at the final readout) and the tanh
soft cap on the logits. The margin read from the final residual x_f is m(x_f) = 30 tanh(w . rms_norm(x_f) / 30) with
w = W_lm[answer] - W_lm[foil]; it is NOT linear in x_f, so a write Delta_f and a vector v_f that are each first-order
in the residual still give a mixed term -- the readout itself is a gate. Design: floor arm as v44 (four runs B, D, V,
DV at alpha = 1), capture x_f at the answer position (residual entering mlp:17 plus mlp:17's replayed output).
Offline readouts of the same four x_f: FULL m(x); RMS-only (tanh removed); LINEAR (rms scale fixed to base, tanh
removed). Interactions in recovery units I_full, I_rms, I_lin; residual-stream interaction iota_f = x_DV - x_D - x_V
+ x_B; autograd mixed derivative d2 m / d alpha d beta at x_B along (Delta_f, v_f) as the analytic check.

REGISTERED BEFORE THE RUN
    pred_a_offline_matches   |I_full(offline) - I_floor(run)| <= 1e-3 on all three (instrument). Worked: 1e-5 True.
    pred_b_readout_is_gate   |I_lin| <= 0.25 |I_floor| on polarity and voice: linearising the readout removes >= 75%
                             of the surviving gate. Worked: 0.10 True; 0.60 False.
    pred_c_rms_not_tanh      |I_rms - I_floor| <= 0.25 |I_floor| on polarity and voice. Worked: 0.05 True; 0.40 False.
    pred_d_stream_first_order ||iota_f|| / ||Delta_f|| <= 0.10 (row mean) on all three. Worked: 0.03 True; 0.30 False.
    pred_e_analytic          |d2 (autograd, recovery units) - I_full| <= 0.25 |I_full| on polarity and voice.
                             Worked: 0.08 True; 0.5 False (higher orders would then matter at alpha = beta = 1).
    Reading rule. b, c, d True: the whole residual gate is the final rms_norm's scale coupling -- the write shrinks or
    grows the readout of the converted vector by changing ||x_f||; a readout artefact of the margin, not a circuit
    computation. Then the MLP-formed products (v42/v43) are the only genuine circuit products.
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
import run_unit_downstream_linearisation_v43 as v43
import run_unit_attention_product_locus_v44 as v44

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_readout_curvature_v45_result.json"
SETS, STACK, N_LAYERS, D = v35.SETS, v35.STACK, 18, 1152
INST_TOL, LIN_MAX, RMS_TOL, IOTA_MAX, AN_TOL = 1e-3, 0.25, 0.25, 0.10, 0.25
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_readout_curvature_v45", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _readout(torch, x, w, mode, scale_b):
    """Margin on the donor-oriented axis (-(answer - foil)) from the final residual x (n, D)."""
    if mode == "linear":
        u = x * scale_b
    else:
        u = x * (D ** 0.5) / x.norm(dim=1, keepdim=True)
    m = (u * w).sum(dim=1)
    if mode == "full":
        m = 30.0 * torch.tanh(m / 30.0)
    return -m


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    t0 = time.perf_counter()
    W = model.lm_head.weight.float()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        positions = list(prep.base_batch.semantic_positions)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first, last = layers[0], layers[-1]
        down = list(range(last + 1, N_LAYERS))
        mlp_first = model.transformer.h[first].mlp
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
        v = v38._cross(mlp_first, ins_b[first], ins_p1[first] - ins_b[first])

        def cfg(write, vec, resid):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
            kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache, "capture_resid": resid}
            if write is not None:
                kw["resid_add"] = {first: write}
            return kw

        RUNS = {"B": (None, None), "D": (delta1, None), "V": (None, v), "DV": (delta1, v)}
        qk, handles = v35._capture_qk(model)
        try:
            _, u_base, out_base = v30._capture(backend, prep, down, **cfg(None, None, {}))
        finally:
            for h in handles:
                h.remove()
        recs, xf = {}, {}
        for key, (w_, vec) in RUNS.items():
            resid = {}
            with v43._linearised(torch, model, down, positions, backend.device, u_base, out_base), v44._qk_frozen(model, qk, {l: None for l in down}):
                rec, _, outs = v30._capture(backend, prep, down, **cfg(w_, vec, resid))
            recs[key] = rec
            xf[key] = torch.stack([resid[(rid, N_LAYERS - 1)] for rid in rids]) + outs[N_LAYERS - 1]
        i_floor = recs["DV"] - recs["D"] - recs["V"] + recs["B"]
        w = W[torch.tensor(prep.base_batch.answer_ids, device=W.device)] - W[torch.tensor(prep.base_batch.foil_ids, device=W.device)]
        scale_b = (D ** 0.5) / xf["B"].norm(dim=1, keepdim=True)

        def inter(mode):
            r = {k: v25._rec(prep, _readout(torch, xf[k], w, mode, scale_b).tolist()) for k in RUNS}
            return r["DV"] - r["D"] - r["V"] + r["B"], r
        i_full, r_full = inter("full")
        i_rms, _ = inter("rms")
        i_lin, _ = inter("linear")
        delta_f, v_f = xf["D"] - xf["B"], xf["V"] - xf["B"]
        iota_f = xf["DV"] - xf["D"] - xf["V"] + xf["B"]
        iota_ratio = float((iota_f.norm(dim=1) / delta_f.norm(dim=1).clamp_min(1e-12)).mean())
        # autograd mixed derivative of the full readout at x_B along (delta_f, v_f), per row, in recovery units
        xb = xf["B"].detach()
        alpha = torch.zeros((), device=xb.device, requires_grad=True)
        beta = torch.zeros((), device=xb.device, requires_grad=True)
        m = _readout(torch, xb + alpha * delta_f + beta * v_f, w, "full", scale_b)
        d2 = []
        for i in range(xb.shape[0]):
            ga, = torch.autograd.grad(m[i], alpha, create_graph=True)
            gab, = torch.autograd.grad(ga, beta, retain_graph=True)
            d2.append(float(gab))
        d2_rec = sum(x / (d - b) for x, b, d in zip(d2, prep.base_axis, prep.donor_axis)) / len(d2)
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows": len(prep.rows),
                        "rec_floor": recs, "i_floor_run": i_floor, "i_full_offline": i_full, "i_rms": i_rms, "i_linear": i_lin,
                        "offline_full_recs": r_full, "iota_over_delta_final": iota_ratio, "d2_autograd_rec": d2_rec,
                        "delta_f_norm_mean": float(delta_f.norm(dim=1).mean()), "v_f_norm_mean": float(v_f.norm(dim=1).mean()),
                        "xb_norm_mean": float(xf["B"].norm(dim=1).mean()),
                        "cos_delta_v_final": float(torch.nn.functional.cosine_similarity(delta_f, v_f, dim=1).mean())}
        print(name, "I_floor %.4f offline %.4f rms %.4f lin %.4f | iota/delta %.3f | d2 %.4f | cos(dv) %.2f" % (
            i_floor, i_full, i_rms, i_lin, iota_ratio, d2_rec, report[name]["cos_delta_v_final"]), flush=True)

    pv = ("polarity_licensing", "voice_frame")
    predictions = {
        'pred_a_offline_matches': all(abs(report[n]["i_full_offline"] - report[n]["i_floor_run"]) <= INST_TOL for n in SETS),
        'pred_b_readout_is_gate': all(abs(report[n]["i_linear"]) <= LIN_MAX * abs(report[n]["i_floor_run"]) for n in pv),
        'pred_c_rms_not_tanh': all(abs(report[n]["i_rms"] - report[n]["i_floor_run"]) <= RMS_TOL * abs(report[n]["i_floor_run"]) for n in pv),
        'pred_d_stream_first_order': all(report[n]["iota_over_delta_final"] <= IOTA_MAX for n in SETS),
        'pred_e_analytic': all(abs(report[n]["d2_autograd_rec"] - report[n]["i_full_offline"]) <= AN_TOL * abs(report[n]["i_full_offline"]) for n in pv),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_readout_curvature_result_v1", "candidate_id": "corpus.unit_readout_curvature_v45",
              "bars": {"inst_tol": INST_TOL, "lin_max": LIN_MAX, "rms_tol": RMS_TOL, "iota_max": IOTA_MAX, "an_tol": AN_TOL},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
