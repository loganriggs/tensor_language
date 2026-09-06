#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the fixed vector, the three linearisation arms and the readout repair fixed before the run.
"""v46: complete the downstream linearisation in x and decompose the floor gate exactly.

v43-v45: with downstream MLPs linearised in u = rms_norm(x) and attention patterns frozen, a floor of 0.22 / 0.10 / 0.59
of the gated readout b survives (polarity / quantifier / voice); attention patterns carry none of it, and a LINEAR
offline readout still shows 67-75% of it, so it is carried by the residual stream itself. The nonlinearities still
live on the path were rms_norm's scale at every block input (the MLP path was linear in u, not in x; the value path
c_v(rms_norm(live)) was untouched) and the final readout. v45's offline readout also applied the tanh cap to the margin
instead of per logit (repaired here: 30 tanh(a/30) - 30 tanh(f/30)). Arms (four runs each, alpha = 1, iso design):
    F0  MLP linear in u, patterns frozen                     (= v44's floor)
    F1  MLP linear in x: u_lin = x / rms(x_B); out = M(u_B) + bil(u_B, u_lin - u_B); patterns frozen
    F2  F1 + value path linear in live: c_v output scaled by ||live|| / ||live_B|| (live = l0 x + l1 x0 recomputed from
        the captured residual, the replayed MLP output and the token embedding)
Offline readouts of the captured final residual: FULL (per-logit tanh, live rms), LINEAR (base scale, no tanh).
Executable check: under F2 every map from the write and the vector to the final residual is first order, so the LINEAR
offline interaction must vanish. Decomposition of the floor gate I_F0: block-input rms curvature on the MLP path
(I_F0 - I_F1), on the value path (I_F1 - I_F2), final readout curvature (I_F2 - I_lin(F2) = I_F2 if the check holds).

REGISTERED BEFORE THE RUN
    pred_a_instrument         repaired FULL offline rec matches the run rec within 1e-3 on every run of F0, all three.
                              Worked: 1e-5 True; v45's 0.04 offset False.
    pred_b_full_linear_zero   |I_lin(F2)| <= 0.05 |I_F0| on all three (the linearisation is complete).
                              Worked: 0.01 True; 0.30 False.
    pred_c_block_rms_carries  (I_F0 - I_F2) / I_F0 >= 0.50 on polarity and voice. Worked: 0.70 True; 0.20 False.
    pred_d_readout_minority   |I_F2| <= 0.40 |I_F0| on polarity and voice (v45's linear readouts imply 0.25 / 0.33).
                              Worked: 0.25 True; 0.60 False.
    pred_e_mlp_path_dominant  (I_F0 - I_F1) >= 0.60 (I_F0 - I_F2) on polarity and voice. Worked: 0.80 True; 0.40 False.
    Reading rule. b True: every surviving piece of the gate is accounted for by named nonlinearities; the circuit's
    genuine products are the MLP-formed ones (v42/v43) and the rest is rms_norm scale coupling -- an artefact of the
    margin readout under a fixed write, not a computation the model performs on the write.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_product_expansion_v42 as v42
import run_unit_downstream_linearisation_v43 as v43
import run_unit_attention_product_locus_v44 as v44

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_full_linearisation_v46_result.json"
SETS, STACK, N_LAYERS, D = v35.SETS, v35.STACK, 18, 1152
INST_TOL, ZERO_MAX, RMS_MIN, READ_MAX, MLP_MIN = 1e-3, 0.05, 0.50, 0.40, 0.60
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 80, 2600


def _plan():
    return {"candidate_id": "corpus.unit_full_linearisation_v46", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


@contextmanager
def _xlinear(torch, model, layers, positions, device, rids, resid_live, x_base, out_base, prev_out_base, x0, vpath, mlp_store):
    """MLP linear in x (all `layers`); optionally the value path linear in live. resid_live is the run's capture dict,
    filled by forward_units before each block's MLP; mlp_store receives the replayed MLP outputs per layer."""
    idx = torch.arange(len(positions), device=device)
    pos = torch.tensor(positions, device=device)
    handles = []
    sqrt_d = D ** 0.5

    def live_norm(l, x_prev):
        lam = model.transformer.h[l].lambdas
        return (lam[0] * x_prev + lam[1] * x0).norm(dim=1, keepdim=True)

    def mk_mlp(l):
        mlp = model.transformer.h[l].mlp
        xb = x_base[l]
        sb = sqrt_d / xb.norm(dim=1, keepdim=True)
        ub = xb * sb

        def hook(m, a, o):
            x = torch.stack([resid_live[(rid, l)] for rid in rids])
            new = out_base[l] + v42._bil(mlp, ub, x * sb - ub)
            mlp_store[l] = new
            y = o.clone()
            y[idx, pos] = new.to(y.dtype)
            return y
        return hook

    def mk_cv(l):
        xb_prev = x_base[l - 1] + prev_out_base[l - 1]
        nb = live_norm(l, xb_prev)

        def hook(m, a, o):
            x_prev = torch.stack([resid_live[(rid, l - 1)] for rid in rids]) + (mlp_store[l - 1] if (l - 1) in mlp_store else prev_out_base[l - 1])
            ratio = live_norm(l, x_prev) / nb
            y = o.clone()
            y[idx, pos] = (y[idx, pos].float() * ratio).to(y.dtype)
            return y
        return hook
    for l in layers:
        handles.append(model.transformer.h[l].mlp.register_forward_hook(mk_mlp(l)))
        if vpath:
            handles.append(model.transformer.h[l].attn.c_v.register_forward_hook(mk_cv(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


def _readout(torch, x, wa, wf, mode, scale_b):
    u = x * scale_b if mode == "linear" else x * (D ** 0.5) / x.norm(dim=1, keepdim=True)
    a, f = (u * wa).sum(dim=1), (u * wf).sum(dim=1)
    if mode == "full":
        a, f = 30.0 * torch.tanh(a / 30.0), 30.0 * torch.tanh(f / 30.0)
    return -(a - f)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    F = torch.nn.functional
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
        x_base = {l: torch.stack([resid_b[(rid, l)] for rid in rids]) for l in range(N_LAYERS)}
        prev_out_base = dict(out_base)
        prev_out_base[last] = outs_b[last]
        tokens = torch.tensor([prep.base_batch.token_rows[i][positions[i]] for i in range(len(rids))], device=backend.device)
        x0 = F.rms_norm(model.transformer.wte(tokens).float(), (D,))
        wa = W[torch.tensor(prep.base_batch.answer_ids, device=W.device)]
        wf = W[torch.tensor(prep.base_batch.foil_ids, device=W.device)]
        pf = {l: None for l in down}

        def four(arm):
            recs, xf = {}, {}
            for key, (w_, vec) in RUNS.items():
                resid, store = {}, {}
                if arm == "F0":
                    ctx = v43._linearised(torch, model, down, positions, backend.device, u_base, out_base)
                else:
                    ctx = _xlinear(torch, model, down, positions, backend.device, rids, resid, x_base, out_base, prev_out_base, x0, arm == "F2", store)
                with ctx, v44._qk_frozen(model, qk, pf):
                    rec, _, outs = v30._capture(backend, prep, down, **cfg(w_, vec, resid))
                recs[key] = rec
                xf[key] = torch.stack([resid[(rid, N_LAYERS - 1)] for rid in rids]) + outs[N_LAYERS - 1]
            return recs, xf

        def inter(r):
            return r["DV"] - r["D"] - r["V"] + r["B"]
        arms = {}
        for arm in ("F0", "F1", "F2"):
            recs, xf = four(arm)
            scale_b = (D ** 0.5) / xf["B"].norm(dim=1, keepdim=True)
            off = {mode: {k: v25._rec(prep, _readout(torch, xf[k], wa, wf, mode, scale_b).tolist()) for k in RUNS} for mode in ("full", "linear")}
            arms[arm] = {"rec_run": recs, "i_run": inter(recs), "rec_offline_full": off["full"], "i_offline_full": inter(off["full"]),
                         "i_offline_linear": inter(off["linear"]),
                         "instrument_max_err": max(abs(off["full"][k] - recs[k]) for k in RUNS)}
        i0, i1, i2 = arms["F0"]["i_run"], arms["F1"]["i_run"], arms["F2"]["i_run"]
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows": len(prep.rows), "arms": arms,
                        "i_F0": i0, "i_F1": i1, "i_F2": i2, "i_lin_F2": arms["F2"]["i_offline_linear"],
                        "share_mlp_path_rms": (i0 - i1) / i0 if i0 else None, "share_value_path_rms": (i1 - i2) / i0 if i0 else None,
                        "share_readout": i2 / i0 if i0 else None, "share_unexplained": arms["F2"]["i_offline_linear"] / i0 if i0 else None}
        print(name, "I F0 %.4f F1 %.4f F2 %.4f lin(F2) %.5f | inst %.1e | shares mlp %.2f v %.2f readout %.2f" % (
            i0, i1, i2, arms["F2"]["i_offline_linear"], arms["F0"]["instrument_max_err"],
            report[name]["share_mlp_path_rms"], report[name]["share_value_path_rms"], report[name]["share_readout"]), flush=True)

    pv = ("polarity_licensing", "voice_frame")
    predictions = {
        'pred_a_instrument': all(report[n]["arms"]["F0"]["instrument_max_err"] <= INST_TOL for n in SETS),
        'pred_b_full_linear_zero': all(abs(report[n]["i_lin_F2"]) <= ZERO_MAX * abs(report[n]["i_F0"]) for n in SETS),
        'pred_c_block_rms_carries': all((report[n]["i_F0"] - report[n]["i_F2"]) / report[n]["i_F0"] >= RMS_MIN for n in pv),
        'pred_d_readout_minority': all(abs(report[n]["i_F2"]) <= READ_MAX * abs(report[n]["i_F0"]) for n in pv),
        'pred_e_mlp_path_dominant': all((report[n]["i_F0"] - report[n]["i_F1"]) >= MLP_MIN * (report[n]["i_F0"] - report[n]["i_F2"]) for n in pv),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_full_linearisation_result_v1", "candidate_id": "corpus.unit_full_linearisation_v46",
              "bars": {"inst_tol": INST_TOL, "zero_max": ZERO_MAX, "rms_min": RMS_MIN, "read_max": READ_MAX, "mlp_min": MLP_MIN},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
