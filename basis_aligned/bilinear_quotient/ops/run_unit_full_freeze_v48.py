"""v48: v47 rerun with the pattern freeze REPAIRED -- bilin18's attention is CausalBilinearSelfAttention.

Read from the loaded model after v47: `type(h[l].attn)` is CausalBilinearSelfAttention with pattern = (q.k/D)(q2.k2/D),
q2 = c_q2(x), k2 = c_k2(x), causal-masked, NOT row-normalised. Every "patterns frozen" instrument from v35 to v47
hooked only c_q / c_k, leaving c_q2 / c_k2 live: a half-freeze. v47's non-zero linear residual (0.36 / 0.11 / 0.19 of
the floor gate) is the expected signature. This run repeats v47's four arms with all four projections frozen
(downstream layers in every arm; stack attention layers in F3p / F3) and the same executable I = 0 check.
    F0   downstream MLP linear in u + downstream patterns fully frozen     (repaired v44 floor)
    F2   full downstream linearisation (MLP in x, value path, patterns)    (repaired v46)
    F3p  F2 + stack attention patterns fully frozen
    F3   F3p + stack value path linear
One-sided instrument failure -> one repair; predictions and bars are v47's, unchanged.

REGISTERED BEFORE THE RUN
    pred_a_complete           |I_lin(F3)| <= 0.05 |I_F0| on all three. Worked: 0.01 True; v47's 0.36 False.
    pred_b_stack_attn_carries (I_F2 - I_F3) / I_F0 >= 0.50 on polarity and voice. Worked: 0.70 True; 0.20 False.
    pred_c_patterns_not_values (I_F2 - I_F3p) >= 0.50 (I_F2 - I_F3) on polarity and voice. Worked: 0.8 True; 0.3 False.
    pred_d_quantifier_too     quantifier (I_F2 - I_F3) / I_F0 >= 0.30. Worked: 0.5 True; 0.1 False.
    pred_e_instrument         offline FULL readout matches the run within 1e-3 on every run of every arm. Worked 1e-7.
    Reading rule. a True: the gate is exactly accounted for (MLP products v42/v43; stack- and downstream-attention
    products; block-input rms curvature; readout curvature). a False: a nonlinearity is still unhooked -- stop and
    read the model, do not raise the bar.
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
import run_unit_downstream_linearisation_v43 as v43
import run_unit_attention_product_locus_v44 as v44
import run_unit_full_linearisation_v46 as v46

PROJ = ("c_q", "c_k", "c_q2", "c_k2")


def _capture_qk4(model):
    store, handles = {}, []
    for l, block in enumerate(model.transformer.h):
        for nm in PROJ:
            handles.append(getattr(block.attn, nm).register_forward_hook(lambda m, a, o, key=(l, nm): store.__setitem__(key, o.detach().clone())))
    return store, handles


@contextmanager
def _qk4_frozen(model, store, layers):
    handles = []
    for l in layers:
        for nm in PROJ:
            handles.append(getattr(model.transformer.h[l].attn, nm).register_forward_hook(lambda m, a, o, key=(l, nm): store[key]))
    try:
        yield
    finally:
        for h in handles:
            h.remove()

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_full_freeze_v48_result.json"
SETS, STACK, N_LAYERS, D = v35.SETS, v35.STACK, 18, 1152
ZERO_MAX, STACK_MIN, PAT_MIN, QUA_MIN, INST_TOL = 0.05, 0.50, 0.50, 0.30, 1e-3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 80, 2600


def _plan():
    return {"candidate_id": "corpus.unit_full_freeze_v48", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


@contextmanager
def _stack_vlinear(torch, model, layers, positions, device, rids, resid_live, x_base, prev_out_base, prev_out_run, x0):
    """Value path linear in live for the stack attention layers; prev_out_* give the (replaced) MLP output of layer l-1."""
    idx = torch.arange(len(positions), device=device)
    pos = torch.tensor(positions, device=device)
    handles = []

    def live_norm(l, x_prev):
        lam = model.transformer.h[l].lambdas
        return (lam[0] * x_prev + lam[1] * x0).norm(dim=1, keepdim=True)

    def mk(l):
        nb = live_norm(l, x_base[l - 1] + prev_out_base[l - 1])

        def hook(m, a, o):
            x_prev = torch.stack([resid_live[(rid, l - 1)] for rid in rids]) + prev_out_run[l - 1]
            ratio = live_norm(l, x_prev) / nb
            y = o.clone()
            y[idx, pos] = (y[idx, pos].float() * ratio).to(y.dtype)
            return y
        return hook
    for l in layers:
        handles.append(model.transformer.h[l].attn.c_v.register_forward_hook(mk(l)))
    try:
        yield
    finally:
        for h in handles:
            h.remove()


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
        stack_attn = list(range(first + 1, last + 1))
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
        qk, handles = _capture_qk4(model)
        try:
            _, u_base, out_base = v30._capture(backend, prep, down, **cfg(None, None, {}))
        finally:
            for h in handles:
                h.remove()
        x_base = {l: torch.stack([resid_b[(rid, l)] for rid in rids]) for l in range(N_LAYERS)}
        prev_out_base = dict(out_base)
        for l in layers:
            prev_out_base[l] = outs_b[l]
        tokens = torch.tensor([prep.base_batch.token_rows[i][positions[i]] for i in range(len(rids))], device=backend.device)
        x0 = F.rms_norm(model.transformer.wte(tokens).float(), (D,))
        wa = W[torch.tensor(prep.base_batch.answer_ids, device=W.device)]
        wf = W[torch.tensor(prep.base_batch.foil_ids, device=W.device)]
        pf_down, pf_all = list(down), list(down) + list(stack_attn)

        def four(arm):
            recs, xf = {}, {}
            for key, (w_, vec) in RUNS.items():
                resid, store = {}, {}
                prev_out_run = dict(prev_out_base)
                if vec is not None:
                    prev_out_run[first] = outs_b[first] + vec
                ctxs = []
                if arm == "F0":
                    ctxs.append(v43._linearised(torch, model, down, positions, backend.device, u_base, out_base))
                    ctxs.append(_qk4_frozen(model, qk, pf_down))
                else:
                    ctxs.append(v46._xlinear(torch, model, down, positions, backend.device, rids, resid, x_base, out_base, prev_out_base, x0, True, store))
                    ctxs.append(_qk4_frozen(model, qk, pf_down if arm == "F2" else pf_all))
                    if arm == "F3":
                        ctxs.append(_stack_vlinear(torch, model, stack_attn, positions, backend.device, rids, resid, x_base, prev_out_base, prev_out_run, x0))
                from contextlib import ExitStack
                with ExitStack() as es:
                    for c in ctxs:
                        es.enter_context(c)
                    rec, _, outs = v30._capture(backend, prep, down, **cfg(w_, vec, resid))
                recs[key] = rec
                xf[key] = torch.stack([resid[(rid, N_LAYERS - 1)] for rid in rids]) + outs[N_LAYERS - 1]
            return recs, xf

        def inter(r):
            return r["DV"] - r["D"] - r["V"] + r["B"]
        arms = {}
        for arm in ("F0", "F2", "F3p", "F3"):
            recs, xf = four(arm)
            scale_b = (D ** 0.5) / xf["B"].norm(dim=1, keepdim=True)
            off = {mode: {k: v25._rec(prep, v46._readout(torch, xf[k], wa, wf, mode, scale_b).tolist()) for k in RUNS} for mode in ("full", "linear")}
            arms[arm] = {"rec_run": recs, "i_run": inter(recs), "i_offline_full": inter(off["full"]), "i_offline_linear": inter(off["linear"]),
                         "instrument_max_err": max(abs(off["full"][k] - recs[k]) for k in RUNS)}
        i0, i2, i3p, i3 = (arms[a]["i_run"] for a in ("F0", "F2", "F3p", "F3"))
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "stack_attention_layers": stack_attn, "downstream": down, "rows": len(prep.rows),
                        "arms": arms, "i_F0": i0, "i_F2": i2, "i_F3p": i3p, "i_F3": i3, "i_lin_F3": arms["F3"]["i_offline_linear"],
                        "share_stack_attention": (i2 - i3) / i0 if i0 else None, "share_stack_patterns": (i2 - i3p) / i0 if i0 else None,
                        "share_readout": i3 / i0 if i0 else None, "share_unexplained": arms["F3"]["i_offline_linear"] / i0 if i0 else None}
        print(name, "I F0 %.4f F2 %.4f F3p %.4f F3 %.4f lin(F3) %.5f | stack %.2f (patterns %.2f) readout %.2f unexpl %.2f" % (
            i0, i2, i3p, i3, arms["F3"]["i_offline_linear"], report[name]["share_stack_attention"], report[name]["share_stack_patterns"],
            report[name]["share_readout"], report[name]["share_unexplained"]), flush=True)

    pv = ("polarity_licensing", "voice_frame")
    qua = report["quantifier_number"]
    predictions = {
        'pred_a_complete': all(abs(report[n]["i_lin_F3"]) <= ZERO_MAX * abs(report[n]["i_F0"]) for n in SETS),
        'pred_b_stack_attn_carries': all(report[n]["share_stack_attention"] is not None and report[n]["share_stack_attention"] >= STACK_MIN for n in pv),
        'pred_c_patterns_not_values': all((report[n]["i_F2"] - report[n]["i_F3p"]) >= PAT_MIN * (report[n]["i_F2"] - report[n]["i_F3"]) for n in pv),
        'pred_d_quantifier_too': qua["share_stack_attention"] is not None and qua["share_stack_attention"] >= QUA_MIN,
        'pred_e_instrument': all(report[n]["arms"][a]["instrument_max_err"] <= INST_TOL for n in SETS for a in ("F0", "F2", "F3p", "F3")),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_full_freeze_result_v1", "candidate_id": "corpus.unit_full_freeze_v48",
              "bars": {"zero_max": ZERO_MAX, "stack_min": STACK_MIN, "pat_min": PAT_MIN, "qua_min": QUA_MIN, "inst_tol": INST_TOL},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
