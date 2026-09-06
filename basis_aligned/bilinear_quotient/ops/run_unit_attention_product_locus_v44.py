#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, the fixed vector, the linearisation and pattern-freeze arms fixed before the run.
"""v44: the part of the write x converted-write product that the downstream ATTENTION forms -- which layer, which head?

v43: with every downstream MLP linearised around base, 0.22 (polarity) / 0.09 (quantifier) / 0.59 (voice) of the gated
readout b survives. Squared attention is itself quadratic in its inputs, so this remainder can be a product formed in
the attention patterns of layers 12-17. Instrument: on top of the MLP linearisation, freeze c_q / c_k outputs (hence
patterns) to their base values -- all downstream layers (everything after the stack is then first order in the
residual, up to rms_norm and the final tanh), all but one layer, or all but one head (a head's q/k block is the
head_dim slice of c_q / c_k). Four runs per arm (B, D, V, DV; iso design, alpha = 1); I = rec(DV) - rec(D) - rec(V) +
rec(B); shares s = I / b_full. Every summand below has the floor s_floor (all frozen, all linearised) subtracted;
the target is s_mlplin (MLPs linearised, attention live) - s_floor. Aggregation matched on both sides.

REGISTERED BEFORE THE RUN
    pred_a_linear_floor        s_floor <= 0.15 on all three. Worked: 0.05 True; 0.30 False.
    pred_b_voice_layers_add    voice: sum_l (s_layer_l - s_floor) within [0.7, 1.3] x (s_mlplin - s_floor).
                               Worked: floor 0.05, s_mlplin 0.59, excess sum 0.54 -> ratio 1.0 True; 0.25 -> 0.46 False.
    pred_c_voice_head          voice: best single head's excess >= 0.25 x (s_mlplin - s_floor). Worked: 0.20/0.54 True;
                               0.05/0.54 False.
    pred_d_polarity_att_small  polarity: s_mlplin - s_floor <= 0.35 (v39's attention share was 0.29). Worked: 0.17 True;
                               0.50 False.
    pred_e_heads_sum_to_layer  voice, every downstream layer: |sum_h (s_head - s_floor) - (s_layer - s_floor)| <= 0.15.
                               Worked: layer 0.20, heads sum 0.22 True; heads sum 0.65 False.
    Control (must hold): B under all freezes equals live B to 1e-4.
    Reading rule. a True: the gate is entirely second-order in downstream MLPs + attention patterns. b, c True: the
    voice gate is an attention product localised to named heads -> next, which of q (the write) and k (the converted
    vector's carriers) moves.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_attention_product_locus_v44_result.json"
SETS, STACK, N_LAYERS, N_HEADS, HEAD_DIM = v35.SETS, v35.STACK, 18, 9, 128
FLOOR_MAX, ADD_BAND, HEAD_MIN, POL_MAX, HEAD_SUM_TOL, CTRL_TOL = 0.15, (0.7, 1.3), 0.25, 0.35, 0.15, 1e-4
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 800, 26000


def _plan():
    return {"candidate_id": "corpus.unit_attention_product_locus_v44", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


@contextmanager
def _qk_frozen(model, store, spec):
    """spec: {layer: None (whole layer) | [heads]} -> c_q / c_k outputs replaced by the stored base values."""
    handles = []

    def mk(key, heads):
        def hook(m, a, o):
            if heads is None:
                return store[key]
            y = o.clone()
            for h in heads:
                y[..., h * HEAD_DIM:(h + 1) * HEAD_DIM] = store[key][..., h * HEAD_DIM:(h + 1) * HEAD_DIM]
            return y
        return hook
    for l, heads in spec.items():
        for nm in ("c_q", "c_k"):
            handles.append(getattr(model.transformer.h[l].attn, nm).register_forward_hook(mk((l, nm), heads)))
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
    t0 = time.perf_counter()
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

        def cfg(write, vec):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + vec if (l == first and vec is not None) else outs_b[l])[i]
            kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
            if write is not None:
                kw["resid_add"] = {first: write}
            return kw

        RUNS = {"B": (None, None), "D": (delta1, None), "V": (None, v), "DV": (delta1, v)}
        qk, handles = v35._capture_qk(model)
        try:
            rec_b_live, u_base, out_base = v30._capture(backend, prep, down, **cfg(None, None))
        finally:
            for h in handles:
                h.remove()

        def four(lin_layers, spec):
            recs = {}
            for key, (w, vec) in RUNS.items():
                with v43._linearised(torch, model, lin_layers, positions, backend.device, u_base, out_base), _qk_frozen(model, qk, spec):
                    recs[key] = v30._capture(backend, prep, down, **cfg(w, vec))[0]
            return recs, recs["DV"] - recs["D"] - recs["V"] + recs["B"]

        _, b_full = four([], {})
        _, i_mlplin = four(down, {})
        floor_recs, i_floor = four(down, {l: None for l in down})
        ctrl_err = abs(floor_recs["B"] - rec_b_live)
        sh = (lambda i: i / b_full) if b_full else (lambda i: None)
        s_floor, s_mlplin = sh(i_floor), sh(i_mlplin)
        per_layer, per_head = {}, {}
        for l in down:
            _, i_l = four(down, {k: None for k in down if k != l})
            per_layer[f"attn:{l:02d}"] = sh(i_l)
            for h in range(N_HEADS):
                spec = {k: None for k in down if k != l}
                spec[l] = [x for x in range(N_HEADS) if x != h]
                _, i_h = four(down, spec)
                per_head[f"attn:{l:02d}:head:{h:02d}"] = sh(i_h)
        target = s_mlplin - s_floor
        layer_excess = {k: x - s_floor for k, x in per_layer.items()}
        head_excess = {k: x - s_floor for k, x in per_head.items()}
        head_sum_by_layer = {f"attn:{l:02d}": sum(head_excess[f"attn:{l:02d}:head:{h:02d}"] for h in range(N_HEADS)) for l in down}
        best_head = max(head_excess, key=head_excess.get)
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "downstream": down, "rows": len(prep.rows), "b_full": b_full,
                        "share_floor": s_floor, "share_mlp_linearised": s_mlplin, "attention_gate": target,
                        "layer_excess": layer_excess, "layer_excess_sum": sum(layer_excess.values()),
                        "head_excess": head_excess, "head_excess_sum_by_layer": head_sum_by_layer,
                        "best_head": best_head, "best_head_excess": head_excess[best_head], "control_base_identity_err": ctrl_err}
        print(name, "b %.4f floor %.2f mlplin %.2f gate %.2f | layer sum %.2f | best %s %.2f | ctrl %.1e" % (
            b_full, s_floor, s_mlplin, target, sum(layer_excess.values()), best_head, head_excess[best_head], ctrl_err),
            {k: round(x, 2) for k, x in layer_excess.items()}, flush=True)

    pol, voi = report["polarity_licensing"], report["voice_frame"]
    vt = voi["attention_gate"]
    predictions = {
        'pred_a_linear_floor': all(report[n]["share_floor"] is not None and abs(report[n]["share_floor"]) <= FLOOR_MAX for n in SETS),
        'pred_b_voice_layers_add': vt > 0 and ADD_BAND[0] * vt <= voi["layer_excess_sum"] <= ADD_BAND[1] * vt,
        'pred_c_voice_head': vt > 0 and voi["best_head_excess"] >= HEAD_MIN * vt,
        'pred_d_polarity_att_small': pol["attention_gate"] <= POL_MAX,
        'pred_e_heads_sum_to_layer': all(abs(voi["head_excess_sum_by_layer"][k] - voi["layer_excess"][k]) <= HEAD_SUM_TOL for k in voi["layer_excess"]),
    }
    control_ok = all(report[n]["control_base_identity_err"] <= CTRL_TOL for n in SETS)
    result = {"predictions": predictions, "control_base_identity_ok": control_ok,
              "schema": "circuit_unit_attention_product_locus_result_v1", "candidate_id": "corpus.unit_attention_product_locus_v44",
              "bars": {"floor_max": FLOOR_MAX, "add_band": ADD_BAND, "head_min": HEAD_MIN, "pol_max": POL_MAX, "head_sum_tol": HEAD_SUM_TOL, "ctrl_tol": CTRL_TOL},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "control_ok": control_ok, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
