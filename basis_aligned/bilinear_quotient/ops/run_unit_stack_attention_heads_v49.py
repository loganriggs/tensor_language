"""v49: WHICH stack-attention heads form the write x converted-write pattern product? (per-head, four-projection freeze)

v48 closed the gate exactly (I_lin(F3) = 0) and put 0.42 (voice) / 0.09 (polarity) / 0.05 (quantifier) of b into stack
attention PATTERNS (layers first+1..last, all four projections). This run localises that pattern term to heads. Baseline
arm is v48's F3p (downstream MLP linear in x, downstream value path linear, all patterns frozen in stack and downstream);
each test arm lets ONE stack head's (q, k, q2, k2) slice run live, everything else frozen. Excesses are floor-subtracted:
    excess(l,h) = I(F3p + head l:h live) - I(F3p);  excess(l) = I(F3p + layer l live) - I(F3p);  T = I(F2) - I(F3p).
Head slices combine through LIVE stack MLPs, so head-additivity within a layer is a real test, not an identity.
Denominators are T (the pattern term itself), not b, so the shares are of the quantity being localised.

REGISTERED BEFORE THE RUN
    pred_a_layers_additive   sum_l excess(l) / T in [0.7, 1.3] on voice and polarity. Worked: 1.10 True; 1.60 False.
    pred_b_voice_head_locus  voice: max_h excess(l,h) >= 0.25 T. Worked: T 0.0048, top head 0.0020 -> 0.42 True; 0.0008 False.
    pred_c_heads_additive    voice and polarity, every stack layer: |sum_h excess(l,h) - excess(l)| <= 0.15 |T|. Worked: T 0.0048,
                             layer 0.0021 vs head-sum 0.0018 -> 0.0003 <= 0.0007 True; head-sum 0.0035 -> False.
    pred_d_polarity_single   polarity: max_h excess(l,h) >= 0.50 T (one head carries the polarity pattern product).
                             Worked: T 0.0023, top 0.0014 -> 0.61 True; 0.0008 -> 0.35 False.
    pred_e_instrument        new head-slice freeze path: spec(live={} in all stack layers) reproduces F3p and spec(live=all heads)
                             reproduces F2 within 1e-5 rec on every run; offline full readout within 1e-3 of each run.
    Reading rule. b True and d True: the pattern product has a head locus in both -> next is the head's pattern change under
    the write (which key tokens gain weight). b or d False: distributed across heads; do not pick the largest post hoc.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_selfterm_sufficiency_v30 as v30
import run_unit_pattern_freeze_v35 as v35
import run_unit_norm_gain_control_v38 as v38
import run_unit_full_linearisation_v46 as v46
import run_unit_full_freeze_v48 as v48

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_stack_attention_heads_v49_result.json"
SETS, STACK, N_LAYERS, D, N_HEADS, HEAD_DIM = v35.SETS, v35.STACK, 18, 1152, 9, 128
ADD_BAND, VOICE_MIN, HEAD_TOL, POL_MIN, PATH_TOL, INST_TOL = (0.7, 1.3), 0.25, 0.15, 0.50, 1e-5, 1e-3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 560, 18000


def _plan():
    return {"candidate_id": "corpus.unit_stack_attention_heads_v49", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


@contextmanager
def _qk4_frozen_spec(model, store, spec):
    """spec: {layer: iterable of LIVE heads}; every other head slice of (q, k, q2, k2) in those layers is replaced by the store."""
    handles = []
    for l, live in spec.items():
        live = tuple(live)
        for nm in v48.PROJ:
            def hook(m, a, o, key=(l, nm), live=live):
                y = store[key].clone()
                for h in live:
                    y[..., h * HEAD_DIM:(h + 1) * HEAD_DIM] = o[..., h * HEAD_DIM:(h + 1) * HEAD_DIM]
                return y
            handles.append(getattr(model.transformer.h[l].attn, nm).register_forward_hook(hook))
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
        qk, handles = v48._capture_qk4(model)
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

        def four(mode, spec=None):
            """mode: 'none' (no linearisation), 'F2', 'F3p' (v48 whole-layer freeze), 'spec' (F2 downstream + head-slice spec in stack)."""
            recs, xf = {}, {}
            for key, (w_, vec) in RUNS.items():
                resid, store = {}, {}
                ctxs = []
                if mode != "none":
                    ctxs.append(v46._xlinear(torch, model, down, positions, backend.device, rids, resid, x_base, out_base, prev_out_base, x0, True, store))
                    if mode == "F2":
                        ctxs.append(v48._qk4_frozen(model, qk, down))
                    elif mode == "F3p":
                        ctxs.append(v48._qk4_frozen(model, qk, down + stack_attn))
                    else:
                        ctxs.append(v48._qk4_frozen(model, qk, down))
                        ctxs.append(_qk4_frozen_spec(model, qk, spec))
                with ExitStack() as es:
                    for c in ctxs:
                        es.enter_context(c)
                    rec, _, outs = v30._capture(backend, prep, down, **cfg(w_, vec, resid))
                recs[key] = rec
                xf[key] = torch.stack([resid[(rid, N_LAYERS - 1)] for rid in rids]) + outs[N_LAYERS - 1]
            scale_b = (D ** 0.5) / xf["B"].norm(dim=1, keepdim=True)
            off = {k: v25._rec(prep, v46._readout(torch, xf[k], wa, wf, "full", scale_b).tolist()) for k in RUNS}
            return recs, max(abs(off[k] - recs[k]) for k in RUNS)

        def inter(r):
            return r["DV"] - r["D"] - r["V"] + r["B"]

        inst = []
        arms = {}
        for mode in ("none", "F2", "F3p"):
            recs, err = four(mode); inst.append(err); arms[mode] = recs
        b_full, i2, i3p = inter(arms["none"]), inter(arms["F2"]), inter(arms["F3p"])
        T = i2 - i3p
        # new-code-path controls: empty spec == F3p; all-heads spec == F2
        recs_e, err = four("spec", {l: [] for l in stack_attn}); inst.append(err)
        recs_a, err = four("spec", {l: list(range(N_HEADS)) for l in stack_attn}); inst.append(err)
        path_err = max(max(abs(recs_e[k] - arms["F3p"][k]) for k in RUNS), max(abs(recs_a[k] - arms["F2"][k]) for k in RUNS))
        per_layer, per_head = {}, {}
        for l in stack_attn:
            spec = {m: ([] if m != l else list(range(N_HEADS))) for m in stack_attn}
            recs, err = four("spec", spec); inst.append(err)
            per_layer[f"attn:{l:02d}"] = inter(recs) - i3p
            for h in range(N_HEADS):
                spec = {m: ([] if m != l else [h]) for m in stack_attn}
                recs, err = four("spec", spec); inst.append(err)
                per_head[f"attn:{l:02d}:head:{h:02d}"] = inter(recs) - i3p
        head_sum_by_layer = {f"attn:{l:02d}": sum(per_head[f"attn:{l:02d}:head:{h:02d}"] for h in range(N_HEADS)) for l in stack_attn}
        top = max(per_head, key=lambda k: per_head[k])
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "stack_attention_layers": stack_attn, "rows": len(prep.rows),
                        "b_full": b_full, "i_F2": i2, "i_F3p": i3p, "T_pattern_term": T, "T_over_b": T / b_full if b_full else None,
                        "excess_layer": per_layer, "excess_head": per_head, "head_sum_by_layer": head_sum_by_layer,
                        "layer_sum_over_T": sum(per_layer.values()) / T if T else None,
                        "head_additivity_gap_over_T": {k: (head_sum_by_layer[k] - per_layer[k]) / T if T else None for k in per_layer},
                        "top_head": top, "top_head_over_T": per_head[top] / T if T else None,
                        "path_control_max_err": path_err, "instrument_max_err": max(inst)}
        print(name, "b %.4f T %.4f (%.2f of b) | layers/T %.2f | top %s %.2f T | path %.1e" % (
            b_full, T, report[name]["T_over_b"] or 0, report[name]["layer_sum_over_T"] or 0, top, report[name]["top_head_over_T"] or 0, path_err), flush=True)
        print("   layers", {k: round(v_ / T, 2) if T else None for k, v_ in per_layer.items()},
              "gaps", {k: round(v_, 2) if v_ is not None else None for k, v_ in report[name]["head_additivity_gap_over_T"].items()}, flush=True)

    pv = ("polarity_licensing", "voice_frame")
    vo, po = report["voice_frame"], report["polarity_licensing"]
    predictions = {
        'pred_a_layers_additive': all(report[n]["layer_sum_over_T"] is not None and ADD_BAND[0] <= report[n]["layer_sum_over_T"] <= ADD_BAND[1] for n in pv),
        'pred_b_voice_head_locus': vo["top_head_over_T"] is not None and vo["top_head_over_T"] >= VOICE_MIN,
        'pred_c_heads_additive': all(gp is not None and abs(gp) <= HEAD_TOL for n in pv for gp in report[n]["head_additivity_gap_over_T"].values()),
        'pred_d_polarity_single': po["top_head_over_T"] is not None and po["top_head_over_T"] >= POL_MIN,
        'pred_e_instrument': all(report[n]["path_control_max_err"] <= PATH_TOL and report[n]["instrument_max_err"] <= INST_TOL for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_stack_attention_heads_result_v1", "candidate_id": "corpus.unit_stack_attention_heads_v49",
              "bars": {"add_band": ADD_BAND, "voice_min": VOICE_MIN, "head_tol": HEAD_TOL, "pol_min": POL_MIN, "path_tol": PATH_TOL, "inst_tol": INST_TOL},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
