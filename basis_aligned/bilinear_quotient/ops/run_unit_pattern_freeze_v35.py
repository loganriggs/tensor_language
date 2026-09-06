#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grid, the two freezes and their combination fixed before the run.
"""v35: is the alpha^2 GENERATED before the stack (intermediate layers) or inside the stack's response, and is it
carried by attention PATTERNS?

Chain: v29 conversion ~ alpha^2; v30 self-term 24-29%; v31 readout linear; v32 no cross-layer product; v34 rms_norm
second order inert and the first-order normalized write w1 carries 96-98% of the cross pull-back at alpha = 1 -- yet
<g, w(alpha)> is superlinear (v33 cross slope 2.31 / 1.68 / 1.95). So the margin-relevant component of the pre-norm
residual delta Delta(alpha) arriving at (and inside) the stack is superlinear even where its norm and cosine look
linear. Two things in the forward can do that: attention patterns (this model's attention squares q.k -- a pattern
responds quadratically to a change of the final-position query), and the layers between the set's heads and the stack.
Two freezes, crossed:
  LIN   no set write; ADD alpha * Delta_first(1) (the measured pre-norm residual delta at the first stack layer under the
        exact set) to the residual at that point. Removes generation; keeps the stack's own response.
  PF    q and k of EVERY layer replaced by their base-run outputs (patterns = base; values/MLPs carry the write).
conv_X(alpha) = rec_X(alpha) - rec_X,stack-frozen(alpha), stack MLPs replaced by base outputs, same alpha. Slopes log-log
over {0.25, 0.5, 0.75, 1}. Polarity's 09:07 stays live (its pattern is frozen under PF like every other head).

REGISTERED BEFORE THE RUN
    pred_a_instrument     rec_LIN(1) (stack live) equals rec_live(1) within 0.005 on all three: the residual delta at the
                          first stack layer is the whole of what the set did up to there.
    pred_b_values_carry   rec_PF(1) >= 0.80 rec_live(1) on all three: the write's effect survives base patterns. Worked:
                          0.50 / 0.585 = 0.85 True; 0.35 -> False.
    pred_c_stack_internal conv_LIN slope >= 1.6 for polarity AND voice: the stack's response to a LINEAR residual write is
                          itself superlinear (the alpha^2 is not generated upstream). Worked: 1.9 True; 1.1 False.
    pred_d_pattern_borne  conv_PF slope in [0.8, 1.3] for polarity AND voice: with base patterns the conversion is linear.
                          Worked: 1.05 True; 1.9 False.
    pred_e_both_linear    conv_LIN+PF slope in [0.8, 1.2] on all three. Worked: 1.0 True.
    Reading rule. c & d True: the converter is the squared-attention PATTERN response of heads at/inside the stack layers
    to the final-position query carrying the write -- the MLPs are downstream of a pattern change (Tier 4 next: which
    heads, the q.k expansion). c True & d False: the stack MLPs' own cross channel responds superlinearly to a linear
    write, contradicting v34's first-order pull-back -- re-examine the third-order rms_norm terms. c False: the alpha^2
    is generated between the set's heads and the stack; expand those layers.
"""
from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier3_readers_v25 as v25
import run_unit_damper_law_v27 as v27
import run_unit_selfterm_sufficiency_v30 as v30

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_pattern_freeze_v35_result.json"
SETS, STACK, EARLY = v30.SETS, v30.STACK, v30.EARLY
GRID = (0.25, 0.5, 0.75, 1.0)
INSTR_TOL, VALUES_BAR, INTERNAL_BAR, PF_BAND, BOTH_BAND = 0.005, 0.80, 1.6, (0.8, 1.3), (0.8, 1.2)
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 120, 4000


def _plan():
    return {"candidate_id": "corpus.unit_pattern_freeze_v35", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _slope(xs, ys):
    if any(y <= 0 for y in ys):
        return None
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def _capture_qk(model):
    """Forward hooks recording every layer's c_q / c_k output (full sequence)."""
    store, handles = {}, []
    for l, block in enumerate(model.transformer.h):
        for nm in ("c_q", "c_k"):
            handles.append(getattr(block.attn, nm).register_forward_hook(lambda m, a, o, key=(l, nm): store.__setitem__(key, o.detach().clone())))
    return store, handles


@contextmanager
def _patterns_frozen(model, store):
    handles = []
    for l, block in enumerate(model.transformer.h):
        for nm in ("c_q", "c_k"):
            handles.append(getattr(block.attn, nm).register_forward_hook(lambda m, a, o, key=(l, nm): store[key]))
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
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        first = layers[0]
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        # base run: q/k of every layer, stack base outputs, pre-norm residual at the first stack layer
        qk, handles = _capture_qk(model)
        resid_b = {}
        try:
            _, _, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        finally:
            for h in handles:
                h.remove()
        resid_p = {}
        rec_live1, _, _ = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        frozen = dict(cache)
        for l, m in zip(layers, stack_mlps):
            for i, rid in enumerate(rids):
                frozen[(rid, m)] = outs_b[l][i]

        def rec(alpha, *, lin, pf, stack_frozen):
            if lin:
                kw = {"units": stack_mlps if stack_frozen else [], "donor_cache": frozen, "base_cache": prep.base_cache,
                      "resid_add": {first: alpha * delta1}}
            else:
                q = v27._scaled_q(backend, units, alpha)
                us = list(units)
                if stack_frozen:
                    for key in g.blocks_of(stack_mlps):
                        q[key] = torch.eye(g.N_EMBD, device=backend.device)
                    us = us + stack_mlps
                kw = {"units": us, "donor_cache": frozen, "base_cache": prep.base_cache, "q": q}
            if pf:
                with _patterns_frozen(model, qk):
                    return v30._capture(backend, prep, layers, **kw)[0]
            return v30._capture(backend, prep, layers, **kw)[0]

        arms = {}
        for label, lin, pf in (("STD", False, False), ("LIN", True, False), ("PF", False, True), ("LIN+PF", True, True)):
            per = {}
            for a in GRID:
                live, fro = rec(a, lin=lin, pf=pf, stack_frozen=False), rec(a, lin=lin, pf=pf, stack_frozen=True)
                per[a] = {"rec": live, "rec_stack_frozen": fro, "conv": live - fro}
            arms[label] = {"per_alpha": per, "conv_slope": _slope(GRID, [per[a]["conv"] for a in GRID]),
                           "direct_slope": _slope(GRID, [per[a]["rec_stack_frozen"] for a in GRID])}
            print(name, label, "conv", [round(per[a]["conv"], 3) for a in GRID], "rec", [round(per[a]["rec"], 3) for a in GRID],
                  "slope", None if arms[label]["conv_slope"] is None else round(arms[label]["conv_slope"], 2), flush=True)
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "rec_live_1": rec_live1,
                        "delta_first_norm_mean": float(delta1.norm(dim=1).mean()), "arms": arms,
                        "instrument_lin_1_minus_live": arms["LIN"]["per_alpha"][1.0]["rec"] - rec_live1,
                        "pf_over_live_1": arms["PF"]["per_alpha"][1.0]["rec"] / rec_live1 if rec_live1 else None}

    def cs(n, arm):
        return report[n]["arms"][arm]["conv_slope"]
    def inband(v, band):
        return v is not None and band[0] <= v <= band[1]
    predictions = {
        'pred_a_instrument': all(abs(report[n]["instrument_lin_1_minus_live"]) <= INSTR_TOL for n in SETS),
        'pred_b_values_carry': all(report[n]["pf_over_live_1"] is not None and report[n]["pf_over_live_1"] >= VALUES_BAR for n in SETS),
        'pred_c_stack_internal': all(cs(n, "LIN") is not None and cs(n, "LIN") >= INTERNAL_BAR for n in EARLY),
        'pred_d_pattern_borne': all(inband(cs(n, "PF"), PF_BAND) for n in EARLY),
        'pred_e_both_linear': all(inband(cs(n, "LIN+PF"), BOTH_BAND) for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_pattern_freeze_result_v1",
              "candidate_id": "corpus.unit_pattern_freeze_v35", "grid": list(GRID),
              "squared_attention": bool(getattr(model.config, "squared_attn", False)),
              "bars": {"instrument": INSTR_TOL, "values": VALUES_BAR, "internal": INTERNAL_BAR, "pf_band": list(PF_BAND), "both_band": list(BOTH_BAND)},
              "slopes": {n: {arm: report[n]["arms"][arm]["conv_slope"] for arm in report[n]["arms"]} for n in SETS},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "slopes": result["slopes"], "squared_attention": result["squared_attention"],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
