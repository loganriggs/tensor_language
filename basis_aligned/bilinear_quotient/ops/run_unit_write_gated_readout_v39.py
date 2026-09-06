#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, grid, the fixed vector and the freeze arms fixed before the run.
"""v39: the readout of a stack layer's output is GATED by the write itself.

v38 on polarity mlp:08 under a linear write: the same fixed vector v = cross_first(1), scaled by alpha and added to the
layer's base output, converts as alpha^1.55 when the residual write is alpha*Delta -- and as alpha^1.08 when the
residual write is held at Delta and only v is scaled. Neither the vector nor the layer changes; what changes is the
write present downstream when v is read. Decomposed at alpha=1: readout(alpha) := rec(alpha Delta + v) - rec(alpha Delta)
= a + b alpha with b ~ 0.030 against a ~ 0.011. This is a write x conversion product read by a LATER layer -- exactly
what v32 called additive; v32 measured the interaction against the whole recovery (0.585, dominated by the direct
path), where a 0.03 term is 5%; against the conversion it is 70%. Design error #9: normalize by what the conclusion is
about. bilin18's later MLPs are bilinear, so a product of two perturbations that both reach them is their natural cross
term L(alpha Delta')R(v') + L(v')R(alpha Delta'). Locate the multiplier by freezing candidate downstream modules to
their value from the alpha*Delta-only run (so they cannot respond to v) in both runs of the readout:
  arm mlp:l        one downstream MLP frozen        arm MLP-all   all MLPs after the stack frozen
  arm ATTN-all     all attention outputs after the first stack layer frozen (final position)
Design as v36/v37 iso: the other stack MLPs are base-frozen in every run, so they cannot be the multiplier here.
alpha grid {0, 0.25, 0.5, 0.75, 1}; a, b by least squares; modulation share = b / (a + b).

REGISTERED BEFORE THE RUN
    pred_a_polarity_gated    modulation share >= 0.5 on polarity. Worked: 0.72 True; 0.2 False.
    pred_b_others_not        modulation share <= 0.25 on quantifier AND voice (their lone-layer cross slopes are 1.15 /
                             1.18). Worked: 0.15 True; 0.4 False.
    pred_c_mlps_multiply     MLP-all removes >= 70% of polarity's b. Worked: 0.030 -> 0.005 True; -> 0.020 False.
    pred_d_not_attention     ATTN-all keeps >= 70% of polarity's b. Worked: 0.030 -> 0.026 True; -> 0.010 False.
    pred_e_single_multiplier one downstream MLP alone removes >= 50% of polarity's b. Worked: mlp:12 60% True; 20% each
                             False.
    Reading rule. a, c, d True: the conversion is AND-ed with the write by a downstream bilinear layer (e says whether
    one layer); the alpha^2 law of the early sets is write x converted-write in a later MLP, and the stack is one factor
    of a product, not the converter. c False & d True: the multiplier is elsewhere (the lm_head tanh cap or the
    final norm) -- test those. c False & d False: distributed; report as such.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_write_gated_readout_v39_result.json"
SETS, STACK, EARLY = v35.SETS, v35.STACK, v35.EARLY
GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GATED_BAR, OTHERS_MAX, MLP_REMOVE, ATTN_KEEP, SINGLE_REMOVE = 0.5, 0.25, 0.7, 0.7, 0.5
N_LAYERS = 18
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 240, 8000


def _plan():
    return {"candidate_id": "corpus.unit_write_gated_readout_v39", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _fit_ab(xs, ys):
    n = len(xs); mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


class _Freezer:
    """Capture (mode='capture') or replace (mode='replace') module outputs at the row positions."""

    def __init__(self, torch, model, positions, device):
        self.torch, self.model = torch, model
        self.idx = torch.arange(len(positions), device=device)
        self.pos = torch.tensor(positions, device=device)
        self.store = {}

    @contextmanager
    def hooks(self, mode, mlps=(), attns=()):
        handles = []

        def mk(key, is_attn):
            def hook(m, a, o):
                y = o[0] if is_attn else o
                if mode == "capture":
                    self.store[key] = y[self.idx, self.pos].detach().clone()
                    return None
                y = y.clone()
                y[self.idx, self.pos] = self.store[key].to(y.dtype)
                return (y, o[1]) if is_attn else y
            return hook
        for l in mlps:
            handles.append(self.model.transformer.h[l].mlp.register_forward_hook(mk(("mlp", l), False)))
        for l in attns:
            handles.append(self.model.transformer.h[l].attn.register_forward_hook(mk(("attn", l), True)))
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
        first, last = layers[0], layers[-1]
        down_mlps = list(range(last + 1, N_LAYERS))
        down_attns = list(range(first + 1, N_LAYERS))
        mlp = model.transformer.h[first].mlp
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        resid_b, resid_p = {}, {}
        _, ins_b, outs_b = v30._capture(backend, prep, layers, capture_resid=resid_b)
        v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, capture_resid=resid_p)
        delta1 = torch.stack([resid_p[(rid, first)] - resid_b[(rid, first)] for rid in rids])
        u_b = ins_b[first]
        _, ins_p1, _ = v30._capture(backend, prep, layers, resid_add={first: delta1})
        v = v38._cross(mlp, u_b, ins_p1[first] - u_b)
        fz = _Freezer(torch, model, list(prep.base_batch.semantic_positions), backend.device)

        def run(alpha, with_v):
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = (outs_b[l] + v if (l == first and with_v) else outs_b[l])[i]
            kw = {"units": stack_mlps, "donor_cache": c, "base_cache": prep.base_cache}
            if alpha:
                kw["resid_add"] = {first: alpha * delta1}
            return v30._capture(backend, prep, layers, **kw)[0]

        arms = {"none": ([], [])}
        arms.update({f"mlp:{l:02d}": ([l], []) for l in down_mlps})
        arms["MLP-all"] = (down_mlps, [])
        arms["ATTN-all"] = ([], down_attns)
        readout = {arm: {} for arm in arms}
        for a in GRID:
            for arm, (ms, ats) in arms.items():
                with fz.hooks("capture", ms, ats):
                    r0 = run(a, False)
                with fz.hooks("replace", ms, ats):
                    r1 = run(a, True)
                readout[arm][a] = r1 - r0
            print(name, a, {arm: round(readout[arm][a], 4) for arm in arms}, flush=True)
        fits = {arm: _fit_ab(GRID, [readout[arm][a] for a in GRID]) for arm in arms}
        a0, b0 = fits["none"]
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "downstream_mlps": down_mlps,
                        "downstream_attns": down_attns, "readout": {arm: {str(a): readout[arm][a] for a in GRID} for arm in arms},
                        "fit": {arm: {"a": fits[arm][0], "b": fits[arm][1]} for arm in arms},
                        "modulation_share": b0 / (a0 + b0) if (a0 + b0) else None,
                        "b_removed": {arm: (b0 - fits[arm][1]) / b0 if b0 else None for arm in arms if arm != "none"}}
        print(name, "a %.4f b %.4f share %.3f" % (a0, b0, report[name]["modulation_share"]),
              {arm: round(x, 2) for arm, x in report[name]["b_removed"].items()}, flush=True)

    pol = report["polarity_licensing"]
    predictions = {
        'pred_a_polarity_gated': pol["modulation_share"] is not None and pol["modulation_share"] >= GATED_BAR,
        'pred_b_others_not': all(report[n]["modulation_share"] is not None and report[n]["modulation_share"] <= OTHERS_MAX
                                 for n in ("quantifier_number", "voice_frame")),
        'pred_c_mlps_multiply': pol["b_removed"]["MLP-all"] is not None and pol["b_removed"]["MLP-all"] >= MLP_REMOVE,
        'pred_d_not_attention': pol["b_removed"]["ATTN-all"] is not None and pol["b_removed"]["ATTN-all"] <= 1 - ATTN_KEEP,
        'pred_e_single_multiplier': any(x is not None and x >= SINGLE_REMOVE for arm, x in pol["b_removed"].items() if arm.startswith("mlp:")),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_write_gated_readout_result_v1",
              "candidate_id": "corpus.unit_write_gated_readout_v39", "grid": list(GRID),
              "bars": {"gated": GATED_BAR, "others_max": OTHERS_MAX, "mlp_remove": MLP_REMOVE, "attn_keep": ATTN_KEEP, "single_remove": SINGLE_REMOVE},
              "modulation_share": {n: report[n]["modulation_share"] for n in SETS}, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "modulation_share": result["modulation_share"],
                      "b_removed": {n: report[n]["b_removed"] for n in SETS}, "seconds": round(result["seconds"], 1)}, indent=2, default=str))


if __name__ == "__main__":
    main()
