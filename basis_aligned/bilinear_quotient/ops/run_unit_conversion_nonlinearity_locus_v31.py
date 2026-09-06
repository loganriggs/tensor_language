#!/usr/bin/env python3
# BQGATE: frozen predictions; sets, stacks, alpha/beta grids and the two replay arms fixed before the run.
"""v31: WHERE is the quadratic? v29 found stack conversion ~ alpha^2 (polarity 2.08, voice 2.04); v30 found the
stack's own bilinear self-term L(w)R(w) carries only 24-29% of conversion at alpha = 1 and the cross-terms
(linear in w by construction) carry 72-79%. Both cannot be true unless either (i) w_l(alpha) at the stack's
input is itself superlinear in alpha (a cascade: attention/MLP feedback between the write and the stack), or
(ii) the margin's response to the stack's OUTPUT is superlinear (the nonlinearity sits downstream of the
stack -- later bilinear layers or the readout). Two replay arms separate them:
  arm X (alpha replay)  scaled write alpha in {0.25, 0.5, 0.75, 1}; capture w_l(alpha); exact cross/self split
                        per layer; rec(cross-only(alpha)), rec(self-only(alpha)).
  arm Y (beta replay)   the alpha = 1 stack output deltas DeltaM_l(1) scaled by beta in {0.25, 0.5, 0.75, 1},
                        written on top of the base stack outputs with the set patched at alpha = 1 -> rec(beta).
                        If rec(beta) - rec_frozen is quadratic in beta, the nonlinearity is downstream of the stack.
Slopes are log-log least squares over the four points, as in v29.

REGISTERED BEFORE THE RUN
    pred_a_instrument          rec(cross-only(1)) and rec(self-only(1)) match v30 within 0.005 on all three sets.
    pred_b_write_linear        slope of mean |w_first(alpha)| in [0.90, 1.10] on all three (the write reaching the first
                               stack layer is linear; rms_norm is mild). Worked: 0.98 True; 1.3 False.
    pred_c_cross_superlinear   slope of rec(cross-only(alpha)) - rec_frozen >= 1.6 for polarity AND voice (the
                               cross channel's margin effect is what v29 measured as quadratic). Worked: 1.9 True.
    pred_d_downstream_locus    slope of rec(beta) - rec_frozen >= 1.6 for polarity AND voice: the superlinearity is
                               downstream of the stack. Worked: 1.8 True; 1.05 False (then the locus is the cascade
                               into w_l, pred_e).
    pred_e_cascade_locus       slope of mean |w_last(alpha)| >= 1.3 for polarity AND voice (the LAST stack layer's
                               input delta is superlinear because it carries earlier conversions). Worked: 1.4 True.
    Reading rule. d True: the "quadratic converter law" is a property of how the margin reads the stack's output,
    not of the stack's algebra -- Tier 4 must expand the reader, not the converter. d False & e True: cascade;
    Tier 4 expands layer-by-layer with self-consistent w. Both False: the locus is the rms_norm / attention
    pattern response between the write and the stack (next: freeze attention patterns).
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
OUT = ROOT / "circuits/followups/unit_conversion_nonlinearity_locus_v31_result.json"
PRIOR = ROOT / "circuits/followups/unit_selfterm_sufficiency_v30_result.json"
SETS, STACK, EARLY = v30.SETS, v30.STACK, v30.EARLY
GRID = (0.25, 0.5, 0.75, 1.0)
INSTR_TOL, WRITE_BAND, SUPER_BAR, CASCADE_BAR = 0.005, (0.90, 1.10), 1.6, 1.3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 80, 3000


def _plan():
    return {"candidate_id": "corpus.unit_conversion_nonlinearity_locus_v31", "sets": {k: v[1] for k, v in SETS.items()},
            "stack": STACK, "grid": list(GRID), "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX, "model_backwards": 0, "model_updates": 0,
            "fit_parameters": 0, "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def _slope(xs, ys):
    if any(y <= 0 for y in ys):
        return None
    lx, ly = [math.log(x) for x in xs], [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    torch, model = backend.torch, backend.model
    prior = json.loads(PRIOR.read_text())["behaviours"]
    t0 = time.perf_counter()
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        stack_mlps = [m for m in STACK[name] if m.startswith("mlp:")]
        layers = [g.unit_layer(m) for m in stack_mlps]
        cache = v25._merged_cache(prep, units)
        rids = list(prep.base_batch.row_ids)
        _, ins_b, outs_b = v30._capture(backend, prep, layers)
        frozen = dict(cache)
        for l, m in zip(layers, stack_mlps):
            for i, rid in enumerate(rids):
                frozen[(rid, m)] = outs_b[l][i]
        all_units = list(units) + stack_mlps

        def rec_with(c, q=None):
            return v30._capture(backend, prep, layers, units=all_units, donor_cache=c, base_cache=prep.base_cache, q=q)[0]
        rec_frozen = rec_with(frozen)

        # arm X: alpha replay
        armX, outs_p1 = {}, None
        for a in GRID:
            q = v27._scaled_q(backend, units, a)
            rec_live, ins_p, outs_p = v30._capture(backend, prep, layers, units=list(units), donor_cache=cache, base_cache=prep.base_cache, q=q)
            if a == 1.0:
                outs_p1 = outs_p
            self_cache, cross_cache, wn, cn, sn = dict(cache), dict(cache), {}, {}, {}
            for l, m in zip(layers, stack_mlps):
                cross, self_ = v30._terms(model.transformer.h[l].mlp, ins_b[l], ins_p[l])
                wn[m] = float((ins_p[l] - ins_b[l]).norm(dim=1).mean())
                cn[m], sn[m] = float(cross.norm(dim=1).mean()), float(self_.norm(dim=1).mean())
                for i, rid in enumerate(rids):
                    self_cache[(rid, m)] = outs_b[l][i] + self_[i]
                    cross_cache[(rid, m)] = outs_b[l][i] + cross[i]
            # the set is written at alpha inside these arms too (q on the set blocks; stack MLPs replaced exactly)
            qx = dict(q)
            for key in g.blocks_of(stack_mlps):
                qx[key] = torch.eye(g.N_EMBD, device=backend.device)
            armX[a] = {"rec_live": rec_live, "rec_self_only": rec_with(self_cache, qx), "rec_cross_only": rec_with(cross_cache, qx),
                       "write_norm": wn, "cross_norm": cn, "self_norm": sn}
            print(name, "alpha", a, {k: round(v, 3) for k, v in armX[a].items() if isinstance(v, float)}, flush=True)

        # arm Y: beta replay of the alpha = 1 stack output deltas, set at alpha = 1
        armY = {}
        for b in GRID:
            c = dict(cache)
            for l, m in zip(layers, stack_mlps):
                for i, rid in enumerate(rids):
                    c[(rid, m)] = outs_b[l][i] + b * (outs_p1[l][i] - outs_b[l][i])
            armY[b] = rec_with(c)
        first, last = stack_mlps[0], stack_mlps[-1]
        slopes = {"write_first": _slope(GRID, [armX[a]["write_norm"][first] for a in GRID]),
                  "write_last": _slope(GRID, [armX[a]["write_norm"][last] for a in GRID]),
                  "cross_norm_first": _slope(GRID, [armX[a]["cross_norm"][first] for a in GRID]),
                  "self_norm_first": _slope(GRID, [armX[a]["self_norm"][first] for a in GRID]),
                  "rec_cross_only": _slope(GRID, [armX[a]["rec_cross_only"] - rec_frozen for a in GRID]),
                  "rec_self_only": _slope(GRID, [armX[a]["rec_self_only"] - rec_frozen for a in GRID]),
                  "rec_live_conversion": _slope(GRID, [armX[a]["rec_live"] - rec_frozen for a in GRID]),
                  "rec_beta": _slope(GRID, [armY[b] - rec_frozen for b in GRID])}
        report[name] = {"units": list(units), "stack_mlps": stack_mlps, "rows": len(prep.rows), "rec_frozen": rec_frozen,
                        "arm_alpha": armX, "arm_beta": armY, "slopes": slopes,
                        "instrument": {"self_only_1_vs_v30": armX[1.0]["rec_self_only"] - prior[name]["rec"]["self_only"],
                                       "cross_only_1_vs_v30": armX[1.0]["rec_cross_only"] - prior[name]["rec"]["cross_only"]}}
        print(name, "slopes", {k: (round(v, 2) if v is not None else None) for k, v in slopes.items()}, "beta", {b: round(r, 3) for b, r in armY.items()}, flush=True)

    def sl(n, k):
        return report[n]["slopes"][k]
    predictions = {
        'pred_a_instrument': all(abs(x) <= INSTR_TOL for n in SETS for x in report[n]["instrument"].values()),
        'pred_b_write_linear': all(sl(n, "write_first") is not None and WRITE_BAND[0] <= sl(n, "write_first") <= WRITE_BAND[1] for n in SETS),
        'pred_c_cross_superlinear': all(sl(n, "rec_cross_only") is not None and sl(n, "rec_cross_only") >= SUPER_BAR for n in EARLY),
        'pred_d_downstream_locus': all(sl(n, "rec_beta") is not None and sl(n, "rec_beta") >= SUPER_BAR for n in EARLY),
        'pred_e_cascade_locus': all(sl(n, "write_last") is not None and sl(n, "write_last") >= CASCADE_BAR for n in EARLY),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_conversion_nonlinearity_locus_result_v1",
              "candidate_id": "corpus.unit_conversion_nonlinearity_locus_v31", "grid": list(GRID),
              "bars": {"instrument": INSTR_TOL, "write_band": list(WRITE_BAND), "superlinear": SUPER_BAR, "cascade": CASCADE_BAR},
              "slopes": {n: report[n]["slopes"] for n in SETS}, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "slopes": result["slopes"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
