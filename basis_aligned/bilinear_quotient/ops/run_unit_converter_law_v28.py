#!/usr/bin/env python3
# BQGATE: frozen predictions; sets (v9), alpha grid, converter stack definition and freeze semantics fixed before the run.
"""v28: the early sets' converters (mlp 7-11) under a scaled write -- polarity and voice.

v26: polarity's set (last layer 8) reaches the logits 33% directly and 67% through mlp 8-11 (+ attn 09:07,
10%); voice's (last layer 7) 16% directly and 84% through mlp 7-11. v27: quantifier's converter stack
(mlp 11-14) is sigmoid-like -- 0.06 / 0.17 / 0.24 / 0.23 at alpha 0.5 / 1 / 1.5 / 2 (convex low end,
saturating high end) -- while dative's dampers are linear. Same alpha grid on the early sets, with the
converter STACK frozen as one unit: polarity mlp 08-11, voice mlp 07-11 (the MLPs from the set's last
layer to 11 inclusive), plus attn 09:07 for polarity.
    rec_live(alpha)          scaled set, everything live
    rec_direct(alpha)        scaled set, ALL downstream modules frozen to base (v25 lists)
    rec_stack_frozen(alpha)  scaled set, only the converter stack frozen to base
    conversion(alpha)        rec_live - rec_direct;  stack_share = (rec_live - rec_stack_frozen) / conversion

REGISTERED BEFORE THE RUN
    pred_a_instrument       rec_live(1.0) equals v26's exact-set recovery (polarity 0.585, voice 0.645) within 0.005.
    pred_b_direct_linear    rec_direct(alpha)/(alpha rec_direct(1)) in [0.85, 1.15] at alpha 0.5, 1.5, 2 on both.
                            Worked: 0.93 -> True; 0.70 -> False.
    pred_c_stack_carries    stack_share(1.0) >= 0.80 on both (mlp 7/8-11 is where the conversion happens, not the
                            later modules). Worked: voice conv 0.544, stack drop 0.495 -> 0.91 -> True; 0.35 -> False.
    pred_d_saturates        conversion(2)/conversion(1) < 1.6 on both (sub-linear high end, as quantifier's 1.37).
                            Worked: 1.30 -> True; 1.90 -> False (linear converter).
    pred_e_no_dead_zone     conversion(0.5)/conversion(1) >= 0.40 on both (quantifier gave 0.37: a convex low end).
                            Worked: 0.48 -> True; 0.30 -> False (threshold-like, as quantifier).
    Reading rule. c True: the Tier 3 reader of the early sets is the mlp 7-11 stack as a unit, with the law given
    by d/e (sigmoid-like if d True and e False; linear if d False and e True; saturating-linear if both True).
    c False: later modules re-convert; extend the stack before any Tier 4 attempt.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g
import run_unit_tier2_characterization_v23 as v23
import run_unit_tier3_readers_v25 as v25
import run_unit_damper_law_v27 as v27

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_converter_law_v28_result.json"
SETS = {k: v23.SETS[k] for k in ("polarity_licensing", "voice_frame")}
STACK = {"polarity_licensing": [f"mlp:{l:02d}" for l in range(8, 12)] + ["attn:09:head:07"],
         "voice_frame": [f"mlp:{l:02d}" for l in range(7, 12)]}
ALPHAS = v27.ALPHAS
INSTR_TOL, LIN_LO, LIN_HI, STACK_BAR, SAT_BAR, DEAD_BAR = 0.005, 0.85, 1.15, 0.80, 1.6, 0.40
V26 = ROOT / "circuits/followups/unit_tier3_readers_zscore_v26_result.json"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 60, 2000


def _plan():
    return {"candidate_id": "corpus.unit_converter_law_v28", "sets": {k: v[1] for k, v in SETS.items()}, "stack": STACK,
            "alphas": list(ALPHAS), "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _eye_blocks(backend, units):
    return {key: backend.torch.eye(sum(g.unit_dim(u) for u in us), device=backend.device) for key, us in g.blocks_of(units).items()}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    v26r = json.loads(V26.read_text())["behaviours"]
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        downstream = v25._downstream(units)
        stack = STACK[name]
        assert all(m in downstream for m in stack), (name, stack)
        cache = v25._merged_cache(prep, units)

        def rec(q_set, frozen):
            q = dict(q_set)
            for key, qb in _eye_blocks(backend, frozen).items():
                assert key not in q, key
                q[key] = qb
            out = g.forward_units(backend, prep.base_batch, units=list(units) + list(frozen), donor_cache=cache,
                                  base_cache=prep.base_cache, q=q)
            return v25._rec(prep, [-(float(x) - float(f)) for x, f in out.tolist()])

        curve = {}
        for a in ALPHAS:
            q_set = v27._scaled_q(backend, units, a)
            live, direct, stack_frozen = rec(q_set, []), rec(q_set, downstream), rec(q_set, stack)
            conv = live - direct
            curve[a] = {"rec_live": live, "rec_direct": direct, "rec_stack_frozen": stack_frozen, "conversion": conv,
                        "stack_share": (live - stack_frozen) / conv if conv else None}
            print(name, a, {k: round(v, 3) for k, v in curve[a].items() if v is not None}, flush=True)
        report[name] = {"units": list(units), "stack": stack, "rows": len(prep.rows), "downstream_count": len(downstream),
                        "v26_rec_set": v26r[name]["rec_set"], "v26_direct": v26r[name]["direct"], "curve": curve}

    def lin(n):
        c = report[n]["curve"]; d1 = c[1.0]["rec_direct"]
        return {a: c[a]["rec_direct"] / (a * d1) for a in ALPHAS if a != 1.0}
    conv = {n: [report[n]["curve"][a]["conversion"] for a in ALPHAS] for n in SETS}
    predictions = {
        'pred_a_instrument': all(abs(report[n]["curve"][1.0]["rec_live"] - v26r[n]["rec_set"]) <= INSTR_TOL for n in SETS),
        'pred_b_direct_linear': all(LIN_LO <= r <= LIN_HI for n in SETS for r in lin(n).values()),
        'pred_c_stack_carries': all(report[n]["curve"][1.0]["stack_share"] is not None and report[n]["curve"][1.0]["stack_share"] >= STACK_BAR for n in SETS),
        'pred_d_saturates': all(conv[n][3] / conv[n][1] < SAT_BAR for n in SETS),
        'pred_e_no_dead_zone': all(conv[n][0] / conv[n][1] >= DEAD_BAR for n in SETS),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_converter_law_result_v1", "candidate_id": "corpus.unit_converter_law_v28",
              "semantics": "block_live_scaled_write_plus_base_freeze", "alphas": list(ALPHAS),
              "bars": {"instrument": INSTR_TOL, "linear": [LIN_LO, LIN_HI], "stack": STACK_BAR, "saturation": SAT_BAR, "dead_zone": DEAD_BAR},
              "direct_linearity": {n: lin(n) for n in SETS}, "conversion": conv,
              "conversion_ratios": {n: {"half_over_one": conv[n][0] / conv[n][1], "two_over_one": conv[n][3] / conv[n][1]} for n in SETS},
              "behaviours": report, "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "conversion": conv, "conversion_ratios": result["conversion_ratios"],
                      "direct_linearity": result["direct_linearity"], "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
