#!/usr/bin/env python3
# BQGATE: frozen predictions; sets (v9), alpha grid, freeze semantics and instrument check fixed before the run.
"""v27: is the downstream feedback on a set's write proportional to the write? (dative dampers, quantifier converters)

v26: dative's set writes ~0.96 of the donor margin into the residual and mlp 15-17 feed back ~40% of it,
against whichever answer leads; quantifier's mlp 11-14 ADD ~27% on top of a 73% direct path. A damper
that is a linear feedback term scales with the write; a thresholded or saturating one does not. Scale the
set's write: block-live patch live + alpha (donor - live) on every set unit (implemented as the existing
block-subspace path with q = sqrt(alpha) I, so alpha = 1 IS the exact replacement -- checked, pred_a),
alpha in {0.5, 1.0, 1.5, 2.0}. For each alpha: rec_live (everything downstream live) and rec_direct (all
downstream modules frozen to base, v25 lists). feedback(alpha) = rec_direct - rec_live (dative, > 0),
conversion(alpha) = rec_live - rec_direct (quantifier, > 0). Recoveries are signed fractions of the donor
margin and may exceed 1 at alpha > 1.

REGISTERED BEFORE THE RUN
    pred_a_instrument     rec_live(1.0) equals the exact-set recovery (v25: quantifier 0.635, dative 0.565)
                          within 0.005 on both. Worked: 0.634 -> True; 0.60 -> False (then nothing below counts).
    pred_b_direct_linear  rec_direct(alpha) / (alpha * rec_direct(1)) in [0.85, 1.15] for alpha in
                          {0.5, 1.5, 2.0} on both (the skip through the final norm + tanh cap is near-linear).
                          Worked: direct(2) = 1.85 vs 1.92 -> 0.96 -> True; 1.50 vs 1.92 -> 0.78 -> False.
    pred_c_dative_monotone   dative feedback strictly increasing over alpha 0.5 < 1 < 1.5 < 2.
                          Worked: 0.20, 0.37, 0.50, 0.60 -> True.
    pred_d_dative_proportional  dative feedback / rec_direct within +-0.10 of its alpha = 1 value at every
                          alpha. Worked: 0.38, 0.41, 0.43, 0.45 -> True; 0.41 at 1 and 0.25 at 2 -> False (saturating).
    pred_e_quantifier_monotone  quantifier conversion strictly increasing over the grid. Worked: 0.10, 0.17, 0.22, 0.26 -> True.
    Reading rule. c,d True: the late MLPs act as a proportional negative-feedback term on the dative margin
    at the final position -- a Tier 3 reader with a stated law. d False: the damper saturates or thresholds;
    report the curve. b False: the "direct path" is not linear in the write and v25/v26 ratios must be
    read at alpha = 1 only.
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

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_damper_law_v27_result.json"
SETS = {k: v23.SETS[k] for k in ("quantifier_number", "dative")}
ALPHAS = (0.5, 1.0, 1.5, 2.0)
INSTR_TOL, LIN_LO, LIN_HI, PROP_TOL = 0.005, 0.85, 1.15, 0.10
V25 = ROOT / "circuits/followups/unit_tier3_readers_v25_result.json"
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 40, 1500


def _plan():
    return {"candidate_id": "corpus.unit_damper_law_v27", "sets": {k: v[1] for k, v in SETS.items()}, "alphas": list(ALPHAS),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0, "gpu_accessed": False,
            "model_loaded": False, "execution_policy": "managed_queue_only"}


def _scaled_q(backend, units, alpha):
    torch = backend.torch
    return {key: (alpha ** 0.5) * torch.eye(sum(g.unit_dim(u) for u in us), device=backend.device)
            for key, us in g.blocks_of(units).items()}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    v25r = json.loads(V25.read_text())["behaviours"]
    report = {}
    for name, (module, units) in SETS.items():
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        downstream = v25._downstream(units)
        cache = v25._merged_cache(prep, units)
        # frozen downstream modules are exact (base) replacements: identity q on their blocks
        frozen_q = {key: backend.torch.eye(sum(g.unit_dim(u) for u in us), device=backend.device)
                    for key, us in g.blocks_of(downstream).items()}
        curve = {}
        for a in ALPHAS:
            q_live = _scaled_q(backend, units, a)
            out = g.forward_units(backend, prep.base_batch, units=list(units), donor_cache=prep.donor_cache,
                                  base_cache=prep.base_cache, q=q_live)
            rec_live = v25._rec(prep, [-(float(x) - float(f)) for x, f in out.tolist()])
            # set (scaled) + all downstream frozen to base: one block dict; a set block and a frozen block never share a key
            q_all = dict(q_live)
            for key, qb in frozen_q.items():
                assert key not in q_all, key
                q_all[key] = qb
            out = g.forward_units(backend, prep.base_batch, units=list(units) + downstream, donor_cache=cache,
                                  base_cache=prep.base_cache, q=q_all)
            rec_direct = v25._rec(prep, [-(float(x) - float(f)) for x, f in out.tolist()])
            curve[a] = {"rec_live": rec_live, "rec_direct": rec_direct, "feedback": rec_direct - rec_live,
                        "feedback_fraction": (rec_direct - rec_live) / rec_direct if rec_direct else None}
            print(name, a, {k: round(v, 3) for k, v in curve[a].items() if v is not None}, flush=True)
        report[name] = {"units": list(units), "rows": len(prep.rows), "downstream_count": len(downstream),
                        "v25_rec_set": v25r[name]["rec_set"], "v25_direct": v25r[name]["direct"], "curve": curve}

    def lin(name):
        c = report[name]["curve"]; d1 = c[1.0]["rec_direct"]
        return {a: c[a]["rec_direct"] / (a * d1) for a in ALPHAS if a != 1.0}
    fb = [report["dative"]["curve"][a]["feedback"] for a in ALPHAS]
    ff = [report["dative"]["curve"][a]["feedback_fraction"] for a in ALPHAS]
    conv = [-report["quantifier_number"]["curve"][a]["feedback"] for a in ALPHAS]
    predictions = {
        'pred_a_instrument': all(abs(report[n]["curve"][1.0]["rec_live"] - v25r[n]["rec_set"]) <= INSTR_TOL for n in SETS),
        'pred_b_direct_linear': all(LIN_LO <= r <= LIN_HI for n in SETS for r in lin(n).values()),
        'pred_c_dative_monotone': all(x < y for x, y in zip(fb, fb[1:])),
        'pred_d_dative_proportional': all(f is not None and abs(f - ff[1]) <= PROP_TOL for f in ff),
        'pred_e_quantifier_monotone': all(x < y for x, y in zip(conv, conv[1:])),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_damper_law_result_v1", "candidate_id": "corpus.unit_damper_law_v27",
              "semantics": "block_live_scaled_write_plus_base_freeze", "alphas": list(ALPHAS),
              "bars": {"instrument": INSTR_TOL, "linear": [LIN_LO, LIN_HI], "proportional": PROP_TOL},
              "direct_linearity": {n: lin(n) for n in SETS}, "dative_feedback": fb, "dative_feedback_fraction": ff,
              "quantifier_conversion": conv, "behaviours": report,
              "seconds": time.perf_counter() - t0, "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "direct_linearity": result["direct_linearity"], "dative_feedback": fb,
                      "dative_feedback_fraction": ff, "quantifier_conversion": conv, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
