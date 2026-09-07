#!/usr/bin/env python3
# BQGATE: five frozen predictions; behaviour list (the five never-lifted non-Codex screens), protocol (= v113) and bars fixed before the run.
"""v190: lift the five fast-screen behaviours that NO batch rung has touched, with the v113 protocol, in one rung.

v112 lifted 21 screens and v113 re-ran its 13 row-2 misses with a larger greedy (pool 20, EVEN target 0.88, <= 14 heads):
row 2 recovered on 8/13, rows 3-5 kept on 11/13, instrument quantifier ODD 0.834 / cdas own-C UB 0.0062 / dim A1 damage 0.601.
Five candidate modules with full A1/A2/P/C row families (32 each) were in NEITHER batch and are not Codex lanes:
countability, possessive_number, possessive_attractor, animacy, existential. Their early fast screens are on record only as
prior-art files (animacy and existential once crashed the fast-screen KERNEL on a non-positive denominator; `g.prepare` handles
all five: 16 ODD rows per family, checked on CPU before writing this). Same protocol as v113, imported from v113.run (constants
identical: pool 20, target 0.88, min_gain 0.02, max_units 14; cdas rank 1, 120 steps, lr 0.05, seed 0, complement 1.0,
own-C EVEN control at weight 30); rows 2-5 as in v112; cross = CE damage on each other batch behaviour's ODD A1.
An error inside one behaviour is recorded and counts as failing every row (v112 rule).

REGISTERED BEFORE THE RUN (counts over the 5 new behaviours; quantifier_number is the instrument, separate). Base rates from
v112/v113: row 2 36 % (8/22) then 62 % (8/13); row 3 100 %; row 4 95 %; row 5 86 %.
    pred_a_localizable   row 2 (ODD A1 head-set extraction >= 0.80) holds on 2 to 5 of 5.           Worked: 2 True; 1 False.
    pred_b_row3_removal  row 3 (dim mean-removal CE damage LB975 > 0 and >= 0.10) on 4 to 5 of 5.   Worked: 4 True; 3 False.
    pred_c_row4_specific row 4 (cdas own-C ODD UB975 <= 0.01) on 4 to 5 of 5.                        Worked: 5 True; 3 False.
    pred_d_row5_a2       row 5 (dim A2 damage LB975 > 0 and >= 0.5 x A1) on 3 to 5 of 5.             Worked: 3 True; 2 False.
    pred_e_instrument    quantifier_number reproduces v113: ODD extraction within 0.834 +- 0.03, cdas own-C UB <= 0.03, and the
                         dim A1 CE damage within 0.601 +- 0.06 (same seed, same rows -> the module must give the old digest).
                         Worked: 0.834 / 0.006 / 0.601 True; 0.79 False.
    Prior: a 55 %; b 80 %; c 70 %; d 65 %; e 85 %.
No CPU smoke of the greedy (162-head sweep is GPU-only in this protocol); the CPU check covered module load + prepare for all five.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import run_unit_tier3_batch_row2_v113 as v113

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_tier3_batch_unlifted_v190_result.json"
BATCH = {"countability": "countability", "possessive_number": "possessive_number", "possessive_attractor": "possessive_attractor",
         "animacy": "animacy", "existential": "existential"}
INSTRUMENT = {"quantifier_number": "quantifier_number"}
BARS = {"k_a": [2, 5], "k_b": [4, 5], "k_c": [4, 5], "k_d": [3, 5], "instr_ext": [0.834, 0.03], "instr_c_ub": 0.03, "instr_dim_a1": [0.601, 0.06]}
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 8000, 256000


def _plan():
    return {"candidate_id": "corpus.unit_tier3_batch_unlifted_v190", "behaviours": len(BATCH), "instrument": list(INSTRUMENT),
            "model_forwards_max": MODEL_FORWARDS_MAX, "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": (len(BATCH) + 1) * v113.STEPS, "model_updates": 0, "fit_parameters": (len(BATCH) + 1) * 14 * 128,
            "gpu_accessed": False, "model_loaded": False, "execution_policy": "managed_queue_only"}


def PREDS(R):
    B = BARS
    new = list(BATCH)
    count = lambda row: sum(R[n]["rows"][row] for n in new)
    inb = lambda x, b: x is not None and b[0] <= x <= b[1]
    qi = R["quantifier_number"]
    ext_ok = inb(qi.get("extraction_odd"), [B["instr_ext"][0] - B["instr_ext"][1], B["instr_ext"][0] + B["instr_ext"][1]])
    dim_ok = "arms" in qi and inb(qi["arms"]["dim"]["A1"]["ce_damage"], [B["instr_dim_a1"][0] - B["instr_dim_a1"][1], B["instr_dim_a1"][0] + B["instr_dim_a1"][1]])
    c_ok = "arms" in qi and qi["arms"]["cdas"]["C"]["ce_ub975"] <= B["instr_c_ub"]
    return {"pred_a_localizable": inb(count("row2"), B["k_a"]), "pred_b_row3_removal": inb(count("row3"), B["k_b"]),
            "pred_c_row4_specific": inb(count("row4"), B["k_c"]), "pred_d_row5_a2": inb(count("row5"), B["k_d"]),
            "pred_e_instrument": bool(ext_ok and dim_ok and c_ok)}


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    R, secs = v113.run({**BATCH, **INSTRUMENT}, OUT, "corpus.unit_tier3_batch_unlifted_v190")
    new = list(BATCH)
    predictions = PREDS(R)
    count = lambda row: sum(R[n]["rows"][row] for n in new)
    tiers = {n: ("tier3_all_rows" if all(R[n]["rows"].values()) else "tier3_rows_" + "".join(k[-1] for k, v in R[n]["rows"].items() if v) if R[n]["rows"]["row2"]
                 else "not_localizable" if "error" not in R[n] else "error") for n in R}
    summary = {n: {"n_units": R[n].get("n_units"), "ext_even": R[n].get("extraction_even"), "ext_odd": R[n].get("extraction_odd"), "rows": R[n]["rows"], "tier": tiers[n],
                   "cross_abs_max": {a: v["cross_abs_max"] for a, v in R[n].get("arms", {}).items()}} for n in R}
    result = {"predictions": predictions, "schema": "circuit_unit_tier3_batch_unlifted_result_v1", "candidate_id": "corpus.unit_tier3_batch_unlifted_v190",
              "counts": {row: count(row) for row in ("row2", "row3", "row4", "row5")}, "all_four": sum(all(R[n]["rows"].values()) for n in new),
              "errors": {n: R[n]["error"] for n in R if "error" in R[n]}, "summary": summary, "behaviours": R,
              "protocol": {"pool": v113.POOL, "target": v113.TARGET, "min_gain": v113.MIN_GAIN, "max_units": v113.MAX_UNITS, "lambda": v113.LAM,
                           "steps": v113.STEPS, "lr": v113.LR, "complement_weight": v113.CW, "controls": "own C EVEN only", "source": "v113.run"},
              "bars": {**BARS, "ext_min": v113.EXT_MIN, "rem_min": v113.REM_MIN, "c_ub_max": v113.C_UB_MAX, "a2_frac": v113.A2_FRAC},
              "seconds": round(secs, 1), "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps({"predictions": predictions, "counts": result["counts"], "all_four": result["all_four"], "tiers": tiers,
                      "errors": result["errors"], "seconds": result["seconds"]}, indent=2, default=str))


if __name__ == "__main__":
    main()
