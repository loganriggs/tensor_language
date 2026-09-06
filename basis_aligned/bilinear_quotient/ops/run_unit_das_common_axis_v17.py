#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), cue pairs, DAS rank 1 / steps / lr / seed / inertness weight, bars fixed before the run.
"""v17: v15's question with the estimator v16 prescribes -- is there a common cue-pair axis, estimated safely?

v15 pooled diff-in-means over three cue pairs and tested on an unseen fourth: polarity in band (0.93),
quantifier over band (1.36), dative not served (0.49). v16 showed the pooled diff-in-means is an unsafe
estimator (the mean of axes ~0.8 apart is a steeper off-axis direction: 1.04 -> 1.57 monotonically
along the rotation, both blocks) and that block DAS rank 1 with complement inertness on the same
rows gives a sane common axis (Either/Some 1.13 / -0.07). This run replaces the estimator and keeps
everything else from v15: same five behaviours, same three fitted pairs (even rows), same held-out
fourth pair, rank 1 per block FIXED, complement weight 1.0, 120 steps, lr 0.05, seed 0. Rank-3 union
of per-pair diff-in-means is kept as the registered ceiling.
  Band [0.50, 1.20]; complement <= 0.30; S + C in [0.85, 1.15]; exact bar 0.50 (fourth pair untested
  below it); over-band = any fraction >= 1.20.

REGISTERED BEFORE THE RUN
    pred_a_das_serves_fourth      on >= 3 tested behaviours the DAS axis serves the fourth pair in band
                                  with complement <= 0.30. Worked example: 0.88 / 0.10 x3 -> True; x2 -> False.
    pred_b_das_keeps_fitted       on >= 3 tested behaviours the DAS axis has S + C in [0.85, 1.15] on the
                                  odd rows of EVERY fitted pair. Worked example: 0.98 / 1.02 / 0.95 -> True;
                                  one 1.19 -> False (v16's quantifier case).
    pred_c_estimator_safe         NO DAS fraction >= 1.20 on any fitted-odd or fourth-pair battery of any
                                  tested behaviour. Worked example: max 1.13 -> True; 1.36 -> False.
    pred_d_polarity_common_axis   polarity_licensing: DAS axis in band with complement <= 0.30 on the
                                  fourth pair (v15's one clean common axis survives the estimator change).
                                  Worked example: 0.91 / 0.02 -> True; 0.40 / 0.50 -> False.
    pred_e_dative_stays_keyed     dative: DAS axis NOT in band on the fourth pair (fraction < 0.50 or
                                  complement > 0.30). Worked example: 0.45 / 0.48 -> True; 0.80 / 0.15 -> False.
    Priors. c expected (that is what inertness is for; if it FAILS the overshoot is a property of the
    set, not of averaging). a, b unsure -- v16's quantifier axis served the unseen pair at 1.13 but
    S + C 1.19 on a fitted pair. d expected. e expected (v13: passed/cooked is a new axis); if e FAILS
    the dative "verb-keyed" reading was itself a diff-in-means artifact and must be retracted.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_dative as m_dative
import circuit_fast_screen_candidate_verb_preposition as m_verb_prep
import circuit_fast_screen_candidate_quantifier_number as m_quantifier
import circuit_fast_screen_candidate_polarity_licensing as m_polarity
import circuit_fast_screen_candidate_verb_complementizer as m_complementizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_das_common_axis_v17_result.json"
SETS = {
    "dative": (m_dative, ["attn:14:head:08", "attn:07:head:08", "attn:06:head:03", "attn:13:head:08", "attn:11:head:03"],
               [{"sent": "handed", "reserved": "bought"}, {"sent": "gave", "reserved": "kept"}],
               {"sent": "passed", "reserved": "cooked"}),
    "verb_preposition": (m_verb_prep, ["attn:06:head:03", "attn:13:head:08", "attn:08:head:08"],
                         [{"relied": "depended", "objected": "listened"}, {"relied": "insisted", "objected": "referred"}],
                         {"relied": "counted", "objected": "spoke"}),
    "quantifier_number": (m_quantifier, ["attn:07:head:08", "attn:11:head:03"],
                          [{"Each": "Neither", "All": "Several"}, {"Each": "One", "All": "Many"}],
                          {"Each": "Either", "All": "Some"}),
    "polarity_licensing": (m_polarity, ["attn:07:head:08", "attn:08:head:01", "attn:04:head:07", "attn:03:head:00"],
                           [{"never": "rarely", "often": "usually"}, {"never": "hardly", "often": "always"}],
                           {"never": "scarcely", "often": "frequently"}),
    "verb_complementizer": (m_complementizer, ["attn:06:head:03", "attn:11:head:03", "attn:07:head:08"],
                            [{"wondered": "asked", "remarked": "said"}, {"wondered": "inquired", "remarked": "insisted"}],
                            {"wondered": "questioned", "remarked": "declared"}),
}
LO, HI, COMP_BAR, EXACT_BAR, NEED, SUM_LO, SUM_HI = 0.50, 1.20, 0.30, 0.50, 3, 0.85, 1.15
STEPS, LR, CW = 120, 0.05, 1.0
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 1200, 30000


def _plan():
    return {"candidate_id": "corpus.unit_das_common_axis_v17",
            "sets": {k: {"units": v[1], "fitted": v[2], "fourth": v[3]} for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 5 * STEPS, "model_updates": 0, "fit_parameters": 5 * 5 * 128,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _sum_ok(b):
    return b["linearity_sum"] is not None and SUM_LO <= b["linearity_sum"] <= SUM_HI


def _band(b):
    return b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI \
        and abs(b["complement_fraction"]) <= COMP_BAR


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return

    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    report = {}
    for name, (module, units, maps, fourth_map) in SETS.items():
        a1 = g.rows_of(module, "A1")
        fitted = {"orig": a1, **{f"v{k + 1}": g.lexical_variant(a1, mp) for k, mp in enumerate(maps)}}
        fourth = g.lexical_variant(a1, fourth_map)
        p_fit = {k: g.prepare(backend, rows[0::2], valid_only=True) for k, rows in fitted.items()}
        p_odd = {k: g.prepare(backend, rows[1::2], valid_only=True) for k, rows in fitted.items()}
        p_fourth = g.prepare(backend, fourth, valid_only=True)
        pooled_rows = [r for k in fitted for r in fitted[k][0::2]]
        p_pool = g.prepare(backend, pooled_rows, valid_only=True)
        d = {k: g.block_diff_in_means(backend, p_fit[k], units) for k in fitted}
        q_das, hist = g.fit_block_subspace(backend, p_pool, units, rank=1, steps=STEPS, lr=LR, seed=0, complement_weight=CW)
        union = g.block_union(*d.values())
        r1 = g.block_random_subspace(backend, units, rank=1, seed=1)
        r3 = g.block_random_subspace(backend, units, rank=len(d), seed=1)
        on_fourth = {"das": g.block_direction_battery(backend, p_fourth, units, q_das, r1),
                     "union": g.block_direction_battery(backend, p_fourth, units, union, r3)}
        das_own = {k: g.block_direction_battery(backend, p_odd[k], units, q_das, r1) for k in fitted}
        exact = on_fourth["das"]["exact_set"]
        fr_all = [on_fourth["das"]["subspace_fraction"] or 0] + [b["subspace_fraction"] or 0 for b in das_own.values()]
        entry = {"units": units, "fitted": maps, "fourth": fourth_map,
                 "dropped": {"fourth": p_fourth.dropped, "pool": p_pool.dropped,
                             **{f"{k}_fit": v.dropped for k, v in p_fit.items()},
                             **{f"{k}_odd": v.dropped for k, v in p_odd.items()}},
                 "exact_fourth": exact, "tested": exact >= EXACT_BAR,
                 "das_fourth_band": _band(on_fourth["das"]), "union_fourth_band": _band(on_fourth["union"]),
                 "das_keeps_fitted": all(_sum_ok(b) for b in das_own.values()),
                 "max_das_fraction": max(fr_all), "das_loss_history": hist,
                 "das_cos_to_pairs": {k: g.block_cosines(q_das, q) for k, q in d.items()},
                 "on_fourth": on_fourth, "das_on_fitted_odd": das_own}
        report[name] = entry
        print(name, "exact4", round(exact, 2),
              "das4", round(on_fourth["das"]["subspace_fraction"] or 0, 2), round(on_fourth["das"]["complement_fraction"] or 0, 2),
              "union4", round(on_fourth["union"]["subspace_fraction"] or 0, 2), round(on_fourth["union"]["complement_fraction"] or 0, 2),
              "own", [(round(b["subspace_fraction"] or 0, 2), round(b["linearity_sum"] or 0, 2)) for b in das_own.values()], flush=True)

    tested = [e for e in report.values() if e["tested"]]
    predictions = {
        'pred_a_das_serves_fourth': sum(e["das_fourth_band"] for e in tested) >= NEED,
        'pred_b_das_keeps_fitted': sum(e["das_keeps_fitted"] for e in tested) >= NEED,
        'pred_c_estimator_safe': bool(tested) and all(e["max_das_fraction"] < HI for e in tested),
        'pred_d_polarity_common_axis': report["polarity_licensing"]["tested"] and report["polarity_licensing"]["das_fourth_band"],
        'pred_e_dative_stays_keyed': report["dative"]["tested"] and not report["dative"]["das_fourth_band"],
    }
    result = {"predictions": predictions, "schema": "circuit_unit_das_common_axis_result_v1",
              "candidate_id": "corpus.unit_das_common_axis_v17", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "exact": EXACT_BAR, "need": NEED,
                       "linearity_sum": [SUM_LO, SUM_HI], "union_rank_per_block": 3,
                       "das": {"rank": 1, "steps": STEPS, "lr": LR, "seed": 0, "complement_weight": CW}},
              "tested": [n for n, e in report.items() if e["tested"]], "sets": report,
              "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "tested": result["tested"],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
