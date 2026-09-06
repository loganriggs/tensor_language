#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), cue pairs, rank (3 = number of fitted pairs), seeds, bars fixed before the run.
"""v15: is there a COMMON component across cue pairs -- an abstract axis -- once it is estimated across pairs?

v13/v14 showed that a direction fit on one cue pair serves another pair only 0.12-0.53 (never in band),
with overlap graded by lexical similarity. That is consistent with two pictures: (i) purely
cue-keyed axes with incidental overlap; (ii) a shared abstract component plus a cue-specific
component, where any single-pair direction is contaminated by its cue part. (ii) predicts that
POOLING pairs averages the cue parts away, so a diff-in-means over three pairs serves an unseen
fourth pair better than any single-pair direction does; (i) predicts pooling helps no more than
the best single pair. Rank-3 union of the three single-pair axes (rank fixed = number of fitted
pairs) is the ceiling: if even the span of three axes does not serve the fourth pair, the fourth
pair's axis is new.

  five behaviours, three fitted pairs each (orig + v13/v14 pairs), one HELD-OUT fourth pair:
  dative            [14:08, 07:08, 06:03, 13:08, 11:03]  sent/reserved, handed/bought, gave/kept | passed/cooked
  verb_preposition  [06:03, 13:08, 08:08]  relied/objected, depended/listened, insisted/referred | counted/spoke
  quantifier_number [07:08, 11:03]  Each/All, Neither/Several, One/Many | Either/Some
  polarity_licensing [07:08, 08:01, 04:07, 03:00]  never/often, rarely/usually, hardly/always | scarcely/frequently
  verb_complementizer [06:03, 11:03, 07:08]  wondered/remarked, asked/said, inquired/insisted | questioned/declared
  Fit rows: even rows of each fitted pair. Test rows: ALL rows of the fourth pair (never fit on), plus
  odd rows of the fitted pairs for the pooled direction's own linearity.
  Band [0.50, 1.20]; complement <= 0.30; exact bar 0.50 (fourth pair untested below it); random
  rank-matched seed 1.

REGISTERED BEFORE THE RUN
    pred_a_fourth_pair_carried    >= 4 of 5 fourth pairs have exact-set recovery >= 0.50.
                                  Worked example: 5 of 5 -> True; 3 -> False.
    pred_b_pooled_beats_single    on >= 3 tested behaviours the pooled direction's fraction on the fourth
                                  pair exceeds the BEST single-pair direction's fraction by >= 0.15.
                                  Worked example: pooled 0.62 vs best single 0.41 (x3) -> True; 0.45 vs 0.41 -> False.
    pred_c_common_axis_exists     on >= 3 tested behaviours the pooled direction serves the fourth pair
                                  in band with complement <= 0.30. Worked example: 0.71 / 0.20 (x3) -> True.
    pred_d_span_serves_fourth     on >= 3 tested behaviours the rank-3 union serves the fourth pair in band
                                  with complement <= 0.30. Worked example: 0.85 / 0.10 (x3) -> True;
                                  0.45 / 0.50 -> False (fourth axis is genuinely new).
    pred_e_pooled_keeps_own       the pooled direction stays in band with complement <= 0.30 on the odd
                                  rows of EVERY fitted pair on >= 3 tested behaviours (pooling did not
                                  destroy the fitted pairs). Worked example: 0.9 / 0.8 / 0.85 -> True; 0.4 -> False.
    Priors. a expected. b, c unsure -- v14's graded overlap (up to 0.53) makes a common component
    plausible but its size unknown; c is the abstract-variable claim, b the weaker "pooling helps".
    d likely (three axes in a 128-d block span a lot); if d FAILS the picture is purely cue-keyed.
    e expected only if a common component is large; failure of e with success of d says the pairs
    are separable but not averageable.
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
OUT = ROOT / "circuits/followups/unit_common_axis_v15_result.json"
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
LO, HI, COMP_BAR, EXACT_BAR, GAIN, NEED = 0.50, 1.20, 0.30, 0.50, 0.15, 3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 800, 20000


def _plan():
    return {"candidate_id": "corpus.unit_common_axis_v15",
            "sets": {k: {"units": v[1], "fitted": v[2], "fourth": v[3]} for k, v in SETS.items()},
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


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
        d_pool = g.block_diff_in_means(backend, p_pool, units)
        union = g.block_union(*d.values())
        r1 = g.block_random_subspace(backend, units, rank=1, seed=1)
        r3 = g.block_random_subspace(backend, units, rank=len(d), seed=1)
        on_fourth = {k: g.block_direction_battery(backend, p_fourth, units, q, r1) for k, q in d.items()}
        on_fourth["pooled"] = g.block_direction_battery(backend, p_fourth, units, d_pool, r1)
        on_fourth["union"] = g.block_direction_battery(backend, p_fourth, units, union, r3)
        pooled_own = {k: g.block_direction_battery(backend, p_odd[k], units, d_pool, r1) for k in fitted}
        exact = on_fourth["pooled"]["exact_set"]
        singles = [on_fourth[k]["subspace_fraction"] or 0 for k in fitted]
        entry = {"units": units, "fitted": maps, "fourth": fourth_map,
                 "dropped": {"fourth": p_fourth.dropped, "pool": p_pool.dropped,
                             **{f"{k}_fit": v.dropped for k, v in p_fit.items()},
                             **{f"{k}_odd": v.dropped for k, v in p_odd.items()}},
                 "exact_fourth": exact, "tested": exact >= EXACT_BAR,
                 "best_single_fraction": max(singles),
                 "pooled_fraction": on_fourth["pooled"]["subspace_fraction"],
                 "pooled_gain": (on_fourth["pooled"]["subspace_fraction"] or 0) - max(singles),
                 "pooled_band": _band(on_fourth["pooled"]), "union_band": _band(on_fourth["union"]),
                 "pooled_keeps_own": all(_band(b) for b in pooled_own.values()),
                 "pooled_cos_to_singles": {k: g.block_cosines(d_pool, q) for k, q in d.items()},
                 "on_fourth": on_fourth, "pooled_on_fitted_odd": pooled_own}
        report[name] = entry
        print(name, "exact4", round(exact, 2), "singles", [round(s, 2) for s in singles],
              "pooled", round(entry["pooled_fraction"] or 0, 2), round(on_fourth["pooled"]["complement_fraction"] or 0, 2),
              "union", round(on_fourth["union"]["subspace_fraction"] or 0, 2), round(on_fourth["union"]["complement_fraction"] or 0, 2),
              "own", [round(b["subspace_fraction"] or 0, 2) for b in pooled_own.values()], flush=True)

    tested = [e for e in report.values() if e["tested"]]
    predictions = {
        'pred_a_fourth_pair_carried': len(tested) >= 4,
        'pred_b_pooled_beats_single': sum(e["pooled_gain"] >= GAIN for e in tested) >= NEED,
        'pred_c_common_axis_exists': sum(e["pooled_band"] for e in tested) >= NEED,
        'pred_d_span_serves_fourth': sum(e["union_band"] for e in tested) >= NEED,
        'pred_e_pooled_keeps_own': sum(e["pooled_keeps_own"] for e in tested) >= NEED,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_common_axis_result_v1",
              "candidate_id": "corpus.unit_common_axis_v15", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "exact": EXACT_BAR, "gain": GAIN,
                       "need": NEED, "union_rank_per_block": 3},
              "tested": [n for n, e in report.items() if e["tested"]], "sets": report,
              "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "tested": result["tested"],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
