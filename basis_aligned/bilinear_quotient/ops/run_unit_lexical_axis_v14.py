#!/usr/bin/env python3
# BQGATE: frozen predictions; head sets (v9 receipt), substitute pairs, seeds and bars fixed before the run.
"""v14: is v13's verb-keyed axis the rule across the corpus? Same frame, new lexical cue pair, 4 behaviours.

v13 (`unit_dative_verb_axis_v13_result.json`) held the dative frame fixed, swapped only the verb pair,
and found one axis PER VERB PAIR in the head set (own axis 1.02 / 1.03; union of the A1+A2 axes serves
a new pair 0.31-0.33; median per-block |cos| 0.12-0.38). Every other v9 behaviour has a single
lexical cue pair in both A1 and A2 (checked: 2 cue pairs x 16 rows, forward/reverse), so v9's
held-out and A2 transfer never left the cue pair. This run asks the same question for four
behaviours whose frames take a drop-in substitute (`g.lexical_variant`, which reproduces v13's rows
exactly): two fresh cue pairs each, on the v9 head sets, batteries on odd rows only.

  verb_preposition  [06:03, 13:08, 08:08]        relied/objected  -> depended/listened, insisted/referred
  quantifier_number [07:08, 11:03]               Each/All         -> Neither/Several, One/Many
  polarity_licensing [07:08, 08:01, 04:07, 03:00] never/often     -> rarely/usually, hardly/always
  verb_complementizer [06:03, 11:03, 07:08]      wondered/remarked -> asked/said, inquired/insisted
  d0 = block dim on the original A1 even rows; dk = block dim on variant k even rows.
  Band [0.50, 1.20]; complement <= 0.30; random rank-matched seed 1; cos bar 0.50; exact bar 0.50;
  not-served bar 0.35. A pair whose exact-set recovery on odd rows is < 0.50 is UNTESTED, not a null.

REGISTERED BEFORE THE RUN
    pred_a_exact_set_transfers   >= 6 of the 8 variant pairs have exact-set odd recovery >= 0.50 (the
                                 head set carries the new pair). Worked example: 7 of 8 -> True; 5 -> False.
    pred_b_variants_own_axis     every TESTED pair's own direction is in band with complement <= 0.30 on
                                 its odd rows. Worked example: 1.01 / 0.02 on all -> True; one 0.44 -> False.
    pred_c_original_axis_serves  the ORIGINAL A1 direction d0 serves the tested variant pairs in band
                                 with complement <= 0.30 on at least HALF of them (abstract axis is the
                                 rule). Worked example: 5 of 7 -> True; 2 of 7 -> False.
    pred_d_lexical_is_the_rule   d0 serves < 0.35 with the variant's own direction at median per-block
                                 |cos| < 0.50 to d0 on at least HALF of the tested pairs (verb-keyed is
                                 the rule). Worked example: 5 of 7 (0.22, cos 0.31) -> True; 2 of 7 -> False.
    pred_e_number_axis_abstract  quantifier_number specifically: d0 serves BOTH variant pairs in band
                                 (v11 showed 11:03's number axis is shared within the number family, so
                                 Neither/Several and One/Many should ride the same axis).
                                 Worked example: 0.92 / 0.03 and 0.88 / 0.05 -> True; One/Many 0.30 -> False.
    Priors. a expected. b expected. c and d are complementary; after v13 I lean d, but e is the one
    case where an abstract axis is already documented, so I expect d to hold on the three non-number
    behaviours and e to hold -- a split verdict is the informative outcome.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_verb_preposition as m_verb_prep
import circuit_fast_screen_candidate_quantifier_number as m_quantifier
import circuit_fast_screen_candidate_polarity_licensing as m_polarity
import circuit_fast_screen_candidate_verb_complementizer as m_complementizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_lexical_axis_v14_result.json"
SETS = {
    "verb_preposition": (m_verb_prep, ["attn:06:head:03", "attn:13:head:08", "attn:08:head:08"],
                         [{"relied": "depended", "objected": "listened"}, {"relied": "insisted", "objected": "referred"}]),
    "quantifier_number": (m_quantifier, ["attn:07:head:08", "attn:11:head:03"],
                          [{"Each": "Neither", "All": "Several"}, {"Each": "One", "All": "Many"}]),
    "polarity_licensing": (m_polarity, ["attn:07:head:08", "attn:08:head:01", "attn:04:head:07", "attn:03:head:00"],
                           [{"never": "rarely", "often": "usually"}, {"never": "hardly", "often": "always"}]),
    "verb_complementizer": (m_complementizer, ["attn:06:head:03", "attn:11:head:03", "attn:07:head:08"],
                            [{"wondered": "asked", "remarked": "said"}, {"wondered": "inquired", "remarked": "insisted"}]),
}
LO, HI, COMP_BAR, COS_BAR, EXACT_BAR, NOT_SERVED = 0.50, 1.20, 0.30, 0.50, 0.50, 0.35
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 600, 16000


def _plan():
    return {"candidate_id": "corpus.unit_lexical_axis_v14",
            "sets": {k: {"units": v[1], "pairs": v[2]} for k, v in SETS.items()},
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
    report, pairs = {}, []   # pairs: one entry per (behaviour, variant)
    for name, (module, units, maps) in SETS.items():
        a1 = g.rows_of(module, "A1")
        fams = {"orig": a1, **{f"v{k + 1}": g.lexical_variant(a1, mp) for k, mp in enumerate(maps)}}
        p = {}
        for k, rows in fams.items():
            p[f"{k}_fit"] = g.prepare(backend, rows[0::2], valid_only=True)
            p[f"{k}_odd"] = g.prepare(backend, rows[1::2], valid_only=True)
        d = {k: g.block_diff_in_means(backend, p[f"{k}_fit"], units) for k in fams}
        r1 = g.block_random_subspace(backend, units, rank=1, seed=1)
        B = {dn: {ev: g.block_direction_battery(backend, p[f"{ev}_odd"], units, d[dn], r1) for ev in fams}
             for dn in fams}
        cos = {k: g.block_cosines(d[k], d["orig"]) for k in fams if k != "orig"}
        med = {k: statistics.median(v.values()) for k, v in cos.items()}
        entry = {"units": units, "maps": maps, "dropped": {k: v.dropped for k, v in p.items()},
                 "exact_set_odd": {ev: B["orig"][ev]["exact_set"] for ev in fams},
                 "median_cos_to_orig": med, "cos_to_orig": cos, "batteries": B}
        report[name] = entry
        for k in fams:
            if k == "orig":
                continue
            ex = entry["exact_set_odd"][k]
            pairs.append({"behaviour": name, "variant": k, "map": maps[int(k[1:]) - 1], "exact": ex,
                          "tested": ex >= EXACT_BAR, "own_band": _band(B[k][k]),
                          "orig_band": _band(B["orig"][k]),
                          "orig_fraction": B["orig"][k]["subspace_fraction"], "median_cos": med[k],
                          "lexical": (B["orig"][k]["subspace_fraction"] or 0) < NOT_SERVED and med[k] < COS_BAR})
        print(name, "exact", {k: round(v, 2) for k, v in entry["exact_set_odd"].items()},
              "cos", {k: round(v, 2) for k, v in med.items()},
              {dn: {ev: (round(b["subspace_fraction"] or 0, 2), round(b["complement_fraction"] or 0, 2))
                    for ev, b in bs.items()} for dn, bs in B.items()}, flush=True)

    tested = [q for q in pairs if q["tested"]]
    n_t = len(tested)
    predictions = {
        'pred_a_exact_set_transfers': n_t >= 6,
        'pred_b_variants_own_axis': n_t > 0 and all(q["own_band"] for q in tested),
        'pred_c_original_axis_serves': n_t > 0 and sum(q["orig_band"] for q in tested) * 2 >= n_t,
        'pred_d_lexical_is_the_rule': n_t > 0 and sum(q["lexical"] for q in tested) * 2 >= n_t,
        'pred_e_number_axis_abstract': all(q["tested"] and q["orig_band"] for q in pairs
                                           if q["behaviour"] == "quantifier_number"),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_lexical_axis_result_v1",
              "candidate_id": "corpus.unit_lexical_axis_v14", "semantics": "block_live",
              "bars": {"band": [LO, HI], "complement": COMP_BAR, "cos": COS_BAR, "exact": EXACT_BAR,
                       "not_served": NOT_SERVED},
              "pairs": pairs, "tested_pairs": n_t, "sets": report,
              "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "tested": n_t,
                      "pairs": [(q["behaviour"], q["variant"], round(q["exact"], 2), q["own_band"], q["orig_band"],
                                 round(q["orig_fraction"] or 0, 2), round(q["median_cos"], 2)) for q in pairs],
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
