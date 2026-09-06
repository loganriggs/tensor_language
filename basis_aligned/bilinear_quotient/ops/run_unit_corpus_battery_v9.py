#!/usr/bin/env python3
# BQGATE: frozen predictions; behaviour list, pool, target, gain floor and bars fixed before the run.
"""v9: the head-set protocol run on every selective screen that does not have a battery yet.

Every selective behaviour in the corpus went through the fast screen (whole-module interchange,
step 0) but only six -- list control_choice, modal_remoteness, polarity_state, correlative
both_vs_neither, possessive (six designs), aspectual has_vs_had -- have been through steps 1-3
(162-head sweep, greedy smallest set, block-live diff-in-means direction with complement, linearity
sum and random). This run puts the remaining SEVENTEEN through the same protocol in one enqueue,
via the new library entry `circuit_unit_greedy.block_battery` (v8's helpers moved into the library
so this and later runners do not copy them).

  behaviours (ledger: terminal, selected site set, no battery yet)
    degree_frame.comparative_vs_equative    interrogative_licensing.question_vs_declarative
    additive_scope.not_only_vs_plain        preposition_selection.on_vs_of
    voice_frame.passive_vs_active           dative_alternation.to_vs_for_v2
    degree_result.too_vs_so                 correlative_pair.both_vs_either
    polarity_licensing.never_vs_often       verb_preposition.relied_vs_objected
    coordination_agreement.and_vs_or        lexical_number.pp_intervener
    quantifier_number.each_vs_all           perfect_number.have_vs_has
    verb_complementizer.whether_vs_that     narrative_tense.past_vs_present
    correlative_state.either_vs_neither
  per behaviour: A1 rows (valid_only, dropped rows counted); 162-head sweep; greedy over the top
  12 (target 0.50, gain floor 0.02, at most 6 heads); if the set reaches 0.50 ("localized"):
  block_battery = exact-set A1 fit / held-out (odd rows) / A2 / P / C, v7 semantics control,
  block diff-in-means (even rows) with complement, S + C, random (rank 1, seed 1) on held-out and
  A2, and its P / C. Bars as in v5-v8: A2 >= 0.50, P <= 0.20, C <= 0.35; direction band
  [0.50, 1.20]; complement <= 0.30; S + C in [0.85, 1.15]; random <= 0.10; block error <= 1e-3.

REGISTERED BEFORE THE RUN
    pred_a_majority_localize        >= 9 of the 17 behaviours reach greedy joint >= 0.50 with <= 6
                                    heads. Worked example: 9 localized -> True; 8 -> False.
    pred_b_localized_sets_selective every localized set's EXACT interchange has A2 >= 0.50,
                                    P <= 0.20, C <= 0.35. Worked example: one set with A2 0.44 -> False.
    pred_c_block_control_exact      every localized set: |block full-rank - exact| <= 1e-3
                                    (the control on the new library path). Worked example: error
                                    0.0004 on all -> True.
    pred_d_direction_in_band        on >= 80% of localized sets the block diff-in-means direction
                                    has held-out AND A2 fraction in [0.50, 1.20], complement <= 0.30,
                                    S + C in [0.85, 1.15]. Worked example: 8 of 10 -> True; 7 of 10 -> False.
    pred_e_random_inert             random rank-1 direction gives |fraction| <= 0.10 on the held-out
                                    rows of every localized set. Worked example: one at 0.12 -> False.
    pred_f_hub_head                 some single head is chosen by >= 3 of the localized greedy sets.
                                    Worked example: attn:09:head:06 in 3 sets -> True; max 2 -> False.

    Priors. a unsure -- the six batteries so far all localized, but they were picked because their
    module sweeps were concentrated; the corpus screens mostly selected resid:17/18 sites. b, c, d,
    e expected from v2-v8 (every localized set so far passed them in block-live mode). f unsure:
    the possessive and aspectual sets share layer 9 but no head; number-like behaviours
    (lexical_number, quantifier_number, perfect_number, coordination_agreement) may share one.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import circuit_fast_screen_producer as producer
import circuit_unit_greedy as g

import circuit_fast_screen_candidate_degree_frame as m_degree_frame
import circuit_fast_screen_candidate_interrogative_licensing as m_interrogative
import circuit_fast_screen_candidate_additive_scope as m_additive
import circuit_fast_screen_candidate_preposition_selection as m_preposition
import circuit_fast_screen_candidate_voice_frame as m_voice
import circuit_fast_screen_candidate_dative as m_dative
import circuit_fast_screen_candidate_degree_result as m_degree_result
import circuit_fast_screen_candidate_both_either as m_both_either
import circuit_fast_screen_candidate_polarity_licensing as m_polarity_lic
import circuit_fast_screen_candidate_verb_preposition as m_verb_prep
import circuit_fast_screen_candidate_coordination_agreement as m_coord
import circuit_fast_screen_candidate_lexical_number_pp as m_lexical_number
import circuit_fast_screen_candidate_quantifier_number as m_quantifier
import circuit_fast_screen_candidate_perfect_number as m_perfect
import circuit_fast_screen_candidate_verb_complementizer as m_complementizer
import circuit_fast_screen_candidate_narrative_tense as m_narrative
import circuit_fast_screen_candidate_correlative_state as m_correlative_state

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_corpus_battery_v9_result.json"
BEHAVIOURS = {
    "degree_frame.comparative_vs_equative": m_degree_frame,
    "interrogative_licensing.question_vs_declarative": m_interrogative,
    "additive_scope.not_only_vs_plain": m_additive,
    "preposition_selection.on_vs_of": m_preposition,
    "voice_frame.passive_vs_active": m_voice,
    "dative_alternation.to_vs_for_v2": m_dative,
    "degree_result.too_vs_so": m_degree_result,
    "correlative_pair.both_vs_either": m_both_either,
    "polarity_licensing.never_vs_often": m_polarity_lic,
    "verb_preposition.relied_vs_objected": m_verb_prep,
    "coordination_agreement.and_vs_or": m_coord,
    "lexical_number.pp_intervener": m_lexical_number,
    "quantifier_number.each_vs_all": m_quantifier,
    "perfect_number.have_vs_has": m_perfect,
    "verb_complementizer.whether_vs_that": m_complementizer,
    "narrative_tense.past_vs_present": m_narrative,
    "correlative_state.either_vs_neither": m_correlative_state,
}
POOL, TARGET, MIN_GAIN, MAX_UNITS = 12, 0.50, 0.02, 6
LO, HI, COMP_BAR, SUM_LO, SUM_HI, RANDOM_MAX, P_BAR, C_BAR = 0.50, 1.20, 0.30, 0.85, 1.15, 0.10, 0.20, 0.35
BLOCK_ERR, MAJORITY, BAND_SHARE, HUB = 1e-3, 9, 0.80, 3
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 7000, 224000


def _plan():
    return {"candidate_id": "corpus.unit_corpus_battery_v9", "behaviours": list(BEHAVIOURS),
            "pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _in_band(b):
    return (b["subspace_fraction"] is not None and LO <= b["subspace_fraction"] <= HI
            and abs(b["complement_fraction"]) <= COMP_BAR
            and b["linearity_sum"] is not None and SUM_LO <= b["linearity_sum"] <= SUM_HI)


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    heads = g.all_head_units()
    t0 = time.perf_counter()
    report = {}
    for label, module in BEHAVIOURS.items():
        t1 = time.perf_counter()
        prep = g.prepare(backend, g.rows_of(module, "A1"), valid_only=True)
        singles, ranked, greedy = g.greedy_heads(backend, prep, pool=POOL, target=TARGET,
                                                 min_gain=MIN_GAIN, max_units=MAX_UNITS, units=heads)
        entry = {"rows_dropped": prep.dropped, "top_heads": {u: singles[u] for u in ranked[:POOL]},
                 "greedy": greedy, "localized": bool(greedy["reached_target"])}
        if entry["localized"]:
            entry["battery"] = g.block_battery(backend, module, greedy["chosen"])
        entry["seconds"] = time.perf_counter() - t1
        report[label] = entry
        line = {"chosen": greedy["chosen"], "joint": round(greedy["joint"], 3), "s": round(entry["seconds"], 1)}
        if entry["localized"]:
            b = entry["battery"]
            line.update({"a2": round(b["exact_set"]["a2"], 3), "P": round(b["exact_set"]["p_effect"], 3),
                         "C": round(b["exact_set"]["c_effect"], 3),
                         "dim_held": round(b["diff_in_means"]["a1_heldout"]["subspace_fraction"] or 0, 3),
                         "comp": round(b["diff_in_means"]["a1_heldout"]["complement_fraction"] or 0, 3),
                         "sum": round(b["diff_in_means"]["a1_heldout"]["linearity_sum"] or 0, 3),
                         "dim_a2": round(b["diff_in_means"]["a2"]["subspace_fraction"] or 0, 3)})
        print(label, json.dumps(line), flush=True)

    loc = {k: v for k, v in report.items() if v["localized"]}
    bats = {k: v["battery"] for k, v in loc.items()}
    in_band = [k for k, b in bats.items()
               if _in_band(b["diff_in_means"]["a1_heldout"]) and _in_band(b["diff_in_means"]["a2"])]
    head_counts = {}
    for v in loc.values():
        for u in v["greedy"]["chosen"]:
            head_counts[u] = head_counts.get(u, 0) + 1
    predictions = {
        'pred_a_majority_localize': len(loc) >= MAJORITY,
        'pred_b_localized_sets_selective': bool(bats) and all(
            b["exact_set"]["a2"] >= LO and b["exact_set"]["p_effect"] <= P_BAR
            and b["exact_set"]["c_effect"] <= C_BAR for b in bats.values()),
        'pred_c_block_control_exact': bool(bats) and all(
            b["semantics_heldout"]["block_error"] <= BLOCK_ERR for b in bats.values()),
        'pred_d_direction_in_band': bool(bats) and len(in_band) >= BAND_SHARE * len(bats),
        'pred_e_random_inert': bool(bats) and all(
            abs(b["diff_in_means"]["a1_heldout"]["random_fraction"] or 0) <= RANDOM_MAX for b in bats.values()),
        'pred_f_hub_head': bool(head_counts) and max(head_counts.values()) >= HUB,
    }
    result = {"predictions": predictions, "schema": "circuit_unit_corpus_battery_result_v1",
              "candidate_id": "corpus.unit_corpus_battery_v9", "semantics": "block_live",
              "bars": {"pool": POOL, "target": TARGET, "min_gain": MIN_GAIN, "max_units": MAX_UNITS,
                       "band": [LO, HI], "complement": COMP_BAR, "linearity_sum": [SUM_LO, SUM_HI],
                       "random": RANDOM_MAX, "P": P_BAR, "C": C_BAR, "block_error": BLOCK_ERR,
                       "majority": MAJORITY, "band_share": BAND_SHARE, "hub": HUB},
              "localized": sorted(loc), "not_localized": sorted(set(report) - set(loc)),
              "direction_in_band": in_band, "head_counts": head_counts,
              "behaviours": report, "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions, "localized": sorted(loc), "in_band": in_band,
                      "head_counts": head_counts, "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
