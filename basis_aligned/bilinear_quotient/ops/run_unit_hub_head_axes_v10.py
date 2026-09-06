#!/usr/bin/env python3
# BQGATE: frozen predictions; hub heads, memberships (from the terminal v9 receipt) and bars fixed before the run.
"""v10: are the hub heads' directions one shared axis or one axis per behaviour?

v9 (`unit_corpus_battery_v9_result.json`) localized 17 behaviours to 1-5 heads each and found the
sets drawn from a small shared pool: attn:07:head:08 sits in 11 of the 17 greedy sets, attn:11:head:03
(the task14 number head) in 7, attn:08:head:01 (the aspectual head) in 6. Two readings are open:
  (i)  a hub head carries ONE axis that many behaviours read (a general "which alternative" or
       answer-writing direction), or
  (ii) it carries a DIFFERENT 128-d direction per behaviour (a multiplexed head).
The two are separated by fitting the head's block diff-in-means direction on each behaviour's even
A1 rows separately, then (a) the pairwise |cosine| matrix and (b) CROSS-PATCHING: behaviour i's
direction applied (block-live) to behaviour j's odd rows, as a fraction of j's own exact single-head
effect. A random 128-d pair has |cos| ~ 0.07; two directions that serve each other's behaviour at
>= 0.50 of the exact effect share the axis causally, not just geometrically.

  hubs and memberships (from v9, fixed here)
    attn:07:head:08  coordination_agreement, correlative_pair.both_vs_either, dative_v2,
                     degree_frame, degree_result, interrogative_licensing, polarity_licensing,
                     preposition_selection, quantifier_number, verb_complementizer, voice_frame
    attn:11:head:03  coordination_agreement, dative_v2, lexical_number, narrative_tense,
                     perfect_number, quantifier_number, verb_complementizer
    attn:08:head:01  additive_scope, correlative_pair.both_vs_either, correlative_state,
                     degree_frame, degree_result, polarity_licensing
  per hub: for each member behaviour, prep even (fit) / odd (evaluation) A1 rows; exact single-head
  effect on odd rows; block diff-in-means on even rows; |cos| matrix; cross-patch matrix
  (direction_i on rows_j, fraction of j's exact single-head effect, complement fraction too).
  Pairs count only where BOTH behaviours' exact single-head effect on odd rows >= 0.10 (v9 singles
  on all rows: 0.10-0.36 for 07:08, 0.06-0.59 for 11:03, 0.09-0.38 for 08:01), so a fraction never
  divides by noise; the number of counted pairs is reported.
  Answer tokens: each behaviour's donor/base answer-id set is recorded; "token-disjoint" pairs
  share no answer id.

REGISTERED BEFORE THE RUN
    pred_a_0708_shared_geometry     attn:07:head:08: median pairwise |cos| over its 11 behaviours
                                    >= 0.70. Worked example: median 0.74 -> True; 0.55 -> False.
    pred_b_0708_shared_causally     attn:07:head:08: on >= 50% of counted ordered pairs (i != j),
                                    direction_i gives >= 0.50 of j's exact single-head effect.
                                    Worked example: 48 of 80 counted pairs -> True; 30 of 80 -> False.
    pred_c_0708_not_a_token_axis    attn:07:head:08: the median |cos| over token-DISJOINT pairs is
                                    >= 0.70 as well (the sharing is not "same answer token").
                                    Worked example: disjoint-pair median 0.72 -> True; 0.40 -> False.
    pred_d_1103_shared_geometry     attn:11:head:03: median pairwise |cos| over its 7 behaviours >= 0.70.
                                    Worked example: 0.81 -> True.
    pred_e_0801_shared_geometry     attn:08:head:01: median pairwise |cos| over its 6 behaviours >= 0.70.
    pred_f_multiplexed_somewhere    at least one hub has median pairwise |cos| <= 0.40 (reading ii
                                    for that head). Worked example: 08:01 median 0.31 -> True;
                                    all three >= 0.41 -> False.

    Priors. a, b: unsure, leaning shared -- 07:08 enters sets for behaviours with unrelated answer
    tokens (by/the, on/of, each/all, whether/that), which is what a general axis looks like; c is
    the control that decides whether "shared" means anything beyond token identity. d: expected
    (11:03 is the number head; five of its seven members are number-like, and the number axis
    was shown to be one direction in task14). e unsure. f: I expect it to FAIL (no multiplexed
    hub) but register it so the multiplexed reading has a registered pass condition.
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

import circuit_fast_screen_candidate_degree_frame as m_degree_frame
import circuit_fast_screen_candidate_interrogative_licensing as m_interrogative
import circuit_fast_screen_candidate_additive_scope as m_additive
import circuit_fast_screen_candidate_preposition_selection as m_preposition
import circuit_fast_screen_candidate_voice_frame as m_voice
import circuit_fast_screen_candidate_dative as m_dative
import circuit_fast_screen_candidate_degree_result as m_degree_result
import circuit_fast_screen_candidate_both_either as m_both_either
import circuit_fast_screen_candidate_polarity_licensing as m_polarity_lic
import circuit_fast_screen_candidate_coordination_agreement as m_coord
import circuit_fast_screen_candidate_lexical_number_pp as m_lexical_number
import circuit_fast_screen_candidate_quantifier_number as m_quantifier
import circuit_fast_screen_candidate_perfect_number as m_perfect
import circuit_fast_screen_candidate_verb_complementizer as m_complementizer
import circuit_fast_screen_candidate_narrative_tense as m_narrative
import circuit_fast_screen_candidate_correlative_state as m_correlative_state

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "circuits/followups/unit_hub_head_axes_v10_result.json"
MODULES = {
    "coordination_agreement": m_coord, "correlative_pair_both_either": m_both_either,
    "dative_v2": m_dative, "degree_frame": m_degree_frame, "degree_result": m_degree_result,
    "interrogative_licensing": m_interrogative, "polarity_licensing": m_polarity_lic,
    "preposition_selection": m_preposition, "quantifier_number": m_quantifier,
    "verb_complementizer": m_complementizer, "voice_frame": m_voice,
    "lexical_number": m_lexical_number, "narrative_tense": m_narrative, "perfect_number": m_perfect,
    "additive_scope": m_additive, "correlative_state": m_correlative_state,
}
HUBS = {
    "attn:07:head:08": ["coordination_agreement", "correlative_pair_both_either", "dative_v2",
                        "degree_frame", "degree_result", "interrogative_licensing",
                        "polarity_licensing", "preposition_selection", "quantifier_number",
                        "verb_complementizer", "voice_frame"],
    "attn:11:head:03": ["coordination_agreement", "dative_v2", "lexical_number", "narrative_tense",
                        "perfect_number", "quantifier_number", "verb_complementizer"],
    "attn:08:head:01": ["additive_scope", "correlative_pair_both_either", "correlative_state",
                        "degree_frame", "degree_result", "polarity_licensing"],
}
COS_BAR, MULTIPLEX_BAR, SERVE_BAR, PAIR_SHARE, EFFECT_FLOOR = 0.70, 0.40, 0.50, 0.50, 0.10
MODEL_FORWARDS_MAX, EXAMPLE_EVALUATIONS_MAX = 1200, 40000


def _plan():
    return {"candidate_id": "corpus.unit_hub_head_axes_v10", "hubs": HUBS,
            "model_forwards_max": MODEL_FORWARDS_MAX,
            "example_evaluations_max": EXAMPLE_EVALUATIONS_MAX,
            "model_backwards": 0, "model_updates": 0, "fit_parameters": 0,
            "gpu_accessed": False, "model_loaded": False,
            "execution_policy": "managed_queue_only"}


def _answer_ids(module):
    return sorted({r[k] for r in g.rows_of(module, "A1") for k in ("base_answer_id", "donor_answer_id")})


def main() -> None:
    if os.environ.get("BQLIB_DRYRUN") or os.environ.get("BQLIB_NO_MODEL"):
        print(json.dumps(_plan(), indent=2, sort_keys=True)); return
    backend = producer.Bilin18TorchBackend.load("cuda")
    t0 = time.perf_counter()
    preps = {}
    for name, module in MODULES.items():
        a1 = g.rows_of(module, "A1")
        preps[name] = (g.prepare(backend, a1[0::2]), g.prepare(backend, a1[1::2]))
    tokens = {name: _answer_ids(module) for name, module in MODULES.items()}

    report = {}
    for hub, members in HUBS.items():
        units = [hub]
        dirs = {m: g.block_diff_in_means(backend, preps[m][0], units) for m in members}
        exact = {m: g.recovery(preps[m][1], g.patched_axis(backend, preps[m][1], units)) for m in members}
        key = next(iter(dirs[members[0]]))
        cos = {i: {j: float((dirs[i][key][:, 0] @ dirs[j][key][:, 0]).abs()) for j in members} for i in members}
        cross, counted, served = {}, 0, 0
        for i in members:
            cross[i] = {}
            for j in members:
                prep = preps[j][1]
                sub = g.recovery(prep, g.patched_axis(backend, prep, units, q=dirs[i]))
                comp = g.recovery(prep, g.patched_axis(backend, prep, units, q=dirs[i], complement=True))
                e = exact[j]
                frac = (sub / e) if abs(e) > 1e-6 else None
                cross[i][j] = {"subspace": sub, "complement": comp, "fraction": frac,
                               "complement_fraction": (comp / e) if abs(e) > 1e-6 else None}
                if i != j and exact[i] >= EFFECT_FLOOR and exact[j] >= EFFECT_FLOOR:
                    counted += 1
                    served += int(frac is not None and frac >= SERVE_BAR)
        pairs = [(i, j) for a, i in enumerate(members) for j in members[a + 1:]]
        all_cos = [cos[i][j] for i, j in pairs]
        disjoint = [cos[i][j] for i, j in pairs if not set(tokens[i]) & set(tokens[j])]
        report[hub] = {"members": members, "exact_single_head_oddrows": exact,
                       "cosines": cos, "median_cos": statistics.median(all_cos),
                       "median_cos_token_disjoint": statistics.median(disjoint) if disjoint else None,
                       "token_disjoint_pairs": len(disjoint), "pairs": len(pairs),
                       "cross": cross, "counted_ordered_pairs": counted, "served_pairs": served,
                       "served_share": (served / counted) if counted else None,
                       "self_fraction": {m: cross[m][m]["fraction"] for m in members}}
        print(hub, json.dumps({"median_cos": round(report[hub]["median_cos"], 3),
                               "disjoint": (round(report[hub]["median_cos_token_disjoint"], 3)
                                            if disjoint else None),
                               "served": f"{served}/{counted}",
                               "exact": {m: round(e, 2) for m, e in exact.items()},
                               "self": {m: round(v or 0, 2) for m, v in report[hub]["self_fraction"].items()}}),
              flush=True)

    h78, h113, h81 = report["attn:07:head:08"], report["attn:11:head:03"], report["attn:08:head:01"]
    predictions = {
        'pred_a_0708_shared_geometry': h78["median_cos"] >= COS_BAR,
        'pred_b_0708_shared_causally': h78["served_share"] is not None and h78["served_share"] >= PAIR_SHARE,
        'pred_c_0708_not_a_token_axis': (h78["median_cos_token_disjoint"] is not None
                                         and h78["median_cos_token_disjoint"] >= COS_BAR),
        'pred_d_1103_shared_geometry': h113["median_cos"] >= COS_BAR,
        'pred_e_0801_shared_geometry': h81["median_cos"] >= COS_BAR,
        'pred_f_multiplexed_somewhere': any(r["median_cos"] <= MULTIPLEX_BAR for r in report.values()),
    }
    result = {"predictions": predictions, "schema": "circuit_unit_hub_head_axes_result_v1",
              "candidate_id": "corpus.unit_hub_head_axes_v10", "semantics": "block_live",
              "bars": {"cos": COS_BAR, "multiplex": MULTIPLEX_BAR, "serve": SERVE_BAR,
                       "pair_share": PAIR_SHARE, "effect_floor": EFFECT_FLOOR},
              "answer_ids": tokens, "hubs": report, "seconds": time.perf_counter() - t0,
              "finished_utc": datetime.now(timezone.utc).isoformat()}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"predictions": predictions,
                      "summary": {h: {"median_cos": round(r["median_cos"], 3),
                                      "served": f"{r['served_pairs']}/{r['counted_ordered_pairs']}"}
                                  for h, r in report.items()},
                      "seconds": round(result["seconds"], 1)}, indent=2))


if __name__ == "__main__":
    main()
