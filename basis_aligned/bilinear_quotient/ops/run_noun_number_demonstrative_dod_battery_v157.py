#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Demonstrative number these/this (ones vs one) DoD battery (v157): the atlas top-4 set {11.2, 7.8, 6.3, 15.1} on FRESH rows -- the atlas residue's
NOUN-NUMBER cluster (determiner / numeral / demonstrative -> plural vs singular pro-form or noun; four live lines at 0.18-0.28, head 11.2 alone 0.40-0.66).

Lane: Claude circuit lane. Frozen before access: F* = 0.28 +/- 0.15 (pred_g). Readers set explicitly and number-free: will-would, who-which, night-day.
ROWS (fresh; `dod_lexicon.fresh` over the screened pools and an adjective list screened here): 16 fresh agents x 16 fresh adjectives, three frames the
module does not use, {these -> " ones", this -> " one"}:
    wanted   "Yesterday the {agent} wanted {det} {adj}"
    kept     "The {agent} by the {obj} kept {det} {adj}"
    counted  "At dawn the {agent} counted {det} {adj}"
96 rows in 3 batches. Battery as v71 (`dod_battery.run`). Kill (registered): LIVE or null failing -> the cluster is recorded as 11.2 co-occurrence.
PREDICTIONS (scored as written by `dod_battery.run`; failures preserved): as v71, pred_g band 0.28 +/- 0.15 per construction.
PRICE (registered maximum): 3 batches x (2 + 1 + 16 + 4 + 2 + 16) = 123 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations

import hashlib
import json

import aspectual_dod_lib as L
import dod_battery
import dod_lexicon

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")

CANDIDATE_ID = "demonstrative_number.these_vs_this.dod_battery_v157"
OUT = "noun_number_demonstrative_dod_battery_v157_result.json"
HEADS = ((11, 2), (7, 8), (6, 3), (15, 1))
FROZEN, BAND = 0.28, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.28 +/- 0.15 per construction"}
ADJ_CANDIDATES = ("shiny", "dusty", "heavy", "tiny", "faded", "sturdy", "glossy", "rusty", "narrow", "smooth", "rough", "sharp", "blunt", "pale", "dark", "bright", "cheap", "fancy",
                  "plain", "fragile", "sticky", "greasy", "damp", "crooked", "spare", "loose", "tight", "thick", "thin", "flat", "round", "square", "odd", "rare", "fresh", "stale", "ripe",
                  "sour", "sweet", "bitter", "warm", "cool", "wet", "dry", "soft", "hard", "loud", "quiet", "cheap", "costly")
CONSTRUCTIONS = {"wanted": lambda agent, obj, adj, det: f"Yesterday the {agent} wanted {det} {adj}",
                 "kept": lambda agent, obj, adj, det: f"The {agent} by the {obj} kept {det} {adj}",
                 "counted": lambda agent, obj, adj, det: f"At dawn the {agent} counted {det} {adj}"}


def build():
    own = ("noun_number_v157_agents", "noun_number_v157_objects", "noun_number_v157_adjectives")   # registered after the run; excluded so replay is stable
    agents = dod_lexicon.fresh(dod_lexicon.AGENT_POOL, 16, exclude=own)
    objects = dod_lexicon.fresh(dod_lexicon.OBJECT_POOL, 16, exclude=own)
    adjectives = dod_lexicon.fresh(ADJ_CANDIDATES, 16, exclude=own)
    pos, neg = L._single(" ones"), L._single(" one")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent, objects[g], adjectives[g], "these" if present else "this")
                ids = L.ENCODING.encode(text)
                answer, foil = (" ones", " one") if present else (" one", " ones")
                a_id, f_id = (pos, neg) if present else (neg, pos)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["noun_number_v157", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, pos, neg, agents, objects, adjectives


def main() -> None:
    rows, pos, neg, agents, objects, adjectives = build()
    print("agents", agents, "objects", objects, "adjectives", adjectives)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, pos, neg, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
