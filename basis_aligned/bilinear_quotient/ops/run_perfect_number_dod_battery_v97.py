#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Perfect have/has DoD battery (v97): the census set {11.3, 7.8, 5.3, 9.7} on FRESH rows -- the number FAMILY's second line.

Lane: Claude circuit lane. Parents: v50b (perfect_number.have_vs_has census on authored rows: 11.3 1.54, 7.8 0.20, 5.3 0.12, 9.7 0.12;
set fraction 0.66, live; selectivity failed there only because the was−were reader is number-marked) and v55 (lexical were/was
battery, the family's first line at fresh grain). The pronoun and selection families each have two batteried lines; the number family
has one. Readers here avoid number: tense will−would, animacy who−which, canonical night−day. Frozen before access: F* = 0.66 +/- 0.15.

ROWS (fresh; `dod_lexicon.fresh`): 16 fresh head nouns with single-token plurals x 16 fresh objects, three frames the module does not use,
each with a PP intervener between the head and the auxiliary, {plural -> " have", singular -> " has"}:
    since       "Since dawn the {noun}(s) by the {obj}"
    lately      "Lately the {noun}(s) inside the {obj}"
    apparently  "Apparently the {noun}(s) behind the {obj}"
96 rows in 3 batches. Battery as v71/v76/v85 (`dod_battery.run`). Evidence tag: edit, fresh rows.

PREDICTIONS (scored as written by `dod_battery.run`; failures preserved): as v71, pred_g band 0.66 +/- 0.15 per construction.
PRICE (registered maximum): 3 batches x (2 + 1 + 16 + 4 + 2 + 16) = 123 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations

import hashlib
import json

import aspectual_dod_lib as L
import dod_battery
import dod_lexicon
import run_pronoun_number_dod_battery_v76 as v76
import run_selection_particle_dod_battery_v86 as v86

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")

CANDIDATE_ID = "perfect_number.have_vs_has.dod_battery_v97"
OUT = "perfect_number_dod_battery_v97_result.json"
HEADS = ((11, 3), (7, 8), (5, 3), (9, 7))
FROZEN, BAND = 0.66, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates (will-would, who-which, night-day)",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.66 +/- 0.15 per construction"}
AGENT_CANDIDATES = v86.AGENT_CANDIDATES + v76.AGENT_CANDIDATES
OBJECT_CANDIDATES = ("plate", "cup", "mug", "pot", "pan", "tray", "jug", "bottle", "can", "brick", "stone", "log", "stick", "wheel", "chain", "hook", "nail", "screw", "bolt",
                     "ladder", "mirror", "candle", "saddle", "helmet", "anchor", "bucket", "shovel", "compass", "hammer", "rug", "sofa", "desk", "shelf", "stool", "crate")
CONSTRUCTIONS = {"since": lambda noun, obj: f"Since dawn the {noun} by the {obj}",
                 "lately": lambda noun, obj: f"Lately the {noun} inside the {obj}",
                 "apparently": lambda noun, obj: f"Apparently the {noun} behind the {obj}"}


def build():
    agents = dod_lexicon.fresh(AGENT_CANDIDATES, 16, plural=True)
    objects = dod_lexicon.fresh(OBJECT_CANDIDATES, 16)
    have, has = L._single(" have"), L._single(" has")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent + "s" if present else agent, objects[g])
                ids = L.ENCODING.encode(text)
                answer, foil = (" have", " has") if present else (" has", " have")
                a_id, f_id = (have, has) if present else (has, have)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["perfect_number_v97", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, have, has, agents, objects


def main() -> None:
    rows, have, has, agents, objects = build()
    print("agents", agents, "objects", objects)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, have, has, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
