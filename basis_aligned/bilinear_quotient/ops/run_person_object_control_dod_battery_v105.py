#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Reflexive object control me/you (myself vs yourself) DoD battery (v105): the atlas top-4 set ((13, 1), (8, 1), (10, 5), (15, 1)) on FRESH rows -- a PERSON readout family candidate.

Lane: Claude circuit lane. Parents: readout atlas v68 (five live lines -- reflexive_person, reflexive_object_control, their plural forms,
possessive my/your -- share {8.1, 13.1, 10.5, 15.1} or three of it; 8.1 leads, the temporal family's token-only cue reader, here reading
a person-marked pronoun token). Frozen before access: F* = 0.54 +/- 0.15 (pred_g). Readers avoid person/number marking: will-would,
who-which, night-day (was-were is person-marked for I / you).

ROWS (fresh; `dod_lexicon.fresh`): 16 fresh agents x 16 fresh objects, three frames the module does not use, {me -> " myself", you -> " yourself"}.
96 rows in 3 batches. Battery as v71/v76/v85/v97 (`dod_battery.run`). Evidence tag: edit, fresh rows.

PREDICTIONS (scored as written by `dod_battery.run`; failures preserved): as v71, pred_g band 0.54 +/- 0.15 per construction.
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
import run_perfect_number_dod_battery_v97 as v97

L.READERS = {"tense_will_would": (" will", " would"), "animacy_who_which": (" who", " which"), "canonical_night_day": (" night", " day")}
L.UNRELATED = ("tense_will_would", "animacy_who_which", "canonical_night_day")

CANDIDATE_ID = "reflexive_object_control.me_vs_you.dod_battery_v105"
OUT = "person_object_control_dod_battery_v105_result.json"
HEADS = ((13, 1), (8, 1), (10, 5), (15, 1))
FROZEN, BAND = 0.54, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.54 +/- 0.15 per construction"}
AGENT_CANDIDATES = v86.AGENT_CANDIDATES + v76.AGENT_CANDIDATES + ("nurse", "pilot", "farmer", "soldier", "teacher", "doctor", "driver", "student", "worker", "lawyer", "judge", "sailor", "miner", "baker", "guard", "clerk")
OBJECT_CANDIDATES = v97.OBJECT_CANDIDATES + v86.OBJECT_CANDIDATES + ("nail", "screw", "bolt", "rug", "sofa", "desk", "shelf", "stool", "crate", "basket", "bench", "lantern", "candle", "mirror")
CONSTRUCTIONS = {"warned": lambda agent, obj, cue: f"The {agent} by the {obj} warned {cue} to protect",
                 "reminded": lambda agent, obj, cue: f"Yesterday the {agent} reminded {cue} to trust",
                 "urged": lambda agent, obj, cue: f"Once again the {agent} near the {obj} urged {cue} to forgive"}


def build():
    agents = dod_lexicon.fresh(AGENT_CANDIDATES, 16)
    objects = dod_lexicon.fresh(OBJECT_CANDIDATES, 16)
    pos, neg = L._single(" myself"), L._single(" yourself")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent, objects[g], "me" if present else "you")
                ids = L.ENCODING.encode(text)
                answer, foil = (" myself", " yourself") if present else (" yourself", " myself")
                a_id, f_id = (pos, neg) if present else (neg, pos)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["person_object_control_v105", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, pos, neg, agents, objects


def main() -> None:
    rows, pos, neg, agents, objects = build()
    print("agents", agents, "objects", objects)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, pos, neg, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
