#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Correlative either/not (or vs but) DoD battery (v119): the atlas top-4 set ((16, 8), (14, 8), (7, 8), (8, 1)) on FRESH rows -- the CORRELATIVE readout family candidate.

Lane: Claude circuit lane. Parents: readout atlas v68 (five live correlative lines share {8.1, 16.8, 14.8, 7.8} or three of it: either/neither ->
or/nor 0.55, both/either -> and/or 0.47, either/not -> or/but 0.43, both/neither -> and/nor 0.45; 8.1 leads, again on a token cue). Frozen before
access: F* = 0.43 +/- 0.15 (pred_g). Readers AS RUN: will-would, who-which, night-day -- importing v104 sets L.READERS at module level (import side effect, found after the run and recorded on the board); the docstring originally said was-were.

ROWS (fresh; `dod_lexicon.fresh`): 16 fresh agents x 16 fresh objects, three frames the module does not use, {either -> " or", not -> " but"}:
    chose    "Yesterday the {agent} chose {cue} the {obj}"
    accept   "The {agent} would accept {cue} the {obj}"
    bought   "At dawn the {agent} bought {cue} the {obj}"
96 rows in 3 batches. Battery as v71 (`dod_battery.run`). Evidence tag: edit, fresh rows.

PREDICTIONS (scored as written by `dod_battery.run`; failures preserved): as v71, pred_g band 0.43 +/- 0.15 per construction.
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
import run_person_reflexive_dod_battery_v104 as v104

CANDIDATE_ID = "correlative_disjoint_either_not.either_vs_not.dod_battery_v119"
OUT = "correlative_either_not_dod_battery_v119_result.json"
HEADS = ((16, 8), (14, 8), (7, 8), (8, 1))
FROZEN, BAND = 0.43, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.43 +/- 0.15 per construction"}
AGENT_CANDIDATES = dod_lexicon.AGENT_POOL
OBJECT_CANDIDATES = dod_lexicon.OBJECT_POOL
CONSTRUCTIONS = {"chose": lambda agent, obj, cue: f"Yesterday the {agent} chose {cue} the {obj}",
                 "accept": lambda agent, obj, cue: f"The {agent} would accept {cue} the {obj}",
                 "bought": lambda agent, obj, cue: f"At dawn the {agent} bought {cue} the {obj}"}


def build():
    own = ("correlative_v119_v120_agents", "correlative_v119_v120_objects")   # shared fresh panel, registered after the runs
    agents = dod_lexicon.fresh(AGENT_CANDIDATES, 16, exclude=own)
    objects = dod_lexicon.fresh(OBJECT_CANDIDATES, 16, exclude=own)
    pos, neg = L._single(" or"), L._single(" but")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent, objects[g], "either" if present else "not")
                ids = L.ENCODING.encode(text)
                answer, foil = (" or", " but") if present else (" but", " or")
                a_id, f_id = (pos, neg) if present else (neg, pos)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["correlative_either_not_v119", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, pos, neg, agents, objects


def main() -> None:
    rows, pos, neg, agents, objects = build()
    print("agents", agents, "objects", objects)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, pos, neg, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
