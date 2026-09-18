#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Adjective preposition selection in/of DoD battery (v85): the atlas top-4 set {8.8, 6.3, 13.8, 7.8} on FRESH rows -- a FOURTH readout family?

Lane: Claude circuit lane. Parents: readout atlas v68 (`atlas_adjective_preposition_in_of_v68_result.json`: 8.8 0.66, 6.3 0.64,
13.8 0.53, 7.8 0.30; set fraction 0.54, live, selective, capability 1.0) and the earlier preposition census v54 (on/of: the SAME
four heads {6.3, 13.8, 7.8, 8.8}). The atlas family assignment left 52 live lines outside the three families; 9 of them share
13.8 + 7.8 and are all complement / particle / preposition SELECTION decisions (a lexical head selects the next function word).
Frozen before access from the atlas: F* = 0.54 +/- 0.15 (pred_g).

ROWS (fresh; freshness asserted by `dod_lexicon.fresh`): 16 fresh agent nouns x 16 fresh objects, three frames the module does not
use, keeping its parenthetical shape (the selection must cross ", <adverb>,"), {interested -> " in", afraid -> " of"}:
    remained  "The {agent} near the {obj} remained {adj}, as always,"
    looked    "By the {obj} the {agent} looked {adj}, frankly,"
    grew      "That {agent} grew {adj}, apparently,"
96 rows in 3 batches. Battery as v71/v76 (`dod_battery.run`). Readers was/were, who/which, night/day. Evidence tag: edit, fresh rows.

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

CANDIDATE_ID = "adjective_preposition_in_of.interested_vs_afraid.dod_battery_v85"
OUT = "selection_dod_battery_v85_result.json"
HEADS = ((8, 8), (6, 3), (13, 8), (7, 8))
FROZEN, BAND = 0.54, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.54 +/- 0.15 per construction"}
AGENT_CANDIDATES = tuple(w for w in v76.AGENT_CANDIDATES if w not in dod_lexicon.PANELS["pronoun_number_v76_agents"]) + ("recruit", "veteran", "cleric", "elder", "apostle", "comedian", "musician", "patient", "customer", "client", "tourist", "passenger", "commuter", "pedestrian", "citizen", "resident", "voter", "taxpayer", "immigrant", "pioneer", "robot", "alien", "zombie")
OBJECT_CANDIDATES = ("blanket", "wagon", "barrel", "cradle", "scroll", "brush", "needle", "lamp", "box", "bag", "cart", "flag", "map", "key", "tent", "boat", "cane", "cloak", "gate", "bench", "stove", "fence", "cage", "chest", "ladder", "vase", "clock", "trunk")
CONSTRUCTIONS = {"remained": lambda agent, obj, adj: f"The {agent} near the {obj} remained {adj}, as always,",
                 "looked": lambda agent, obj, adj: f"By the {obj} the {agent} looked {adj}, frankly,",
                 "grew": lambda agent, obj, adj: f"That {agent} grew {adj}, apparently,"}


def build():
    agents = dod_lexicon.fresh(AGENT_CANDIDATES, 16)
    objects = dod_lexicon.fresh(OBJECT_CANDIDATES, 16)
    pos, neg = L._single(" in"), L._single(" of")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent, objects[g], "interested" if present else "afraid")
                ids = L.ENCODING.encode(text)
                answer, foil = (" in", " of") if present else (" of", " in")
                a_id, f_id = (pos, neg) if present else (neg, pos)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["selection_v85", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, pos, neg, agents, objects


def main() -> None:
    rows, pos, neg, agents, objects = build()
    print("agents", agents, "objects", objects)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, pos, neg, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
