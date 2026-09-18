#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Verb particle selection up/down DoD battery (v86): the atlas top-4 set {13.8, 14.8, 7.8, 8.8} on FRESH rows -- the selection FAMILY test.

Lane: Claude circuit lane. Parents: v85 (adjective preposition in/of: {8.8, 6.3, 13.8, 7.8}, 7/7 on fresh rows) and the atlas
(`atlas_verb_particle_up_down_v68_result.json`: 13.8 0.69, 14.8 0.55, 7.8 0.27, 8.8 0.27; set fraction 0.52, live, selective,
capability 1.0). Three heads shared with v85's set: if this second selection decision passes the same battery on fresh rows, the
selection family is a component family. Frozen before access: F* = 0.52 +/- 0.15 (pred_g).

ROWS (fresh; `dod_lexicon.fresh`): 16 fresh agents x 16 fresh objects, the v85 frames with the verb pair, {woke -> " up", calmed -> " down"}:
    remained  "The {agent} near the {obj} {verb}, as always,"
    looked    "By the {obj} the {agent} {verb}, frankly,"
    grew      "That {agent} {verb}, apparently,"
96 rows in 3 batches. Battery as v85 (`dod_battery.run`). Evidence tag: edit, fresh rows.

PREDICTIONS (scored as written by `dod_battery.run`; failures preserved): as v85, pred_g band 0.52 +/- 0.15 per construction.
PRICE (registered maximum): 3 batches x (2 + 1 + 16 + 4 + 2 + 16) = 123 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations

import hashlib
import json

import aspectual_dod_lib as L
import dod_battery
import dod_lexicon

CANDIDATE_ID = "verb_particle_up_down.woke_vs_calmed.dod_battery_v86"
OUT = "selection_particle_dod_battery_v86_result.json"
HEADS = ((13, 8), (14, 8), (7, 8), (8, 8))
FROZEN, BAND = 0.52, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.52 +/- 0.15 per construction"}
AGENT_CANDIDATES = ("resident", "voter", "taxpayer", "immigrant", "pioneer", "robot", "alien", "zombie", "vampire", "ghost", "giant", "dwarf", "goblin", "dragon", "villain",
                    "toddler", "infant", "teenager", "adult", "sibling", "twin", "attorney", "detective", "inspector", "warden", "marshal", "trooper", "professor", "lecturer",
                    "dean", "mentor", "pupil", "intern", "trainee", "apprentice", "graduate", "freshman", "bartender", "barber", "florist", "grocer", "jeweler", "cashier",
                    "janitor", "courier", "porter", "usher", "listener", "viewer", "speaker", "caller", "sender", "player", "gamer", "fan", "chemist", "geologist", "historian",
                    "linguist", "designer", "analyst", "manager", "director", "producer", "merchant", "peddler", "tailor", "weaver", "potter", "carpenter", "mason", "miller",
                    "fisher", "trapper", "logger", "rancher", "herder", "officer", "captain", "admiral", "lieutenant", "cadet", "bishop", "rabbi", "imam", "preacher", "deacon",
                    "magician", "juggler", "acrobat", "drummer", "guitarist", "pianist", "violinist", "composer", "conductor", "neighbor", "settler", "nomad")
OBJECT_CANDIDATES = ("cane", "cloak", "gate", "bench", "stove", "fence", "cage", "chest", "vase", "clock", "trunk", "hat", "coat", "boot", "glove", "belt", "scarf", "shirt",
                     "sock", "shoe", "plate", "cup", "mug", "pot", "pan", "tray", "jug", "kettle", "bottle", "can", "brick", "stone", "log", "stick", "wheel", "chain", "hook",
                     "nail", "screw", "bolt")
CONSTRUCTIONS = {"remained": lambda agent, obj, verb: f"The {agent} near the {obj} {verb}, as always,",
                 "looked": lambda agent, obj, verb: f"By the {obj} the {agent} {verb}, frankly,",
                 "grew": lambda agent, obj, verb: f"That {agent} {verb}, apparently,"}


def build():
    agents = dod_lexicon.fresh(AGENT_CANDIDATES, 16)
    objects = dod_lexicon.fresh(OBJECT_CANDIDATES, 16)
    pos, neg = L._single(" up"), L._single(" down")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent, objects[g], "woke" if present else "calmed")
                ids = L.ENCODING.encode(text)
                answer, foil = (" up", " down") if present else (" down", " up")
                a_id, f_id = (pos, neg) if present else (neg, pos)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["selection_particle_v86", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, pos, neg, agents, objects


def main() -> None:
    rows, pos, neg, agents, objects = build()
    print("agents", agents, "objects", objects)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, pos, neg, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
