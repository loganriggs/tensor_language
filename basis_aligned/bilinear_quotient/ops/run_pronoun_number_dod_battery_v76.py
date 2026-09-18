#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Pronoun number they/he DoD battery (v76): the atlas top-4 set {9.6, 12.4, 15.1, 10.5} on FRESH rows -- the pronoun FAMILY claim.

Lane: Claude circuit lane. Parents: readout atlas v68 (`atlas_pronoun_number_v68_result.json`: 9.6 0.81, 12.4 0.28, 15.1 0.15,
10.5 0.14; set fraction 0.71, live, selective) and the pronoun-gender line v71-v75 (set {10.1, 9.6, 12.4, 15.1}, all five
properties at head grain). If the same core {9.6, 12.4, 15.1} carries a second pronoun decision on fresh rows with the same
battery, the pronoun family is a component family and not one line's co-occurrence. Frozen before access: F* = 0.71 +/- 0.15.
Note the contrast is they vs he (the module's vocabulary), so number is confounded with gender on the negative side; the
gender line's set shares three heads, which is the point under test, not a nuisance.

ROWS (fresh; freshness asserted against `dod_lexicon.used_words()`): 16 fresh agent nouns with single-token plurals x 16
fresh objects, three frames the module does not use, {plural -> " they", singular -> " he"}:
    lost      "The {noun}(s) lost the {obj} and so"
    because   "Because the {noun}(s) wanted the {obj},"
    later     "The {noun}(s) bought the {obj} and later"
96 rows in 3 batches. Battery as v71 (`dod_battery.run`). Evidence tag: edit, fresh rows.

PREDICTIONS (scored as written by `dod_battery.run`; failures preserved): as v71, pred_g band 0.71 +/- 0.15 per construction.
PRICE (registered maximum): 3 batches x (2 + 1 + 16 + 4 + 2 + 16) = 123 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations

import hashlib
import json

import aspectual_dod_lib as L
import dod_battery
import dod_lexicon

CANDIDATE_ID = "pronoun_number.they_vs_he.dod_battery_v76"
OUT = "pronoun_number_dod_battery_v76_result.json"
HEADS = ((9, 6), (12, 4), (15, 1), (10, 5))
FROZEN, BAND = 0.71, 0.15
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.71 +/- 0.15 per construction"}
AGENT_CANDIDATES = ('trader', 'critic', 'senator', 'diver', 'wrestler', 'cyclist', 'physicist', 'biologist', 'economist', 'philosopher', 'programmer', 'developer', 'consultant', 'investor', 'marine', 'commander', 'major', 'recruit', 'veteran', 'cleric', 'elder', 'apostle', 'comedian', 'musician', 'patient', 'customer', 'client', 'tourist', 'passenger', 'commuter', 'pedestrian', 'citizen', 'resident', 'voter', 'taxpayer', 'immigrant', 'pioneer', 'robot', 'alien', 'zombie')
OBJECT_CANDIDATES = ("coin", "rope", "torch", "bell", "jar", "drum", "crown", "sword", "shield", "purse", "wallet", "ticket", "spoon", "knife", "bowl",
                     "pillow", "blanket", "wagon", "barrel", "cradle", "scroll", "brush", "needle", "lamp", "box", "bag", "cart", "flag", "map", "key")
CONSTRUCTIONS = {"lost": lambda noun, obj: f"The {noun} lost the {obj} and so",
                 "because": lambda noun, obj: f"Because the {noun} wanted the {obj},",
                 "later": lambda noun, obj: f"The {noun} bought the {obj} and later"}


def build():
    own = ("pronoun_number_v76_agents", "pronoun_number_v76_objects")   # registered after the run; excluded so replay is stable
    agents = dod_lexicon.fresh(AGENT_CANDIDATES, 16, plural=True, exclude=own)
    objects = dod_lexicon.fresh(OBJECT_CANDIDATES, 16, exclude=own)
    they, he = L._single(" they"), L._single(" he")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, agent in enumerate(agents):
            for present in (True, False):
                text = make(agent + "s" if present else agent, objects[g])
                ids = L.ENCODING.encode(text)
                answer, foil = (" they", " he") if present else (" he", " they")
                a_id, f_id = (they, he) if present else (he, they)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["pronoun_number_v76", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, they, he, agents, objects


def main() -> None:
    rows, they, he, agents, objects = build()
    print("agents", agents, "objects", objects)
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, they, he, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
