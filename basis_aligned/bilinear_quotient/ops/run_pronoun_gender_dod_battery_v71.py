#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_native_capability pred_c_set_live_and_beats_null pred_d_set_selective pred_e_set_is_additive pred_f_keep_only_retains_most pred_g_set_fraction_within_band_of_frozen
"""Pronoun-gender he/she DoD battery (v71): the atlas top-4 readout set {10.1, 9.6, 12.4, 15.1} on FRESH rows.

Lane: Claude circuit lane. Parent: readout atlas v68 (`atlas_pronoun_gender_v68_result.json`, authored rows of the
pronoun_gender candidate: top-4 set fraction 0.89, live, selective on all three readers, capability 1.0 in every cell).
The triple {9.6, 12.4, 15.1} is the atlas's most recurrent top-4 core (16 live lines, all pronoun gender/number/person
cells), so this is the first fresh-row battery of the PRONOUN readout family. Frozen before access from the atlas:
F* = 0.89 +/- 0.15 (pred_g).

ROWS (fresh, built here; freshness asserted against `dod_lexicon.used_words()` and the module's GENDER pairs):
10 gender noun pairs x 10 objects (only 10 fresh single-token pairs exist outside the module's 16 and the lane's used
words), three constructions the module does not use, {male -> " he", female -> " she"}:
    lost      "The {noun} lost the {obj} and so"            -> " he" / " she"
    because   "Because the {noun} wanted the {obj},"         -> " he" / " she"
    later     "The {noun} bought the {obj} and later"          -> " he" / " she"
60 rows in 2 batches. Removal along `O_h^T (u_he - u_she)` per head at the final token; null = norm-matched random
direction x 16 seeds; readers was/were, who/which, night/day (gates as every line); singles + additivity; zero /
keep-only / 16 random keeps. Evidence tag: edit, fresh rows.

PREDICTIONS (scored as written by `dod_battery.run`; failures preserved)
    pred_a_instrument_replays_native          manual forward = producer native <= 1e-4
    pred_b_native_capability                  >= 0.85 correct in every (construction x side) cell. Prior: unsure --
                                              the "because" frame puts the pronoun after a comma.
    pred_c_set_live_and_beats_null            fraction >= 0.10, positive >= 0.75, above every null
    pred_d_set_selective                      all three reader gates
    pred_e_set_is_additive                    |joint - sum(singles)| <= 0.25 x min(single)
    pred_f_keep_only_retains_most             keep-only retention >= 0.70 and every random keep <= 0.30
    pred_g_set_fraction_within_band_of_frozen every construction's fraction within 0.89 +/- 0.15

PRICE (registered maximum): 2 batches x (2 + 1 + 16 + 4 + 2 + 16) = 82 forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations

import hashlib
import json

import aspectual_dod_lib as L
import circuit_fast_screen_candidate_pronoun_gender as pg
import dod_battery
import dod_lexicon

CANDIDATE_ID = "pronoun_gender.he_vs_she.dod_battery_v71"
OUT = "pronoun_gender_dod_battery_v71_result.json"
HEADS = ((10, 1), (9, 6), (12, 4), (15, 1))
FROZEN, BAND = 0.89, 0.15
# scored by dod_battery.run, listed here so the queue gate sees the literal keys
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_native_capability": ">= 0.85 per cell",
               "pred_c_set_live_and_beats_null": "fraction >= 0.10, positive >= 0.75, > null max", "pred_d_set_selective": "three reader gates",
               "pred_e_set_is_additive": "gap <= 0.25 x min single", "pred_f_keep_only_retains_most": "retention >= 0.70, random <= 0.30",
               "pred_g_set_fraction_within_band_of_frozen": "0.89 +/- 0.15 per construction"}
PAIRS = (("hero", "heroine"), ("god", "goddess"), ("grandson", "granddaughter"), ("boyfriend", "girlfriend"), ("spokesman", "spokeswoman"),
         ("dad", "mom"), ("bull", "cow"), ("groom", "bride"), ("male", "female"), ("guy", "gal"))
OBJECTS = ("compass", "ladder", "hammer", "mirror", "candle", "saddle", "helmet", "anchor", "bucket", "shovel", "pillow",
           "blanket", "wagon", "barrel", "cradle", "scroll")[:10]
CONSTRUCTIONS = {"lost": lambda noun, obj: f"The {noun} lost the {obj} and so",
                 "because": lambda noun, obj: f"Because the {noun} wanted the {obj},",
                 "later": lambda noun, obj: f"The {noun} bought the {obj} and later"}


def build():
    # own panel registered in dod_lexicon.PANELS after the run; exclude it so replay keeps working
    used = dod_lexicon.used_words(exclude=("pronoun_gender_v71_nouns", "pronoun_gender_v71_objects")) | {w for p in pg.GENDER for w in p}
    words = [w for p in PAIRS for w in p] + list(OBJECTS)
    if set(words) & used:
        raise L.RowError(f"not fresh: {sorted(set(words) & used)}")
    if len(set(words)) != len(words):
        raise L.RowError("duplicate word")
    for w in words:
        L._single(" " + w)
    he, she = L._single(" he"), L._single(" she")
    reader_ids = {name: (L._single(a), L._single(b)) for name, (a, b) in L.READERS.items()}
    rows = []
    for construction, make in CONSTRUCTIONS.items():
        for g, (male, female) in enumerate(PAIRS):
            for present in (True, False):
                text = make(male if present else female, OBJECTS[g])
                ids = L.ENCODING.encode(text)
                answer, foil = (" he", " she") if present else (" she", " he")
                a_id, f_id = (he, she) if present else (she, he)
                if L.ENCODING.encode(text + answer) != ids + [a_id] or L.ENCODING.encode(text + foil) != ids + [f_id]:
                    raise L.RowError(f"joint tokenization changed for {text!r}")
                rows.append(L.Row(hashlib.sha256(json.dumps(["pronoun_gender_v71", construction, g, present, text]).encode()).hexdigest()[:24],
                                  construction, g, present, text, tuple(ids), answer, foil, a_id, f_id, len(ids) - 1, (), reader_ids))
    return rows, he, she


def main() -> None:
    rows, he, she = build()
    dod_battery.run(dod_battery.LineSpec(CANDIDATE_ID, OUT, rows, he, she, HEADS, frozen_fraction=FROZEN, band=BAND))


if __name__ == "__main__":
    main()
