#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""correlative_either_not DoD (v126): the run_correlative_either_not_dod_battery_v119 set on Pile rows (out-of-corpus) (PREDICTS OOD). Body: `dod_natural_line.run`; rows receipt `circuits/followups/correlative_either_not_dod_pile_rows_v126.json` (outcome-blind
miner; any-sense cue filter stated). Congruent cells ('natural_either_or', 'natural_not_but') are the readout test; the other cells are corpus counter-cases. Emitted by dod_line.py.
PREDICTIONS (bars frozen from v20; failures preserved): pred_a instrument <= 1e-4; pred_b congruent capability >= 0.75 per cell; pred_c congruent mean
damage >= 0.15 and >= 60% positive; pred_d > null max; pred_e three gates; pred_f incongruent mean damage <= 0 (None if the cells are empty).
PRICE (registered maximum): batches x (native + producer + set + 16 nulls) forwards; 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import run_correlative_either_not_dod_battery_v119 as line

CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_pile_v126"
ROWS = N.ROOT / "circuits/followups/correlative_either_not_dod_pile_rows_v126.json"
OUT = "correlative_either_not_dod_pile_v126_result.json"
CONGRUENT = ('natural_either_or', 'natural_not_but')
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, pos, neg = N.rows_from_receipt(ROWS, "or", " or", " but", lambda r: f"natural_{r['cue']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, pos, neg, line.HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
