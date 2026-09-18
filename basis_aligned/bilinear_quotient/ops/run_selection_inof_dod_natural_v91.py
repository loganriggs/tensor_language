#!/usr/bin/env python3
# BQGATE: EXPERIMENT pred_a_instrument_replays_native pred_b_congruent_native_capability pred_c_congruent_removal_shifts_away_from_label pred_d_congruent_removal_beats_null pred_e_congruent_removal_selective pred_f_incongruent_removal_shifts_toward_label
"""Selection family DoD (v91): the run_selection_dod_battery_v85 set on natural FineWeb rows (training corpus, out-of-panel) (PREDICTS OOD).
Body: `dod_natural_line.run`. Rows: `selection_dod_natural_rows.py` (generic miner `dod_natural_miner.py`, receipt
`circuits/followups/selection_inof_dod_natural_rows_v91.json`), cue token within 12 tokens of a next token ' in' / ' of'; congruent cells ("natural_interested_in", "natural_afraid_of")
are the readout test; the incongruent cells are whatever the corpus offers (a far cue with another preposition/particle; may be
short or empty -- recorded, not padded). Any-sense cue filter stated. The final query is the token before the function word, so on
adjacent natural uses the readout is asked one position earlier than the panel's parenthetical.
PREDICTIONS (bars frozen from the aspectual natural line v20; failures preserved): pred_a instrument <= 1e-4; pred_b congruent
capability >= 0.75 per cell; pred_c congruent mean damage >= 0.15 and >= 60% positive; pred_d > null max; pred_e three gates;
pred_f incongruent mean damage <= 0 (None if the cells are empty). Prior for b: unsure on the far-cue rows.
PRICE (registered maximum): batches x (native + producer + set + 16 nulls); 0 backwards; 0 fits.
"""
from __future__ import annotations
import dod_natural_line as N
import run_selection_dod_battery_v85 as line

CANDIDATE_ID = line.CANDIDATE_ID.rsplit(".", 1)[0] + ".dod_natural_v91"
ROWS = N.ROOT / "circuits/followups/selection_inof_dod_natural_rows_v91.json"
OUT = "selection_inof_dod_natural_v91_result.json"
CONGRUENT = ("natural_interested_in", "natural_afraid_of")
PREDICTIONS = {"pred_a_instrument_replays_native": "<= 1e-4", "pred_b_congruent_native_capability": ">= 0.75 per congruent cell",
               "pred_c_congruent_removal_shifts_away_from_label": "mean >= 0.15, >= 60% positive", "pred_d_congruent_removal_beats_null": "> null max",
               "pred_e_congruent_removal_selective": "three gates", "pred_f_incongruent_removal_shifts_toward_label": "mean <= 0"}


def main() -> None:
    rows, sha, p, n = N.rows_from_receipt(ROWS, "in", " in", " of", lambda r: f"natural_{r['cue']}_{r['label']}")
    N.run(CANDIDATE_ID, OUT, rows, sha, p, n, line.HEADS, CONGRUENT)


if __name__ == "__main__":
    main()
