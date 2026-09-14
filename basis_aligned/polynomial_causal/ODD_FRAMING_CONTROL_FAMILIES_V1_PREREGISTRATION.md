# Multi-control robustness of the fresh O framing effect

Reuse the frozen 48 rows from `ODD_FRAMING_FRESH_V1_ROWS.json` and only the
native and remove-O-from-framing arms.  The original target and work/jobs
readouts are replay anchors.  Before further model scores, freeze four new,
outcome-blind token pairs: cat/dog `[3797,3290]`, red/blue `[2266,4171]`,
Monday/Tuesday `[3321,3431]`, and apple/orange `[17180,10912]`.

This is 96 body forwards under a 180-second cap.

- `pred_a`: target and work/jobs values for both arms replay the frozen fresh
  artifact within `1e-5`; source recomposition is at most `1e-10`; all values
  are finite and exactly 96 forwards execute.
- `pred_b`: native target cue is positive for at least 10/12 pairs and the
  framing target-effect RMS is at least `1e-5` in each template.
- `pred_c`: in each template the median of the four new control-effect RMS to
  target-effect RMS ratios is at most `.5`, and at least three of four ratios
  are at most `.5`.
- `pred_d`: every new control ratio is at most `1` in each template.  This is a
  descriptive robustness bar stronger than C.

A pass does not rescue the failed work/jobs control in the earlier receipt.
The test diagnoses control-family sensitivity; it does not establish a unique
semantic unit.  Native upstream states, full weights, and the suffix remain.

