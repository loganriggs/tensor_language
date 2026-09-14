# Native equal-count early versus late framing split

Reuse the 48 fresh rows and partition each framing mask by source order into
equal-count earlier and later halves: 8/8 sources for local-history and 6/6 for
radio. Arms are native, remove early-half O, remove late-half O, and remove all
framing O. Read the target plus the four frozen new controls. This is 192 body
forwards under a 180-second cap.

- `pred_a`: early plus late reconstructs framing sources within `1e-10`;
  native and framing target/work-jobs arms replay the frozen fresh artifact
  within `1e-5`; values are finite and exactly 192 forwards execute.
- `pred_b`: native target cue is positive for at least 10/12 pairs and framing
  target-effect RMS is at least `1e-5` in each template.
- `pred_c` (stable early-position hypothesis): early-half cue error versus all
  framing is at most 35% and late-half error is at least 50% in each template.
- `pred_d` (stable late-position hypothesis): late-half error is at most 35%
  and early-half error is at least 50% in each template.

Four-control ratios and separate-effect composition are descriptive. Failure of
both opposing hypotheses rejects a template-stable early/late role on this
panel. The split is fixed without model scores or fitting. Native upstream
states and suffix remain; no semantic identification, corpus OOD, static
compression, or quantization claim follows.

