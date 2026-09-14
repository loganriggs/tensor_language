# Frozen S/O cue-response reduction on new constructions

Use all 72 rows built without native scores. The original S/R/O source weights
are frozen. Run every mask 0..7 (S=1, R=2, O=4), 576 forwards, 180 second cap.
Predict masks 2/3/6/7 from observed masks 0/1/4/5 by masking out the R bit.
This retains S, O and SO response terms without estimating them from held-out arms.

- A: shared write vs separate S/R/O writes relative error <=1e-10 for every
  nonzero mask; full factorial reconstruction absolute error <=1e-12.
- B: each family has at least 10/12 positive native British-minus-American
  paired contrasts, and full-head-removal paired effect norm is >=10% of native
  paired contrast norm. Failure is retained without selecting replacement rows.
- C: each family's maximum held-out-corner paired-cue L2 prediction error is
  <=10% of its full-head removal paired effect norm (denominator floor 1e-8).
  Report C separately from native capability; adoption requires all three.

Null: the omitted R branch becomes material after the template shift, or the
model lacks the cue behavior. Unrelated-control errors and scalar/additive
comparators are descriptive. No general preservation, autonomous prediction,
new semantic identity or static model compression claim. All model weights,
prefix/suffix and four observed intervention corners remain explicit costs.
