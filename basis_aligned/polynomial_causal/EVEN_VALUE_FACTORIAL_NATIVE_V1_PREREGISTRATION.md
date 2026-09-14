# Complete the scalar/remainder/odd intervention cube

S is the prior cue-selective scalar, R is full-even minus S, O is the odd-key
component. All weights and shared graph are frozen. Mask bits 1/2/4 denote
removed S/R/O. Prior native tests cover masks 0,1,2,3,4,7. Masks 5 and 6 remain
unmeasured, so two pair interactions and the triple cannot be separately inferred.

Use the same 72 regional and 32 natural prefixes: native plus missing masks 5/6
requires 312 body forwards, 120 seconds, managed GPU lane1. The shared graph must
consume actual native first-value inputs, combine branches before the output map,
and retain the actual nonlinear suffix. No model weights or row choices are fit.

A: repeated native scores <=1e-5 relative to previous anchors, and both live
shared graph writes agree with separate S/R/O writes <=1e-10. B: the complete
eight-corner Boolean/Mobius decomposition replays scores <=1e-10 relative.
C: dropping the triple term predicts the full-removal baseline-subtracted effect
within 1% relative L2 on both regional readouts in each of three families and
newline CE in each natural half. Report every coefficient and failed criterion.

This tests how many interaction orders are needed for these fixed interventions.
It is not data-guided fitting of model weights, a held-out text predictor, universal
linearity, an autonomous model, or identification of new semantic circuits.
