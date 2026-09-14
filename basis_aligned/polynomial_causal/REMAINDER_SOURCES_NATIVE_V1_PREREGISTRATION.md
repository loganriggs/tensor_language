# Current versus inherited full-even remainder

Freeze all 72 SRO_FRESH_CUE_V1 rows, including the weak opposed-role family.
Rc = even((1−lambda)Vcurrent)−S; Rf = even(lambda Vfirst), hence R=Rc+Rf.
Use existing weight-derived routing and output maps, supplied native first values,
and the unchanged native suffix. Arms: native, remove Rc, remove Rf, remove R.
Price: 288 forwards, 180 seconds; two shared routing evaluations per nonzero arm,
no added model weights, all original weights and contexts remain required.

- A: native/R anchors relative error <=1e-5 against the prior full cube;
  live source recomposition relative error <=1e-10.
- B: Rc removal paired cue effects approximate R removal within 20% relative
  L2 error in each family, denominator floor 1e-8. Null: inherited values make
  a material contribution to the newly consequential remainder cue effect.
- C: separate Rc and Rf removal effects sum to R removal within 5% relative L2
  error for paired cue and individual control readouts in each family.

All arms are baseline-subtracted. Record B/C independently of A; adoption would
require A. No new selection, independent OOD, selective circuit or static
compression claim follows from this source localization.
