# Does the source-group reversal persist at small intervention strength?

Opened follow-up after the preregistered fresh direction failure and self/mixed
causal diagnostic. Source-group prediction remained accurate, while native present,
self and mixed removals reversed in documents4,8,11,19. Do not use these as fresh data.

Use the exact captured MLP7-present reference write from CITY_SOURCE7_FRESH_V1,
with strengths0,.1,.5,1 at attention8; full native MLP8/suffix recomputed. Four padded
batches/160sequence-equivalent forwards/72block calls,2CPUthreads,180second cap.
No changed source definition, row selection, fit, query or denominator recomputation
inside the reference write. This is a finite-strength test, not an exact derivative.

pred_a: native and strength1 match prior native/reference-group margins at
<=1e-4absolute AND<=1e-5relative; all finite/support0 and callcounts correct; CPUonly.
pred_b: both.1and.5strengths have negative attenuation at all six endpoints in each
of the four previously reversing documents4,8,11,19. Failure rejects persistence of
those reversals toward small amplitude.
pred_c: relative L2 error of10*effect(.1) and2*effect(.5) against effect(1) is<=.10
for each globally and within the four-document reversal stratum. Failure means
simple amplitude linearity cannot explain the full-strength effect at this bar.

Report each document's native contrast, signed margin changes and attenuations,
plus four control effects. Persistent small-strength signs would locate the problem
in conditional source/readout alignment rather than exclusively large-edit curvature;
it would not identify a token-level cause. No fresh/selectivity/composition promotion.
