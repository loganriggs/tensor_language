# MLP9 contextual DCT range test V1

## Question

Raw weight-only MLP9 DCT contains four reproducible factors but its output span
captures no more of the CrossFirst suffix reader than random subspaces.  Test
the briefing's next required distinction: retain the native state and RMS
normalizer as open context slots instead of decomposing the context-free raw
bilinear tensor.

For the exact first consumer

`g_z(theta) = z + theta + MLP9(RMS(z + theta))`,

evaluate `D2 g_z(v,v)` at every token position.  Input probes are fixed before
model access: 16 deterministic orthonormal Gaussian directions plus the four
raw-weight DCT input factors that were independently stable across seeds.  No
suffix reader, endpoint, logit, or behavioral outcome enters these responses.

On the 48 opened `CROSSFIRST_HESSIAN_TOP2_FRESH_V1` discovery prefixes, form
the output covariance of all contextual Hessian responses and freeze its top
eight eigenvectors.  Also freeze random-probe-only and stable-DCT-probe-only
comparators.  Evaluate, without refitting, on the disjoint 48 new-endpoint
`CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1` prefixes:

1. fraction of contextual Hessian response norm retained by each basis; and
2. fraction of the exact downstream UK-minus-US suffix-reader norm lying in
   each basis.

Compare reader coverage to 16 deterministic Haar rank-eight output subspaces
and to the earlier raw-weight DCT rank-eight output span.  This is a randomized
range finder for prompt-conditioned DCT, not a full CP recovery or a causal
circuit test.

## Predictions

- **A — instrument:** MLP9-stage native replay is at most `2e-6`, and four
  swapped mixed-JVP order checks agree within `2e-5` relative error.
- **B — contextual compression:** the combined top-eight basis retains at least
  `.50` of contextual Hessian response norm on discovery and `.40` on the
  disjoint new-endpoint panel.
- **C — discovery reader:** combined-basis reader coverage is at least `.30`,
  exceeds the median random control by `.15`, and exceeds raw-weight DCT by
  `.15`.
- **D — endpoint transfer:** combined-basis reader coverage is at least `.25`
  overall and `.20` in every confirmation family, and exceeds every random
  control on the confirmation panel.
- **E — context value:** the combined basis exceeds both the random-input-only
  and stable-DCT-input-only contextual bases by at least `.03` confirmation
  reader coverage.
- **F — audit:** serialize frozen bases, covariance spectra, response retention,
  prompt/family reader coverage, comparator coverage, and exact price.

If A fails, repair mechanics before interpretation.  B failure means rank eight
does not compress this open-context response family.  C failure means context
does not rescue an unsupervised reader on discovery.  D failure rejects
transfer to new endpoints.  E failure means the stable raw DCT inputs add no
value beyond behavior-blind random probing.  Passing nominates the frozen basis
for a genuinely fresh causal extraction/removal test; it does not itself prove
any of the four target traits.

## Price

One checkpoint load; 96 already-open prefixes split 48/48; one batched
double-JVP over 20 directions and one suffix gradient per prefix; four symmetry
checks; three 1,152-dimensional covariance eigendecompositions; 16 random
rank-eight output controls; no physical intervention, behavioral fit,
coefficient fit, or parameter update.
