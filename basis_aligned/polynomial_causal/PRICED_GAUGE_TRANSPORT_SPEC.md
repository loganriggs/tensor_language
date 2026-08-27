# Preregistered priced gauge-transport test

## Question

Does a discovered latent slice define a reusable causal program interface, or is it
only a coordinate-dependent locator?  A reusable interface must transport unseen
interventions, retain its meaning in a second background, compose with another
intervention, and have a description length that is stable under equivalent gauges.

This test is deliberately downstream of `writer_floor_question.py` and the frozen
ship oracle.  Those results may choose a registered source site or license an OOD
arm, but they may not change the metrics, controls, or gates below.

## Typed transport and its gauge

At source and destination residual boundaries let

```text
E_s : physical source residual -> source coordinates
D_t : destination coordinates -> physical destination residual
A   : source coordinates -> destination coordinates
tau = D_t A E_s
```

`tau` is the physical intervention map.  Under coordinate rewrites

```text
c_s' = G_s c_s,                         c_t' = G_t c_t
E_s' = G_s E_s,                         D_t' = D_t G_t^-1
A'   = G_t A G_s^-1,
```

the physical map is exactly unchanged: `D_t' A' E_s' = tau`.  Claims about a
transport therefore use `tau`, not entries, sparsity, or norms of `A`.  In an
orthonormal interface grammar the admissible coordinate gauge is `O(k)`; in a
general encoder/decoder grammar it is `GL(k)` and the condition number is reported.

These interface gauges are not the CP gauge of a bilinear MLP.  A generic CP layer
has only per-product scale/sign and product-permutation freedom because arbitrary
mixing destroys the frozen elementwise-product core.  CP factors are balanced and
canonicalized under that smaller group.  A factorized *linear* bond does have an
internal `GL(r)` gauge.  The three groups may not be interchanged in analysis.

### RMSNorm semantics

Every endpoint is named as pre- or post-RMSNorm.  Post-norm interventions use the
ordinary physical residual metric.  Pre-norm live interventions execute the actual
`rsqrt(mean(x^2)+epsilon)` and are a separate arm; finite epsilon means input scaling
is not asserted to be an exact gauge.  A frozen-norm arm is a diagnostic linearized
semantics and cannot license a live interface by itself.

## Price

The primary standalone price is a canonical encoding of the physical operator
`tau`, its graph endpoints, precision, and norm semantics.  A dense/factorized
linear rewrite is canonicalized by the SVD of its product, eliminating its internal
`GL(r)` gauge.  If `E_s` or `D_t` is not already an admitted frozen library object,
its standalone bytes are paid here.  Activation patching is not a deployable program
by default: a position-indexed transport also pays for its quantized coordinate
field, position/address rule, and every repeated clamp site.  Downstream material
created after a clamp is either reproduced by the candidate forward program or
counted as response distortion; it cannot be credited to the basis for free.

An amortized price may replace admitted encoder/decoder bytes with immutable library
references and price `A` once.  It is reported only after a joint orbit
canonicalizer makes that representation invariant to simultaneous interface gauge
rewrites.  Until that codec exists, the amortized number is explicitly a compiler
upper bound; pointing into the original model's weights is not free.

Report both the current Theseus benchmark price and the experimental canonical-codec
price.  The latter does not replace benchmark policy.  A transport result is
structural only if its matched-price ranking is stable in both grammars.

## First concrete ladder

### Stage 0: algebraic and measurement preflight (CPU)

1. Generate nonsingular `G_s,G_t` with condition numbers in `[1, 100]` and verify
   physical-map and response equality after the complete gauge rewrite.
2. Verify an exact same-boundary identity intervention through the clean harness.
3. Verify an exact CP scale/sign/permutation rewrite of one bilinear layer.
4. Confirm that an intentionally incomplete gauge rewrite and a position shuffle
   are detected by the response metric.
5. Verify canonical price drift at most 1% across all complete rewrites.

Stages 1--3 are barred if an exact positive control fails.  A failed negative
control makes the relevant metric underpowered and also bars promotion.

### Stage 1: no-teacher-forcing commuting triangle

Use post-block residual boundaries `L8 -> L11 -> L14` at rank 64.  This rank is a
registered first test, not a claim of minimality: ranks 8 and 16 did not beat the
existing random-256 alignment control, while rank 64 did.  Split data three ways:

1. **basis rows:** independently fit local token-deviation bases `U8,U11,U14` and
   freeze them before any intervention is observed;
2. **response-fit rows:** inject small natural-scale perturbations at L8 and,
   separately, L11; fit centered response maps `T8,11`, `T11,14`, and direct
   `T8,14` by ridge from `dc_j = U_j^T(x_j^I - x_j)`;
3. **evaluation rows:** choose no basis, rank, ridge penalty, scale, or map.

Use raw orthonormal coordinates.  If whitening is later added, store the full
coordinate metric `C_l`, transformed as `C_l' = Q_l^T C_l Q_l`; a diagonal scale
would reduce the declared gauge to signed permutations.  Downstream RMSNorm stays
live.  Discovery calibrates perturbation RMS once so median early-intervention KL is
in `[0.01, 0.20]`, then freezes it.

On untouched rows and a wholly held-out perturbation family (coordinate cuts or
position-matched donor-minus-target swaps; fitting uses isotropic perturbations),
intervene once at L8 and compare:

- the true L8-intervened forward pass;
- a full-state L14 oracle patch, as a harness control;
- a true projected-`U14` oracle patch, testing interface sufficiency;
- the direct prediction `U14 T8,14 dc8`;
- the chain `U14 T11,14 T8,11 dc8`, with no true L11 value;
- natural-state regression, shuffled-response, and matched random-subspace maps.

Every predicted patch is added to the *baseline* L14 state.  Applying it to the true
early-intervened L14 state leaks the response and invalidates the run.

### Stage 2: behavior and alternate-background extension

Only after the triangle passes its oracle-interface and direct/chain gates, freeze
the maps and score question, pronoun, copy, and novel/rare cells.  Then rerun the L8
intervention in the current full ship and one registered rank-32 compressed
background.  Any background-specific refit is a separately priced diagnostic; the
headline uses the one clean-fitted map.

The behavior-specific extension uses the already frozen two-dimensional
`question@mlp11` and rank-8 `pronouns@mlp17` interfaces.  The question source is the
top nonredundant writer group selected by `writer_floor_question.py`; if its written
gates fail, use the inherited `{attn10, attn9, mlp9, mlp10}` group and label it as
such.  Hold out pair/triple writer and reader/final-cut families.  The scalar Mobius
pair model is a required baseline, not vector-transport fit evidence.

### Stage 3: conditional OOD arm

Run the code-corpus intervention arm only if the authoritative same-realization
FineWeb oracle licenses a content site.  Use the frozen file-disjoint code corpus,
file-cluster uncertainty, and the identical ship realization.  OOD cannot rescue a
transport that failed Stages 1 or 2.

## Controls

Every family uses the same rows, amplitudes, RMS semantics, and output metrics for:

- exact identity/same-site positive control and the full-state destination oracle;
- the true projected-destination-subspace oracle;
- the candidate physical transport;
- zero transport and identity-coordinate transport where shapes permit;
- Haar-rotated coordinates with matched rank and singular spectrum;
- position-shuffled transport;
- wrong-behavior and wrong-depth transports;
- same-token other-context raw donor and discovery-fitted affine donor;
- token-mean/lexical residual transport;
- unperturbed-state regression fit between the same local bases;
- the existing scalar pairwise-response model when scoring scalar effects.

Random and shuffled controls match rank, physical intervention RMS/energy, patched
position count and coverage pattern, and clamp multiplicity.  Each random basis is
Haar-rank-64 inside that layer's independently frozen top-256 activation support and
gets its own optimally fit maps at identical price.  The same seeded null triplets
are reused across behaviors, backgrounds, and OOD arms.  Control choice and rank
never use evaluation outcomes.

## Metrics and uncertainty

For intervention family `f`, with physical response vectors `Delta_i` and predictions
`Delta_hat_i`, define

```text
NRE_f = sqrt(sum_i ||Delta_hat_i - Delta_i||^2 / sum_i ||Delta_i||^2).
```

The primary coordinate causal error is `max_f NRE_f`; do not pool families.  At the
output define the commuting-diagram error

```text
E_out = sum_i KL(p_early_i || p_transported_i)
        / sum_i KL(p_early_i || p_baseline_i).
```

Use ratios of aggregate sums, not means of unstable per-example ratios.  Also report
normalized MSE of centered pre-softcap logits, coordinate response `R2`, the
90th percentile per-example relative error with a discovery-frozen denominator
floor, response cosine, and sign agreement on preregistered powered scalar effects.
All deltas are intervention-minus-own-baseline responses, not intervened-output
agreement.

Report target-effect recovery, collateral CE/KL outside the target class, and the
joint response produced by actually installing simultaneous interventions.  Do not
substitute a sum of singleton effects for the composed forward pass.

Use row-cluster bootstrap on FineWeb and file-cluster bootstrap on code.  Comparisons
to nulls are paired.  A candidate must beat every one of 20 frozen stochastic nulls;
the finite-null p-value is `(1 + #null >= candidate) / 21` in the direction of the
registered score.  Confidence intervals and failures are preserved even if a later
arm succeeds.

## Frozen gates

A Stage-1 triangle passes only if all are true on held-out families:

1. exact identity relative error is at most `1e-5`, full gauge-rewrite response
   drift is at most `1e-6` in float64, and canonical price drift is at most 1%;
2. the full-state L14 oracle has `E_out <= 1e-3` and centered-logit relative error
   at most `1e-3`;
3. the true projected-`U14` oracle has `E_out <= 0.25` in every held-out family;
4. the direct map has coordinate `NRE <= 0.50` (`R2 >= 0.75` under the same
   aggregate convention) and `E_out <= 0.25` in every held-out family;
5. the chain has `E_out <= 0.35`, is no more than `0.10` worse than the direct map,
   and retains at least 75% of its response alignment and powered target effect;
6. direct and chain response `R2` exceed unperturbed-state regression and the best
   of 20 matched random triplets by at least `0.10`; both beat all 20, while the
   shuffled-response `R2 <= 0.05`;
7. no powered behavior/token cell has `E_out > 0.35`, and powered target-effect sign
   agreement is at least 0.80.

Generic-interface promotion additionally requires, without gate changes:

8. both question and pronoun meet output `E_out <= 0.35`, sign agreement at least
   0.80, and target recovery at least 0.70 with collateral at most 0.25 times the
   target effect;
9. moving from clean to either alternate background has `E_out <= 0.35` and retains
   at least 75% of the candidate's advantage over its best null;
10. matched-price candidate rankings have Kendall tau at least 0.8 across the two
   frozen price grammars.

If gate 3 fails, the target basis is a locator and no fitted transport can rescue it.
If gate 4 fails, response dynamics are not transported.  If gate 5 fails, the maps
are local but not composable.  If gate 8 fails, the object is not selectively
editable; if gate 9 fails, it is not a whole-program API.  Gauge or exact-control
failure invalidates the measurement rather than falsifying the model claim.

## Existing evidence and expected information gain

This experiment is needed because the present evidence separates transport-like
correlation from interface semantics:

- same-context self transplant has mean recovery `0.781` (median `1.0`), while a
  raw same-token donor is `-0.357` and a fitted global affine donor is `0.000`;
- position-matched activation patching produces large alignment/KL signals and
  replicates in a SwiGLU control model, but its original content-vs-random ratio was
  only `1.345`, below its registered `1.5` gate.  Content-vs-function masks do no
  better than size-matched random masks and several depth/contiguity predictions
  fail.  It is a sensitive positive measurement control, not a prior interface win;
- the patch is a position-indexed field: patched positions align at about `0.891`
  and unpatched positions at `0.358`; a shuffled coordinate field reaches only
  `0.565`.  Full nine-site clamping reaches `0.899`, while no unique early/middle/
  late band dominates.  Roughly 32% of full-patch KL remains after stripping the
  final content coordinates, so basis dimension alone underprices the mechanism;
- the question cut ledger predicts its unseen triple substantially better with
  pair interactions than additively (`0.150` versus `0.696` normalized question-CE
  error), showing that composition is measurable and non-negligible;
- an output rank-8 locator retained half of oracle head recall but only 13.5% of
  oracle removal damage, so basis alignment alone is not a causal API.

Thus a pass would add something the current project does not have: a gauge-honest,
priced interface that predicts edits across components and backgrounds.  A clean
failure is also high-value because it prunes latent transport from the program
grammar and sends effort back to common physical boundaries and explicit interaction
terms.
