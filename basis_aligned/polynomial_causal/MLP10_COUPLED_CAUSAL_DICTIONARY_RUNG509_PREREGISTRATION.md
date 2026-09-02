# Rung 509: source-conditioned coupled MLP10 causal dictionary

Status: prospectively frozen after rung508's zero-family strong null and before any exact-term finite outcome on
documents 500:1000 is computed for this rung.

## Question

Rung508 showed that the same equality-score action is stable at the whole-action level but is implemented through
different MLP10 source-family effects under `N`, `P`, `Z7`, and `Z8`. A single hand-chosen partition therefore asks
for the wrong invariance. This rung asks whether different exact input terms under the four implementations can be
matched to the same downstream causal variable.

This directly targets within-MLP splitting, grouping across native components, held-out/OOD prediction, selective
finite manipulation, and later composition. It is not a low-rank, quantization, reconstruction-error, or CE-only
experiment. The fitted dictionary is only a candidate generator until its weighted terms survive physical removal
on unopened documents.

## Exact terms and finite response tensor

Use rung507's frozen 22 normalized input sources and all 253 unordered source pairs. For score implementation `a`
and exact term `p=(s,t)`, let `deltaY[a,p]` be that term's exact change in the float32 semantic MLP10 output relative
to the score-absent run. The term is physically removed by subtracting `deltaY[a,p]` from the deployed BF16 MLP10
write and recomputing layers11--17.

For each removal, record a 34-coordinate causal response:

- four cross-entropy changes on the fixed copy contexts `(near, far, one earlier match, multiple earlier matches)`;
- thirty member-minus-matched-control cross-entropy changes for the frozen held-out circuit tags already loaded by
  rung508.

All coordinates come from finite forward passes. Gradients may not enter fitting, selection, or scoring. The tensor
`R[a,p,h,c]` has four score implementations, 253 exact terms, two document halves, and 34 downstream coordinates.

## Coupled dictionary

Fit exactly eight candidate atoms. Eight is a fixed identification budget inherited from rung508's registered upper
bound, not a searched rank and not an adoption claim.

For atom `k`, score implementation `a`, and exact pair `(s,t)`, define a nonnegative assignment

`g[a,(s,t),k] = softmax_k(q[a,k,s] + r[a,k,t] + q[a,k,t] + r[a,k,s])`.

Thus every exact term is divided across eight atoms, assignments sum to one, and each atom is explicitly coupled to
the sources read by the Left and Right bilinear branches. The symmetric formula makes swapping the two branches an
allowed gauge. The fitted `q/r` logits themselves are not interpreted; the resulting assignments `g` are.

Each atom also has one shared 34-coordinate downstream response `w[k]`. On the first discovery half, fit `q`, `r`,
and `w` to predict the measured finite singleton responses

`R_hat[a,p] = sum_k g[a,p,k] * w[k]`.

Coordinates are divided by their first-half root-mean-square value before fitting. Use fixed Adam settings:
2,000 steps, learning rate `.02`, weight decay `1e-4`, and assignment-entropy penalty `.01`. Run seeds
`5090,5091,5092`. Repeat the complete fit independently on the second discovery half. There is no hyperparameter,
rank, seed, or checkpoint selection. Report prediction error, but it is never sufficient for a circuit claim.

Atom permutation is resolved by the maximum-total-cosine matching of the eight shared responses. Additive gauges in
`q/r` are ignored because only `g` is scored.

## Candidate stability before intervention

An atom is eligible only if, after permutation matching:

- its shared response has cosine at least `.80` across all three restarts within each half;
- its 4-by-253 assignment table has cosine at least `.75` across those restarts;
- the independently fitted half0 and half1 atoms have response cosine at least `.70` and assignment cosine at
  least `.65`;
- under every score implementation, at least two exact terms have assignment at least `.50`, and no one term carries
  more than `.80` of that atom's assignment mass; and
- every pair of eligible atoms has shared-response cosine at most `.90`, preventing duplicated labels.

Retain every eligible atom without ranking. The dictionary is identifying only if two through eight atoms are
eligible. Outside that range is the registered strong null.

## Physical atom removal and prediction

For fixed fitted assignments, the atom's exact output change is

`deltaY_atom[a,k] = sum_p g[a,p,k] * deltaY[a,p]`.

Because assignments sum to one, summing all eight atoms must reproduce the complete named score-induced MLP10 change
up to the already registered numerical remainder. Subtract each `deltaY_atom` from the deployed BF16 MLP10 output and
recompute the real suffix. This is a new finite group intervention; it is not inferred from the singleton fit.

On discovery, an eligible atom passes only if for every score implementation its four-task effect norm is at least
`.00025` nat, its two-half cosine is at least `.50` with norm ratio at most3, and its all-copy magnitude is at least
`.00025` nat and twice its off-target magnitude. Its pooled effects under `P/Z7/Z8` must each have cosine at least
`.70` with `N` and norm ratio at most3.

Without refitting or rematching, rerun the same weighted removals on documents752:1000, halves752:876 and876:1000.
An atom confirms only if every source has confirmation norm at least`.00025`, discovery-to-confirmation cosine at
least`.60`, confirmation-half cosine at least`.50`, norm ratios at most3, and the same all-copy/off-target rule;
`P/Z7/Z8` must each have confirmation cosine at least`.65` with `N`.

For each confirmed atom report its source-assignment map, the exact terms with assignment at least`.50`, and whether
the same downstream variable uses different source pairs under different score implementations. Those are the
candidate explanations; names are not assigned from weight similarity alone.

## Composition gate

If two through eight atoms confirm, physically remove every unordered atom pair on confirmation. Fit the same
discovery-only additive, left-redundant, right-redundant, or one-scalar interaction rule used by rung508, then require
confirmation cosine at least`.70`, relative residual at most`.65`, positive prediction cosine in both halves, and
the all-copy/off-target rule. At least one pair must predict to license a composition claim.

## Controls, price, and stopping rules

The run must pin rung508's result and its A-true/B-false route; verify all 22 sources, 253 exact terms, 34 response
coordinates, data supports, calls, and patches; reproduce score calibration; preserve raw-input, normalized-input,
float32-output, BF16-output, and eight-atom partition remainders separately; and require every requested edit to be
nonzero. Documents748:752 remain unused.

Singleton collection costs, per 248-document phase, `62 * (1 + 4*(1+253)) = 63,054` full forwards: one absent
capture and, per score source, one intact plus 253 removals. The eight fitted atom removals add
`62 * 4 * 8 = 1,984` forwards per phase. Confirmation is opened only with two through eight eligible atoms. Pair
confirmation adds `62 * 4 * choose(q,2)` forwards for `q` confirmed atoms. The maximum complete price is therefore
`2*(63,054+1,984) + 248*choose(8,2) = 137,020` full forwards, zero backwards, and six small CPU fits
(three seeds fit independently on each of the two discovery halves). No deployed parameters are added or saved.

### Registered predictions

- **A:** the exact-term, response-coordinate, assignment-partition, numerical, calibration, call, and liveness
  instrument passes.
- **B:** two through eight atoms satisfy every restart and half-stability rule.
- **C:** at least two atoms pass the physical discovery removals and confirm without refitting.
- **D:** at least one discovery-frozen atom-pair rule predicts its confirmation removal.
- **E:** at least one confirmed atom uses different majority source pairs under two calibrated score implementations
  while preserving the same downstream response. This is the proposed action-level/shared-variable explanation of
  rung508's replacement-dependent internal realization.

The strong null is A false, failed score calibration, fewer than two or more than eight stable atoms, fewer than two
finite confirmations, no predictable pair, or no source-changing shared atom.

- A false: repair only the instrument.
- B false: the eight-atom coupled vocabulary is not identifiable; test a downstream predictive-state quotient that
  does not require a latent dictionary, rather than sweeping atom count or penalties.
- B true/C false: the fit describes singleton responses but not executable grouped computations; close this
  dictionary form.
- C true/D false: retain identified atoms but model higher-order suffix state before claiming composition.
- D true/E false: the stable atoms do not explain the action-level/internal mismatch; audit score-specific paths.
- A--E true: validate on OOD code and construct an executable MLP10 subprogram using only confirmed atoms.

No outcome licenses a rank sweep, quantization, threshold tuning, favorable-seed selection, or calling response
reconstruction a circuit.
