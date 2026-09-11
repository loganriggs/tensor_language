# Hourly strategic review — 11 September 09:22 UTC

The circuit targets remain explicit read/operation/write computation; grouping
across modules and splitting within them; held-out/OOD prediction; executable
extraction or sufficiency; selective removal/interchange; predictable composition
and reuse; and stable identification. The full goal is a simpler transparent
program with the four behavioral properties, not a weight-reconstruction score.
This review covers 08:22–09:22; the snapshot audit below was completed just after
the boundary while writing the review.

## Decision and evidence

Continue the already running overcomplete L1 discovery, with its frozen
weight-only objective and matched untrained-dictionary controls. Do not extend
the completed MSP geometry comparisons. Ordinary nonorthogonal bases converged
but failed their quality targets; Tyler fits timed out and remain inconclusive
about their local optima. Exact conditional Down fits improve all eight
programs, but no learned family gets the registered one-percentage-point gain.
The best ordinary result is 30.76% coefficient capture versus refitted PCA's
29.17%. Its two refitted functions have cosine only 0.368 despite converged
bases and exact output solves. This is an actual function difference, not just
coordinate relabeling, and prevents a stable-identification claim.

The full-U sparse-support update on 128 products settled locally after its
saved-state polish: 29.906% to 29.944% full coefficient capture. The original
timeout remains a miss; further gain after convergence was negligible. No
identical continuation is needed. Reuse the exact conditional operator when
testing a materially different feature/support hypothesis.

The exact radial correction is a useful example of metric sensitivity:
ideal-sphere capture reaches about 74%, but radial-plus-bias alone scores
63.22%. It does not establish natural-input preservation. The small, already
queued FineWeb diagnostic uses eight frozen alternatives and 8192 positions,
with native/physical replay controls and no fitting. This explicitly tests
the metric interpretation of weak candidates; it is not promotion following
a passed structural screen. No million-token discovery sweep is added.

Primary evidence: [method index](WEIGHT_ONLY_METHODS_INDEX.md),
[completed result explanation](explanations/2026-09-11/explanation_2026-09-11_0902.md#update0918),
[L1 protocol](OVERCOMPLETE_L1_READER_V1_PREREGISTRATION.md),
[frozen FineWeb protocol](FROZEN_RADIAL_FINEWEB_V1_PREREGISTRATION.md).

## Initialization and encoding assumptions

For a unit reader x and unit-ball dictionary rows B, the Lasso row objective is

$$
\tfrac12\|x-zB\|_2^2+\lambda\|z\|_1
\ge \tfrac12(1-r)^2+\lambda r
\ge \lambda-\tfrac12\lambda^2,\qquad r=\|zB\|_2,
$$

for 0<lambda<1. The first inequality uses the reverse triangle inequality and
the atom norm constraint. If an atom equals x, using its coefficient 1-lambda
attains the bound. Thus the 2304 sampled initial atoms already give globally
optimal individual code objectives for 37.5% of the 6144 training readers.
This can anchor features to selected readers, but does not prove that the
joint optimum is memorization: moving atoms can help the other readers.

An immutable iteration81 snapshot shows median squared-code participation
2.33 readers, with 60.42% of code energy on each atom's own initial reader.
Yet median nonzero usage is 88 readers and median initial-atom cosine is
0.8906, missing the registered >=0.9 prediction. Selected-reader capture is
77.89% versus 49.92% for other training readers. Their row objective has worsened
from its attainable initial bound, so the atoms are not unchanged copies.
These are intermediate training/code statistics, not final results or additive
folded-function energies. Keep the held-out weight comparison untouched.

The first snapshot audit incorrectly accepted an absent same-iteration
objective replay. Its original instrument verdict is **unverified**, despite
the stored boolean. A separate Gram-contraction check on the exact frozen
bytes agrees with the computed objective to 5.55e-17; it cannot invent the
missing historical diagnostic. Preserve both receipts and this correction.
[Snapshot](L1_READER_ANCHOR_V1_AUDIT.json),
[independent check and correction](L1_READER_ANCHOR_REPLAY_V1_AUDIT.json).

Another concrete assumption needs a small check before interpreting the final
encoder: its fixed top128 rule can pad genuinely sparse Lasso solutions with
zero-score ties. Subsequent least-squares refitting can make those arbitrary
extra features active. Test feature-permutation invariance on a small exact
example with the existing encoder. Do not change the live job or its thresholds.
This distinguishes inference ambiguity from unsuccessful feature discovery.

## Alternatives and confounds

Priority order:

1. Finish and interpret the L1 convergence, held-out reader, full tensor and
   sampled-dictionary comparisons. Training concentration alone is insufficient;
   local convergence plus failure to beat its untrained control prevents
   promotion of this specific objective/encoder, not all sparse structure.
2. Test zero-support padding with a bounded synthetic counterexample. If
   permutation changes the fitted function, audit the encoder separately before
   drawing a negative feature conclusion. If it does not, retain the null.
3. Run the frozen FineWeb diagnostic already queued. Failure of physical replay
   invalidates interpretation; disagreement with sphere predictions rejects
   that metric transfer, not the algebraic radial identity.
4. For subsequent discovery, favor direct folded-polynomial objectives and
   adaptive overlapping components over another orthogonal rotation. Existing
   free-product, block and conditional-solve receipts must be checked first;
   an unfinished optimization is not an exhausted structural family.

Reader normalization gives equal weight to different native readers and ignores
output importance during dictionary discovery. Native product pairings, fixed
penalty and support size remain restrictive. Weight-row holdout is not document
holdout. FineWeb is training-corpus validation; Pile is separately labelled
shift testing. Sign/scale gauges, nonlinear RMS/tanh composition, subtraction
of a large background, and post-result selection remain explicit confounds.

## Throughput and continuation

The phase log for 08:22–09:22 records science36.19, implementation5.45,
validation0.49, publication11.26 and review6.61 minutes. Scientific work plus
implementation totals41.64 versus18.36 minutes for the other categories.
These coarse foreground intervals overlap asynchronous GPU work. Zero new
behavioral circuit screens or identified circuits were produced; the user's
current directive prioritizes weight-only discovery.

- **CIRCUIT_FOCUS: PASS within the user-directed discovery scope.** The work
  tests reusable feature computation and stable identification, with no claim
  of meeting the older one-circuit-per-ten-minutes target.
- **CEREMONY_BUDGET: PASS.** Overhead stayed below scientific work; preserve
  detailed updates at hourly/major/user boundaries and reuse existing kernels.
- **NOVELTY_LESSON_GATE: PASS.** Existing ALS, sparse encoders, readout execution
  and CP inner products were reused. The initialization bound and padding test
  address different failure mechanisms; none is counted as an identified unit.

Continuation is concrete: native L1 is live, frozen FineWeb is queued, and the
training-only snapshot and independent objective audit were executed. The next
CPU step is the encoder permutation test. Next hourly10:22; next math10:51.
