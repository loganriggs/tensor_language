# Rung 460: code-OOD confirmation of the shared equality score

Status: prospective confirmation design, frozen after rung 459 and before loading the `ood_code` row tensor or
running any code-corpus score transplant. This is OOD identification, not compression or adoption.

## Frozen natural-text hypothesis

Rung 459 selected exactly one object from 144 fitting candidates and confirmed it on held-out natural text:

- source score: L5H5;
- target term: L8H4;
- transplanted factor: the complete equality-restricted double-QK score product;
- retained target payload: L8H4's own value-after-output vector;
- first selected downstream reader: MLP9;
- fitting score RMS ratio: the exact `L5H5->L8H4.score_ratio` stored in the hash-bound rung-459 result;
- frozen between-group control: transplant L7H3's score into L8H4 using the exact natural fitting ratio stored for
  `L7H3->L8H4`.

No pair, factor, reader, scale, task condition, threshold, or control may be reselected on code.

## Data and arms

Use only the 192-document `ood_code` role from the already-frozen induction-equality-v2 receipt. This role is
repository-file-disjoint from the natural roles by its original authority. Split it into fixed reporting waves
`0:96` and `96:192`; neither wave fits anything.

Run native and empty analytical replay for instrument integrity. For the selected and control early-to-late pairs,
run only:

- base: remove the early equality term and L8H4 equality term;
- reference: remove the early term but leave native L8H4;
- score transplant: remove both, then add the early score multiplied by the **natural-fit frozen** score ratio and
  L8H4's current code-input payload at layer 8.

Capture only MLP9. Recompute code score RMS values and direct score cosines solely as diagnostics; they cannot alter
the natural-fit ratios or arms. Do not run code payload transplants, relocated-whole terms, other readers, other
pairs, subset searches, SEALED attention0 outcomes, or another row role.

## Metrics

For MLP9 and each task condition, compare the reference response and score-hybrid response relative to base exactly
as in rung 459. Report positive cosine, reference-relative error, liveness, and

`task_margin = cosine_positive - max(cosine_matched_negative, cosine_off_target)`.

For all-positive CE define

`recovery = [CE(base)-CE(score hybrid)] / [CE(base)-CE(reference)]`.

Use 20,000 shared document-bootstrap draws and the same seed family with a new rung-460 suffix. Report matched-
negative and off-target hybrid-minus-reference CE. Compute recovery and response metrics in both fixed waves.

For interchange, use the per-document absolute score-hybrid-minus-reference CE discrepancy. The selected L5H5
pair is “within”; the frozen L7H3 pair is “between.” Use 10,000 label permutations at seed 460.

## Registered predictions

### A. Instrument

All parent-result/source/model/row/role/mask hashes, exact factor reconstruction, empty replay, arm identity, natural-
scale reuse, source-before-target cache timing, and no-extra-outcome clauses hold. Factor reconstruction relative
squared error is at most `1e-10`; replay relative squared logit error is at most `1e-12`.

### B. MLP9 response transfers to code

For the frozen selected pair, all-positive response cosine is at least `.65`, task margin at least `.05`, reference-
relative error at most `.70`, and both responses at least `1e-4` of the MLP9 write RMS. Each fixed wave has positive
response cosine and positive task margin.

### C. The score transplant preserves the code task effect

The native L8H4 reference stake is positive in the point estimate and every bootstrap draw. Recovery is at least
`.40`, its simultaneous 95% lower bound is above `.20`, and recovery is positive in both waves. Absolute off-target
hybrid-minus-reference CE is at most `.01 nat`.

### D. The frozen between-group control transfers

Between/within mean absolute per-document CE discrepancy is at least `2.0` with label-permutation `p <= .05`.

### E. Direct score geometry supports the same operational grouping

On code all-positive equality edges, direct L5H5/L8H4 score cosine is at least `.60` and exceeds direct
L7H3/L8H4 score cosine by at least `.30`. This is supporting geometry only; it cannot compensate for failure of B,
C, or D.

The strong null is instrument failure, selected response cosine below `.30`, recovery at most `.10`, interchange
separation at most `1.2`, or selected direct score cosine no larger than the frozen control.

## Claim boundary and successor

Only A+B+C+D+E identifies a natural-and-code shared equality matcher. It still deploys no replacement and saves zero
parameters. If it passes, the next circuit question is which of the two QK score branches supplies the shared
feature: freeze branch-preserving hybrids of `score1 × score2` and test Q/Q2 versus K/K2 feature reuse without
changing the now-identified payload-specific boundary. If code fails, retain the natural-text identification but do
not call the matcher OOD-general.

