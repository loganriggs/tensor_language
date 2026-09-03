# Rung 522 CPU preflight addendum

**Frozen:** 2026-09-03 05:53 UTC, after the CPU planted toy and before any rung522 model fit, CUDA smoke, or model
outcome. This addendum fixes the implementation choices and resolves two ambiguities in the preregistration. It does
not change any scientific outcome threshold.

## Mathematical and optimizer choices

- The learned variable is a `1152 x 4` float32 frame. Every model use applies reduced QR retraction with
  deterministic signs from `diag(R)`. This is a differentiable Stiefel-manifold retraction. Symmetric polar
  retraction through `eigh(Q^T Q)` was rejected for optimization because its eigenvector derivative is undefined at
  an exactly orthonormal frame, where the four eigenvalues coincide; the CPU preflight produced non-finite gradients
  at update zero. All scientific comparisons use the projector and therefore ignore QR's column gauge.
- Initialization starts from a CPU Gaussian matrix determined by the fit seed. It is symmetrically polar-retracted in
  float64 and then transferred to the model device as float32. CUDA random-number generation does not affect the
  initial subspace.
- Optimizer: Adam, learning rate `.03`, betas `(.9,.999)`, epsilon `1e-8`, no weight decay, no gradient clipping,
  exactly `200` updates, and the final iterate as checkpoint. There is no best-loss checkpoint selection.
- Objective epsilon is `1e-12`. The task-conditioned loss is exactly
  `max_target (L_member + 24 L_control)`. Recovery-only controls use coefficient zero. One differentiable
  `torch.max` is taken over all participating targets; targets are never averaged or alternated between updates.
- Health windows are the first and last 20 updates. The preregistered gates remain unchanged: finite values and
  gradients, no model-parameter gradients, attention8 called once, frame orthonormality error `<=1e-5`, projector
  movement `>.02`, final-window loss below initial-window loss, fixed VALIDATION objective below initialization, and
  an exact self-donor activation/logit no-op.

## Balanced training batches

One leave-one-target-out update contains four recipient/donor row pairs: a member-anchored row and a matched-control-
anchored row for each of the two fitted targets. Within each role, rows follow a seed-defined permutation and cycle
only after exhaustion. FIT `D0` donor maps cycle `0,1,2,3` by `update mod 4`. All eligible tokens in the selected row
contribute to that role's mean; tokens belonging to another role do not. The two separately normalized target losses
enter the exact maximum.

A target-specific oracle uses two member and two control roles. A later all-three fit, if opened, uses six roles—one
member and one control row for every target—rather than dropping or averaging a target to force batch size four.
Label-permutation fits use the same role counts after permuting labels within the frozen overlap lattice and matching
strata.

If any role has no eligible row, if one projected model execution cannot include all required roles, or if the
callback changes target identities between updates, the fit is invalid and stops. No batch rule may be changed in
response to model outcomes.

## Toy calibration, including the failed first setting

The single planted problem contains two orthogonal rank-4 subspaces in 32 dimensions. Members carry identical codes
and downstream readers in both copies, so the selective and broad projectors have exactly the same member loss,
`0.25`. Controls carry only the broad copy. Member response power therefore cannot distinguish the two projectors.

The first coefficient, `8`, is retained as a failed calibration receipt. At 400 updates it found the correct
projector—selective overlap `0.97275`--`0.97332`, broad overlap `0.02668`--`0.02725`, and concentration
`6.469`--`6.489`—but missed the toy scaled-residual gate: `0.05357`--`0.05986`, required `<=0.05`. Coefficients
`12`, `16`, and `24` were then checked on this same planted toy only. The first complete pass was `24`. After the
combined event-budget audit, the same coefficient passed all five seeds at 200 updates:

- selective-projector overlap `0.992747`--`0.992855`;
- broad-projector overlap `0.007143`--`0.007252`;
- held-out signed cosine `0.998824`--`0.999175`;
- optimally scaled residual `0.04061`--`0.04849`; and
- held-out member/control RMS concentration `11.631`--`11.847`.

No model activation, circuit-mask outcome, VALIDATION/TEST result, or CUDA computation entered this calibration.

## Clarification 1: the whole-attention8 comparator uses the new RMS statistic

Rung521's published concentration was

`mean_member |delta CE| / mean_control |delta CE|`.

Rung522's frozen gate instead uses

`RMS_member(delta CE) / RMS_control(delta CE)`.

These are different statistics, so Prediction A.6 may not compare a new RMS projector score with rung521's old
mean-absolute number. Rung522 will recompute the complete-attention8 swap on every Rung522 evaluation cell and score
it with the Rung522 RMS definition. The old `1.262`--`1.790` values remain context only. Prediction A.6 compares the
projector and full-attention8 RMS concentrations on exactly the same split, target mask, matched controls, donor
ensemble, and direction. The required improvement remains at least `1.0` absolute.

This resolution was fixed before any rung522 model outcome and does not change the `4.0` projector threshold.

## Clarification 2: evaluation splits, donor averaging, and batching

For Prediction A, “both document halves” means the two already frozen held-out data splits, VALIDATION and TEST—not
the two FIT power halves used by rung521. FIT `D0` is used by gradients. VALIDATION `D0`, VALIDATION `D1`, TEST `D0`,
and TEST `D1` are split-specific maps and are all unseen by gradients.

Within one donor ensemble, the signed per-token response is averaged over its four maps before cosine, residual,
recovery, RMS concentration, or fingerprint statistics are computed. `D0` and `D1` remain separate required cells.
Forward and reverse directions remain separate required cells. Concatenating arms in a physical GPU batch changes
neither mathematical averaging nor cell identities.

All 83 Prediction-A frames—15 real leave-one-out, 20 target-specific oracles including the reserved fourth target,
and 48 overlap-preserving label-permutation fits—are trained and frozen before TEST opens. TEST is executed once,
after health checks, VALIDATION outputs, and the no-selection rule are frozen. No TEST statistic selects a seed,
threshold, coefficient, checkpoint, or projector.

Inference evaluation uses six recipient rows per logical chunk. The eight donor maps and two swap directions are
concatenated, giving a physical no-gradient batch of `6*8*2=96` sequences. For a 216-row held-out split this costs
`ceil(216/6)=36` forward calls per frame. A managed instrument-only smoke must demonstrate that this batch is live
and fits GPU memory. If batch 96 fails, model science remains closed; it is not lawful to exceed the frozen call
ceiling or silently score fewer arms.

Training uses physical batch 4 for two-target/oracle/permutation fits and batch 6 for the conditional all-three fit.
The smoke must also execute one differentiable batch-6 projected intervention, confirm a finite frame gradient, and
confirm that every frozen model parameter has no gradient. No scientific response is retained from the smoke.

## Exact maximum price

There are at most 103 fits:

- real leave-one-out: `3*5 = 15`;
- recovery-only leave-one-out: `3*5 = 15`;
- target-specific oracles, including the reserved fourth target: `4*5 = 20`;
- label permutations: `3*16 = 48`; and
- conditional all-three fit: `5`.

At 200 updates, with exactly one projected forward and one backward event per update, the optimization price is

`103*200 = 20,600 forward events + 20,600 backward events = 41,200 combined events`,

below the frozen 45,000-event ceiling. Haar controls do not train. Any implementation needing a second projected
forward inside an update is invalid.

The worst-case inference ledger before the separately capped removal stage is:

- native capture: `95+36+36 = 167` forwards over FIT/VALIDATION/TEST;
- independent exact native replay: `167`;
- one self-donor batch in each split: `3`;
- FIT/D0 full-attention8 training targets, four donor maps concatenated: `95`;
- fixed initialization/final health batch for all 103 fits: `2*103 = 206`;
- complete-attention8 RMS comparators on VALIDATION and TEST: `2*36 = 72`;
- Prediction-A real/oracle/permutation evaluation: `(15+20+48)*2*36 = 5,976`;
- conditional recovery-only evaluation: `15*2*36 = 1,080`;
- 20 Haar controls on both held-out splits: `20*2*36 = 1,440`; and
- conditional all-three selection and test: `5*36 + 36 = 216`.

Total worst case: `9,422` inference-only forwards, below 12,000. Prediction D removal retains its separate 2,000-
forward ceiling. Fingerprints, ensemble averaging, response metrics, projector overlaps, principal angles, and
permutation quantiles are computed from saved outputs and add no model calls.

## Fail-closed stage order

1. CPU toy, unit tests, dependency hashes, dry-run, and the managed no-science smoke must pass.
2. Train the 15 real leave-one-out, 20 oracle, and 48 null frames; freeze all 83 before TEST.
3. Evaluate Prediction A exactly. If A fails, stop without recovery-only, Haar, final-shared, or removal stages.
4. If A passes, train/evaluate recovery-only controls and evaluate Haar controls for Prediction B. Failure stops.
5. If A/B pass, fit all-three projectors, select the VALIDATION Grassmann medoid, and evaluate C on TEST once.
6. Only an A--C pass opens mean-centered removal for D.

This stage order saves work after a negative result but never changes the worst-case registered prices or reads a
later-stage outcome early.

## Frozen CPU artifacts

- mathematical core SHA-256:
  `6cff6f7726dd8f76e786d64abf913fc31adbdfec101a97741a1aa3396f8431c2`
- planted-toy executable SHA-256:
  `5abbb09ec0871e0d7ad5b8cb63a3f6103027848700df36fcc3dc85ce21c42935`
- CPU unit tests SHA-256:
  `42b7b2f41fccf1c4f662f0b7dfeddc6f59836c9a9b26b611977007d8c00542c7`
- passing toy result SHA-256:
  `398842217e729e743dc4b5fe4947dc7837a40e01a42b2c267faa2249a6ad0fe4`
- toy canonical content hash stored in that result:
  `307997401e75d92162f978526ac3e09628f42d4219bd726e590f28fee21aa718`

The model entrypoint and smoke do not yet exist at this freeze. Their byte hashes and the frozen dependency census
must be added in a second pre-outcome implementation receipt before the managed smoke is enqueued.

Instrument-sequencing clarification, made before any rung522 CUDA execution: the no-science smoke may be hashed and
enqueued before the full scientific entrypoint exists, because its purpose is to establish whether the frozen
physical batch shapes and gradient isolation are feasible before spending implementation time on the complete
runner. The smoke's byte hash must be in a pre-outcome receipt before the smoke runs. The eventual scientific
entrypoint's byte hash must likewise be in a later pre-outcome receipt before any scientific rung522 run. Neither
receipt permits changing the registered scientific statistics, thresholds, fit count, or model-call ceilings.

## Pre-model red-team corrections — 2026-09-03 06:01 UTC

An independent audit found that the first stage order would open TEST before the recovery-only controls and final
shared projector existed. It also found that the proposed projector-overlap null did not match the real overlap
statistic. No rung522 model or CUDA outcome exists. The following corrections supersede the affected clauses above
and in the preregistration; all numerical response gates not mentioned here remain frozen.

### TEST remains closed until every learned object is frozen

All 103 possible frames are trained before TEST opens: the 15 real leave-one-out frames, 15 recovery-only frames,
20 target-specific oracle frames, 48 label-permutation frames, and five all-three frames. Only FIT labels/responses
enter gradients. VALIDATION is then used for health checks and provisional gates. If those provisional gates fail,
the run stops without TEST.

For a leave-one-out fit, the initialization/final health minibatch contains only the two fitted target identities;
the omitted target cannot enter its objective or health selection. For an all-three fit, only the three fitted
targets enter health and selection. `r.2.0.1` never enters a shared-projector gradient, health gate, medoid, or seed
choice. Its independent target-specific oracle does use its mask, but that oracle cannot affect any shared frame.
Therefore “unseen” for the fourth circuit means unseen by the shared fit, not unseen by every control fit.

Among all-three frames that pass the fixed VALIDATION health gates, choose the Grassmann medoid solely by geometry:
the frame minimizing the sum of projector Frobenius distances to the other eligible frames, with the lower seed as
the exact tie-break. VALIDATION performance determines eligibility only; it does not rank healthy frames. The
fourth-circuit mask is absent from both steps.

After every frame, scheduler fingerprint, and decision is written and hashed, execute one TEST evaluation sweep.
That sweep computes the final A/B/C statistics for all frozen objects. It also computes the already-fixed mean-
removal response of the selected final frame without changing or selecting anything. Prediction D is scored only if
A--C pass. No TEST result can cause another fit, checkpoint choice, threshold change, or alternate frame selection.

### Exact FIT label-permutation null

Represent the four quartet memberships of every FIT token position by one complete 4-bit vector. Also represent
membership in each circuit's parent slice by a second 4-bit vector. For null seed `p`, group FIT positions by

`(token_class, position_bin_32, native_CE_decile, four-bit parent-slice vector)`.

Within each group, apply one SHA-256-keyed permutation to the complete membership vectors. The same permutation moves
all four membership bits together, so the global 16-cell overlap lattice and every within-stratum pattern count are
preserved exactly. Including the parent-slice vector prevents a permuted circuit member from leaving that circuit's
legal parent population. Recompute exclusive masks and their matched controls from these permuted FIT masks using the
frozen control ladder. VALIDATION and TEST masks, labels, and controls are never permuted.

Each null must change the 4-bit code at at least 90% of originally nonzero FIT positions; otherwise that seed is
invalid and the whole null family is unavailable. Seeds are fixed as `52300..52315`. One null fit is trained for each
of the three leave-one-out folds under each seed, for the already priced 48 fits. Relabeling target names without
moving complete membership vectors is forbidden because the symmetric max-target objective would be unchanged.

### Matched projector-stability statistic

For each real optimizer seed, compute the three pairwise normalized projector overlaps among its three leave-one-out
frames and retain their minimum. This gives five real cross-fold stability values. For each label-permutation seed,
compute the same minimum across that seed's three leave-one-out null frames, giving 16 matched null values. Prediction
A's geometric clause holds only if at least four of five real values strictly exceed the higher-interpolation 95th
percentile of the 16 null values.

Cross-seed overlaps within one leave-one-out fold are reported descriptively but cannot support identification,
because the 48-fit price does not provide five optimizer restarts for every permuted problem. This correction keeps
the registered 48 null fits and the 41,200-event ceiling unchanged.

### Bounded selectivity and uncertainty

For member RMS `M` and matched-control RMS `C`, retain the concentration `M/C` and the fixed gates `M/C>=4` and
projector-minus-full-attention8 concentration `>=1`. Add the bounded quantities

`selectivity = (M-C)/(M+C+1e-12)` and `fourfold_margin = M-4C`.

For every required projector cell, resample matched member/control pairs at the row level for 2,000 deterministic
bootstraps. The higher-interpolation lower 95% bound of `fourfold_margin` must be strictly positive. This prevents a
large ratio caused by a nearly zero denominator from passing. Prediction B's real-vs-Haar/permuted joint statistic
is now the bounded

`minimum_heldout_selectivity * minimum_heldout_aligned_recovery`,

not unbounded concentration times recovery.

For the recovery-only comparison, pair frames by leave-one-out fold and optimizer seed. At least four of five seeds
must improve the minimum cell concentration by at least `0.5`, retain signed cosine within `0.05`, and have a
strictly positive row-bootstrap lower 95% bound on the bounded-selectivity improvement in every cell. In addition,
the observed mean of the five paired minimum-cell improvements must strictly exceed the higher-interpolation 95th
percentile of the exact 32-value seed-wise sign-flip null. A merely positive numerical difference cannot pass.

### Oracle liveness and generic-damage controls

Every target-specific oracle must be healthy and, in every VALIDATION/TEST donor/direction cell for its owner, have
member RMS at least `.02` nat and aligned recovery at least `.05`. Otherwise the target is instrument-invalid; “50%
of oracle” cannot become an easy gate through a dead or negative denominator.

The control-matching report separates exact-next-token tiers 0 and 1. If a target/split/donor/direction cell contains
at least 32 such matched pairs, the projector must also satisfy `M/C>=4` and a positive row-bootstrap lower bound on
`M-4C` within that subset. If fewer than 32 pairs exist, exact-token specificity is explicitly underpowered and even
an otherwise positive rung is called **within-census operational extraction**, not semantic identification.

Every final shared frame is also scored on:

1. every non-quartet circuit whose registered best native component is attention8;
2. all 32 fingerprint circuits; and
3. every token position outside the quartet union.

For each TEST donor/direction cell, the RMS projected effect over all outside-union positions must be at most 25% of
the smallest quartet-member RMS. Failure is generic attention8/CE damage and makes C false. These controls are in
addition to the pre-existing matched controls, not replacements for them. A/B alone are described only as selective
response fitting; ownership requires C and the second physical action in D.

### Exact C fingerprint statistic and null

For each of the four TEST cells (`D0/D1 x forward/reverse`), define circuit coordinate

`v[j] = RMS_member_j(projected delta CE) - RMS_matched_control_j(projected delta CE)`.

The quartet separation statistic is

`S = min_(j in quartet) v[j] - max_(j not in quartet) v[j]`.

Thus `S>0` means every quartet coordinate exceeds every non-quartet coordinate. No averaging occurs across donor
ensembles or directions; all four cells must pass. For each cell, create 20,000 CPU nulls by applying one common
permutation of the saved per-token projected responses within `(token_class, position_bin_32, native_CE_decile)`
groups, then recompute all 32 coordinates and `S`. A common response permutation preserves the circuit-mask overlap
lattice, matching strata, and circuit base rates. The observed `S` must be positive and strictly exceed the higher-
interpolation null 95th percentile in every cell. This is calibration of the historically proposed quartet's
operational extraction, not a fresh cluster-discovery p-value.

### Price clarification after corrections

The matched stability null still uses exactly 48 fits, so the maximum optimization price remains 20,600 forward and
20,600 backward events. All VALIDATION health objectives are one fixed balanced minibatch at initialization and the
final iterate, exactly matching the 206-forward ledger; they are not full-split evaluations. Moving every fit before
TEST and computing bounded/bootstrap/null statistics from saved per-token responses add no model calls. The maximum
pre-removal inference count therefore remains 9,422.

## Final implementation definitions — 2026-09-03 06:17 UTC

A second independent code audit found details that had not yet been specific enough to implement without choices.
No rung522 scientific model call, fit, VALIDATION outcome, or TEST outcome exists. These definitions close those
choices prospectively and supersede any less-specific wording above.

### Fixed health batch and exact no-op

Each fit gets a separate VALIDATION balanced-row scheduler with the same fitted target list and integer fit seed as
its FIT scheduler. The health call always uses that scheduler's batch zero, VALIDATION donor ensemble D0 map zero,
and the forward direction. It never uses the omitted target; a null fit is evaluated against the original,
unpermuted VALIDATION masks for its two named fitted targets. The validation callback ignores the optimizer's
sentinel step `-1` and always returns this one fixed batch. Both schedulers and their selected row IDs are written to
the pre-TEST manifest.

The exact self-donor check uses the six numerically lowest row IDs in each split (or every row if a split had fewer
than six). Native attention8 writes from that same execution are reinserted without arithmetic. Both the write
difference and logits must be bitwise equal, so the tolerance is exactly zero. FIT and VALIDATION checks occur
before fitting; the TEST check is part of the sealed TEST sweep.

For the target-specific oracle, the two member roles and two control roles use independent SHA-256 orderings keyed
by their distinct role names. They are not a fixed rotation of one another. Coinciding row choices are allowed,
because independent deterministic permutations can coincide; both copies still contribute separately to the one
concatenated target response.

### Paired row bootstrap

The ordered member/control arrays from the frozen matcher define token pairs. Bootstrap clusters are the member
recipient rows: sampling one member row carries every member token in that row and each token's aligned matched
control, even when that control lies in another row. Each required cell uses exactly 2,000 replicates. For replicate
`b` and draw `k`, the sampled cluster index is the first eight SHA-256 bytes of

`a8-r522-row-bootstrap-v1:<full cell ID>:<b>:<k>`

read as an unsigned little-endian integer, modulo the number of member-row clusters. Thus there is no library RNG or
unrecorded seed. RMS values are recomputed from all carried token pairs, with repeated clusters repeated in both
member and control sums. The lower bound is the higher-interpolation 5th percentile.

For the paired task-conditioned versus recovery-only comparison, use the difference in each frame's **minimum
held-out concentration** across the required split/donor/direction cells. Apply the `>=0.5` gate and the exact 32
sign choices separately inside each omitted-target fold over its five matched optimizer seeds. The sign-flip null is
the mean of the five signed differences; the observed unsigned mean must strictly exceed its higher-interpolation
95th percentile. Separately, every individual cell uses paired row clusters and must have a strictly positive lower
95% bound for the task-conditioned minus recovery-only bounded-selectivity difference. All three omitted-target
folds must pass.

### Prediction B aggregation

For one frame, define the joint statistic as

`minimum held-out bounded selectivity * minimum held-out aligned recovery`,

where each minimum ranges over the required VALIDATION and TEST donor/direction cells for that omitted target. For
each omitted-target fold, at least four of the five task-conditioned seeds must strictly exceed both (a) the maximum
of the 20 Haar values evaluated on that same omitted target and (b) the higher-interpolation 95th percentile of the
16 matched label-null values for that fold. Every one of the three folds must satisfy this rule. Negative recovery
is retained rather than clipped.

### Fingerprint masks and deterministic null

All 32 fingerprint coordinates use rung521's `fingerprint` member/control arrays, including for the quartet; they do
not substitute the quartet-exclusive arrays used by Prediction A. A coordinate is therefore comparable across all
32 circuits. For null replicate `b=0..19999`, positions inside each
`(token_class, position_bin_32, native_CE_decile)` group are ordered by global position and permuted by one
SHA-defined affine bijection. For a group of size `n>1`, hash

`a8-r522-fingerprint-null-v1:<TEST cell ID>:<b>:<group tuple>`.

The first eight bytes choose offset `c modulo n`; the next eight choose an initial multiplier in `1..n-1`, which is
incremented cyclically until it is coprime to `n`. Recipient rank `i` receives the saved response at donor rank
`(a*i+c) modulo n`. A size-one group is the identity. This makes every replicate a directly checked bijection with
constant-time address calculation rather than relying on a library shuffle or an underspecified SHA ordering.

One common permutation is applied before all 32 coordinates are recomputed. The masks, pair ordering, and circuit
base rates stay fixed. This is explicitly a coarse-stratum randomization: it does not preserve exact token identity,
so passing it supports the registered within-census operational extraction but cannot by itself establish a
semantic variable.

### Exact removal action and Prediction D

For the validation-selected all-three frame `Q`, `mu_Q` is the four-number mean of `yQ` over **all** native FIT
attention8 writes at all 568 rows and 256 predicted-token positions. It is computed and hashed before TEST opens.
No circuit mask and no amplitude fit enters this mean.

Removal is executed once over the TEST rows as

`y_removed = y - (yQ - mu_Q)Q^T`.

It has no donor or direction. Its saved per-token response is `CE_removed - CE_native`. Score TEST fold 8 and TEST
fold 9 separately, using the 32 rung521 fingerprint member/control pairs restricted to that fold. For each circuit
let

`u[j] = RMS_member_j(removal delta CE) - RMS_control_j(removal delta CE)`.

In each TEST fold, D requires: every quartet `u[j]` is positive; `min_quartet u > max_nonquartet u`; and every
quartet member RMS is at least twice the median non-quartet member RMS. It also requires the sign of
`mean_member(delta CE)-mean_control(delta CE)` to match the selected projector-swap sign in each corresponding TEST
fold, D0/D1 ensemble, and forward/reverse cell. Zero has no sign and fails.

Finally, apply the same 20,000 common coarse-stratum response permutations and SHA rule above, with cell IDs
`removal:fold8` and `removal:fold9`, to the removal response. In each fold the observed
`min_quartet u-max_nonquartet u` must be positive and strictly exceed the higher-interpolation null 95th percentile.
The negative set is exactly the other 28 circuits in the frozen 32-circuit battery. D uses no VALIDATION response,
no donor-specific removal, and no fitted scale. Its TEST forwards are charged only to the separate 2,000-forward
removal ceiling.

### TEST execution boundary and arm order

Native TEST capture, independent replay, and the TEST self-donor check are deferred until after the canonical
pre-TEST manifest has been atomically written and hashed. Before that point, no model call may contain a TEST row.
The manifest contains all 103 frame hashes and health states, FIT and VALIDATION scheduler hashes, null-design
hashes, the five all-three eligibility states, the geometry-only
medoid and lower-seed tie-break, all 20 Haar hashes, the call ledger, and the FIT-only `mu_Q`. Opening TEST is a
one-way state change; afterward fitting, optimizer updates, threshold changes, and selection are forbidden.
Every scalar VALIDATION output and response hash used by the provisional decision is first written to a separate
create-only JSON evidence file. The manifest stores that file's byte hash and canonical-content hash, and refuses
to open TEST unless the evidence file's embedded provisional decision and exact call ledger equal the independently
supplied manifest decision and ledger. This binds the decision to its complete inspectable inputs without copying
the same large nested evidence object into the manifest itself.

The physical evaluator labels every arm explicitly as `(ensemble, map, direction, recipient row)` before
concatenation and reconstructs saved outputs from those labels. It may not infer mathematical axes from tensor
reshape order. The registered output cells remain D0/D1 crossed with forward/reverse after averaging four maps.

## Label-null feasibility correction — 2026-09-03 06:28 UTC

The first execution of the exact FIT label-permutation constructor was CPU-only and occurred before any scientific
model call. It exposed an algebraic impossibility in the earlier 90% movement check. There are 1,442 FIT positions
with a nonzero quartet membership code. Under the already-frozen
`(token class, position bin, CE decile, parent-slice code)` strata and exact within-stratum code counts, at least 253
of those positions are forced to retain their code. For example, a stratum containing only three copies of code 8
cannot move any of them while preserving that stratum's code multiset. The exact maximum movement is therefore
1,189/1,442 = 0.8245492372. The 16 first SHA-order permutations moved only 857--893 positions
(0.594313--0.619279), and all correctly failed the impossible 90% check.

The movement rule is replaced prospectively by a constraint-relative rule with no fitted percentage: in every
stratum, solve the minimum-cost one-to-one assignment from donor codes to recipient positions, with primary cost 1
exactly when an originally nonzero recipient retains the same complete 4-bit code and 0 otherwise. SHA-256 values
keyed by null seed, stratum, recipient position, and donor position provide the secondary tie-breaking cost; the
primary penalty is larger than the greatest possible sum of all secondary costs, so tie breaking cannot sacrifice a
possible move. Every null seed must attain the computed global minimum number of unchanged nonzero positions—253
for the frozen FIT data—and hence the global maximum 1,189 moved positions. The observed movement and the
independently computed theoretical maximum are both written and hashed. Any seed that misses this maximum, changes
a within-stratum code count, moves a bit outside its parent slice, or is not a bijection invalidates the full null
family.

This correction strengthens the randomization relative to the failed 59--62% SHA-order versions while preserving
every original conditioning variable and overlap-lattice count. It does not use model activations, responses,
VALIDATION, TEST, or an outcome-chosen threshold.

## Final pre-science execution clarifications — 2026-09-03 06:59 UTC

An independent audit of the still-sealed scientific runner found four choices that must be explicit before any
rung522 scientific model call. The following definitions are prospective. They use no fitted frame, model response,
VALIDATION value, or TEST value.

### What must pass on VALIDATION before TEST can open

The provisional VALIDATION decision applies the already-registered A and B rules to the four VALIDATION cells only
(`D0/D1` crossed with forward/reverse). It requires all of the following:

1. for each omitted-target fold, at least four of five real seeds pass every A response, selectivity, paired-row
   bootstrap, whole-attention8 comparison, and powered exact-token condition in all four cells;
2. every target-specific oracle fit is healthy and live for its owner in all four cells;
3. at least four of five matched real-seed three-fold projector-overlap values strictly exceed the 16-null higher
   95th percentile;
4. in each fold, at least four of five task-conditioned seeds pass every paired recovery-only comparison, and the
   five-seed mean concentration improvement strictly exceeds its exact sign-flip 95th percentile; and
5. in each fold, at least four of five task-conditioned joint statistics strictly beat both the 20-Haar maximum and
   the 16-label-null higher 95th percentile.

All three folds must pass. This provisional decision only controls whether the already-frozen TEST sweep may occur;
the scientific A/B claims are recomputed over both VALIDATION and TEST afterward. A failed provisional decision is
written as a terminal pre-TEST result and cannot be repaired by changing a seed, threshold, fit, or control.

The oracle denominator is paired by optimizer seed: real seed `s` is compared with the target-specific oracle for
the omitted target at the same seed `s`. No best-oracle or across-seed choice is allowed. Every one of the 48 label-
null fits must also pass its optimizer health checks, or its null comparison is unavailable and TEST stays closed.
All 20 target-specific oracle fits, including the five reserved-`r.2.0.1` oracles, must be healthy and must satisfy
the `.02`-nat member-RMS and `.05` aligned-recovery liveness bars in all four VALIDATION cells for their own target.

The all-three medoid eligibility set is not caller-chosen. An all-three seed is eligible exactly when its optimizer
health checks pass and, for each of the three fitted targets in all four VALIDATION cells, it passes the A-style
signed-cosine, residual, positive and half-same-seed-oracle recovery, member-RMS, concentration, improvement over the
whole-attention8 comparator, row-bootstrap, and powered exact-token gates. This predicate uses no `r.2.0.1` mask or
fingerprint. Full-split VALIDATION performance therefore determines eligibility but never ranks eligible frames;
the geometry-only medoid and lower-seed tie-break remain the only ranking rule. No eligible all-three frame means a
terminal pre-TEST failure.

### Independent replay and complete arm reconstruction

The native capture uses the observed-model dispatch path, while the independent replay uses the literal
embedding/block/output loop from rung521. Both paths must execute attention8 exactly once and produce bitwise-equal
logits. Repeating the dispatch path twice is not an independent replay. Every saved evaluation tensor is initialized
to nonfinite sentinels; after explicit `(ensemble, map, direction, recipient row)` assignment, every element must be
finite before map averaging. This makes a missing or duplicated physical arm fail closed.

### Fold-specific removal pairs

For `removal:fold8` and `removal:fold9`, a frozen TEST matched pair belongs to the fold of its **member recipient
row**. Its aligned matched control is carried with it even if that control row lies in the other TEST fold. This is
the same member-row clustering convention used by the paired bootstrap and avoids rematching after TEST opens.
For either fold statistic, the common response permutation covers the complete TEST population and uses groups
ordered as `(document fold ID, token class, position bin, native CE decile)`. The SHA payload is
`a8-r522-fingerprint-null-v1:<cell ID>:<replicate>:<fold>:<class>:<position bin>:<CE decile>`. Thus a member or control response is
permuted only within its own document fold, including when the two endpoints of one frozen pair lie in different
folds. The separate cell IDs `removal:fold8` and `removal:fold9` still give independent deterministic null maps; each
statistic then reads only the pairs anchored by member rows in its named fold. The reported pair counts, cross-fold
pair counts, and pair-array hashes make this restriction auditable.

### Equivalent computation of the 20,000 response-permutation nulls

The common affine permutation remains exactly the one already specified. An implementation may compute donor
addresses only for response positions actually queried by the 32 frozen member/control arrays, rather than
materializing all 55,296 permuted TEST values on every replicate. For a queried recipient, the group ordering,
multiplier, offset, and donor-rank formula are identical to the full permutation, so every queried value and every
null statistic are bit-for-bit the same. The receipt hashes the complete 20,000-value statistic vector, the
algorithm/namespace definition, and the complete donor maps for the first and last replicate. It need not construct
or hash the unused entries of the other 19,998 full maps. A small CPU equivalence test against the literal full-map
implementation is required before science.
