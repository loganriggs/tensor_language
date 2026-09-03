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
