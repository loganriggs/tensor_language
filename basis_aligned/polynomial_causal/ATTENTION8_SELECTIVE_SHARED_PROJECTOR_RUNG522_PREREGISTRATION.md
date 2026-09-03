# Rung 522 preregistration: selective shared projector inside attention8

**Frozen:** 2026-09-03 05:46 UTC, after rung521 Stage A and before any rung522 implementation, optimizer, smoke,
or model outcome.

## Why this is a new experiment

Rung521 asked whether the complete attention8 output was already selective enough to be a trustworthy fitting
target. Its registered composite prediction A failed and therefore closed every learned stage in that rung. We do
not reopen it.

The failure was specific. Across all 24 target/half/donor/direction cells, the complete attention8 swap changed
exclusive circuit members only `1.2621489`--`1.7898961` times as much as matched non-members, below the frozen `3.0`
bar. However, every bootstrap lower bound on member-minus-control effect was positive (`0.0426963`--`0.2613011`
nat), both donor ensembles transferred in every registered cell, and the 32-circuit fingerprints reproduced across
document halves with Pearson correlations `0.9581688`--`0.9727829`, above their permutation controls. Exact replay,
self-donor no-op, and edit-liveness checks also passed.

Thus the whole module is a stable but broad causal component. Rung522 asks a different question: **can one learned
subspace inside that broad response isolate a computation shared by multiple circuits?** Selectivity is now the
outcome being tested, not a prerequisite that the native module boundary must already satisfy.

The TEST split and all VALIDATION/TEST outcomes remain unread by rung521, which used FIT only. To minimize post-result
freedom, rung522 retains rung521's previously frozen rank, projector action, circuit identities, data splits, donor
maps, response metrics, and numerical thresholds wherever they apply.

## Circuit claim and non-claim

The proposed shared circuit is a rank-4 projector in the 1,152-dimensional post-output-projection write of
attention8. It is trained from two of three circuit identities and must predict the third:

- fitted/leave-one-out set: `r.2.0.2`, `r.2.1.1`, `r.2.2.1`;
- reserved historical reuse target: `r.2.0.1`;
- negatives: every other circuit whose best native component is attention8, plus all other available circuits in
  the 32-circuit scoring battery.

This is not a compression or rank-reduction claim. Rank 4 is fixed capacity shared by all real and control fits. A
positive claim requires held-out circuit-identity prediction, fresh-document and fresh-donor transfer, selective
physical intervention, and stability across restarts. A low reconstruction loss or a small basis alone is a null.

No private projector is part of rung522. Private fitting remains closed unless a later preregistration first shows
that the response left after removing the shared prediction is reproducible across documents and donors.

## Exact intervention and gauge

Let `y in R^1152` be attention8's native output write for a recipient token and `d` the write at its registered
natural donor. For an orthonormal frame `Q in R^(1152 x 4)`, define `P=Q Q^T` and physically swap only that subspace:

`I_P(y,d) = y + (d-y) P = y + ((d-y)Q)Q^T`.

The learned object is `P`, not the four columns of `Q`: replacing `Q` by `Q R` for any 4-by-4 orthogonal matrix
does not change the intervention. Stability is therefore measured by projector overlap `tr(P1 P2)/4` and principal
angles, never by column matching.

For every edited write, the real model suffix is executed. CE changes are not added across interventions.

## Frozen data, controls, and donors

Reuse rung521's hash-defined document split, exclusive quartet masks, matched controls, and coherent row-level donor
maps without alteration:

- FIT folds 0--5 train and choose no scientific threshold;
- VALIDATION folds 6--7 choose a representative among healthy seeds only;
- TEST folds 8--9 are opened once after the fit and all decisions are frozen;
- training uses FIT donor ensemble `D0` only;
- FIT `D1`, both VALIDATION/TEST donor ensembles, reverse swaps, and the reserved fourth circuit are unseen by
  gradients.

Controls remain in their own document half and parent/data cell. The exact relaxation levels and hashes in the
rung521 preflight are reused. No control or donor may be regenerated in response to rung522 outcomes.

## Fits

### Shared leave-one-target-out fits

For each of the three omitted targets, fit one rank-4 projector from the other two targets, with five fixed real
seeds `52200..52204`. Each update balances the two training targets and minimizes the exact maximum of their losses,
not their average.

For full-attention8 signed per-token CE response `f` and projected response `p`, the member fidelity term is

`L_member = mean((p-f)^2) / (mean(f^2)+eps)`.

The matched-control penalty is

`L_control = mean(p_control^2) / (mean(f_member^2)+eps)`.

The coefficient or constrained form joining these two terms, optimizer, learning rate, update count, batch balance,
initialization, checkpoint choice, and health rules must be frozen in a preflight addendum. A single small CPU toy
must show that the rule can recover a planted shared selective subspace while rejecting an equally powered broad
subspace before CUDA science. No rung522 model result may choose these settings.

### Matched controls

Run all of the following at rank 4 under the same health and evaluation rules:

1. **Recovery-only shared fit:** omit `L_control`. This tests whether causal-response approximation by itself is
   selective.
2. **Target-specific oracle:** one independent fit per target. This bounds what rank 4 can recover when sharing is
   not required; it is not evidence of reuse.
3. **Haar-random projectors:** at least 20 fixed random rank-4 projectors.
4. **Label permutations:** 16 complete retrainings with the target labels permuted within the frozen overlap lattice
   and matching strata. The higher-interpolation 95th percentile is the null threshold.

## Health and liveness gates

Before reading any model metric, every required fit must satisfy:

- finite losses and gradients;
- no model-parameter gradients;
- attention8 called exactly once per execution;
- `max|Q^T Q-I| <= 1e-5` after retraction;
- projector distance from initialization above `.02`;
- mean loss over the final 20 updates below the first 20;
- VALIDATION objective better than initialization; and
- a self-donor remains an exact activation and logit no-op.

Any unhealthy required real fit invalidates the rung. Failed null fits are reported and make their corresponding
comparison unavailable rather than silently shrinking the null.

## Prediction A: a shared selective response generalizes to an omitted circuit identity

For each leave-one-target-out fold, at least four of five seeds must pass every following condition on the omitted
target in both document halves, both unseen donor ensembles, and both swap directions:

1. signed response cosine to the full-attention8 member response is at least `.75`;
2. optimally scaled relative residual is at most `.55`;
3. aligned recovery is positive and at least 50% of that target's rank-4 oracle recovery;
4. projected member RMS is at least `.02` nat;
5. matched-control RMS is at most 25% of exclusive-member RMS, equivalently member/control concentration is at
   least `4.0`; and
6. concentration strictly exceeds the complete-attention8 baseline in the same cell by at least `1.0` absolute.

Item 5 is deliberately stricter than rung521's failed `3.0` gate because the old 25%-control rule already implies
`4.0`; retaining the stronger pre-outcome rule avoids a post-result relaxation. Item 6 prevents a pass created by an
easier evaluation cell rather than learned selectivity.

Prediction A additionally requires the five-seed projector overlaps to strictly exceed the retrained label-
permutation q95 in all three omitted-target folds. If response metrics pass but projector stability does not, report
an operationally predictive but unidentified family, not a stable shared unit.

## Prediction B: selectivity comes from the task-conditioned objective

For every omitted-target fold, the real shared projector must strictly outperform the recovery-only shared control
on the minimum held-out concentration across halves/donors/directions, without losing more than `.05` signed cosine.
It must also strictly beat both the Haar-random maximum and the label-permutation q95 on a frozen joint statistic:

`min_heldout_concentration * min_heldout_aligned_recovery`.

This distinguishes a task-conditioned selective computation from generic low-dimensional approximation or optimizer
luck.

## Prediction C: the shared projector reuses on a fourth circuit and is not generic attention8 damage

After A/B, fit the same rank-4 objective to all three FIT targets with five fixed seeds. Select the Grassmann medoid
using VALIDATION only, then open TEST once. Without using `r.2.0.1` in fitting or selection, the projector must pass
every response and selectivity threshold from A on that reserved circuit in both TEST donor ensembles and directions.

On the 32-circuit fingerprint, the four quartet circuits must separate from all non-quartet circuits above the
higher-interpolation 95th percentile of 20,000 overlap-lattice-preserving max-statistic label permutations. All
circuits are printed. If several non-quartet circuits pass equally, C fails under the quartet name; the result may be
reported only as a broader attention8 variable.

## Prediction D: a second physical action preserves ownership

With all choices frozen, apply mean-centered projection removal

`R_P(y) = y - (yQ - mu_Q)Q^T`,

where `mu_Q` is the FIT-only mean projected attention8 write. On TEST, removal must preserve the quartet-vs-negative
ordering and the sign of the fitted shared intervention's member-minus-control effect in both document halves. It
must change each quartet target at least twice as much as the median non-quartet circuit and beat the same max-
statistic permutation q95. This is a selective manipulation test; no amplitude may be fitted after TEST opens.

## Registered outcomes

- **Health failure:** invalid optimization instrument; no model inference.
- **A false:** no held-out shared selective rank-4 projector in this optimizer family. Good response recovery with
  concentration below threshold is only a broad low-rank approximation.
- **A true, B false:** an unstable or generic low-dimensional coordinate, not task-conditioned identification.
- **A/B true, C false:** a three-target object without demonstrated fourth-circuit reuse, or a broader attention8
  variable if negatives also respond.
- **A--C true, D false:** a donor-swap direction with no demonstrated removal ownership, not a manipulable circuit.
- **A--D true:** a stable, reusable, selectively manipulable attention8 subspace for this quartet on the registered
  data. This still does not provide a semantic algorithm, shifted-corpus OOD result, joint composition with other
  circuits, or a compressed deployed model.

Increasing rank after failure is not an interpretability successor. A new rank or private stage would require a new
mechanistic hypothesis and preregistration.

## Literal price ceiling

The preflight addendum must derive a tighter exact call formula. Hard ceilings before the final removal stage are
45,000 forward/backward optimization calls and 12,000 inference-only calls; removal may add at most 2,000 inference
calls. The largest retained learned artifact is one `1152 x 4` frame, or 4,608 floating values, plus a 4-number FIT
mean and metadata. The full training search may transiently hold five frames per fold/control. Model weights remain
frozen and are not duplicated in a deployed artifact.

## Frozen dependencies

- rung521 Stage-A result:
  `6a303e0e62ef3d2443ed6d667f74bc28c703a79ce5f462657bff212c1c5a676c`
- rung521 preflight:
  `42639d35ef6317104c6e0e684aeb00cb4c550df77d496733bcfe8be790fed650`
- rung521 Stage-A science executable:
  `3067dba2f3e0f4fefb4c718c86bbae7201efcaac16dd93ee46baa7c8265b0fcf`
- rung521 measurement/instrument module:
  `d5ca962c16cd8f454adac79916a9cf3272b91debac0d27ebba2ce77804fb9ebd`
- shared projector/control library:
  `edcf3d750e8fbdcb2ae479bcc6e68bd7ccc5078217b62cf981570656b6a773e4`
- `census_state_diverse.pt`:
  `c785f3d938091253535aa4f613ab2b4107bf297c8d615da4f7eab4f8282f5e0b`
- `curated_rows.pt`:
  `faaf89f38ddf1471234a1d30d978213367a566a9927bb3c73b274ab32afaa9dd`
- `circuits/BATTERY.json`:
  `86d7ac72eeb95f9ec80a3e92ef65e28c0df66a36b9291d2d1d2d01f7bb6c5030`
