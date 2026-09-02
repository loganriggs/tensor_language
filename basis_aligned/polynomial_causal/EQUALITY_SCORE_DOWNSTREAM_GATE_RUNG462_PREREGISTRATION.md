# Rung 462: causal localization of the equality context gate

Status: prospective intervention design, frozen after rung 461 and before running any later-write patch. This is an
explanatory experiment on the already-open code role, not a new OOD confirmation or a compression test.

## Motivation

The frozen L5H5-score→L8H4-payload transplant preserves the native code effect across near/far and
one/multiple-predecessor contexts with four-cell Spearman 1.0. Yet MLP9 response size has the opposite ordering:
the equality service matters more in far and unique-predecessor contexts even though the local response is smaller.
Therefore raw response norm is not the gate.

Rung 462 asks whether one later attention or MLP **write** causally mediates the context-dependent effect. It uses
write patching, not low-rank response fitting: record a module's write on the reference trajectory, insert that exact
write into the base trajectory at the same module, and measure which task effects it restores.

## Frozen data, object, and trajectories

- role: the same 192-document `ood_code` tensor;
- discovery half: documents `0:96`; validation half: `96:192`;
- selected object: L5H5 equality score → L8H4 payload, read only as the already-frozen mechanistic parent;
- scale: rung 459's natural-fit score ratio, unchanged;
- base: remove the L5H5 and L8H4 equality terms;
- reference: remove L5H5 but retain native L8H4;
- hybrid: remove both and restore L5H5's scaled score times L8H4's payload;
- candidate later writes, in execution order:
  `MLP8`, then `attention9`, `MLP9`, ..., `attention17`, `MLP17` (19 candidates);
- cells: near positive, far positive, one-predecessor positive, multiple-predecessor positive, all positive, and
  off target, with masks and support frozen by rungs 460--461.

Native and empty analytical replay are instrument controls. Do not test other source/target pairs, score factors,
readers, scales, row roles, context definitions, QK branches, or SEALED attention0 outcomes.

## Exact patch intervention

For one batch, first run base, reference, and hybrid while caching every candidate write. A candidate `j` defines:

- `reference_patch(j)`: rerun base, but replace only write `j` by the cached reference write from the same document,
  positions, and batch;
- `hybrid_patch(j)`: rerun base, replacing only write `j` by the cached hybrid write;
- `permuted_patch(j)`: validation control that replaces only write `j` by the reference write rolled by one document
  within each four-document batch.

For an attention patch, replace its residual-stream write but retain the base trajectory's internal first-value
state. This isolates the public attention output rather than silently patching a second hidden channel. For an MLP
patch, replace its complete 1,152-dimensional write. No future activation is moved backward in depth.

Fit runs all 19 `reference_patch` candidates only on documents 0:96. It then freezes one candidate before any
validation patch outcome. Validation runs reference, hybrid, and permuted patches only for that frozen candidate.

## Metrics

For cell `C` and patch `P`,

`patch_effect(P,C) = [sum CE_base(C) - sum CE_P(C)] / token_count(C)`.

`patch_recovery(P,C) = patch_effect(P,C) / native_stake(C)`,

where `native_stake(C) = [sum CE_base(C)-sum CE_reference(C)]/token_count(C)` and is required positive for primary
cells. Report raw response RMS and response cosine for every candidate as companion measurements, but they cannot
select or pass a candidate.

Discovery candidates require all-positive patch recovery at least `.10`, positive far-minus-near and
one-minus-multiple patch effects, and absolute off-target patch effect at most `.01 nat`. Select the candidate with
largest all-positive recovery, breaking exact ties by the frozen execution order.

On validation, use 20,000 shared document-bootstrap draws for all-positive reference-patch recovery. Report fixed
48-document waves `96:144` and `144:192` for recovery and both context contrasts.

## Registered predictions

### A. Instrument

All parent/source/model/row/mask/scale/candidate-order hashes hold. Replay relative-squared error is at most `1e-12`,
factor reconstruction error at most `1e-10`, every reference patch uses the same-document cached write, the
permuted control is an exact one-document roll, call census is exact, and SEALED remains closed.

### B. A later write is selected on discovery

At least one of the 19 candidates satisfies the fixed recovery, two context-order, and off-target requirements. Its
identity and all discovery metrics are frozen before validation patch outcomes.

### C. The selected reference write causally mediates held-out effect

On validation, reference-patch all-positive recovery is at least `.10`, its simultaneous 95% bootstrap lower bound
is above `.02`, native stake is positive in every bootstrap draw, and recovery is positive in both fixed waves.
Absolute off-target patch effect is at most `.01 nat`.

### D. The selected write carries the context law

On validation, reference-patch effect is larger for far than near and for one predecessor than multiple, pooled and
in both fixed 48-document waves. These are signed causal effects, not response-norm comparisons.

### E. Alignment and transplant controls

The correct reference patch has at least `2.0` times the absolute all-positive effect of the document-permuted patch
and exceeds it by at least `.005 nat`. Across the four primary cells, hybrid-patch versus reference-patch effects
have Spearman at least `.80`; hybrid-patch all-positive effect is between `.50` and `1.50` times the reference-patch
effect. This asks whether the transplanted matcher drives the same downstream mediator.

The strong null is instrument failure, no discovery candidate, validation reference-patch recovery at most `.03`,
correct/permuted separation at most `1.10`, or hybrid/reference four-cell Spearman at most zero.

## Claim boundary and successor

A full pass identifies one later module write as a causally sufficient partial mediator of the equality service's
context-dependent effect. It does not prove that this is the only mediator, that its raw coordinates are semantic,
or that the full natural-plus-code matcher claim passed rung 460. It saves zero parameters.

If it passes, the next held-out test patches that mediator under targeted removal/interchange and checks unrelated
copy/equality controls before the QK branch split. If no candidate or validation fails, treat the gate as distributed
across the suffix or carried by the residual path rather than a single component write; next test a cumulative
suffix-boundary patch, not another rank sweep.
