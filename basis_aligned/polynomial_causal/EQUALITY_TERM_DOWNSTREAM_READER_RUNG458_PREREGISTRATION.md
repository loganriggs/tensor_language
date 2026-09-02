# Rung 458: downstream-reader-defined equality-term grouping

Status: prospective design, frozen before response capture or model execution. This is a screen plus held-out natural-
text identification test, not OOD confirmation or adoption. The repository-code subset outcomes remain closed.

## Question

Rung 457 found robust overlap between the early equality block `{L5H5,L7H3}` and the layer-8 block
`{L8H3,L8H4}`, but did not identify L8H3/L8H4 as one group. Determine whether a specific later attention or MLP write
treats one cross-depth pair of equality terms as the same task-conditioned variable.

This is below head-term grain. Native heads are sources of exact intervention terms, not assumed semantic units.
Response cosine is only the fitting screen. A group is identified only if it predicts held-out component responses,
causal patch repair, and safer within-group than between-group interchange.

## Rows and split

Use only the exact rung-457 `final_natural` row file
`5f2813eacc3ec66162c2ce695b978264137c66126fdc25e3d49b4efd44a9d759` and its already-frozen masks. Documents
`0:96` are the response-fitting half; documents `96:192` are the validation half. Rung 457 already exposed CE subset
effects on both halves, so this is not a fresh task-effect final. The new component responses, pair choice, reader
choice, patch outcomes, and interchange outcomes are unopened.

No `ood_code`, attention0 confirmation, SEALED consequence family, or other row role may be loaded.

## Screen arms and captured objects

Use seven exact configurations:

- native;
- remove each singleton equality term L5H5, L7H3, L8H3, or L8H4;
- remove the early block L5H5+L7H3; and
- remove the layer-8 block L8H3+L8H4.

For every attention and MLP write in layers 9 through 17, capture on the fitting half

`Delta_j(h,x) = write_j(remove h,x) - write_j(native,x)`.

Layer 9 is the first common suffix after every source term. The fitting artifact retains only task-conditioned response
Grams, norms, and document sufficient statistics, not token rows or full hidden-state caches.

For component `j`, condition `c`, and terms `h,h'`, define normalized response similarity

`cos_j,c(h,h') = sum_x <Delta_j(h,x),Delta_j(h',x)> / sqrt(sum_x||Delta_j(h,x)||^2 sum_x||Delta_j(h',x)||^2)`.

The sum is over exact tokens in condition `c`. Also report response RMS relative to the native component-write RMS.

## Deterministic pair and reader choice

Candidate pairs are the four cross-depth pairs

`{L5H5,L8H3}`, `{L5H5,L8H4}`, `{L7H3,L8H3}`, `{L7H3,L8H4}`.

For each pair and each of the 18 common-suffix components, compute on `all_positive`:

`task_margin = cos_all_positive - max(cos_matched_negative, cos_off_target)`.

A fitting candidate is live only if both term-response RMS values are at least `1e-4` of native write RMS, positive
cosine is at least `.70`, and task margin is at least `.15`. Choose the candidate with largest task margin, then
largest positive cosine, then lexical component/pair identity. If none qualifies, stop before patch/interchange and
record the registered null.

This rule searches over 72 pair-reader candidates on fitting documents only. The validation thresholds below are not
changed after selection.

## Held-out response prediction

On documents `96:192`, recompute only the frozen pair and reader response summaries. Require:

- positive cosine at least `.60`;
- task margin at least `.10`;
- both response RMS values live under the same `1e-4` relative rule; and
- the sign of near-minus-far and one-minus-multiple response cosine matches the fitting half whenever the fitting
  difference has magnitude at least `.05`.

This establishes response transfer but not causal use.

## Patch test

For each term in the frozen pair, run its singleton removal on validation documents. At the frozen reader, replace
only that component's changed write with its cached native write from the same document and token, leaving the reader
input and every other write as in the removal run.

For all-positive CE, define

`patch_recovery = [CE(remove h) - CE(patch native reader write)] / [CE(remove h) - CE(native)]`.

Require positive removal stakes in point estimates and every bootstrap draw. A reader is a causal mediator only if
patch recovery is at least `.15` for both terms with simultaneous 95% lower bound above zero. Report matched-negative
and off-target patch effects; require absolute off-target CE change from the unpatched removal no larger than `.01
nat`. This patch can identify mediation but does not prove the reader is the first or only consumer.

## Interchange test

For frozen pair terms `a,b`, cache the frozen reader writes under native, remove-a, and remove-b on each validation
input. In remove-a, replace the reader write by the remove-b reader write; symmetrically interchange b into a. These
are within-proposed-group swaps.

Between-group controls use the two remaining cross-depth terms, each swapped into the corresponding removal arm with
the same procedure. Score the absolute change in registered all-positive recovered effect caused by every swap.

Use the already-shipped commutation statistic:

`separation = mean absolute between-group change / mean absolute within-group change`.

Require separation at least `2.0`, exact label-permutation `p <= .05`, and mean within-group change no more than 25%
of the native-to-removal stake. Matched-negative and off-target swap effects are reported and may not be used to choose
the group.

## Registered predictions

### A. Instrument

All source/row/mask/arm hashes, model identity, replay, site dispatch, component capture counts, split boundary, and
no-new-role clauses hold. Relative squared native replay is at most `1e-12`. Every patch changes only the frozen
reader write. Any failure invalidates the result.

### B. Fitting screen finds a task-conditioned cross-depth pair

At least one fitting candidate meets the `.70` positive cosine, `.15` task-margin, and response-liveness bars.

### C. Response grouping transfers

The frozen pair/reader meets every held-out response-prediction bar without refitting or reselection.

### D. The reader causally mediates both terms

Both frozen terms meet the held-out patch-recovery and off-target bars.

### E. The proposed group passes interchange

Within-group swaps are materially safer than frozen between-group controls under the separation, permutation, and
stake-relative bars.

The strong null is instrument failure, no live fitting candidate, either validation positive cosine below `.30`, both
patch recoveries at most `.05`, or interchange separation at most `1.2`.

## Claim boundary and price

B alone is a response-geometry screen. B+C is stable response similarity. B+C+D identifies a shared causal reader
but not interchangeability. Only B+C+D+E proposes a natural-text operational group eligible for a separately frozen
code OOD test.

No result is a compressed program. Literal deployed saving is zero. Report exact outer-forward count, wall time, peak
GPU memory, retained sufficient-statistic bytes, and the number of searched pair-reader candidates. Rank or response-
Gram dimension may be reported as a diagnostic, but cannot satisfy any prediction.

If the null wins, do not tune cosine thresholds or response rank. Split each equality term into continuous Q/K score
features versus value/output features and repeat the downstream-reader criterion at that finer algebraic grain.
