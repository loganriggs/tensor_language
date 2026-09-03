# R584 pre-outcome implementation review

**Reviewed:** 2026-09-03 UTC, before any R584 model run.

## Verdict: BLOCK the model run pending four repairs and one clarified null rule

The scientific object is sound and the central implementation is mostly faithful to R582: it deletes the exact R576
cached-value term, forms separately normalized source-present/source-deleted MLP inputs, computes the exact finite
background-cross and contrast-self terms, subtracts one frozen response vector from the native MLP write, recomputes
the suffix, evaluates FIT before SELECT, and never constructs a FINAL_TEST/OOD model batch.

The current runner should nevertheless not be run. Four adversarial contract failures would make a landed null or
pass incompletely auditable, and R582's null-comparison sentence still has an outcome-relevant unresolved reading.
The owner should repair these before updating the implementation hash or enqueueing R584.

## Frozen authorities inspected

- R582 preregistration:
  `e7832dc77cabe7a1afba61c759188a0aca73802163cef1abe013ffaff5c987b3`
- R582 rows:
  `84c6a78882a33c266b3875285f63ceaed746dac7810fce16b591f7b57763cf3b`
- R582 row receipt:
  `1511cfd7fcfe729edf4427f9f88f8552c32230e013d01a0661767713fdc29148`
- R582 helper/builder:
  `b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c`
- R584 implementation note:
  `bb20396590836bddc3ddad972f498a37e0f5454de4826aaa5adc196c591c3e5a`
- reviewed R584 runner:
  `41f854ece413d2c8575aae5c4a320dd7fa12781acc75e09895ccd3bbd6b81774`
- owner's R584 tests:
  `41cdcf54e649246dae69d2af716030726b7ee5688ef6fba0632108408d37e3cb`
- reviewed R584 dry run:
  `d79ea4ae360e5ad02caf21a9710f4e867d14115ba78b4c221fcc9c7048569c9f`

These are review-time hashes, not approval of a future changed runner.

## What passes review

### Exact R576 term deletion

The deletion in `trajectory` computes the cached layer-0 value directly from embeddings and the layer-0 value weights,
multiplies it by attention-8's cached-value coefficient, reads the native layer-8 score at the frozen final query and
source position for H7 and H3, projects each head through its own output block, sums them, and subtracts only that
1,152-vector at the final query after attention 8. The source and query positions come from the frozen R582 row, and
the row validator checks that the source ID occurs at that position and that the query is final.

The runner carries the same independent checks as R576 for cached-bus equality, projected-term equality, complete
head source summation, and own/cached value addition. It uses float32, so the `1e-10` relative-squared algebra bars are
plausible rather than BF16 cancellation traps.

### Normalized-state C/Q algebra

For each of MLP8/10/12/14, the native and term-deleted trajectories recompute independently to the MLP input. The code
captures the model's separately RMS-normalized inputs (x^+) and (x^-), sets δ to (x^+-x^-), and computes

\[
C=D[(L\delta)(Rx^-)+(Lx^-)(R\delta)],\qquad
Q=D[(L\delta)(R\delta)].
\]

The direct comparison is (D[(Lx^+)(Rx^+)-(Lx^-)(Rx^-)]), so MLP bias correctly cancels. The two ordered cross terms
remain together. Existing tests cover direct reconstruction and L/R swap. The new adversarial test also confirms
reciprocal per-product scaling invariance at the frozen relative-squared tolerance.

The intervention subtracts the frozen C, Q, or C+Q vector from the native MLP write at the final query and recomputes
all later blocks. This is the R582 causal intervention; it does not replace the whole MLP state.

### Counterfactual rows and active null maps

The row authority has exactly 1,440 unique prompts in 40 semantic groups:

- FIT: 16 groups / 576 rows;
- SELECT: 8 groups / 288 rows;
- FINAL_TEST: 8 groups / 288 rows;
- OOD: 8 groups / 288 rows.

Each group contains all 36 representation × source × condition cells. The successor/copy contrast keeps the final
source token fixed. Split words, source ranges, and group identities are disjoint.

The null-map census is exact:

- FIT has 384 eligible successor/copy/surface rows; both null maps contain exactly those 384 keys.
- SELECT has 192 eligible rows; both maps contain exactly those 192 keys.
- Every different-group donor changes group while preserving representation, source level, and exact condition.
- Every other-action donor preserves group, representation, source level, source token, and surface status while
  changing copy versus successor.

The actual vector removed in a null is the donor response vector, so its saved vector norm is the intervention norm.
The 0.8--1.25 norm gate is live rather than a label-only control.

### FIT/SELECT closure and price

All twelve candidates are evaluated on FIT in the correct order. Exactness failure prevents selection. Only the first
non-null-eligible candidate receives the two nulls; null failure is terminal under the prospective R584 implementation
note. SELECT opens only after both FIT nulls pass. SELECT evaluates the selected site's three companion components and
the two nulls, but only the originally selected component can pass. No code path evaluates FINAL_TEST or OOD tokens.
Reading outcome-blind metadata for row validation is not opening either split.

The price calculation is correct at batch size 24:

- 27 FIT batches and 20 FIT null-eligible batches: 419 forwards maximum;
- 14 SELECT batches and 10 SELECT null-eligible batches: 91 conditional forwards;
- literal maximum 510, below R582's conservative ceiling of 530;
- zero backwards and no weight updates.

The metadata fields `selection_order`, `opened_splits`, and `forbidden_splits_opened` are lists; selected/provisional
component and `next_step` are scalar string-or-null fields. This avoids R581's list-versus-string error. The scientific
result should rename the nested dry-run `execution` object to `execution_plan`, however: as written it would contain
`model_forwards: 0` and `opened_splits: []` beside different top-level observed values.

## Required repair 1: fail closed on exact raw membership

`score_candidate` infers its census from the rows it receives. Deleting an entire FIT semantic group reduces all cells
from 16 to 15 groups and still returns a passing report. The adversarial fixture does exactly this and the current code
returns `passed_without_nulls=True`.

Before scoring, require equality to the authority-derived row-ID set, group-ID set, and per-cell counts:

- real FIT: all 576 row IDs, 16 groups, one row per group in every representation/source/condition cell;
- real SELECT: all 288 row IDs, 8 groups, same complete cell structure;
- each null: exactly the 384 FIT or 192 SELECT eligible recipient IDs, with exactly one registered donor ID per row.

Do not merely require nonempty cells. A missing row/group is an integrity error, not a smaller bootstrap sample or a
scientific null.

## Required repair 2: every normal scientific null must serialize finite JSON

When an ordinary-prompt mean action gap is nonpositive, `score_candidate` emits `-inf` for surface recovery. Zero target
scales and zero row norms can likewise emit `inf`. Python's permissive `json.dumps` writes `Infinity`, which is not
standard JSON and can break an independent schema/audit precisely on the expected null path.

Represent undefined ratios as `null` plus `passed: false` and a named reason such as
`nonpositive_ordinary_action_gap`. Apply this to surface ratios, margin/RMS fractions, null norm ratios, and every other
denominator. Write both dry run and result with `allow_nan=False`; a nonfinite scalar then becomes an integrity failure
before artifact publication.

## Required repair 3: exactness must cover every row/site, not the first batch

The finite bilinear response is computed for every row/site, but only each row's maximum across sites is saved. More
importantly, the manual native trajectory is compared to `native_logits` only for `batch_index == 0`. R582 requires
every prompt/site/arm exactness to be auditable; a replay error in any later length batch can currently pass.

Recommended zero-forward-price repair:

1. At attention 8, compute the native attention write alongside the analytic replay inside each existing trajectory
   call and save a per-row full-valid-sequence or final-query relative-squared error. This adds local attention compute,
   not another whole-model forward.
2. Save `native_replay_relative_squared_error_by_row` and
   `bilinear_response_relative_squared_error_by_site` under every capture row.
3. Gate their maxima at `1e-10`. Retain the independent end-to-end native comparison as an additional smoke check, but
   do not let its first-batch-only coverage stand in for all-row replay evidence.

Running a separate native whole-model pass for all batches would instead raise the maximum to 549 and violate R582's
530 ceiling, so that alternative requires a prospective price correction.

## Required repair 4: save the literal R582 audit envelope and code provenance

Current intervention records omit token IDs, query/source positions, and literal source value. Capture records omit the
source-deletion full-vocabulary squared-difference sum/count and per-site response errors. The result also does not save
the R584 implementation or test hash. An audit could join some fields back from the pinned row artifact, but R582
explicitly requires them in the row-level sufficient statistics.

At minimum, each capture/intervention record must add:

- `token_ids`, `query_position`, `source_position`, `source_value`, answer/candidate IDs;
- per-site C/Q/joint norms and per-site reconstruction errors;
- native-versus-source-deleted registered logits, log-sum-exp, squared full-vocabulary difference sum, and vocabulary
  count;
- native-versus-intervened values already present;
- null donor row ID for null arms;
- explicit component/site/arm identity on each record.

The result envelope must add runtime SHA-256 values for the exact runner, owner test, adversarial test (if adopted), dry
run, R582/R584 documents, helper, rows, receipt, checkpoint, and null-map specification. A self-hash belongs in a
separate atomic receipt if one is required; do not fake a result self-hash inside the result itself.

## Required prospective clarification: the null comparison statistic

R582 says: “The real candidate's minimum action-gap lower bound across representations must exceed each null's
corresponding bound.” Current `score_null` instead compares real and null bounds separately in each of twelve
representation/source/surface cells. It also recomputes the same real lower bound under a different bootstrap cell ID
for each null rather than reusing the candidate's already frozen bound. The implementation note does not resolve this
wording, and the choice can change the selected/null verdict.

Before model output, freeze one exact formula. The most literal conservative reading is, for each source/surface cell
and null,

\[
\min_{r\in\{list,digit,word\}} L_{real,r}
>
\max_{r\in\{list,digit,word\}} L_{null,r},
\]

using the real lower bounds already computed under the candidate's fixed bootstrap IDs. If the intended rule was
cellwise (L_{real,r}>L_{null,r}), say so prospectively and reuse the original real bound rather than redrawing it once
per null.

Likewise freeze whether the null activity number is
`median(null_norm) / median(real_norm)` (the literal wording) or the current
`median(null_norm / real_norm)`. Either can be reasonable, but they are not equal and cannot be chosen after outcomes.

## Adversarial tests

New review-only tests are in
`basis_aligned/bilinear_quotient/ops/r584_preoutcome_adversarial_tests.py`. The filename deliberately does not match
pytest's default `test_*.py` collection pattern, so it documents the blocked contract without silently breaking the
owner's existing suite. Run it explicitly during repair.

Current outcomes:

- expected pass: reciprocal product-rescaling gauge invariance;
- expected pass: scalar/list/hash-container metadata types;
- failure: missing complete semantic group is accepted;
- failure: negative action-gap null emits six `-inf` values;
- failure: intervention row lacks token/position/source fields;
- failure: per-row/site exactness, source-deletion RMS, and implementation/test hashes are absent.

## Commands run

```text
python -m py_compile basis_aligned/bilinear_quotient/ops/numbered_list_cached_value_downstream_use_rung584.py
```

Passed.

```text
pytest -q basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung582.py
```

`14 passed in 0.51s`.

```text
pytest -q basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung584.py
```

`9 passed in 12.42s`.

```text
pytest -q basis_aligned/bilinear_quotient/ops/r584_preoutcome_adversarial_tests.py
```

At the reviewed implementation: `4 failed, 2 passed`. These are pre-outcome contract failures, not scientific results.

## Unblocking criterion

R584 is ready for parent review only when the four failing adversarial tests pass, all original R582/R584 tests still
pass, `json.dumps(..., allow_nan=False)` succeeds on both planted held and planted null fixtures, the null statistic is
prospectively fixed, the updated literal price is recomputed, and new hashes are published before any model call.

No rank sweep, threshold change, or later candidate may rescue a failed outcome. If the repaired instrument runs and no
candidate clears the action/copy/null gates, the required scientific result remains
`downstream_use_decomposition_null`.
