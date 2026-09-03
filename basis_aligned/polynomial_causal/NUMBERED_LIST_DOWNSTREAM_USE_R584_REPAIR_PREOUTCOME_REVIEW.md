# R584 repaired downstream-use experiment: independent pre-outcome review

**Reviewed:** 2026-09-03 UTC

**Target:** exact R584 repair committed as `55b138ed7d270fa6b103f06006091f761cf54af8`

## Verdict: APPROVED for managed model execution at these exact hashes

The repaired R584 instrument closes all four blockers and the unresolved null definition from the original pre-outcome review. The model-facing computation, scientific thresholds, candidate order, frozen rows, and FIT-before-SELECT policy remain those of R582. This approval is only permission to execute the exact reviewed program through the managed runner. It is not evidence that an MLP component exists, passes the gates, or should be promoted; a landed result still requires an independent CPU audit before any circuit claim.

No R584 result or result receipt existed during this review. No model was loaded, no GPU call was made, and FINAL_TEST/OOD were not opened.

## Exact reviewed authorities

- commit: `55b138ed7d270fa6b103f06006091f761cf54af8`
- R582 preregistration: `e7832dc77cabe7a1afba61c759188a0aca73802163cef1abe013ffaff5c987b3`
- R582 helper: `b0d99eeeef834091cf9ddfe77b58682f0e9a7e101e143a18570808dacb57bc1c`
- R582 rows: `84c6a78882a33c266b3875285f63ceaed746dac7810fce16b591f7b57763cf3b`
- R582 rows receipt: `1511cfd7fcfe729edf4427f9f88f8552c32230e013d01a0661767713fdc29148`
- R584 implementation note: `612005760bccda8f1a9f16b540b0734de3241e5da1c40246f514509733539181`
- original blocked review: `a2afe7185a07528a29435cce7cafe5c63758667ec9fc5ba02d9bd6fe1e37a25a`
- repaired R584 runner: `50609756d97de2f13f717774f13d72b1c743f38a172375e9b08efc2b055336c7`
- R584 owner tests: `37cc8f73ed128ebdb17b5cfcdb1248bc240291e9a10d38c526ac7d4a76ea3cce`
- R584 adversarial tests: `900883046b648c7c9aa0714fff3d7d0da678b70ab8598623321e4f9d32bb5cd2`
- generic result contract: `af8fb9557dcb77e038319b0fffa919927f3925497a0edafe27fc951125dfb272`
- regenerated R584 dry run: `b2ebe65c92ea5170ab13394c1ffee8562ff4241f481a6fce392a00b200149fe8`
- observed-model facade: `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`
- exact R576 deletion helper: `91db3a2a9210aef915ce2e4f0a62253274e0b5470cbfaa05a95d50a3c0cf985a`
- attention replay helper: `5723e42e2a5f72a4ddab7a20b631e18e0b6d28875ff53f3db2d37d1845d6e076`

The R584 files in the repair commit remain byte-identical in the current worktree. The regenerated dry-run JSON is also byte-identical to the committed dry run.

## Resolution of the original blockers

### 1. Complete row, group, and donor membership: closed

Every real arm is passed through `validate_real_raw` before scoring. The generic contract requires exact equality with every authority row in the opened split, including row IDs, group IDs, and split assignments. The R584-specific validator additionally binds representation, source level/value/token, condition, action, query/source positions, answer IDs, token IDs, site, component, and arm identity. Dropping an entire group now fails as an integrity error rather than reducing the bootstrap sample.

The authority recount is:

- FIT: 576 rows in 16 groups;
- SELECT: 288 rows in 8 groups;
- FINAL_TEST: 288 rows in 8 groups;
- OOD: 288 rows in 8 groups.

Each null is restricted to the exact successor/copy ordinary/surface recipients: 384 rows on FIT and 192 rows on SELECT. `validate_null_donor_map` recomputes the deterministic R582 map and requires exact key/value equality. Different-group donors preserve representation, source level, and condition while changing group. Other-action donors preserve group, representation, source level/value/token and swap copy with successor. Every saved null record must name the exact registered donor and null arm.

### 2. Finite scientific-null behavior: closed

Every outcome-dependent division uses `_safe_ratio`. A non-finite input, nonpositive denominator, or non-finite result becomes JSON `null`, sets the associated gate false, and records a reason such as `nonpositive_ordinary_action_gap`, `nonpositive_successor_margin_scale`, or `nonpositive_real_intervention_norm`. Planted negative-gap, zero-scale, and dead-null tests all serialize with `allow_nan=False`.

The candidate report and null report are checked with `validate_standard_json`; the complete result is checked again by the generic result contract and written with `allow_nan=False`. A normal scientific null therefore remains a complete finite artifact instead of crashing or emitting `Infinity`.

### 3. Per-row replay, decomposition, and source-deletion evidence: closed

Inside every source-present and source-deleted trajectory, layer-8's manual attention replay is compared with the native attention write. The per-row relative-squared errors for both trajectories and their maximum are saved for every capture row. This check runs in every length batch, not only the first batch.

For each row and each candidate MLP in `{8,10,12,14}`, the runner saves the separate `C`, `Q`, and `C+Q` norms and the relative-squared error between `C+Q` and the direct finite MLP-write difference. The global exactness decision takes the maximum over all rows and sites and requires it, the native replay errors, and the other R576 algebra checks to be at most `1e-10`. The first-batch end-to-end native call remains only an additional smoke test.

Capture and intervention records now include the complete token/semantic coordinates, registered candidate endpoint statistics, source-deleted endpoint statistics, squared full-vocabulary logit-difference sums, vocabulary counts, independently recomputable RMS values, site/component/arm identity, and null donor IDs. This is sufficient evidence for a later CPU audit without saving every vocabulary vector.

### 4. Result validation and provenance: closed

The dry run contains 19 exact provenance hashes: frozen rows/documents, the R576/R579 authorities, runner, owner and adversarial tests, generic result contract, model facade, R576/R573 computation helpers, and four deterministic null maps. The eventual result adds the exact dry-run hash and verified checkpoint hash.

Before publication, `validate_scientific_result` validates finite JSON, exact opened-split capture membership, scalar/string and container field types, the exact conditional model-call count, zero backwards, no weight updates, required provenance values, and absence of FINAL_TEST/OOD. The nested zero-call object is named `execution_plan`, so it cannot be mistaken for the observed top-level execution count.

### 5. Null inequality and activity definition: prospectively fixed

For each source level, surface condition, and null, the repaired code uses the conservative rule

$$
\min_{r\in\{\mathrm{list},\mathrm{digit},\mathrm{word}\}} L_{\mathrm{real},r}
>
\max_{r\in\{\mathrm{list},\mathrm{digit},\mathrm{word}\}} L_{\mathrm{null},r}.
$$

The real lower bounds are reused directly from the selected candidate report and are not redrawn under null-specific bootstrap IDs. Activity is separately checked in every representation/source/surface cell as

$$
\frac{\operatorname{median}\lVert v_{\mathrm{null}}\rVert}
     {\operatorname{median}\lVert v_{\mathrm{real}}\rVert}
\in [0.8,1.25].
$$

A zero real median becomes `null` with a reason and fails. The planted test distinguishes this conservative cross-representation inequality from the weaker cellwise interpretation.

## Conditional execution accounting

The independently recomputed legal paths are:

- no FIT candidate passes the non-null gates: 379 forwards;
- a provisional FIT candidate exists but one of its FIT nulls fails: 419 forwards;
- both FIT nulls pass and SELECT opens: 510 forwards.

The maximum decomposes as 419 FIT forwards plus 91 conditional SELECT forwards. It is below R582's conservative ceiling of 530. Every path has zero backwards and zero weight updates. The code recomputes the expected count from `provisional_fit_selection` and `selected_component` and requires exact equality before result publication.

## Verification performed

```text
pytest -q \
  basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung582.py \
  basis_aligned/bilinear_quotient/ops/test_numbered_list_cached_value_downstream_use_rung584.py \
  basis_aligned/bilinear_quotient/ops/r584_preoutcome_adversarial_tests.py \
  basis_aligned/bilinear_quotient/ops/test_result_contract.py
```

Result: `48 passed in 13.20s`.

```text
python basis_aligned/bilinear_quotient/ops/gate.py \
  basis_aligned/bilinear_quotient/ops/numbered_list_cached_value_downstream_use_rung584.py
```

Result: `no findings`; `GATE: PASS`.

```text
python basis_aligned/bilinear_quotient/ops/preflight.py \
  basis_aligned/bilinear_quotient/ops/numbered_list_cached_value_downstream_use_rung584.py
```

Result: `preflight: no findings`.

```text
CUDA_VISIBLE_DEVICES='' BQLIB_DRYRUN=1 python \
  basis_aligned/bilinear_quotient/ops/numbered_list_cached_value_downstream_use_rung584.py
```

The regenerated receipt reports 1,440 authority rows, the exact 419+91 price, 19 provenance hashes, zero model calls/backwards/updates, no opened split, and `FINAL_TEST_or_OOD_opened: false`. Strict finite JSON validation and equality to the in-memory current execution plan both pass.

## Approval boundary

This review found no remaining pre-execution blocker in the repaired contract. Approval is revoked by any change to the hashes above. A future result must preserve a null honestly if no candidate passes, and must not be interpreted until a separately frozen CPU auditor reconstructs its raw rows, summaries, split-opening decision, exactness gates, null comparisons, conditional price, and provenance.
