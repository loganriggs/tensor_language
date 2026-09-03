# Rung 588 preregistration: independent CPU audit of the future R584 result

**Frozen:** 2026-09-03 UTC, after independent approval of the exact repaired instrument and before this auditor inspected, queried, or even checked the existence of the R584 result namespace

## Purpose and independence boundary

R588 decides whether a future R584 JSON result contains the complete frozen evidence and whether that evidence independently implies the reported selection and terminal decision. It does not load the model, run a prompt, fit a vector, alter a threshold, or open FINAL_TEST/OOD. It must not import or call the R584 runner's scoring, selection, result-validation, or decision functions.

The audit is pinned to repaired commit `55b138ed7d270fa6b103f06006091f761cf54af8`, the independently approved review with SHA-256 `9294bdf8df18a56cdae8705b69e0129bfe2d6376d642d4c9dc86386c0d898310`, the R582 rows and mathematical helper, and every code/document hash declared in the frozen R584 dry-run receipt. R582's outcome-blind deterministic group-bootstrap and null-map definitions may be independently reimplemented or hash-pinned as authority; no R584 outcome computation may be reused.

During construction and dry-run, the R584 result path must not be opened, read, statted, or existence-checked. The non-dry-run entry point is the only code allowed to read it, after the managed R584 job has terminally produced it. Because R584 has no separate scientific-result receipt, the auditor must stable-read the result bytes twice, record their SHA-256, validate the embedded frozen dry-run receipt/provenance, and refuse a changing or malformed byte stream. The R588 dry-run JSON is the pre-outcome audit-package receipt.

## Exact result paths

The only source outcome is the immutable future file
`basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung584_results.json`.
The only R588 outputs are:

- `basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung588_audit.json`; and
- `basis_aligned/bilinear_quotient/numbered_list_cached_value_downstream_use_rung588_dryrun.json`.

R588 never rewrites the source.

## Independent raw reconstruction

For every opened split and every saved arm, R588 must require exact equality to the R582 authority rows, not merely nonempty cells:

- FIT has 576 rows in 16 semantic groups;
- conditional SELECT has 288 rows in 8 semantic groups;
- each FIT null has exactly 384 registered recipients;
- each SELECT null has exactly 192 registered recipients; and
- no raw record may contain FINAL_TEST or OOD.

Every capture and intervention record must reproduce the frozen token IDs, group/split, representation, source level and value, source/query positions, source token, action/condition, and registered answer IDs. Real arms must have one exact site/component identity. Null arms must contain the exact deterministic recipient-to-donor map and its semantic constraints.

R588 must independently verify all saved sufficient-statistic identities:

1. ordinary endpoint margin equals answer logit minus the saved maximum other candidate logit;
2. ordinary endpoint CE equals log-sum-exp minus answer logit, and `answer_best` agrees with those two logits;
3. conflict preference equals arithmetic-candidate logit minus structural-candidate logit;
4. margin damage, CE increase, and conflict-sign preservation agree with native/intervened endpoints;
5. each full-vocabulary RMS equals the square root of its saved squared-difference sum divided by its vocabulary count; and
6. norms, squared sums, exactness errors, and elapsed time are finite with the required sign.

Capture rows must contain per-row source-present and source-deleted attention-replay errors; per-site MLP `C`, `Q`, and `C+Q` norms; per-site finite-response reconstruction errors; the R576 term norm; and source-deletion full-vocabulary evidence. Their maxima must equal the opened split's saved exactness summary. Exactness passes only when the checkpoint hash is pinned, every opened error is at most `1e-10`, and every R576 term norm is positive.

## Independent scientific reconstruction

For each real candidate, R588 independently rebuilds all frozen R582 statistics with exactly 2,000 deterministic semantic-group bootstrap replicates:

- successor positive-damage fraction and lower mean margin/CE bounds;
- copy answer-best fraction, mean CE increase, intervention activity, and margin/RMS fractions using FIT scales on SELECT;
- source-matched successor-minus-absolute-copy action gaps and lower bounds;
- relation-break and conflict-row activity;
- conflict-sign preservation;
- source-level sign agreement;
- surface-gap recovery; and
- broken-relation characterization.

Undefined ratios must be JSON `null`, fail their gate where gated, and carry the exact frozen reason. Every independently rebuilt report must match the saved report recursively, with zero relative tolerance and absolute numeric tolerance `1e-12`.

For each active null, the auditor independently rebuilds its action-gap bounds and checks activity in each representation/source/surface cell as median null norm divided by median real norm in `[0.8,1.25]`. The real lower bounds must be reused byte-for-value from the already reconstructed real report. The frozen null gate is, separately for each source and surface,

$$
\min_{r\in\{\mathrm{list},\mathrm{digit},\mathrm{word}\}}L_{\mathrm{real},r}
>
\max_{r\in\{\mathrm{list},\mathrm{digit},\mathrm{word}\}}L_{\mathrm{null},r}.
$$

R588 must also reconstruct every saved two-removal interaction at the selected site:

$$
I = Y_{C+Q\ \mathrm{removed}}-Y_{C\ \mathrm{removed}}
    -Y_{Q\ \mathrm{removed}}+Y_{\mathrm{native}},
$$

along with the separate `C` and `Q` removal effects, on every non-conflict row in both opened splits.

## Selection, split opening, price, and terminal decision

The candidate order is fixed as `C`, `Q`, then `C+Q` at MLPs 8, 10, 12, and 14. R588 independently derives the only legal execution path:

1. If exact FIT capture fails or no candidate passes its non-null FIT gates, there is no provisional or selected candidate, only FIT is open, null/SELECT evidence is absent, and the price is exactly 379 forwards.
2. If the first passing FIT candidate exists but either FIT active null fails, it remains the provisional candidate but there is no selected candidate, only FIT is open, and the price is exactly 419 forwards.
3. If both FIT nulls pass, the provisional candidate becomes the selected candidate, SELECT opens, exactly the three components at that site plus two SELECT nulls are present, and the price is exactly 510 forwards.

All paths require zero backwards and no weight updates. `pred_a`, `pred_b`, `pred_c`, their conjunction, `downstream_use_component_held` versus `downstream_use_decomposition_null`, and the scalar `next_step` must exactly equal the independently reconstructed state. A scientific null is valid when complete; malformed evidence is an audit failure rather than a null.

## Envelope, provenance, and decision

The source must be strict finite JSON with the exact frozen top-level fields and path-dependent container types. Its embedded `execution_plan` must equal the pinned R584 dry-run receipt, including its zero-call plan semantics, 419+91 ceiling, selection order, closed splits, and 19 hashes. The result must add the exact dry-run hash and verified checkpoint hash, bind the runner/tests/reviewed dependencies and four null maps, report the frozen runner/test/contract hashes, and contain no unexpected provenance key.

The R588 verdict is `held_independent_audit` only if every byte, authority, schema, membership, endpoint identity, reconstruction, bootstrap, null, exactness, interaction, selection, split, price, terminal, and provenance check holds. Otherwise it is `failed_independent_audit` with named failures and, when possible, the independently recomputed R584 scientific decision.

## Pre-outcome validation and price

Before source access, model-free planted fixtures must cover:

- a complete held path that opens SELECT and costs 510 forwards;
- a complete scientific-null path with FIT only and 379 forwards;
- malformed missing-arm/missing-row evidence;
- a non-finite nested number;
- an invalid null donor;
- a changed conditional forward count;
- a changed interaction identity; and
- a stale provenance or dry-run receipt.

The real audit uses all 2,000 bootstrap replicates. Focused fixtures may use fewer replicates only to keep the CPU-only dry run small. R588 is `# BQLANE: cpu`; it performs zero model forwards, zero backwards, zero updates, and no GPU work.
