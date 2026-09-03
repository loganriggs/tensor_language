# Independent pre-outcome review of the R585 replacement package

**Review date:** 2026-09-03 UTC  
**Reviewed commit:** `96b420e98facf50bb560d8860f84ea91d208219e`  
**Verdict:** **APPROVED FOR IMPLEMENTATION ON THESE EXACT BYTES; NOT YET APPROVED FOR MODEL EXECUTION**

## Scope and outcome boundary

This is an outcome-blind, CPU-only review of the committed R585 replacement amendment, model-free manifest, manifest dry run, dependency lock, and R578 row authority. I did not inspect any future or live uncommitted R585 runner or test, open an R585 outcome, load a model, use a GPU, or touch the queue, registry, or agent board.

The replacement repairs the mathematical blockers in the first R585 preregistration. The causal operation, cells, scale units, bootstrap, dependency gate, price, terminal precedence, and licensed claim are now sufficiently determined for an implementation to be written. This approval does not license execution merely because a runner exists. The runner must pin these exact bytes, the dependency-lock hash, an exact canonical equality-term implementation and hook convention, and must pass the adversarial tests named below before a model call.

## Exact reviewed authorities

- replacement amendment: `98ed34711ada83bbe1591887edf17164efd443d4c6a47559f43dec33f60aa5bf`;
- model-free manifest: `7addbb8c07cbf29b985f5713e28d949c11a8da44e01c85c2044cbe764c04c962`;
- original manifest tests: `0439219a0281a57ec31e88b250acb669eafe225a6fc2fc0bbcaee74bced6050e`;
- manifest dry run: `dc81109bed0ef44c51224988a53d57143751a3f078a889c156a7a8862e52114f`;
- dependency lock: `908826844336fe7a073ae16a5ef9123434514c21a73f8d3b331b4bab6e9f49b7`;
- R578 rows: `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`;
- locked R586 result: `14e7414bc7cf6b4a6a221079ac378752602b021b8b411124149dcc2c311666b8`;
- locked R586 receipt: `afd7533b1838b7d230858696a059f9c3a5903e75f031aa0c86f175f4bc0d9384`; and
- locked R587 audit: `72f0261fe32aa3d048c442ea1c08af932af6a368894610833e79aaaabf98bfe9`.

The committed R586 result and receipt both say `held_capability_screen`, evaluated exactly FIT and SELECT, and opened no forbidden split. The committed R587 artifact says `held_independent_audit` and binds the same R586 result and receipt hashes. The dependency lock reproduces those exact paths, hashes, verdicts, and split closure. With independently supplied observed hashes, the manifest validator returns `dependency_held`; a changed digest raises, while a non-held upstream verdict returns `not_executed_upstream_dependency` rather than an R585 scientific null.

## Re-derived intervention

For endpoint (x), site (h), query (q_x), and semantic role (r\in\{A,C\}), the fixed factor is

$$
e_h^x(r)=p_h^x(q_x,k_x(r))\mathbf 1[t^x_{k_x(r)-1}=t^x_{q_x}],
\qquad
u_h^x(r)=W_h^O v_h^x(k_x(r)).
$$

For recipient (x) and donor (y), the four complete role sums are

$$
\begin{aligned}
t_h^{xx}&=\sum_{r\in\{A,C\}}e_h^x(r)u_h^x(r),\\
t_h^{yx}&=\sum_r e_h^y(r)u_h^x(r),\\
t_h^{xy}&=\sum_r e_h^x(r)u_h^y(r),\\
t_h^{yy}&=\sum_r e_h^y(r)u_h^y(r).
\end{aligned}
$$

These are replay, score-only, payload-only, and joint. At each of L5H5, L7H3, L8H3, and L8H4, the hook receives `inserted - live equality term`. The inserted hybrid is frozen from native endpoint factors; the removed term is recomputed from the live state after earlier-layer changes. L8H3 and L8H4 share one pre-modification layer-8 state. Roles are summed within each site, and sites are inserted separately; neither an omitted role nor a product of cross-site score/value sums is licensed.

The exact R578 authority has no semantic-token ambiguity: both A/C source tokens and positions are distinct, payload positions immediately succeed their registered sources, all query/source/payload IDs agree with the saved token sequence, and equality support is exactly one on coherent endpoints and zero on the broken match endpoint. Six physical pair-order/position patterns occur, so token search or a fixed absolute position would be wrong even though the frozen rows themselves are clean.

The structural identities follow algebraically:

- selector swap: native payload values are causally before the changed query, so payload equals replay and joint equals score;
- lag control: added filler is after the source/payload pairs, so the same two identities hold;
- coherent-to-broken match: donor equality support is zero, so joint equals score; and
- broken-to-coherent match: recipient equality support is zero, so payload equals replay.

This gives exactly 32 full-vocabulary identities per split. No other arm is declared a numerical no-op.

## Re-derived cells, mappings, and statistics

The included authority is exactly:

| Split | Rows | Directed pairs | Unique endpoints | Groups per directed cell |
|---|---:|---:|---:|---:|
| FIT | 1,872 | 3,744 | 1,728 | 72 |
| SELECT | 936 | 1,872 | 864 | 36 |

`contrast_target_source_edit` is absent. Per split, the manifest has 20 target cells and 32 control cells before intervention arms. Crossing with score, payload, and joint gives 60 target arm cells and 96 control arm cells. Eight lag/payload cells are structural no-ops, leaving 88 activity-eligible control arm cells and 24 `(arm, direction, recipient_condition)` coverage keys per split.

The answer-changing recipient mapping is exact:

- selector variants map `s0p0→s1p0` and `s0p1→s1p1`;
- payload variants map `s0p0→s0p1` and `s1p0→s1p1`;
- joint diagonals map `s0p0→s1p1` and `s1p0→s0p1`; and
- reversing a stored direction makes the donor endpoint the recipient without synthesizing a second reversed row.

For answer-changing targets, the natural coordinate is donor-answer minus recipient-answer logit. For match break, the common-answer coordinate is correct minus other-payload logit, signed negative for coherent-to-broken and positive for broken-to-coherent. These choices make donor-directed natural movement positive in both directions. The CE coordinate is signed the same way for match rows. The repaired whole-cell recoveries are

$$
R_{\rm mean}=\frac{\operatorname{mean}_g n_g}{\operatorname{mean}_g d_g},
\qquad
R_{\rm median}=\frac{\operatorname{median}_g n_g}{\operatorname{median}_g d_g},
$$

not a mean or median of rowwise ratios. A nonpositive mean, lower 95% mean bound, or median denominator is deterministically `native_denominator_or_scale_null`; the planted zero-denominator case confirms this path.

Selector score/joint, payload payload/joint, and match score/joint use the frozen recovery, positive-effect, direction-frequency, and signed-CE gates. Payload-swap score and coherent-to-broken match payload are opposing arms. Joint-diagonal rows use the two single-factor harms and the four-way interaction rather than a nonexistent natural-answer denominator. This closes the first preregistration's match-break and undefined-median defects.

## Scale units and active controls

The scale source is unique by `(arm, recipient_condition)` and is always FIT-frozen:

- score controls use the selector-score target;
- payload controls use the payload-swap-payload target; and
- joint controls use the selector-joint target.

Physical control direction does not select a different target. The 64 control cells across both splits times three arms produce exactly 192 control-to-target mappings. Each mapping has three distinct physical scale coordinates—inserted residual norm, target-margin logit movement, and full-vocabulary logit RMS—so a fully expanded runtime lookup has 576 unit-preserving entries, 288 per split. The adversarial test rejects mapping an insertion-norm control threshold to a logit-margin target scale.

Activity is measured from the median across the four per-site norms of the actual `inserted - live_removed` deltas, never from the inserted term alone. A family is adequately active only if at least 75% of its groups exceed 10% of the matching FIT insertion scale. Each coverage key needs two distinct active families. All groups in an active family cell—not only groups above the activity threshold—enter the margin, vocabulary-RMS, CE, and answer-preservation checks.

## Bootstrap and price

An independent enumeration reproduces exactly 124 bootstrap cells per split, 248 total, and dry-run hash `16e07d0400c37a4270adad09d1e28e04dc079138b61e7bb903d7abebb2ab027e`. It uses the exact canonical family ID, variant, recipient condition, direction, arm, and metric. The independent 2,000-replicate reconstruction of the 72-group sentinel gives big-endian uint16 draw hash `a9faec7440b0cb64f09bbb79de32d9e4b07bfd22b3a271830f4c2b6b4fde885f`. The namespace, lexicographic group order, first-eight-byte big-endian SHA interpretation, float64 statistics, and lower/higher quantile methods are unambiguous.

At batch size 32, the conservative price is:

$$
\mathrm{FIT}=\lceil1728/32\rceil+3\lceil3744/32\rceil+\lceil1728/32\rceil
=54+351+54=459,
$$

$$
\mathrm{SELECT}=\lceil864/32\rceil+3\lceil1872/32\rceil+\lceil864/32\rceil
=27+177+27=231.
$$

Thus SELECT-closed execution costs at most 459 and full FIT+SELECT costs at most 690, with zero backwards, fitted vectors, or weight updates. A contrast call would breach this registered census and price.

## Terminal decision and licensed claim

FIT is decided completely before SELECT opens, and all failed clauses remain saved. Within each split, dependency/integrity, instrument, denominator/scale, target capacity, factorization, active coverage, broad-write, and held outcomes have a unique precedence. SELECT failures receive the corresponding `select_` label only after every FIT clause passed.

Only `held_operational_selector_payload_factorization` licenses the narrow statement that the complete fixed four-site equality-gated terms causally carry an operational selector/value factorization on R578 FIT/SELECT under this role-aligned intervention. It does not establish unique Q/K features, OOD generalization, a weight-level compiler, individual-site necessity, selective removal, or the project's complete reusable-circuit goal.

## Adversarial validation

The separate test file is
`basis_aligned/bilinear_quotient/ops/test_induction_selector_payload_frozen_factor_rung585_replacement_adversarial.py`.
Together with the original manifest tests it passes 27 tests. The tests cover exact package hashes, semantic position/token consistency, planted ambiguous canonical tokens, metadata drift, recipient/donor conditions, match signs, zero denominators, target/control/coverage/identity counts, omitted A/C roles, omitted sites, distinct scale units, scale collisions, all bootstrap IDs and the 2,000-draw sentinel, dependency hash/verdict tampering, exact price, FIT-first precedence, and the licensed conclusion. Regenerating the dry run to a temporary path is byte-content equivalent after JSON parsing to the committed artifact.

## Requirements before model execution

These are implementation obligations already implied by the frozen amendment, not permission to amend it after seeing data:

1. Hard-code the exact dependency-lock SHA and independently open, hash, and parse the three fixed production dependency paths. The manifest validator intentionally trusts a caller-supplied observed-hash map and is not sufficient by itself.
2. Pin the exact canonical equality-term source hash, continuous-score scaling/sign, projected-value matrix orientation, and hook location in the runner, tests, dry run, and receipt.
3. Materialize and hash the full endpoint × four-site × two-role operation census. FIT has 13,824 such native factor keys and SELECT has 6,912. Reject omitted roles/sites rather than allowing a smaller set to run.
4. Materialize scale kind in the runtime control lookup, or enforce an equivalent typed structure, so residual norms can never collide with either logit scale.
5. Validate semantic positions and token IDs against the exact R578 rows before batching. Never infer roles by searching for a token value or assuming an absolute position.
6. Resolve the amendment's minor evidence-timing wording conservatively: an early instrument failure may stop below the ceiling, but the invalid artifact must explicitly state which phases and rows were not executed. It cannot claim a complete scientific null.
7. Keep integrity failures distinct from complete scientific nulls and preserve every failed clause even when terminal precedence selects one label.

Subject to those checks, implementation may proceed on the exact reviewed bytes. Execution remains blocked until the resulting runner, tests, dry run, dependency binding, canonical-term hash, and artifact schema receive their own pre-outcome review.

## Reusable two-agent workflow lessons

### Ranked builder failure modes to preempt

1. **Wrong semantic coordinate:** searching for token values or using a fixed position instead of the registered A/C role positions. This can silently choose two matches or the wrong payload after pair-order/lag changes.
2. **Wrong intervention time:** recomputing supposedly frozen donor factors after an earlier intervention, removing a frozen rather than live recipient term, or updating L8H3 before caching the state needed by L8H4.
3. **Incomplete operation:** omitting one role or one of the four fixed sites while still reporting an all-four result, or multiplying sums across sites and creating nonexistent cross-site products.
4. **Circular exactness:** declaring replay valid only because the same buggy tensor was subtracted and added. Canonical-term, non-equality remainder, native comparator, padding, and structural identities must be independently checked.
5. **Unit collision:** using insertion-vector norm to normalize logit margin or vocabulary RMS, or allowing a direction/outcome-dependent target scale.
6. **Bad causal orientation:** applying the answer-changing formula to answer-preserving match rows, reversing the match sign, accepting a zero denominator, or summarizing rowwise ratios.
7. **Broad damage mistaken for transfer:** moving a margin by destroying the recipient answer, counting inactive controls as selective, or filtering control outcome statistics to only the active rows.
8. **Membership/bootstrap drift:** duplicating reverse directions, pooling variants or directions, dropping a group, using display family labels in cell IDs, or changing quantile/hash conventions.
9. **Dependency or split leakage:** trusting lock strings without hashing/parsing the fixed files, opening SELECT before FIT holds, touching FINAL_TEST/OOD, or treating an upstream failure as an R585 null.
10. **Overclaiming:** converting an all-four, oracle-supported behavior result into a unique-feature, individual-site, reusable, removable, or OOD circuit claim.

### Cross-circuit invariant tests

The following tests generalize beyond R585 and should become a shared builder/critic contract:

- exact authority hash plus exact row, group, split, direction, arm, and site membership;
- strict finite JSON and hash-bound binary artifacts with dtype, shape, byte size, and row-order hashes;
- semantic-coordinate reconstruction from metadata, including planted duplicate-token and shifted-position cases;
- an explicit Cartesian operation census over all frozen components and semantic roles, with omission tests;
- independently reconstructed endpoint identities, intervention deltas, RMS sufficient statistics, and algebraic no-op/composition identities;
- typed normalization scales whose physical units must match at lookup time;
- deterministic group-bootstrap IDs, draw/statistic hashes, quantile methods, and a known sentinel;
- complete planted held, complete scientific-null, invalid-instrument, inactive-control, broad-damage, malformed-membership, and dependency-failure paths;
- exact conditional call accounting, zero updates/backwards where registered, and forbidden-split closure; and
- deterministic terminal precedence with all subordinate failed clauses retained.

### Counterfactual quality

The counterfactuals are causally meaningful for the narrow R578 task. Selector swaps change the selected source while holding payload assignment fixed; payload swaps change payload assignment while holding the selected source fixed; joint diagonals change both factors while preserving the correct answer; match breaks toggle equality support while preserving payload; and neutral, filler, and lag edits test invariance. Both physical directions are retained separately.

They also represent multiple valid counterfactual realizations rather than one prompt pair: each directed cell has 72 disjoint FIT groups or 36 disjoint SELECT groups, with token, pair-order, filler, and length variation. The four control families supply multiple ways to preserve the intended computation while altering context. This is enough to test an operational factorization on the task, but it remains oracle-equality-supported and in-distribution. It does not establish that these are all valid natural counterfactuals or that a learned feature would extrapolate OOD.

### Shared computation versus shared difficulty or broad damage

Evidence for the same computation is the conjunction of donor-directed numerator and CE movement, opposing single-factor predictions, joint-diagonal factorial interaction, exact selector/lag/match identities, reuse of one FIT-frozen typed scale on group-disjoint SELECT, and preservation on adequately active unrelated controls. Shared task difficulty alone can make several interventions fail or succeed together, but it does not predict the crossed selector-versus-payload arm pattern or the exact no-op identities. Broad damage can move an answer margin, but it does not predict donor-answer CE improvement, the answer-preserving joint interaction, preserved target margins on active controls, or small control full-vocabulary RMS.

The present all-four intervention cannot establish that the four sites individually share, duplicate, or divide the computation. That question requires a separately frozen site-subset or interchange-equivalence experiment with the same active controls; it cannot be inferred from a held all-four arm.

### Proposed wave-2 critic prompt amendment

> Review only the exact committed R585 runner/test/dry-run bytes and the wave-1 amendment, manifest, dependency lock, review, and adversarial tests; do not inspect outcomes or live uncommitted files. Independently reconstruct the full endpoint × site × role operation manifest and typed scale lookup before importing the runner. Require planted ambiguous-token, omitted-role, omitted-site, same-layer-order, live-versus-frozen, zero-denominator, scale-unit collision, inactive-control, broad-damage, duplicate-direction, missing-group, bootstrap-sentinel, dependency-byte, FIT-first, and price/split-closure failures. Verify the runner opens and parses the fixed dependency files itself rather than trusting caller-supplied verdicts/hashes, and verify independent canonical/remainder/native-comparator checks prevent self-cancelling replay. Report separately whether the instrument is executable, whether its counterfactuals are causally valid, and what the strongest licensed circuit claim would be if held.
