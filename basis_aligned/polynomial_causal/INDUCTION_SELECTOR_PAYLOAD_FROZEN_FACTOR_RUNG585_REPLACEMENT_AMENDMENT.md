# Rung 585 replacement/amendment: frozen selector × payload factor intervention

**Frozen:** 2026-09-03 UTC, before any R585 model output was inspected  
**Status:** prospective replacement for the blocked first R585 preregistration

## Authority and scope

This document supersedes `INDUCTION_SELECTOR_PAYLOAD_FROZEN_FACTOR_RUNG585_PREREGISTRATION.md` for any future R585 implementation or execution. The blocked original remains immutable at SHA-256 `b927d1ae92a5ab749a888badcfaaa0f5e7301d79b7169be8dec18babccfbd116`; it is retained as provenance, not silently edited. The outcome-blind review identifying the blockers remains immutable at `b8b4bcae6d2a24781383a5595a7c78d2d58623df209e9b98f7037ecc10566b2c`.

This replacement asks the same narrow question: do the four fixed equality-gated attention terms at

\[
H=(\mathrm{L5H5},\mathrm{L7H3},\mathrm{L8H3},\mathrm{L8H4})
\]

causally carry an operational selector-by-payload factorization on the repaired R578 FIT/SELECT prompts? It does not search heads, sites, ranks, dimensions, subsets, thresholds, or normalizers. It does not claim unique Q/K features, OOD generalization, a weight-level compiler, individual-site necessity, or selective removal.

Frozen model-free authorities include:

- R578 rows: `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`;
- R580 preregistration/result/receipt: `8f80926d0a90360a66ebce605732d32ff3e283a3428eb7245f4813a521d12580`, `7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84`, and `6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a`;
- model-free R586 preregistration/implementation/test/dry-run: `a139948085a99a6e745d3e8bf5d08ae11b58480d30ddf5e75467b506dda3a9a5`, `ab33c1afea27d624151ad68ca230fb36ae03833e95349eb6da409778e9ea271b`, `748400e0675d37d9fd7fc7ce306ac7549b73db4b50bab2fe365abdb44b4d7841`, and `0134f0218c3ec135abaec30d2028abae6e1da2c4f0c30bc78cf57d4d4aac0d30`.

No R586 or R585 scientific outcome is an authority of this document.

## Hard upstream dependency closure

R585 must not load a model until a separate `induction_selector_payload_frozen_factor_rung585_dependency_lock.json` has been frozen and its SHA-256 has been hard-coded into the R585 implementation and tests. The lock is created only after upstream work completes and before any R585 model call. It must contain:

```json
{
  "schema": "induction_selector_payload_frozen_factor_rung585_dependency_lock_v1",
  "r578_rows_sha256": "8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6",
  "replacement_amendment_sha256": "<hash of this document>",
  "r586_result_path": "<fixed path>",
  "r586_result_sha256": "<exact hash>",
  "r586_receipt_path": "<fixed path>",
  "r586_receipt_sha256": "<exact hash>",
  "r586_verdict": "held_capability_screen",
  "r587_audit_path": "<fixed path>",
  "r587_audit_sha256": "<exact hash>",
  "r587_audit_verdict": "held_independent_audit",
  "evaluated_splits": ["FIT", "SELECT"],
  "forbidden_splits_opened": []
}
```

The runner verifies every file hash and parsed field. Filename existence is insufficient. A non-held or malformed upstream dependency means R585 is not run; it is not an R585 scientific null. The final implementation, tests, dry-run, dependency-lock, this amendment, canonical term implementation, checkpoint, result, and receipt hashes must all be saved in the R585 receipt.

## Exact canonical term and four arms

For endpoint \(x\), site \(h\), final query position \(q_x\), and semantic target-source role \(r\in\{A,C\}\), let \(k_x(r)\) be the payload position immediately after source role \(r\). Define

\[
E_x(q_x,k_x(r))=\mathbf 1[t^x_{k_x(r)-1}=t^x_{q_x}],
\]

\[
e_h^x(r)=p_h^x(q_x,k_x(r))E_x(q_x,k_x(r)),\qquad
u_h^x(r)=W_h^O v_h^x(k_x(r)).
\]

The future implementation must pin the existing canonical equality-term library by exact hash and use its exact definition of the continuous bilinear score \(p\), including all model scaling/sign conventions. It must also pin the exact head-output hook location and matrix orientation. No softmax, rescaling, or raw-Q/K reinterpretation may be substituted. R578 guarantees that coherent endpoints have exactly one registered equality-successor edge and broken match endpoints have zero.

For directed recipient \(x\) and donor \(y\), cache every native \(e_h^x,e_h^y,u_h^x,u_h^y\) before any intervention. Define the frozen role-aligned insertions

\[
\begin{aligned}
t_h^{xx}&=\sum_{r\in\{A,C\}}e_h^x(r)u_h^x(r),\\
t_h^{yx}&=\sum_r e_h^y(r)u_h^x(r),\\
t_h^{xy}&=\sum_r e_h^x(r)u_h^y(r),\\
t_h^{yy}&=\sum_r e_h^y(r)u_h^y(r).
\end{aligned}
\]

These are respectively replay, score-only, payload-only, and joint. Role labels, not absolute positions, align donor and recipient factors. At each site in increasing layer/head order, compute the canonical recipient equality term \(t_h^{\rm live}\) from the current live state and apply

\[
\Delta_h=t_h^{\rm inserted}-t_h^{\rm live}.
\]

Later live removal is allowed to reflect earlier interventions; every inserted hybrid remains frozen native. L8H3 and L8H4 are modified in one same-layer transaction from the same pre-modification layer-8 state. The model receives \(\sum_h\Delta_h\) through the appropriate sequential hooks; scores and values are never summed separately and multiplied.

### Independent canonical replay checks

Before a scientific decision:

1. The factor library's \(\sum_r e_h^x(r)u_h^x(r)\) must match its independently computed canonical equality term at every endpoint/site within absolute tolerance `1e-5` in float32.
2. Canonical equality term plus canonical non-equality remainder must reconstruct the native per-head final-query output within `1e-5`.
3. Replay final logits must match a separate native comparator for every endpoint and length class within `1e-5` elementwise over the full vocabulary.
4. Saved `inserted - live_removed` must match the observed hook delta at every executed row/direction/arm/site within `1e-5` elementwise.
5. Padded and unpadded native final logits must match within `1e-5` for every observed length class; padding is strictly after each saved final query and cannot be a key.
6. Every saved scalar/vector is finite.

Any failure is `invalid_instrument`; a self-cancelling replay alone is not evidence that the named term was reconstructed.

## Frozen rows: contrast is dropped

R585 uses exactly these R578 families on FIT and, conditionally, SELECT:

1. `two_valid_sources_selector_swap`;
2. `payload_swap_match_preserved`;
3. `selector_payload_joint_answer_preserved`;
4. `match_break_payload_preserved`;
5. `irrelevant_source_edit`;
6. `irrelevant_payload_edit`;
7. `copy_relation_preserved_nuisance_change:filler_change`; and
8. `copy_relation_preserved_nuisance_change:lag_extension`.

`contrast_target_source_edit` is **not evaluated or reported by R585**. Its R580 native diagnostic remains available separately. This choice freezes the conservative R585 ceiling at 690 forwards.

Each stored row is evaluated in exactly its two declared directions, `base_to_donor` and `donor_to_base`; no reversed duplicate row is synthesized. FIT contains 1,872 rows, 3,744 directed pairs, and 1,728 unique endpoints. SELECT contains 936 rows, 1,872 directed pairs, and 864 endpoints. Each directed cell contains one row per semantic group: 72 on FIT or 36 on SELECT.

## Exact directed-cell manifest

`recipient_condition` is the recipient's R578 factorial condition. For answer-preserving and match/control rows it is the condition prefix of `family_variant`; for answer-changing rows it is obtained from the recipient endpoint's registered R578 condition. Only the combinations below exist.

| Family | Variant | base→donor recipient | donor→base recipient | Arms scored |
|---|---|---|---|---|
| selector swap | `payload_assignment_0` | `s0p0` | `s1p0` | score, payload, joint |
| selector swap | `payload_assignment_1` | `s0p1` | `s1p1` | score, payload, joint |
| payload swap | `selector_0` | `s0p0` | `s0p1` | score, payload, joint |
| payload swap | `selector_1` | `s1p0` | `s1p1` | score, payload, joint |
| joint diagonal | `payload_B` | `s0p0` | `s1p1` | score, payload, joint |
| joint diagonal | `payload_D` | `s1p0` | `s0p1` | score, payload, joint |
| match break | each of `s0p0,s0p1,s1p0,s1p1` | coherent recipient | broken recipient | score, payload, joint |
| neutral source | each condition | same condition | same condition | score, payload, joint |
| neutral payload | each condition | same condition | same condition | score, payload, joint |
| filler | `<condition>:filler_change` | same condition | same condition | score, payload, joint |
| lag | `<condition>:lag_extension` | same condition | same condition | score, payload, joint |

This expands per split to 4 selector cells, 4 payload cells, 4 joint-diagonal cells, 8 match cells, and 8 cells for each of four control families before crossing with three intervention arms. Thus there are 20 directed target cells (60 arm cells), 32 directed control cells (96 arm cells), and 24 control-coverage keys `(arm, direction, recipient_condition)` per split.

### Structural identity manifest

Replay is always an instrument baseline, never a target or selectivity success. The following and only the following additional arm identities are structural:

| Family/direction | Exact identity | Scientific use |
|---|---|---|
| selector swap, both directions | payload = replay; joint = score | exactness; payload is opposing no-op |
| lag, both directions | payload = replay; joint = score | exactness; payload excluded from activity/selectivity |
| match coherent→broken | joint = score because donor equality support is zero | exactness; payload remains an opposing arm |
| match broken→coherent | payload = replay because recipient equality support is zero | exactness; payload excluded from activity/selectivity |

Every identity is checked in full-vocabulary final logits within `1e-5`. No other arm is declared a no-op. The manifest yields eight structural lag-payload control cells and 88 activity-eligible control arm cells per split.

## Directed target statistics

All row/group quantities use replay as intervention baseline; the separate native comparator is only an instrument check.

### Answer-changing selector and payload rows

For recipient answer \(a_x\), donor answer \(a_y\), and any state \(z\), define

\[
m(z)=\operatorname{logit}_{a_y}(z)-\operatorname{logit}_{a_x}(z),\quad
n_g=m(I_g)-m(\mathrm{replay}_g),\quad
d_g=m(\mathrm{native\ donor}_g)-m(\mathrm{native\ recipient}_g).
\]

### Answer-preserving match-break rows

Let \(a\) be the fixed correct answer and \(a'\) the other target payload,

\[
c(z)=\operatorname{logit}_{a}(z)-\operatorname{logit}_{a'}(z).
\]

Set \(s=-1\) for coherent-to-broken and \(s=+1\) for broken-to-coherent, then define

\[
m_{\rm dir}(z)=s\,c(z),\quad n_g=m_{\rm dir}(I_g)-m_{\rm dir}(\mathrm{replay}_g),\quad
d_g=m_{\rm dir}(\mathrm{native\ donor}_g)-m_{\rm dir}(\mathrm{native\ recipient}_g).
\]

This makes positive movement mean movement toward the donor state in both directions without pretending the answer token changed.

### Whole-cell recovery and CE movement

For every selector, payload, or match target cell,

\[
R_{\rm mean}=\frac{\operatorname{mean}_g n_g}{\operatorname{mean}_g d_g},\qquad
R_{\rm median}=\frac{\operatorname{median}_g n_g}{\operatorname{median}_g d_g}.
\]

Ratios \(n_g/d_g\) are never formed or summarized. A cell is `native_denominator_null` if the mean denominator is nonpositive, its bootstrap lower 95% bound is not strictly positive, or its median denominator is nonpositive. Save all \(n_g,d_g\), both unnormalized point summaries, both ratios, and the lower bound of mean \(n_g\). No recovery-ratio confidence interval is used as a gate.

For answer-changing rows, donor-answer CE movement is

\[
q_g=\mathrm{CE}_{\rm replay}(a_y)-\mathrm{CE}_{I}(a_y).
\]

For match rows use donor-coherence signing,

\[
q_g=s[\mathrm{CE}_{\rm replay}(a)-\mathrm{CE}_{I}(a)].
\]

Positive \(q_g\) means CE moves in the donor state's expected direction. Intended transfer arms require the bootstrap lower 95% bound of mean \(q_g\) to be strictly positive.

## FIT gates, repeated unchanged on SELECT

Every clause applies separately to every directed cell in the manifest; no direction, variant, or recipient condition is pooled.

### Selector swap

- score and joint: \(R_{\rm mean}\ge0.30\), \(R_{\rm median}\ge0.30\), lower mean-\(n\) bound `>0`, fraction \(n_g>0\) at least 75%, and lower donor-CE-\(q\) bound `>0`;
- payload: structural replay identity;
- joint equals score within `1e-5` full-vocabulary elementwise.

### Payload swap

- payload and joint: the same `0.30/0.30`, positive-effect, 75%, and donor-CE gates;
- score: \(|R_{\rm mean}|\le0.25\);
- joint \(R_{\rm mean}\) is not below payload \(R_{\rm mean}\) by more than `0.10`.

### Match break

- score and joint in both directions: \(R_{\rm mean}\ge0.30\), \(R_{\rm median}\ge0.30\), lower mean-\(n\) bound `>0`, fraction \(n_g>0\) at least 70%, and lower signed-CE-\(q\) bound `>0`;
- coherent-to-broken payload: \(|R_{\rm mean}|\le0.25\), and joint equals score exactly;
- broken-to-coherent payload: structural replay identity; score and joint may differ and must pass independently.

### Answer-preserving joint diagonal

For each row let \(c_r,c_s,c_p,c_{sp}\) be correct-minus-other margins under replay, score, payload, and joint. Require:

- \(c_r>0\) and \(c_{sp}>0\) in at least 75% of groups separately;
- bootstrap lower means of \(c_r-c_s\) and \(c_r-c_p\) strictly above zero;
- bootstrap lower mean of \((c_{sp}-c_s-c_p+c_r)/4\) strictly above zero;
- arithmetic mean across groups of `CE_joint(correct)-CE_replay(correct)` at most `0.10` nat; and
- median across groups of the joint/replay full-vocabulary logit RMS at most `0.25 T_vocab(joint, recipient_condition)` as defined below.

## Separate FIT scales and exact control mapping

For the unique FIT target cell with recipient condition \(c\), define per-group target quantities

\[
A_g=\operatorname{median}_{h\in H}\|t^{\rm inserted}_{h,g}-t^{\rm live}_{h,g}\|_2,
\]

\[
M_g=|m(I_g)-m(\mathrm{replay}_g)|,qquad
V_g=\sqrt{\frac1{|\mathcal V|}\sum_{v\in\mathcal V}
(\ell_{I_g,v}-\ell_{\mathrm{replay}_g,v})^2}.
\]

The three FIT scales are the group medians \(T_{\rm insert}=\operatorname{median}A_g\), \(T_{\rm margin}=\operatorname{median}M_g\), and \(T_{\rm vocab}=\operatorname{median}V_g\). All must be finite and strictly positive. They are reused unchanged on SELECT. Physical control direction does not select a scale; both control directions at recipient condition \(c\) use the same unique target cell below.

| Control arm | Matched FIT target family | Matched target arm | Matching key |
|---|---|---|---|
| score | selector swap | score | recipient condition |
| payload | payload swap | payload | recipient condition |
| joint | selector swap | joint | recipient condition |

The answer-preserving joint-diagonal vocabulary gate uses the joint row's recipient condition and the last row of this table. No residual norm is ever compared with a logit quantity.

## Active selectivity controls

For each control group, define its activity \(A_g\) exactly as above from the actual `inserted-live_removed` per-site deltas. A control cell is **adequately active** when at least 75% of its groups satisfy

\[
A_g\ge0.10T_{\rm insert}(\text{arm},c).
\]

For each of the 24 `(arm, direction, recipient_condition)` coverage keys per split, at least two distinct adequately active families among neutral-source, neutral-payload, filler, and lag are required. Lag/payload is a structural no-op, excluded from the candidate count; its other identities are still checked. Failure is `insufficient_active_controls`.

Every adequately active family cell is scored on **all groups in the cell**, not only groups passing the activity threshold. It must satisfy:

- median \(|c_I-c_{\rm replay}|\le0.25T_{\rm margin}(\text{arm},c)\);
- median full-vocabulary logit RMS \(\le0.25T_{\rm vocab}(\text{arm},c)\);
- arithmetic mean `CE_I(correct)-CE_replay(correct) <= 0.10` nat; and
- \(c_I>0\) in at least 75% of groups.

Here \(c\) is always correct-answer minus the other target payload. A target pass with any adequately active control failure is `broad_contextual_equality_write`. Inadequate cells are reported but cannot count as selectivity successes.

## Exact bootstrap

All confidence intervals use 2,000 semantic-group cluster replicates. The namespace is

```text
a8-r585-replacement-group-bootstrap-v1
```

Each cell ID is the pipe-joined string

```text
<split>|<family>|<variant>|<recipient_condition>|<direction>|<arm>|<metric>
```

using the literal manifest labels above, directions `base_to_donor` or `donor_to_base`, arms `score`, `payload`, or `joint`, and metric in:

```text
denominator_mean, numerator_mean, donor_ce_mean,
single_score_harm_mean, single_payload_harm_mean, factorial_interaction_mean
```

Only metrics applicable to that family/arm are instantiated. The implementation must generate this expected set before loading the model, save the sorted set and its SHA-256, and tests must compare it with an independently generated set from R578.

Within a cell, group IDs are sorted lexicographically. For replicate `b=0..1999` and draw `k=0..G-1`, select index

\[
\operatorname{uint64}_{\rm big}\left(
\operatorname{SHA256}(\texttt{namespace:cell_id:b:k})[0:8]
\right)\bmod G.
\]

The selected group contributes all observations in that cell. Point means and bootstrap statistics are float64. Lower bounds use `numpy.quantile(0.025, method="lower")`; any reported upper bound uses `numpy.quantile(0.975, method="higher")`. Save ordered group IDs plus SHA-256 of the big-endian uint16 draw matrix and big-endian float64 statistic vector for every instantiated cell.

## FIT-first opening and terminal precedence

Run and decide all FIT cells before opening SELECT. FIT scales are frozen before the FIT decision and merely reused on SELECT; no mapping, scale, threshold, family, site, or arm may change. SELECT opens exactly once only if every FIT scientific and instrument clause passes. FINAL_TEST and OOD remain closed.

Record every failed clause. The single terminal label follows this precedence:

1. dependency not held or lock absent: `not_executed_upstream_dependency` and no model call;
2. authority/hash/census/schema/price preflight failure: integrity abort, no scientific result;
3. replay, canonical-term, padding, hook-delta, finiteness, or structural-identity failure: `invalid_instrument`;
4. any required natural denominator or scale invalid: `native_denominator_or_scale_null`;
5. any intended full-four-site target transfer gate fails: `factor_capacity_null`;
6. targets pass but an opposing single-factor or joint-diagonal interaction gate fails: `factorization_not_identified`;
7. factorization gates pass but active-family coverage fails: `insufficient_active_controls`;
8. coverage passes but any adequately active control fails: `broad_contextual_equality_write`;
9. all FIT clauses pass but any corresponding SELECT clause fails: prefix the applicable labels 3–8 with `select_`;
10. all clauses pass: `held_operational_selector_payload_factorization`.

A complete null/invalid artifact is written for cases 3–9 with all raw evidence collected before the decision. Integrity aborts never masquerade as scientific nulls. No smaller site set may rescue a failure.

## Compact sufficient-stat evidence schema

The result is a small JSON manifest plus hash-bound little-endian float32 `.npy` arrays and JSONL sufficient-stat tables. `.npy` arrays are C-contiguous with fixed row order recorded in the manifest; all aggregate arithmetic and sums of squares use float64.

1. **Authority table:** one record for each included R578 row, with row/group/split/family/variant IDs, both endpoint sequence IDs, directions, token/answer IDs, semantic source/payload/query positions, equality support, and length. Expected counts are 1,872 FIT and 936 SELECT rows.
2. **Unique endpoint table:** one record per 1,728 FIT or 864 SELECT endpoint. Save native target logits, target CEs, log-normalizer, length/final position, and references into native factor arrays.
3. **Native factor arrays:** keyed by `(endpoint_id, site, role)`, save score scalar \(e\), projected value \(u\), canonical role-summed equality term, native head output, and non-equality remainder. Endpoint factors are stored once, not copied into every direction.
4. **Directed record table:** one record per 3,744 FIT or 1,872 SELECT directed pair. It references recipient/donor endpoint factors and contains replay plus score/payload/joint target logits, both target CEs, log-normalizer, per-site delta-array indices, exactness errors, and every \(n_g,d_g,q_g,c\) sufficient statistic.
5. **Intervention arrays:** keyed by `(directed_id, arm, site)`, save live removed term and actual hook delta. Frozen inserted terms are reconstructed exactly from referenced endpoint factors, so they are not duplicated. Save insertion-delta norms separately in float64.
6. **Full-vocabulary sufficient statistics:** for every replay-relative arm output save vocabulary size, float64 sum of squared logit differences, and the resulting RMS. The scientific artifact need not duplicate full 50,257-vectors; a streaming implementation computes the sum of squares from the actual logits before releasing the batch. Tests verify `RMS=sqrt(sum_sq/vocab_size)` and target-logit entries come from the same tensor.
7. **Cell/bootstrap table:** save exact member directed IDs, ordered group IDs, point statistics, scales, activity flags, draw/statistic hashes, intervals, gates, and failed clauses.

Every file path, byte size, dtype, shape, row-order hash, and SHA-256 is included in the result receipt. The future CPU auditor verifies exact R578 membership, no duplicated directions, all sufficient-stat identities, every scale and aggregate, the complete bootstrap trace, price, split closure, and terminal precedence without loading a model.

## Frozen execution price

The conservative schedule is fixed; no retrospective savings are claimed.

- FIT: 54 capture/replay forwards over 1,728 endpoints, `3*117=351` directed intervention forwards, and 54 independent native-comparator forwards: exactly 459 if FIT completes.
- Conditional SELECT: 27 capture/replay forwards, `3*59=177` intervention forwards, and 27 comparator forwards: exactly 231 if SELECT completes.
- Maximum: 690 forwards, zero backwards, zero fitted vectors, and zero weight updates.

The runner records calls by phase and arm. Exceeding a phase or total ceiling is an integrity abort without a scientific result. Early instrument failure may stop below the ceiling and writes only an invalid-instrument artifact. No contrast call is permitted.

## Licensed conclusion

Only `held_operational_selector_payload_factorization` licenses the statement that the complete fixed set of four equality-gated terms causally carries an operational selector/value factorization on R578 FIT/SELECT prompts under the frozen role-aligned intervention. It remains a behavior-level, oracle-equality-supported factorization. It is not a complete circuit under the project's OOD prediction, executable weight extraction, selective removal, or reusable-composition goals; it licenses separately frozen translation, active-removal, and OOD tests.
