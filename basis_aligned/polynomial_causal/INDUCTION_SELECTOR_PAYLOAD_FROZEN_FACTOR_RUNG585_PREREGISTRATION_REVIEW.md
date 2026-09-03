# Review of R585 frozen selector × payload preregistration

**Review date:** 2026-09-03 UTC  
**Verdict:** **BLOCKED**

This is an outcome-blind, CPU-only specification review. I read the R585 preregistration, its pre-freeze red team, the R578 row authority, the R580 contract/result envelope, the model-free R586 package, the R557 planted semantics, the R558 preregistration, and the circuit-bootstrap playbook. I did not open an R585 or R586 scientific outcome, load the model, or use a GPU.

The high-level experiment is worth doing: it fixes a complete four-term set, freezes recipient and donor factors before intervention, uses semantic positions across unequal lengths, includes crossed single-factor predictions, keeps active controls, and limits the licensed conclusion. However, the present text cannot produce a unique audited decision. Several ambiguities are mathematical rather than editorial.

## Blocking findings

### 1. The match-break recovery formula is identically undefined

The preregistration defines, for answer-changing rows,

\[
m(z)=\operatorname{logit}_{a_d}(z)-\operatorname{logit}_{a_r}(z).
\]

It then requires recovery for both directions of `match_break_payload_preserved`. In the frozen R578 authority, all 432 FIT/SELECT match-break rows have `answer_changes=false` and `base_answer_id == donor_answer_id`. Therefore this definition gives `m(z)=0` for every state, a zero natural denominator, and an invalid cell in both directions. This is not a possible scientific null; it is a contradictory metric.

**Required correction:** define match-break recovery separately using correct-minus-other margin

\[
c(z)=\operatorname{logit}_{a}(z)-\operatorname{logit}_{a'}(z).
\]

For each direction let \(s=+1\) when the donor is coherent and \(s=-1\) when the donor is broken, and define \(m_{\rm dir}(z)=s\,c(z)\). Then use the same whole-cell formula with denominator \(\mathbb E[m_{\rm dir}(d)-m_{\rm dir}(r)]\). Freeze the donor-oriented CE sign as well: \(s[\mathrm{CE}_r(a)-\mathrm{CE}_I(a)]\), or explicitly exempt coherent-to-broken necessity from the positive-CE-improvement gate. State the exact rule rather than inheriting the answer-changing definition.

### 2. “Mean and median recovery” contradicts the only defined recovery

The text defines one whole-cell recovery ratio, \(R=\mathbb E[n_g]/\mathbb E[d_g]\), and forbids rowwise ratios. It later gates both “mean and median recovery,” but no median recovery exists under that definition. An implementation could choose median row ratios, a ratio of medians, a median numerator divided by a mean denominator, or simply relabel \(R\).

**Required correction:** either remove every median-recovery gate or define a second group-level statistic explicitly. A coherent choice is \(R_{\rm median}=\operatorname{median}_g(n_g)/\operatorname{median}_g(d_g)\), with a frozen rule for zero/nonpositive median denominator. Continue to forbid \(\operatorname{median}_g(n_g/d_g)\). Define “positive lower mean effect” explicitly as the lower bootstrap bound of \(\mathbb E_g[n_g]\).

### 3. The active-control scale is dimensionally inconsistent and its matching rule is absent

The only named target scale is

\[
T=\operatorname{median}\|\Delta_{\rm insert}\|_2,
\]

which is a residual-stream norm. The control gates then compare a logit-margin change and a full-vocabulary logit RMS to `0.25` times “that scale.” Residual-vector units cannot normalize logit units. The same undefined “matched FIT target scale” appears in the answer-preserving joint-diagonal logit-RMS gate.

It is also unspecified which target family supplies the scale for each control arm, direction, and condition. Score-only has selector and match-break targets; payload-only has payload-swap targets; joint has several possible targets. This leaves an outcome-dependent choice.

**Required correction:** use separate frozen scales in matching units. For example, define \(T_{\rm insert}\) from \(\|\text{inserted term}-\text{live removed term}\|_2\) solely for activity, \(T_{\rm margin}\) from matched target logit-margin changes, and \(T_{\rm vocab}\) from matched target full-vocabulary RMS changes. Give a complete lookup table from every control `(arm, direction, condition)` to one target family/variant and one FIT scale. State whether group summaries take a median over sites, a norm of the four-site intervention, or another fixed aggregation. Do not use the norm of the inserted term itself: replay can insert a large term while making zero intervention.

### 4. The four-arm tensor operation is underspecified

The factor is indexed by semantic role \(r\in\{A,C\}\), but the four arms drop that index. It is unclear whether the runner inserts only one selected role or the sum over both role-aligned terms. That distinction changes score-only and payload-only interventions.

**Required correction:** freeze the actual operation, for every site \(h\), as

\[
\begin{aligned}
t_h^{rr}&=\sum_{r\in\{A,C\}}e_h^{r_0}(r)u_h^{r_0}(r),\\
t_h^{dr}&=\sum_r e_h^d(r)u_h^{r_0}(r),\\
t_h^{rd}&=\sum_r e_h^{r_0}(r)u_h^d(r),\\
t_h^{dd}&=\sum_r e_h^d(r)u_h^d(r),
\end{aligned}
\]

and define the per-site hook delta as `inserted_h - live_removed_h`. Pin the exact definition of the continuous score \(p\), equality support \(E\), projected value orientation, and hook location to a specific implementation hash. In particular, say whether \(p\) is a raw bilinear score or any normalized/scaled quantity, and define \(E(q,k)=1[t_{k-1}=t_q]\) on semantic payload positions.

The existing replay and hook-delta checks are partly circular: subtracting and re-adding any self-consistently computed tensor can replay exactly. Add an independent reconstruction check that the cached \(\sum_r e_hu_h\) equals the canonical, hash-pinned equality term in the head output, and that equality plus the non-equality remainder reconstructs that output.

### 5. Contrast reporting and the price envelope disagree

R585 says the contrast-target-source edit is retained and reported, and the evidence section asks for every row/direction/arm/site. Its 690-forward ceiling explicitly excludes contrast. From R578, excluding contrast gives exactly 1,872 FIT and 936 SELECT rows, matching 459 + 231 forwards. Including the diagnostic gives 2,160 FIT and 1,080 SELECT rows and the red team's conservative 531 + 268 = **799** forwards.

**Required correction:** choose one protocol before execution:

- exclude contrast entirely from model evaluation and state that R585 reports no new contrast intervention; or
- include it, enumerate its evidence cells, and use the 799-forward conservative ceiling (or freeze and test a specific cheaper reuse schedule).

It cannot be both reported and absent from the price.

### 6. Bootstrap and aggregation choices are not reproducible

“2,000 SHA-defined bootstrap replicates” does not specify the namespace, cell-ID grammar, digest-byte interpretation, group ordering, draw rule, or NumPy lower/upper quantile methods. CE-increase gates do not say mean, median, maximum, or confidence bound. “Every active control cell” does not say whether its outcome statistics include all groups or only groups classified active.

**Required correction:** freeze the complete hash bootstrap algorithm as R580 did, list every expected cell ID, and define every statistic. For each control cell, state prospectively whether activity is a cell-level eligibility test followed by scoring all groups, or a filter followed by scoring only active groups. The former is less selection-prone. Define the answer-preserving and control CE bars as a specific point statistic and, if intended, a specific confidence bound.

### 7. Active-cell coverage and terminal precedence are not enumerable

“Every non-structural arm × direction × factorial condition” does not enumerate whether replay is excluded, which family-specific arms are structural, or the expected number of target/control cells. The listed no-ops are useful but not a full machine-checkable cell table. Several failures can also occur together, while the terminal labels have no precedence: `invalid_instrument`, invalid denominator, `factor_capacity_null`, `factorization_not_identified`, `insufficient_active_controls`, and `broad_contextual_equality_write`.

**Required correction:** add a literal cell manifest with family, variant, recipient condition, direction, arm, structural/no-op status, target/control role, scale source, and gate. Freeze expected row/group counts for every cell. Give deterministic terminal precedence and require preservation of all failed clauses even when one label wins.

### 8. The execution dependency is not yet hash-closed

The conditional dependency on held R586 and R587 is conceptually correct, but this R585 text does not pin their eventual result, receipt, audit, implementation, test, and dry-run hashes. “Bound through R586” is not an exact authority relation by itself.

**Required correction:** before any R585 model call, freeze a pre-execution addendum or implementation receipt that pins the exact held R586 result/receipt and held R587 audit plus the R578 rows and R585 preregistration. Enforce the dependency by parsed verdict and hash, not filename or existence. This can be done prospectively without changing a scientific threshold.

### 9. The evidence representation is too ambiguous to audit or price operationally

Taken literally, saving full-vocabulary changes and 1,152-dimensional vectors for every row, direction, arm, and site in JSON creates billions of numeric entries and duplicates native endpoint factors many times. The text does not say whether “full-vocabulary logit change” means the vector, its RMS, or a sufficient statistic. It also does not define how repeated endpoint factors are deduplicated while preserving exact row ownership.

**Required correction:** freeze an explicit artifact schema. Store unique endpoint factor tables once, keyed by endpoint/site/semantic role, and let directed row-arm records reference them. For each full-vocabulary RMS, save at least the vocabulary count and float64 sum of squared replay-relative logit differences (or a hash-bound binary vector) so an auditor can recompute the RMS. Specify binary/JSON files, dtypes, serialization, hashes, and expected record counts. This changes storage, not the scientific gates.

## Checks that passed

- R578 confirms exact group-disjoint FIT/SELECT rows, two declared physical directions per row, semantic source/payload/query positions, unequal lag lengths, and no answer change for match-break or control families.
- The no-contrast census and conservative price arithmetic are internally correct: FIT `1872 rows -> 3744 directions`, `1728 endpoints`, `54 + 351 + 54 = 459`; SELECT `936 -> 1872`, `864 endpoints`, `27 + 177 + 27 = 231`.
- Frozen-factor capture before any intervention correctly prevents an earlier layer's intervention from contaminating the supposedly fixed factor at a later layer.
- Same-site products are summed as \(\sum_h e_hu_h\), avoiding nonexistent cross-site products.
- Semantic-coordinate mapping rather than equal-length tensor indexing is correct.
- Selector and lag payload-only identities, plus broken-recipient payload-only replay, are valid instrument checks.
- The opposing selector/payload predictions and active unrelated controls directly address factor identification rather than rank reduction.
- The licensed conclusion appropriately disclaims unique Q/K features, OOD generalization, weight-level compilation, per-site necessity, and selective removal. Those limitations should remain unchanged after repair.

## Required disposition

Do not implement or execute the current preregistration. Repair findings 1–8 before code freeze; finding 9 must at least be resolved in the implementation contract before execution. Because the match-break gate is mathematically impossible and the scale/aggregation choices can change pass/fail outcomes, these are not safe post-outcome clarifications.

No model-free test file was needed: direct R578 metadata inspection already proves the decisive contradiction (`432/432` FIT+SELECT match-break rows have equal base/donor answers), and the row census proves the contrast/price mismatch.

## Reviewed hashes

- R585 preregistration: `b927d1ae92a5ab749a888badcfaaa0f5e7301d79b7169be8dec18babccfbd116`
- R585 red team: `fe84eeb7d3a200028264064b3671da2b95217fd30b422962c0bc27b45ede7d59`
- R578 rows: `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`
- R580 result / receipt: `7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84` / `6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a`
- model-free R586 preregistration / implementation / test / dry run: `a139948085a99a6e745d3e8bf5d08ae11b58480d30ddf5e75467b506dda3a9a5` / `ab33c1afea27d624151ad68ca230fb36ae03833e95349eb6da409778e9ea271b` / `748400e0675d37d9fd7fc7ce306ac7549b73db4b50bab2fe365abdb44b4d7841` / `0134f0218c3ec135abaec30d2028abae6e1da2c4f0c30bc78cf57d4d4aac0d30`
