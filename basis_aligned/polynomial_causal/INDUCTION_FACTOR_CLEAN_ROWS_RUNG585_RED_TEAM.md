# R585 pre-freeze red team: clean-row induction selector × payload factors

**Reviewed:** 2026-09-03 UTC. This is a CPU-only design review. No R585 model output was opened.

## Bottom line

The R578 rows can distinguish an equality-gated selector contribution from a projected value contribution, but only
if R585 uses a true four-arm factorial clamp at the four fixed terms and judges the opposing predictions. Merely
getting recovery from a score, value, or joint transplant is not enough: the transplanted 1,152-vector can carry
general context, and R577 has just shown that broad state transfer can succeed on every target while failing active
controls.

The following are required before freezing R585:

1. Resolve the R581 authority failure. The present R581 artifact independently reproduces the scientific R580 verdict
   with no failed scientific clause, but its formal verdict is `failed_independent_audit` because
   `envelope.next_step:value` differs: R580 stored a one-element list where the audit expected a string. R585 may be
   preregistered, but it must not execute while its stated hard dependency is formally failed. This needs a prospective
   authority resolution and successful audit, not an instruction to ignore the failure.
2. Do not reuse R558's later-site live-factor behavior. Capture native recipient and donor factors first. At each of
   L5H5, L7H3, L8H3, and L8H4, subtract the equality term computed from the current live recipient state, then insert
   the requested combination of **frozen native** recipient/donor score and value factors. Otherwise an earlier score
   transplant changes a later native value, so the nominal score-only arm contains an uncontrolled value change (and
   conversely for the value-only arm).
3. Map factors by the R578 semantic source/payload positions, not by padded absolute tensor shape. The lag controls have
   unequal lengths and would be rejected by R558's equal-length assertion.
4. Gate active selectivity and target CE improvement, not answer-margin recovery alone. A generic damaging write can
   lower the recipient answer and imitate donor-minus-recipient margin recovery.
5. Keep the full four-term set fixed. A pass localizes a distributed equality-gated factor policy; it does not prove
   that all four terms are necessary, that any term is a unique head-level basis element, or that a low-rank circuit
   has been found.

## What the existing authorities do and do not establish

- R557 exactly checks the discrete selector/value combinatorics on the older rows. It is a model-free instrument check,
  not evidence that the four neural terms carry the behavior.
- No R558 result exists in the repository. Its preregistration and implementation therefore supply prior semantics and
  thresholds, not a held four-term causal result.
- R578 supplies 180 disjoint semantic groups and exact positions. R580 establishes native FIT/SELECT behavior on 3,024
  unique prompts. The present R581 audit recomputes that scientific result but is formally failed for the envelope type
  mismatch above.
- R578's complete prompt census contains exactly one equality-successor edge in every coherent endpoint and zero in the
  broken endpoint of each match-break row. There are no extra equality edges elsewhere in these prompts. Thus the
  isolated term is already a final-query, selected-successor computation; R585 should assert this again rather than
  silently transplanting arbitrary full-sequence tensors.
- R583's audit of R577 is the relevant failure warning. Complete H7/H3 and all-attention-8 swaps transferred all six
  numeric target-direction cells and both relation cells, yet passed only 7/10 active controls. Later complete-state
  swaps were less selective. Recovery without active controls therefore cannot identify a semantic factor.

Frozen source hashes observed during this review:

- R578 rows: `8893ff83ea6080ad704f38376715d19be8971867178a4edc3bfd61fe025b39b6`
- R580 result: `7c7463a95931a51cd848ff9e8033bed77a26f7889a1a5fd1a3512ec2d1224b84`
- R580 receipt: `6a1ef728bca424ed27ec145adad1918923e91f190b96a9ff452b6838413b670a`
- current R581 audit: `8ecc1562632212ee876a794377e31966776ec15de02b5cb8d31798e438502cdb`
- R557 result: `28157bf2eb40538ea2cb9431665f6fbfe27f2dad9f5221a33228b36f629fe7cb`
- R583 audit: `3682ba0cc65363268246569f6e4fea8a4cff9051c74bdbc0cbde2c524ef9c4fd`

## Exact intervention R585 should mean

For each fixed term (h), endpoint (x), final query (q_x), and semantic target role (r\in\{A,C\}), cache

\[
e_h^x(r)=p_h^x(q_x,k_x(r))E^x(q_x,k_x(r)),\qquad
u_h^x(r)=W_h^O v_h^x(k_x(r)),
\]

where (k_x(r)) is the payload position immediately after source role (r). The score factor is really the continuous
bilinear score **together with the oracle token-equality support** (E). It is not evidence that a learned Q/K feature
has itself been identified.

For a directed recipient (r_0) and donor (d), use these four arms at every fixed term:

- replay: (e^{r_0}u^{r_0});
- score-only: (e^d u^{r_0}), with the value read at the semantic role selected by the donor edge;
- payload-only: (e^{r_0}u^d), with the donor value read at the semantic role selected by the recipient edge;
- joint: (e^d u^d).

The sum must preserve same-term pairing, ​\(\sum_h e_hu_h\). Never multiply a sum of scores by a sum of values;
that would introduce cross-term products absent from the model. At each later site, subtract the **live** equality term
before inserting the frozen hybrid. All non-equality attention terms and all downstream computation remain live.

This is a factorial clamp of four distributed terms, not four independent head ablations. Because the sites occur at
different layers, a simultaneous pass can include downstream interactions among them. Without a separately
preregistered subset/removal experiment, it licenses no statement about which one is necessary.

## Opposing predictions that make the experiment identifying

Every row has two physical directions. Report and gate them separately before any pooled summary.

| R578 family | Score-only prediction | Payload-only prediction | Joint prediction |
|---|---|---|---|
| selector swap, payload assignment fixed | moves toward donor answer | exact replay/no-op | equals score-only and moves toward donor |
| payload swap, selected source fixed | does not move toward donor answer | moves toward donor answer | moves toward donor, and should not be worse than payload-only |
| joint answer-preserving diagonal | moves from the shared correct answer toward the other payload | likewise moves toward the other payload | restores/preserves the shared correct answer |
| coherent → broken selected match | removes the selected contribution and harms its fixed answer | must not imitate match removal | equals or closely tracks score-only because the donor edge is zero |
| broken → coherent selected match | restores the fixed answer | exact no-op because the recipient edge is zero | restores; difference from score-only measures source-context content in (u) |
| neutral-source, neutral-payload, filler, or lag edit | preserves the target answer when the factor delta is active | preserves the target answer when active | preserves the target answer when active |

Two exact algebraic checks are especially valuable:

- Selector rows differ only at the final query. By causality, every earlier projected value is identical. Payload-only
  must replay the recipient, and joint must equal score-only, within the frozen numerical tolerance.
- Lag rows insert filler only after all three source/payload pairs. Their earlier projected values are identical.
  Payload-only must replay and joint must equal score-only even though the continuous score may change with query
  position.

Failure of either identity is an implementation/position-mapping failure, not a scientific null. Payload-swap
score-only is not an exact no-op: payload changes can alter the final contextual score. It is a causal crossed-factor
control and must remain small in the donor direction, not be forced to numerical zero.

## Denominators and minimum scientific gates

For answer-changing rows, orient

\[
m(z)=\operatorname{logit}_{a_d}(z)-\operatorname{logit}_{a_r}(z),\qquad
D=\mathbb E_g[m(d)-m(r)].
\]

Use a whole-cell ratio (R=\mathbb E_g[m(I)-m(r)]/D), never a mean or median of row-wise ratios. Cells are at least
split × family variant × physical direction; a semantic group is the bootstrap cluster. A cell is invalid and fails
instead of being omitted when the natural denominator is nonpositive or its 95% bootstrap lower bound is not above
zero. Save numerator and denominator in logits as well as the ratio.

Margin recovery can be obtained by damaging the recipient answer. Add the donor-answer CE coordinate

\[
C=\mathbb E_g[\mathrm{CE}_r(a_d)-\mathrm{CE}_I(a_d)]
\]

and require its 95% lower bound to be above zero for the intended score and payload transfers. Also require positive
intervention effect in at least 75% of groups. Recommended inherited FIT bars are mean and median recovery at least
0.30 and lower mean effect above zero. The crossed score/payload recovery must have absolute value at most 0.25. These
are the prior R558 scale, strengthened by R577's median and active-control lessons rather than tuned to R580 outcomes.

The answer-preserving joint diagonal has no valid full-prompt recovery denominator: base and donor have the same
answer, so their natural output difference can be arbitrarily small. Do not use R558's outcome-dependent
`min(score,payload)` ratio as the primary gate. For correct-minus-other margin (c), require separately:

- native and joint margins positive in at least 75% of groups;
- both single-factor changes (c_r-c_s) and (c_r-c_p) positive, each with a bootstrap lower mean above zero;
- factorial interaction \((c_{sp}-c_s-c_p+c_r)/4\) with bootstrap lower mean above zero;
- joint CE increase at most 0.10 nat and joint full-vocabulary RMS no more than 0.25 of the matched FIT target scale.

For match break, score the coherent→broken drop and broken→coherent restoration separately. Require mean and median
recovery at least 0.30, lower mean effect above zero, and positive fraction at least 0.70. Payload-only is an opposing
control, not a denominator. In the broken→coherent direction it must be exact replay.

All required FIT cells must pass before SELECT opens. On SELECT, evaluate exactly the same fixed all-four arms and
reuse FIT-frozen normalizers and thresholds. No term subset, threshold, control family, or direction may be changed
after FIT. FINAL_TEST and OOD remain closed.

## Active controls and the broad-write failure mode

The neutral and nuisance rows are not automatically active for every arm. A changed token that occurs after the
selected payload cannot change that payload's causal value. Neutral-pair order rotates, and lag extension is a
structural payload no-op. Therefore R585 must save, for each row, direction, arm, and term:

- the semantic query/key positions and equality support;
- cached recipient/donor score scalar and projected value vector;
- the live term removed and frozen term inserted;
- insertion-delta norm per site and the resulting final-query residual delta norm where defined.

Mark structural no-ops prospectively (selector payload-only, lag payload-only, and broken-recipient payload-only) and
test their exactness, but never count them as selectivity successes. For each non-structural arm × direction ×
`s0p0/s0p1/s1p0/s1p1` cell, require at least two distinct control families to be active in at least 75% of their
groups. "Active" should mean median insertion norm at least 0.10 of that arm's matched FIT target median, following
R577. If this coverage is absent, the terminal result is `insufficient_active_controls`, not a factor pass.

For every active control cell require:

- median absolute correct-minus-other margin change at most 0.25 of the matched FIT target scale;
- median full-vocabulary final-logit RMS at most 0.25 of the matched FIT target scale;
- correct-answer CE increase at most 0.10 nat;
- correct target answer remains above the other target payload in at least 75% of rows.

Use neutral-source, neutral-payload, filler-change, and lag-change rows. Keep contrast-target-source as a reported
diagnostic, as R580 did: it changes the context immediately before the competing payload and is not a clean invariance
control. A pass on target transfer plus failure of any active-control gate is the R577-like conclusion
`broad_contextual_equality_write`, not selector/payload identification.

## Direction, position, and padding traps

- A stored R578 row is one pair with both declared directions. Evaluate it twice; do not also synthesize a reversed row
  and thereby count the same physical contrast four times.
- Pair order rotates, so the selected payload is not at one fixed absolute index. R578 metadata is authoritative.
- FIT coherent prompts have lengths 19/20 and lag donors 27/28; SELECT has 21/22 and lag donors 29/30. R558's
  `len(base)==len(donor)` assertion would either crash or silently drop the essential lag control.
- Extract a donor score at donor `(final_query, semantic_payload)` and insert it at the recipient final query with the
  corresponding semantic target role. Do not transplant raw rotary Q/K vectors across different query positions.
- Right padding is acceptable only after each example's final query, with all metrics read at the saved final index and
  no padded key eligible for intervention. Freeze and test padded-versus-unpadded final logits for every length class.
  Reconstruction errors must be computed over valid positions, not dominated by padding.
- Save results separately for both directions. Continuous scores and contextual values need not be antisymmetric even
  when the token-level edit is reversible; pooling can hide a one-direction broad-state failure exactly like R577's
  direction-specific control failures.

## Minimum raw evidence and independent audit contract

Save row-level, not only aggregate, evidence:

- R578 row/group/split/family/variant IDs, direction, endpoint sequence IDs, semantic positions, and answer IDs;
- native, replay, score-only, payload-only, and joint B/D logits, both target CEs, log-normalizer, and full-vocabulary
  RMS from recipient;
- every factor scalar/vector and intervention norm listed above, plus exactness errors;
- every unnormalized numerator and denominator, cell membership, ordered group IDs, bootstrap draw/hash, confidence
  interval, activity flag, gate, and failed clause;
- all input/code/prereg/checkpoint/result hashes, evaluated splits, forward count, zero backwards/updates, and terminal
  decision.

The CPU audit must rebuild exact row and directed-cell membership from R578, verify no duplication, recompute every
whole-cell denominator and clustered bootstrap, check the selector/lag algebraic identities, verify that every control
counted as selective was active, recompute all gates and the terminal decision, and bind the raw artifact bytes. A
scientific null must be written normally rather than turned into an exception.

## Bounded FIT/SELECT price

The minimum decision set excludes the non-gated contrast diagnostic but includes selector, payload, joint, match
break, neutral source, neutral payload, filler, and lag rows in both directions.

At batch size 32:

- FIT: 1,872 undirected rows, 3,744 directed pairs, 1,728 unique endpoint sequences. One replay/capture pass per
  endpoint is 54 forwards; three arm passes are `3*ceil(3744/32)=351`; an independent native comparator over all
  endpoints is another 54. Conservative FIT ceiling: **459 forwards**.
- Conditional SELECT: 936 undirected rows, 1,872 directed pairs, 864 unique endpoints. Replay/capture is 27, three
  arm passes are 177, and native comparison is 27. SELECT ceiling: **231 forwards**.
- Maximum decision price: **690 model forwards, zero backwards, zero fitted vectors, zero weight updates**.

If factor capture/replay is compared directly to R580's bound native B/D logits and log-normalizer and to native
attention writes inside the same calls, the independent endpoint passes can be eliminated, giving 405 FIT and 204
SELECT forwards. This cheaper accounting must be frozen and tested before execution; it cannot be claimed
retroactively. Including the contrast diagnostic raises the conservative ceilings to 531 FIT and 268 SELECT (799
total). The runner must derive whichever price is chosen from frozen row membership and abort on excess.

## Licensed interpretations

- Intended target arms pass, opposing arms fail, joint interaction holds, and active controls pass: the fixed four
  equality-gated terms causally carry an operational selector/value factorization on the R578 task.
- Target transfer holds but active controls fail: a context-rich equality write transfers the answer, but the semantic
  factor is not isolated.
- Joint transfer holds without the opposing single-factor pattern: broad donor-state transfer or a mapping error; no
  selector/value claim.
- Exactness or algebraic identities fail: invalid instrument.
- Full all-four ceiling fails: preserve the factor-capacity null; do not rescue it by a post-outcome term or rank sweep.

Even the strongest pass is not yet a weight-level, reusable, removable circuit. It licenses a separately frozen
translation/removal test; it does not by itself satisfy circuit extraction or selective removal.
