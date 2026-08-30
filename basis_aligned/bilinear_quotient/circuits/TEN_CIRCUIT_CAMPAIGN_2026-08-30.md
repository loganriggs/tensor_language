# Ten-circuit campaign — 2026-08-30

The ordering favors exact executable objects and surgical known effects.  Existing
bilinear-quotient scripts are discovery harnesses unless their own lifecycle says
otherwise; reuse their masks, hooks, and known answers, not their row provenance or
publication semantics.  New promotion runs use fresh, document-disjoint roles and the
gates in [TIER_RUBRIC.md](TIER_RUBRIC.md).

Per-circuit specifications live in
[`campaign_2026_08_30/`](campaign_2026_08_30/README.md):
[previous-token](campaign_2026_08_30/01_previous_token_bigram.md),
[induction](campaign_2026_08_30/02_induction_copy.md),
[successor](campaign_2026_08_30/03_ordered_successor.md),
[brackets](campaign_2026_08_30/04_bracket_closure.md),
[articles](campaign_2026_08_30/05_article_choice.md),
[newline](campaign_2026_08_30/06_newline_boundary.md),
[copied entities](campaign_2026_08_30/07_copied_entity.md),
[novel capitalization](campaign_2026_08_30/08_novel_capitalization.md),
[quotes](campaign_2026_08_30/09_quote_parity.md), and
[numeric formatting](campaign_2026_08_30/10_numeric_formatting.md).

## 1. Previous-token and bigram lookup

**CURRENT tier: 5.** Endpoint: next-token CE where L0H3's native top read is offset
`-1`; matched cells are self, other-offset, and diffuse queries.  The QK pattern is the
product of two token-and-RoPE bilinear forms; values are exact
`c_v(rms_norm(wte(token)))` lookup rows followed by native projection.

Evidence: [`head_0_3_exact.py`](../head_0_3_exact.py) and
[`head_0_3_exact_results.json`](../head_0_3_exact_results.json) give `-0.0`-nat exact
replacement and `+0.14675` shuffled cost.  [`layer0_fold.py`](../layer0_fold.py) and
[`layer0_fold_results.json`](../layer0_fold_results.json) give layer-0 `-0.0`, shuffled
`+0.23687`, and the layer-1 boundary `+1.47026`.

Extraction replaces L0H3, then all L0 attention, with the exact lookup.  Removal deletes
only L0H3's offset-`-1` contribution.  Collateral includes self/other-offset positions,
induction, and global text.  OOD holds out bigram pairs, frequency quartiles, domains,
and sequence lengths; require `|dCE| <= 0.001` for extraction and shuffled cost at least
`0.05`.  This owner also relays induction, so no induction credit follows automatically.

**Next promotion:** fresh-role terminal certification of exact extraction,
offset-conditioned removal, and unseen-bigram OOD; this certifies rather than raises its
Tier-5 mechanistic status.

## 2. Induction and repeated-bigram copying

**CURRENT tier: 4.** Endpoint: synthetic `AB ... A -> B` change in `log p(B)` and CE,
plus natural CE where the current/next bigram occurred earlier.  Matched negatives repeat
`A` with a different earlier follower; first mentions and noninductable repeats are
controls.

Evidence: [`induction_mechanism.py`](../induction_mechanism.py),
[`circuit_induction.py`](../circuit_induction.py), and
[`induction_injection_family.py`](../induction_injection_family.py) supply behavior,
lag, and cross-model harnesses.  The source-closed terminal result
[`terminal_copy_selection_v1_attempt2_result.json`](../../polynomial_causal/terminal_copy_selection_v1_attempt2_result.json)
finds `+0.44869993` nat for `L5H5/L7H3/L8H3/L8H4` but collateral margin
`-0.01440914`; unconditional mean replacement is rejected.

Tensor target: exact double-bilinear QK matchers, rank-128 identity payloads, and named
L4H7/L6H3 relays.  Extraction retains/rebuilds that program in an attention-null
background.  Removal must be source-match gated.  Collateral includes first mentions,
wrong followers, copied non-targets, entity/capitalization, successor, and global CE.
OOD holds out token identities, lag bands, natural versus synthetic rows, and domain.
The heads overlap copied-entity service, so parameters/effects are not counted twice.

Fresh natural FINAL and code OOD now confirm exact replay, extraction recovery
`0.9085/1.0104`, and positive target removal `+0.4686/+1.5017` nat.  The overall
certificate is nevertheless NO-GO because the frozen collateral gate fails
(`+0.00345` natural point estimate with a wide simultaneous upper bound, and
`+0.13831` nat on code).  Equality fetching is broader than the registered
repeated-bigram target, especially in code.

**Next promotion:** keep the terminal verdict fixed; factor the broad equality matcher
from behavior-specific payload/use branches or prospectively enumerate all equality
copy affordances before defining unrelated controls.  Then recursively replay the
resulting matcher, payloads, and relays to token/position primitives.

## 3. Ordered successor

**CURRENT tier: 2.** Endpoint: CE and successor-minus-self logit margin after members
of ordered families; controls use the same tokens outside successor contexts and
frequency-matched ordinary tokens.

Evidence: [`succ_map.py`](../succ_map.py) and
[`succ_map_results.json`](../succ_map_results.json) show L8H7 ranks all eight digit
successors first.  [`succ_general.py`](../succ_general.py) gives target damage `0.1478`
versus `0.00267` elsewhere.  [`succ_twin_scale.py`](../succ_twin_scale.py) and
[`year_succ.py`](../year_succ.py) extend the screen to weekdays, months, and years and
show L14H4 is mostly behaviorally dormant.

Tensor target: low-rank QK sequence-membership selector plus L8H7's successor OV map.
Extraction installs only that QK/OV program; removal deletes its successor-map
contribution.  Collateral includes self prediction, copying, numeric formatting,
punctuation, and global CE.  Select on digits; freeze weekdays/months/years and held-out
cycle members as OOD.  The numeric/date campaign below excludes these successor cells.

**Next promotion:** a first-order L8H7 query/key/value writer census with matched
same-layer controls and exact QK/OV replay.

## 4. Matched bracket closure

**CURRENT tier: 4.** Endpoint: closer-token CE when an unmatched compatible opener is
present in the preceding 64 positions.  Controls kill the nearest incompatible
delimiter, previous token, random source, or use no unmatched opener.

Evidence: [`bracket_match.py`](../bracket_match.py) and
[`bracket_match_results.json`](../bracket_match_results.json) give L13H8 deletion
`+0.8254` target/`+0.00376` global and true-match deletion `+0.6890`.
[`bracket_query_rank.py`](../bracket_query_rank.py) provides rank/random controls;
[`bracket_pointer_pairs.py`](../bracket_pointer_pairs.py) exactly replays the dense
writer-pair algebra; [`bracket_nested.py`](../bracket_nested.py) supplies nesting cells.

Tensor target: two bilinear QK factors over query/key writer parts, compatible-opener
identity/distance, and closer OV payload.  Extraction rebuilds L13H8 from registered
match terms; removal kills only the true match edge.  Collateral includes incompatible
delimiters, quotes, punctuation, and global text.  OOD holds out delimiter type, depth,
distance, and code/prose; each subtype must keep the sign.  L13H8 also serves quotes, so
the final includes bracket-only, quote-only, and joint-owner cells.

**Next promotion:** recursively reduce the sufficient matched-opener writer-pair
program to embeddings and exact upstream programs.

## 5. Article choice: a/an versus the

**CURRENT tier: 3.** Endpoint: `logsumexp(logit(a),logit(an))-logit(the)` at article
positions, with target CE and AUC secondary; controls are other determiners and matched
nonarticles.

Evidence: [`article_choice_verify.py`](../article_choice_verify.py) and result give AUC
`0.87`, with front-attention/MLP ablation drops `0.1671/0.1330`.
[`article_circuit_depth.py`](../article_circuit_depth.py) and
[`article_trigger_trace.py`](../article_trigger_trace.py) trace the front chain, while
[`article_write_rank.py`](../article_write_rank.py) identifies rank 16.  The trigger
harness's failed null is not promotive evidence.

Tensor target: exact layer-0 bigram lookup composed with MLP0 quadratic gates and the
rank-16 article write.  Extraction executes it in a front-null background; removal
deletes only its article direction/gates.  Collateral includes other determiners,
prepositions, capitalization, and global CE.  OOD holds out noun bigrams, noun
frequency, vowel/consonant onset, and sentence position.  Layer-0 lookup storage is
shared with circuit 1 and counted once.

**Next promotion:** exact attention-0 × MLP0-gate × rank-16 compositional replay and a
sufficiency curve against matched random gates.

## 6. Newline and structural boundary

**CURRENT tier: 4.** Endpoint: newline-token CE against position-jitter, count-matched
random, other punctuation, and all remaining positions.

Evidence: [`newline_crew_screen.py`](../newline_crew_screen.py) and result identify
`{7.2,8.2,10.2,11.0,12.6}`.  [`newline_crews.py`](../newline_crews.py) gives all-five
damage `0.6166` target/`0.0049` elsewhere.  [`newline_head_pairs.py`](../newline_head_pairs.py)
and [`newline_head_rebuild.py`](../newline_head_rebuild.py) give exact writer-pair
algebra; top 200/625 terms retain `0.918` on newline but `0.933` on controls, so the
compression is not behavior-specific.

Tensor target: shared low-rank structure across five double-bilinear QK scores and OV
payloads.  Extraction retains the crew plus a frozen reconstruction rank; removal
deletes all five.  Collateral includes punctuation, capitalization, quotes/brackets,
and global CE.  OOD splits prose, code, lists/tables, line length, and domain.  A
compressed score must beat equal-rank random writer-pair controls on target-minus-control
retention; exact nonspecific compression is a certified negative.  Structural owners
overlap other punctuation families.

**Next promotion:** recursively test whether a shared tensor grammar can beat the known
dense 200-pair requirement on fresh target/control roles.

## 7. Copied entity continuation

**CURRENT tier: 2.** Endpoint: CE on capitalized targets with an earlier identical
antecedent.  Matched cells are novel capitalized and copied non-capitalized targets;
global text is the collateral denominator.

Evidence: [`circuit_capcopy.py`](../circuit_capcopy.py) and
[`circuit_capcopy_results.json`](../circuit_capcopy_results.json) give committee13
damage `1.6060` copied-capitalized, `0.3456` novel-capitalized, `0.7471` copied-
noncapitalized, and `0.0277` global.  [`circuit_copy2.py`](../circuit_copy2.py) provides
target-logit/pattern candidates; [`capitalized_committee12.py`](../capitalized_committee12.py)
provides late payload hooks.

Tensor target: antecedent-match QK routing composed with capitalized/name-fragment OV
payload.  Extraction installs only this interaction; removal is antecedent-gated.
OOD holds out entity strings, BPE forms, lag, frequency, and domain.  Copied-capitalized
damage must exceed both matched control classes with simultaneous lower bounds.  This
is the interaction of copy and capitalization, not a third disjoint parameter set.

**Next promotion:** conditional first-order writer/reader census followed by a
matcher-by-payload factorial.

## 8. Novel capitalization and register

**CURRENT tier: 2.** Endpoint: capital-initial targets without an earlier identical
antecedent, excluding registered sentence/newline starts.  Controls are copied
capitalized, sentence-initial capitalized, and lowercase frequency matches.

Evidence: [`capitalized_committee12.py`](../capitalized_committee12.py) gives pooled
all-12 damage `0.6031` target/`0.0403` elsewhere.
[`capitalized_removal_greedy.py`](../capitalized_removal_greedy.py) shows no small
equivalent removal set.  [`capitalized_committee_grain.py`](../capitalized_committee_grain.py)
separates QK/OV contributions.  `circuit_capcopy_results.json` shows why copied and
novel cells cannot be pooled.

Tensor target: shared HOSVD/paired-product basis across the late crews after projecting
out the antecedent-copy route.  Extraction retains only the novel residual; removal
leaves copied-entity service live.  OOD holds out proper nouns, acronyms, title case,
sentence starts, and BPE forms.  The novel target lower bound must remain positive after
complete exclusion of copied targets.  Pooled capitalization cannot promote this entry.

**Next promotion:** novel-versus-copied conditional writer census with source-match and
sentence-start covariates.

## 9. Quote parity and closure

**CURRENT tier: 2.** Endpoint: close-quote CE under odd unmatched-quote parity.
Controls are opening quotes, balanced positions, bracket closers, and nearby nonmatching
quote tokens.

Evidence: [`quote_close_heads.py`](../quote_close_heads.py) gives L13H8 target damage
`0.5240` with about `0.003` elsewhere.  [`quote_destination.py`](../quote_destination.py)
gives recent-quote pattern share `0.0695` target versus `0.0125` control.
[`quote_state.py`](../quote_state.py) decodes state, but
[`quote_state_causal_results.json`](../quote_state_causal_results.json) shows its removal
loses only `0.0049` of the gap.  [`quote_mechanism.py`](../quote_mechanism.py) supplies
QK instrumentation and negative weights-only controls.

Tensor target: a causal parity carrier plus L13H8/L10H6 matched-opener QK and quote OV
payload.  Extraction requires all three; removal deletes only the parity-conditioned
closer contribution.  OOD splits dialogue/prose, nesting, opener distance, and quote
form.  Collateral includes brackets, punctuation, and newline.  A decoded but causally
unnecessary state earns no extraction credit.  L13H8 is shared with brackets.

**Next promotion:** fit-only causal state search with matched random subspaces, frozen
before testing whether the selected state feeds the matched-opener head.

## 10. Numeric, unit, and date formatting

**CURRENT tier: 1.** Endpoint: CE on units or registered delimiters immediately after a
numeral.  Matched negatives are numeral contexts followed by non-units.  Successor
digits/months/weekdays/years are explicit exclusions.

Evidence: [`number_word_verify.py`](../number_word_verify.py) and result give exact
identity delta `1.84e-7`, but only `77` targets and a failed null, so no promotive
behavior claim.  [`digit_copy_split.py`](../digit_copy_split.py) supplies copy/fresh
numeral masks.  [`year_succ.py`](../year_succ.py) is used only to exclude successor
contamination.

Tensor target: MLP0 quadratic number gates factored into magnitude, surface-form, and
layout factors followed by unit/delimiter readout.  Extraction installs the minimal
native-gate subset or paired-product program; removal deletes only formatting service.
Collateral includes words, punctuation, successor digits, and copied numbers.  OOD
holds out magnitudes, decimals, grouping, percentages, currencies, units, and date
layouts.  Before terminal gates, require a shuffled-label null and at least 200 target
positions per SELECT and OOD role.  Copy/successor effects remain exclusions/covariates.

**Next promotion:** fresh powered numeral-to-unit/delimiter screen with matched nonunit,
shuffled-label, successor-exclusion, and identity controls.  Another null failure
retires the candidate.

## Shared-owner accounting summary

L13H8 is shared by bracket and quote closure.  Copy heads overlap copied entities and
capitalized prediction.  Layer-0 lookup infrastructure serves previous-token and
article circuits.  Late structural writers overlap newline, punctuation, and
capitalization.  Every terminal scorer therefore includes the other nine endpoints as
prespecified collateral cells; parameters and causal CE are counted once globally even
when multiple behavioral claims pass.
