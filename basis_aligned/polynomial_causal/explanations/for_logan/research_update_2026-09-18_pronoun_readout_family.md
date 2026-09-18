# 18 September — Pronoun gender he/she: a third readout family, {10.1, 9.6, 12.4, 15.1}

**Path now.** A gendered noun one clause earlier (king/queen, hero/heroine …) → four late heads writing along
`O_h^T(u_he − u_she)` at the final query (the conjunction or comma before the pronoun) → he/she. The core triple
{9.6, 12.4, 15.1} is the readout atlas's most recurrent top-4 core: it appears in 19 live lines, every one a pronoun
gender / number / person decision (subject, object, possessive, reflexive), and in none of the auxiliary lines. The
temporal family {9.1, 9.4, 15.5}+11.3 and the number family {5.7, 7.8, 9.7}+11.3 do not overlap it. **Delta since the
last update:** the atlas finished (225 of 226 lines scored, one line had no usable rows), and the pronoun set went
through the fresh-row battery.

**Result on fresh rows (v71, edit).** 60 rows: 10 fresh single-token gender pairs (all that exist outside the module's 16
and the lane's used words), 10 fresh objects, three frames the module never uses. Native capability 1.00 (male) / 0.90
(female) per frame.

| measure | value |
|---|---|
| set removal, fraction of native margin | lost 0.99, because 0.96, later 1.11 (pooled 1.02) |
| positive rows | 60 / 60 |
| norm-matched random null, max of 16 | 0.02 |
| reader moves was−were / who−which / night−day | 0.02 / 0.02 / 0.04, all within null + 0.25 × damage |
| singles 12.4 / 10.1 / 9.6 / 15.1 | 0.70 / 0.43 / 0.30 / 0.26; additivity gap 0.063 vs bar 0.064 |
| zero the four slices / keep only the readout direction | 1.69 / −0.62 (retention 1.37); random keep ≤ 0.013 |

**Failed prediction, kept.** The fraction frozen from the atlas (0.89 ± 0.15) was exceeded on the 'later' frame (1.11):
the set removes more than the whole margin there. Six of seven registered predictions held; the band is a miss, not a
pass. A re-frozen band from v71 will be tested on a further fresh panel before the OOD property is called.

**Natural rows (v73, edit, FineWeb = training corpus, out-of-panel).** 64 rows mined outcome-blind: a gendered noun
within 12 tokens, next token he/she, 16 per (noun gender × pronoun) cell; the noun filter is any-sense ("female dorms",
"groom a successor" pass). Congruent rows (noun gender = pronoun, native capability 1.00): the set removes 3.22 of the
4.16-logit margin (77%), positive on 31/32, null max 0.05, selective; bars frozen from the aspectual natural line held.
Incongruent rows (noun gender ≠ pronoun, capability 0.44 / 0.38): removal moves the model toward the pronoun the text
actually uses (−0.25 logits), as a cue-reading set should. Pile rows (v75, out-of-corpus, same miner and bars): 83% of the congruent margin (2.76 of 3.35), positive 30/32, null max 0.07,
selective; incongruent rows shift toward the text's pronoun (−0.68). Random four-head-set null (v72): best random quadruple 0.12 vs
the set's 1.75, none live.

**What is different from the auxiliary families.** Removal along one weight-only direction per head takes the *entire*
margin (the auxiliary sets take 45–60%), keep-only raises the margin above native, and additivity sits exactly at the
bar with 12.4 the largest single. The response census (v74, exact λ-recurrence split, closure 7e-5) confirms it: the four heads' own writes carry 94% of
the linear attribution, the downstream MLPs net −6% (MLPs 12/13 amplify, the calibrator MLP 17 pushes back), and the
nonlinear remainder is 0.8%. Unlike the auxiliary families, there is no MLP relay port to declare.

**Scope.** One behaviour, one fresh panel, two natural panels (FineWeb, Pile), head grain. All five properties have head-grain
evidence; not yet: source folds (what the four heads read), the other 18 pronoun lines on fresh rows. Scorecard: `basis_aligned/claude_hourly_review/PRONOUN_GENDER_DOD_SCORECARD.md`; atlas table:
`basis_aligned/claude_hourly_review/READOUT_ATLAS_TABLE.md`; receipts `pronoun_gender_dod_battery_v71_result.json`,
`readout_atlas_v68_result.json` under `basis_aligned/bilinear_quotient/circuits/followups/`.
