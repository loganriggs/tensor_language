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

**What is different from the auxiliary families.** Removal along one weight-only direction per head takes the *entire*
margin (the auxiliary sets take 45–60%), keep-only raises the margin above native, and additivity sits exactly at the
bar with 12.4 the largest single. The pronoun decision looks like a nearly pure readout at these four heads, with no
MLP relay share to declare yet (no response census run).

**Scope.** One behaviour, one fresh panel, head grain. Not yet: matched-count random-set null, natural rows, response
census, source folds. Scorecard: `basis_aligned/claude_hourly_review/PRONOUN_GENDER_DOD_SCORECARD.md`; atlas table:
`basis_aligned/claude_hourly_review/READOUT_ATLAS_TABLE.md`; receipts `pronoun_gender_dod_battery_v71_result.json`,
`readout_atlas_v68_result.json` under `basis_aligned/bilinear_quotient/circuits/followups/`.
