# 18 September — The readout collection so far: five families, eleven lines (index)

One page across the per-behaviour reports. Every set is weight-only: heads writing along `O_h^T(u_a − u_b)` at the final
query, removed by projection, judged against a norm-matched random-direction null (16 seeds), three unrelated readers
(was−were, who−which, night−day), matched-count random head-set nulls, keep-only sufficiency, additivity, frozen numbers
on fresh rows, and natural FineWeb (training corpus) / Pile (out-of-corpus) rows. Failed predictions are kept in the
scorecards. Tags: fresh = rows unused by any selection step; natural = mined outcome-blind.

| family | core heads | lines at head grain | fresh | natural in / out of corpus | direct readout? | open port |
|---|---|---|---|---|---|---|
| temporal (tense / mood / aspect) | 9.1, 9.4, 15.5 + 11.3; cue reader 8.1 | aspectual has/had, temporal will/had, narrative was/is, modal would/will | yes ×4 | yes / yes (aspectual, temporal) | no: 40–60% MLP relay | MLP relay parts (diffuse folds) |
| number (agreement) | 5.7, 7.8, 9.7 + 11.3 | lexical were/was (within scope), perfect have/has | yes ×2 | yes / yes (have/has) | have/has no: 60%, MLP 9–17 relay | adjacent-subject case MLP-borne; have/has over-additive (11.3 dominant) |
| pronoun (gender / number / person) | 9.6, 12.4, 15.1 (+10.1 gender, +10.5 number) | gender he/she, number they/he | yes ×2 | yes / yes ×2 | yes: 94% / 103% | MLP 8 (writes the noun and verb states) |
| person (I / you antecedents) | 8.1, 13.1, 10.5, 15.1 | reflexive I/you, object control me/you | yes ×2 | yes / yes (reflexive) | yes: 85% / 90% | — |
| selection (complement choice) | 13.8, 7.8, 8.8 (+6.3, +14.8) | adjective preposition in/of, verb particle up/down | yes ×2 | yes / yes ×2 | up/down yes 86%; in/of no 38% | in/of MLP suffix |

**A regularity across number lines.** On natural text, removing a number readout set hurts rows whose far cue noun has the
*other* number just as much as congruent rows (pronoun number v77/v78, perfect have/has v100/v101): number sets carry the
number the model resolved for the real subject; the gender set instead tracks the noun (v73/v75). Registered the opposite
way each time and falsified each time; kept.

**Cross-family facts (weights and edits).** One temporal axis carries ≥ 50% of each temporal line's own removal (v70).
The number and temporal directions are entangled at the shared head 11.3 and orthogonal at the temporal heads (v69). The
two pronoun contrasts share an axis at the pronoun core; the two selection contrasts do not share one at the selection
heads, they share the heads (v95). The verb-number direction carries 18% of the pronoun-number readout at the pronoun
heads, not half (v96). Head 7.8 is in the number and selection cores and is the most recurrent head of the atlas (65 of
100 live lines); heads 8.1 (temporal cue reader) and 10.1 (pronoun gender noun reader) are token-only readers of the same
shape.

**What the atlas says about the rest.** With five cores, 72 of the 100 capable live lines are assigned (pronoun 26, number 17,
person 13, selection 13, temporal 3). The 28 left are structured: a verb-inflection cluster on {11.3, 7.8, 17.4} (7 lines:
agreement, finiteness, subjunctive, modal perfect), the four determiner-number variants of one task on {11.2, 5.7, 16.8,
15.1}, and the correlative lines on {16.8, 14.8, 8.1, 7.8}. Table:
`basis_aligned/claude_hourly_review/READOUT_ATLAS_TABLE.md`.

**Reports.** Aspectual: `research_update_2026-09-17_aspectual_readout_component.md`; temporal set:
`research_update_2026-09-17_temporal_readout_set.md`; two auxiliary families: `research_update_2026-09-18_readout_families.md`;
pronoun family: `research_update_2026-09-18_pronoun_readout_family.md`; selection family:
`research_update_2026-09-18_selection_readout_family.md`. Scorecards under `basis_aligned/claude_hourly_review/*_DOD_SCORECARD.md`;
cross-line summary `SHARED_READOUT_COMPONENTS.md`; receipts under `basis_aligned/bilinear_quotient/circuits/followups/`.
