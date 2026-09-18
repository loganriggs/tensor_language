# 18 September — The readout collection so far: seven families, sixteen lines (index)

One page across the per-behaviour reports. Every set is weight-only: heads writing along `O_h^T(u_a − u_b)` at the final
query, removed by projection, judged against a norm-matched random-direction null (16 seeds), three unrelated readers
(was−were, who−which, night−day), matched-count random head-set nulls, keep-only sufficiency, additivity, frozen numbers
on fresh rows, and natural FineWeb (training corpus) / Pile (out-of-corpus) rows. Failed predictions are kept in the
scorecards. Tags: fresh = rows unused by any selection step; natural = mined outcome-blind.

| family | core heads | lines at head grain | fresh | natural in / out of corpus | direct readout? | open port |
|---|---|---|---|---|---|---|
| temporal (tense / mood / aspect) | 9.1, 9.4, 15.5 + 11.3; cue reader 8.1 | aspectual has/had, temporal will/had, narrative was/is, modal would/will | yes ×4 | yes / yes (aspectual, temporal) | no: 40–60% MLP relay | MLP relay parts (diffuse folds) |
| number (agreement) | 5.7, 7.8, 9.7 + 11.3 | lexical were/was (within scope), perfect have/has | yes ×2 | yes / yes ×2 (were/was fails the tense-reader gate on natural rows; the leak is MLP-borne downstream, not a direction artifact: v138–v142) | have/has no: 60%, MLP 9–17 relay | adjacent-subject case MLP-borne; have/has over-additive (11.3 dominant) |
| pronoun (gender / number / person) | 9.6, 12.4, 15.1 (+10.1 gender, +10.5 number) | gender he/she, number they/he | yes ×2 | yes / yes ×2 | yes: 94% / 103% | MLP 8 (writes the noun and verb states) |
| person (I / you antecedents) | 8.1, 13.1, 10.5, 15.1 | reflexive I/you, object control me/you | yes ×2 | yes / yes (reflexive) | yes: 85% / 90% | 10.5's diffuse source; 8.1 / 13.1 / 15.1 close to a token lookup |
| correlative (either/both/neither/not → or/and/nor/but) | 8.1, 16.8, 14.8, 7.8 | either/not, both/neither | yes ×2 | yes / yes ×2 | yes: 117% / 106% (late MLPs push back) | none: all four heads close to a token lookup with constant patterns |
| noun number (determiner / numeral / demonstrative → plural vs singular noun) | 11.2, 7.8, 15.1 | demonstrative these/this, numeral three/one | yes ×2 | — | — | numeral keep-only 0.62 (sufficiency open) |
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
100 live lines). Head 8.1 is one token lookup serving four families: its block-0 copy of the cue token, read along each family's
contrast, has the right sign for since/by, tomorrow/earlier, I/you, me/you, either/not and both/neither and beats random token pairs on
every one (v133); the lookups are not fully contrast-specific (cross-talk up to 0.37 of own for the temporal and either/not cues).

**Shared heads, separate directions (v134–v136).** Families overlap in heads (15.1 and 10.5 sit in both the pronoun and person
sets; 7.8 in number and selection), but removing another family's direction at a set's own heads removes nothing: 0.00 at the
pronoun-number set (person direction), −0.001 at the person set (pronoun-number direction), 0.006 at the in/of set (number
direction), against 1.55 / 2.27 / 2.14 for the own directions. The components overlap in heads, not in what they write.

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
