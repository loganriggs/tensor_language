# 18 September — The readout collection so far: four families, nine lines (index)

One page across the per-behaviour reports. Every set is weight-only: heads writing along `O_h^T(u_a − u_b)` at the final
query, removed by projection, judged against a norm-matched random-direction null (16 seeds), three unrelated readers
(was−were, who−which, night−day), matched-count random head-set nulls, keep-only sufficiency, additivity, frozen numbers
on fresh rows, and natural FineWeb (training corpus) / Pile (out-of-corpus) rows. Failed predictions are kept in the
scorecards. Tags: fresh = rows unused by any selection step; natural = mined outcome-blind.

| family | core heads | lines at head grain | fresh | natural in / out of corpus | direct readout? | open port |
|---|---|---|---|---|---|---|
| temporal (tense / mood / aspect) | 9.1, 9.4, 15.5 + 11.3; cue reader 8.1 | aspectual has/had, temporal will/had, narrative was/is, modal would/will | yes ×4 | yes / yes (aspectual, temporal) | no: 40–60% MLP relay | MLP relay parts (diffuse folds) |
| number (agreement) | 5.7, 7.8, 9.7 + 11.3 | lexical were/was (within scope), perfect have/has | yes ×2 | — | — | adjacent-subject case MLP-borne; have/has over-additive (11.3 dominant) |
| pronoun (gender / number / person) | 9.6, 12.4, 15.1 (+10.1 gender, +10.5 number) | gender he/she, number they/he | yes ×2 | yes / yes ×2 | yes: 94% / 103% | MLP 8 (writes the noun and verb states) |
| selection (complement choice) | 13.8, 7.8, 8.8 (+6.3, +14.8) | adjective preposition in/of, verb particle up/down | yes ×2 | yes / yes ×2 | up/down yes 86%; in/of no 38% | in/of MLP suffix |

**Cross-family facts (weights and edits).** One temporal axis carries ≥ 50% of each temporal line's own removal (v70).
The number and temporal directions are entangled at the shared head 11.3 and orthogonal at the temporal heads (v69). The
two pronoun contrasts share an axis at the pronoun core; the two selection contrasts do not share one at the selection
heads, they share the heads (v95). The verb-number direction carries 18% of the pronoun-number readout at the pronoun
heads, not half (v96). Head 7.8 is in the number and selection cores and is the most recurrent head of the atlas (65 of
100 live lines); heads 8.1 (temporal cue reader) and 10.1 (pronoun gender noun reader) are token-only readers of the same
shape.

**What the atlas says about the rest.** Of 100 capable live lines, 52 fall outside the four cores at ≥ 2 shared heads;
the largest unassigned clusters are 11.3 + 7.8 (16 heterogeneous agreement / finiteness / subjunctive lines), 13.8 + 7.8
(the selection family, now claimed) and 10.5 + 15.1 + 8.1 (5 reflexive-person lines). Table:
`basis_aligned/claude_hourly_review/READOUT_ATLAS_TABLE.md`.

**Reports.** Aspectual: `research_update_2026-09-17_aspectual_readout_component.md`; temporal set:
`research_update_2026-09-17_temporal_readout_set.md`; two auxiliary families: `research_update_2026-09-18_readout_families.md`;
pronoun family: `research_update_2026-09-18_pronoun_readout_family.md`; selection family:
`research_update_2026-09-18_selection_readout_family.md`. Scorecards under `basis_aligned/claude_hourly_review/*_DOD_SCORECARD.md`;
cross-line summary `SHARED_READOUT_COMPONENTS.md`; receipts under `basis_aligned/bilinear_quotient/circuits/followups/`.
