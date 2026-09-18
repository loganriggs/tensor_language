# 18 September — Correlatives: a sixth readout family, {8.1, 16.8, 14.8, 7.8}

**Path now.** A correlative's first element (either / both / neither / not) selects its second (or / and / nor / but) after the object
noun; four late heads read it out along `O_h^T(u_a − u_b)` at the noun. Five live atlas lines share the set; 8.1 — the token-only cue
reader of the temporal and person families — leads or co-leads on a token cue again. **Delta since the person update:** two lines through
the head-grain battery, natural rows on both corpora, nulls and censuses.

| line (edit) | set | fresh rows: fraction vs frozen, positive, null | random-set null | census (direct) | natural FineWeb / Pile (congruent) |
|---|---|---|---|---|---|
| either/not → or/but (v119, v121, v122, v125, v126) | 16.8, 14.8, 7.8, 8.1 | 0.46 / 0.44 / 0.42 vs 0.43 ± 0.15; 96/96; 0.06 | 1.99 vs 0.20, none live | 117% (MLPs 15–17 push back +1.0) | 21% and 25% of 5.5–5.7 logits; either/or 29–34%, not/but 5–14% |
| both/neither → and/nor (v120, v123, v124, v127, v128) | 8.1, 7.8, 16.8, 14.8 | 0.47 / 0.45 / 0.48 vs 0.47 ± 0.15; 96/96; 0.09 | 3.37 vs 0.10, none live | 106% (push-back +1.4) | 43% and 42% of 7.2–7.9 logits; neither/nor 116–130% (the set is the whole 'nor' decision), both/and 5–6% |

Thirty-five of thirty-six registered predictions held; the one failure is kept: the both/neither set is over-additive on fresh rows
(joint 3.37 vs singles 3.11, with 8.1 alone at 1.22). On natural text the family behaves as a token tracker: rows where a far cue
disagrees with the text's second element move toward the text under removal, on all four panels. The two halves of each decision are
unequal in the wild: either/or and neither/nor are carried by the set, not/but and both/and barely (an any-sense "not" and the default
"and" dominate those margins).

**Readers caveat.** Both batteries ran with will−would / who−which / night−day rather than the documented was−were, because importing
another runner for its word list set the readers at import time; found after the runs, corrected in the docstrings, recorded on the
board, and turned into a rule (readers explicit; word pools from `dod_lexicon`).

**Scope.** Two behaviours, one fresh panel each, natural on both corpora, head grain. Not yet: source folds (is 8.1 a token-only
either / both reader?), the three remaining correlative lines. Scorecards `basis_aligned/claude_hourly_review/CORRELATIVE_EITHER_NOT_DOD_SCORECARD.md`,
`CORRELATIVE_BOTH_NEITHER_DOD_SCORECARD.md`; receipts under `basis_aligned/bilinear_quotient/circuits/followups/correlative_*`.
