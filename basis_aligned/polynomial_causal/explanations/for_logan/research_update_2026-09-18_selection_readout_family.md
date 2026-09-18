# 18 September — Complement selection: a fourth readout family, {13.8, 7.8, 8.8} (+6.3 / 14.8)

**Path now.** A lexical head selects the function word that follows it across a parenthetical: an adjective its
preposition (interested → in, afraid → of), a verb its particle (woke → up, calmed → down). Both are read out by late heads
writing along `O_h^T(u_a − u_b)` at the comma before the function word: {8.8, 6.3, 13.8, 7.8} for in/of and {13.8, 14.8, 7.8, 8.8}
for up/down. The in/of set is exactly the set the earlier on/of census found (v54), and 13.8 + 7.8 recur in nine live atlas
lines that are all complement / particle / preposition selections. **Delta since the pronoun update:** the readout atlas's
family assignment (100 capable live lines: pronoun 28, number 17, temporal 3, other 52) pointed at this cluster among the
52 unassigned lines; two of its lines went through the fresh-row battery.

| line (fresh rows, edit) | set | fraction vs frozen | positive | null max | selective | additive gap / bar | keep-only |
|---|---|---|---|---|---|---|---|
| adjective preposition in/of (v85, 96 rows) | 8.8, 6.3, 13.8, 7.8 | 0.67 / 0.62 / 0.49 vs 0.54 ± 0.15 | 96/96 | 0.06 | yes (night−day 4× null, inside the gate) | 0.056 / 0.103 | 1.08 |
| verb particle up/down (v86, 96 rows) | 13.8, 14.8, 7.8, 8.8 | 0.43 / 0.47 / 0.40 vs 0.52 ± 0.15 | 96/96 | 0.09 | yes | 0.036 / 0.074 | 0.97 |

Fourteen of fourteen registered predictions held. The two decisions share three heads and pass the same battery on fresh
rows, so the selection family is a component family like the pronoun family, not one line's co-occurrence.

**Nulls and censuses (v87–v90).** Both sets beat sixteen matched-count random quadruples (2.14 vs 0.19; 1.76 vs 0.09; no random
quadruple live). The response censuses differ: the up/down set is a direct readout (86% of the effect is the heads' own
writes, closure 1e-5), but the in/of set is not — its heads carry 38% and the MLP suffix (MLPs 8–15, largest MLP 9) carries
the rest, with MLPs 16/17 pushing back. Two registered predictions failed there and are kept. The early members 6.3 / 7.8 /
8.8 remove 0.4–0.8 logits each when edited but show up as small direct terms, so their effect runs through the MLPs.

**What is different.** Head 7.8 is the most recurrent head of the whole atlas (65 live lines) and belongs to the number
family's core as well; here it is the smallest single in both sets (0.41, 0.33). Head 13.8 leads the particle set (0.70) and
8.8 the preposition set (0.82). Selectivity holds against the three standing readers, but the canonical night−day reader
moved four times its null on the in/of line while staying inside the damage-scaled gate; that is reported, not hidden.

**Scope.** Two behaviours, one fresh panel each, head grain. Not yet: natural rows, source folds; the in/of MLP response part is open. Scorecards: `basis_aligned/claude_hourly_review/SELECTION_DOD_SCORECARD.md`,
`SELECTION_PARTICLE_DOD_SCORECARD.md`; receipts `selection_dod_battery_v85_result.json`,
`selection_particle_dod_battery_v86_result.json` under `basis_aligned/bilinear_quotient/circuits/followups/`.
