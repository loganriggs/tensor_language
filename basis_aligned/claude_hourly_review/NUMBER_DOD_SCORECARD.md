# Number family (agreement at the auxiliary slot) — definition-of-done scorecard

Opened 2026-09-18 00:22 UTC (fourth path; Claude lane). Rubric: `better_circuits.md` §1; battery as in the other
scorecards. Readers for this family: has−had (tense), who−which, night−day — was−were is the target axis.

**Path now.** Number of the subject noun (singular / plural token) → readout set {11.3, 5.7, 7.8, 9.7} writing along
`O_h^T(u_were − u_was)` at the final query → was/were. Selected by the reuse censuses on four number lines (v50b–v53).

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Removing the set along its readout directions damages was/were | edit | fresh lexical-number rows (v55) | 2.67 logits = 70%; positive 64/64; null max 0.18 | passes |
| 2 | The set is selective on has−had, who−which, night−day | edit | v55 | has−had moves 0.88 vs null 0.20 (gate fails); who−which 0.22 vs 0.25 and night−day 0.25 vs 0.15 pass | fails: the number readout shifts tense |
| 3 | The set is additive | edit | v55 | singles 11.3 1.43, 5.7 0.50, 7.8 0.42, 9.7 0.28; gap 0.06 ≤ bar 0.07 | passes |
| 4 | Keeping only the readout projection keeps the heads' was/were service | edit | v55 | retention 0.47 (random keep ≤ 0.02) | fails: these heads' was/were service is not one direction each |
| 5 | 11.3 leads | edit | v55 | 1.43 vs 0.50 | passes |
| 6 | The number heads' were−was and has−had readout directions overlap in weight space: cos -0.54 (11.3), -0.57 (7.8), -0.41 (9.7), -0.16 (5.7); temporal heads -0.06 / 0.01 / -0.07; unembedding -0.26 | fold (weights only) | v56 | 0 forwards | passes 3/3: the tense movement in row 2 is a weight-level overlap of the two contrasts at the number heads |
| 7 | Removing only the part of each were−was direction orthogonal to its has−had direction (weights only, no fit) keeps 68% of the set's damage (1.82 logits, 48%, positive 100%, null max 0.13) and spares has−had (0.32 vs null 0.18); who−which / night−day within gate | edit | fresh (v57) | 38 forwards | passes 5/5 |
| 8 | Rank-2 keep (span of the were−was and has−had directions), implemented as replacement by the projection of the NATIVE slices: rank-1 0.62, rank-2 0.32, random rank-2 ≤ 0.06 | edit | v58 | identity replay exact | 2/4: pred_c (rank-1 replays v55's 0.47) failed because this protocol fixes later heads' slices at native values while v55's in-forward keep-only projects the slices the edited forward actually produces — a protocol difference, not a science result; rank-2 claim unsupported here; rerun in-forward as v58b |
| 9 | In-forward rank-2 keep (span of were−was and has−had per head): rank-1 0.47 (replays v55's 0.47 — instrument), rank-2 0.19, random rank-2 ≤ 0.093 | edit | v58b | 42 forwards | 2/3: the declared two-contrast span does not capture the number heads' service either; Extracted stays unheld |
| 10 | Per-head keep-only: 11.3 retention 0.97 and 7.8 1.05 (one-directional); 5.7 -0.30 (keeping its readout projection makes things WORSE than zeroing: its readout-direction component opposes its was/were service); 9.7 zero damage -0.08 (inert alone) | edit | v59 | 20 forwards | 2/3: my registered guess that 11.3 is not one-directional FAILED — the non-one-directional member is 5.7 |
| 11 | Matched-count random four-head-set null | edit | v60 | set 2.67 vs random max 0.11, median 0.010; none live | passes 4/4 |
| 12 | Head 5.7's readout coefficient (oriented plural−singular) is sourced from prep 0.69, subject 0.34, the2 0.08; inherited branch 0.01 | fold | v61 | closure ok | 0/2 registered guesses (subject token leads; token-only branch dominates) FAILED — 5.7 is not a token-only number-cue reader |
| 13 | Keeping only the readout projection at the one-directional pair {11.3, 7.8} (5.7 and 9.7 native) retains 0.99 of zeroing those two slices (2.19 logits) | edit | v61 | | passes |

## Five-property status
| property | status | next |
|---|---|---|
| Simple | held at head grain: 4 heads + contrast; sweeps on 4 lines and the random-quadruple null (row 11) | — |
| Predicts OOD | first fresh panel only | frozen prediction on new templates |
| Extracted | held at the head boundary for the readout pair {11.3, 7.8} (rows 10, 13); 5.7 is an upstream contributor whose service is not a readout projection and whose sources are not token-only (row 12) — declared open port | — |
| Selective | held with the orthogonalized (weight-only) number direction (row 7); the raw were−was direction is entangled with tense at the number heads (rows 2, 6) | — |
| Composes | additive (row 3) | pairwise + random-split null |

## Receipts
- v61 5.7 fold + pair keep: `.../number_family_dod_head57_fold_v61_result.json`
- v59 per-head keep: `.../number_family_dod_per_head_keep_v59_result.json`
- v60 random-set null: `.../number_family_dod_random_set_null_v60_result.json`
- v58b in-forward rank-2 keep: `.../number_family_dod_keep_rank2_v58b_result.json`
- v58 rank-2 keep (replace protocol; protocol mismatch noted): `.../number_family_dod_keep_rank2_v58_result.json`
- v56 cosines (CPU): `.../number_family_dod_readout_cosines_v56_result.json`
- v57 orthogonalized readout: `.../number_family_dod_orthogonal_readout_v57_result.json`
- v55: `bilinear_quotient/circuits/followups/number_family_dod_battery_v55_result.json`; code `ops/run_number_dod_battery_v55.py`
