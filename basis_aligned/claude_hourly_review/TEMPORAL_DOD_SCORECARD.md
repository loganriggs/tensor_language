# Temporal will/had — definition-of-done scorecard

Living document (Claude circuit lane), opened 2026-09-17 23:09 UTC after review 3. Rubric:
`basis_aligned/better_circuits.md` §1; style: `basis_aligned/communicating_results.md`; battery and
metrics as in `ASPECTUAL_DOD_SCORECARD.md` (damage, fraction, norm-matched random null, retention).

**Path now.** Cue token (tomorrow/earlier) → readout set S = {11.3, 9.1, 15.5, 9.4}, each head writing along
`O_h^T(u_will − u_had)` at the final query → will/had. S was selected by the blind 162-head sweep on the
line's authored rows (v27) and is scored here on fresh rows only. **Delta:** first rung landed.

## Claim table

| # | claim | tag | rows | key numbers | status |
|---|---|---|---|---|---|
| 1 | Removing S along its weight-only readout directions damages will/had | edit | fresh lexicon + places, 2 constructions (v28) | 3.45 logits = 81% of the margin; positive 64/64; null max 0.05 | passes |
| 2 | S is selective on was−were, who−which, night−day | edit | fresh (v28) | all three within null + 0.25×damage (11.3 is the subject-number head, yet was−were stays within gate) | passes |
| 3 | S is additive | edit | fresh (v28) | joint 3.45 vs Σ singles 3.34: gap 0.11 vs bar 0.09 (25% of the smallest single, 9.4 = 0.36) | fails as registered (superadditive by 3% of the joint) |
| 4 | 11.3 leads on fresh rows | edit | fresh (v28) | singles 11.3 1.39, 9.1 1.18, 15.5 0.41, 9.4 0.36 | passes |

| 5 | Keeping only the readout projection at the four heads keeps their will/had service | edit | opened (v29) | zero four slices 3.45 logits; keep-only retention 1.19; random-direction keep ≤ 0.015 | passes |
| 6 | The S removal transfers to three new constructions (adverb phrase, expected-that frame, agent-first comma-final) | edit | fresh templates (v29) | fractions 0.82 / 0.78 / 0.36; positive 32/32 each; null max ≤ 0.10; selective in all three; all six capability cells 1.00 | 2 of 3 inside the registered [0.40, 1.21] band; the comma-final construction (0.36) is live, selective and null-beating but below the band — fails as registered |

## Five-property status

| property | status | next |
|---|---|---|
| Simple | 4 heads + the will/had contrast (sweep-selected); matched-count null not yet run | random 4-head-set null |
| Predicts OOD | partial: sign, positivity and selectivity transfer to three new templates; magnitude band held on two of three (row 6) | third lexicon with a frozen number; natural rows |
| Extracted | held at the head boundary: four scalar ports (row 5) | cue-source fold: which token/branch each head reads |
| Selective | held on fresh rows and three templates (rows 2, 6) | — |
| Composes | measured once, fails the additivity bar by 0.02 (row 3) | pairwise Möbius terms; random-split null |

## Receipts
- v27 sweep (aspectual lane receipt): `bilinear_quotient/circuits/followups/aspectual_anchor_dod_reuse_temporal_v27_result.json`
- v28: `.../temporal_auxiliary_dod_removal_v28_result.json`; code `ops/run_temporal_dod_removal_v28.py`
- v29: `.../temporal_auxiliary_dod_keep_and_templates_v29_result.json`; code `ops/run_temporal_dod_keep_and_templates_v29.py`
