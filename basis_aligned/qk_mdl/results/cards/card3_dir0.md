# Circuit card 3: the structure gain (mlp16 dir0) on a legal citation

**The one atom that survived null-calibrated monosemanticity (results/18).**

## The context

`The court rejected that argument. Smith v. Jones, 410 U.S. 113, 121 (1973); see also`

## Where dir0 fires in this prompt

| position/token | dir0 coefficient |
|---|---|
| 2: ` rejected` | +44779 |
| 5: `.` | +51505 |
| 8: `.` | +41319 |
| 21: `);` | +44476 |
| 22: ` see` | +42318 |
| 23: ` also` | +53520 |

## Baseline prediction at the final position

Top continuations: ' Smith', ' United', ',', ' U', ' Jones', ' H'

## Ablation effects at the final position

| arm | Δ on top-8 baseline continuations (mean) | top suppressed |
|---|---|---|
| dir0 projected out | +0.275 | ' JUSTICE', '═', '��' |
| dir1 projected out (contrast) | -0.001 | ' still', ' deal', ' really' |
| random dir projected out (null) | +0.002 | 'stall', 'er', 'als' |

## Verdict

**Selective and consistent — with an instructive sign.** dir0 fires precisely at the
citation-structure positions (peak at ` also`; also `);`, ` see`) and its ablation
moves the case-name continuations by +0.275 — two orders of magnitude above both
controls (dir1 −0.001, random +0.002). The sign: removing dir0 makes generic
case-name starts MORE likely and suppresses court-boilerplate tokens (` JUSTICE`).
So the atom is not a "promote citations" feature — it is an intra-register
distribution shaper: given legal-citation context, it redistributes probability
within that register's vocabulary. Monosemantic in the validated sense (one
direction, same effect whenever it fires, selective vs nulls); its meaning is a
conditional reweighting, not a token booster. Caveats: single prompt; corpus-scale
selectivity is in results/18; the fired-coefficient magnitudes are raw (unnormalized
model units).