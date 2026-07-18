# Causal monosemanticity: three rounds of falsification, one survivor

Logan's verification criterion (2026-07-20): a reduction's atoms should have
*monosemantic causal effects* — concentrated, consistent across contexts, falsifiable.
Three rounds, applying the falsifiability first to the metrics themselves.

## Round 1: energy-selected class-pair blocks — falsified twice over

TN-native candidate atoms: (head, class_q, class_k) blocks of the exact layer-0 pattern
tensor, ranked by data-weighted pattern-energy mass, causally probed by zeroing each
block alone. All 14 top blocks: concentration ≈ 0.00, consistency ≤ 0.30 — diffuse and
inconsistent. AND the selection was compromised: energy mass ranks junk-token classes
(unicode debris, katakana) because the unnormalized bilinear pattern explodes on rare
tokens — the energy-vs-causal mirage, fifth appearance, now in the discovery step.

## Round 2: positive controls falsify two metrics and confound the third

Scoring known-real atoms (H7's rank-1 direction, the H5 head, mlp16's named gains)
plus a random-direction control: participation ratio and class-mass share fail to
discriminate knowns from random (mean-vector-based — wrong object); fire-consistency
discriminates but is mechanically confounded — ANY fixed direction ablated near the
unembedding yields near-rank-1 Δlogits (∝ U·d̂): random control 0.69. What carried
signal anyway: the decoded token lists (mlp16's markup direction suppresses ` fmt`,
`="`, ` []`; its structure direction suppresses capitalized sentence-starters) —
qualitative where-it-fires ↔ what-it-pushes alignment.

## Round 3: null calibration — percentiles against matched random atoms

Each candidate vs 8 random atoms of the same type at the same site:

| atom | fire-consistency (pct vs null) | verdict |
|---|---|---|
| **mlp16 dir0** | **0.98 (pct 1.0 — above every null)** | **the one survivor** |
| mlp16 dir1, dir3 | 0.76 / 0.35 (pct 0.0 — BELOW all nulls) | see below |
| H7 principal dir | 0.14 (pct 0.62) | indistinguishable from null |
| L0 content blocks ×3 | 0.03–0.16 (pct 0.0) | decisively falsified |

The below-null scores are themselves informative: dir1/dir3 are *contextual gains* —
their ablation effects legitimately vary with context, so they score less consistent
than random fixed directions. **Consistency-vs-null cannot define monosemanticity for
contextual atoms**; it detects output-aligned constancy (dir0 is a near-constant
output-basis feature, and passes decisively).

## What three rounds established

1. One validated causally-monosemantic atom in this model: **mlp16 dir0** (constant
   output-aligned structure feature; ablation effect essentially identical whenever it
   fires, beyond every matched null).
2. Layer-0 class-pair blocks are not output-monosemantic under any metric tried —
   consistent with the program's picture: layer 0 does transport and generic selection,
   not output-aligned features.
3. A metric taxonomy: mean-vector concentration measures nothing; consistency measures
   output-alignment (mechanically inflated near the unembedding, legitimately deflated
   for contextual atoms); the robust cross-round signal is qualitative token-list
   alignment between firing contexts and effect directions.
4. Methodological: every one of the three rounds was saved by a control (positive
   controls, random-atom nulls) — the falsifiability loop Logan requested works, and
   its first casualties were the rulers, which is the correct order.

Where meaningful-circuit work goes from here: behavior-targeted cards (cards/, which
carry their own set-ablation checks) rather than atom-first discovery; and MDL
structure measured by ΔCE ladders (block-sparse pattern experiment, running) where no
monosemanticity assumption is needed.

Files: `../cp_circuits.py/.json`, `../cp2_controls.py/.json`, `../cp3_calibrated.py/.json`.
