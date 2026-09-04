# CIRCUIT BATTERY — BILINEAR EIGENDECOMPOSITION, SCORED CAUSALLY (preregistration)

Registered 2026-09-04 07:16Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_bilinear_eigen_causal`. Script: `ops/circuit_battery_bilinear_eigen_causal.py`.
Input receipt: `bilinear_eigen_cpu_probe_results.json` (the CPU half, sha a88ef6ddd7481ff2ebd26ded2731df2d16364a8a7ebcaa1b0954385b1d2139b5).
Move 1 of `MATHEMATICAL_REVIEW_2026-09-04_0713.md`. IMMUTABLE.

## Object: a published method, our architecture, and a reason to doubt it

Pearce, Dooms, Rigg, Oramas & Sharkey, **"Bilinear MLPs enable weight-based mechanistic interpretability"**, arXiv:2410.08417,
ICLR 2025 Spotlight (predecessor arXiv:2406.03947; originating note Sharkey arXiv:2305.03452) fold a gate-free bilinear MLP into a
third-order tensor, contract it with an output direction `u` to obtain the symmetric form
`M_u = sym(Leftᵀ diag(Downᵀ u) Right)`, and propose **its top eigenvectors as interpretable directions obtainable from weights
alone**. bilin18 is exactly that architecture (`gated=False`, `squared_mlp=False`, verified in the checkpoint config at §2812).

Their demonstrations are toy, vision and small-LM; the literature check for this review found no treatment of RMSNorm or of
multi-layer residual composition, and no test of the RANKING against causal ground truth at scale. We have specific reason to doubt
it here: §2822–§2826 measured that energy and magnitude rankings do not track causal effect in this model — an in-sample rank-4
subspace of a removal effect holds **.700** of its energy and delivers **.139** of its damage, while one unembedding-defined
direction holding **.0021** of the energy delivers **.199** at 2.4× the block's specificity. |Eigenvalue| is that family of
statistic.

The CPU half of this move already built `M_u` from weights alone for mlp8, mlp10 and mlp11 against a pooled numeric-answer axis and
a random control, and found the spectrum **nearly flat**: effective rank (participation ratio of |eigenvalues|) **731–759 of 1152**,
top-8 holding **2.0–2.7%** of the absolute eigenvalue mass, signs split about evenly (≈574 positive / ≈578 negative). The top
eigenvalue is larger for the numeric axis than the random one (mlp8 45.1 vs 32.9; mlp10 53.6 vs 23.5) so the contraction direction
carries some signal, but the SHAPE does not concentrate.

This rung asks the causal question: ablate the block's normalized input along an eigendirection and measure the damage to the
numbered-list successor margin. Sign convention: d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS. **No CE and no §312 L2 — the
frontier's L2 is CE ADDED ABOVE THE REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing here installs.**

Fixed before the run: blocks mlp8 and mlp10; output axes = the pooled unembedding direction of the single-token numbers 0–99 and a
seeded random control; 12 top-|eigenvalue| directions and 12 seeded random eigenindices per block; task
`numbered_list.index_successor`, split OOD, 24 rows per cell.

## Predictions

```
BARS  = {rho: .50, top_over_random: 4.0, axis_gain: .20, flat_share: .10, exact: .05, floor: .5}
NULLS = {rho_le: .10, top_over_random_le: 1.5, axis_gain_le: 0.0, flat_share_ge: .40}
```

**pred_a_eigenvalue_predicts_damage** — median over the two blocks of the Spearman correlation between |eigenvalue| and measured
damage, over the 24 scored directions, ≥ .50. **Registered in the direction that VALIDATES the published method**, because that is
the outcome that would give this campaign a data-free way to enumerate a block's causal directions and would be a real step toward
compiling. *Worked example:* if the ranking works, ρ .6–.9; if |eigenvalue| is another energy statistic, ρ ≈ 0 and §2822–§2826
generalise from effect-space to weight-space. Null: ≤ .10.

**pred_b_top_eigendirection_beats_random** — median over blocks of |top-1 eigendirection damage| / |median random-eigenindex damage|
≥ 4.0. *Worked example:* a genuinely dominant direction should be several times a random one; the flat spectrum from the CPU half
predicts ~1–2. Both operands are absolute damages; the denominator is floored at 1e-6 and reported alongside so a tiny denominator
is visible. Null: ≤ 1.5.

**pred_c_the_output_axis_matters** — median over blocks of (mean top-12 damage for the numeric axis) − (same for the random axis)
≥ .20 margin units. *Worked example:* the CPU half found the numeric axis gives a larger top eigenvalue, so if the contraction
direction selects anything task-relevant the numeric-axis directions should damage the successor more. If it reads ≈ 0, `M_u`'s
eigenvectors are insensitive to which output direction is contracted, which would be a strong negative about the method. A
DIFFERENCE of damages in the same units. Null: ≤ 0.

**pred_d_the_spectrum_is_flat** — median over blocks of the top-8 share of absolute eigenvalue mass ≤ .10. *Worked example:* the CPU
half measured .020–.027, so this should pass comfortably; it is registered so that the CPU observation is SCORED rather than merely
asserted, and so that anyone reading pred_a's outcome knows whether it was tested on a concentrated or a flat spectrum. Null: ≥ .40.

**pred_e_full_basis_removal_is_the_block_input_ablation** — |damage(all 1152 eigendirections removed) − damage(block output zeroed)|
/ max(|the latter|, .5) ≤ .05. *Worked example:* projecting out a complete orthonormal basis leaves the zero vector, and a bilinear
form of zero is zero, so the two arms are the same computation up to the bias term; ~.00–.02 expected. A larger gap means the
projection arm is not doing what it claims and no direction-level number here is readable. Instrument check.

## Stated null

|Eigenvalue| does not predict damage (ρ ≤ .10), the top eigendirection is no better than a random one, and the contracted axis does
not matter. Combined with the flat spectrum that would say **the weight-only bilinear eigendecomposition of arXiv:2410.08417 does not
transfer to a normalized 546M model** — a citable negative, and one this campaign is unusually well placed to state because §2812
gives the exact composition law across RMSNorm and §2826 supplies an independently established causal axis to compare against.

## Price

2 blocks × 2 axes × 24 directions × 24 OOD rows × 2 forwards, plus two reference arms per block.
Literal budget: ≤ 2,600 GPU forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes. Uses `ops/fastload.py`.
**The price recorded in the ledger section for this rung will be read from this receipt's `price` field in the same command the
section is written from (§2853).**

## What this does NOT claim

Single directions, not subspaces; two blocks; one task and one split. Eigenvectors of `M_u` act on the block's NORMALIZED input, and
the map from residual directions to normalized directions is state-dependent — §2812 characterises it exactly but this rung does not
correct for it, so a null result bounds the method AS SPECIFIED rather than every possible variant of it. Nothing installs; no L2.
Does not satisfy Codex's four-phase integration contract; updates no circuit record.
