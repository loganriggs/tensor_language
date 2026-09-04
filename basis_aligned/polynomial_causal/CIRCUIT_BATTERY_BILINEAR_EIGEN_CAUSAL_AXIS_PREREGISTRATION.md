# CIRCUIT BATTERY — BILINEAR FORM CONTRACTED AGAINST THE CAUSAL AXIS (preregistration)

Registered 2026-09-04 07:32Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_bilinear_eigen_causal_axis`. Script: `ops/circuit_battery_bilinear_eigen_causal_axis.py`.
Input receipt: `circuit_battery_bilinear_eigen_moment_results.json` (§2855, sha 6e04172e20e774c70605697ecd406245e2385c00145b3162879d5752583fdfa8).
IMMUTABLE.

## The last version of the method worth testing

§2854 tested the weight-only eigendecomposition of Pearce et al. (arXiv:2410.08417, ICLR'25 Spotlight) as specified: |eigenvalue| came
out **anti-correlated** with causal damage (median Spearman −.446) on a flat spectrum. §2855 gave it the algebraically-correct repair —
weight each eigenvalue by the input's occupancy of its eigenvector, `λ·E[⟨z,v⟩²]`, which is EXACTLY the change in the block's output
along the contraction direction `u` — and it still failed (median Spearman **−.191**, gain over raw **−.024**), while a float64
diagnostic confirmed the identity itself to **2.0e-4**.

§2855's conclusion was therefore not "the statistic is wrong" but "the mediation is wrong": the form is algebraically correct about the
block's output along `u`, and that component is not what drives the behaviour. **Both prior rungs contracted against the pooled numeric
unembedding axis.** §2826 independently established a different axis that IS causally live for these tasks —
`W_U[answer] − W_U[best competing candidate]`, which carried .199 of the block's damage at 2.4× its specificity while holding .0021 of
the effect's energy. This rung contracts the bilinear form against THAT axis. It is the last version of the method this campaign has a
principled reason to try, and if it also fails the negative is complete across both plausible output directions.

Fixed before the run: blocks mlp8 and mlp10; contraction axis `u` = the mean over this task's OOD rows of
`W_U[answer] − W_U[competitor]`, competitor taken per row from that row's own native logits, then normalized; 12 directions by moment
score plus 12 by raw |eigenvalue|, deduplicated; task `numbered_list.index_successor`, split OOD, 24 rows per cell. Sign convention:
damage d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS. **No CE and no §312 L2 — the frontier's L2 is CE ADDED ABOVE THE REAL MODEL
where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing installs.**

## Predictions

```
BARS  = {rho_moment: .60, gain_over_numeric: .50, exact64: 1e-3, top_ratio: 2.0, axis_cos: .90, floor: .5}
NULLS = {rho_moment_le: .10, gain_le: 0.0, top_ratio_le: 1.0, axis_cos_ge: .99}
§2855 reference: numeric-axis moment-weighted Spearman −.191
```

**pred_a_causal_axis_moment_predicts_damage** — median over the two blocks of the Spearman correlation between `λ·E[⟨z,v⟩²]` and
measured causal damage ≥ .60. *Worked example:* if §2855's diagnosis is right — correct algebra, wrong output direction — then
contracting against the axis the behaviour is actually measured on should make the form predictive, giving .6–.9. If it lands near 0
again, then no output direction this campaign can motivate makes the weight-space bilinear form causally predictive for this block, and
the negative is complete. Null: ≤ .10.

**pred_b_causal_axis_beats_the_numeric_axis** — (this rung's median moment-weighted Spearman) − (−.191) ≥ .50.
*Worked example:* the comparison value is §2855's published number on the same task, split, blocks and statistic, differing only in the
contraction axis, so this isolates the axis. Expected ~.8 if the axis is the problem, ~0 if it is not. A DIFFERENCE of two rank
correlations. Null: ≤ 0.

**pred_c_the_identity_holds_in_float64** — median over blocks of the float64 relative error of `Δf_u = λ⟨z,v⟩²` ≤ 1e-3.
*Worked example:* §2855 measured 2.0e-4 with fp32 eigenvectors, so this should pass at the same level. **The bar is set from that
MEASUREMENT rather than from the wishful 1e-4 I registered in §2855 and which its fp32 arm could never have met** — the identity is
exact in exact arithmetic, and the residual is dominated by the eigenvectors' own fp32 accuracy. Instrument check.

**pred_d_moment_top_beats_eigen_top** — median over blocks of |mean damage of the top-12 by moment score| / |mean damage of the top-12
by raw |eigenvalue|| ≥ 2.0. *Worked example:* §2855 measured 1.13 on the numeric axis; if the causal axis makes the form predictive
this should rise well above 2. The practical form of pred_a. Null: ≤ 1.0.

**pred_e_the_contraction_changes_the_object** — |cos(causal axis, numeric axis)| ≤ .90. *Worked example:* if the two axes are nearly
parallel then this rung is re-running §2855 and cannot answer anything; the answer-minus-competitor axis is a DIFFERENCE of two members
of the numeric class while the numeric axis is their MEAN, so near-orthogonality (.0–.3) is expected. Registered so that a null result
cannot be attributed to having accidentally repeated the previous contraction. Null: ≥ .99.

## Stated null

The causal axis does not help either (ρ ≤ .10, gain ≤ 0, top ratio ≤ 1) while the identity holds and the axes are genuinely different.
That would complete the negative: for this model, at this block, the weight-space bilinear form of arXiv:2410.08417 is causally
uninformative under both the output direction its authors' framing suggests and the one this campaign independently established as
causally live.

## Price

2 blocks × [1 axis-collection pass + 1 moment-collection pass + ≤ 24 directions × 24 rows × 2 forwards] over 24 OOD rows.
Literal budget: ≤ 2,800 GPU forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes. Uses `ops/fastload.py`.
**Per §2853, §2854's correction and §2855: every figure in the resulting ledger section, including the price, will be read from this
receipt in the same command the section is written from, and none from a smoke run.**

## What this does NOT claim

Two blocks, one task, one split, a POOLED axis (the per-row axis varies and `M_u` requires one fixed `u`). Single directions, not
subspaces. A negative bounds the method for this architecture and these contractions, not for the settings its authors tested. Nothing
installs; no L2. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
