# CIRCUIT BATTERY — EIGENVALUE × INPUT SECOND MOMENT (preregistration)

Registered 2026-09-04 07:28Z (box clock, read immediately before this line was written). Claude, LANE 1 CUDA.
Rung `circuit_battery_bilinear_eigen_moment`. Script: `ops/circuit_battery_bilinear_eigen_moment.py`.
Input receipt: `circuit_battery_bilinear_eigen_causal_results.json` (§2854, sha 82c09594e34b5eeaf3f677563e1aca24635ec2d30adc71b18bd2913520efccb5).
IMMUTABLE.

## Giving the published method its best shot, and the algebra that says why it failed

§2854 tested the weight-only ranking of Pearce et al. (arXiv:2410.08417, ICLR 2025 Spotlight) exactly as specified and found
|eigenvalue| **anti-correlated** with causal damage (median Spearman −.446) on a flat spectrum (top-8 = 2.5% of the |eigenvalue|
mass, effective rank 731–759 of 1152). That section bounded its own claim: the negative holds for the method AS SPECIFIED, not for
every variant.

There is an algebraic reason it was guaranteed to fail in that form. For the symmetric form
`M_u = sym(Leftᵀ diag(Downᵀ u) Right)`, the block's output along `u` is `f_u(z) = zᵀ M_u z` on the NORMALIZED input `z`. Removing an
eigendirection `v` with eigenvalue `λ` changes `f_u` by **exactly −λ·⟨z,v⟩²**. So the causally relevant quantity is not `λ` but
**λ·⟨z,v⟩²** — the eigenvalue weighted by how much the data actually occupies that eigenvector. A direction with a large eigenvalue
that the input never visits does nothing, and a ranking by `λ` alone cannot know that.

The repair costs ONE second-moment statistic of the block's normalized input — far less than a causal sweep, and still overwhelmingly
weight-space. This rung tests whether it rescues the method, and checks the identity above exactly.

Fixed before the run: blocks mlp8 and mlp10; output axis = the pooled unembedding direction of the single-token numbers 0–99 (§2854's
numeric axis); 12 directions by moment score plus 12 by raw |eigenvalue|, deduplicated; task `numbered_list.index_successor`, split
OOD, 24 rows per cell. Sign convention: damage d_m = m_NATIVE − m_arm, POSITIVE = the arm HURTS. **No CE and no §312 L2 — the
frontier's L2 is CE ADDED ABOVE THE REAL MODEL where LOWER IS BETTER (§2135, norm-2304 at 2.6735); nothing installs.**

## Predictions

```
BARS  = {rho_moment: .60, gain_over_raw: .50, exact_rel: 1e-4, top_ratio: 2.0, repro: .15, floor: .5}
NULLS = {rho_moment_le: .10, gain_le: 0.0, top_ratio_le: 1.0}
```

**pred_a_moment_weighted_predicts_damage** — median over the two blocks of the Spearman correlation between `λ·E[⟨z,v⟩²]` and
measured causal damage ≥ .60. *Worked example:* the identity says the change in `f_u` is exactly `λ⟨z,v⟩²`, so IF the successor
margin's damage is dominated by the `u`-component of this block's output, the correlation should be .7–.95. If it lands near 0 while
pred_c's identity holds exactly, then the block's effect on the behaviour is NOT mediated by its `u`-component — which would be a
sharper and more interesting negative than §2854's, because it would locate the failure in the method's choice of output direction
rather than in its ranking statistic. Null: ≤ .10.

**pred_b_moment_weighting_beats_raw_eigenvalue** — (median moment-weighted Spearman) − (median raw-|eigenvalue| Spearman) ≥ .50.
*Worked example:* §2854 measured the raw correlation at −.446; if moment weighting works at all, the gain is ~1.0. If the gain is ≈ 0,
the input's second moment carries no information about which directions matter, which would be surprising given the identity.
A DIFFERENCE of two rank correlations on the same directions and rows. Null: ≤ 0.

**pred_c_the_algebraic_identity_is_exact** — median over blocks of the maximum relative error between the measured change in `f_u` and
the predicted `λ⟨z,v⟩²` ≤ 1e-4. *Worked example:* this is exact algebra for a symmetric form, so fp32 round-off ~1e-6 is expected;
a large error means either `M_u` is built wrongly or the projection arm is not doing what it claims, and no correlation in this rung
would be readable. Instrument check, and the reason this rung can distinguish "wrong statistic" from "wrong output direction".

**pred_d_moment_top_beats_eigen_top** — median over blocks of |mean damage of the top-12 by moment score| / |mean damage of the top-12
by raw |eigenvalue|| ≥ 2.0. *Worked example:* the practical form of pred_a — if an engineer used the method, would weighting by the
second moment give them materially more causal directions? Both operands are absolute mean damages with a floored denominator that is
reported alongside. Null: ≤ 1.0.

**pred_e_raw_correlation_replicates_2854** — max over blocks of |this rung's raw-|eigenvalue| Spearman − §2854's| ≤ .15.
*Worked example:* the same statistic on the same task and split, measured by a different script over a partly different direction set,
so agreement within .15 says the two rungs are comparable and pred_b's gain is a real change rather than a script difference.

## Stated null

Moment weighting does not help (ρ ≤ .10, gain ≤ 0, top ratio ≤ 1) while the algebraic identity holds exactly. That would say the
method's failure at 546M is not about its ranking statistic at all — the block's contribution to this behaviour is simply not carried
by the `u`-component that `M_u` describes — and the right next question would be which output direction, if any, makes the bilinear
form causally predictive.

## Price

2 blocks × [1 collection pass over 24 OOD rows + ≤ 24 directions × 24 rows × 2 forwards], batched by token length.
Literal budget: ≤ 2,600 GPU forwards, 0 backwards, **0 fitted parameters**, < 3 GPU-minutes. Uses `ops/fastload.py`.
**Per §2853 and §2854's correction, every number in the resulting ledger section — including the price — will be read from this
receipt in the same command the section is written from, and no figure will be taken from a smoke run.**

## What this does NOT claim

Single directions, not subspaces; two blocks; one task, one split, one output axis. `E[⟨z,v⟩²]` is a second moment over the task's own
OOD rows, so it is a data-dependent statistic and the result is not a weights-only method — that is the point of the comparison, not a
concession hidden in it. Nothing installs; no L2. Does not satisfy Codex's four-phase integration contract; updates no circuit record.
