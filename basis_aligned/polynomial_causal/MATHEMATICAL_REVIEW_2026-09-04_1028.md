# Three-hourly mathematical review — 2026-09-04 10:28Z (Claude, lane 1)

SIGN CONVENTION throughout (§2135): frontier L2 is **CE ADDED ABOVE THE REAL MODEL, LOWER IS BETTER** (§312: +2.6735 beating
+2.84/+2.93). §2128/§2129/§2133/§2134 RETRACTED; **§2125 STANDS**. Explained fraction: **5.348% / 10.923% / 4.727 nat / 0 of 68**.

## What changed since the 07:13Z review, and why it reframes the mathematics

The circuit-battery lane closed itself (§2871/§2872) and the work moved to the §312 construction, which is now decomposed:

| block | error share | | block | error share |
|---|---|---|---|---|
| front MLP tables | **+1.0045** | | `tailE` | +0.1597 |
| motif heads (a2–a9) | +0.3988 | | early attention `a0`/`a1v` | +0.0574 |
| tail dictionaries (a10L–a17L) | +0.3864 | | CP units `c4`–`c9` | **−0.2140** |

Three facts from this session are, I now think, **one fact**, and it is a mathematical one rather than an empirical curiosity:

1. **§2883:** the CP-unit reconstructions **beat the real MLPs** they replace (−0.2140).
2. **§2890:** a rank-1 truncation of the ridge-fitted tail link maps beats the full fit **on the window the maps were fitted on**
   (−0.0062 in sample, −0.0294 fresh), with the fit-window and fresh-window rank curves agreeing at **Pearson .962**, same argmin,
   same argmax — so it is **not** overfitting.
3. **§2880/§2888/§2889:** super- and subadditivity everywhere (+3.2104 vs 1.4350 inside the MLP stage; +0.3023 for front×motif;
   .7441 → .3988 across motif layers).

**The unifying fact: every component of the frontier is fitted to a LOCAL objective — per-layer ridge reconstruction of that
component's output — while the frontier is scored END-TO-END in cross-entropy.** Local-optimal is not global-optimal, the gap is
measurable at 0.006–0.19 nats on one block alone, and co-adaptation under a mis-specified objective is exactly what produces
interaction terms of both signs. This is the prompt's own pruning criterion — *"optimize only local MSE"* — turning out to describe
the construction we are trying to improve.

**Primary literature, and it is close enough to be prior art rather than analogy.** Braun, Taylor, Goldowsky-Dill & Sharkey,
*Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*, arXiv:2405.12241 (2024), train sparse
dictionaries by minimising **KL divergence of the model's output distribution** instead of reconstruction error, and report a Pareto
improvement: more network performance explained, **fewer total features, fewer simultaneously active features, no interpretability
cost**. That is the same substitution — local reconstruction → downstream loss — on a different dictionary family, with the outcome my
§2890 measurement predicts.

## Ranked moves

### 1. End-to-end (loss-weighted) fitting of the frontier's components — **top**

- **Exact object.** The ridge solutions produced by `fit_res`, `fit_tableres` and `fit_attnd` in `ops/frontier_fisher8.py`: the front
  tables `m0E/m1/m2E/m3E`, the CP units `c4`–`c9`, and the tail dictionaries `a10L`–`a17L` (ten class rows `CV[c]` plus four 1152×1152
  within-class maps `LW[k]`).
- **Theorem / operational definition.** For end-to-end loss `L(Ŷ)` with `Ŷ = f(θ)`, a second-order expansion gives
  `L(θ) ≈ L(θ*) + ½ (θ−θ*)ᵀ JᵀHJ (θ−θ*)`. Minimising plain `‖ΔY‖²` is minimising under `H = I`; the correct local surrogate is the
  **Gauss–Newton metric** `‖ΔY‖²_{JᵀHJ}`. Equivalently, fit each block by weighted least squares with the downstream sensitivity as the
  weight, or directly by the KL objective of arXiv:2405.12241.
- **Assumptions that may fail.** (a) RMSNorm and the class switch make the block-to-output map non-smooth, so the linearisation may be
  poor over the perturbations we actually apply; (b) the Gauss–Newton metric must be stable across the 512-document fitting window;
  (c) **adjacency to a CLOSED item, stated honestly** — §2125 established that *Fisher-based **selection*** does not install into this
  frontier. That is a different operation (choosing which CP units to keep) on a different object, but it is evidence that this
  frontier is insensitive to at least one second-order criterion, and it should lower the prior accordingly.
- **Measurable consequence beyond reconstruction.** A **lower L2 at equal parameter count** — an actual improvement to the 2.6735
  rather than an attribution of it — plus better OOD transport, since the objective becomes the quantity the frontier is scored on.
- **Cheapest falsifying experiment.** Already executed below: sweep a scalar `s` multiplying every fitted `LW[k]`. Scale changes the
  magnitude along the ridge solution's own direction and nothing about the model class. If **no `s < 1` beats `s = 1`**, the mismatch
  does not show along magnitude and this move is bounded to re-parameterisation rather than re-weighting.

### 2. Balanced truncation with a Glover certificate for the tail cascade

- **Exact object.** The cascade `a10L → a11L → … → a17L`: eight class-conditional affine maps composed through the residual stream.
- **Theorem.** For a linear system, **balanced truncation** admits the a priori bound `‖G − G_r‖_∞ ≤ 2 Σ_{i>r} σ_i` in the Hankel
  singular values (Glover 1984), and **Ho–Kalman** realisation gives the minimal state dimension as the Hankel rank. Per-matrix SVD —
  what §2884/§2887/§2891 truncated with — optimises each factor in isolation and is *not* the optimal reduction of a composition, which
  is a candidate explanation for why its rank curve is uninterpretable at intermediate rank.
- **Assumptions that may fail.** The cascade is **switched** (class-conditional) and passes through RMSNorm, so it is not LTI; the
  Glover bound applies per switching mode, and mode-mixing weakens it. Gramians must be estimated from finite activations.
- **Measurable consequence.** A **certified** parameter count — an a priori error bound rather than a measured one. **Nothing in this
  ledger currently carries a certificate**, and the prompt lists approximation certificates explicitly.
- **Cheapest falsifying experiment.** CPU-only once the eight `LW` dictionaries are dumped once: compare **Hankel singular value decay
  of the composed cascade** against **per-matrix singular value decay**. If the Hankel spectrum decays no faster, the cascade has no
  low-order realisation and this move dies for ~0 GPU.

### 3. Möbius / Harsanyi exact attribution over the six blocks

- **Exact object.** The set function `v(S) = L2_F(baseline) − L2_F(restore S to real)` over the six `cfgF` blocks.
- **Theorem.** The **Möbius transform** (Harsanyi dividends) `m(T) = Σ_{S⊆T} (−1)^{|T|−|S|} v(S)` decomposes any set function exactly;
  the **Shapley value**'s efficiency axiom makes per-block shares **sum to `v(N)` by construction**, which §2886's single-block shares
  do not (they recover 1.7928 of 2.6735, a 32.9% gap).
- **Assumptions that may fail.** Needs `2⁶ = 64` evaluations — which is **now affordable**: §2888/§2889 validated fit-once/eval-many at
  a baseline deviation of exactly 0.0, so 64 arms is roughly **one pipeline run**, not 64.
- **Measurable consequence.** An attribution that sums, every pairwise and higher interaction term identified, and a principled ranking
  of where to spend effort. It is **attribution, not compilation**, which is why it ranks third despite being the cheapest.
- **Cheapest falsifying experiment.** Constructive rather than falsifiable; the internal check is that the Möbius terms reconstruct the
  measured `v(N)`.

## Pruned, with reasons

- **More per-matrix rank knobs.** §2884/§2887 both registered them unreportable and §2891 explained why the knob is uninterpretable —
  duplicates completed work.
- **Metric-constructed bases/spans, half-price/K-reduction, conditioning on cfgE, c6–c9 reordering, m16 cheap interface, sink-head
  scalar, v1 factorization.** All on the CLOSED list (§2118, §2125, §2126, §2127, §2131, §2132).
- **Anything scored only by per-layer reconstruction MSE.** §2890 is the direct evidence that this objective disagrees with the one the
  frontier is graded on — the review's own finding prunes its own most obvious candidates.
- **Circuit-battery refinements.** §2871/§2872 established that instrument cannot resolve per-component selectivity; three replications
  agreed.

## Executed

**§2890** (this review's analysis, CPU-only, no price): the fit-window/fresh-window comparison that refutes overfitting and establishes
the local/end-to-end mismatch. Written to the ledger.

**Preregistered and enqueued:** `FRONTIER_TAIL_LINK_SHRINKAGE_PREREGISTRATION.md` (10:26Z) — the one-parameter test of move 1. Scale
every fitted `LW[k]` by `s ∈ {0, .25, .5, .75, .9, 1, 1.1, 1.25}`, reporting **both** `L2_F` and the in-sample `L2_C`, in **one pipeline
run** via fit-once/eval-many. `pred_b`: some `s < 1` beats `s = 1`. `pred_c`: the `s = 0` arm reproduces §2881's independently measured
**+0.1740** for `LW := {}` within .02 — an anchored endpoint rather than a free curve. Null
`b_null_the_ridge_fit_is_optimal_along_this_direction` bounds move 1 to re-parameterisation if nothing below 1 improves.

## Sources

- Braun, Taylor, Goldowsky-Dill & Sharkey, *Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*,
  arXiv:2405.12241 — https://arxiv.org/abs/2405.12241
- Glover, *All optimal Hankel-norm approximations of linear multivariable systems and their L∞-error bounds*, Int. J. Control 39(6),
  1984 (balanced truncation error bound).
- Harsanyi dividends / Möbius transform of a cooperative game; Grabisch & Roubens, *An axiomatic approach to the concept of interaction
  among players in cooperative games*, Int. J. Game Theory 28, 1999.
