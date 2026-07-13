# Results log — tensor-similarity-regularized bilinear transcoders

Authoritative: `FINDINGS_metric.md` (metric correctness) + this file. Spec: `HANDOFF_tensor_sim_transcoder.md`.

---

## Tick 1 — metric built + verified BEFORE any training loop (`tensor_sim.py`, `sanity_checks.py`)

9 check groups, all pass, verified against **two independent references** (exact brute-force Isserlis to
1.4e-16, and Monte Carlo) on random layers, a jacclust DGP layer, and a real 500M bilinear MLP (r=4608,d=1152).

- **FINDING 1 (bug in handoff, fixed):** the "plug empirical Σ of the *lifted* inputs into the Gram recursion"
  recipe is silently wrong — Isserlis needs a **zero-mean** Gaussian and `x̃=(1,x)` is not one. It predicts
  `E[x̃₀⁴]=3` when the truth is `1`; measured **37–95% error**. Fixed with **non-central Wick** (mean + centered
  covariance): `tensor_inner_mean` / `fid_loss_mean` / `lifted_moments`. Exact vs MC, same `O(r r' d)` cost.
- **FINDING 2:** the CP gauge group is **permutation × rescaling, NOT `GL(r)`**. A random invertible `U` on the
  hidden index must (and does) break invariance — kept as a control that must fail. Bonus gauge the handoff
  omits: **L↔R swap is exactly invariant** (transposes each slice; a quadratic form sees only the symmetric
  part, so the function is unchanged — verified `forward(L,R)==forward(R,L)`).

---

## Tick 2 — E1 synthetic recovery (`e1_synthetic_recovery.py`)

Ground truth: rank-8 bilinear layer with 3-sparse CP factors, `d_in=16`, presented under a random CP gauge.
Transcoder: rank 32 (4× overcomplete), random init, BatchTopK k=4. It never sees `(D,L,R)` — only forward
passes (MSE) and the closed-form `L_fid`. 5 seeds. Two data regimes × two choices of the metric's `Λ`.

### FINDING 3 (major) — a data-matched `Λ` DESTROYS the off-distribution guarantee

The handoff says *"Data-matched metric (do this, residual streams are not N(0,I))."* **Do not do this** — it
defeats the entire point. `Λ`'s covariance decides **which input directions the fidelity term protects**. Fit
`Σ` to the observed data and the metric becomes blind exactly where an off-distribution mechanism (backdoor)
hides. Measured, in the regime that matters (data on a 6-dim subspace of R^16, 10 directions never probed):

| arm | tsim (train metric) | **tsim (TRUE, full-support)** | MSE(in) | **MSE(OOD)** | gt-recovery |
|---|---|---|---|---|---|
| MSE+TopK | 0.998 | **0.024** | 0.002 | **0.973** | 0.201 |
| L_fid (data-matched) | 1.000 | **0.076** | 0.000 | **0.919** | 0.133 |
| L_fid (full-support) | 1.000 | **1.000** | 0.000 | **0.000** | 0.434 |
| MSE + L_fid (data-matched) | 1.000 | **−0.021** | 0.000 | **1.014** | 0.178 |
| **MSE + L_fid (full-support)** | 1.000 | **1.000** | 0.001 | **0.000** | **0.574** |
| control (random init) | | | | | 0.054 = chance |

The data-matched metric **reports 1.000 (perfect) while the true global fidelity is 0.076** and the transcoder
is wholly wrong off-distribution. Added to MSE it provides **zero** protection (true tsim −0.02) — it silently
degenerates into MSE.

### Handoff hypothesis: CONFIRMED, but only with a full-support metric, and only off-isotropic data

With `Λ` = full-support (lifted `N(0,I)`):
- **MSE+TopK alone** — fits in-distribution (0.002) but true fidelity collapses (tsim 0.024, OOD 0.973). It
  silently drops every mechanism the data doesn't probe. *This is the paper's Figure-1 story, reproduced.*
- **L_fid alone** — perfect global fidelity (1.000), **data-free**, but poor factor recovery (0.434): with an
  overcomplete `r'` the CP decomposition is non-unique, so it lands on a dense, non-ground-truth factorization.
- **Both** — perfect fidelity **and** the best recovery (**0.574**, vs 0.20 / 0.43 alone; chance 0.054). ✓

In the **isotropic** regime (data probes every direction) the two metrics coincide and MSE+TopK alone nearly
suffices (tsim 0.984, recovery 0.877) — the fidelity term adds little. **So E1 is only informative in the
restricted-data regime**; an isotropic toy would have hidden the entire effect.

### Practical recommendation

Use a **full-support `Λ`** (identity, or `Σ + λI` ridge). The `Σ`-vs-`I` choice is *the* key design knob — it
trades data-realism against off-distribution coverage — and should be swept (`Σ_temp = (1-t)Σ + tI`). Real
residual streams have full-rank but *ill-conditioned* `Σ`, so a data-matched metric isn't literally blind, but
it heavily down-weights the low-variance directions where backdoors live. Expect the same failure, softened.

### Bug caught in-flight (methodology note)

A first version of E1 drew a **fresh random subspace on every `sample_x` call**, so "in-distribution" test data
was silently on a *different* subspace than train. It produced a plausible-looking all-arms-fail table. Caught
by an independent check (training loss converged fine while reported test MSE was 0.84). Basis is now built
once per run and shared. *Standing rule: verify against an independent reference, not internal consistency.*

---

## Next

1. Sweep the metric temperature `Σ_temp = (1-t)Σ + tI` — quantify the fidelity/realism tradeoff (FINDING 3's knob).
2. Structural-sparsity architectures (hierarchical / block-CP / fixed support), fit with **`L_fid` alone —
   fully data-free weight-space optimization**, which E1 shows is viable (tsim 1.000 with no data at all).
3. E3 Pareto (λ × K sweep: L0 vs tensor-sim vs MSE).
4. E2 SVHN backdoor (flagship) once 1–3 are solid.

---

## Tick 3 — E2 structural priors, fit on `L_fid` ALONE (DATA-FREE) (`e2_structural_priors.py`)

Same planted ground truth (rank 8, 3-sparse CP factors, d_in=16, random CP gauge). Transcoder rank 32.
**No data is used at all** — the only loss is the closed-form `L_fid` with a full-support Λ. 5 seeds.

| prior | tensor-sim | **gt-recovery** | factor-sp | eff-L0/row |
|---|---|---|---|---|
| dense (E1 baseline) | 1.000±.000 | 0.479±.058 | 0.490 | 8.6 |
| topk-rows (hard k=3) | 1.000±.000 | 0.654±.060 | 1.000 | 1.7 |
| **L1 (soft row sparsity)** | **1.000±.000** | **0.900±.045** | 0.870 | **3.2** |
| block (hierarchical) | **0.478±.097** | 0.317±.050 | 0.951 | 2.5 |
| symmetric (l'=r') | 1.000±.000 | 0.440±.057 | 0.598 | 6.4 |
| control (random init) | — | 0.066 = chance | — | — |

*(ground truth: factor-sp 1.000, eff-L0/row = 3.0)*

### FINDING 4 (headline) — data-free structured fitting BEATS data-driven fitting at recovering the truth

**L1 prior + `L_fid` alone, with zero data: tensor-sim 1.000 and gt-recovery 0.900** (chance 0.066) — and it
finds the *correct* sparsity organically (eff-L0 3.2 vs the planted 3.0). Compare E1's best arm *with* data
(MSE+TopK+`L_fid`, full-support): recovery **0.574**. So the purely weight-space fit recovers the planted
structure **far better than the data-driven one** — you can reverse-engineer a bilinear layer's sparse CP
structure with **no data whatsoever**.

Mechanism: the sparsity prior **breaks the CP non-uniqueness**. Dense `L_fid` hits fidelity 1.000 but recovery
only 0.479, because an overcomplete CP decomposition is non-unique and it lands on an arbitrary dense one.
Sparsity selects the ground-truth gauge among the infinitely many exact factorizations.

### Prior comparison (which structure to use)

- **L1 (soft) is the clear winner.** Hard top-k projection (0.654) is *worse* — it over-concentrates
  (eff-L0 1.7 < the true 3.0), i.e. projected gradient is too aggressive and gets stuck. Let sparsity emerge.
- **symmetric (`l'=r'`, "squared readout") is fidelity-free-of-charge**: tensor-sim 1.000 with **half the
  parameters**. It *can* be exact because both the Gaussian metric and the layer's function see **only the
  symmetric part** (FINDING 2). It doesn't recover the asymmetric planted factors (0.440), but for
  interpretability (each feature is a squared linear form → eigendecomposable, handoff E4) it's a free win.
- **block / hierarchical FAILS (tsim 0.478)** — an honest negative, and a control that *could* fail and did.
  The planted structure doesn't respect the block partition, so a **mis-specified** hierarchical prior cannot
  even represent the layer. Lesson: a wrong structural prior costs **fidelity**, not just recovery. Hierarchy
  is only safe when the true structure respects it (or when the blocks are learned, not imposed).

### Handoff's E1 open question: "can sim = 1 coexist with sparsity at achievable overcompleteness r'?" — YES

Hard-sparse (k=3) transcoder vs rank:

| r' | 8 (= gt rank) | 16 | 32 | 64 |
|---|---|---|---|---|
| tensor-sim | 0.887±.058 | 0.997±.003 | **1.000±.000** | 1.000±.000 |
| gt-recovery | 0.661 | 0.662 | 0.654 | 0.678 |

Sparsity and exact fidelity coexist from ~2–4× overcompleteness. At `r'` = the true rank it is *not* achievable
(0.887) — the **sparse-decomposition rank exceeds the tensor's rank**, exactly as the handoff anticipated.

### Practical recommendation

For reverse-engineering a bilinear layer's structure: **`L_fid`(full-support) + L1 on the factor rows, no data**,
at ~4× overcompleteness. Add the symmetric (`l'=r'`) constraint for free interpretability. Do **not** impose a
hierarchical/block partition unless you know the layer respects it.
