# Results log — tensor-similarity-regularized bilinear transcoders

Authoritative: `FINDINGS_metric.md` (metric correctness) + this file. Spec: `HANDOFF_tensor_sim_transcoder.md`.

**Figures: [FIGURES.md](FIGURES.md)** — the five headline results as plots.

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

---

## Tick 4 — hierarchy is a SPECTRUM + the metric-temperature knob (`e3_hierarchy_spectrum.py`)

E2's block prior failed (tsim 0.478) — but that was one extreme (a *hard* mask) against a ground truth that did
not respect blocks. The honest question isn't "does hierarchy help" but "does it help **when the layer is
hierarchical**". So: two ground truths (`gt=random` 3-sparse anywhere; `gt=block` = each unit reads one block
of a 4-block partition) × hierarchy granularity `n_blocks` ∈ {1..16} × **hard** mask vs **soft** graded
off-block L1 penalty (s=0.03). All fits data-free (`L_fid` only, full-support Λ), rank 32, 5 seeds.

| n_blocks | gt=random HARD tsim | HARD recov | SOFT tsim | SOFT recov | | gt=block HARD tsim | HARD recov | SOFT tsim | SOFT recov |
|---|---|---|---|---|---|---|---|---|---|
| 1 (dense) | 1.000 | 0.856 | 1.000 | 0.856 | | 1.000 | 0.676 | 1.000 | 0.676 |
| 2 | 0.665 | 0.712 | 1.000 | 0.886 | | 1.000 | 0.765 | 1.000 | 0.821 |
| 4 | 0.478 | 0.556 | 1.000 | 0.961 | | **1.000** | 0.805 | 1.000 | 0.872 |
| 8 | 0.396 | 0.437 | 1.000 | 0.982 | | **0.621** | 0.550 | 1.000 | 0.874 |
| 16 | 0.319 | 0.372 | 1.000 | **0.986** | | **0.442** | 0.322 | 1.000 | 0.895 |

### FINDING 5 — hard-prior fidelity is a DATA-FREE STRUCTURE-DISCOVERY DIAGNOSTIC

Hard and soft hierarchy are **qualitatively different objects**, and E2 conflated them:

- **HARD masks are a *probe*, not a prior.** Fidelity collapses precisely when the mask is mis-specified. On
  `gt=block` (a truly 4-block layer) tsim stays **1.000 for every partition at or coarser than the truth**
  (n_blocks 1, 2, 4) and **breaks the moment the mask is finer than the truth** (0.621 at 8, 0.442 at 16). On
  `gt=random` it breaks immediately (0.665 at n_blocks=2). **So: the finest hard partition that still holds
  tsim = 1.0 IS the layer's true block structure** — 4 for the block gt, 1 (i.e. *not hierarchical*) for the
  random gt. You can scan structural hypotheses and read off which ones the layer admits, **with no data**.
  This turns E2's "negative" into an instrument.
- **SOFT (graded) hierarchy is a free lunch.** It keeps tsim = **1.000 everywhere** — even at 16 blocks against
  a gt that respects no blocks — *and* improves recovery monotonically (random gt: 0.856 → **0.986**; block gt:
  0.676 → 0.895). A soft locality penalty can't cost fidelity (it's a penalty, not a constraint), and it acts
  as an extra symmetry-breaker on top of L1. **Beats E2's best prior (L1 alone, 0.900).**

Rule of thumb: **use soft hierarchy to fit, use hard hierarchy to test.**

### FINDING 6 — a 1% identity ridge completely undoes FINDING 3's blindness

FINDING 3 said a data-matched Λ goes blind off-distribution. It's the *knob* that matters. Sweeping
`Σ_t = (1−t)·Σ_data + t·I` (data on a 6-dim subspace of R^16; scored under the full-support metric + an OOD probe):

| t | TRUE tsim | MSE(OOD) | gt-recovery | |
|---|---|---|---|---|
| 0.00 | 0.196±0.048 | 0.807 | 0.251 | data-matched — **BLIND** (the handoff's recipe) |
| **0.01** | **0.985±0.009** | **0.015** | 0.813 | **1% ridge — essentially fixed** |
| 0.05 | 0.999±0.000 | 0.001 | 0.861 | |
| 0.20 | 1.000±0.000 | 0.000 | **0.882** | |
| 0.50 | 1.000±0.000 | 0.000 | 0.857 | |
| 1.00 | 1.000±0.000 | 0.000 | 0.856 | full-support — safe |

The tradeoff I warned about in Tick 2 **barely exists**: you do not have to choose between data-realism and
off-distribution coverage. `Σ + εI` with **ε ≈ 0.01–0.05** keeps the data covariance's shape while restoring
essentially all of the global guarantee (TRUE tsim 0.196 → 0.985 → 0.999). Recovery even *peaks* at t=0.2
(0.882), slightly above pure identity. **Practical recipe: always ridge the metric's Σ; ε=0.05 is a safe default.**

---

## Tick 5 — DEPTH: the degree-4 metric, and hierarchy as something you *measure*, not impose

`tensor_sim_deep.py` extends the metric to a **stack of two bilinear layers**. Two layers is degree-4 in `x`:
`y_k = Σ_g D2_kg (xᵀQ_g x)(xᵀP_g x)`, so `⟨A|Λ|Â⟩` needs `E[∏ of 4 quadratic forms]` — a sum over the 15 set
partitions of {1..4}, each block contributing a joint cumulant `κ(k) = 2^{k-1} Σ_{(k-1)! cyclic orderings}
tr(∏ M_i)`, `M_i = A_iΣ`. Implemented for **general n** (so degree-2 × degree-4 cross terms come free) and
**verified against Monte Carlo** before use — closed==MC to 3.9e-4 (Σ=I) / 1.8e-2 (Σ=SPD), `L_fid == E‖y−ŷ‖²/E‖y‖²`,
`L_fid(A,A)=0` exactly. The vectorised variant used for training was re-checked against the reference (rel **0.0**)
and MC. *A whole tick's conclusions ride on this formula; none of it is trusted unchecked.*

### FINDING 7 — depth is provably necessary, and the metric prices it exactly (`e5_hierarchy_via_depth.py`)

Target: `x∈R^12` split into 4 disjoint groups of 3 → each group is squeezed into **one mid-level feature** →
layer 2 mixes them densely → `y∈R^3`. A **flat** (1-layer, degree-2) transcoder has a hard fidelity ceiling:

| flat rank | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| tensor-sim | 0.593 | 0.613 | 0.616 | 0.618 | **0.618** |

**Rank buys nothing** (0.618 at rank 4× the input dim). A degree-2 model cannot represent a degree-4 target at
any width: this is a **property of the function class**, and `L_fid` computes it in closed form with no data.
This is the quantitative version of "you need depth to get composition."

### FINDING 8 (headline) — hierarchy WIDTH IS A SPECTRUM, read off the `tsim(dz′)` curve (`e5b_...py`)

The natural conjecture — *"the smallest bottleneck `dz′` holding tsim=1 is the number of mid-level features"* —
**FAILED as first run**: tsim was already 0.954 at `dz′=1`. Diagnosis (not a bug, a fact): `L_fid` is a
**relative-norm** error, and the first ground truth had wildly unequal mid-level feature scales, so one `z_j`
dominated `E‖y‖²` and one squared quadratic form already captured 95% of it. So the sweep does not report an
integer — **it reports a spectrum**, weighted by each sub-feature's contribution to the output. Rebuilt the gt
with an explicit scale ladder over the `z_j` (everything else identical) and swept `dz′`:

| `dz′` | **BALANCED** (4 equal sub-features) | **SKEWED** (1, ½, ¼, ⅛) | **ONE-FEATURE** gt (control) |
|---|---|---|---|
| 1 | 0.529±.111 | 0.799±.179 | **1.000±.000** |
| 2 | 0.758±.130 | 0.943±.052 | 1.000 |
| 3 | 0.973±.005 | 0.996±.004 | 1.000 |
| **4** | **1.000±.000** ← true width | 0.999 | 1.000 |
| 5–6 | 1.000 | 1.000 | 1.000 |

**The curve is a scree plot for mid-level features.** Where it saturates = the *effective* number of
sub-features; how sharply it turns = whether they matter equally. A balanced 4-feature hierarchy gives a **sharp
knee exactly at 4**. A skewed one gives a **graded ramp** — correctly, because that computation really is mostly
one sub-feature plus corrections. The control that had to work does: a target that truly needs one mid-level
feature reports **1.000 at `dz′=1`**, so the sweep is measuring structure, not rewarding capacity.

This is the depth analogue of FINDING 5 (hard masks as a probe), and it is **strictly better**: a block mask
presupposes *which coordinates group together*, whereas the bottleneck sweep presupposes nothing — the grouping
is discovered. Confirmed by adding L1 (E2's winner) at `dz′=4` on the balanced gt:

| λ_L1 | 0 | 0.001 | 0.003 | 0.01 | **0.03** |
|---|---|---|---|---|---|
| tensor-sim | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** |
| group purity of recovered layer-1 features | 0.466 | 0.478 | 0.502 | 0.594 | **0.813** |

*(chance 0.328; planted gt 1.000)* — sparsity recovers **which input coordinates form each sub-feature**, at
**zero fidelity cost**, never having been told the groups exist.

### E4 (`e4_multilayer_hierarchy.py`) — cross-layer block priors: works, but is the weaker tool

Fitting a 2-layer transcoder to a cross-layer "tree" gt (blocks confined at *every* layer), data-free:

| arm | TRUE tsim | MSE(OOD) | layer-1 feature recovery |
|---|---|---|---|
| MSE (subspace data) | 0.181±.112 | 0.810 | 0.385 |
| deep `L_fid` (dense) | 0.997±.003 | 0.003 | 0.559 |
| deep `L_fid` + L1 | 0.995±.003 | 0.004 | 0.558 |
| **deep `L_fid` + cross-layer hierarchy** | 0.966±.036 | 0.037 | **0.720±.054** |
| *chance (random init)* | | | *0.434* |

FINDING 3 replicates at depth (MSE on subspace data → true tsim **0.18**, OOD **0.81** — blind), the degree-4
`L_fid` fixes it data-free (0.997), and the cross-layer prior buys the best factor recovery. And FINDING 5's
diagnostic replicates at depth: the same hard cross-layer prior scores **0.966** on a truly-hierarchical target
but **0.543** on a gt that respects no blocks — it betrays a mis-specified structure. Still, prefer the
bottleneck sweep: it discovers the structure instead of assuming it.

### The synthesis (Tick 5)

**Hierarchy is not a prior you impose — it is a measurement you make, and depth is the instrument.**
- To **fit**: soft/graded penalties (L1, soft locality) — they never cost fidelity and they break CP
  non-uniqueness (FINDING 5).
- To **test/discover**: hard constraints — a block mask (FINDING 5) or, better, a **bottleneck sweep**
  (FINDING 8) — because a *correct* structural hypothesis costs no fidelity and a *wrong* one must break it.
- Depth converts "is this layer hierarchical?" from a qualitative question into a **curve**, computable with
  **no data at all**.

---

## Tick 6 — the Pareto frontier (`e6_pareto.py`) and the FIRST REAL-MODEL number (`e7_real_layer_rank.py`)

### FINDING 9 — the program's thesis, stated as cleanly as it can be stated

Sweep BatchTopK `k` × the fidelity weight `λ`, on the honest regime (data on a 6-dim subspace of R^16; Λ
full-support). Reported both ways, because the closed form scores the transcoder's *dense tensor* while the
deployed model is *gated* — so `gated-sim` is a Monte-Carlo global fidelity of the **actual gated model** over
the full space. (Stating the confound and then measuring it away, rather than quoting only the flattering one.)

`k = 32` (= rank, so the **gate is a no-op** — no confound at all, the cleanest row in the program):

| λ | MSE(in) | tensor-sim | gated-sim | gt-recovery |
|---|---|---|---|---|
| **0 (MSE-only)** | **0.000** | **0.079±.031** | **0.084±.029** | 0.132 |
| 0.1 | **0.000** | **1.000±.000** | **1.000±.000** | **0.647±.033** |
| 1 | 0.000 | 1.000 | 1.000 | 0.506 |
| 10 | 0.000 | 1.000 | 1.000 | 0.444 |

**An MSE-only transcoder reaches PERFECT reconstruction (0.000) while its true global fidelity is 0.08 — i.e.
essentially ZERO.** It has learned a function that agrees with the layer everywhere the data goes and is
unrelated to it everywhere else. Adding the fidelity term with λ=0.1 takes tensor-sim 0.079 → **1.000** and
gt-recovery 0.132 → **0.647** (chance 0.066) **at no MSE cost whatsoever**. There is no tradeoff to trade off.

Same story at every sparsity level (MSE-only tensor-sim is ~0 throughout: −0.061, −0.060, 0.024, 0.037, 0.079
for k = 1, 2, 4, 8, 32), and the fidelity term always fixes it:

| k | MSE(in), λ=0 | tensor-sim, λ=0 | tensor-sim, λ=0.1 | gated-sim, λ=0 → 0.1 |
|---|---|---|---|---|
| 1 | 0.085 | −0.061 | 0.999 | 0.005 → 0.594 |
| 2 | 0.031 | −0.060 | 0.999 | −0.010 → 0.751 |
| 4 | 0.005 | 0.024 | 0.999 | 0.031 → **0.906** |
| 8 | 0.001 | 0.037 | 0.999 | 0.042 → **0.974** |
| 32 | 0.000 | 0.079 | 1.000 | 0.084 → **1.000** |

The `L_fid`-ONLY arm (no data at all) hits tensor-sim **1.000 at every k**, and its *gated* fidelity is limited
only by how much the gate throws away (0.537 at k=1 → 1.000 at k=32) — i.e. the gap between "the tensor is
right" and "the deployed sparse model is right" is **entirely the gate's cost**, not a failure of the fit.
Cost of sparsity, priced honestly: k=4 keeps 0.845 of the layer, k=8 keeps 0.948.

### FINDING 10 (honest negative) — a REAL bilinear MLP is NOT low-rank under an isotropic metric

First real-model measurement: the L8 bilinear MLP of a 500M 18-layer bilinear GPT (`r=4608`, `d=1152`,
`K=1152`). Fit rank-`r′` CP transcoders on `L_fid` alone — **no data, no forward passes, only weights**:

| r′ | 32 | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|---|
| r′/r | 0.007 | 0.014 | 0.028 | 0.056 | 0.111 | 0.222 |
| tensor-sim | 0.118 | 0.136 | 0.161 | 0.201 | 0.265 | **0.373** |
| +L1: tensor-sim | 0.118 | 0.136 | 0.160 | 0.201 | 0.265 | 0.373 |
| +L1: eff-L0/row | 713 | 721 | 728 | 730 | 732 | 732 |

*(control: random-init tensor-sim = −0.000 = chance)*

**No compression.** At 22% of the layer's own rank we recover only 37% of it, the curve is smooth and roughly
linear in `r′` (no knee — contrast FINDING 8's toys), L1 changes **nothing** (identical to 3 decimals) and the
factors stay dense (eff-L0 ≈ 730 of 1152). Under this metric the real layer looks close to **incompressible and
unstructured** — the opposite of every toy here.

**But the honest reading is that this is a statement about Λ, not (yet) about the layer.** Λ = N(0,I) demands
the transcoder match the layer *equally in all 1152 residual directions*, including the overwhelming majority
the model never visits — real residual streams are violently anisotropic. FINDING 3 said don't use a
data-matched Σ; FINDING 6 said the fix is a *ridge*, `Σ_data + εI`, which needs the real Σ. **This run used
neither** — it used the ε=1 endpoint. So FINDING 10's correct claim is narrow: *a real bilinear MLP has no
low-rank structure with respect to isotropic inputs.* Whether it has low-rank structure **on its own data
manifold** is the open question, and it is the obvious next experiment: estimate `Σ_resid` from a text batch,
ridge it (ε≈0.05, FINDING 6), and re-run this exact sweep. If a knee appears, that knee is the layer's real
feature count. **Until then, no claim about real-model structure should be drawn from this program.**
