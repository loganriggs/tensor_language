# Mechanism decomposition — what is actually true

**Read this instead of `results_mechdecomp.md`.** That file is a chronological log: retracted claims sit
next to surviving ones, and reading it linearly will mislead you.

*Rewritten 2026-07-10; supersedes all earlier versions.* Every numeric claim carries its **provenance** —
how many seeds, what statistic, which control could have falsified it.

**Standing rule, adopted after three reversals:** no single-seed, top-k, or max-over-sample statistic may
be reported as a finding. Medians, full distributions, rank tests, >=3 seeds.

---

## 1. What the method is

Given a linear map `W` and activations `x`, learn a dictionary `D` so `W x` is reconstructed by a sparse
set of rank-1 mechanisms `W d_j d_j^T`. The premise: optimising for the map's **action** (`Wx`) yields
better atoms than optimising for the **activations** (`x`), as a plain SAE does.

---

## 2. Established, with controls

### 2.1 The theory is correct where it was proved
Closed form (§1.1) verified in float64: matches the optimal value, beats Adam (529.52 vs 1037.97),
recovers `W P_S` to 1.8e-10 relative error, solves the counterfactual-edit variant. §1.2's negative
result reproduces. *Deterministic tests against exact identities.* **Solid.**

### 2.2 On synthetic data with a known generator, the method recovers it
`X = D_true C + noise`. Recovery (mean over true atoms of max |cos| to a learned atom) reaches
**0.96–0.98** against a **measured** chance level of 0.32–0.37, for K/d from 1 to 4. The 4x-overcomplete
case needs 60 rounds, not 20 (convergence, not failure). *Guard: init at `D_true` must be a fixed point —
it is (held-out delta -0.0002, cos 0.9998).* **Solid.**

### 2.3 The objective is learnable on real maps
Pythia-410m L3 `down_proj`, held-out, D frozen: R2(`Wx`) = **0.4591 +/- 0.0003** (3 seeds) vs random
**0.0887 +/- 0.0018**. It beats every matched-L0 floor: PCA-32 0.3509, PCA-64 0.4231, OMP-32-over-PCA
0.3897 (learned 0.6020 at K=2048, k=32). *Floors that could have won and didn't.* **Solid.**

### 2.4 The spec's premise holds: `Wx` beats `x` — modestly
Matched activations, K=1024, k=16, identical probe indices, 3 seeds, vs a **strengthened** (8k-step) SAE
so the baseline is not handicapped:

| dictionary | R2(`Wx`) | irrepl median | purity median |
|---|---|---|---|
| **masked-projector** | **0.4591 +/- 0.0003** | **0.000134** | **0.100** |
| SAE-8k (x-only) | 0.4291 +/- 0.0042 | 0.000016 | 0.050 |
| random | 0.0887 +/- 0.0018 | 0.000028 | 0.050 |

Mann-Whitney on pooled per-atom losses: SAE vs MP `z = -8.88`; on purity `z = -6.16`. Measured chance
purity 0.070. The SAE's *median* atom is more replaceable than a random direction.
*3 seeds, medians, rank tests.* **Solid.**

**The two objectives do different things.** `Wx` spreads the map's action uniformly across atoms
(uniformly irreplaceable, uniformly mildly pure; selection effect 1.4x). `x` concentrates structure in a
minority (mostly replaceable and impure, but a crisp tail — top-10 purity 0.477, selection effect 4.3x).
Neither dominates: pick by whether you need a faithful basis or a few clean features.

### 2.5 The atoms are NOT demonstrably mechanisms
The central negative — and it survived the standing rule while three positive claims did not.

Resolved with a **usage- and R2-invariant** statistic (open problem #1, closed 2026-07-10):

> **uniqueness** = `loss / (base_R2 / K)` = the fraction of an atom's fair share of explained variance
> that is uniquely its own. Usage is `k/K` for any dictionary, so equal-`k` is usage-matched; base R2
> is divided out. The metric is **not** K/d-invariant, so it was re-validated at the real proportions.

Power check at the **real** K/d = 0.25 and rank/d_in = 0.25 (exactly Pythia's 1024x4096, K=1024):
TRUE generator **0.9558** vs random **0.1961** (4.88x). At K/d=1.0: 0.7933 vs 0.1860.

Pythia `down_proj`, 3 seeds, medians, Mann-Whitney on pooled per-atom uniqueness:

| dictionary | median uniqueness | vs MP |
|---|---|---|
| **masked-projector, k=16** | **0.2990** | — |
| random, k=16 (equal sparsity) | 0.3280 | z = +0.95, **n.s.** |
| random, k=96 (matched R2) | 0.2717 | z = -1.12, **n.s.** |
| SAE-8k, k=16 | 0.0448 | z = -8.59, lower |
| *(TRUE generator at this K/d)* | *0.9558* | |

**CORRECTED 2026-07-10 (next tick).** The table above was measured at K/d=0.25 with only 12.5
firings/atom. Redone with N_EVAL=4000 and probes restricted to atoms firing >=5 times:

| K/d | MP uniq | random uniq | ratio | MW z |
|---|---|---|---|---|
| 0.25 | 0.2597 | 0.2248 | 1.16x | -1.11 (n.s.) |
| 0.50 | 0.3347 | 0.2193 | 1.53x | -1.81 (marginal) |
| **1.00** | **0.2711** | 0.1519 | **1.79x** | **-3.33 (significant)** |
| *(TRUE generator at K/d=1)* | *—* | *—* | *4.06x* | |

**At or above critical completeness the MP atoms ARE significantly more unique than random (1.79x) —
but less than half as unique as a known generator (4.06x).** Below critical completeness the advantage
vanishes. The SAE is *more* redundant than random (correlated decoder directions).

*Measurement caveat now enforced: with sparse codes `usage = k/K`, so the eval set must scale with K or
per-atom medians are taken over atoms that never fire. A K=8192 run at N_EVAL=600 gave 1.2 firings per
atom and produced pure noise.*

*Confound chain closed: usage matched by construction; R2 normalised out and separately matched; K/d
matched, with the metric's power re-validated by a known generator at that K/d.*

### 2.6 Single-atom ablation is not a valid test of dictionary quality
On the toy where `D_true` **is** the generator, single-atom ablation separates true from random by only
**1.54x (mean) / 1.00x (top1)**; the conditional variant 1.57x. It cannot detect localization that
provably exists: with codes fixed, deleting any used atom removes ~`1/k` of the reconstruction whether or
not it is a real factor. **Irreplaceability** (drop the atom from the *dictionary*, re-select codes)
separates them 8.4x, and 34x under matched R2. *This invalidates the common "ablate a feature, watch
behaviour" move as evidence of dictionary quality.*

### 2.7 Circuit read-off in a gated (bilinear) layer is data-conditioned
`ds = a1*b2 + a2*b1 - b1*b2`: each branch is cross-weighted by the other's score, so §1.5's per-branch
`d_k^T W1 d_j` is exact only for **stacked linear maps**. Empirically the per-branch contraction is
ill-posed — under `|G|` it names L0H0 for K1 and L0H3 for K2; under signed selectivity, L0H3 and L0H2. No
variant identifies the causal head in both branches, and the answer changes with the statistic.

### 2.8 Identifiability is bounded by `row(W)`
Atoms are recoverable only up to their `row(W)` component: `cos(true, identifiable) = sqrt(rank/d_in)`,
verified to 4 decimals (0.4992 at 256->64, 0.5000 at 2048->512). Pythia `down_proj`: **0.50** — 75% of
every atom is invisible to `W`. GPT-2 OV: 0.993.

---

## 3. Retracted, with cause

| claim | cause |
|---|---|
| "Structural failure — objective can't be sparsified on real models" | measured on a failed optimization (degraded M-step, L0~294, never sparse) |
| "Atoms sit at the random-cosine floor vs SAE features" | site mismatch (post-ln2 vs `resid_pre`) + failed run; max-cos promoted into an oracle the spec declined to make decisive |
| identifiability table (41/23/12% overlap) | compared at L0 40/77/160; overlap falls mechanically with L0 |
| "SAE features are the reference for identifiability" | the SAE's own support is **not** the reconstruction optimum at its own L0 (OMP 0.9328 vs OLS-on-SAE-support 0.8807, 23% overlap) |
| "atoms are 12x more causal than random" (`down_proj`) | max-over-sample across two non-identical runs; does not replicate |
| Tier-1.5 gate PASSes (4.07x, 8.81x, rank test) | **a random dictionary passes them, with a larger margin (25.59x)** |
| "the attn2 OV mechanism is distributed / nothing sparse to find" | measured with single-atom ablation, which has no power (§2.6) |
| "reconstruction and localization are independent properties" | the localization axis was never validly measured |
| "attention maps are not decomposable" | pythia `attention.dense` gap +0.606, GPT-2 OV +0.365 |
| "the decomposability pre-test separates the regimes" | confounded by `K/rank(W)` (gap 0.0026 -> 0.0271 as K/rank 16 -> 1) |
| "weight-aware does NOT beat an SAE" | single-seed mean + max; reverses under 3 seeds and medians |
| "purity tracks the selection rule, not the dictionary" | single-seed top-10; the dictionary effect is significant (z = -6.16) |
| "atoms are weakly mechanism-like (1.84x matched-random)" | single-seed **mean**; seeded medians give 1.09x, n.s. |
| "release-D: the optimizer wanders" | inline M-step was stale-residual Jacobi + renormalized with codes frozen; it degraded a **known-good** dictionary |
| §1.3 free-ablatability of a thresholded feature | 1.89x concentration = the raw-gate ratio exactly; never evidence about atoms |
| §1.5 contraction read-off for gated layers | ill-posed (§2.7) |
| "matched-compute SAE also fails to sparsify Pythia" | my SAE was under-trained (L0 1918/4096) |
| "selectivity beats magnitude for circuit scoring" | wrong write vector (OV applied to the position's own residual) |

**Program-A retractions forced by this work.** The copy-burst lever installs a **positional** copier (0.90
at the trained period 128; chance at 96/85/150/180; copies iff `P | 128` or `P | 129`, which predicts the
untrained P=129 -> 0.4013), not induction. `attn2-seed0` copies arbitrary repeated tokens at **1.2x
chance**, so its L1H2 is a repeated-bigram circuit, not an induction head. A **tiled random-period** burst
schedule does install genuine content-based induction (0.96-0.99 across trained periods, generalising to
untrained P=16/32), but formation is **stochastic** (mix 0.5: seed 0 hits at 0.7483, seed 1 misses at
0.0011) — so there is no mixture "threshold".

---

## 4. Proposed spec amendments

1. **§3 (E-step):** the L1 lasso is not deficient — it was mis-calibrated. At matched L0 it ties OMP on a
   clean toy (0.958-0.966 vs 0.960). Calibrate lambda to a **target L0 per dictionary**; prefer an
   L0-targeting solver only on coherent real dictionaries (GPT-2 OV: OMP 0.902 vs lasso+debias 0.665).
2. **§1.4:** state the `row(W)` bound (§2.8) before interpreting any wide map's atoms.
3. **§1.5:** restrict "circuit discovery = reading a matrix" to **stacked linear maps**. For gated layers,
   ablate the atom-reconstructed write and measure the change in the downstream score.
4. **§1.3:** delete. A static rank-1 weight edit cannot localize a thresholded feature.
5. **New — mandatory controls.** Every gate runs against (i) a **random unlearned dictionary** and, where
   a reconstruction gap is claimed, (ii) at **matched K/rank(W)**. Dictionary-quality claims use
   **irreplaceability**, never single-atom ablation. No single-seed / top-k / max statistics.

---

## 5. The honest bottom line

The `Wx` objective yields a basis that reconstructs a map's action far better than random, and better than
an activation-only SAE — on the map's action, on atom-level uniformity, and on typical-atom purity.
**Its atoms are only weakly mechanism-like:** at critical completeness they are significantly more
unique than a random basis (1.79x, z = -3.33) but less than half as unique as a known generator (4.06x);
below critical completeness the advantage disappears entirely.

**It is a better basis, and at best a weak feature finder.**

---

## 6. Open problems

1. ~~Can the matched-R2 control's usage confound be removed?~~ **RESOLVED 2026-07-10** by the
   uniqueness statistic (§2.5): usage is `k/K` for any dictionary, R2 divides out, K/d matched and
   power re-validated. Verdict: MP atoms are indistinguishable from random.
2. `attention.dense` and GPT-2 OV both have large reconstruction gaps; their **irreplaceability** under the
   matched-R2 control is unmeasured.
3. Gemma-2B tier hard-blocked: `GatedRepoError: 401`, no HF token configured.
4. The bilinear-`M_i` quartic has no matched-L0 solution (cold init collapses to `c=0`; warm init densifies
   to L0 839).
