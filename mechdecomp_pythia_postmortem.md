# Pythia Tier-2 post-mortem: exactly what was run, and exactly where it failed

Companion to `results_mechdecomp.md`. Written because the summary line ("real dense
activations don't provide W-aligned sparsity") compresses five distinct runs and two
different *kinds* of failure into one sentence. Here is the full account.

---

## 0. First, a correction to the mental model

> *"We have the first reconstruction map part, then the SAE on those to separate them."*

**There is no two-stage pipeline.** The method is a *single* alternating optimization.
There is no "reconstruct first, then run an SAE on the reconstruction."

The objective (spec §1.3) is one loss over a dictionary `D = [d_1 … d_m]` and codes `C`:

```
L(D, C) = Σ_i ‖ ( Σ_j c_ij · (d_jᵀ x_i) · W d_j ) − W x_i ‖²  +  λ Σ_i ‖c_i‖₁
                  └──────── reconstruction of the MAP'S OUTPUT ────────┘
```

- `x_i` = one activation vector (an input to the layer W).
- `W x_i` = what the layer actually computes on it. **This is the reconstruction target** —
  not `x_i` itself. (An SAE would target `x_i`.)
- Each *mechanism* is the rank-1 matrix `W d_j d_jᵀ`; the atom `d_j` is an **input-side**
  direction, `W d_j` is its output-side image.
- Sparsity + shared dictionary is the "SAE-like" ingredient, but it is applied **to the
  map's action**, one level up from activations. That is the entire novelty.

Optimization alternates:
- **E-step** (`mechdecomp/estep.py`): codes `C` given `D` — a per-datapoint non-negative
  lasso, solved with the Hadamard-Gram factorization `G_i = (a_i a_iᵀ) ⊙ G_W` where
  `a_ij = d_jᵀ x_i` and `G_W = (WD)ᵀ(WD)`. Vectorized coordinate descent.
- **M-step** (`mechdecomp/mstep.py`): dictionary `D` given `C` — exact per-atom update
  (solve `W d = r̄` for the residual this atom must fit, renormalize `‖d‖=1`).

So when I say "it failed," the question is always: *failed in the E-step, the M-step, or
in the science after both converged?* All three happened, at different sites.

---

## 1. What was run: modules, data, code

Data for every Pythia run: **`NeelNanda/pile-10k`**, streamed, ≤128 tokens/doc, 160–250
docs, then a random subsample of token positions. Activations captured with forward hooks.

| # | Model | `W` (the map being decomposed) | Shape / rank | `X` (the data) | Hook site |
|---|---|---|---|---|---|
| 1 | Pythia-70m | L3 MLP **down-projection** `layers[3].mlp.dense_4h_to_h.weight` | (512, 2048), **wide**, 1536-dim null space | post-GELU MLP hidden (the down-proj's own input) | `forward_hook` on `dense_4h_to_h`, take `inp[0]` |
| 2 | Pythia-70m | L3 attention **OV of head 2** = `W_O[:,h] @ W_V[h,:]` | (512, 512), **rank 64** | **residual stream** entering L3 | `forward_pre_hook` on `layers[3]` |
| 3 | Pythia-410m | L6 attention **OV of head 3** | (1024, 1024), **rank 64** | residual stream entering L6 | `forward_pre_hook` on `layers[6]` |
| 4 | Pythia-410m | L6 **full attention output** (all 16 heads' OV concatenated) | (1024, 1024), **rank 1016** | residual stream entering L6 | same |
| 5 | Pythia-410m | *(none — baseline)* vanilla ReLU **SAE on `x` directly** | m=4096 | same residual stream | same |

Everything ran through `mechdecomp/{estep,mstep,objective,tier0}.py` with `svd_init`
(top right-singular vectors of `WX`) as the dictionary initializer.

---

## 2. Run-by-run: where each one broke

### Run 1 — MLP down-projection (wide W). Failed in the **M-step**, then failed the science.

**First attempt: numerical divergence.**
```
Pythia-70m L3 down-proj: W (512, 2048), X (2048, 30000)
  m 1024 lam 0.02  : R2 -11,990,633
  m 1024 lam 0.005 : R2 -1.27e15
  m 2048 lam 0.01  : R2 -3.20e11
```
Codes exploded. **Root cause, precisely:** the exact M-step solves `W d = r̄` by
`lstsq(W, r̄)`. `W` is 512×2048, so this system is **underdetermined** — a 1536-dimensional
null space. Atoms drift into `null(W)`, where they change `d_jᵀ x` (the gate) while leaving
`W d_j` (the output) untouched. Two atoms can then have nearly identical `W d_j` and grow
with mutually cancelling codes. This is textbook **CP degeneracy**, which the spec §1.5
explicitly warns about — and which the toys never exposed, because their `W` was square and
full-rank (trivial null space).

**Fixes applied:** elastic-net ridge on the E-step Gram diagonal + a row-space-regularized
M-step. Divergence stopped:
```
  m 1024 lam 0.02 ridge 0.02: R2 0.3959 L0 21.1
```

**Then it failed the science.** With a *stable* solve, compare to a dense low-rank baseline
(closed-form optimal rank-r map from `mechdecomp/closed_form.py`, no sparsity at all):

| model | R² | L0 |
|---|---|---|
| closed-form **rank-21 dense** | **0.551** | — |
| closed-form rank-50 dense | 0.724 | — |
| masked-projector, m=1024 | 0.489 | 66 |

The method used **66 active atoms per datapoint** and still reconstructed *worse* than a
plain 21-dimensional dense map. And its atoms' top-activating tokens were incoherent
grab-bags (`' from' ' Thus' ' going' '87'`). **Fails both Tier-2 soft criteria** (competitive
reconstruction; interpretable atoms).

*Why reconstruction is poor here:* `X` is post-GELU MLP hidden — dense and high-rank
(top-40 PCs hold only ~55% of variance). Reconstructing `Wx` for dense `x` needs roughly
effective-rank-many active atoms; sparse codes cannot do it. This is not a bug, it's a
mismatch between the objective's sparsity premise and the data.

---

### Run 2 & 3 — attention OV head (low-rank W). **Reconstruction succeeded**, atoms still failed.

Switching the site to a rank-64 OV map, with `X` = the **residual stream** (where SAE
features are believed to live):

| run | R² | L0 |
|---|---|---|
| Pythia-70m, L3 OV head 2 | **0.9689** | 23.6 |
| Pythia-410m, L6 OV head 3 | **0.9897** | 46.9 |

Reconstruction is now excellent — 24 sparse atoms capture 97% of the map's action. So the
Run-1 failure really was *site* (wide W + dense hidden), not the method.

**But an intermittent divergence appeared.** The ablation-faithfulness run, on the *same
code that had just produced 0.99*, blew up:
```
faith410.log : R2 -37,655,420,928   L0 44.3
```
Same disease, different geometry: OV is rank-64 in 1024-dim, so it has a **960-dimensional
input null space**. The ridge only *stochastically* suppressed the wandering (the
`resample_dead` reinit is random). **This retroactively meant the earlier 0.99 runs were
lucky draws**, and any faithfulness/interpretability number computed on them was untrustworthy.

**Principled fix** (`mechdecomp/mstep.py::rowspace_basis`): project every atom onto
`row(W)` after each M-step, making null-space wandering *impossible by construction* rather
than penalized. Verified across seeds:
```
robust410.log: seed 0: R2 0.9907  seed 1: R2 0.9913  seed 2: R2 0.9914   (L0 ≈ 46)
```

**Now, on a trustworthy decomposition, the science failed.** Ablation faithfulness — remove
atom `j` from the map (`W ← W − (W d_j) d_jᵀ`), measure `‖ΔW·x‖` on datapoints where the
atom is active vs inactive:

| | median concentration | frac > 3× | frac > 5× |
|---|---|---|---|
| **Tier-0 toys** (clean features) | **32–64×** | 100% | 100% |
| **Pythia-410m OV atoms** (508 live) | **2.6×** | 18% | 4% |

An atom's ablation barely localizes to its own active contexts. Context-window
interpretability of even the highest-faithfulness atoms: incoherent.

---

### Run 4 — full attention output (high-rank W). Reconstruction collapsed.

Hypothesis from a synthetic control: a *full-rank* map should let atoms separate features
(a rank-64 row space is too narrow to hold many distinct features). So decompose all 16
heads' OV at once — rank 1016 instead of 64:

```
full-attn OV map rank 1016 (d=1024)
R2 0.3513  L0 74.0
ABLATION FAITHFULNESS: median 1.1×   (single head was 2.6×)
```

**Worse on both axes.** And this is the moment the whole picture unified: the high-rank map
*could* separate features in principle, but now sparse codes can't reconstruct its action on
dense activations — the exact Run-1 failure. There is a scissors:

| | low-rank W (1 head) | high-rank W (full attn) |
|---|---|---|
| reconstruction | ✅ R² 0.99 (small output space) | ❌ R² 0.35 (dense data) |
| feature separation | ❌ 64-dim row space too narrow | ✅ in principle |

**No single-map site on Pythia gives both.** The synthetic control succeeded only because
its data was sparse *by construction*.

---

### Run 5 — SAE baseline. ⚠️ **An overclaim I need to retract.**

I trained a vanilla ReLU SAE directly on the same Pythia-410m L6 residual:

```
vanilla (3k steps, m=4096) : R2 0.9989  L0 1918.3
tuned (12k steps, annealed λ, unit-norm decoder) : R2 0.9997  L0 1891.9
```

L0 ≈ 1900 out of 4096 means **it never became sparse** — half the features fire on every
token. I wrote that "a matched-compute SAE also fails to sparsify," implying this was
evidence about *Pythia's data*.

**That inference was wrong, and later evidence refutes it.** When I loaded a *properly
trained* pretrained SAE (`gpt2-small-res-jb`, sae_lens) on GPT-2's residual stream, it
achieved **R² 0.9928 at L0 56.2** — real sparsity, clean features. So a well-trained SAE
*does* sparsify a real LM's residual stream. My Pythia SAE failure was a **training-quality
artifact** (SAE training is notoriously finicky — needs ghost grads, resampling, long runs),
**not** evidence that Pythia's activations lack feature-sparsity.

**Corrected status:** we have *no* evidence that Pythia-410m's residual lacks sparse
features. The honest statement is that I never established the data property either way on
Pythia. The definitive test therefore had to move to a model with a *known-good* SAE — which
is exactly what the GPT-2 run did.

---

## 3. Failure taxonomy

Two categorically different things went wrong, and conflating them would be a mistake:

**(A) Numerical / optimization failures — all in the M-step, all fixed.**

| symptom | cause | fix |
|---|---|---|
| R² → −10¹⁵ (wide W) | underdetermined `lstsq(W, r̄)`; atoms wander in `null(W)` → CP degeneracy | E-step ridge + row-space M-step |
| *intermittent* R² → −10¹⁰ (low-rank W) | 960-dim input null space; ridge only stochastically suppresses | project atoms onto `row(W)` (impossible by construction) |
| Tier-0 recovery regressed 0.999 → 0.945 | the ridge added for wide-W perturbs the *exact* solution on full-rank W | revert ridge; use **pinv** M-step (exact full-rank, stable rank-deficient) |
| codes all zero (GPT-2) | activations un-normalized (‖x‖ ≫ 1), λ kills every code | RMS-normalize `X` to ‖x‖ ≈ √d |
| hang at m=2048 | exact M-step is a Python loop over atoms | vectorized gradient M-step for large dictionaries |

These are engineering. They cost most of the debugging, and each one silently invalidated
prior numbers until caught (hence the discipline: **re-run Tier-0 gates after every solver
change** — that's how the regression above was found).

**⚠️ SECTION SUPERSEDED (2026-07-09, after Logan's review).** The "structural failure" reading below
is RETRACTED. Three of its planks are now measured artifacts: (i) the load-bearing GPT-2 run was in the
dense regime (L0 294) with a degrading solver, while the feature basis lives at L0≈56 — at matched
sparsity the feature basis DOMINATES (0.777 vs 0.61-0.66) and OMP reaches R² 0.90; (ii) the atoms were
compared to res-jb SAE features across an attention block and a LayerNorm (site mismatch); (iii) the
weak ablation localization (2.6×) is what the rank-1 weight edit MUST produce for any thresholded
feature — it equals the raw-gate ratio and is independent of atom quality. See results_mechdecomp.md,
sections "★ DECISIVE: matched-L0" onward.

**(B) The structural failure — not fixable by better optimization.**

Once the solver was sound, the atoms still didn't correspond to features. The final GPT-2
test (a model whose residual is *proven* sparse: its SAE hits L0 56 / R² 0.99) measured this
directly, against 24,576 validated SAE feature directions:

| atoms | median max-cos to any SAE feature | frac > 0.5 |
|---|---|---|
| svd-init top-64 (clean, R² ≈ 1) | 0.229 | **0%** |
| svd-init top-512 | 0.153 | 0% |
| trained atoms (512) | 0.162 | 0% |
| **random directions (baseline)** | **0.149** | 0% |
| *scale reference:* SAE features' own inter-feature overlap | 0.539 | — |

Everything sits at the random-direction floor. A genuine feature match would read ≥ 0.5.

**Interpretation.** The objective ties each atom to reconstructing `Wx`. The directions that
efficiently reconstruct a linear map's action are its **principal / singular directions** —
dense, orthogonal, PCA-like. Sparse features are neither. So the method converges to the
wrong *kind* of object, regardless of how well it is optimized. This is structural.

---

## 4. So what actually failed, in one paragraph

On Pythia we ran the masked-projector decomposition at three sites (wide MLP down-proj on
GELU hidden; single rank-64 OV head on the residual; full rank-1016 attention output on the
residual). The **M-step** broke first, twice, from null-space-driven CP degeneracy on
rank-deficient maps — fixed by row-space projection and a pinv update. With a numerically
sound solver, the **science** then broke in a site-dependent way: the wide/high-rank maps
can't be sparsely reconstructed on dense activations (R² 0.35–0.49, below a dense rank-21
baseline), while the low-rank map reconstructs beautifully (R² 0.99) but its 64-dimensional
row space can't hold separable features (ablation localization 2.6× vs the toys' 32–64×). No
site gives both. The SAE baseline I ran alongside was under-trained and proves nothing about
Pythia's data — a claim I have now retracted. The decisive test moved to GPT-2 + a pretrained
SAE, where the data's sparsity is established, and there the atoms landed at the
random-direction floor (max-cos 0.16 vs 0.15) — showing the failure is **structural**: the
objective recovers the map's principal directions, not its sparse features.

---

## 5. What would have to change

- **Not** more optimization effort. The clean `svd-init` (R² ≈ 1) is already at the
  random floor w.r.t. features; a better solver converges *harder* to principal directions.
- The objective needs a term that **biases atoms toward sparse-feature directions** rather
  than variance-explaining ones. Candidates: penalize atom *density* / reward non-Gaussianity
  of `d_jᵀ x` across the corpus (an ICA-flavoured criterion); or tie atoms to an
  activation-side sparse code (a hybrid: SAE features as the dictionary, `W`-relevance as the
  weighting) — which would recover the spec's stated goal ("feature discovery weighted by
  mechanistic relevance") without asking reconstruction of `Wx` to *find* the features.
- The parts that **do** work and are worth keeping: the closed-form rank-r theorem
  (`closed_form.py`, unit-tested), exact toy recovery, and the **contraction-based circuit
  analysis** — which recovered our causally-verified `L0H3 → L1H2` induction edge once scored
  by *selectivity* rather than magnitude (see `results_mechdecomp.md`, Tier 1.5).

---

## 6. Reproduce

```bash
cd /workspace/tensor_language && source /venv/main/bin/activate

python tests/test_closed_form.py            # Tier 0.1 theorem (float64) — PASSES
python -m mechdecomp.tier0                  # toy recovery + ablation gates — PASSES
python -m mechdecomp.tier2_pythia           # Run 1 (MLP down-proj)  [needs guards]
# Runs 2-5 & the GPT-2 test were driven from logs/: tier15.log, robust410.log,
# faith410b.log, pythia_fullattn.log, sae_proper.log, gpt2_compare.log
```
Raw stdout for every run quoted above is preserved in `logs/`.
