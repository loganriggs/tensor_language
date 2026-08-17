# What the toys say about bilin18 — carries over, breaks, and what to test

Written after Part A (A1–A5) and B2. The question: these are 12-to-48-dimensional
planted-answer models; bilin18 is a 546M-parameter 18-layer language model. Which of
these results are about *bilinear layers* and therefore transfer, which are about *toys*,
and where do the toys quietly assume something bilin18 does not do?

Everything below is marked: **[measured here]** in the toy program, **[on record]** already
established about bilin18 elsewhere in this repo, **[inferred]** a consequence I argue for
but have not run, **[test]** a proposed measurement.

---

## 1. The architecture match is exact for the MLP, and not for the attention

Verified at source, not from documentation:

**MLP — exact match.** `jacclust/tt_model.py:188-195`, with `config.gated` false for bilin18:

```python
x = self.Left(x) * self.Right(x)
x = self.Down(x) + self.Down_bias
```

That is `y = D((Lx) ⊙ (Rx)) + b`, the toy layer, with no gate and no input biases. Width
`expansion_factor * n_embd = 4 * 1152 = 4608` (`tt_model.py:182-185`). So every A-series
object — the per-output interaction form `Q_i`, the gauge group, the Λ metric, the
canonicalisation, the block and component machinery — is *literally* the same object on
bilin18, not an analogy. The only additions are the `Down_bias` (an affine term, absorbed
by the constant coordinate exactly as in A4) and the affine-free RMSNorm on the input,
which is a positive per-position scalar and therefore commutes with the form.

**Attention — NOT what Part B built.** `tt_model.py:134-144`:

```python
pattern = (scores / D) * (scores2 / D)
pattern.masked_fill_(causal_mask.logical_not(), 0.0)
```

No softmax — the only softmax in the file is in `differential_attention`, a variant bilin18
does not use. So bilin18's placement is a **third** option that the plan's B0 does not
list: an unnormalised product of two score fields. B0 offers "logit-level product" (product
then softmax) and "post-softmax product" (product of two distributions). Part B implemented
exactly those two. Neither is bilin18.

This is not a detail. It changes what is even definable — see §3.

---

## 2. What carries over

### 2.1 Neurons are not the parts — and now there is a number for it

**[on record]** `qk_mdl/METHODS.md:26-27`: "Neuron permutation and rescaling leave `T`
unchanged… which is why we never do neuron-level interpretation." That is the gauge
argument, and it is correct.

**[measured here]** A3 adds a quantitative version: at 48 components in 16 input dimensions
**the reconstruction is perfect to machine precision (1.7e-32) while a third of the
recovered components are wrong.** A low fit residual is not evidence the parts mean anything.

**[corrected after review]** My first version of this section argued from `K/d` — bilin18 at
`4608/1152 = 4` being "twice" A3's breakdown around 2. That extrapolates along an axis A3
cannot resolve: A3 varied only the component count at fixed `m = 8, d = 16`, and the form
family's effective rank is exactly `m` at every count, so the breakdown is equally
consistent with a limit in `R/m`. The correct argument is Kruskal's bound applied to
bilin18's own shape (`THEORY.md` T6): for an `(m, d, d)` partially symmetric family,
uniqueness is guaranteed only to `R ≤ (m + 2d − 2)/2`, which at `m = d = 1152` is **1727**.

**[inferred]** bilin18's MLP has `R = 4608 ≫ 1727`. So any decomposition of a bilin18 MLP
into its 4608 rank-1 pieces sits far outside any uniqueness guarantee, and the binding
constraint is the *output* dimension, not the input one. The conclusion survives the
correction; the previous argument for it did not.

This bears directly on the registered next step "(b) estimation-noise-robust CP fitting"
(`STATUS_2026-08-10.md:57-60`). A3 says robustness to estimation noise is necessary but not
sufficient: even a noiseless, perfectly-fit decomposition at this ratio is not identified.
The fix has to come from a constraint that breaks the degeneracy (sparsity in a shared
dictionary across readers, symmetry, or a data metric), not from a better fitter.

### 2.2 The weight-space form is dense; the part that matters is small — already found, and A2 sharpens what to do about it

**[on record]** MA-1/MA-2 (`qk_mdl/LOG.md:2294-2316`): the exact per-direction quadratic
form `M_d` has effective rank **580–640 of 1152** in weight space, but the data-whitened
form `Σ^½ M_d Σ^½` collapses to effective rank **25–52**, and a rank-64 whitened form runs
live at ΔCE +0.028.

**[measured here]** A2-3 is the same phenomenon with a sharper instrument. Whitening
reweights directions by how much the data uses them; **canonicalisation** goes further and
removes the directions the data cannot constrain *at all* — the components of `Q` orthogonal
to the span of the realised lift `{vec(x xᵀ)}`. On modular addition that is 29–35% of the
mass, removed with a function change of 6e-12.

**[measured here]** And A1-2 shows why this is not bookkeeping: the unconstrained part is
causally live off-distribution. Two inputs the task calls identical came out 36–51% apart.

**[on record]** bilin18 already shows the matching symptom: "Coarsening layer-0 QK genuinely
helps off-distribution… All headlines are FineWeb" (`BILIN18_LAYERS_0_1.md:38-41`). A
component that helps when you delete it on one distribution and hurts on another is
precisely a direction the training distribution never constrained.

**[test]** Compute the identifiable fraction of each MLP's form: build the second-moment
operator of the lifted, rms-normed block input over a FineWeb sample, and report what
fraction of `‖M_d‖²` survives projection onto its support. My prediction is that it is
*much* lower than the toys' 65–71%, because the rms-normed residual stream at width 1152
occupies a small manifold inside the 664,128-dimensional `Sym²`. Cheap: one pass to
accumulate the moment, then a projection. If it is very low, then most of the reported
weight-space density of `M_d` is not a fact about the computation at all.

### 2.3 Weight-space surgery as a *test*, not just a description

**[measured here]** A2-4 is the result I would most want to try at scale. Projecting the
grokked model's forms onto the recovered block structure — a pure weight edit, no data, no
gradients — took held-out accuracy 0.978/0.934/0.953 → **1.0000** on all three seeds, and
the deleted remainder, run alone, was a lookup table (95% train, 0% test). A2-8 adds a
second such edit: imposing the task's input symmetry, again data-free, again to 1.0000.

The general shape: *if you have correctly identified the structure, deleting everything
else should not hurt and may help.* That is a much sharper gate than reconstruction error,
and bilin18 already has the harness for it — the ΔCE substitution gates in `tier2_model.py`
and `METHODS.md`.

**[test]** For each MLP, take the rank-64 whitened form that already runs at ΔCE +0.028
(MA-3) and ask the A2-4 question rather than the reconstruction question: is there a
*structured* projection (not a rank truncation) whose ΔCE is **negative**? A rank
truncation ranks by size. A2-4's projection ranked by structure and improved the model. If
no structured projection of a bilin18 MLP ever improves CE, that is a real difference
between the toy and the model, and worth knowing.

### 2.4 The linearization band, and a specific correction to existing code

**[proved, see THEORY.md T3]** The error left by straightening a component, relative to
deleting it, is invariant under scaling that component's gain — identically, because both
quantities are homogeneous of degree 2 in the form. In A4's design the ratio is exactly
`1/(1+ρ²)²` where `ρ` is the mean-to-fluctuation ratio along the component's directions.
So curvature governs straightenability and the *planted gain* cannot, as a matter of
algebra rather than measurement.

**[correction]** An earlier version of this section said "correlation with the component's
**size** is +0.00". That `+0.00` is against the planted gain. Measured against a component's
*realised* contribution — which is what a practitioner would use — `a4_nulls.py` reports
−0.71 to −0.89, because the realised contribution inherits curvature. Size is not useless;
it is entangled.

**[measured here]** Allowing the tangent substitution beats keep-or-delete at 22 of 25
parameter budgets, up to 10×.

**[measured here, and this is the caveat]** A4's four nulls, run afterwards, show the
frontier advantage is **not task-specific**: a randomly initialised model shows it at 23/25
budgets (median gain 9.3×) and a label-shuffled model at 23/25. That is correct and not a
failure — straightenability is a property of quadratic forms on an input distribution with
a large mean, not of anything training put there. What *is* task-specific is which
components exist at all (recovery cos 1.00 trained vs 0.06 random).

**[on record]** bilin18's input to each MLP is an rms-normed residual stream — exactly the
"large mean, smaller fluctuation" regime. The repo already computes the right object:
`qk_analytic_bilin.py:1-19` does "exact constant/linear/quadratic split about the data mean
`μ = E[x̂]`, plus Eckart–Young-optimal k general quadratic forms".

**[inferred, motivation corrected]** Eckart–Young picks the k features by size, which
answers "which k features best reconstruct". The question for a linearisation budget is
"which features must stay quadratic", and T3 says the answer to that is set by curvature.
The two rankings coincide only if size and curvature happen to be correlated in bilin18 —
which is an empirical question nobody has asked. Note the honest version of the claim: the
best ranking for a budget is by `err_prune − err_lin` per parameter saved, which depends on
both.

**[test]** Cheap and self-contained: for each quadratic feature in `qk_analytic_bilin.py`,
compute its mean-to-fluctuation ratio through the data, then compare two error-vs-parameter
frontiers — features ranked by singular value (current) versus ranked by curvature. A4
predicts the curvature ranking dominates, and predicts the win is largest at intermediate
budgets. This needs no new machinery; it is a re-ranking of an existing computation.

### 2.5 Recovery is blocked by a residual that is harder than noise

**[measured here]** A2-5/A2-8: at the *same* amount of structural mess, a real trained
model's residual defeats block recovery where three synthetic surrogates (isotropic,
structured, memorisation-shaped) do not — 4/11 versus 11/11. Scaling the model's own
residual locates a sharp threshold. About half of the difficulty is symmetry-breaking; the
rest is unexplained.

**[on record]** bilin18's situation rhymes: structure *detection* works (§37, 32/36 channels
beat the corrected null), while *recovery and naming* are below the bar
(`RESULTS_l0_mdl.md:6104`).

**[inferred]** The toys say this gap is expected and is not primarily a fitter problem. In a
setting with exact ground truth, no depth and no estimation noise, the residual of a
*trained* model was 2–3× harder than matched artificial noise. That is a reason to expect
the bilin18 downgrade to survive better fitters, and a reason to spend effort on
*canonicalising away* the residual (A2-8's symmetry projection is the template) rather than
on fitting through it.

### 2.6 Shared computation, and two corrections to how to look for it

**[measured here]** A5. Three corrections to the plan, all analytically confirmed:

1. When one form is shared by `R` readers and each also has a private form, the
   reader-weighted second moment's top eigenvalue ratio is **R+1**, not `R` — and the top
   eigenvector is **not** the shared form. It is `√(R/(R+1))` of the shared form plus
   `1/√(R(R+1))` of *every* private form (0.866 / 0.289 each at R=3; measured 0.8374).
   Reading the top eigenvector as "the shared computation" imports a slice of every private
   one. To isolate it you must deflate.
2. **The reader shuffle cannot test sharing.** A component identical across readers
   contributes identical coordinates to every reader, so permuting readers leaves it
   invariant: measured ratio 4.00 before, 3.5–4.0 after. The control that works is a matched
   family with no sharing, which separates cleanly.
3. Decomposing one reader's form on its own does **not** find the shared part (best cosine
   0.41), because from a single reader the shared/private split is unidentifiable — any
   rotation within their span gives the same form. This is the plan's prediction (iii) in
   its strongest form and the concrete argument for fitting globally.

**[measured here]** And the PCA claim, which needed the right DGP to see: with *orthogonal*
planted parts, PCA and a sparse dictionary are indistinguishable — identical error at every
budget. With *overlapping* parts (planted mutual cosine 0.64), they are still identical on
error (both 0.0000 at the true budget) but completely different on identification: the
dictionary matches the planted parts at 0.96/0.98/1.00, PCA's own components at
0.87/0.39/0.53. **PCA's failure is invisible in reconstruction error and only shows up
against ground truth.**

**[inferred]** This is a direct warning for the bilin18 superposition results. §73–§74 use
SVD directions and report "individual directions account for just 24% of the layer's own
effect; the other 76% appears ONLY under joint removal" (`RESULTS_l0_mdl.md:4213-4260`).
A5 shows that exact signature — components that are individually poor but jointly complete
— is what PCA produces when the true parts *overlap*, even when the true parts are
individually clean and causally separable. The "irreducibly distributed" reading may be an
artefact of the basis, not a fact about the layer.

**[test]** Re-run the §74 joint-removal analysis on dictionary atoms rather than SVD
directions, at matched count and matched removed energy (the repo already knows to match on
energy, `BILIN18_LAYERS_0_1.md:298-301`). A5 predicts the individually-attributable fraction
rises substantially. If it does not, "irreducibly distributed" survives a real challenge and
is worth much more.

---

## 3. What the toys get wrong for bilin18

### 3.1 Part B's entropy census is undefined on bilin18

This is the biggest mismatch and the toys are what exposed it.

The plan's B1 taxonomy tests each head with entropies: (a) factor collapse is `H(A₂) ≈ log n`,
(c) genuine conjunction is `H(A₁), H(A₂)` high with `H(A₁ ⊙ A₂)` low. Every one of those
quantities presumes the pattern is a probability distribution.

**bilin18's pattern is not a distribution.** It is `(q₁·k₁)(q₂·k₂)/D²`, masked to zero
above the diagonal, never normalised. Entries are routinely negative — **[on record]**
`jacclust/SUMMARY.md:69` measures negative mass fraction 0.49 ± 0.27 across heads. There is
no entropy of a signed unnormalised vector, and **[on record]** `HANDOFF.md:207-209` warns
"there is no per-query gauge… Do not row-center as a gauge fix."

So B1 as written cannot be run on the model it was designed to be run on. Part B needs a
third placement implemented — unnormalised score product — and a taxonomy built from
scale-free statistics that survive sign, such as participation ratio, which the earlier
program already used.

### 3.2 …and the toys predict the existing analogue is not evidence of conjunction

**[on record]** `jacclust/SUMMARY.md:69` reports `PR(product) = 0.32 < min(PR(s₁), PR(s₂)) ≈ 0.59`
at **100% of heads** — the unnormalised analogue of "individually vague, jointly precise".

**[measured here]** B2 finds exactly this signature firing where there is nothing to
conjoin. On control tasks where a *single* property suffices, the score-level multiplicative
head still registers as "(c) genuine conjunction" — both factor entropies high (0.86–0.92),
product entropy very low (0.02–0.04) — in both seeds. Multiplying two score fields sharpens
the result whatever they encode. The signature is a property of the operation, not of the
task.

**[inferred]** A statistic that fires at 100% of heads is doing no discriminating work, and
B2 gives the reason it would fire regardless. The sharpening at 100% of bilin18's heads
should not be read as 100% conjunctive heads.

**[retracted after review]** I previously claimed ablation was the statistic that
discriminated. That ablation was broken — it replaced a factor by its per-example mean over
keys, which is negative on 56% of examples and inverts the attention ordering, and it drops
the head to 0.27 even on control tasks whose ceiling is 1.000. See `REVIEW_RESPONSE.md` R3.
A correct ablation (positive constant matched on RMS and on resulting pattern entropy, or
key-shuffling) has not been run.

**[proved, THEORY.md T8]** What the toys *do* establish is which statistics are admissible
at all. `(W₁, W₂) → (cW₁, W₂/c)` is exactly function-preserving, so any per-factor statistic
must be invariant under it. Softmax entropy is not; the participation ratio is. Since
bilin18's pattern is unnormalised and signed, PR is the only one of the two even defined
there — two independent arguments landing on the same statistic.

### 3.3 Depth, and the value bus

Every A experiment is one layer. bilin18 is 18 blocks with a `x = λ₀x + λ₁x₀` skip and a
value bus mixing each block's `v` with block 0's (`tt_model.py:114-116, 213`). A6 (path
kernels) and B3 (composed testbed) are exactly the missing pieces and are not run. Until
they are, nothing in Part A licenses a claim about a *path* through bilin18 — only about a
single layer's form.

### 3.4 The input distributions are not comparable

A1/A3/A4 use Gaussian inputs; A2 uses one-hot pairs. bilin18's MLP input is an rms-normed
residual stream: a fixed-norm shell, strongly anisotropic, with heavy token-frequency
structure. Two consequences. The identifiable-fraction number will differ a lot (§2.2). And
A4's curvature axis is defined against the *input mean*, which for a fixed-norm shell is a
well-defined but distribution-specific object — the ranking should be recomputed, not
imported.

### 3.5 Scale of superposition

A2 and A4 plant 11–18 components in 40–48 dimensions. bilin18 has 4608 hidden units in 1152
dimensions, with **[on record]** ~8 directions per block for 50% of the feed-forward
residual and 90% not reached at 64 (`RESULTS_l0_mdl.md:4179-4211`). A3's calibration curve
covers `K/d` up to 3; bilin18 sits at 4 and its *behavioural* rank is far lower than either.
The regime is not covered by anything measured here, and extending A3 to `K/d = 4` with a
realistic input metric is a cheap way to cover it.

---

## 4. The four tests, ordered by value per unit of work

1. **Re-rank `qk_analytic_bilin.py`'s quadratic features by curvature instead of singular
   value** and compare error-vs-parameter frontiers (§2.4). No new machinery; directly
   tests A4's correction on the real model.
2. **Measure the identifiable fraction of each MLP's interaction form** under the realised
   lift metric (§2.2). One accumulation pass; recalibrates every existing weight-space mass
   statistic.
3. **Re-run the §74 joint-removal analysis on dictionary atoms rather than SVD directions**,
   matched on count and removed energy (§2.6). Tests whether "irreducibly distributed" is
   a property of the layer or of PCA.
4. **Rebuild the B1 census on ablation with matched-scale factor replacement**, and
   implement the unnormalised-product placement so Part B is about bilin18's actual
   architecture (§3.1–3.2).

**Correction to an earlier version of this document**, which said the four tests were not
runnable here because the weights were absent. They were merely *uncached*: `load_elriggs`
downloads from the Hub, the box has 113 GB free and a 32 GB GPU, and the checkpoint pulls
in about 100 seconds. Test #4 has now been run — see below — and the model loads and gates
correctly (546M parameters, 18 layers, d=1152, ungated MLPs of width 4608, CE 3.46 on a
32-chunk pile-10k sample).

---

## 5. RESULT: test #4, run on the real model (`bilin18_attention.py`)

The census rebuilt on bilin18's own attention, with participation ratio in place of
entropy (undefined there, and gauge-dependent anyway — `THEORY.md` T8), and compared
against **the same architecture with random weights**, which is the control the toys said
would be decisive and which had not been run.

| | sharpening signature fires | median PR drop | median negative mass | median \|cos(W₁,W₂)\| |
|---|---|---|---|---|
| bilin18, trained | **162 / 162 heads (100%)** | 0.097 | 0.490 | 0.031 |
| same architecture, random weights | **162 / 162 heads (100%)** | **0.207** | 0.500 | 0.002 |

**The signature carries no information.** An untrained network of the same shape shows
"individually vague, jointly precise" at exactly the same universal rate, and with a
*larger* median drop. The recorded 100%-of-heads result
(`jacclust/SUMMARY.md:69`) is therefore not evidence that bilin18's branches are
conjunctive — it is what this architecture does before it has learned anything.

**The negative-mass figure too.** The recorded 0.49 ± 0.27 negative mass is reproduced
here at 0.490 for the trained model — and random weights give **0.500**. That statistic
also fails to distinguish trained from untrained.

Two things this does establish, both new:

- **Regime (b), factor alignment, does not occur in bilin18.** Zero of 162 heads have their
  two QK circuits near-identical (|cos| > 0.9); the median is 0.031, barely above the 0.002
  of random weights. So whatever the two branches are doing, they are not duplicates.
- **The reimplementation is validated against the existing record.** An independently
  written measurement reproduces the recorded 100% firing and the recorded 0.49 negative
  mass, so the comparison against the null is like-for-like rather than a different
  statistic disagreeing.

What this does *not* say: that the branches are not conjunctive. It says the two statistics
on record cannot tell. The instrument that survived the toys is the readout of what each
branch *reads* (B2-4, A6, B3), and that is the measurement to run next on these heads.

---

## 6. RESULT: test #2, run on the real model — the identifiable-subspace question

Files: `bilin18_identifiable.py`, `bilin18_identifiable_power.py`, `bilin18_blind_direction.py`.

### 6.1 What was being asked

A2-3 on the toy established a procedure. A bilinear layer's per-output *interaction
form* `M_d` is a symmetric matrix, and the function only ever uses it through
`x^T M_d x`. So any part of `M_d` orthogonal to `span{x x^T : x in data}` is invisible
to the function no matter how much Frobenius mass it carries. On the toy, one-hot
inputs probe 529 of Sym²'s 1081 dimensions, and a trained model kept 65–71% of its
mass inside that span against a 49% chance level — enough that projecting first
changed conclusions, and little enough that mass statistics stayed roughly meaningful.

The registered prediction for bilin18 was that the identifiable fraction is *much*
lower, because Sym² at width 1152 has 664,128 dimensions. That prediction is confirmed,
but the interesting part is why, and it is not the reason I expected.

### 6.2 The raw measurement, and why it means less than it looks like

With 4000 sampled MLP inputs per layer (`bilin18_identifiable.py`, 19 s):

| layer | identifiable fraction | random-form null | ×null |
|---|---|---|---|
| 0 | 0.0161 | 0.0061 | **2.6×** |
| 1 | 0.0070 | 0.0058 | 1.2× |
| 5 | 0.0060 | 0.0062 | 1.0× |
| 9 | 0.0047 | 0.0058 | 0.8× |
| 13 | 0.0045 | 0.0061 | 0.7× |
| 16 | 0.0059 | 0.0061 | 1.0× |
| 17 | 0.0083 | 0.0059 | 1.4× |

Read naively: **less than 1% of an interaction form's mass can affect the function, and
the forms are no better aligned with the data than a random symmetric matrix.** The
first half of that is real. The second half is not a licensed conclusion from this
table, because two very different worlds produce the same number — a form whose learned
part is a tiny share of its mass, and a form whose learned part is large but spread over
far more than 4000 directions. So the table was not written up until the confound was
tested.

### 6.3 The power analysis, which is where the actual result is

An unrelated form has identifiable fraction exactly `N/dim`, dead linear in `N`. A form
concentrated on `k` data directions saturates as `N → k`. So sweep `N` and read the
*shape* (`bilin18_identifiable_power.py`, 23 s). Expressed as a multiple of chance:

| layer | N=250 | 500 | 1000 | 2000 | 4000 | 8000 |
|---|---|---|---|---|---|---|
| 0 | **4.93×** | 4.15× | 3.68× | 3.15× | 2.69× | 2.31× |
| 5 | **0.10×** | 0.15× | 0.28× | 0.65× | 0.95× | 1.09× |
| 13 | **0.19×** | 0.23× | 0.36× | 0.58× | 0.79× | 0.89× |
| 17 | **2.25×** | 1.67× | 1.39× | 1.29× | 1.27× | 1.29× |
| *unrelated-form null* | 1.14× | 1.09× | 1.11× | 1.05× | 0.99× | 0.99× |

The null is flat at 1.0× across a 32× range in `N`, exactly as the theory says it must
be, which validates the estimator. Against that, three things are visible that the
single-`N` table hid completely:

1. **The 4000-sample table is a sampling artefact, not a fact about the weights.** Every
   layer's curve is still moving at N=8000. The absolute fraction is not measurable by
   this route at this scale — it would need ≳10⁵ samples and a 10⁵×10⁵ solve.
2. **Layers 0 and 17 are concentrated.** Nearly 5× chance at N=250, decaying as more
   directions are added: the signature of mass sitting on the few directions the data
   visits most.
3. **Layers 5 and 13 are the opposite, and this is the finding.** At 0.10× and 0.19×
   chance they are an *order of magnitude below* what an unrelated matrix scores. Below
   chance is not "no structure" — an unrelated form gets chance by construction. It
   means the form has **less** mass on the data's leading directions than a random
   matrix does, i.e. it is actively orthogonal to them.

### 6.4 The mechanism, confirmed directly

The same run measured the other half. The Gram of `x x^T` at layers 5 and 13 has
participation ratio 1.3 and 2.1 — essentially *one* direction carries the second-moment
mass of the rms-normed MLP input. That makes a sharp prediction: the MLP's quadratic
form should nearly annihilate that direction. Tested on all 11 sampled layers
(`bilin18_blind_direction.py`, 20 s), with `v` the top PC of the normalised input and
curvature `|v^T M v| / ||M||_F` compared against an equally-sized random form along the
same `v`:

| layer | top-PC share of the input | mean \|v·x̂\| | curvature along v, vs a random form |
|---|---|---|---|
| 0 | 0.086 | 0.244 | **5.23×** |
| 1 | 0.862 | 0.928 | 0.03× |
| 3 | 0.271 | 0.512 | 0.15× |
| 5 | 0.941 | 0.970 | 0.03× |
| 7 | **0.944** | **0.972** | **0.00×** |
| 9 | 0.895 | 0.946 | 0.03× |
| 11 | 0.855 | 0.924 | 0.14× |
| 13 | 0.831 | 0.911 | 0.19× |
| 15 | 0.769 | 0.876 | 0.82× |
| 16 | 0.701 | 0.835 | 1.79× |
| 17 | 0.543 | 0.715 | **4.41×** |

Two clean monotone trends, and they are the same story from both ends:

**Through the middle of the network the rms-normed MLP input is nearly a single fixed
vector.** At layers 5–7 the mean \|cos\| between a token's normed MLP input and the top
PC is 0.97 — every token, at every position, arrives within about 14° of the same
direction. (This is the massive-activation / attention-sink phenomenon that is already
documented in the literature for other transformers; the contribution here is not that
it exists but that it is measured on bilin18 and connected to the forms.)

**And the bilinear MLPs there are built to not see it.** Curvature along `v` runs
0.00–0.19× a random form's — layer 7 annihilates it to four significant figures. The
MLP reads only the small residual variation *around* the dominant direction. Layer 0,
whose input is the embedding and genuinely spread (top-PC share 0.086), is 5.2×
*enriched* along its top PC instead, and the last two layers swing back the same way.

This is what produces the below-chance numbers in §6.3, and it also explains the
sample-starvation: the directions these MLPs actually use are precisely the ones the
data visits rarely, which is the worst possible regime for a random-sample probe.

### 6.5 What this does to the rest of the program

This is the sharpest transfer failure found so far, and it cuts against the toys rather
than against bilin18.

- **A2-3's projection procedure does not transfer by this route.** It is not that the
  answer is different; the measurement is not executable at width 1152. The substitute
  that *does* transfer is the curve-shape diagnostic above, which is cheap (23 s) and
  gave a stronger answer than the absolute number would have.
- **Every Frobenius-mass statistic the toys rely on is compromised on bilin18.** Block
  mass, eigenvalue participation ratio, dictionary-atom mass — all of them weight
  directions by `||·||_F`, and on layers 1–13 the overwhelming majority of that norm sits
  where the data never goes. A form can look "concentrated" in weight space while its
  functionally live part is elsewhere entirely.
- **The toys could not have caught this.** Their one-hot inputs have no dominant
  direction by construction (top-PC share 1/d), so the entire phenomenon — a nearly
  one-dimensional input distribution that the layer is engineered to ignore — has no
  analogue in any toy run in this program. That is a gap in the toy design, not a
  wrong result.
- **The concrete fix for future measurements** is to whiten by the input second moment
  before any mass statistic — i.e. work with `S^{1/2} M S^{1/2}` rather than `M`, which
  is the Λ-weighted functional metric the program already defined in `bq_common.py` and
  then, on bilin18, did not use. That is now the recommended default for §2.2's
  weight-space reads.

---

## 7. Testing §6.5's own recommendation — and a registered prediction that failed

Files: `bilin18_whitened.py`, `bilin18_whitened_dirs.py`.

§6.5 ended with a recommendation: whiten by the input second moment before reading any
mass statistic off bilin18 weights. That was an argument, not a result, so it was tested.

**The measurement.** For an output direction `d` with interaction form `M_d`, take the
rank-`k` approximation two ways — top-`k` eigenpairs of `M_d` by |eigenvalue| ("raw",
which is what a Frobenius-mass statistic sees), and top-`k` eigenpairs of
`S^{1/2} M_d S^{1/2}` mapped back ("whitened", the Λ-weighted functional metric) — and
score both by fraction of variance unexplained of `x^T M_d x` on 6000 held-out inputs.
FVU is purely functional: it does not care about Frobenius norm at all.

**Registered prediction, written before the run:** the gap is large through the middle
of the network (layers 5–13, where §6.4 found curvature along the dominant direction at
0.00–0.19× random, so the norm is mostly dead weight) and small at layers 0 and 17.

### 7.1 Result — the recommendation holds, the prediction does not

Rank needed for 90% of the function, and the FVU ratio at k=16, on the top-8 principal
directions of each MLP's own output:

| layer | raw rank for 90% | whitened rank for 90% | whitening gap at k=16 |
|---|---|---|---|
| 0 | 128 | 64 | 1.5× |
| 1 | >128 | **32** | 3.0× |
| 5 | >128 | **32** | 2.6× |
| 9 | >128 | 64 | 2.2× |
| 13 | >128 | **32** | 2.7× |
| 17 | 8 | **4** | **5.2×** |

Whitening helps everywhere, by 1.5–5.2×, and at four of six layers it is the difference
between reaching 90% of the function inside rank 32–64 and not reaching it at all by
rank 128. **The recommendation is validated.**

**The prediction is wrong, and consistently so.** The gap is *largest* at layer 17
(5.2×), which is the layer where the input is *least* dominated by a single direction
(top-PC share 0.543 against 0.94 at layer 7). It is smallest at layer 0. So whitening's
benefit does not track the blind-direction mechanism of §6.4 at all — it is right for a
different reason than the one I argued for. The same ordering appeared in the
random-direction run (`bilin18_whitened.py`: 3.9× at the ends against 1.9× in the
middle), so this is not a basis artefact. I do not have a mechanism for it and am not
going to invent one.

### 7.2 A methodological correction that matters more than the above

The first run used **random** output directions `d`. A random `d` mixes all 4608 neurons,
and it turns out to overstate rank badly. Same layers, same metric, random `d` against
output-PC `d`:

| layer | whitened FVU at rank 128, random d | …output-PC d |
|---|---|---|
| 9 | 0.151 | **0.026** |
| 13 | 0.179 | **0.019** |
| 5 | 0.089 | 0.015 |

On random directions layers 9 and 13 never reach 90% of the function even at rank 128,
and the honest-looking conclusion would have been "bilin18's mid-network interaction
forms are irreducibly high rank." On the directions the layer actually uses, rank 32–64
suffices. **Measuring interaction-form rank on random output directions overstates it by
roughly 2–8×.** Anyone repeating this analysis should sample `d` from the layer's own
output distribution; this program's toy runs used the natural output basis throughout,
so the error is specific to scaling up, and it is an easy one to make.

### 7.3 The one robust structural fact

Layer 17 is qualitatively different from every other layer measured, in both bases and
by every statistic here: **whitened rank 4 captures 90% of its function and rank 16
captures 99.5%.** The final bilinear MLP is very nearly a four-dimensional quadratic
form. Layers 1–13 need 32–64. That is an 8–16× separation, and it is the strongest
candidate so far for a place in bilin18 where the toys' decomposition machinery
(dictionary learning, simultaneous block diagonalisation, CP) would actually have enough
signal-to-parameter ratio to work — a rank-4 object in a 1152-dimensional space is
tractable in a way a rank-64 one is not.

---

## 8. Reading layer 17: 8 numbers that reproduce a 16M-parameter layer

Files: `bilin18_layer17.py`, `bilin18_layer17_readout.py`, `bilin18_layer17_verify.py`.

§7.3 identified layer 17 as the one place on the real model where the decomposition
machinery has enough signal per parameter to be worth pointing at. This does that. It is
the first result in the program that reads a real computation out of the 546M model
rather than measuring a property of it.

### 8.1 The gate: does the rank claim survive as cross-entropy?

Held-out FVU on a quadratic feature is not the same thing as the model still working, so
the layer-17 MLP was actually **replaced** by its truncation — output confined to the
top-`R` principal output directions, each carrying a rank-`k` form — and the model
re-scored. The output variance decides `R`: layer 17's MLP output needs just **4**
principal directions for 90% of its variance (2 for 50%), so `R = 4`.

| replacement | CE | Δ vs baseline | share of the way to a dead layer |
|---|---|---|---|
| baseline | 3.4557 | — | — |
| **quadratic part of layer 17 removed** | 4.5330 | +1.0773 | 100% |
| output projected to top-4 PCs, forms exact | 3.5505 | +0.0948 | 8.8% |
| …+ **whitened rank 2** | 3.5576 | +0.1018 | **0.7%** |
| …+ raw rank 2 | 3.6049 | +0.1491 | 5.0% |
| …+ whitened rank 4 | 3.5567 | +0.1009 | 0.6% |
| …+ raw rank 4 | 3.5716 | +0.1158 | 2.0% |

Two things come out of this table.

**The layer compresses enormously.** Deleting layer 17's quadratic part costs 1.08 nats.
Keeping **4 output directions × rank-2 forms — 8 signed squared projections — recovers
all but 0.7% of that.** The replacement is about 13.8k numbers standing in for a layer of
roughly 15.9M parameters, at a cost of 0.007 nats.

**Whitening matters to the model, not just to a reconstruction error.** At the same rank
budget, ranking eigenpairs by |eigenvalue| does 5.0% of a dead layer's damage where the
Λ-weighted ranking does 0.7% — a **7× difference in functional damage for identical
parameter cost**. This is the strongest confirmation of §6.5's recommendation, and it is
in the currency that matters.

*(A first version of this gate reported "project only" costing more than deleting the
layer, which is impossible. The cause was mine: `mu` was the mean of the MLP output,
which already contains `Down_bias`, so the bias was added twice. Fixed by passing
`mu − bias` and restoring only the component of the mean lying outside span(P).)*

### 8.2 What it computes

A symmetric form is a signed sum of squares, `x^T M_d x = Σ λ_i (w_i·x)²`, so the layer
is literally: take a few projections of the residual stream, square them, and add them
with signs. With `R = 4` and rank 2, that is eight terms. The two leading output
directions, named on both ends:

**Output direction 1** — 48.4% of the MLP's output variance. Writing along it promotes
` challenge, draw, promise, presented, designed, battle, tackle, pose`.
- feature 1 (59% of the form, **subtracts**): squared projection onto a direction whose
  strongest excitation is auxiliaries and function words — ` has, who, 're, 've, have,
  include, will`.
- feature 2 (21%, **adds**): a delimiter/punctuation direction — `', **, ", (, \n, £`.

**Output direction 2** — 22.3% of output variance. Promotes determiners and possessives
` the, their, our, a, your`; demotes verbs and participles.
- feature 1 (59%, **subtracts**): sentence-final and closing punctuation — `####, )., ?,
  ###, ]).`
- feature 2 (19%, adds): copulas and pronouns — ` are, he, be, they, is, it, were, was`.

Direction 2 reads as a syntactically sensible next-token rule: after a sentence-closing
punctuation mark, suppress "predict a determiner"; in the presence of a copula or
pronoun, boost it.

### 8.3 Verifying the names — including a first attempt that had no power

Naming a direction by the tokens whose unembedding rows it points along is the standard
shortcut, and it is weak: it assumes the residual direction the MLP *reads* is aligned
with the one the unembedding *writes*. So the names were tested directly — run the
corpus, record `(w·x)²` at every position, and ask which tokens actually sit where the
feature fires.

**The first attempt failed for lack of power and is recorded rather than quietly
replaced.** It used the 32×513 eval set, where only 48 tokens clear 30 occurrences; the
chance overlap between a 20-name list and a 10-token excitation list against a pool of 48
is 4.2/10, and the measured 2.7/10 sat *below* it. That run establishes nothing either
way. Two fixes: 512×513 = 262,144 positions (1,026 tokens clearing the threshold, 21×
more), and a Spearman correlation across all qualifying tokens against a 200-draw
permutation null, instead of an overlap between mismatched lists.

| feature | ρ, current token | ρ, next token | permutation null (95th) | |
|---|---|---|---|---|
| dir 1, feature 1 | **0.657** | 0.447 | 0.057 | supported |
| dir 1, feature 2 | **0.505** | 0.253 | 0.063 | supported |
| dir 2, feature 1 | **0.392** | 0.086 | 0.074 | supported |
| dir 3, feature 1 | 0.306 | 0.279 | 0.058 | supported |
| dir 3, feature 2 | 0.147 | 0.189 | 0.067 | weakly supported |
| dir 2, feature 2 | 0.114 | −0.050 | 0.060 | **barely above null** |

All six clear their null, but the spread matters and "6/6 confirmed" would overstate it:
the three leading features are solidly supported (ρ 0.39–0.66 against ~0.06), while
dir 2 feature 2 at 0.114 is barely distinguishable from chance and its name should be
treated as unverified. **ρ(current token) exceeds ρ(next token) in five of six**, so
these features key on the token actually present, not on the prediction being made.

### 8.4 What this settles for the program

- The toys' central methodological claim — that the useful decomposition is into
  interaction-form eigendirections under the functional metric, not into neurons —
  **holds on the real model, gated by cross-entropy**, at least at layer 17.
- The Λ-weighted metric is not a technicality. Same parameter budget, 7× less damage.
- Interpretability by unembedding alignment is **partially** justified here: it agrees
  with measured excitation for the strong features and not for the weak ones. It should
  be reported with the correlation attached, never on its own.
- The obvious limit: this worked at layer 17 because layer 17 is nearly rank-4. Layers
  1–13 need rank 32–64 on 4–8× more output directions, which is two orders of magnitude
  more terms and is not readable the same way. Whether the same treatment produces
  anything legible in the middle of the network is untested and is the natural next step.

---

## 9. Does layer 17's treatment work anywhere else? A profile of all 18 MLPs

File: `bilin18_depth_profile.py` (503 s).

§8.4 recorded the limit of the layer-17 result — it worked because layer 17 is nearly
rank-4 — and named the natural next step: whether the same treatment produces anything
at the other seventeen layers. It does not.

Each MLP was replaced by `R` principal output directions carrying rank-`k` forms, with
`R` set by that layer's own 90%-of-output-variance point (capped at 48) and `k` swept to
32, then the model re-scored. Damage is measured against the untouched model and
normalised by what deleting that layer's quadratic part costs, so each layer is graded
on its own scale.

| layer | cost of deleting it | cheapest config within 5% | compression |
|---|---|---|---|
| 0 | 1.802 | R=48, k=32 → 4.0% | 9× |
| 1 | **5.650** | — | — |
| 2–15 | 0.024–0.520 | — | — |
| 16 | 1.167 | **R=9, k=2 → 2.2%** | **512×** |
| 17 | 1.077 | — | — |

**Two of eighteen layers compress**, and the winner is layer 16, not 17: nine output
directions carrying rank-2 forms — eighteen squared projections, 31,122 numbers for a
15.9M-parameter layer — at 2.2% damage.

Three things need saying about this table rather than leaving it to be misread.

**Layer 17 appears to fail its own result, and does not.** §8.1 reported 0.7% damage at
R=4, k=2. The difference is the reference point: §8.1 measured damage *beyond the
projection step*, this table measures it from the untouched model, and layer 17's
projection-to-4-directions step alone costs 8.8%. Both numbers are right and they answer
different questions. The deeper problem is that `R` was fixed by a variance rule chosen
in advance, which is not the same as choosing it by what it costs — resolved in §10.

**Layer 1 is the most important MLP in the model by a wide margin.** Deleting its
quadratic part costs 5.65 nats, against 1.80 for layer 0 and about 1.1 for layers 16 and
17. Every layer from 2 to 15 costs less than 0.52, most of them under 0.06. Nothing in
the toy program predicted this concentration and it was not looked for.

**The 5% tolerance is far harsher on the middle than on the ends.** 5% of layer 12's
0.024-nat delete cost is 0.0012 nats — a bar an order of magnitude tighter in absolute
terms than the same 5% imposes on layer 1. So "layers 2–15 do not compress" may be the
wrong reading of these rows; "layers 2–15 barely do anything individually" is at least
as consistent with them. Which of those is true is what §10 tests.

---

## 10. Resolving §9 — and a correction to §8's headline number

File: `bilin18_depth_followup.py` (237 s).

### 10.1 Correction: layer 17's replacement costs 9.5%, not 0.7%

§9 flagged that `R` was fixed by a variance rule rather than chosen by cost. Sweeping
`R ∈ {4,8,16,32,64}` against `k ∈ {2,4,8,16}` and taking the cheapest configuration
inside each damage tolerance, measured from the untouched model:

| layer | cheapest ≤5% | cheapest ≤10% |
|---|---|---|
| 0 | — | R=16, k=16 → 7.6%, 51× |
| 16 | **R=4, k=2 → 4.2%, 1151×** | R=4, k=2 → 4.2%, 1151× |
| 17 | R=32, k=4 → 4.8%, 86× | R=4, k=2 → 9.5%, 1151× |

**§8.1 said the eight-term replacement of layer 17 "recovers all but 0.7%" of the cost of
deleting the layer. That sentence conveys the wrong quantity.** 0.7% is the damage added
by truncating the forms to rank 2, measured *beyond* the projection step. Measured from
the untouched model — which is what "recovers all but X" means to a reader — the same
replacement costs **9.5%**. Both numbers appear in §8.1's table; the prose picked the
flattering one. The correct decomposition of layer 17 at R=4, k=2:

- total cost of the replacement: 9.5% of what deleting the layer costs (+0.102 nats)
- of which 8.8 points come from confining the output to 4 directions
- and only 0.7 points from truncating those 4 forms to rank 2

So the *rank* claim is as strong as §8 said — rank 2 is nearly free once you have the
four directions — but the *compression* claim was overstated by an order of magnitude in
damage terms.

**And layer 16, not 17, is the model's most compressible layer.** At the identical
parameter budget (R=4, k=2, 13,832 numbers, 1151×) it costs 4.2% where layer 17 costs
9.5%. §9's table already showed this and I read past it. The §8 machinery is sound; it
was pointed at the wrong layer.

Unaffected by this: the whitened-versus-raw comparison in §8.1, since both arms are
measured beyond the same projection step at matched parameter cost, and the 7× gap
stands. Also unaffected: the feature naming and its verification in §8.2–8.3.

### 10.2 The middle layers are individually cheap and jointly not

§9's third caveat was that a 5% relative tolerance is brutal for layers whose delete cost
is 0.024 nats, so "layers 2–15 do not compress" might really be "layers 2–15 barely do
anything". Deleting all fourteen at once settles it:

| | cost |
|---|---|
| each of layers 2–15 deleted alone | 0.024 – 0.520 nats |
| **sum** of the fourteen individual costs | 1.790 nats |
| **all fourteen deleted together** | **5.142 nats** (CE 3.456 → 8.598) |

**2.87× superadditive.** The middle of this network is not doing little — single-layer
ablation massively understates it, because when one mid-layer's quadratic part is
removed the other thirteen absorb the loss, and when all fourteen go there is nothing
left to absorb it. Any claim of the form "layer *n* contributes almost nothing" built on
a one-at-a-time ablation is unsafe in this model, and §9's own table is the example.

This also reframes §9's compression result. The two compressible layers, 0 and 16, are
exactly the two whose individual delete cost is large (1.80 and 1.17 nats). The layers
that resist compression are the ones whose individual contribution is small *and*
collectively large — which is the signature of computation distributed across depth
rather than localised in any layer. That is the honest reason layer 17's treatment does
not generalise, and it is a stronger statement than "the other layers are higher rank".

---

## 11. RESULT: test #1, and a correction to how it was specified

File: `bilin18_centered_features.py` (62 s).

§4 listed as the highest-value test: "re-rank `qk_analytic_bilin.py`'s quadratic features
by curvature instead of singular value", on the grounds that T3 makes size uninformative
about which features must stay curved.

**Reading the code changed the test.** That script's ranking is the SVD of `Down·C^{1/2}`
with `C = E[hh^T]` — already in the data-weighted metric, i.e. exactly the whitening §7
independently validated. "Ranked by raw size" was a wrong description of existing code
and §2.4's framing of it should be read with that correction.

But the same reading exposed a real defect. `C` is the **uncentered** second moment, and
the constant term contains `Down[(Lμ)⊙(Rμ)] + Down_bias` but **not** `Down·E[h]`. Since
`h = (Ld)⊙(Rd)` is quadratic, `E[h] ≠ 0` even though `d` is centred by construction. So
there is an unmodelled constant, and Eckart–Young must spend singular directions
representing it — rank spent on a constant that could be stored for free in the bias.

Registered prediction: centring wins, most at small `k`, with the win tracking the
constant's energy share.

| block | constant's share of the quadratic term's energy | k=8 | k=32 | k=128 |
|---|---|---|---|---|
| 0 | **52.2%** | +0.0131 | −0.0012 | −0.0001 |
| 1 | 15.8% | **+0.0400** | −0.0008 | +0.0005 |
| 5 | 44.1% | +0.0007 | −0.0006 | −0.0001 |
| 7 | 22.4% | +0.0017 | +0.0010 | +0.0003 |

*(improvement in held-out CE, positive = centring better)*

**The diagnosis is confirmed and the predicted consequence is not.** The omitted constant
really does carry 16–52% of the quadratic term's energy, so the defect is large in the
terms that motivated looking for it. But the benefit is negligible: centring wins 7 of 12
cells, mean improvement +0.0045 nats, best +0.0400 at block 1 `k=8` — a 3% relative gain
against that cell's +1.3251 error — and at `k ≥ 32` it is a wash in both directions. The
predicted correlation with energy share is absent: block 0 has the largest constant
(52.2%) and a third of block 1's improvement.

The reason is that Eckart–Young absorbs a rank-one constant nearly for free. One singular
direction out of eight is a real cost only when eight is already too few, which is why
the entire effect lives at `k=8`.

**Verdict on test #1: no meaningful frontier improvement is available here.** Centring is
still worth doing — it makes the constant exact instead of approximated, costs nothing,
and removes a conceptual wart — but it is a tidiness fix, not a result. The test is
closed negative.

*(Data note: the original script reads `data_fineweb_tokens.npy`, which is not on this
box; the pile-10k sample built for the other bilin18 runs was substituted. Both arms see
identical data so the contrast is unaffected, but the absolute ΔCE values here are not
comparable to those in `qk_analytic_bilin.json`.)*

---

## 12. RESULT: test #3 — "irreducibly distributed" is real at its operating point, and the number is not a property of the layer

Files: `bilin18_joint_removal.py` (216 s), `bilin18_joint_removal_matched.py`.

§74 of `RESULTS_l0_mdl.md` reports the repo's boundary result: MLP1's top-32 SVD output
directions removed together cost 0.161 nats while the sum of the 32 individual removals
is 0.039, so **individual directions account for 24% and 76% appears only jointly**. A5
predicted this is what PCA produces when the true parts overlap, and that dictionary
atoms would raise the individually-attributable fraction substantially.

§78 already red-teamed §74 with a sparse dictionary and settled that a dictionary crosses
the *nameability* boundary (23/32 monosemantic against SVD's 0/32) and not the *causal*
one. But it never measured §74's own statistic — it used a different denominator (the
full-layer knockout, 5.57 nats) — so the number A5 predicts on was still open.

### 12.1 At §74's operating point, no basis helps

Four arms, all 32 directions, all mean-ablated identically on the same data:

| arm | removed energy | joint ΔCE | sum of solos | attributable |
|---|---|---|---|---|
| svd32 (§74's own) | 70.5% | +0.3832 | +0.0345 | **9.0%** |
| rot32 (random rotation in the same span) | 70.5% | +0.3832 | +0.0350 | **9.1%** |
| dict32 (32-atom dictionary) | 64.3% | +0.1146 | +0.0151 | 13.1% |
| dict4096 (top 32 of 4096, §78's object) | 53.0% | +0.1820 | +0.0374 | 20.5% |

The rotation control reproduces svd32 to within 0.1 points, confirming §74's own
basis-independence claim exactly. And at fixed count 32, **no basis makes this layer
individually attributable** — the best is 20.5%, so four fifths of the effect is still
joint-only. §74's qualitative claim survives.

*(Gate caveat: this reproduces §74's signature but not its number — 9.0% here against
their 24%, with joint ΔCE 0.383 against their 0.161. The fineweb token file their script
uses is not on this box, so this runs on the pile-10k sample. Same layer, same method,
different corpus; the comparison across arms is internally matched, the comparison to
§74's absolute figure is not.)*

### 12.2 A5's prediction fails once removed energy is matched

The energies above are not matched, and the test specification required matching them
precisely because they confound this statistic — directions that remove less of the
layer's output have less overlap to interfere. dict4096 removes 53% where svd32 removes
70.5%, and dict4096 is the arm carrying the result. So: hold energy fixed, vary count.

| arm | n | removed energy | joint ΔCE | sum of solos | attributable |
|---|---|---|---|---|---|
| **svd** | **4** | 55.8% | +0.0411 | +0.0264 | **64.1%** |
| dict4096 | 32 | 53.0% | +0.1820 | +0.0374 | 20.5% |
| svd | 32 | 70.5% | +0.3832 | +0.0345 | 9.0% |
| **dict4096** | **82** | 69.5% | +0.5706 | +0.0789 | 13.8% |

At ~53% energy, SVD is **3× more** individually attributable than the dictionary — the
opposite of A5's prediction. At ~71% the dictionary is ahead by 1.5×. The direction of
the effect flips depending on which energy you match at, which is what a confounded
comparison looks like. **A5's prediction is refuted**; the apparent dictionary advantage
in §12.1 was an energy artefact, and my own 1.5× threshold passed it only because the
script did not match energy — the one control the test called for.

### 12.3 What this actually establishes, which is more useful than either prediction

**The attributable fraction is not a property of the layer.** On the same MLP1, unchanged,
it takes values from **9% to 64%** depending only on how many directions you remove:

| directions removed | 4 | 32 | 82 |
|---|---|---|---|
| attributable fraction | 64.1% | 9.0% | 13.8% |
| joint ΔCE | 0.041 | 0.383 | 0.571 |
| sum of solos | 0.026 | 0.035 | 0.079 |

The mechanism is plain in the last two rows: the joint effect grows 14× from 4 to 32
directions while the solo sum grows 1.4×. Interference grows superlinearly in the size of
the ablation set. So "76% appears only under joint removal" is substantially a statement
about *removing 32 things at once*, not a scale-free fact about how distributed the layer
is. Quoting the fraction without its count and removed energy is not meaningful, and the
repo's phrasing does quote it that way.

This is the same phenomenon as §10.2's 2.87× superadditivity across depth, measured
within a single layer instead of across fourteen. Ablation superadditivity is pervasive
in this model, and any "X% is joint-only" or "layer *n* contributes almost nothing" claim
needs its ablation-set size attached to mean anything.

**Net verdict on test #3.** §74's boundary result survives at its own operating point and
survives the basis challenge that A5 aimed at it — that is a real win for §74, and it is
worth more than the challenge would have been. What does not survive is the number's
status as a layer property. Both my prediction and the framing I inherited were wrong in
the same way: treating a budget-dependent statistic as though it measured the layer.

---

## 13. The budget-free version of §74's question, answered

File: `bilin18_shapley.py` (478 s, 660 model evaluations).

§12 left the program without a usable statistic: "X% individually attributable" moves
from 9% to 64% with the analyst's choice of ablation-set size, so the question §74
actually cared about — is MLP1's causal content concentrated in a few directions or
spread across many? — had no budget-free answer. The Shapley value is the standard
repair, and it is the *right* repair here for a precise reason: it averages each
direction's marginal contribution over all coalition sizes (so no budget is chosen), and
its efficiency axiom forces the attributions to sum to the joint effect exactly — the
"91% unexplained" residual that solo ablation leaves cannot exist by construction.

Twenty random permutations of the 32 SVD directions, marginal contributions measured by
held-out CE at every step. *(A gate note against my own script's description: with
full-permutation sampling, efficiency holds identically — each permutation's marginals
telescope to v(all) — so the measured 0.0000 gap validates the bookkeeping only. The
honest uncertainty is the per-direction standard error, ±0.004–0.014 nats.)*

| | solo ablation (§74's numerator) | Shapley |
|---|---|---|
| sum of attributions | 0.0345 (9.0% of joint) | **0.3832 (100%, by axiom)** |
| participation ratio | 1.4 of 32 | **9.5 of 32** |
| top direction | 0.0224 | 0.1034 (27.0% of the layer) |
| top 4 / top 8 share | — | 50% / 67% |
| negative contributions | 14 of 32 solo values ≤ 0 | 1 of 32 |

**The budget-free answer is "about ten directions".** Participation ratio 9.5 of 32: the
causal content of MLP1's top-32 subspace is carried by roughly ten effective directions,
with the largest single one holding 27% and the top eight holding two thirds.

> **Corrected by §16:** the count was inflated by sampling noise in the basis. With a
> 4.8×-data basis the participation ratio is **5.6**, the leader holds **39%**, and two
> of the six "leaders" turn out to be one real component split in two by the small-sample
> PCA. The qualitative verdict (oligarchy with a leader) stands; "ten" does not.

Three consequences:

1. **Both prior readings were artefacts of their instruments.** Solo ablation says
   "1.4 effective directions but 91% unexplained" — concentration by interference
   blindness. The §74 phrasing says "irreducibly distributed" — a smear, by the
   budget effect of §12. The layer is neither: it is *oligarchic*, ten-ish directions
   with a clear leader, and that answer required an attribution that handles
   interference rather than a bigger or smaller ablation set.
2. **Solo ablation understates every important direction, non-uniformly.** The top
   direction's Shapley value is 4.6× its solo effect; directions 4 and 5 have *negative*
   solo effects (+0.019/+0.018 Shapley) — they look actively helpful to remove one at a
   time while genuinely carrying part of the layer. Ranking directions by solo ablation
   — which the repo's nameability batteries do — misranks them.
3. **§74's nameability conclusion is untouched**: its z-score battery asks whether any
   single direction is *interpretable*, not how the causal mass is distributed, and
   nothing here contradicts 0/32 nameable. What changes is the phrase "irreducibly
   distributed": the distribution has ten-ish parts, and the reason no part shows up
   solo is interference, not absence of parts.

The natural follow-on — are the ten Shapley-leading directions the ones a dictionary
names, i.e. do nameability and causality align once interference is handled — is
registered as an open question, not run.

---

## 14. The follow-on: nameability and causality are aligned after all

File: `bilin18_shapley_dict.py` (no new model evaluations — geometry on §12/§13's objects).

§78 (repo) reported the nameable axis and the causal axis of MLP1 are "nearly
orthogonal": a dictionary names 23/32 atoms but 0/32 are load-bearing. §13 showed the
causal instrument behind that second number — solo ablation — misranks by up to 4.6× and
assigns negative effects to genuinely load-bearing directions. So the recorded
orthogonality could be an artefact of the instrument. With Shapley values in hand the
question becomes geometric: does the dictionary's nameable structure live where the
causal mass lives?

For each of the top 32 dictionary atoms, the fraction of its energy inside three
10-direction spans of the same top-32 SVD subspace:

| span | mean atom energy | usage-weighted |
|---|---|---|
| top-10 Shapley directions | **0.238** | **0.288** |
| random 10-subsets (200 draws) | 0.177 [0.143–0.214] | — |
| bottom-10 Shapley directions | 0.132 | 0.135 |

Top beats the 95th percentile of the null; top/bottom ratio 1.80×; weighting by atom
usage strengthens it (0.288), meaning the *most-used* atoms are the most concentrated on
the causal directions.

**The nameable atoms preferentially occupy the causally responsible subspace.** §78's
"naming does not explain the hub" conclusion was at least partly an artefact of scoring
causality with solo ablation. The corrected statement: the dictionary's atoms do point
where the causal mass is — what remains true from §78 is only that no *single* atom is
individually load-bearing, which §13 explains as interference among ~10 genuine parts
rather than absence of parts.

Caveats stated rather than buried: the alignment is a 1.8× enrichment, not identity
(atoms still hold most of their energy outside the top-10 span); the dictionary explains
0.82 of output variance (FVU 0.179), so a fifth of the structure is unmodelled; and this
is one layer of one model.

---

## 15. What the ten directions are — token structure, but not crisp names

File: `bilin18_mlp1_leaders.py` (11 s).

§8.3's verified naming instrument (unembedding alignment tested against measured
excitation over 262k positions, 200-draw permutation null), pointed at the six leading
Shapley directions of MLP1.

**A methodological catch first, recorded because it will bite anyone repeating this: the
direction basis must come from the same data as the Shapley run.** The first two
versions of this script recomputed the output SVD from their own (larger) corpora, and
the tail directions visibly rotated — "direction 1" changed its unembedding profile
between runs that should have differed only in statistical power. Caught because the
`writes toward` lists changed; fixed by taking the basis from the Shapley run's exact
data and using the large corpus only for excitation.

All six clear their permutation nulls (ρ up to 0.47 against nulls of ~0.06) — the
excitation of every causally-leading direction has genuine token structure. But the
honest reading is weaker than "named":

| rank | share | writes toward | fires on (measured) | ρ |
|---|---|---|---|---|
| #1 (27%) | pronouns/numerals ` one, it, I, 2, they, we` | overwhelmingly plain space (16.3×; everything else ≤1.3×) | 0.24 |
| #2 (9%) | concrete nouns ` grade, line, circuit, plate`; against punctuation | word-fragment suffixes `ines, onse, ower` | 0.29 |
| #3 (9%) | ` case, method, solution, way`; against `The/the` | ` largest, problem, look, simple` | 0.11 |
| #4 (5%) | **`the/The`, against finite verbs & pronouns** | markup ` <, ([, }, </` | **0.47** |
| #5 (5%) | suffix fragments | fragments/space | 0.29 |
| #6 (5%) | punctuation/suffixes | **sentence-initial discourse openers `As, For, To, One, If`** | 0.29 |

Some of this is legible — #4 is a determiner-vs-verb axis (exactly the kind of feature
layer 17 also carries), #6 keys on sentence-initial position, #1's firing is dominated
by whitespace/indentation. But none of the six has the crispness of layer 17's verified
features, and several read as morphological/positional rather than semantic.

**This does not overturn §74's "0/32 nameable", and should not be quoted as doing so.**
§74's bar is causal-clearness (mean-ablation effect z ≥ 3 localised at the direction's
own top firing positions) — a *causal* criterion. This instrument's bar is statistical
token-dependence of excitation — a *correlational* one, and weaker. The combined honest
statement: the causally-leading directions of MLP1 have real, verifiable token structure
(all six, p < 0.05 against permutation), but that structure is diffuse in exactly the
way §74's stricter causal bar registers as unnameable. Both instruments are right; they
measure different grades of "having a name".

---

## 16. The sample-size control — §13's count was inflated, the weight basis is not exonerated

Files: `bilin18_shapley_bigdata.py` (472 s), plus the length-confound split (scratch).

Prompted by the observation that every §13–§15 number rests on a PCA basis fit on 32,768
positions. Refit on 153,900 positions (4.8×, rows disjoint from the evaluation rows),
re-derive everything.

### 16.1 What survived and what did not

| | 32k basis (§13) | 154k basis | verdict |
|---|---|---|---|
| joint effect of the span | 0.383 | 0.434 | stable |
| participation ratio | 9.5 | **5.6** | **§13's count was inflated** |
| leader share | 27% | **39%** | leader underestimated |
| top-8 share | 67% | 72% | stable |
| leader identity (dirs 0, 1) | — | cos 0.985, 0.946 | rock solid |

**Correction to §13: "about ten directions" was an overestimate; the better-estimated
basis gives five-to-six, with the leader carrying 39%.** The qualitative verdict — an
oligarchy with a clear leader, neither a smear nor a single star — survives and in fact
sharpens. The count did not.

The mechanism of the inflation is visible in the basis correspondence: old directions 6
and 4 — Shapley ranks #3 and #5, with 9% and 5% of the layer — **both** best-match new
direction 3 (cos 0.41 and 0.62). The small-sample PCA split one real component into two
noisy ones, and the Shapley attribution dutifully spread that component's mass across
both, pushing the participation ratio up. Sampling noise in the basis *inflates* the
apparent number of parts. §15's naming of ranks #3 and #5 is correspondingly
contaminated (two names for one thing); ranks #1, #2 and their names stand.

### 16.2 The weight basis: not exonerated, but its standing is unchanged, and the
in-sample numbers were misleading for a different reason than overfitting

Held-out output energy (fresh rows, neither basis fit on them): data-154k **37.5%**,
data-32k 35.4%, weight SVD 23.5%. Two things:

- **4.8× more data moves the data basis by 2 points** (35.4 → 37.5). The basis estimate
  was already essentially converged at 32k; the inflation in §16.1 lives in a handful of
  tail directions (27/32 principal cosines > 0.9, min 0.007), not in the subspace.
- **The in-sample numbers (70.5% data / 44.3% weight) were roughly double the held-out
  ones — but by row-group heterogeneity, not overfitting.** The tell: the weight basis
  fits *nothing*, yet drops by the same factor (×0.53) as the data basis (×0.51) on the
  new rows. A context-length split confirms length is irrelevant (36.3% at 128 vs 35.4%
  at 513). Different document groups simply concentrate different amounts of MLP1's
  output in *any* fixed 32-dim subspace. Consequence: every "X% of the layer" figure in
  §12–§15 is row-group-relative; ratios between bases are trustworthy, absolute
  percentages are not.

The weight verdict, updated: 60% of the data span's causal effect (0.261/0.434) and 63%
of its held-out energy — both essentially identical to the small-basis numbers (59%,
64%). **More data neither exonerates nor further indicts the weights; the data basis's
~1.6× edge is real, held-out-stable, and not an artefact of sample size.**

---

## 17. What the data is, and what writes the causal directions

Files: `bilin18_data_structure.py` (6 s), `bilin18_source_folding.py` (5 s).

Two questions asked together because the answer to "which compression is right" depends
on both: what shape MLP1's output distribution has, and what upstream writers actually
drive its causal directions.

### 17.1 The data: not a dense 32-dim subspace — a mixture with a long tail

Measured on 153,900 positions, held-out rows for anything fittable:

- **Spectrum.** One enormous direction (effective rank of the full 1152-dim output: 3)
  followed by a long, slowly-decaying tail: 90% of energy needs ~241 dims, and held-out
  energy at k = 8/32/128/512 is 19/38/60/88%. The top-32 slice §12–16 worked in is a
  reasonable *causal* slice but is nowhere near the whole distribution. "Dense low-rank
  subspace" is the wrong model.
- **Sparsity: modest, not SAE-shaped.** Excess kurtosis along the top-32 PCA directions
  has median **1.2** (Gaussian = 0), with a handful of heavy directions (up to 27).
  Random directions score 3.7. This is not the strongly leptokurtic regime that makes a
  global sparse dictionary the right code — most coefficients are near-Gaussian with a
  few heavy-tailed exceptions.
- **Document mixture: the real structure.** The leader's coefficient has **ICC 0.56 by
  document** — more than half its variance is *which document you are in*, not which
  token. Directions 2–8 sit at 0.02–0.14. Combined with §15 (the leader fires almost
  exclusively on whitespace): the leading causal direction of MLP1 is substantially a
  **document-register feature** (whitespace-heavy material — code, tables, lists —
  against prose).
- **No hierarchy.** Mean |off-diagonal| Spearman of the top-8 magnitude co-activations
  is 0.10; the leader *anti*-correlates with tail energy (−0.30), which is what a
  register mixture predicts (in whitespace documents the prose-tail goes quiet) and what
  gating would contradict.

### 17.2 The fold-in: exact for a bilinear layer, and the leader is not a token feature

The residual stream at MLP1's input is an exact sum of four writers — embedding path
(with the per-block λ re-injections), attn0, MLP0, attn1 — and the per-position rms
scalar divides each, so `xhat = Σ xhat_a` exactly. Because the layer is bilinear, each
output coefficient then splits exactly into writer pairs,
`c_d = Σ_{a≤b} (2−δ) xhat_a^T M_d xhat_b` (reconstruction gates 3e-7–6e-7; measured, not
assumed). Variance shares of `c_d`, top pairs:

| | attn1×attn1 | mlp0×attn1 | emb×attn1 | emb×mlp0 | mlp0×mlp0 |
|---|---|---|---|---|---|
| direction 0 (39% leader) | **76.1%** | 12.6% | 9.3% | 0.5% | 0.9% |
| direction 1 | 17.3% | **32.1%** | 11.4% | 14.9% | 13.3% |
| direction 5 | 22.1% | **23.1%** | 12.2% | 18.7% | 14.2% |

**The leader is an attention-squared feature.** 76% of its variance is attn1's output
interacting with itself — the current token's embedding contributes 9% at most through
cross terms. So the picture assembled across §15–17 is coherent: attn1 aggregates a
summary of the local context; MLP1 squares it; the result is a register signal that is
half document-identity by variance. The leading causal object in this layer is **not a
token feature and cannot be named by token lists** — the §15 instrument found its
correlate (whitespace) without being able to say what it was.

Directions 1 and 5 are genuinely mixed — every writer pair contributes 11–32% — which is
what "diffuse token structure, no crisp name" (§15) looks like mechanically.

The embedding-only curvature `rmsnorm(wte_t)^T M_d rmsnorm(wte_t)`, computable for the
whole vocabulary from weights alone, adds a per-token layer to each: direction 0 is
positive on function words and whitespace, negative on content nouns; direction 1 is
negative on all punctuation, positive on content nouns; direction 5 is negative on
sentence-initial discourse openers (` As, So, When, After`) — each consistent with its
§15 excitation profile.

### 17.3 What this says the right compression is

The measured structure rejects both default codes. A global sparse dictionary is wrong
because the coefficients are mostly near-Gaussian (median excess kurtosis 1.2); plain
PCA + Gaussian is wrong because the distribution is a document mixture with a
heavy-tailed minority. The MDL-shaped recommendation, from the measurements rather than
from preference:

1. **Condition on register first.** A small discrete context state (predictable from
   attn1's output; the leader is 56% document identity) should be the first code
   symbol. Coding the register once per document region is nearly free and removes the
   single largest variance component.
2. **Within register, a moderate-rank Gaussian code** for the near-Gaussian bulk
   (the spectrum says ~100–250 dims for the full distribution, ~5 for the causal slice).
3. **A sparse exception code** only for the few heavy-tailed directions (kurtosis > 5),
   which is where SAE-style atoms genuinely fit.
4. **Interpretation should factor through the writers**: the exact pair decomposition
   means "what does direction k mean" reduces to "which writer pair drives it, and what
   does *that writer* respond to" — for the leader, the next unfold is through attn1's
   value circuit (which past tokens, with what pattern weights), not through the
   current-token vocabulary.

---

## 18. The leader unfolded: one head, squared

File: `bilin18_leader_unfold.py` (6 s; exactness gates 5e-7 head level, 3e-7 key level).

§17 said the register leader is attn1's output squared. Splitting that exactly by head
pair (attention output is linear in heads through c_proj, so the quadratic splits into
an exact 9×9 grid), then attributing each key position's exact share:

- **Head 4 of layer 1, squared, is 90.4% of the leader's variance.** The next term
  (head 1 × head 4) is 5.0%. One head carries the register feature almost alone.
- **The keys that drive it are layout characters.** Attending to ` `, ` <`, `</`, `\r`,
  `\n\n`, `#`, `$`, `%`, `!` pushes the signal up; attending to `The`, `).`, ` we`,
  ` my`, `.`, ` and` pushes it down. The feature is literally "how much of the attended
  context is markup/layout rather than prose".
- **It is a local-context feature**: half the attribution mass lies within 15 tokens;
  offsets 0–4 carry 16%, beyond 64 only 19%.

Full mechanistic sentence, every step exact: *MLP1's dominant causal direction squares
head 4's aggregation of layout tokens in the recent context, producing a
document-register signal that carries 39% of the layer's causal effect.* Every clause
above is a measurement (§13/§16 for the 39%, §17 for the writer, this for the head, the
keys, and the range).

---

## 19. Phase A verdict: the compression is real, the register semantics are not causal

File: `bilin18_leader_verify.py` (12 s). *(A first run of A1/A2/r1 was void — it patched
`mlp1.forward`, which the custom forward never calls, and silently scored the intact
model. Caught because every rung reported identical CE. Fixed with an explicit hook in
the forward; r2/r3 never used patching and were valid in both runs.)*

### 19.1 The compression claim: causally verified, red-teamed, MDL-scored

Deleting only the leader direction (mean-ablating one coefficient of one layer) costs
+0.0158 nats. The ladder of replacements, all fit on rows the evaluation never sees:

| replacement for c₀(x) | params | CE vs base | repairs |
|---|---|---|---|
| delete (mean) | 0 | +0.0158 | 0% |
| **story surrogate: a·(u·x̂)² + b** | **1,154** | +0.0012 | **92.1%** |
| **rank-2 whitened truncation** | **2,308** | −0.0001 | **100.9%** |
| full form | 664,128 | 0 | 100% |

**A 2,308-parameter object replaces a 664,128-parameter one at full functional
fidelity — 288× smaller — and 1,154 parameters get 92%.** The red-team control that
makes this meaningful: the *same* surrogate form with a random unit direction (a, b
refit) repairs **0.0%**. The win is entirely in the direction u, which was derived from
the story (top whitened eigenvector of the form restricted to the attn1 component).
This is the causal MDL result the plan asked for.

One nuance worth its ink: the surrogate's raw coefficient fit R² collapses from 0.984
on fit rows to **0.265** on unseen documents (r2) — yet its CE repair, measured on
those same unseen documents, is 92%. Both numbers are right. The R² drop is dominated
by the document-level variance component (ICC 0.56), and that component evidently does
little work for next-token prediction; the part of c₀ that matters downstream is the
part u carries across documents. A fidelity metric on the *coefficient* and one on the
*function* disagree, and the function is the one that matters.

### 19.2 The semantic claim: fails both causal tests

The register story made two causal predictions. Both failed.

**A1 — damage should concentrate in layout-heavy contexts.** It does the opposite:
binned by the layout fraction of the trailing 32 tokens, deletion damage is +0.0165 /
+0.0186 / +0.0136 / **+0.0066** from prose to most-layout — *smallest* where the story
says largest. The control direction shows the reverse (rising) profile, so the
instrument can detect gradients; the leader's just goes the wrong way.

**r3 — injecting layout into prose should raise the leader.** Injecting 24 layout
tokens shifts the leader by −26 ± 2 against a natural std of **3,449** — under 1% of a
standard deviation, *negative*, and statistically indistinguishable from injecting
random prose words (−28 ± 2). Stripping all layout from markup documents shifts it
+27 ± 9 — also the wrong sign for the story, also negligible. Token-level layout does
not causally drive this direction at any scale the intervention can produce.

**Reconciliation with §18, which is a methods lesson.** The per-key attribution
(layout keys strongly positive) is exact *on the natural data distribution* — it
decomposes covariance, and layout keys covary with high-leader documents. But
attribution over a distribution is still correlational *across* that distribution:
inserting isolated layout tokens into prose does not recreate the document context in
which head 4's layout-attention naturally occurs. **An exact decomposition is not an
intervention**, and this is now the program's cleanest demonstration of the gap.

### 19.3 The corrected story

What survives, all of it causal: MLP1's dominant direction is a cheap function of
attn1's output — one-to-two squared projections, 288× smaller than its parameter count,
92–101% of its function — whose *correlates* are register-like (document identity,
layout density) but whose *causal role* is not the register story: its deletion damage
sits in prose, and layout tokens do not drive it interventionally. "Register detector"
is demoted from mechanism to correlate. What it actually computes for the network
remains open; what is closed is that you can swap it for 2,308 numbers.

---

## 20. Phase B: layer 0 at full depth — the clean layer

File: `bilin18_layer0_battery.py` (459 s; writer-decomposition gate 8e-8). Basis fit on
153,900 positions, rows disjoint from evaluation, §16's lessons baked in from the start.

**Concentration (Shapley, 20 permutations).** Joint effect of the top-32 span +0.126
nats; participation ratio 8.1 of 32; leader 28%, top-4 55%. Same oligarchic shape as
layer 1, a little flatter.

**Writers.** All three leading directions are **embedding-dominated**: emb×emb carries
75% / 67% / 62% of their variance, attention pairs the rest, and no attention head
exceeds 6%. Exactly what bottom-up predicts at depth 0 — these are token-identity
features, and the input side is fully characterizable: each leader is, to first order, a
function of the current token alone.

**Names — including the program's strongest verified name so far.**

| leader | share | fires on (measured) | ρ vs null |
|---|---|---|---|
| dir 1 | 28% | ` (`, `).`, `.`, `\n`, `!`, `?`, `..`, `),` | **0.950** / 0.139 |
| dir 8 | 11% | ` 10`, ` first`, ` not`, `3`, ` more`, ` no`, ` one` | 0.797 / 0.153 |
| dir 3 | 9% | `.`, ` make`, `!`, ` work`, `;` | 0.803 / 0.175 |

The layer-0 leader is a **punctuation-vs-content axis** (curvature positive on nouns
like ` Boeing, Marvel, PlayStation`, negative on `).`, ` (`, `.`), verified at ρ = 0.95
against a 0.14 null — the crispest feature naming in the program. The second is a
number/quantifier axis. Layer 0 has what layer 1 lacked: nameable variables.

**Structure.** Effective rank 24 (layer 1's was 3), dims for 50/90%: 19/430, kurtosis
0.8–5.3, ICC ≤ 0.07 except one direction at 0.39. Layer 0's output is a moderate-rank,
near-dense, token-driven code with little document-mixture — the opposite regime from
layer 1, and the right regime for PCA+Gaussian description.

**Causal MDL ladder.** Deleting the leader costs +0.0153; the 1,154-parameter surrogate
repairs **65.8%**, rank-2 63.7% (tied within noise), random-u control 0.6%. So layer 0's
leader is genuinely *less* compressible than layer 1's (66% vs 92–101%): a
token-identity feature over a 50k vocabulary intrinsically uses more directions of its
form than a squared context summary does. The compression each layer admits tracks what
it computes — which is itself a finding: **surrogate compressibility is a probe of
mechanism class** (context-summary features compress to 1–2 terms; token-identity
features do not).

---

## 21. Phase A′: the variable graph under interchange interventions

File: `bilin18_interchange.py` (8 s). The §19 surrogate implies a causal abstraction:
`z := u·x̂` (computed, per §18's attribution, mostly by attn1 head 4) → `c₀ := az²+b` →
`write := c₀d₀`. §19 tested only the output. Interchange interventions test each edge:
patch the variable's value from a *source* input into a *base* input and demand the
downstream behaviour match patching the low-level realizer. Base and source pairs come
from different documents, which makes this deliberately the hard regime (§16/§19 showed
cross-document transfer is where fits break).

**E2 — the head4 → z edge: verified interventionally.** Swapping head h's attn1 context
(pre-projection) from source into base, one head at a time, and measuring how much `z`
moves: **head 4 produces 79.1%** of all z-movement; the runner-up (head 1) 14.3%; no
other head above 1.7%. §18's on-distribution attribution said 90%. Unlike the layout
semantics — where attribution and intervention disagreed — this edge holds up when
actually moved. "Head 4 computes z" is now an interventional claim.

**E1 — the z → c₀ edge: a partial abstraction, and honestly scored.** Replacing
`c₀(base)` with the true `c₀(source)` changes downstream log-probs by KL 0.0109
(top-half positions) and flips 218 top-1 predictions. Replacing it with the
abstraction's value `a·z(source)²+b` instead reproduces that downstream behaviour at
**68.0% faithfulness** (KL mismatch 0.0035) and matches **61.0%** of the flipped top-1
predictions. The shuffled-c control sits at 13.3%, so the pairing is doing real work —
but 68% is far from the 92% CE repair §19 measured on-distribution.

The gap between 92% and 68% is informative rather than embarrassing: interchange
transports values *across documents*, which is exactly where the surrogate's coefficient
fit degrades (transfer R² 0.265, §19). On-distribution, z carries c₀'s function almost
completely; transported across contexts, about a third of c₀'s downstream influence
comes from parts of the form z does not see. The abstraction is real but leaky, and the
leak is localised to cross-context generalisation — a sharper statement than either
previous number alone.

Graph status after Phase A′:

| edge | claim | test | verdict |
|---|---|---|---|
| head4 → z | head 4 computes z | context interchange | **79% of z-movement — holds** |
| z → c₀ | c₀ = az²+b abstracts the form | value interchange | **68% faithful (13% control) — partial** |
| c₀ → CE | the write is the leader's causal role | ablation + ladder (§19) | **holds, 288× compression** |
| layout → z | layout tokens drive z | token injection (§19 r3) | **refuted** |

---

## 22. Phase C: layer 16 — two directions run the layer, and a surrogate that beats the model

File: `bilin18_layer16_battery.py` (461 s, generated from the layer-0 battery with the
writer tracker swapped to coarse groups: embedding path / all-attention / all-MLPs,
tracked exactly through the 16-block recurrence; gate 1.8e-7).

**Concentration: the extreme of the depth profile.** Participation ratio **2.5** of 32
(layer 0: 8.1, layer 1 refit: 5.6). The top two directions carry 42% and 40%; the third
25%; top-4 share is 109% because tail values go negative. Layer 16 is effectively a
**two-direction layer**, which independently corroborates §10's finding that R=4, k=2
replaces it at 4.2% damage.

**Writers: deep cross-terms, embedding irrelevant.** All three leaders are dominated by
**attn×mlps** cross-variance (47–56%), with emb×emb at 0–1% and no single head of block
16's own attention above 0% variance share. These are accumulated-computation features —
the interaction of everything attention has gathered with everything the MLPs have
written — which is what "deep feature" means mechanically, and the opposite of layer 0's
emb-dominated leaders.

**Names: present but weak** (ρ 0.26–0.39 against nulls ~0.11–0.17). Leader #2 is the
recurring syntax axis: curvature positive on pronouns/copulas (` we, can, she, I, he`),
negative on sentence-enders (`. ). ! ? :`), firing on exactly those enders — the same
axis §8.2 read out of layer 17's output direction 2. It is visible at layer 16 too,
which suggests it is a *bus signal* maintained across the late layers rather than a
single layer's property — a concrete cross-layer variable for the causal-abstraction
program.

**Structure**: effective rank 3, dims for 50/90% = 2/15, low kurtosis, mild ICC
(0.12–0.23). A genuinely low-rank, near-Gaussian code — the easiest regime in the model.

**The B5 anomaly, reported rather than smoothed over.** Deleting the leader costs
+0.0337. The 1,154-parameter surrogate does not merely repair it — it lands **0.0252
nats BELOW the intact baseline** (repair "174.7%"); rank-2 also edges the baseline
(−0.0024). The random-u control behaves normally (repairs 9.5%), so this is not an
artifact of the patching machinery. A replacement beating the model it replaces means
the full form carries a component that *hurts* on this evaluation set. The obvious
hypothesis, registered not asserted: the model was trained on fineweb, this evaluation
is pile-10k, and the rank-1 whitened core generalises across the shift while the
discarded 664k-parameter remainder is distribution-specific — truncation as
regularisation. Testable by rescoring on fineweb-like data (the original token file is
not on this box); until then the honest statement is that layer 16's leader is
over-parameterised for out-of-distribution prediction, in the direction the whitened
metric identifies.

### The emerging depth taxonomy

| layer | leader mechanism | concentration (PR/32) | surrogate repair | naming |
|---|---|---|---|---|
| 0 | emb×emb — token identity | 8.1 | 66% | **ρ 0.95** (punctuation axis) |
| 1 | attn1² — squared context summary | 5.6 | 92% | diffuse; register correlate |
| 16 | attn×mlp — accumulated computation | **2.5** | **175%** (beats baseline) | ρ 0.3 (syntax bus) |
| 17 | near-rank-4 output; syntax rules | — | 99% at rank 2 (§8) | verified, uneven |

Compressibility, writer profile, and nameability all shift together with depth — the
battery is measuring mechanism class, not just size.

---

## 23. Phase D: the weights knew — one data matrix from prediction

File: `bilin18_theory_pass.py` (5 s). The bilinear MLP is a third-order tensor
`T = Σⱼ Downⱼ ⊗ Leftⱼ ⊗ Rightⱼ`, and its output-mode Gram has closed form:
`G_plain = Down[(LLᵀ)∘(RRᵀ)]Downᵀ` from weights alone, and
`G_lam = Down[(LSLᵀ)∘(RSRᵀ)]Downᵀ` with one data statistic — the input second moment S.
(`G_lam` is what Isserlis gives for a Gaussian input with that second moment; it is the
Λ-weighted object the toys have used since `bq_common`.) The question: how much of the
program's expensive empirical pipeline was already in the weights?

**D1 — the causal leader was predictable at every depth tested.** Energy of the
*measured Shapley leader* inside the top-8 eigenvectors of each Gram:

| layer | plain weights | **weights + S** | random |
|---|---|---|---|
| 0 | 0.140 | **0.898** | 0.009 |
| 1 | 0.594 | **0.983** | 0.015 |
| 16 | 0.569 | **0.981** | 0.011 |
| 17 | 0.958 | **0.996** | 0.004 |

The Λ-weighted Gram — pure linear algebra on the weights plus a single 1152×1152
matrix of data — holds 90–99.6% of the causal leader in its top-8, at all four depths.
Plain weights alone degrade with shallowness (0.96 at layer 17 down to 0.14 at layer 0,
where the input distribution is furthest from isotropic). **The entire
Shapley-and-basis pipeline, for the leader, was one Gaussian-equivalent computation
away from the weights.** The empirical top-8 output basis is also substantially
predicted (energy 0.47–0.79 lam vs 0.005–0.008 random).

**D2 — the head was predictable too.** The per-head folded operators
`B_h = W_proj,hᵀ M_d0 W_proj,h` — 128×128 matrices, pure weight algebra — give head 4 a
61.2% squared-norm share at layer 1, ranked first by a factor of 5 over the runner-up.
The interchange measurement said 79%. The expensive discovery of §18/§21 (which head
computes z) required no forward passes to find, only to *verify*.

**D4 — the tensor-network accounting.** The verified layer-1 leader surrogate
(`z = u·x̂; c₀ = az²+b; write = c₀d₀`) is a three-node network: 2,306 parameters and
2,307 flops/token against the layer's 15.9M — **6,903× cheaper** — with §19's measured
92% on-distribution fidelity and §21's 68% interchange faithfulness as its quality
certificates. Layer 16's four-direction rank-2 replacement: 13,832 parameters at 4.2%
damage.

**What the theory pass changes going forward.** The battery order inverts: compute
`G_lam` and the folded head operators *first* (seconds, closed-form), take their top
components as the hypothesis set, and spend the model evaluations only on Shapley
*verification* and interchange tests of weight-derived candidates. The measurement
budget so far was spent discovering things the weights already contained; from here it
should be spent testing them.

---

## 24. The syntax bus: a verified two-layer variable, with a rectified edge

File: `bilin18_syntax_bus.py` (5 s). §22 noticed the same pronoun-vs-sentence-ender
axis in layer 16's #2 causal direction and layer 17's #2 output direction. That was an
observation; this makes it a tested graph edge.

**T1 — the pair is special.** corr(c16-axis, c17-axis) = **0.935** across positions.
Other layer-16 directions against the same layer-17 coefficient: median |r| 0.131, max
0.284. The two sites carry the same signal to first approximation.

**T2 — the edge is causal, specific, and one-sided.** Steering c16 by +2σ at every
position moves c17 by **+0.95σ** — essentially unit gain — while a causally-irrelevant
layer-16 direction steered identically moves it 0.035σ (**14.8× specificity**). But
steering by −2σ moves c17 by only −0.04σ: the edge is **rectified**. That asymmetry is
exactly what the architecture predicts: layer 17's coefficient is a *quadratic* form,
and a squared readout of a shifted input responds linearly on one side of its operating
point and flattens near the vertex on the other. The bilinear mechanism is visible in
the intervention's shape — a structural signature no linear-circuit picture would
produce.

**T3 — the token-level semantics fail specificity, again.** Steering ±2σ moves
determiner log-probs by large amounts (+1.45/−0.98), but matched control tokens move
*more* (+1.71/−1.53). At this magnitude the intervention shifts the whole common-token
distribution rather than the determiner slice — so "writes toward determiners" is not
supported as a specific causal effect at the steering scale tested. Two caveats
recorded: ±2σ at every position is a large, out-of-distribution intervention (log-prob
shifts of 1.5 nats say the model is far from its operating regime), and a
smaller-magnitude, positionally-targeted version might yet show specificity. As
measured: unsupported.

**The pattern across the program is now three-for-three**: structural/graph claims
(head4→z, the L16→L17 bus, compression ladders) verify under intervention; token-level
semantic claims (layout drives the register leader, the bus writes determiners) fail
specificity every time they are tested causally. The reliable currency of this model's
mechanisms is *directions, edges, and gains* — not token stories. That is itself a
finding about what interpretability claims this architecture licenses.

---

## 25. The anomaly resolved: truncation as regularisation, confirmed on fineweb

File: `bilin18_l16_anomaly.py` (79 s; fresh fineweb sample streamed from the Hub,
90×257 tokens, saved as `fineweb_eval_tokens.pt`).

§22's registered hypothesis for the layer-16 anomaly — the surrogate beats the intact
model because the eval corpus (pile) is shifted from the training corpus (fineweb), and
the discarded remainder is distribution-specific — made a sharp prediction: the
improvement should vanish on fineweb. It does, cleanly:

| corpus | baseline | delete | surrogate − base | rank-2 − base |
|---|---|---|---|---|
| pile (shifted) | 4.0345 | 4.0589 | **−0.0285** | −0.0001 |
| fineweb (training-like) | 3.0837 | 3.1253 | **+0.0011** | +0.0004 |

(The pile improvement replicates on fresh rows, so §22's number was not a fluke; and
baseline CE 3.08 vs 4.03 confirms fineweb is the in-distribution corpus.)

**On the training distribution, the 1,154-parameter surrogate and the 664,128-parameter
form are functionally equivalent** (+0.0011 nats, within noise). The remainder
contributes nothing in-distribution and *hurts* under shift. So the compression result
strengthens from "92–100% fidelity" to: **the leader's effective content is the
surrogate; the other 663k parameters are dead weight in-distribution and a liability
out of it.**

This also hands the program a general protocol: score surrogate-vs-full on both the
training-like and a shifted corpus, and the difference measures how much of a
component is distribution-robust computation versus distribution-specific fit. Layer
16's leader: entirely robust core. Worth running on the other layers' surrogates
(registered, not yet run — layer 1's 92% and layer 0's 66% may decompose differently).

---

## 26. The robustness split for all three verified leaders

File: `bilin18_robustness_split.py` (17 s). §25's protocol applied to layers 0 and 1,
with registered expectations: layer 1's missing 8% should close in-distribution (its
leader is a context summary and its gap looked shift-shaped); layer 0's missing 34%
should not (token identity does not shift between corpora).

| layer | corpus | delete − base | surrogate − base | repair |
|---|---|---|---|---|
| 0 | pile | +0.0120 | +0.0048 | 60.1% |
| 0 | **fineweb** | +0.0065 | +0.0016 | **76.1%** |
| 1 | pile | +0.0067 | **−0.0007** | 109.8% |
| 1 | **fineweb** | +0.0121 | +0.0016 | **86.8%** |
| 16 (§25) | pile | +0.0244 | −0.0285 | >100% |
| 16 (§25) | fineweb | +0.0416 | +0.0011 | ~100% |

**Layer 1 shows the same shift pattern as 16**: on pile its surrogate now *beats* the
full form (109.8% on these rows), in-distribution it genuinely misses 13%. **Layer 0's
gap does not close** (76.1% in-distribution) — as registered, a token-identity feature
really does use more of its form than any rank-1 surrogate carries, on any corpus.
(Both registered expectations held qualitatively; layer 1's "should close" closed to
87%, not 100%.)

Two structural observations the table adds:

- **In-distribution missing fraction orders by mechanism class**: token identity (24%)
  > context summary (13%) > accumulated computation (~0%). The deeper and more
  aggregated the feature, the more completely a rank-1 whitened core captures it.
- **Leaders matter more in-distribution where they are deep**: layer 1 and 16's delete
  costs are ~2× larger on fineweb than pile, layer 0's is 2× smaller. Shallow
  token-features carry relatively more of the load on shifted text; deep context
  features carry more on the training distribution.

The distribution-robust core of every verified leader is its rank-1 whitened surrogate;
every remainder is neutral-to-harmful under shift. That is now a three-layer regularity,
not a curiosity.

---

## 27. The middle, attributed fairly — and a correction to the 2.87× headline

Files: `bilin18_middle_shapley.py` (673 s, 20 permutations under a clean operator),
plus the operator check (scratch).

### 27.1 Correction: the superadditivity is 1.42×, not 2.87×

Cross-checking this run's gates against §10.2 exposed two defects in that measurement.
Its numerator and denominator used **different deletion operators** — the solo costs
came from §9's machinery (which writes the layer's mean *minus its component in the
top-R output span*, R ≤ 48) while the joint deletion used a top-1-PC variant — and the
joint run computed each layer's mean **on a model whose earlier layers were already
ablated** (stale means). Under one clean operator (write the exact intact-model mean;
no span removal, no staleness), on intact means throughout:

| | §10.2 (mixed operators, stale means) | clean operator C |
|---|---|---|
| sum of 14 solo deletions | 1.790 | 2.087 |
| all 14 deleted together | 5.142 | 2.963 |
| **superadditivity** | **2.87×** | **1.42×** |

The qualitative claim — the middle is jointly more than its parts, so one-at-a-time
ablation understates it — survives. The magnitude was inflated 2×. Everything
downstream that quoted 2.87× (§12.3's analogy, RESULTS, the report chart) carries this
correction. A general lesson worth the ink: **"delete cost" is operator-dependent** —
§9's span-removal operator distorts the layer's *mean* along with its variation, and
per-layer costs move by up to 3.5× between operators (layer 4: 0.381 under A, 0.110
under C). Any ablation number needs its operator stated.

### 27.2 The fair shares: two working layers, one saboteur, eleven understudies

Shapley over the fourteen quadratic parts, everything under operator C:

| layer | solo | Shapley | amplification | share |
|---|---|---|---|---|
| 2 | +0.679 | +0.861 | 1.3× | 29.1% |
| 3 | +0.801 | +1.024 | 1.3× | **34.6%** |
| **4** | +0.110 | **−0.668** | — | **−22.5%** |
| 5–15 (each) | 0.017–0.071 | +0.09–0.22 | **2.9–5.4×** | 3–7% |

Participation ratio 3.5 of 14. Three structural facts:

1. **Layers 2 and 3 are the middle.** Together 64% of the fair share — the "distributed
   middle" of §10 is mostly two adjacent early layers plus a long tail. With §9's
   layer-1 result, the model's MLP story is heavily front-loaded: layers 1–3 dominate.
2. **Layer 4's Shapley value is large and negative** (−0.668, robust across both
   operators). Averaged over deletion contexts, *removing layer 4's quadratic part
   repairs part of the damage of removing the others*. Its computation is useful only
   when its downstream partners are intact; with them gone, its output is actively
   harmful. That is a strong, testable coupling signature — layer 4 writes something
   that later layers must process — and no solo or joint number could have shown it
   (solo: +0.110, unremarkable).
3. **The tail's importance was uniformly hidden**: layers 5–15 carry 3–7% each at
   amplification 2.9–5.4×. Solo ablation understated every one of them, by more the
   later they sit.

---

## 28. Layer 4's coupling located — and my §27 hypothesis was backwards

Files: `bilin18_layer4_coupling.py` (13 s) + backward weight check (scratch).

§27 read layer 4's negative Shapley value as "layer 4 writes something later layers
must process." The interventional test inverts that reading.

**T1 — the marginal flip locates the partner, and it is upstream.** Layer 4's marginal
deletion cost inside coalitions, operator C:

| coalition already deleted | m(4 \| S) |
|---|---|
| ∅ | +0.110 |
| {5} … {5..9} | +0.20 → **+0.28** (rises) |
| {5..15} | +0.226 |
| **{2,3}** | **−1.335** |
| {2,3,5..15} | −3.232 |

Deleting the *downstream* layers makes layer 4 more valuable, not less — the forward
consumers hypothesis is dead. The sign flips, violently, exactly when **layers 2–3**
enter the coalition: with its upstream suppliers gone, layer 4's computation becomes
toxic (−1.34 nats of extra damage). **Layer 4 is a reader of layers 2–3.** Its §27
negative Shapley came entirely from coalitions containing {2,3}; averaged over orders,
that dominated. The correct pipeline statement, interventionally grounded: layers
**2 → 3 → 4** form a front-loaded chain at the entrance to the middle, and layer 4's
quadratic is tuned to operate on what 2–3 write — on an unwritten bus it misfires and
amplifies damage.

**T2 — the weight-side Gram predicts both directions correctly.** The input-mode Gram
of layer 4's tensor in the Λ metric ranks the writers it is sensitive to: **layer 3 at
0.097 (12× random), layer 2 at 0.063 (8×)**, above layers 1, 5, 6 (0.03–0.04). And the
forward direction exists but is weaker: layer 5 reads layer 4's output at 10× random,
decaying to 2× by layer 9. Closed-form weight algebra, seconds, agreeing with the
interventional flip on both the direction and the ranking of the edge — the Phase-D
protocol's second confirmed prediction.

The method note that generalises: **a negative Shapley value identifies a coupled
stage but not the direction of coupling** — the marginal-flip sweep (which coalition
flips the sign) is the cheap follow-up that orients the edge, and the Λ-metric
input-mode Gram predicts the answer from weights before any model evaluation is spent.

---

## 29. Layers 2 and 3: where the model refuses to compress

Files: `bilin18_layer2_battery.py`, `bilin18_layer3_battery.py` (461+466 s, gates ~1e-7).
The two layers carrying 64% of the middle's fair share (§27), given the full battery.

| | layer 2 | layer 3 | (layer 16, for contrast) |
|---|---|---|---|
| Shapley PR / 32 | 8.7 | **13.8** | 2.5 |
| leader share | 22% | **16%** | 42% |
| output effective rank | **87** | **80** | 3 |
| dims for 90% of output | 545 | 509 | 15 |
| leader's rank-1 repair | **3.5%** | 67.8% | ~100% |
| leader's rank-2 repair | 17.6% | 68.7% | ~100% |
| leader naming ρ | 0.47 | **−0.01** | 0.26–0.39 |
| dominant writers | mlps×mlps 47%, attn×mlps 37% | attn×mlps 43%, mlps×mlps 38% | attn×mlps 47–56% |

**Layer 3 is the flattest layer measured** (PR 13.8 — half its 32 directions genuinely
matter) and its leader is the program's first with *no verifiable token structure at
all* (ρ = −0.005 against a 0.155 null). **Layer 2's leader is the least compressible**:
the rank-1 whitened surrogate repairs 3.5%, rank-2 only 17.6% — against 66–100%
everywhere else. Both layers' outputs are high-rank (effective rank 80–87, needing
~500 dims for 90%), and their leaders read almost entirely from accumulated MLP and
attention writes (embedding ≤0.5%).

**This bounds §26's regularity honestly.** "The distribution-robust core of every
verified leader is its rank-1 whitened surrogate" was measured on layers whose leaders
*have* verified surrogates (0, 1, 16). Layers 2–3 show the construction failing: where
the causal mass of the middle actually lives, no low-rank surrogate exists to be
robust. The regularity is real but conditional on mechanism class.

**And it closes the loop on the repo's boundary result.** §74 called MLP1's tail
"irreducibly distributed"; §13–16 partially rehabilitated MLP1 into five-to-six
directions. Layers 2–3 are what genuinely irreducible distribution looks like on this
model: flat fair-share spectra, high-rank outputs, unnameable leaders, failed
surrogates — measured with instruments that succeeded on four other layers, so the
failure is informative rather than instrumental. The middle's workhorses are
distributed in exactly the way the rest of the model is not.

The pipeline picture assembled across §27–29: layers 2–3 perform a high-rank,
uncompressible transformation; layer 4 reads it (and misfires without it); the tail
5–15 adds small redundant refinements; 16–17 collapse everything back to a few
readable directions for the unembedding. Distribution rises then falls with depth, and
the compressible ends were exactly where every earlier success lived.

---

## 30. The formula's first blind test: 3/4, with the miss and its caveats on record

File: `bilin18_blind_prediction.py` (731 s). Predictions computed and frozen from
weights + input second moment *before* any causal measurement of layers 7, 9, 11, 13
(never previously profiled); success bar registered in advance at leader-energy 0.5 in
the predicted top-8. Empirical leaders from 8-permutation Shapley over each layer's
top-32 output directions.

| layer | span effect | measured leader (share, PR) | energy in predicted top-8 | cos w/ predicted #1 | verdict |
|---|---|---|---|---|---|
| 7 | +0.0202 | dir 4 (51%, PR 2.7) | **0.403** | 0.195 | **MISS** |
| 9 | −0.0068 | dir 0 (−90%, PR 0.4) | 0.892 | 0.781 | HIT |
| 11 | +0.0177 | dir 0 (61%, PR 2.1) | **0.954** | **0.917** | HIT |
| 13 | +0.0060 | dir 0 (113%, PR 0.5) | 0.972 | 0.861 | HIT |

(random-span baseline 0.003–0.010 throughout)

**The formula predicts blind.** Three of four layers clear the registered bar, two of
them at 0.95+, and at layer 11 the *single* predicted #1 eigenvector is the measured
leader at cos 0.92. The miss is real and stays a miss: layer 7's leader sits at 0.403 —
45× random, but below the bar I set, and its predicted-#1 cosine (0.195) is poor.

Caveats that keep this honest rather than triumphant:

- **Layers 9 and 13's attributions are noisy at 8 permutations** (PR 0.4–0.5, shares
  −90%/113% — negative-value-dominated spectra). Their *leader identities* may be
  unstable even though the subspace score is not. The clean hits are 11 and 13's
  energy numbers; the clean identity hit is layer 11.
- **Layer 9's span effect is negative** (−0.0068): deleting its top-32 output span
  *improves* pile CE. That is §25–26's shift-regularisation pattern appearing
  unprompted in a tail layer — the tail's top directions are partly fineweb-specific
  fit, which is consistent with their small fair shares in §27.

Verdict for the protocol: **weights + one data matrix locates the causal subspace of
an unseen layer, usually precisely, and its failures are detectable** (layer 7's low
cos flagged itself). The battery order proposed in §23 — weight-side candidates first,
model evaluations to verify — survives its first contact with layers it had never seen.

---

## 31. The 3→4 edge has no small set of channels

File: `bilin18_chain_bus.py` (4 s after a caught bug: the first version's forward
dropped the residual add, self-flagged by an exactly-zero reference effect).

Weights-first candidates for the 2→3→4 chain's bus variables: top eigenvectors of the
coupling operator `K(3→4) = C3^{1/2} G2(4) C3^{1/2}` — what layer 3 writes weighted by
what layer 4's quadratic reads. Registered predictions: P1, coupling directions beat
layer 3's own output PCA at matched k; P2, eight channels carry half the full-edge
transplant effect.

| k | coupling T(k) | L3-PCA T(k) | random |
|---|---|---|---|
| 1 | 0.021 | 0.020 | 0.000 |
| 4 | 0.128 | 0.121 | 0.002 |
| 8 | **0.211** | 0.195 | 0.004 |
| 16 | 0.297 | 0.281 | 0.008 |

**P2 failed decisively** (0.211 against a 0.5 bar), and P1 held only trivially (an 8%
relative edge over plain PCA — real but useless). The 3→4 edge is **broadband**: no
small set of variables carries it, and the coupling operator adds almost nothing over
"layer 3's biggest outputs" here, in sharp contrast to the leader predictions where the
same algebra was nearly exact.

This is the edge-level twin of §29's verdict and it closes the question the
causal-abstraction phase posed for the chain: **the 2→3→4 pipeline does not abstract
into few variables.** Its existence and orientation are established (§28); its content
is high-bandwidth by every instrument — output rank, surrogate failure, naming failure,
and now channel count. For this model, "understand the middle" cannot mean "name its
variables"; the honest unit is the subspace and its gain, not the coordinate.

---

## 32. The composition gate: surrogates don't compose across the one tested edge

File: `bilin18_composition_gate.py` (10 s). Every compression was verified with the
rest of the model intact; §27's superadditivity warned that joint installation could
fail while parts pass alone. Registered prediction: joint ≤ 1.3× the sum, with any
excess expected on the 16→17 bus edge, separable via a pair arm.

| installed | ΔCE |
|---|---|
| L1 leader surrogate | +0.0040 |
| L16 replacement | +0.0307 |
| L17 replacement | +0.1018 |
| **L16 + L17** | +0.1974 |
| all three | +0.2019 |

**The prediction failed (1.48×) — and the excess is entirely the predicted edge.** The
16–17 pair interaction is +0.0649; the full three-way excess is +0.0654; the residual
beyond the pair term is **+0.0005, zero within noise**. L1's surrogate composes
perfectly with everything. L16's and L17's replacements interfere with each other and
with nothing else — exactly what the verified L16→L17 syntax-bus edge (§24) requires:
L17's replacement was fit on inputs produced by an *intact* layer 16, and replacing 16
shifts the very distribution 17's whitened truncation was optimised for.

Three things this settles:

1. **"Correct compression" needs a composition criterion**, and it is now part of the
   battery: parts verified alone must be re-gated jointly, and the causal graph tells
   you which pairs to test (only linked pairs can interact — the unlinked L1 pairs
   contributed +0.0005).
2. **The causal graph made a successful advance prediction about a failure mode.**
   The 16–17 edge was found by steering (§24); it just showed up, quantitatively, in a
   completely different experiment. That is the strongest validation the graph has.
3. **The fix is prescribed by the mechanism**: refit L17's replacement on the
   L16-replaced model (scheduled/sequential fitting, downstream-after-upstream).
   Registered, not yet run; predicted to recover most of the 0.065.

---

## 33. The prescribed fix mostly fails: refitting closes 21%, not two thirds

File: `bilin18_sequential_refit.py` (13 s). §32's mechanism story — L17's replacement
was fit on intact-L16 inputs — prescribed scheduled fitting: install L16's replacement,
refit L17's on the modified model. Registered prediction: at least two thirds of the
+0.0649 interaction closes.

**It closed 21%** (excess +0.0649 → +0.0516; joint +0.1974 → +0.1841). And the sanity
arm deepened the puzzle: the refit-for-modified-upstream L17 replacement, installed
alone on the *intact* model, does *better* (+0.0916) than the intact-fit one (+0.1018)
— fit-distribution matching is evidently not what governs these replacements' quality.
The registered prediction failed; "wrong fitting distribution" is a minor mechanism.

The surviving candidate, being tested next (rank sweep, in flight): **shared-wire
truncation compounding** — both replacements truncate the same bus signal, and losses
on a wire that L17 reads *quadratically* compound rather than add. If right, the
interaction excess should collapse with form rank faster than the individual damages.

## 34. Second hypothesis down: the interaction is rank-independent

File: `bilin18_interaction_rank.py` (32 s). Registered prediction: shared-wire
truncation compounding — the excess collapses with form rank. **Failed**: the excess is
flat, +0.0649 / +0.0655 / +0.0632 / +0.0616 at k = 2/4/8/16, while both individual
damages barely move. Neither refitting (§33, 21%) nor form truncation explains the
16–17 interaction. By elimination the candidate is the **projection step** — confining
both layers' outputs to their R=4 spans — which is the one component the k-sweep holds
fixed. Registered next: sweep R at fixed k; if the excess persists at R=16, the bus
signal between 16 and 17 lives substantially outside both top-4 output spans, which
would connect this directly to the coverage-gap finding (top spans carry only a
fraction of causal effect).

## 35. The interaction obeys a product law — a composition rule, not a bug

Files: `bilin18_interaction_span.py` (25 s) plus the joint analysis of §33–35's seven
configurations. The span sweep also missed its registered bar (excess 0.0649 → 0.0452
from R=4 to 16; 30% closed, bar was half). Three failed single-mechanism hypotheses in
a row — refitting (21%), form rank (0%), output span (30%) — and then the data
volunteered the actual structure:

| config | d16 | d17 | excess | excess/(d16·d17) |
|---|---|---|---|---|
| k=2 | 0.0307 | 0.1018 | 0.0649 | 20.7 |
| k=4 | 0.0304 | 0.1009 | 0.0655 | 21.4 |
| k=8 | 0.0304 | 0.0968 | 0.0632 | 21.5 |
| k=16 | 0.0298 | 0.0952 | 0.0616 | 21.7 |
| R=8 | 0.0267 | 0.0770 | 0.0526 | 25.6 |
| R=16 | 0.0258 | 0.0664 | 0.0452 | 26.4 |

**excess ≈ c · d16 · d17 with c = 22.9 ± 2.4** — 9% mean prediction error across six
configurations spanning two different fidelity knobs. The three fixes "failed" because
there is nothing to fix: each knob shrinks a *factor*, and the excess follows the
product, exactly as a quadratic reader of a sum of errors predicts (the cross-term of
`(e16 + e17)` through a bilinear form is proportional to `e16·e17`). The architecture's
signature again, this time in the composition algebra.

**The practical composition rule for this model**:
`joint damage ≈ Σᵢ dᵢ + c·Σ_(linked pairs) dᵢdⱼ`, with the graph supplying which pairs
are linked (unlinked pairs contributed +0.0005 in §32) and c ≈ 23 for the 16→17 edge.
Compression budgeting becomes quantitative: keep each replacement's solo damage small
enough that the product terms stay inside the additive budget — e.g. d16·d17 < 0.002
keeps the cross-term under 0.05 nats. Registered follow-ups: measure c on a second
linked pair (needs one more verified edge with two surrogates), and test the law's
out-of-sample prediction at an untried (R, k) corner.

## 36. The product law predicts out of sample

File: `bilin18_product_law_test.py` (17 s). Two corners never measured, both knobs off
the fitted axes, prediction from `excess = 22.9·d16·d17` registered before measuring:

| corner | predicted excess | measured | error |
|---|---|---|---|
| R=8, k=8 | +0.0413 | +0.0475 | 13% |
| R=16, k=8 | +0.0317 | +0.0411 | 23% |

Both inside the registered 25% bar. The composition law is now a validated predictive
tool, not a fit: `joint ≈ Σdᵢ + 23·Σ_linked dᵢdⱼ` for this edge, with a hint of mild
underprediction at high fidelity worth watching if the law gets leaned on hard.

## 37. The tail swept weights-first: prediction works where there is signal, and the tail is shift-fragile

File: `bilin18_tail_sweep.py` (29 s — seven layers, three evaluations each, the adopted
weights-first economy).

| layer | delete pred-2 | delete rand-2 | ratio | delete span-32 |
|---|---|---|---|---|
| 5 | +0.0107 | −0.0001 | **75×** | +0.0412 |
| 6 | +0.0020 | +0.0002 | 11× | −0.0003 |
| 8 | −0.0008 | −0.0001 | — | +0.0084 |
| 10 | −0.0011 | +0.0007 | — | +0.0099 |
| 12 | +0.0100 | −0.0009 | 12× | **−0.0025** |
| 14 | +0.0054 | +0.0002 | 24× | +0.0060 |
| 15 | **−0.0091** | −0.0004 | — | **−0.0065** |

Registered (a) — pred-2 ≥ 5× random — holds at **4/7** (5, 6, 12, 14; layer 5 at 75×).
The three misses are informative rather than random: layers 8 and 10 have span effects
under 0.01 nats — near the evaluation floor, nothing for the formula to find — and
**layer 15's predicted top-2 are actively harmful** (−0.0091): deleting the two
directions the Λ-Gram ranks most important *improves* pile CE.

Registered (b) confirmed and extended: negative span effects at layers 12 and 15, on
top of §30's layer 9. **The shift-fragility pattern is now the tail's norm, not an
anomaly**: in layers 9–15, the highest-Gaussian-variance quadratic directions — exactly
what G_lam finds — are substantially fineweb-specific fit that hurts on shifted text.
The formula is working correctly and measuring something real: in the tail, "most
important by variance" and "most distribution-specific" largely coincide, which is the
§25/§26 robust-core story at layer scale. A fineweb arm for the tail negatives is the
registered confirmation (predicted: positive deletion costs in-distribution).

## 38. The tail's negative effects flip sign in-distribution — prediction held

File: `bilin18_tail_fineweb.py` (11 s). All registered predictions held, cleanly:

| layer | span-32, pile | span-32, fineweb | pred-2, fineweb |
|---|---|---|---|
| 9 | −0.0068 | **+0.0109** | +0.0007 |
| 12 | −0.0025 | **+0.0083** | +0.0036 |
| 15 | −0.0065 | **+0.0124** | +0.0052 |

Every negative deletion effect in the tail is positive on training-like data, including
layer 15's harmful-on-pile predicted top-2 (+0.0052 in-distribution). The tail layers
do real (small) in-distribution work in exactly the directions the Λ-Gram identifies;
that work is fineweb-tuned and inverts to a liability under shift. The shift-fragility
of the tail is now fully confirmed on both sides of the split, and every negative
ablation number in this program's pile evaluations should be read through it.

## 39. The coverage curve: causal mass is spread over the whole output space, and the leader-predictor does not generalise to coverage

File: `bilin18_coverage_curve.py` (22 s). Deletion cost of the top-k output span as k
grows to full rank, layers 1 and 0, two orderings. Gate: both bases agree at k=1152
(4.9037 / 0.6983 — also layer 1's first operator-C full-delete measurement; §9's 5.65
was operator A, consistent with the known operator dependence).

| k | L1, PCA | L1, G_lam | L0, PCA | L0, G_lam |
|---|---|---|---|---|
| 32 | 0.116 | 0.068 | 0.100 | 0.072 |
| 128 | 0.568 | 0.366 | 0.221 | 0.166 |
| 512 | 3.699 | 2.200 | 0.523 | 0.454 |
| 1152 | 4.904 | 4.904 | 0.698 | 0.698 |

**(a) held, strongly: the causal mass is nearly space-filling.** Layer 1's 128-dim span
carries 12% of the full cost; 512 dims carry 75%. Per-direction cost actually *peaks in
the mid-spectrum* (dims 256–512 carry ~0.0095 nats each against 0.0055 for 128–256 and
0.0019 for the final 640) — the §6 blind-direction theme at its sharpest: this layer
puts its heaviest causal weight per direction in the *middle* of the variance spectrum,
not the top.

> **Corrected by §44.2:** the mid-spectrum-peak reading was an artifact. Cumulative
> differences do not measure per-band mass under superadditivity; disjoint band
> deletions show per-dim cost decreasing monotonically with rank, and the causal mass
> living in cross-band *interactions* (bands sum to 0.24 vs 4.90 jointly, 20×). The
> space-filling conclusion stands; the mechanism stated here does not. There is no low-dimensional causal core to find at layer 1; the
direction-level program's <10% coverage was not a sampling failure but the geometry.

**(b) failed, inverted: PCA dominates G_lam for cumulative coverage at every k on both
layers** — the exact opposite of the leader-identity results (§23/§30). The Λ-Gram is
the better *leader* predictor and the worse *coverage* ordering; variance ordering wins
for bulk. Two instruments, two jobs, and my assumption that the first generalised to
the second is now a recorded failed prediction. Whatever ordering is causally optimal
is neither — the mid-spectrum peak says both orderings misplace the heavy directions.

## 40. The interchange leak resists two hypotheses — and localises to error amplification

Files: `bilin18_leak_origin.py`, `bilin18_leak_rank2.py` (6 s each). §21 left a 32% gap
between the z→c₀ abstraction's on-distribution repair (92%) and its interchange
faithfulness (68%). Two registered hypotheses, both failed:

- **Document mixture (predicted ≥85% same-document): FAILED at 60.8%.** Transplanting z
  within the same document is no more faithful than across documents. The leak is not
  the ICC-0.56 document component.
- **The form's second direction (predicted ≥85% with a rank-2 transplant): FAILED at
  72.5%** (cross-doc; 69.3% same-doc) — despite the rank-2 coefficient fit reaching
  R² = 0.987 on natural data.

What survives elimination is sharp: a transplant whose coefficient errors are 1.3% in
variance terms still loses ~27% of the downstream effect. The remaining candidate is
**position-heavy error amplification** — the mismatch between ĉ₀(source) and c₀(source)
is small on average but the downstream stack (quadratic layers) amplifies the few
positions where it is large, so KL-faithfulness saturates well below the R² story.
Consistent with everything else measured: the composition product law (§35) is exactly
this amplification mechanism at layer scale, and the coverage curve (§39) says function
concentrates where variance statistics do not look. Registered next test: per-position
mismatch distribution (prediction: the top 5% of positions carry >60% of the mismatch
KL), which would close the leak's accounting without closing the leak.

## 41. The leak's accounting closed: local coefficient error, diluted by propagation

Files: `bilin18_leak_accounting.py`, `bilin18_leak_singlepos.py` (6 + 10 s).

The aggregate accounting test failed both its bars (top-5% share 48% vs a 52% generic
control; ρ(mismatch, coeff error) = 0.18) — but with a design confound recorded before
interpreting: a patch at position q affects every later position through causal
attention, so per-position mismatch KL mixes all upstream errors and dilutes any local
correlation. The clean design patches **one position at a time** (the KL downstream of
q is then attributable to q alone; 24 forwards):

- **(a) held: ρ(single-position mismatch, that position's coefficient error) = +0.52.**
  The leak is locally coefficient-error-driven after all; the aggregate test was
  measuring its own confound.
- (b) failed narrowly: the quadratic operating-point scaling (mismatch per unit error
  growing with base coefficient magnitude) lands at ρ = +0.29 against a 0.3 bar, n=12
  positions. Suggestive, unproven; kept as failed.

**The leak thread's verdict across §40–41**: the 32% z→c₀ interchange gap is the
accumulation of small per-position coefficient errors (fit residual 1.3% of variance),
each locally amplified (ρ 0.52) and mixed forward by propagation — not document
identity, not a missing second direction. It is irreducible for any fixed-rank
surrogate and calibrated by the same product-coupling that governs composition (§35).
Practical consequence: interchange faithfulness for this architecture has a ceiling set
by fit residual × propagation, so "the abstraction is 68% faithful" should be read as
"the abstraction has 1.3% coefficient error in a model that amplifies error ~25×" —
two descriptions of one number, the second one predictive.

## 42. Second linked pair: the product coupling is real, the scalar constant is not

File: `bilin18_product_law_pair2.py` (26 s). A 3×3 damage grid on the 3→4 edge (partial
span deletions as the lossy replacements) plus the unlinked pair (3, 14) as control.

**The control is the cleanest graph validation yet**: the unlinked pair's excess is
≤ 0.0013 at every cell — c ≈ 0–1 — while the linked pair at comparable damages shows
excess up to 0.19. No edge, no interaction, across eighteen cells.

**Both registered predictions about the constant failed, in an instructive direction.**
On the 3→4 edge, c ranges **6.1–30.2** (rel sd 59%; (a) failed), with clear structure:
it depends on *which part* of layer 3's span is damaged — the k∈(8,32] band of L3's
output couples ~4× more strongly per unit damage (c ≈ 17–30) than the (32,128] band
(c ≈ 6–7). And (b) failed trivially — the unstable mean (14.3) sits within 2× of 22.9.

**The refined law**: the composition excess is a *bilinear form in the two damage
profiles*, `excess ≈ e_aᵀ C_edge e_b`, and the scalar law of §35–36 is its rank-1
approximation — valid exactly when the damage profile's *shape* is held fixed, which
the 16→17 sweeps did (varying R and k changes the amount, barely the shape) and the
3×3 grid deliberately did not. This is the bilinear architecture's signature stated
completely: composition coupling is itself a quadratic object, edge-local and
direction-resolved. The scalar c is a useful engineering number per (edge,
damage-family); it is not a physical constant of the edge.

## 43. Two queue-runner results: the anisotropy is not in K, and operating-point scaling stays unproven

Files: `bilin18_cedge_direction.py`, `bilin18_opscaling.py` (first runs under the new
supervisor-managed queue runner; 9 + 21 s).

**The coupling operator does not predict the coupling anisotropy.** Registered: damage
along K(3→4)'s top-8 eigendirections yields c ≥ 1.5× the PCA-8 value. **Failed**:
c_K = 10.7 vs c_PCA = 12.8 — K is no better than variance ordering here, consistent
with §31's transfer result. (The K-bottom control "held" trivially: its d3 is ≈ 0, so
its c is a divide-by-tiny artifact, recorded as uninformative rather than a win.) The
§42 anisotropy — the 8–32 band coupling 4× stronger — has no weight-side predictor yet;
`C_edge`'s structure remains empirical.

**Operating-point scaling failed again at n=48** (§41(b) rerun with 4× the positions).
The quadratic-sensitivity story for *which positions* amplify most is not supported;
what stands is only the local error→mismatch link (ρ 0.52, §41(a)).

## 44. Queue batch: signed coupling channels, a §39 correction, and a hidden front edge

Files: `bilin18_coupling_bands.py`, `bilin18_coverage_bands.py`, `bilin18_l0_l1_edge.py`
(12 + 6 + 10 s under the runner).

### 44.1 C_edge(3→4) has signed structure — a coupling channel and an anti-coupling bulk

Both registered predictions held, and the measured structure exceeds them:

| L3 band | d3 | excess with L4 | c |
|---|---|---|---|
| [0,8) | +0.035 | +0.042 | +12.8 |
| **[8,32)** | +0.036 | **+0.102** | **+30.8** |
| [32,64) | +0.011 | **−0.006** | −5.8 |
| [64,128) | +0.016 | **−0.010** | −6.9 |
| [128,256) | +0.015 | −0.009 | −6.1 |

The 3→4 coupling is a **narrow positive channel at ranks 8–32** (c = +31) sitting on a
broad **anti-coupled** region: damaging L3's ranks 32–256 *reduces* the joint damage
with L4 — layer 4 partially compensates mid-band damage while amplifying channel
damage. C_edge is signed, not just anisotropic. (The random-span control is again a
divide-by-near-zero artifact and is recorded as uninformative.)

### 44.2 CORRECTION to §39: there is no mid-spectrum peak — cumulative differences measured interference, not band mass

Both §44 predictions derived from §39's reading **failed**, and rightly: disjoint band
deletions at layer 1 give per-dim costs of 0.0036 / 0.0006 / 0.0002 / 0.0001 / 0.00002
for bands [0,32) → [512,1152) — **monotonically decreasing**. §39's "per-direction cost
peaks mid-spectrum" came from differencing a *cumulative* curve, and under
superadditivity cumulative differences do not measure band mass. I should have caught
this — it is the §12 budget-dependence lesson recurring inside a single layer's
spectrum, and §39 carries the correction notice.

What replaces it is stronger: the disjoint bands sum to **0.24** nats against the
full-span deletion's **4.90** — the within-layer cross-band superadditivity is **20×**.
Layer 1's causal mass is not *in* any band of directions; it is overwhelmingly in the
**interactions between bands**. The space-filling conclusion of §39 stands, with the
mechanism corrected: not mid-band mass, but interaction dominance — the product-coupling
theme at its most extreme, inside one layer.

### 44.3 The front of the graph is chained after all: a hidden L0→L1 edge

The registered *absence* claim failed spectacularly: steering L0's punctuation leader by
+2σ moves L1's leader coefficient by **+1.04σ** (potency control passed), while −2σ
moves it +0.07σ — **unit gain, rectified**, the same quadratic-readout signature as the
16→17 bus. §17's writer decomposition had assigned emb×mlp0 only 0.5% of the L1
leader's variance; the resolution is not a contradiction but a blind spot of
proximate-writer accounting: **L0's write feeds attn1's input, and attn1 is the L1
leader's dominant writer** — the influence routes through the attention block, which
the folding attributes to attn1 rather than to what attn1 read. Proximate-writer
variance shares do not bound upstream causal influence. The graph gains
`L0-leader → (attn1) → z/c₀`, and the front of the model is a chain, not parallel
tracks; mediation-through-attn1 is the registered next test, now queued.

## 45. The hidden edge is attention-mediated (with overshoot), and layer 1's interactions are 80% higher-order

Files: `bilin18_edge_mediation.py`, `bilin18_band_interactions.py` (10 + 14 s).

**Mediation of the L0→L1 edge: held, with structure.** Freezing attn1's context while
steering L0's leader kills the entire +1.04σ effect *and overshoots* (−0.35σ): the
residual direct path (L0's write surviving in the stream L1's MLP reads directly) is
slightly *negative*, and attention carries more than the whole net effect. Freezing
head 4 alone kills 51% — **below the registered 60% bar**: unlike z's own context
dependence (79% head 4), the L0-leader signal reaches c₀ through *several* attn1 heads.
The attn0-freeze control is perfectly clean (0% killed). Confirmed graph edge:
`L0-leader → attn1 (distributed across heads, 4 largest) → c₀`, minus a small negative
direct path. The front chain is `token → L0 → attention → L1`, and the §17 folding's
0.5% proximate share was measuring only the direct path.

**Layer 1's within-layer interaction is dominantly higher-order.** All ten band pairs
measured: pairwise excesses sum to 0.94 nats of the 4.66-nat interaction — **80% of the
interaction is order-3 and above** (registered prediction held). The pair-level product
law, which prices *between-layer* composition well, captures only a fifth of
*within-layer* structure. Layer 1 is not a sum of parts plus pairwise couplings; it is
holistic in the concrete, measured sense that most of its causal effect exists only in
combinations of three or more direction-bands. That is the final and strongest form of
"no small core": not only is the mass not in any band, it is not even in any *pair* of
bands.

## 46. The edge rides head 1, and the interaction hierarchy is graded to all orders

Files: `bilin18_mediation_heads.py`, `bilin18_band_triples.py` (10 + 20 s).

**The L0→L1 edge is carried by head 1, not head 4.** Registered prediction (a) — no
non-4 head kills more than 30% — **failed in the best way**: freezing **head 1 kills
96%** of the steered effect, nearly double head 4's 51%, with every other head at ≤3%
(concentration prediction (b) held). Combined with §21 (head 4 produces 79% of *z*'s
natural movement, head 1 second at 14%), the front of the graph resolves into two
distinct channels through the same attention block: **head 4 computes z's ordinary
context dependence; head 1 transports the L0-leader signal** into the coefficient. The
single-head kills sum well past 100% — mediation itself is non-additive, consistent
with everything else in this model.

**The within-layer interaction hierarchy is graded, not truncated.** Möbius
decomposition through order 3 at layer 1: solo bands 0.24 (5%), pairwise 0.94 (19%),
pure order-3 1.70 (35%), leaving **order-4-and-above at 2.02 (41%)**. The registered
bar (order-3 sum < 1.86, i.e. order-4+ still dominates) held, narrowly and honestly.
The full picture: each interaction order contributes *more* than the last through at
least order 4 — layer 1 is holistic all the way up, with no order at which a truncated
interaction model captures it.

## 47. Head 1 works through its pattern, and interaction depth is compressibility's twin

Files: `bilin18_head1_route.py`, `bilin18_band_orders_l16.py` (10 + 14 s).

**The L0 signal moves head 1's attention, not its cargo.** Registered value-mediation
**failed**, inverted: freezing head 1's *pattern* kills 54% of the steered effect,
freezing its *values* only 30% (inert-head control clean at ±5%). The injected
L0-leader offset changes *where head 1 looks* — its q/k geometry — more than *what it
copies*. The two routes sum to 84% against the 96% joint kill, so they overlap
non-additively, as everything in this model does. Graph annotation updated: the
L0→L1 edge is pattern-dominant through head 1.

**Layer 16's interaction hierarchy is shallow — the contrast case held at 99%.**
Solo bands + pairwise excesses capture **99%** of layer 16's full-span deletion cost,
against layer 1's **24%**. The registered unification survives its first test:
**compressibility and interaction shallowness are the same property seen from two
sides** — a layer whose causal effect decomposes into few directions is also a layer
whose effect decomposes into low-order interactions, and the uncompressible middle is
uncompressible precisely because its effect exists only in high-order combinations.
Generalisation across layers 0/2/3/17 is queued; if the monotone relation holds, this
becomes the program's unifying statement about what "understandable" means for this
architecture.

## 48. The unification holds: interaction shallowness IS compressibility, across the model

Files: `bilin18_depth_shallowness.py`, `bilin18_head1_aim.py` (51 + 5 s).

**The ordinal prediction held at all six layers.** Share of each layer's full-span
deletion cost captured by solo bands + pairwise interactions:

| layer | 1 | 2 | 3 | 0 | 16 | 17 |
|---|---|---|---|---|---|---|
| solo+pair share | **24%** | **39%** | **57%** | 67% | **99%** | **100%** |

Every layer known to compress (16, 17, and partially 0) sits above every layer known
not to (1, 2, 3), exactly as registered. **Interaction shallowness and compressibility
are one property seen from two sides**: a layer is replaceable by a few directions
precisely to the extent that its causal effect decomposes into low-order combinations
of directions — and the uncompressible early layers are uncompressible because 43–76%
of their effect exists only in third-order-and-higher combinations. This is the
program's unifying statement about what "understandable" means for this architecture,
and it is now measured on all of layers 0–3, 16, 17 with the ordinal prediction
registered in advance.

**The head-1 re-aiming test invalidated itself and is recorded as such.** The
punctuation-vs-content class comparison had 980 punctuation keys against **28** content
keys (underpowered), the inert-head control violated its own bound, and steering
reduced absolute pattern mass *globally* — a scale confound (the injected offset
perturbs the rms normalisations, shrinking q/k products everywhere). No conclusion
about where head 1 re-aims is licensed from this run; a scale-controlled redesign
(per-row normalised patterns, corpus-frequency-matched classes) is queued.

## 49. Layer 1 factorizes after all — one entangled core, one inert complement — plus the completed shallowness map

Files: `bilin18_factorization.py` (15 s, run for the user's independence question),
`bilin18_tail_shallowness.py` (140 s), `bilin18_head1_aim2.py` (5 s).

### 49.1 The factorization test: my "no factorization" prediction failed

Two-way 576+576 splits of layer 1's output space, synergy share = (full − dA − dB)/full:

| split | d(A) | d(B) | synergy share |
|---|---|---|---|
| **PCA low/high** | **+4.169** | **+0.009** | **15%** |
| G_lam top/bottom | +2.759 | +0.011 | 44% |
| PCA interleaved | +0.224 | +0.209 | 91% |
| random ×5 | ~0.12 | ~0.12 | **95–96%** |

Registered prediction (a) — no split below 50% — **failed**, and the failure sharpens
§45: layer 1 is *not* unstructured holism. The PCA-aligned top-576 subspace carries 85%
of the layer *by itself*, the complement is nearly inert (+0.009), and cross-synergy is
only 15%. **Layer 1 ≈ (deeply entangled 576-dim core) ⊕ (inert complement)** — a
factorization exists; it is just that the working factor is half the space and
internally graded to all orders (§46). Meanwhile any *misaligned* cut sees 95%
synergy (random splits, isotropy prediction (b) held at 1pp spread), which is why the
band analysis read as total holism. All synergies are positive everywhere: the
structure is need-both synergy, never mutual exclusivity. §45's "hard boundary on any
pairwise calculus" stands *within the core*; "no small core" stands (the core is big,
not absent); "no factorization" is corrected here.

### 49.2 The 18-layer shallowness map is complete

All eleven tail layers (5–15) land **above** layer 3's 57% solo+pair share — most at
85–95% — with the registered prediction held and no exclusions needed (every tail
layer cleared the 0.02-nat power floor). Final map: **deep interaction is exclusive to
layers 1–3.** One narrow early region of genuinely high-order computation feeds an
otherwise shallow, compressible, weight-predictable machine.

### 49.3 The re-aiming mechanism is real; head-specificity is not licensed

The scale-controlled redesign: under L0-leader steering, head 1's *relative* pattern
mass on punctuation keys shifts **4.1×** more than under a matched random steer
(registered bar 3×, held). But the inert-head control was violated again — head 6's
punctuation mass moves 57× its random-steer baseline — so the class re-aiming is
attention-wide, not head-1-specific. Combined with §46 (only head 1's freeze affects
c₀): **the steer re-aims many heads toward punctuation; only head 1's re-aiming is
read by the leader coefficient.** The mechanism claim is supported; the selectivity
lives in what c₀ reads, not in which heads move.

## 50. Two user questions, answered with runs

Files: `bilin18_operator_composition.py`, `bilin18_input_orders.py` (5 + 26 s).

**Was the routing derivable from weights (operator-level composition)?** Partially —
the important half, yes. The steered direction's qk-circuit enrichment per head ranks
**head 1 first** (prediction (a) held): pure weight algebra identifies which head
carries an injected signal, before any intervention. But (b) failed — head 1's raw
v-enrichment slightly exceeds its qk-enrichment, so the *pattern-dominance* of the
route is not readable from first-order enrichments (patterns respond nonlinearly to
q/k perturbations; the operator calculus needs the quadratic response, not norms) —
and (c) failed: head 4's qk rank is 8th, consistent with §46 (head 4 carries z's
*natural* context, not the injected signal; the two channels really are different
operators). Verdict: scalar edges don't compose; first-order operator signatures
compose enough to find the carrier head; the route's internal character needs
second-order signatures.

**Does shallow=compressible go the other way, on the input side?** Yes, with exactly
the architectural signature predicted. Input-side band Möbius (patching only the xhat
each MLP reads):

| | layer 1 | layer 16 |
|---|---|---|
| input-side solo+pair share | **89%** | 90% |
| output-side (for contrast) | 24% | 99% |

All three registered predictions held: layer 1's *input* side is dramatically shallower
than its output side (89% vs 24%) because **a bilinear layer's own input interaction is
architecturally bounded at order 2** — the depth we measure on the output side is
manufactured *downstream* of the layer, by the stack that consumes it. The L1–L16 gap
nearly vanishes on the input side (1pp vs 75pp). So the asymmetry of the unification is
itself informative: compressibility is a property of *how a layer's output is consumed*,
not of how the layer consumes its input — every layer reads shallowly; only the middle
is read deeply.

## 51. The operator calculus at second order: route character recovered; edges are signal-specific

Files: `bilin18_blind_routing.py`, `bilin18_secondorder_route.py` (10 + 5 s; the
latter's first run crashed on dead code, fixed and rerun — and the crash exposed a
runner flaw, `_completed.txt` logging `exit_ok` unconditionally, now fixed to record
real exit codes).

**The blind routing test failed its own gate, informatively.** Weights-only
qk-enrichment named head 7 as the predicted carrier for L0's #2 causal direction (the
number/quantifier axis) — but the registered edge-existence prediction failed: steering
that axis moves the L1 leader by only **+0.23σ** (bar 0.3), against the punctuation
axis's +1.04σ. The carrier prediction was therefore untestable, and the finding stands
on its own: **the L0→L1 edge is signal-specific** — the front chain routes the
punctuation axis at unit gain and largely ignores the number axis. Edges in this model
are not pipes; they are filters.

**The second-order signature recovers everything first-order norms missed.** Computing
response *energies* from weights + cached activations (pattern response = score
perturbations through the standing cross-scores, times standing value energy; value
response = injected value content through the standing pattern):

- head 1's pattern-response exceeds its value-response **69×** (registered (a) held) —
  matching the measured 54/30 pattern-dominant route where first-order norms had
  predicted the reverse;
- head 1's *absolute* pattern-response (4.1e11) exceeds every other head's by 30–300×
  — the carrier identity falls out too, more sharply than from first-order enrichment.

The operator-composition picture is now: **first-order signatures find carriers
sometimes; second-order response energies find carriers, route character, and — next
registered test — edge strength.** If the number axis's weak edge is also predicted by
its small maximal response energy, the calculus predicts which signals route at all,
from weights.

## 52. Response energy detects routing at 6,000× — and found a stronger edge than the one we knew

File: `bilin18_edge_strength.py` (10 s).

| signal | response energy (weights+cache) | measured \|Δc₁\| |
|---|---|---|
| punct (#1) | 4.20e11 | 1.04σ |
| **#3** | **3.15e11** | **1.60σ** |
| number (#2) | 5.47e10 | 0.23σ |
| random-a | 6.77e7 | 0.015σ |
| random-b | 1.36e8 | 0.002σ |

Spearman(E, |Δc₁|) = **0.80 exactly** — the registered ≥0.8 bar formally failed on the
strict float comparison and is reported as at-bar: the ordering has two adjacent swaps
among five. Prediction (b) failed genuinely: the punctuation axis tops the energy list
but **not** the measured list, because **L0's #3 causal direction routes into the L1
leader at 1.60σ — the strongest front edge yet observed, previously unknown**, and the
calculus had flagged it (second-highest energy) before the measurement.

The substantive scorecard for the weights-side calculus: it separates routed from
unrouted signals by **3–4 orders of magnitude** (3e11 vs 7e7), ranks the routed ones
roughly (two swaps), and its false ordering at the top is between two signals it
correctly identifies as both strongly routed. As a *detector* of which signals travel
an edge, it is essentially solved; as a fine ranker, approximate. The newly found #3
edge inherits the full protocol next: registered carrier prediction from enrichment,
then the freeze sweep — the blind test §51 could not run now has an edge that exists.

## 53. The blind carrier test fails: the #3 edge has no carrier

File: `bilin18_blind_routing2.py` (10 s). Both registered predictions **failed**, and
the failure characterises the edge: freezing any single head kills at most **32%**
(head 8), with head 1 second at 28%, head 4 at 16%, and the rest ≤7% — the #3 edge is
**distributed across heads**, unlike the punctuation edge's 96% single-head carrier.
First-order qk-enrichment predicted head 1 (4.03) and is wrong twice over: wrong head,
and wrong presumption that a dominant carrier exists.

Scorecard for the blind carrier protocol so far: 0-for-1 on testable attempts (the
punctuation case was retrodiction; the number axis had no edge; the #3 edge has no
carrier). The honest state of the operator calculus: **routing detection from weights
is solved** (6,000× separation, §52); **carrier prediction is not** — first-order
enrichment neither ranks heads correctly when mediation is distributed nor predicts
whether a dominant carrier exists. The registered next step is the per-head
second-order response energy against the measured kill profile (the §51 signature that
recovered pattern-dominance may also recover the kill distribution), queued.

## 54. Three runner results: a one-sided routing guarantee, a metric lesson, and the reader-coupling test confounded by its own aggregation

Files: `bilin18_mediation_profile.py`, `bilin18_edge_census.py`,
`bilin18_reader_coupling.py` (5 + 10 + 5 s).

**The edge census (8 signals): response energy is necessary, not sufficient.**
Spearman 0.64 (bar 0.7, failed — close), but the asymmetry is the finding: **zero
low-energy signals route** (E < 1e10 never reaches 0.5σ; registered (b) held), while
one high-energy signal barely routes (dir 4: the *highest* E at 5.8e11, only 0.33σ).
The detector gives a one-sided guarantee — low energy proves no edge; high energy only
licenses testing. Two new edges surfaced: rank-5 (0.36σ) and rank-6 (0.33σ), both
modest.

**The mediation-profile test: right on the spread edge, wrong metric on the
concentrated one.** For the #3 edge, per-head response energies track the kill profile
at ρ = 0.77 (held). For the punctuation edge, ρ = 0.05 — but both the energy top-1 and
the kill top-1 are head 1; Spearman over nine heads is dominated by the seven
noise-level entries. The registered metric was ill-chosen for concentrated profiles;
top-1 agreement (2/2 edges) and concentration transfer (held) are the meaningful
reads. Recorded as a metric lesson, not silently rescored.

**The reader-coupling test failed both disjointness predictions — and the instrument
is confounded, stated plainly.** Median per-reader top-5% mass 0.08 (near-uniform) and
mean cross-reader cosine **0.99**: at whole-MLP resolution every downstream reader's
coupling matrix over L1's output directions is essentially the same dense matrix. But
the aggregation (energy over each reader's 1152 output directions) is central-limit
flattening: per-output-direction coupling could be arbitrarily sparse and disjoint and
this statistic would still come out uniform-and-identical. The user's path-separation
hypothesis is NOT refuted at the resolution it was posed — individual forms are
rank-bounded, hence necessarily structured — only the whole-reader version is dead.
The per-form version is queued with the aggregation removed.

## 55. Path-separation tested at form resolution: the coupling is genuinely shared

File: `bilin18_perform_coupling.py` (11 s; instrument clean of §54's aggregation
confound — each form's coupling matrix is rank-bounded and examined individually).

Per-form coupling of L1's top-48 output directions, 32 forms × 6 readers (L2–L17):

- **(a) failed**: per-form top-5% entry mass is 0.15–0.18 — about 3× the uniform
  baseline (0.05) but far from the registered 0.35. Individual forms couple L1's pairs
  *diffusely*, not sparsely.
- **(b) failed**: within-reader cosine between different forms' coupling matrices is
  **0.63** — different outputs of the same reader largely agree on which L1-pairs
  matter.
- **(c) held, weakly informative**: cross-reader cosine 0.60 ≈ within-reader 0.63 —
  readers agree with each other almost as much as with themselves.

Verdict on the structural-exclusivity hypothesis for MLP readers, now tested at the
resolution where the architecture reads: **the density is real, not an aggregation
artifact**. There is a shared coupling template — one dominant pattern of which
L1-direction pairs interact, common across forms and across all six readers sampled —
plus modest per-form variation. Layer 1's interaction is dense at every resolution
tried: whole-model CE (§45), whole-reader (§54, confounded but consistent), and now
per-form (clean).

Two caveats recorded: the cosines are between **absolute-value** matrices, which
inflates overlap (all-positive vectors correlate; a signed version is queued and
predicted lower), and **attention QK readers are untested** — the user's original
suggestion included them, and heads are the component most likely to specialise.

## 56. Signed coupling reverses §55's reading: shared envelope, near-orthogonal functionals

File: `bilin18_signed_qk_coupling.py` (6 s).

**(a) held decisively: the signed within-reader cosine is 0.11** against the unsigned
0.64. §55's "shared coupling template" is a *magnitude envelope* only — which pairs
have large |coupling| is common (plausibly inherited from the L/R singular structure) —
but the **signed** couplings of different forms are nearly orthogonal. Each form reads
an almost-independent quadratic functional of L1's output, over a shared dense support.

**(b) failed: QK heads are no more specialised** than MLP forms at magnitude level
(top-5% mass 0.15, cross-head |cos| 0.79) — the envelope is universal across reader
types. (Limit: the signed version was computed for MLP forms only; the QK signed
cosine is queued.)

**Corrected synthesis of §§54–56, and the honest resolution of the user's question.**
Layer 1's interaction is *dense in support* — no reader, form, or head couples a sparse
or disjoint set of direction-pairs; the magnitude envelope is shared by everyone — but
it is *diverse in functionals*: the signed quadratic forms the readers actually compute
are close to mutually orthogonal (cos ≈ 0.1). So the structural-exclusivity picture
fails in the support sense and substantially succeeds in the functional sense: the
downstream stack reads layer 1 through an overcomplete family of nearly-independent
quadratic measurements on a common dense substrate. Path-separation of *supports* is
impossible; path-separation of *functionals* is nearly free by construction — which is
also, in hindsight, why whole-model band deletions (which destroy all functionals over
a support region at once) read as inseparable holism.

## 57. The orthogonal-functionals picture is universal

File: `bilin18_signed_completion.py` (9 s). Both registered predictions held:

- signed cross-reader form cosine **0.089** (bar ≤0.15) — functional orthogonality
  extends across readers, not just within one;
- signed cross-head QK cosine **0.156** (bar ≤0.25) — attention heads' signed score
  functionals are near-orthogonal too, despite their 0.79 shared magnitude envelope.

The §56 synthesis is now measured everywhere it was posed: **the front of the model
reads layer 1 through an overcomplete family of nearly-orthogonal quadratic
functionals on one shared dense support** — MLP forms within readers (0.11), across
readers (0.089), and attention heads (0.156). This closes the interaction-structure
arc that began with the user's independence question, and it re-aims the compression
program: direction coordinates failed on the middle (space-filling coverage, holistic
bands) because they cut across every functional at once; functional coordinates are
the natural frame, and the two registered next steps are (i) the spectrum of the
functional family itself — quantifying "overcomplete and diverse" against the rank-≲5
envelope — and (ii) the causal payoff: single-functional steering with cross-talk
measured, i.e. path-selective intervention despite dense support.

## 58. Functional coordinates: an 80-dim basis, and a finite steering range

Files: `bilin18_functional_spectrum.py`, `bilin18_functional_steering.py` (11 + 12 s).

**The spectrum quantifies the §57 picture exactly.** Envelope family: effective rank
**2.6** (one shared template, as predicted). Signed family: effective rank **80** of
240 sampled functionals — my "≥100, no small basis" prediction *failed in the
compressive direction*: the readers' functional diversity, though real (top-1 component
only 7%), is captured by **~80 principal functionals**. That is the functional-
coordinates compression number for the front of the model: not 240 independent
measurements, not a handful — eighty.

**Single-functional steering works, with a range.** Perturbing L1's output along the
top eigenvector of one reader-form's coupling matrix:

| target | own movement | median cross-talk | selectivity |
|---|---|---|---|
| L2, form 0 | 1.38σ | 0.07σ | **20.3×** |
| L2, form 3 | 0.12σ | 0.08σ | 1.5× |
| L5 (×2) | 0.38 / 0.26σ | ~0.1σ | 3.0 / 2.5× |
| L13 (×2) | 0.05–0.09σ | ~0.24σ | **0.2–0.4×** |

Registered bars failed (mean 4.6× vs 5; not every case ≥3×) — and the failure has
clean structure: **selectivity decays with reader depth**. Adjacent, targeted
intervention despite dense support is real (20× with the random-direction control at
noise level); by eleven layers of separation the perturbation no longer arrives as the
functional it left as. Functional coordinates are *local* coordinates — near-
orthogonality enables surgical intervention on nearby readers, and propagation through
the quadratic stack (the same product-amplification that governs composition and the
interchange leak) scrambles the targeting over depth. Secondary observation: L2-form-3
barely moves even adjacently — steerability also needs appreciable absolute coupling,
not just an eigvector.

Registered next: the **coherence length** — same-protocol targets at every depth
L2–L9, decay curve of selectivity, half-range estimate, and the B-norm gate for
steerability.

## 59. The coherence length is one layer

File: `bilin18_coherence_length.py` (14 s; B-norm confound fixed by taking each
reader's largest-coupling form).

| reader | L2 | L3 | L4 | L5 | L7 | L9 | L11 | L13 |
|---|---|---|---|---|---|---|---|---|
| own movement (σ) | **1.46** | 0.28 | 0.49 | 0.38 | 0.10 | 0.29 | 0.03 | 0.05 |
| selectivity | **13.0×** | 2.1× | 0.8× | 1.5× | 0.3× | 2.1× | 0.6× | 0.4× |

- (a) at-bar: Spearman(depth, own) = −0.79 against a −0.8 bar — strong decay with
  wiggles (L9 bounces), reported as at-bar rather than held.
- **(b) failed informatively: the half-range is one layer.** Own-movement collapses
  from 1.46σ at L2 to 0.28σ at L3 and never recovers past 0.5σ. My registered 2–6
  layer half-range was optimistic by half an order of magnitude.
- (c) held: absolute coupling gates steerability (ρ = 0.62).

**Functional identity survives approximately one layer of the quadratic stack.** Note
what this does *not* yet distinguish: the steering vector for a deep reader is that
reader's weight-side coupling eigvector in L1's output basis — the *direct-path*
functional — and its failure at depth could mean either (i) the range limit is
intrinsic (propagation noise/attenuation swamps any targeting) or (ii) the direct-path
vector is simply the wrong direction once layers 2–12 have transformed the
perturbation. The discriminating experiment is exact and cheap: the **gradient** of the
deep coefficient with respect to an additive perturbation at L1's output is the true
end-to-end sensitivity direction, computable by one backward pass. Registered: gradient
steering restores own-movement ≥ 0.5σ at L13 (targeting artifact) — or fails to
(intrinsic limit), and either answer settles the arc. Queued.

## 60. The range limit is intrinsic: deep coefficients are not individually addressable from L1

File: `bilin18_gradient_steering.py` (9 s). Both registered predictions failed, and the
failures constitute the answer:

| target | gradient own | direct own | gradient cross-talk | cos(grad, direct) |
|---|---|---|---|---|
| L5 | 1.02σ | 0.43σ | 0.60σ | +0.14 |
| L9 | 0.37σ | 0.30σ | 0.09σ | −0.08 |
| L13 | **0.06σ** | 0.05σ | **3.70σ** | −0.09 |

The gradient — the *exact* end-to-end sensitivity direction, accounting for everything
layers 2–12 do to the perturbation — restores nothing at L13 (0.06σ, bar 0.5) while its
cross-talk explodes to 3.70σ: the optimal direction for moving one deep coefficient
moves *other* coefficients sixty times more. And cos(gradient, direct-path) ≈ ±0.1 at
depth: the two candidate targeting directions are nearly orthogonal *and both fail*,
which rules out every static-direction strategy at once.

**Verdict: the one-layer coherence length is an intrinsic property of the quadratic
stack.** A static perturbation at L1, in any direction, diffuses into broadband
collective motion within a few layers; deep coefficients respond to the *collective*
state, not to any injectable direction. Individual addressability of deep functionals
from shallow layers does not exist in this model at static-steering magnitudes —
long-range control would require input-dependent (per-token, per-context) injection,
which is a different class of intervention. At L5 the gradient does double the direct
effect (1.02σ vs 0.43σ) at the cost of selectivity (0.60σ cross-talk): even at short
range, power and precision trade off.

This closes the functional-coordinates arc with a complete characterisation: **~80
principal functionals, surgically steerable at range one, collectively but not
individually influential at depth** — the quadratic stack is locally transparent and
globally opaque to static intervention, with the transition happening in a single
layer.

## 61. The 80-functional basis is shared structure: leave-one-reader-out R² = 0.71

File: `bilin18_functional_basis_fidelity.py` (11 s). All three bars held:

| basis rank | leave-one-reader-out median R² | random-basis control |
|---|---|---|
| 8 | 0.152 | −0.08 |
| **80** | **0.711** | −0.05 |

A reader entirely excluded from fitting has its L1-coupling matrices reconstructed at
R² 0.71 from the other readers' principal functionals. The basis is **shared
vocabulary**, not per-reader bookkeeping; r=8 captures under a quarter of it
(diversity is real); a random basis captures nothing.

**The functional-coordinates arc, complete.** The middle's reading of layer 1 is: one
dense support envelope (rank ~2.6) carrying **~80 shared principal quadratic
functionals** (verified out-of-reader at 0.71) that are nearly orthogonal pairwise
(0.09–0.16), surgically steerable at range one (20×), and individually unaddressable
beyond that by any static direction (§60, intrinsic). This is the compressed
description of the "incompressible" middle — not in direction coordinates, where every
instrument failed, but in the coordinates the architecture reads with. Registered open
items, honestly deferred: a CE-level gate (replacing live readers' L1-couplings with
their rank-80 reconstructions — implementation is nontrivial because forms act on the
full residual input), and the semantics of the principal functionals themselves.

## 62. Context-dependent injection recovers power, not addressability

File: `bilin18_contextual_steering.py` (9 s). Per-sequence gradient steering — each
sequence steered along its own gradient direction:

| target | own movement | cross-talk | (static gradient, for reference) |
|---|---|---|---|
| L5 | 0.71σ | 0.42σ | 1.02σ / 0.60σ |
| L13 | **0.22σ** | **2.88σ** | 0.06σ / 3.70σ |

Registered (a) held: context-dependence lifts L13 own-movement 3.7× over the static
gradient — the *reachability* deficit is partly input-independence. Registered (b)
failed: cross-talk remains 13× the own-movement. **Power returns; selectivity does
not.** The refined final verdict on the stack's opacity: deep coefficients can be
*influenced* by context-adaptive injection but not *addressed* — any injection strong
enough to move one deep functional moves the collective state more, at every
intervention class tested (static direct-path, static gradient, per-sequence
gradient). Per-token injection is the next rung and is registered-open; the pattern
across three rungs predicts it buys further power and no selectivity.

## 63. The vocabulary's words are not the verified axes

File: `bilin18_principal_semantics.py` (11 s; ran under the queue runner during a
session-restart gap — the cron died with the session, the runner survived, and this
write-up is the recovery; the cron now carries a restart-recovery note).

Both registered predictions **failed**, and the failures characterise the vocabulary:

| principal | mass | eff-rank | \|cos\| with verified z | null p95 |
|---|---|---|---|---|
| #1 | 0.198 | 10.8 | 0.14 | 0.28 |
| #2 | 0.105 | 15.8 | 0.09 | 0.33 |
| #3–5 | 0.04–0.08 | 12–20 | 0.01–0.15 | ~0.31 |

- **(a) failed**: the top principal functional does not align with the verified
  z/register direction — below its own permutation null. The causally-verified axes of
  the layer batteries are *not* the principal axes of the cross-reader vocabulary.
- **(b) failed**: no top-5 principal is low-rank (eff-ranks 11–20 of 48). The
  vocabulary's words are medium-rank quadratics — structured well below the 48-dim
  ceiling, but not few-term-nameable objects.

The reconciliation matters: the LORO result (§61) says the vocabulary's *span* carries
the readers' forms at R² 0.71, and the verified surrogates are reader forms — so the
verified axes should live *inside* the span while not being principal. That is the
third appearance of the same split (G_lam leaders vs coverage ordering; solo vs
Shapley): **importance orderings and identity orderings differ in this model**.
Registered next: the containment test — energy of the verified axes' coupling matrices
inside the top-80 principal span, against matched random quadratics.

## 64. The intervention ladder completes; the containment test was ill-posed

Files: `bilin18_pertoken_steering.py`, `bilin18_vocab_containment.py` (9 + 11 s).

**The opacity characterisation is now complete across the full intervention ladder:**

| intervention class | L13 own-movement | L13 cross-talk | addressable? |
|---|---|---|---|
| static direct-path | 0.05σ | — | no |
| static gradient | 0.06σ | 3.70σ | no |
| per-sequence gradient | 0.22σ | 2.88σ | no |
| **per-token gradient** | **0.91σ** | **1.70σ** | **no** |

Registered (a) held (power keeps rising — 15× from static to per-token) and (b)
confirmed: cross-talk still exceeds own-movement at every rung. **Power is fully
recoverable with enough context-dependence; addressability is recoverable at no rung
tested.** The quadratic stack's depth-opacity is now a complete, graded, measured
statement — the strongest single characterisation the program has produced.

**The containment test is recorded as an instrument error, not a finding.** The
numbers came back low (z-surrogate 0.156, leader coupling 0.351 in the top-80 span) —
but the test compared objects from different spaces: the verified z and leader form
act on L1's *input*; the vocabulary is quadratics over L1's *output* directions.
Projecting input-space forms through the output-PCA basis is a category confusion, and
the registered prediction should never have been posed on these objects. The honest
state: the program has no *verified* reader-side (L1-output-coupling) axis independent
of the family the vocabulary was built from, so containment is currently untestable —
what is testable instead is whether vocabulary words are *causally individuated*,
which is queued: steer along principal functional #1's top output direction vs #2's,
and ask whether they move distinct constituencies of reader coefficients.

## 65. Word constituencies: the envelope artifact again — and a post-hoc flip awaiting confirmation

File: `bilin18_word_constituencies.py` (11 s, completed under the runner).

As registered, the test **failed with a violated control** — and the violation is
diagnostic: raw movement profiles of the three vocabulary words correlate at +0.70,
but so does the *random-direction* control (>0.3), meaning the profiles are dominated
by a shared **movability envelope** (which reader coefficients respond to any steering
at all — the same envelope-vs-identity artifact as §55's |abs| matrices, now in its
causal form). Every word is causally live (max moves 0.46–0.55σ, prediction (b) held).

**Post-hoc, labelled as such**: dividing out the movability envelope (each
coefficient's response normalised by its mean response across all four steers), the
residual constituencies are **anti-correlated or uncorrelated** — word1×word2 −0.76,
word1×word3 −0.38, word2×word3 −0.11, and each word distinct from random. If this
survives pre-registration, the vocabulary's words are causally individuated with
*complementary* constituencies — the causal counterpart of their signed
near-orthogonality. Confirmatory version queued with the envelope-normalised statistic
registered in advance (5 words, 2 random controls; mean residual pairwise correlation
≤ 0.1, all pairs < 0.5).

## 66. Pre-registered confirmation: the words are causally individuated, with one degenerate pair

File: `bilin18_word_constituencies2.py` (12 s). The envelope-normalised statistic,
registered in advance this time, on 5 words + 2 random controls:

- **(a) held: mean residual pairwise correlation −0.20** (bar ≤ 0.1). The post-hoc
  flip of §65 survives pre-registration: vocabulary words move complementary
  constituencies of reader coefficients on average.
- (c) held: every word causally live (max moves 0.39–0.67σ).
- (b) failed: one pair reaches +0.64 — two of the top-five words share a constituency.
  The plausible mechanism (recorded as hypothesis): adjacent principal components with
  close singular values rotate freely within their subspace, so neighbouring "words"
  need not be individually well-defined even when the family is.

Verdict: **the vocabulary is causally individuated as a family — distinct, largely
complementary constituencies under steering — with individual word identity unstable
where the principal spectrum is nearly degenerate.** This is the causal counterpart of
the signed near-orthogonality (§56-57), now established at the pre-registered level,
and it completes the chain from the user's original independence question: dense
shared support → near-orthogonal signed functionals → ~80-word shared vocabulary →
causally complementary constituencies.

## 67. The constituencies are corpus-robust: the individuation arc closes

File: `bilin18_constituency_transfer.py` (12 s). Both registered predictions held:

- individuation holds in-distribution: fineweb mean residual pairwise correlation
  **−0.24** (bar ≤ 0.1);
- and they are the *same* constituencies: per-word cross-corpus profile correlations
  0.52–0.94, **mean +0.78** (bar ≥ 0.5).

The vocabulary's causal structure is not a pile artifact — the same words move the
same complementary reader-sets on the training-like corpus. With this, the arc that
began with the user's independence question is complete at every level it was posed:

1. dense shared support (no disjoint sub-structure, §54–55);
2. near-orthogonal signed functionals, universal (§56–57);
3. an ~80-word shared vocabulary, leave-one-reader-out verified (§58, §61);
4. words causally individuated with complementary constituencies, pre-registered and
   corpus-robust (§66–67);
5. steerable at range one, power-recoverable but never addressable at depth (§59–64).

That is the program's final answer to "can the middle be compressed, and in what
coordinates": yes — into a corpus-robust functional vocabulary of ~80 words whose
causal identities are real, whose supports are inseparable, and whose reach is one
layer.

## 68. The words have names after all — correlationally

File: `bilin18_word_naming.py` (11 s). Both registered predictions **failed in the
informative direction** — I predicted, from the program's token-story record, that at
most 2 of 5 words would be token-nameable and none register-shaped. Instead:

| word | ρ (null ~0.08) | ICC | fires on |
|---|---|---|---|
| #1 | **+0.50** | 0.56 | ` your, their, both, the` + whitespace |
| #2 | **+0.56** | 0.56 | `(, [, ", We` + whitespace |
| #3 | −0.17 | 0.22 | (not nameable) |
| #4 | +0.18 | 0.56 | (not nameable) |
| #5 | **+0.48** | 0.57 | ` detected, levels, samples, As` |

**Three of five words carry verified token structure** (determiner/possessive context;
clause-and-bracket openers; a quantitative/measurement register), and four of five have
document-level ICC ≈ 0.56 — *the same number as the original register leader's* (§17).
The vocabulary's principal words are substantially register-shaped, which in hindsight
is what a variance-driven principal decomposition must find when the document mixture
is the dominant variance structure.

Status of these names, stated carefully per the three-for-three record: they are
**verified correlational descriptions** (nulls cleared decisively), not causal claims —
the program's every causal test of a token story has failed, and the queued test
(steering word #5 and reading measurement-vocabulary log-probs) will either make it
four-for-four or record the first causal token-story success. Either outcome is worth
having, and the bar is registered accordingly modest.

## 69. Four-for-four: readable but not steerable

File: `bilin18_word5_causal.py` (11 s). Word #5's causal name test **failed** at 1.3×
against a 1.5× bar — the lowest bar the program has ever set — with the familiar
signature: both steering signs move the *whole* common-token distribution (control
tokens shift 0.23–0.88 nats alongside the measurement set).

The causal token-story record now stands at **four-for-four failures** — layout→
register (§19), bus→determiners (§24), head-1 aiming (§48), measurement-register
steering (here) — while the *correlational* token structure verified decisively in
every one of those same cases. That is no longer a run of bad luck; it is a
regularity of the model worth stating as such:

> **In bilin18, token-level semantics are readable but not steerable.** Directions'
> activations carry verifiable token structure (ρ up to 0.95 against nulls of ~0.08),
> and interventions along those directions never move their named tokens selectively —
> the write always shifts broad distributional mass instead.

The mechanism is plausibly the §56 structure itself: a direction's token correlate
lives in its *activation statistics* on natural data, while its causal write feeds an
overcomplete family of readers that respond collectively — the same
selectivity-not-reachability opacity as §62–64, at the vocabulary level.

Registered decisive test, queued: the strongest correlational name in the program —
layer 0's ρ = 0.95 punctuation axis. If even that fails causal selectivity, the
regularity holds at maximal strength; if it succeeds, the boundary is "steerable only
where naming is near-perfect," which would be its own finding.

## 70. The regularity is graded, not absolute

File: `bilin18_punct_causal.py` (5 s). The five-for-five prediction **failed**: the
ρ = 0.95 punctuation axis shows a **1.7×** causal swing ratio — above the 1.5×
regularity bar, below the 2× "cleanly bounded" alternative. Even the crispest name in
the program steers its tokens only 1.7× more than controls, but it *does* steer them.

Revision, stated plainly: **causal token selectivity in bilin18 is graded by
correlational crispness, and weak even at its best.** The data points so far: ρ 0.95 →
1.7×; ρ 0.48 → 1.3×; weaker names → ≤1×. "Readable but not steerable" overstated the
dichotomy; the true statement is a steep, lossy conversion from correlational to
causal token structure — most of what makes a name readable does not survive into
steering, but a residue does, in proportion. The systematic test (selectivity ratio vs
naming ρ across ~8 named directions, registered Spearman ≥ 0.6) is queued; if the
monotone relation holds, the program closes with a quantitative law for when token
semantics can be causally used at all.

## 71. No law: selectivity is direction-idiosyncratic — and the numbers axis is the first real token-steerable direction

File: `bilin18_selectivity_law.py` (17 s). The graded-law prediction **failed** at
Spearman +0.14:

| direction | ρ | selectivity |
|---|---|---|
| L0 numbers | 0.80 | **3.82×** |
| L0 punct | 0.95 | 1.72× |
| word1 determiners | 0.50 | 1.40× |
| word5 measurement | 0.48 | 1.30× |
| word2 openers | 0.56 | 1.22× |
| L0 #3 | 0.80 | 1.02× |

Naming crispness does not predict causal usability — two directions at identical
ρ = 0.80 span 1.02× to 3.82×. Two things stand:

1. **The numbers axis is the program's first strongly token-steerable direction**
   (3.82×) — causal token control exists in this model; it is just rare and
   unpredicted by every correlational property measured so far.
2. The regularity's third formulation also dies. The honest state: token steering
   works for *some* directions, selectivity ranges 1.0–3.8×, and no measured property
   yet predicts which. Registered next (queued): the **direct-write hypothesis** — 
   selectivity tracks how much of the steered direction's write survives to the
   unembedding aligned with its named set (a mechanism, not a semantic story: layer-0
   writes ride the residual bypass to the logits, and the numbers axis may simply
   write number-logits directly). Predictor computable from weights; registered
   Spearman ≥ 0.7 across the six.

## 72. Selectivity is direct logit-writing: the weights-computable predictor holds

File: `bilin18_direct_write.py` (15 s). The direct-write contrast |DW| — the mean
unembedding-alignment difference between a direction and its named token set,
pure weight algebra — predicts measured steering selectivity at **Spearman +0.77**
(bar 0.7, held) across the six §71 directions. Prediction (b) failed informatively:
punct has the largest DW (183) yet the numbers axis (DW 109) out-steers it 3.82× to
1.72× — DW ranks well but is modulated downstream (a large direct write can be
partially cancelled by the stack's response; the residual ordering is exactly the
§59-64 propagation story).

**The closing synthesis of the token-semantics arc**: where token steering works in
this model, it works because the steered direction *writes its named tokens' logits
directly through the residual bypass* — the most mechanically shallow channel that
exists — and the amount is predictable from weights. It does not work through semantic
routing, which failed every causal test at every depth. This coheres with the whole
program: mechanisms and structure verify; token *stories* fail; and the one causal
token lever is the one that involves no story at all.

## 73. Blind DW: a top-pick predictor, not a ranker — and the arc's missing null

File: `bilin18_direct_write_blind.py` (16 s). Blind on four never-steered directions:
(a) **failed** — Spearman 0.20; the measured range compressed to 1.2–1.7× and the fine
ordering did not transfer. (b) held — the max-DW direction did steer best (1.70×).
Honest scorecard: |DW| picks the best candidate blind and cannot rank the rest; the
retrodictive 0.77 was partly fit to the spread of the original six.

**And a gap the user's "why is this optimal" question exposed, stated plainly: the
orthogonal-functionals arc (§54–57) never ran its random-weights null** — the control
that the program's very first real-model result (§5, the attention census firing
identically on random weights) established as mandatory. Signed near-orthogonality of
high-dimensional objects is exactly the kind of statistic typicality can produce for
free. Queued as the decisive split: recompute the entire §54–61 battery on a randomly
initialised bilin18. Registered predictions carve "SGD found it" from "generic":
signed orthogonality and the dense envelope are predicted to REPRODUCE on random
weights (typicality), while the family compression (eff-rank 80 of 240) and the shared
vocabulary (LORO R² 0.71) are predicted NOT to (training's actual contribution).
Alongside it, the user's robustness hypothesis is queued as a direct test.

## 74. The null broke, the robustness hypothesis half-held

Files: `bilin18_functional_null.py` (broken instrument, recorded),
`bilin18_functional_robustness.py` (8 s).

**The fresh-init null is degenerate**: NaN eff-ranks, cosines of exactly 0, LORO
trivially 1.0 — the freshly initialised model's couplings are ~zero (consistent with
zero-initialised output projections at init). A fresh-init model is the wrong null for
"what did training build" when training grows structures from zero; the right null is
**weight-shuffled** (permute the trained weights' entries — same marginals, destroyed
structure). Queued as the fix.

**The user's robustness hypothesis: main bar held, discipline check caught a
confound.** Real functionals' worst single-direction deletion damage is **0.25σ vs
0.51σ** for sparse-support controls — dense support halves targeted fragility, as
hypothesised. But the isotropic-noise null-prediction failed (real functionals also
show 2.6× better noise SNR, which pure support-density should not provide) — meaning
the sparse controls are not covariance-matched, and part of the "robustness" is the
real forms' alignment with the data covariance rather than support density per se.
Registered fix queued: sparse controls built in the whitened basis, isolating support
density with covariance held equal. Until then: the hypothesis is *supported but not
isolated*.

## 75. The decisive null: SGD built the vocabulary, inherited everything else

Files: `bilin18_functional_null.py` v2 (weight-shuffled), `bilin18_functional_robustness.py`
v2 (covariance-matched controls); 13 + 8 s.

**All four registered splits held** on the weight-shuffled null (same weight marginals,
destroyed structure):

| statistic | trained | weight-shuffled | verdict |
|---|---|---|---|
| signed cosines (within/cross) | 0.11 / 0.089 | 0.03 / 0.03 | **generic** — orthogonality is typicality |
| envelope eff-rank | 2.6 | 2.5 | **generic** — the magnitude template comes from weight marginals |
| family eff-rank | **80** | 191 | **trained** — 2.4× compression |
| LORO R² at r=80 | **0.71** | 0.26 | **trained** — the shared vocabulary is learned |

The answer to "why is this structure there": **dense support and near-orthogonality are
the architecture's defaults — SGD did not select them. What SGD built is the shared
80-dimensional vocabulary**: compressing a generically ~191-dimensional functional
family into 80 dimensions that transfer across readers (0.26 → 0.71 out-of-reader).
Training's fingerprint is the compression and the sharing, nothing else.

**And the robustness hypothesis loses its support under the fixed control**: with
covariance-matched sparse controls, the targeted-damage advantage shrinks to 1.6×
(0.25σ vs 0.39σ, below the registered 2× bar → failed) and the isotropic-noise
discipline check still fails — most of v1's apparent robustness was data-covariance
alignment, not support density. Consistent with the null: density isn't selected *for*
anything; it is inherited, and its incidental robustness benefits are modest.

## 76. The vocabulary is built, not carved

File: `bilin18_vocab_vs_generic.py` (19 s). Both registered predictions held, at the
stronger end: the trained top-80 vocabulary has **0.075** of its energy in the
shuffled-weights top-80 span (random baseline 0.035) and only **0.170** in the *full*
generic 191-dimensional structure. Training did not compress the inherited functional
family — it discarded it and built a nearly-orthogonal-to-generic subspace essentially
from scratch. Combined with §75, the complete origin story of the middle's reading
code: **architecture supplies dense, orthogonal, ~191-dimensional generic machinery;
SGD replaces its content with an 83%-new, 80-dimensional, cross-reader-shared
vocabulary.** The gradient-coupling hypothesis (readers training against a shared
writer converge onto a common code) remains the candidate mechanism, with its
registered scaling test queued: vocabulary dimension should track the writer's output
complexity across writer layers.

## 77. Vocabulary scaling: right ordering, wrong normalisation

File: `bilin18_vocab_scaling.py` (22 s). Results with an instrument inconsistency
flagged before interpretation: this run stacked **un-normalised** coupling matrices,
so large-norm forms dominate the spectrum and the absolute eff-ranks (13–16 for
L0/L1/L3 writers; 5 for L16) are **not comparable** to §58's unit-normalised 80. What
is internally valid is the cross-writer ordering: L16 (5) < L0 (13) < L1 (15) ≈ L3
(16), matching the writers' output-complexity ordering with one adjacent swap —
Spearman exactly 0.80 against the ≥0.8 bar (reported at-bar; n = 4, underpowered).
Registered (b) held: the simple writer (L16) gets a small code. The scaling hypothesis
has weak, suggestive support; the normalised rerun is queued for real numbers.

## 78. Scaling is dead; the code size is universal

File: `bilin18_vocab_scaling.py` v2 (unit-normalised, 22 s). Both registered
predictions **failed decisively**: normalised family eff-ranks are **85 / 79 / 98 /
112** for writers L0 / L1 / L3 / L16 — no relation to writer output complexity
(Spearman −0.20; the *simplest* writer has the *largest* family rank). The
gradient-coupling scaling story is dead.

What replaces it is cleaner: **the reading code's effective dimension is roughly
constant (~80–110) regardless of which layer is the writer**, against the generic
(weight-shuffled) value of ~191. L1's "80" is not a fact about L1 — it is an instance
of a model-wide constant: training compresses every writer's functional family by
about 2× to a near-universal size. Registered next (queued): confirm the denominator —
shuffled-weights families for the other writers should also sit near 191, making the
~2× compression factor universal rather than L1-specific.

## 79. The compression constant is universal — the arc's closing number

File: `bilin18_universal_compression.py` (27 s). Both bars held: shuffled-weights
family eff-ranks are 195–198 for every writer (the generic value is
writer-independent), and the trained/generic compression ratio is **1.77–2.46×
across all four writers**. Training applies a roughly constant ~2× functional
compression to every layer's reading code, onto a near-universal ~80–110-dimensional
size. This is the origin-story arc's closing number: architecture supplies ~195
generic dimensions everywhere; training halves it everywhere; the result is shared
across readers (LORO) and mostly new relative to the generic content (§76).

## 80. Confirmed out-of-sample: the origin story is complete

File: `bilin18_vocab_writer9.py` (19 s). Writer L9, never used in the arc: trained
family eff-rank **76**, shuffled **198**, compression **2.60×** — all three registered
bars held. Five writers now agree: architecture supplies a ~195–198-dimensional
generic reading structure everywhere; training compresses it ~2–2.6× into an
~76–112-dimensional code that is shared across readers, mostly new relative to the
generic content, and writer-independent in size. The "why did SGD find this" question
has its measured answer: **SGD's contribution to the middle's reading structure is one
thing — a universal, shared, built-from-scratch compression of the functional family —
and everything else about the structure (density, orthogonality, the envelope) came
free with the architecture.**

## 81. The matrix-SAE test was vacuous — instrument error, recorded

File: `bilin18_matrix_sae.py` (98 s). All three bars nominally failed, but the run
proves nothing either way: **the dictionary had 300 atoms for 240 functionals**, so
sparse perfect reconstruction is trivially available (assign each functional its own
atom), and that is what both arms found — R² = 1.00 at L0 ≈ 10–12 for trained *and*
shuffled families alike. With more atoms than samples, neither the sparsity nor the
"trained vs generic" comparison carries information. The atom-complexity reading
(median eff-rank 32.5) is likewise uninterpretable, since degenerate atoms ≈ copies of
individual functionals.

The honest version, queued: **held-out sparse coding** — fit 120 atoms on five
readers' 200 functionals, code the sixth reader's 40 held-out functionals, and compare
(R², L0) against the dense 80-basis baseline (LORO R² 0.71 at "L0" = 80). The question
the flawed run failed to ask: does sparse-over-complete beat dense-orthogonal
*out of sample*? Only that comparison can open or close the sparsity door.

## 82. Quiet steering: the projection is free — and useless. Cross-talk is second-order.

File: `bilin18_quiet_steering.py` (12 s), answering the user's penalized-objective
question. Both registered predictions failed, and *not* via the named alternative:

| L13 target | own | monitored cross-talk | unmonitored |
|---|---|---|---|
| raw gradient | 0.06σ | 0.74σ | 0.57σ |
| projected (off 5 monitored gradients) | 0.10σ | 0.68σ | 0.47σ |

- **(d) failed at 0.09**: the target's gradient is nearly *orthogonal* to the other
  coefficients' gradient span — sensitivity subspaces are not shared at first order.
  My shared-subspace story is refuted.
- **(a) failed the informative way**: the projection costs nothing (own-movement
  154% of raw — consistent with 0.09 overlap) **and buys nothing** (monitored
  cross-talk cut only 1.1×). First-order quiet is not quiet.

**The mechanism, at last.** Every coefficient is a quadratic: Δc ≈ gᵀδ + δᵀHδ. The
projection zeroes the *linear* response of the monitored coefficients — and their
movement barely changes, so their response to the injection is dominated by the
**second-order term**: they respond to the injection's *energy* through their forms'
curvature, which no direction choice can remove. Addressability fails not because
sensitivities overlap (they don't) but because **control is linear and collateral is
quadratic** — at magnitudes large enough for deep reach, δᵀHδ beats gᵀδ for every
bystander at once. This is the composition product law and the interchange leak in
their cleanest form yet, and it makes a sharp registered prediction (queued):
selectivity should *improve as 1/‖δ‖* at short range (linear own vs quadratic cross),
while at L13 no magnitude rescues it (the linear own-term is ≈ 0 there).

## 83. CORRECTION to §§60–64: depth is addressable at small magnitude

File: `bilin18_magnitude_sweep.py` (12 s). Registered (b) held emphatically
(selectivity 21.6× at 0.25× magnitude vs 1.5× at 2×; cross-talk log-log slope 1.53,
superlinear as the second-order mechanism predicts; own-slope 0.22 — sublinear,
saturating, (a) formally failed on that half). And **(c) failed in the direction that
forces a correction**: at 0.25–0.5× magnitude, the L13 target steers at **4.7× and
3.0× selectivity** (own-movement 0.15–0.34σ).

> **Correction.** §§60–64 concluded deep coefficients are "individually unaddressable
> from L1 by any tested intervention." All of those tests ran at 1–2× magnitude —
> inside the regime where quadratic collateral dominates. The corrected statement:
> **deep coefficients are weakly but selectively addressable at small magnitudes
> (~0.15–0.35σ at 3–5× selectivity); what is impossible is *large* selective deep
> control** — beyond ~0.5σ at depth, collateral (∝‖δ‖²) necessarily overtakes the
> target (∝‖δ‖, saturating). The intrinsic-limit verdict survives only in this
> magnitude-qualified form. Chain of discovery for the record: the user's
> penalized-objective question → projection null result → second-order mechanism →
> magnitude sweep → correction.

## 84. The vocabulary's edges: one dissident reader, no sparse structure, and a QK code that avoids the MLP code

Three runs close the sample-completeness questions about the 80-dim functional basis
(`bilin18_all_readers.py`, `bilin18_matrix_sae2.py`, `bilin18_qk_vocabulary.py`).

**All 16 MLP readers** (was the six-reader sample biased?): mostly no. Nine of ten
held-out readers reconstruct from the six-reader basis at median R² 0.47–0.92
(registered (b) held). Two registered bars failed informatively: the all-16 family
eff-rank is **139**, above the [70,130] bar — the vocabulary grows sublinearly
(~6 new dimensions per added reader, vs 191 available per reader) but it does grow;
80 was the six-reader vocabulary, not the model's. And **L11 is a dissident**: its
functionals reconstruct at R² −0.10 — worse than predicting the mean. One reader
of sixteen speaks essentially none of the shared code.

**Sparse coding, honest version** (`matrix_sae2`, after §81's vacuous v1): held-out
sparse coding reaches R² 0.64 only at L0 ≈ **105 of 120 atoms** — that is dense
coding wearing a dictionary, and it still loses to the dense-80 baseline (0.71).
Atoms are complex (eff-rank ~32). Registered (a) failed. **The sparsity door is
closed**: the functional vocabulary has no sparser atomic decomposition; it is a
genuinely distributed 80–140 dimensional code.

**QK query-side couplings** (do attention readers share the MLP vocabulary?): the
strongest no on record, with an instrument caveat. The 27 heads' query-side
quadratics reconstruct from the MLP basis at median R² **−0.26**, while random
symmetric matrices reconstruct at **+0.37** — the random control is violated, so
the registered bar's frame was miscalibrated (the fitted basis captures a sizable
generic share of any symmetric matrix). The meaningful comparison is QK *versus*
random: QK couplings sit far below chance, i.e. they concentrate in directions the
MLP vocabulary actively avoids. Attention-pattern reading and MLP reading use
**disjoint quadratic codes**. (Joint family eff-rank 88 — the QK code is itself
compact.) Cross-reader-type sharing: refuted.

## 85. Constraint-release test: pruning the "harmful" spans does not beat the finetune control

File: `bilin18_constraint_release.py` (71 s), testing the user's hypothesis that the
residual stream constrains the model — that removing certain connections plus a short
finetune could beat the intact model's CE. Best available candidates: the three tail
spans (top-32 output directions of L9/L12/L15) whose deletion *improved* pile CE with
no finetune at all (§37–38). Arms, all evaluated on held-out rows never touched by
training: A intact/no finetune 3.943; B finetune-only (200 steps, lr 1e-5) 3.571;
C prune-then-finetune **3.597**.

Registered (b) **failed**: C is 0.026 nats *worse* than B — six times the bar in the
wrong direction — and the shift-artifact null (C ≈ B) is also excluded. Once the
model is allowed to adapt, those spans are genuinely load-bearing: the finetune could
not rebuild what they carry. The earlier "deletion improves CE" was a
distribution-shift artifact, exactly as the §37 fineweb/pile asymmetry hinted — the
spans hurt on *shifted* data while carrying real function on the home distribution.
No evidence yet for released constraints; the dose–response control (600 steps —
was recovery just slow?) is queued, with the note that a 6×-bar gap rarely flips at
3× dose. A fairer future candidate: connections selected by interchange-leak rather
than by raw deletion effect.

**§85 addendum — dose control.** At 600 steps the gap *widens*: C−B = +0.054
(registered b6 ≥ 0.01: held). The spans are load-bearing; recovery is not just slow.
One caveat for the record: finetune-only at 600 steps (3.871) is worse than at 200
(3.571) — the finetune overfits the 256 training rows past ~200 steps — so these
comparisons live in an early-stopping regime. The conclusion is unaffected because
both arms share every schedule detail and the gap holds at both doses (+0.026,
+0.054).

## 86. OV census: the linear sector is a third, indifferent code

File: `bilin18_ov_census.py` (7 s), closing the last registered-open item. Per-head
OV transmit maps (27 heads, layers 2–4) read L1's top-48 output span at median
energy **0.071** — only 1.7× the uniform null of 0.042 (registered (a), ≥3×,
**failed**), and their within-span coordinate weighting is completely uncorrelated
with the quadratic vocabulary's usage (Spearman **−0.01**; registered (b) held).

The sector census is now complete, and the answer is threefold disjointness:
**MLP quadratic readers** share an 80–139-dim vocabulary; **QK quadratic readers**
use a code that sits *below chance* in that vocabulary (§84); **OV linear readers**
are close to indifferent — they transmit L1's principal content barely above a
uniform-random rate and pay no attention to the coordinates the quadratic code
works in. The functional vocabulary is a property of MLP-to-MLP reading
specifically, not of the model's reading in general.

## 87. The dissident is engaged, foreign, and load-bearing; constraint-release refuted twice

Two runs (`bilin18_dissident_l11.py`, `bilin18_prune_l11.py`) close the L11 question
and give the constraint-release hypothesis its second, structurally motivated test.

**Diagnosis** — L11 is not disengaged: its median coupling norm to L1 is 20.7,
rank 4 of 16 ascending (range 17.9–42.4; registered "bottom-2" failed). Its own 40
functionals have eff-rank 27.0, just past the ≤25 coherence bar — a moderately
diffuse family. So L11 reads L1 at normal strength and computes functionals that
simply live outside the shared span: **a foreign code, not a weak signal.**

**Constraint-release, candidate two** — if the foreign code were vestigial or
interfering, removing L11's entire MLP write plus a finetune should beat the
finetune-only control. It does not: C−B = **+0.033** (registered skeptical bar
≥ +0.01 held), slightly worse than the tail-span candidates' +0.026. The dissident
carries ~0.033 nats the finetune cannot rebuild in 200 steps.

The user's residual-stream-constraint hypothesis is now refuted on two structurally
different candidate classes: connections flagged by *deletion benefit* (shift
artifacts, §85) and a connection flagged by *vocabulary foreignness*. Both are
load-bearing once the model is allowed to adapt. If a third candidate class is ever
tried, interchange-leak edges (the product-law cross-terms) are the remaining
motivated choice.

## 88. The dissident has no dedicated consumer, and the QK code is per-layer

**L11 consumers** (`bilin18_l11_consumers.py`): no concentration. Mean-ablating
L11's entire MLP write moves downstream coefficients almost uniformly (0.15–0.21σ
across L12–17; top "consumer" L14 at only 1.2× the cross-layer median; registered
(a) ≥3× and (b) adjacency both failed). The random-shift control came in at 1.9×,
below its 2× bar — so roughly half the measured movement is generic response to an
energy-matched perturbation, and the per-layer numbers are weak evidence. The
robust finding is the absence: **L11's foreign, load-bearing output is consumed
diffusely** — no reader owns it. Together with §87 this looks like a distributed
contribution (calibration-like or genuinely spread mass), not a private channel.

**QK own-family** (`bilin18_qk_family.py`): the 27 heads' couplings have eff-rank
**15.5** against 26.4 for matched random matrices — a real 1.7× compression,
at-bar for the registered ≤15 (another float-boundary case, reported as at-bar).
But leave-one-layer-out reconstruction is weak (median R² 0.18 < 0.3): the sharing
is mostly **within-layer**. QK reading has its own compact code, organized
per-layer rather than model-wide — unlike the MLP vocabulary, which crosses
fourteen of fifteen readers.

## 89. The dissident does concentrated content work; the QK "code" is mostly per-head

**L11's function** (`bilin18_l11_function.py`): both registered signatures returned
the *content* verdict. Ablation damage is concentrated — the top decile of tokens
carries **51%** of total CE damage (registered diffuse-bar <35% failed; past
content features ran >50%) — and the entropy signature is strong too (mean
next-token entropy shifts 8.3× more than under an energy-matched random shift).
The dissident's profile is now complete: normal engagement with L1, a foreign
functional code, no dedicated downstream reader, damage concentrated on a specific
token subset, and a large calibration side-effect. A reader that works alone on
its own token business — written diffusely, felt specifically.

**QK per-layer** (`bilin18_qk_perlayer.py`): within-layer eff-ranks are 6.0–7.9
of 9 — heads barely compress even inside a layer (registered ≤5 failed) — while
the layer codes are mutually distinct (median principal cos 0.24, held). Combined
with §88 (family 15.5/27, LOLO 0.18): QK reading is **mostly per-head**, with mild
within-layer sharing and near-zero cross-layer sharing. The strong shared-vocabulary
phenomenon is specific to MLP quadratic reading at every grouping we can form.

## 90. Constraint-release closed on three candidate classes; the dissident's business is contextual

**Candidate three — the interchange edge** (`bilin18_constraint_interchange.py`):
cutting L17's read of L16's top-8 output span costs 0.183 nats raw; after a 200-step
finetune the model recovers most but stays **+0.067 nats worse** than the finetuned
control — the largest gap of the three candidates (registered skeptical bar held).
The user's residual-stream-constraint hypothesis is now refuted on all three
motivated candidate classes, with an instructive ordering: the more structurally
implicated the connection, the *more* load-bearing it proves —
deletion-benefit spans +0.026, foreign-code reader +0.033, product-law edge +0.067.
Nothing tested behaves like a constraint whose removal frees capacity; everything
tested carries function the finetune cannot rebuild. The arc is closed unless a
fundamentally different candidate class appears.

**L11's damage is contextual, not lexical** (`bilin18_l11_tokens.py`): the §89
concentration (top decile of token *occurrences* = 51% of damage) does not project
onto token *types*: damage is uncorrelated with corpus frequency (Spearman −0.09;
"rare-token specialist" failed) and the per-type profile is only weakly stable
across disjoint held-out halves (split-half 0.35 < 0.5 bar). The random-shift
control held (0.08), so the profile is L11-specific — but it is organized by
context, not by vocabulary item. Consistent with everything else about this model:
token-level stories fail, structural/contextual ones verify.

## 91. Blind edges v1: instrument error (score = loudness), plus a real depth decline

File: `bilin18_blind_edges.py`. All three registered bars failed, and the failure is
diagnostic of the instrument, not the idea: the score trace(C·G₂) grows monotonically
with depth because activation magnitudes grow, so it ranks *identically* to the
loudness-only null (both Spearman −0.85) — the alignment content of the score never
got to speak. The registered comparison (c) existed precisely to catch this, and did.

The anti-correlation itself contains a genuine observation: measured **relative**
edge strength (transplant-induced change over output variance) declines monotonically
through the tail — 0.28–0.36 for edges 5→6 through 7→8, down to 0.04 at 14→15.
Adjacent layers couple progressively less as depth increases, consistent with the
tail's shallow-compressibility. v2 is queued with the magnitude-free score
trace(C·G₂)/(trace C · trace G₂/1152) — pure alignment; registered: Spearman ≥ 0.5
against the same measured effects, and the score must now *beat* the loudness null
by construction-independent margin ≥ 0.4.

## 92. Blind edges v2: tail writes are unaimed — alignment ratio ≈ 1.0 everywhere

File: `bilin18_blind_edges2.py`. With the magnitude-free score, the answer is a clean
negative with the control finally behaving: every adjacent tail edge's alignment
ratio sits at **0.81–1.18** — indistinguishable from the isotropic baseline of 1.0.
Layer i's write is *not* preferentially aimed at layer i+1's quadratically sensitive
directions, at any tail edge. Rank prediction accordingly has nothing to grab
(Spearman −0.34, noise; registered (a2)/(b2) failed; (c) held — alignment beats the
loudness null, which was anti-correlated at −0.85).

Contrast with the node story: the same weights-plus-S machinery located four tail
layers' causal-leader *subspaces* blindly (three of four, energies 0.89–0.97). The
formula sees within-layer structure but the tail's layer-to-layer routing carries no
alignment signature — which redirects the explanation of §91's depth decline
(0.36 → 0.04) to **dilution**: the write shrinking relative to the accumulated
residual, not de-aiming. That is registered and queued as v3: Spearman(write-to-
residual energy ratio, measured effect) ≥ 0.7 over the ten edges.

## 93. The tail routes by dilution: edge strength is the write's share of the stream

Instrument note first: v3's ratios came out ~5×10⁴ — physically impossible — because
the denominator captured the MLP's RMS-*normalized* input while the numerator was the
raw write (units mismatch). v3.1 (`bilin18_edge_dilution2.py`) divides by the raw
residual stream entering the next block. Both registered predictions then held:

| edge | write/stream | measured effect |
|---|---|---|
| 5→6 | 0.252 | 0.277 |
| 9→10 | 0.100 | 0.277 |
| 14→15 | 0.040 | 0.041 |

The dilution ratio declines **perfectly monotonically** through the tail (zero
inversions in nine comparisons) and predicts measured edge strength at Spearman
+0.79. Combined with §92, the tail's layer-to-layer story is complete and simple:
**writes are unaimed (alignment ratio ≈ 1.0), and an edge is exactly as strong as
the writer's share of the stream the reader sees.** Each layer speaks at roughly
constant absolute volume into an accumulating stream, so its marginal voice fades
with depth — no routing structure, just arithmetic. This is the mechanism behind
the depth map's tail: shallow-compressible because nothing downstream is aimed at
anything specific upstream.

## 94. The syntax bus has no upstream supplier — long-range influence is a flat floor

File: `bilin18_bus_origin.py`. Both registered operationalizations failed, and the
decisive number is the control: for every source layer 5–15, transplanting its full
write moves L16's bus coordinates **no more than a matched random 8-dim span**
(bus/random ratios 1.00–1.15). No layer feeds the bus specifically. The registered
excess-over-share metric misfired for an instructive reason: bus movement is nearly
*flat* across sources (0.16–0.32σ) while dilution shares vary 24-fold, so
"excess" was just the reciprocal of the share. Share-proportionality — the §93 law —
describes *adjacent* influence; at ten layers' range, every source's influence has
saturated into the same undifferentiated ~0.2σ floor, the long-range diffusion the
steering arc measured as cross-talk. Even L15, the largest mover (0.32σ), is only
1.14× its own random-span control.

Conclusion: the syntax content L16 forwards is assembled *by L16* from diffusely
accumulated stream state (or arrives via attention, untested) — there is no
station-to-station relay behind the bus. The two-station L16→L17 structure is a
local phenomenon sitting on top of an unstructured supply.

## 95. The chain refuses variables: the 3→4 edge is high-rank, and the weights' aim is marginal

File: `bilin18_chain_bus.py` (written in the operator arc with frozen predictions;
executed only now). P1 held: the coupling operator's directions beat layer 3's own
top output-PCA directions at every k ≤ 8 — the weights do know what layer 4 reads,
slightly better than what layer 3 writes loudest. But the margin is thin (T(8):
0.211 vs 0.195, ~8%), and P2 failed decisively: eight directions of 1152 carry only
**21%** of the full-edge effect (sixteen carry 30%), nowhere near the registered
half. The 2→3→4 chain's long-anticipated "abstraction treatment" is hereby answered:
**the edge has no small variable set.** Its content is genuinely high-rank, and even
at the front of the model — where adjacent coupling is strong (§93: share 0.13–0.25)
— the writes are barely aimed (echoing §92's tail isotropy). The open item closes
with a negative that the report registered in advance as an acceptable outcome.

## 96. Tail profiles complete — and truncation-as-regularization is endemic, not shift-specific

File: `bilin18_tail_profiles.py` (23 s), the report's last open item. Every tail
layer 5–15 now has a measured 8-dim leader-span deletion effect, with three results:

1. **The blind formula extends**: its weights-plus-S spans do real causal damage at
   8 of 11 layers (≥5× the random-span noise of ±0.0009; registered (a) held).
   Damage magnitudes are small everywhere (≤0.025 nats) — individual tail spans
   matter little, exactly as the dilution picture predicts.
2. **Formula-vs-PCA is muddied by a real phenomenon** (registered (b) failed at
   4/11): the comparison breaks down because PCA deletion sometimes does *negative*
   damage — at L9 the formula span costs +0.0060 while the PCA span *improves* CE
   by −0.0156.
3. **Five of eleven layers have a span whose deletion improves home-corpus CE**
   (registered (c) held, far beyond its ≥1 bar): L8/L14/L15 formula-spans and
   L6/L9/L15 PCA-spans, best −0.0156 at L9. The truncation-as-regularization
   pattern, previously seen only on shifted corpora, is endemic to the tail on the
   model's own distribution.

Finding 3 reopens the constraint-release question with the candidates it always
needed: §85's spans were chosen by *shifted*-corpus benefit and proved load-bearing
at home. These new spans benefit the **home** corpus with no finetune. Queued:
stability check (do the negatives replicate on disjoint held-out rows?) and
constraint-release candidate four (delete L9's PCA span + finetune vs finetune-only
control — registered skeptical after three refutations, but this is the first
candidate whose no-finetune sign already points the right way).

## 97. Constraint-release closed for good: even genuinely beneficial deletions lose after adaptation

Files: `bilin18_negative_stability.py`, `bilin18_constraint_c4.py`. The stability
check passed all three bars — the home-corpus deletion benefits are real and
replicate *stronger* on disjoint rows (L9 −0.0211, L15 −0.0133; fresh random spans
inert at ±0.0003). Then candidate four: pruning exactly those two spans plus the
standard 200-step finetune lands **+0.011 nats worse** than the finetune-only
control (registered skeptical bar ≥ +0.005 held).

This is the sharpest form of the phenomenon and it closes the hypothesis on four
candidate classes, the last one selected by the hypothesis's own success criterion:

| candidate class | no-finetune deletion | after both arms finetune |
|---|---|---|
| shifted-corpus-benefit spans | helps (shifted only) | +0.026 worse |
| foreign-code reader L11 | hurts | +0.033 worse |
| product-law interchange edge | hurts (−0.183) | +0.067 worse |
| **home-corpus-benefit spans** | **helps (−0.021, replicated)** | **+0.011 worse** |

The resolution of the apparent paradox: the negative spans genuinely act as noise
at the model's *frozen* operating point — removing them helps the un-adapted model —
but they also carry function, and the finetune re-tunes the rest of the network
around them, reclaiming more than the regularization benefit was worth. Deletion
benefit measures a local property of the frozen model, not spare capacity.
**Nothing in this model is a removable constraint**; the residual stream's
"pressure" is load, all the way down.

## 98. The negative spans redistribute; the bus is assembled, not delivered

**Regularizer signature** (`bilin18_regularizer_signature.py`): both registered bars
held, and the shape is sharper than predicted. Deleting the L9+L15 spans is not a
uniform improvement but a **redistribution**: the hardest-token quartile captures
173% of the net gain — i.e. the deletion *hurts* everywhere else (easy quartile
+0.028 per token) and helps confidently-wrong tokens by more than the total. The
spans sharpen easy predictions and overshoot on hard ones; removing them trades
sharpness for calibration at a net frozen-model profit. One caveat: the random
control was violated at small magnitude (−0.004 on hard tokens, ~9× smaller than
the real spans' effect) — a sliver of hard-token help is generic damping, the bulk
is span-specific. This also explains §97: a finetune can re-tune the sharpness the
deletion sacrificed, reclaiming more than the calibration gain was worth.

**Bus assembly** (`bilin18_bus_attention.py`): both registered predictions failed,
completing the origin story with a double negative. Transplanting L16's *attention
output* across documents moves the bus coordinates by only **0.052σ** — below even
the §94 MLP-source floor (0.16–0.32σ), specificity 1.05. The syntax content L16
forwards arrives neither from any upstream MLP write nor through L16's own
attention gathering: **L16's MLP computes the bus from the accumulated token-local
residual state.** The model's strongest interaction is assembled in place from
ingredients no single component supplies — the interchange is real, its supply
chain is the whole stream.

## 99. The finetune erases the signature; assembly regression needs an aligned rerun

**Reclaim anatomy** (`bilin18_reclaim_anatomy.py`): the registered *alternative*
landed. After both arms finetune, the pruned model's hard-token advantage is gone —
the +0.010 gap spreads roughly evenly (hard quartile +0.002, easy +0.005). The
frozen-model redistribution signature (§98) says nothing about adapted models: a
200-step finetune re-tunes the calibration/sharpness trade across the board rather
than fighting the deletion locally. §97's conclusion is unchanged and now fully
mechanistic: deletion benefits are frozen-point local properties, erased by the
first opportunity to adapt.

**Bus assembly v1** (`bilin18_bus_assembly.py`): instrument error, recorded. The
stream features were captured from a separate 256-token forward while the bus
targets came from a 257-token run — per-row misalignment garbles the stream and
early-stream R²s (−0.08, −0.02), and the one internally aligned feature (L16's own
attention output, R² +0.33) is the only valid number. It creates a genuine tension
worth resolving: attention output *correlates* with bus coordinates (0.33) though
its cross-document content is causally inert (0.05σ, §98). The aligned v2 is queued;
prediction updated: stream-in R² ≥ 0.6 stands, and the attention correlation should
be explained away as shared position/context signal (partialling out the stream
should drop it below 0.15).

## 100. The bus is linearly readable, carried from the bottom, and deliverable only in place

File: `bilin18_bus_assembly2.py` (aligned rerun after §99's instrument error). Three
numbers complete the syntax-bus origin story:

- The stream entering L16 predicts the bus coordinates at held-out **R² 0.979 —
  with a linear map**. The bus is a function of token-local stream state, and an
  almost perfectly linear one, despite the MLP computing it being quadratic.
- Attention's apparent correlation (0.33) is fully explained away on the stream
  residual (−0.16): shared context signal, no independent contribution — matching
  its causal inertness (§98).
- Registered (c) **failed informatively**: the stream at *L2* already predicts the
  bus at R² 0.659 — two-thirds of the variance is determined near the bottom of the
  network and carried upward, refined from 0.66 to 0.98 along the way, readable
  everywhere, but causally *deliverable* only as L16's own write (§94: transplants
  at every station move nothing).

The picture: the model's strongest interaction runs on content that is old,
distributed, and linearly exposed — the L16→L17 relay is special not because its
information is special but because L16 is where that information is written loudly
enough, and freshly enough, for L17's quadratic to couple to it. Queued next:
whether the early-determined share is mere token identity (lexicality control),
and whether the tail's MLPs are *generally* this linear at their operating point —
which would tie the whole tail phenomenology (unaimed writes, dilution routing,
shallow compressibility) to effective linearity.

## 101. The bus is lexical-headed, and nonlinearity lives in the middle

**Lexicality** (`bilin18_bus_lexicality.py`): the registered alternative landed. The
current token's embedding alone predicts the bus coordinates at held-out R² 0.536 —
the early-determined core (§100's 0.659 at L2) is mostly **token identity**, with
early context adding only ~0.12. The "syntax bus" label overreads: the channel's
composition is ≈0.54 token identity + 0.12 early context + 0.32 accumulated later
context (to 0.98 at L16). A more accurate name: a lexical-headed channel with
contextual refinement. Propagated to the report.

**Effective linearity map** (`bilin18_effective_linearity.py`): all three registered
bars failed, and the measured shape is better than the hypothesis. Linear
predictability of each MLP's write from its input stream is not monotone in depth:

    L1-3: 0.65-0.76 | L4: 0.89 | L6-L10: 0.52-0.62 (minimum) | L15: 0.82 | L16: 0.97 | L17: 0.95

**Nonlinearity concentrates in the middle** — exactly the layers the depth map
called incompressible and the functional-vocabulary arc found hardest — while the
final two layers are almost purely linear at their operating point. Two
consequences: (1) an independent, registered-prediction-free cross-validation of
the depth map by an entirely different instrument; (2) a sharp tension, queued for
test: the model's strongest verified interaction (L16→L17, the composition product
law, +0.067 under cut-and-finetune) sits between its two *most linear* layers. If
replacing L17's MLP with its fitted linear map (R² 0.95) kills the 16→17 excess,
the whole interaction lives in the 5% nonlinear residue — the product law as a
thin quadratic skin on a linear pipe.

## 102. The product law is the quadratic skin on a linear pipe

Files: `bilin18_linearized_interchange.py` (v1 VOID — the linear-replacement hook
ran after the span-ablation hook and overwrote it, making d17 ≡ 0 and the excess
drop an artifact; recorded), `bilin18_linearized_interchange2.py` (hook order
fixed). The v2 numbers, all three registered bars held:

| arm | d16 | d17 | joint | excess |
|---|---|---|---|---|
| real L17 | +0.212 | +0.490 | +0.844 | **+0.143** |
| linearized L17 (R² 0.95 stand-in, base within 0.10) | +0.312 | +0.489 | +0.804 | **+0.002** |

Replacing L17's MLP with its fitted linear map kills **98%** of the 16→17
interaction excess while leaving L17's own function intact (d17 unchanged at
+0.489 — the span's contribution is carried by the linear part). The composition
product law, the interchange leak, the +0.067 cut-and-finetune load — the model's
strongest interaction is mechanistically identified as **the quadratic cross-term
of a 5%-by-variance nonlinear residue** riding on an otherwise linear layer.

One new thread, registered and queued: under linearized L17, upstream damage costs
*more* (d16 +0.312 vs +0.211) — the nonlinear residue appears to *absorb* upstream
damage (gain-control/compensation). And the map's causal complement: linearizing
middle layers (R² 0.52–0.62) should be catastrophic where linearizing L16/17 was
cheap — queued as the causal version of §101's nonlinearity map.

## 103. CORRECTION to §101's reading: functional nonlinearity is front-loaded

Files: `bilin18_quadratic_compensation.py`, `bilin18_linearize_middle.py`.

**Compensation is channel-specific, not gain-control** (registered (a) failed 2/4):
the nonlinear residue absorbs damage to L16's principal span (real +0.212 vs
linearized +0.312, ratio 0.68) but not matched additive noise (0.95), and random
spans are inert under both arms (no damage to absorb — a design flaw in the
prediction, recorded). The absorption lives on the same channel as the product-law
coupling: the quadratic skin both *creates* the 16→17 interaction and *cushions*
that channel's damage — one mechanism, two faces.

**Linearization cost does not track nonlinear variance** (all §101-followup bars
failed, Spearman −0.02):

    L2: +0.228 | L4: +0.080 | L7: +0.026 | L9: +0.035 | L13: +0.041 | L16: +0.015

> **Correction to §101's interpretation.** "Nonlinearity concentrates in the
> middle" is true of *variance* (linear maps explain only 52–62% of L6–10's
> outputs) but false of *function*: replacing a middle layer's MLP with its 55–60%
> linear map costs almost nothing (+0.03 nats), while replacing L2's 70% map costs
> +0.23. The middle's unexplained variance is largely loss-irrelevant on
> distribution; **functionally necessary nonlinearity is front-loaded at L2**. The
> report's "nonlinear computation concentrates where the program found
> incompressibility" is corrected accordingly.

The compression implication is queued as the joint test: if layers 5–17 are all
individually near-linearizable, is the model literally "a few nonlinear layers on
a linear pipe"? Registered: (a) joint linearization of 5–17 costs ≤ 1.3× the sum
of individual costs (linearization removes the very machinery that makes damages
interact); (b) the full joint cost stays ≤ 0.6 nats; (c) adding L2 to the joint
adds ≥ 0.15 (the front's function is real).

## 104. The naive linear pipe fails: approximation errors compound, and L5 is an outlier

File: `bilin18_linear_pipe.py`. Registered (a) and (b) failed, (c) held (+0.29 for
adding L2):

- **L5 is an outlier**: its individual linear stand-in costs **+1.51 nats** — 58×
  L7's cost — despite L5 having a *higher* linear R² (0.78 vs 0.55). Whatever L5's
  low-variance components do, they are functionally critical and the ridge fit
  loses them. (This inverts §103's lesson once more: neither high nor low linear
  R² predicts functional cost; the map from variance to function is genuinely
  uninformative in this model.)
- **Joint linearization is superadditive at 2.68×** (sum of individual +2.05,
  joint 5–17 +5.49; even the middle block 6–10 alone is 2.1× superadditive). My
  registered subadditivity story — "linearization removes the interaction
  machinery" — was wrong. The better explanation: each linear stand-in is a local
  approximation, valid on the distribution it was fit on; stacking them drifts
  every downstream layer's input off that distribution, and the fits degrade in
  cascade. Composition fails here not through quadratic cross-terms but through
  **approximation-validity drift**.

The rescue is queued with a registered bar: refit each stand-in *sequentially* on
the partially-linearized model's own activations (front-to-back). If drift is the
mechanism, sequential refitting should recover most of the gap (registered: refit
joint ≤ 1.5 nats, and the L5 refit alone ≤ 0.5). If even refitting fails, the
tail's function is irreducibly nonlinear *in composition* — a stronger statement
than any single-layer measurement can make.

## 105. INSTRUMENT CORRECTION: the lambda-mixing mismatch contaminates §§102–104's stand-in costs

The sequential-refit run (`bilin18_pipe_refit.py`) solved §104's L5 anomaly by
accident: under a self-consistent protocol L5's stand-in costs **+0.029**, not
+1.51. Diagnosis, confirmed in the architecture: each block applies
`x = λ₀·x + λ₁·x₀` *inside* its forward, so the hook-based evaluators in
§§102–104 captured the block input **before** λ-mixing while the linear maps were
fit on activations recorded **after** it. Every stand-in was applied to an input
it was never fit for; the damage depends on how far a block's λs sit from (1, 0),
which is why some layers looked cheap and L5 looked catastrophic.

Status of affected claims, pending the consistent rerun (queued):
- §104's "L5 outlier" and the 2.68× superadditivity: **withdrawn** (artifacts).
- §103's per-layer costs and the "functional nonlinearity is front-loaded"
  correction: **suspended** — to be re-measured.
- §102's 98% interaction-kill: internally consistent (both arms shared the bug)
  but the stand-in's true fidelity needs re-verification.

What survives cleanly is the refit run itself (self-consistent by construction):
**a 13-layer sequentially-refit linear pipe costs +1.56 nats** — at-bar against
the registered 1.5, far from the irreducible-nonlinearity alternative (>3), with
per-layer marginals growing with depth (registered (c) held: composition drift is
real, just not catastrophic). Registered for the rerun: (a2) consistent individual
costs are ≤0.1 everywhere except L2, and L2 ≥ 3× the median (front-loading
retested); (b2) the §102 interaction-kill persists at ≥70% under the consistent
protocol.

## 106. The linearization arc, final numbers on a clean instrument

File: `bilin18_consistent_linearization2.py` (v1's interchange arm was void — mlp
hooks never fire in the manual forward; its individual-cost arm was valid and is
carried over). The consistent protocol settles everything the λ-mixing bug touched:

- **Individual linearization costs**: L2 +0.109 | L4 +0.054 | L5 +0.025 | L7 +0.036
  | L9 +0.032 | L13 +0.046 | L16 +0.033 | L17 +0.096. **Front-loading reinstated**
  (§103's claim, halved magnitudes): L2 is 3× the median; the middle stays cheap
  despite its high nonlinear variance. L17's stand-in genuinely costs +0.096 (the
  contaminated +0.10 was accidentally right).
- **The §102 interaction-kill survives, revised 98% → 79%**: real excess +0.143,
  linearized-L17 excess +0.030. Four-fifths of the model's strongest interaction is
  the quadratic skin; a real +0.030 residue survives linear L17 (routed through
  attention or the final norm — open thread, small).
- **The sequential-refit pipe stands**: 13 layers linearized for +1.56 nats total,
  marginals growing with depth — composition drift is real and bounded.

Arc summary: variance-nonlinearity peaks in the middle, functional nonlinearity
peaks at the front (L2) and at L17's interaction skin, and thirteen of eighteen
layers can be replaced by refit linear maps for about a tenth of a nat each.

## 107. The functional-nonlinearity map is complete — and it crowns L1

File: `bilin18_front_map.py`, all three registered bars held (a rare clean sweep):

    L0 +0.176 | L1 +0.282 | L2 +0.109 | L3 +0.092 | L4 +0.054 | mid/tail ~0.03 | L17 +0.096

The map is **unimodal with its peak at L1** — the layer this program spent the most
effort reading (densest interactions, source of the 80-word functional vocabulary,
target of every steering experiment) is independently the model's most functionally
nonlinear layer. Two entirely different hardness measures — interaction density
measured by intervention, and linearization cost measured by replacement — agree on
where the real computation is.

Final synthesis of the arc: **bilin18 is five genuinely nonlinear layers (L0–L4,
declining from L1), a nearly-linear thirteen-layer pipe (individually ~0.03 nats
each to replace; +1.56 cumulatively due to drift), and one quadratic interaction
skin at L17 that carries four-fifths of the model's strongest coupling.** Queued:
whether the pipe's cumulative drift flows through attention reading off-manifold
state (pattern-clamped hybrid; registered: clamping patterns to base-model values
cuts the refit-pipe cost ≥40%).

## 108. The drift is stream-borne; patterns exonerated

File: `bilin18_clamped_pipe.py`. Registered (a) failed cleanly with the control
perfect: clamping every attention pattern to base-model values is exactly free
(+0.0000 — the dual-forward instrument is sound) and cuts the refit pipe's cost by
only **1%** (+1.564 → +1.547). Attention patterns are not the drift channel. The
pipe's compounding cost accumulates in the residual stream itself: each linear
stand-in's approximation error feeds the next stand-in's input directly, no
attention mediation required. Combined with §§92–93 (unaimed writes, dilution
routing), the tail's error dynamics are as plain as its information dynamics:
everything is carried by the stream, additively.

Queued, to unify the two crowns: L1 is both the source of the functional
vocabulary (§58) and the most functionally nonlinear layer (§107, +0.282). Are
these the same fact? Partial linearization: replace L1's MLP output only within
its top-48 principal span (the coordinates every reader's quadratic consumes),
or only in the complement. Registered: (a) span-only linearization costs ≥ 60% of
full; (b) complement-only ≤ 40%; (c) the two parts sum to full within 15%
(additivity sanity).

## 109. The two crowns are different facts: the vocabulary is the linear part of L1

File: `bilin18_l1_span_linearization.py`. All three registered bars failed, in the
most informative direction of the whole linearization arc:

| what is linearized | share of write energy | cost | share of full cost |
|---|---|---|---|
| top-48 principal span (the vocabulary channel) | 59% | +0.034 | **12%** |
| complement (1104 low-variance dims) | 41% | +0.150 | 53% |
| both (full) | 100% | +0.282 | 100% |
| random-48 control | ~4% | +0.001 | matches energy share |

The vocabulary span — the coordinates every reader's quadratic consumes, the
target of every steering and naming experiment — is the comparatively **linear**
part of L1's computation. The functional nonlinearity that makes L1 the model's
most expensive layer to replace lives in the **low-variance complement** (53%)
and in the interaction between the halves (the missing 35%: halves sum to 0.183,
full is 0.282 — superadditive across the split). Variance ≠ function, this time
spectrally within a single layer: the model's readable interface is nearly
linearly generable, and its hidden computation sits below the principal spectrum.

Queued: spectral localization — linearization cost of the top-k span for
k ∈ {48, 128, 256, 512}, with random-k controls. Registered: cost stays ≤30% of
full through k = 256 (the nonlinearity is deep in the spectrum, not just past 48);
monotone in k; random controls track energy share.

## 110. L1's nonlinearity lives in a mezzanine band

File: `bilin18_spectral_nonlinearity.py`. Registered (a) failed — the nonlinearity
is not deep in the spectrum either. The cost curve by rank band (marginal):

    ranks 1-48: +0.034 | 49-128: +0.034 | 129-256: +0.056 | 257-512: +0.086 | 513-1152: +0.007

Monotone in k (held), random controls track energy share (held), and the
complement-of-512 alone is nearly free. Three spectral zones of L1's write:

1. **The interface** (top ~48, 59% of energy): what every reader consumes, nearly
   linearly generable.
2. **The mezzanine** (ranks ~50–500, a third of energy): where the functionally
   necessary nonlinear computation concentrates — half the layer's replacement
   cost in these mid-variance directions.
3. **The deep tail** (beyond ~512): functionally inert.

The model's most important layer keeps its loud output linear and its hard
computation quiet — mid-variance, below everything the reading program measured.
Queued: the causal closure — mean-ablating the mezzanine band (ranks 129–512)
versus the top-48 interface versus a random-384 span. Registered: (a) mezzanine
ablation damage ≥ 2× interface ablation (the function is there, not in the loud
part); (b) random-384 well below both.

## 111. Content vs computation: the two operators disagree, and that completes the picture

File: `bilin18_mezzanine_ablation.py`. Registered (a) failed, informative: for
*ablation* the ordering reverses — deleting the top-48 interface costs +0.179 while
deleting the mezzanine costs +0.078 (random-384 control +0.017, held). No
contradiction with §110; the two operators measure different things:

- **Ablation removes content.** The interface carries the important content
  (+0.179) — content that §109 showed is *generated almost linearly*.
- **Linearization removes generation.** The mezzanine's content is smaller, but
  producing it *requires* the quadratic (half the layer's linearization cost).

L1 in one line: **loud linear content, quiet nonlinear computation** — the
interface is what matters downstream, the mezzanine is where the layer works.

**The λ table** (read from weights, user's request): every block mixes
`x = λ₀·x_prev + λ₁·x₀` (x₀ = token embeddings) before attention, and the values:
λ₁ ≈ **8.0 at nearly every layer** — each block re-injects the raw token embedding
at weight 8 — while λ₀ collapses to ≈ 0.013 at L1 and ≈ 0.065 at L5: **L1 and L5
nearly discard the accumulated stream and recompute from embeddings.** This
explains at a stroke: the bus's lexical head (§101 — every layer's input is
embedding-dominated by construction), L1's role as the vocabulary source (it
computes on nearly raw embeddings), and why the λ-mixing instrument bug (§105) hit
hardest exactly at L5. Relative-norm measurement queued (λ-weighted term sizes per
layer), plus the user-suggested compressed-quadratic-on-the-residual test.

## 112. Corrections from the norm table; the residual quadratic is not PCA-compact

**Correction to §111's inference.** The raw λs misled me: λ₀ ≈ 0.013 at L1 looks
like a stream reset, but the stream it multiplies is huge, so the λ₀·x_prev term
(RMS 18.3 at L1, 66.5 at L5) still exceeds the 8·x₀ embedding term (RMS 8.0, 5.1).
**No layer is embedding-dominated** (registered (a) failed); the embedding
re-injection is a constant-size 8.0 whisper that fades below 1% of the stream by
the tail. The lexical bus (§101) is therefore carried by embedding information
*persisting in the stream*, not by re-injection dominance. What the norm table
found instead:

- **L0 is the real reset**: its MLP writes at RMS **1436 into a stream of ~6** —
  the post-L0 stream essentially *is* L0's computation on raw embeddings.
- **The tail turns the volume up**: MLP write norms fall through the middle
  (dilution, §93) then surge at the end — L15: 332, L16: 938, **L17: 1851**, the
  loudest write in the network, exceeding the stream it joins. The last layers
  are output-preparation amplifiers; registered (c) (monotone decline) failed on
  exactly this surge.

**The user-suggested compressed quadratic** (`bilin18_quadratic_residual.py`):
the linear-fit residual is **not compact in the input's principal subspace**. A
full quadratic on the top-64 input PCs captures only 19% of L17's residual
variance and ~0% of L9's (registered (a) failed, (b) held). Since each MLP *is*
exactly quadratic, this means the quadratic's active input subspace is
misaligned with the input's principal directions — the same off-principal motif
as L1's mezzanine (§110), now on the input side. Queued: the same fit in the
**input-mode Gram eigenbasis** (the quadratic's own preferred coordinates, from
weights) — registered: Gram top-64 captures ≥ 2× what PCA-64 did at L17, and
≥ 0.25 absolute at L9; if it also fails, the residual is diffuse in every basis
and "compressed quadratic" is closed.

## 113. The compressed-quadratic door closes; the residues are diffuse in every natural basis

File: `bilin18_gram_residual.py`. The user-suggested rescue — fit the quadratic in
the layer's own input-mode Gram eigenbasis instead of input PCA — fails too: L17's
residual capture peaks at 0.203 (k=32) and *drops* at k=64 (0.126; the added
features fit noise), L9 stays at ~0 and goes negative. Combined with §112:
**the nonlinear residues are high-rank and diffuse in both the data's principal
basis and the quadratic's own weight-preferred basis.** The MLPs are exactly
quadratic, so the residue is a quadratic form — but one spread across so many
input directions that no 64-dim restriction sees more than a fifth of it. The
compressed-quadratic direction is closed; this is the same diffuseness the
program has met everywhere (dense interactions, distributed vocabulary, mezzanine
computation), now in the approximation algebra.

**User prediction registered** (2026-08-17, verbatim intent): L11, the dissident
w.r.t. L1's vocabulary, "reads from other layers" — i.e. it should share a code
with some *other* writer. Queued: L11's reconstruction from the shared reader
basis computed over writers L0 and L9 (the two other writers with measured
vocabularies). Registered per the user: R² ≥ 0.4 for at least one other writer;
alternative: L11 is a universal dissident (its own private input coordinates).

## 114. The dissidence is L1-specific — and the strong shared code is too

File: `bilin18_l11_other_writers.py`, testing the user's registered prediction
(L11 "reads from other layers"). The 0.4 bar failed, but the *control* failed
with it, and that combination is the finding:

| writer | L11 | healthy reader L12 (control) | random |
|---|---|---|---|
| L0 | +0.15 | +0.12 | +0.07 |
| L9 | +0.25 | +0.19 | +0.07 |

Over writers other than L1, cross-reader code sharing is weak for *everyone*
(control violated at 0.12–0.19 vs the ≥0.5 bar) — the famous LORO 0.71 belongs to
writer L1 specifically. And inside that weak regime **L11 is indistinguishable
from a normal reader — in fact slightly above the control at both writers.** The
§84 anomaly sharpens: L11 is not a generally foreign layer; its dissidence is
specific to the one writer (L1) whose vocabulary the rest of the model shares
most strongly. The user's prediction holds in its qualitative form — L11 relates
normally to other layers — while the strong-sharing bar was unreachable because
strong sharing itself is an L1 phenomenon. (Caveat: the reader set here swapped
L9→L15 to avoid writer/reader overlap; a weaker basis than §84's.)

## 115. Attention compensates too; the last excess suspect is the final norm

File: `bilin18_residual_excess_locator.py`. Registered (a) failed with a perfect
control (clamp exactly free under no damage), and the failure is a discovery:
freezing L17's attention to its clean-run values **doubles** the surviving excess
(+0.030 → +0.066). L17's attention does not carry the residual interaction — it
**absorbs** part of it, responding to the damaged stream in a way that reduces
the joint penalty. That is the second damage-compensator found at this layer
(§103 found the MLP's quadratic residue cushioning the same channel). The
interaction accounting so far: 79% L17's quadratic skin, and the remainder is
*more* than measured — attention hides a chunk of it. By elimination, the
surviving excess must live in the **final rms_norm**, the only nonlinearity left
on the linear-L17 path. Queued: freeze the final norm's per-token scale to
clean-run values under the same arms; registered (a) that kills ≥60% of the
free-attention excess (+0.030), closing the interaction ledger exactly; control
(b) norm-freeze free under no damage.

## 116. Span-ablation damage is largely norm-mediated at the tail

File: `bilin18_final_norm_test.py`. Registered (a) failed at −419% and the control
was violated (freezing the hybrid's final gain to the *real* model's clean values
shifts base by 0.0097 — wrong reference; instrument flaw recorded). But the effect
that showed up dwarfs both problems:

| arm | d16 | d17 | excess |
|---|---|---|---|
| free final norm | +0.283 | +0.485 | +0.030 |
| gain frozen to clean | **+0.046** | **+0.083** | +0.158 |

**Six-fold collapse of the individual damages.** Mean-ablating a top-8 span
removes a large share of the residual's energy; the final rms_norm then re-scales
the whole vector, distorting every logit. Most of the measured "content damage"
of tail span-ablations is this global gain response — the spans' content-level
value is closer to +0.05/+0.08 than +0.28/+0.49. And at content level the 16→17
interaction is *bigger* (+0.158), not smaller.

Scope: this does not void earlier results (the rms_norm is part of the model;
the damages are real for the intervention) but it re-frames their
*interpretation* wherever a large-energy span was ablated at the tail — most
importantly the composition-law arc's d16/d17 inputs. Queued, with the control
fixed (each arm frozen to its **own** no-damage gain): (a) control exact ≤0.002;
(b) content-level excess ≥0.05 — the interaction is real at content level, not a
gain artifact; (c) norm-mediated share of individual damage ≥60% at both spans,
on the real (non-hybrid) model.

## 117. Content-level accounting: the interaction is even bigger than we said

File: `bilin18_norm_mediation.py` (control exact at 0.0000). Final numbers on the
real model, per-arm gain-frozen:

| | raw (free norm) | content-level (gain frozen) | norm-mediated share |
|---|---|---|---|
| d16 | +0.212 | +0.138 | 35% |
| d17 | +0.490 | +0.089 | **82%** |
| excess | +0.143 | **+0.205** | (negative — norm *masks* interaction) |

Registered (c) failed at the 60%-both bar (35%/82%) but the refined picture is
sharper than the prediction: L17's famous span damage is largely the final norm
re-scaling a suddenly-quieter residual (it writes at RMS 1851; deleting its top-8
span guts the final vector), while L16's is mostly content. And the interaction
excess **grows** when the gain channel is frozen: at content level the 16→17
interaction (+0.205) is 90% the size of both individual content damages combined.
The composition law is not a gain artifact — the gain was hiding a third of it.
Queued: the content-level version of §106's quadratic-skin kill (does linearizing
L17 still remove ≥60% of the *content* excess?), which either re-grounds or
re-opens the skin story at the corrected level.

## 118. REVISION of the skin story: at content level, the quadratic skin carries only 21%

File: `bilin18_content_kill.py` (controls exact). Linearizing L17 under per-arm
frozen final gain kills only **21%** of the content-level excess (+0.205 → +0.162).

> **Revision to §§102/106.** The "linearizing L17 kills 79–98% of the interaction"
> result was dominated by the *norm channel*: L17's quadratic amplifies the two
> ablations' joint effect on residual energy, and the final gain converts that to
> loss. At **content level** the 16→17 interaction survives a linear L17 nearly
> intact. The quadratic-skin story explains the gain-mediated interaction, not
> the content interaction.

What can carry a +0.162 interaction through a *linear* L17 with a *frozen* final
gain? Logit deltas through a linear map superpose exactly, so the prime suspect
is **loss curvature**: CE is convex in logit perturbations — two deltas each
costing little can cost more together even when they add exactly. Queued, the
final decomposition: (a) measure logit-delta additivity under linear-L17 +
frozen-gain (registered: relative residual ≤15% — deltas do superpose); (b)
evaluate CE at base+Δ₁₆+Δ₁₇ (synthetically added logits) vs the true joint run —
registered: this curvature term accounts for ≥70% of the surviving excess. If
both hold, the composition law finally decomposes into three named parts:
quadratic content cross-term (~21%), norm-gain channel, and loss curvature.

## 119. The composition excess decomposed: curvature drives it, compensation halves it

File: `bilin18_curvature_decomposition.py` (stale print labels in the log — the
registered predictions and numbers are in the JSON; recorded). Both registered
bars failed, in a direction that completes the accounting:

- Logit deltas through the linear-L17/frozen-gain pipeline are **not** additive
  (relative residual 0.239) — L17's attention and the upstream real layers still
  interact.
- The synthetic arm (forcing the logits to superpose) produces an excess of
  **+0.461 — 2.8× the true +0.162**. Convex loss curvature alone would make joint
  damage far worse than observed; the network's genuine joint response
  *destructively interferes*, cancelling roughly two-thirds of the curvature
  penalty.

Final decomposition of the 16→17 composition excess (raw +0.143): a **norm-gain
channel** (§§116–117, masks part of it), a **loss-curvature term** (would
contribute +0.46 at content level), and a **compensation term** (−0.30, the
network attenuating joint perturbations) leaving +0.162 content-level net, of
which L17's quadratic carries ~21%. Compensation is now the third-time motif:
the quadratic residue cushions its channel (§103), L17's attention absorbs
interaction damage (§115), and the joint logit response subadds (here). This
model does not merely fail gracefully — it actively cancels compound damage,
and the "product law" headline number was always the small residue of that war.

## 120. Norm-mediation is loudness-concentrated; L16 is the true content heavyweight

File: `bilin18_content_profiles.py` (control exact). Both registered bars failed
in clarifying directions:

- **No depth gradient** (Spearman 0.16): norm-mediation concentrates at the
  *loudest writers* — L17 (81%, RMS 1851) and L5 (59%) — while mid-tail spans are
  mostly content (5–36%). The mediator is the span's share of final-vector
  energy, not depth per se.
- **Ranking mostly survives** (tau 0.79) with one big exception: L17's span
  demotes from +0.489 to +0.093, making **L16 the model's largest content-level
  span (+0.148)**. §48's syntax-bus emphasis lands better than ever: the content
  is at L16; L17's apparent dominance was mostly gain.
- The deletion-improves spans are **content-real and slightly norm-masked**:
  L9 −0.023, L6 −0.016, L15 −0.015 at content level (more negative than raw).
  §96–97's regularization story survives the accounting intact.

Queued, closing the compensation arc: is compound-damage cancellation (§119)
general? Six random tail-layer span pairs, true joint excess vs synthetic
superposed-logit excess. Registered: (a) synthetic > true for ≥5 of 6 pairs;
(b) median cancellation ≥40%.

## 121. Compensation is an output-end phenomenon

File: `bilin18_general_compensation.py`. Registered generality failed (3/6 pairs,
median cancellation 24%): compound-damage cancellation concentrates at pairs
involving the output end — L13–L17 (36%), L11–L15 (24%), and L9–L16 (244%: the
joint deletion is net *negative*, −0.024) — while small mid-tail pairs show
**anti-compensation** (L7–L13 at −148%: the true joint excess exceeds what loss
curvature alone predicts; genuine amplifying interaction). Caveat: most mid-tail
pair effects are small (+0.002–0.007), so their percentages are noisy; the
16/17-involving numbers are solid.

Reading: the three absorbers found in this arc (L17's quadratic residue §103,
L17's attention §115, the 16→17 destructive interference §119) are all part of
the same **output-preparation machinery actively stabilizing logits** — a local
property of the model's final stage, not a network-wide principle. The mid-tail
neither absorbs nor amplifies much; it just carries.

The L9–L16 joint-negative (−0.024, deleting the §96 regularizer span together
with the model's biggest content span *helps* overall) is queued for replication
on disjoint rows before it earns interpretation.

## 122. The regularizer-content interaction is real: −0.022, replicated

File: `bilin18_l9l16_replicate.py`. Registered (b) held with a textbook
replication: the beneficial L9×L16 interaction is **−0.0216** on disjoint rows
(vs −0.0237 original), with the random-pair control perfectly additive (−0.0005).
Registered (a) failed for an embarrassing reason recorded here: I had misread
§121's −0.024 as the *joint total* when it was the *excess*; the joint total is
positive (+0.165) on both row sets. The replicated fact: **deleting L9's
overshoot-trimming span cushions L16 content damage by ~0.022 nats** — when the
model's biggest content span is gone, the sharpener (§98) sharpens corrupted
content, so removing it helps. The regularization story and the composition
accounting meet: negative spans are not free-floating benefits but interaction
partners whose sign depends on what else is broken.

## 123. The product law's final scope

File: `bilin18_content_product_law.py`. Across a 3×3 grid of *damage shapes*
(span sizes 2/8/32 at both layers), the scalar product law fails at both levels —
raw R² 0.12, content R² 0.47 — **as §48 predicted it would**: the law is
per-damage-family (a bilinear form in the damage profiles), and this grid mixes
families. The scalar version's validated domain was always fixed-shape sweeps.
Two new facts from the grid:

- **Raw excess flips sign** for small-L17-damage cells (−0.07 to −0.11): with a
  2-dim L17 deletion, adding L16 damage *helps* relative to additivity — the
  output-end compensation machinery (§121) is strong enough to invert the
  interaction. The product law's positive-coupling regime requires both damages
  substantial.
- **The content-level constant is ~13, not 23** (registered (b) held): the
  historic c = 22.9 was roughly norm-doubled. Compression budgeting with the
  §law formula should use the content-level constant and expect ±families.

Also logged: raw d17 is non-monotone in span size (k=32 < k=8) — ranks 9–32 of
L17's output PCA include beneficial-to-delete directions. Band scan queued
(registered: (a) some band beyond rank 8 has negative content-level damage;
(b) the non-monotonicity localizes to that band).

## 124. No beneficial band at L17 — the wiggle was the generic-removal floor

File: `bilin18_l17_bands.py`. Registered (a) failed and the control violation is
the explanation: bands 9–32 (+0.019) and 33–128 (+0.010) cost the same as a
**random 24-dim span** (+0.019). At a writer as loud as L17, removing *any*
energy costs ~0.02 nats through the norm channel — a generic-removal floor — and
§123's non-monotonicity (0.018) sits inside it. No beneficial-to-delete
directions exist in L17's ranks 9–128; the content of L17's write is effectively
all in its top-8 (+0.508, 25× the floor). The loudest layer is also the most
concentrated: one band of eight directions, everything else indistinguishable
from ballast at the CE level.

## 125. The pattern census: attention is mostly content-based, with a positional hub at L2

File: `bilin18_pattern_census.py` (shuffled null clean at negative values). Both
registered bars failed short of their thresholds while the shape is clear:

    positional R² medians -- L0: 0.03 | L2: 0.26 | L5: 0.03 | L9: 0.09 | L13: 0.01 | L16: 0.02

Offset alone explains little of most heads' patterns at any depth. **L2 is the
positional hub** (median 0.26, one head at 0.76), a handful of positional
specialists sit elsewhere (L0 h3: 0.71, L9 h7: 0.49, L16 h8: 0.48), and L13 has
none above 0.26 (registered (c) failed there). The early-vs-late gradient exists
(0.14 vs 0.02 median) but under the registered 0.2 bar. bilin18's attention is
predominantly content-routing from the start — consistent with score-*product*
attention having no softmax pressure toward sharp positional templates. Queued:
the lexical complement — is the content part predictable from the key token's
identity (per-head mean score per key token, held out)? Registered: (a) lexical
R² > positional R² for the majority of heads at every layer (the model's lexical
theme extends to attention); (b) shuffled-key null ≤ 0.05.

## 126. Patterns are contextual: the census's triple-negative

File: `bilin18_pattern_lexical.py` (after a shape fix; shuffled null clean). The
lexical hypothesis fails everywhere: held-out key-token identity explains
essentially none of any head's pattern (medians −0.12 to +0.00; the negative
values are honest held-out overfit of per-token means), and offset+lexical
combined peaks at a median of 0.19 (L2). With §125:

- **Position**: weak except the L2 hub.
- **Token identity**: nothing, at any layer.
- **Therefore: contextual.** The patterns are irreducibly driven by the full
  query-key content interaction.

The contrast is the finding: the residual stream is embedding-dominated and the
bus is lexical-headed (§§101, 111–112), but **attention is the one component
family that is not lexical** — it is where context enters the computation. The
division of labor in one line: the stream remembers the token, attention reads
the context, the front MLPs do the nonlinear work, and the output end amplifies
and stabilizes.
