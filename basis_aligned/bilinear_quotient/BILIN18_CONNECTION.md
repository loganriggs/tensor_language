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
