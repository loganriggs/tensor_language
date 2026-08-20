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

## 127. Attention heads are matched filters: effective score rank 4.6 of 128

File: `bilin18_score_rank.py` — all three registered bars held, the program's
second clean sweep:

    median score eff-rank -- L0: 1.4 | L2: 3.5 | L5: 7.4 | L9: 4.5 | L13: 7.1 | L16: 4.9
    overall median 4.6 of 128 | isotropic-input null: 44.4 | factor gap 0.37

On the actual activation distribution, each of a head's two score factors is a
~5-dimensional bilinear form — the 128-dim head dimension is scaffolding, and the
compression comes from the *data* (the isotropic null is 10× higher), not the
weights alone. The two factors of a head have comparable rank (product attention
composes two similar filters). With §126: attention reads *context*, and it reads
it through a handful of data-aligned matched filters per factor. Caveat registered
in the follow-up: these operators ignore RoPE, which sits between projection and
dot product — the realized-pattern-rank run (queued) checks whether the
weights-level ranks survive it.

## 128. RoPE fans the filters out: realized rank 22.7, and the weights-level rank does not survive

File: `bilin18_pattern_rank.py`. The §127 caveat matures into a scope
qualification: realized pattern matrices run at median effective rank **22.7**
(registered ≤12 failed), and the pre-RoPE weights-level rank does **not** predict
realized rank across heads (Spearman −0.20; registered ≥0.5 failed). The
shuffled-weights null held (48.7, 2.1×) — patterns are still structured, just not
in a way the static bilinear operator captures.

Corrected statement of §127: **pre-rotation, each score factor is a ~5-dim
content filter; RoPE's position-dependent rotation fans those few filters into a
~20-30-dim family of realized patterns.** The matched-filter picture is true of
what the head *computes on content* and false of what the pattern *looks like*
over positions — rank is not preserved through position-dependent rotation, and
any pattern-level compression scheme must work post-RoPE. (This also explains
§125's L2 positional hub cleanly: positional structure emerges precisely where
the filters engage RoPE's frequency ladder hardest.)

## 129. The filters look shared — pending the covariance-matched null

File: `bilin18_filter_sharing.py`. All three skeptical registrations failed and
the named alternative fired: content-filter subspaces align at median principal
cos **0.71 within layers** and **0.41 across layers** (isotropic random floor
0.10). Read at face value this is a *shared attention lexicon* — and it would
complete a beautiful symmetry: heads watching the same few stream directions
while combining them into private quadratic forms is exactly the "dense shared
support, private functionals" structure the program found in MLP reading (§§49–58,
§84: shared support, per-head couplings).

**Held provisionally.** The C-metric concentrates any operator's singular vectors
toward the covariance's top directions, so isotropic random subspaces are the
wrong floor — the same envelope-style artifact the constituency arc caught in
§55. Queued: covariance-matched null (random subspaces drawn inside the C-metric
ball). Registered: (a) within-layer 0.71 exceeds the matched null by ≥ 0.2 (the
lexicon is real beyond concentration); alternative: a matched null near 0.6 would
attribute the sharing to covariance concentration and void the claim.

## 130. The attention lexicon is real, and it is a within-layer institution

File: `bilin18_filter_null.py`. The covariance-matched null lands at **0.35** —
half of §129's raw 0.71 was concentration, and the remaining **+0.36 is genuine
sharing** (registered ≥ +0.2 held). Cross-layer alignment (0.41) is almost
entirely concentration (+0.06 above the null): the lexicon does not extend across
layers. Final statement of the motif, now measured in both component families:

> **Shared watch-list, private combinations.** A layer's nine heads watch
> substantially the same few stream directions and combine them into per-head
> score forms (this section + §§125–127); the model's MLP readers watch the same
> L1 output coordinates and combine them into per-reader functionals (§§49–58).
> The MLP watch-list is a model-wide institution; the attention watch-list is a
> per-layer one.

Queued, the unifying question: does a layer's attention lexicon coincide with its
own MLP's input watch-list (the input-mode Gram's top directions)? Registered:
(a) alignment exceeds the covariance-matched null by ≥ 0.15 (one watch-list per
layer, shared across component types); alternative: two separate lexicons — also
an answer, and it would echo §84's cross-type disjointness.

## 131. Two watch-lists per layer: the component types divide the stream

File: `bilin18_one_watchlist.py`. The registered alternative fired: a layer's
attention lexicon and its MLP input watch-list align at 0.13–0.40 — at or below
the covariance-matched null (0.35), and at L9/L13 well below it (the two lists
*avoid* each other more than chance). (Caveat: the null was computed with L2's
covariance; per-layer nulls would shift slightly, not enough to flip sub-null
values into wins.) There is no "one watch-list per layer": **attention and the
MLP each maintain their own**, and cross-type separation is now measured three
independent ways — QK couplings reconstruct below chance from the MLP vocabulary
(§84), OV paths are indifferent to it (§86), and the input watch-lists are
disjoint (here). The component types divide the stream between them: same
carrier, separate channels, separate reading institutions.

## 132. Separate write channels too: the stream is multiplexed by component type

File: `bilin18_write_channels.py` — third clean sweep (5/5, instrument
discriminating): attention writes and MLP writes occupy separate stream
subspaces at every measured layer (median principal cos 0.08–0.25, at or below
the covariance-matched nulls of 0.20–0.26; four of five *below* null — mild
active avoidance). Together with §131 the component-type separation is complete
on both sides:

> **The residual stream is a multiplexed bus.** At every layer, attention and
> the MLP each keep their own input watch-list (§131) and their own output
> channel (here); the quadratic codes of the two types are mutually unreadable
> (§§84, 86). One carrier, two component types, and essentially no cross-talk in
> either direction — the "everything is diffuse" theme of this model holds
> *within* each component type, while *between* types the separation is clean.

## 133. The watch-lists are causally real but individually light — except one attention edge

File: `bilin18_watchlist_causal.py`. All three registered bars held (2/3, 2/3,
floor clean), but the effect sizes tell the fuller story: deleting a writer's
component within the next layer's institutional watch-lists costs little
(+0.002–0.005 — the within-type diffuseness again; no 8-dim channel carries much
alone), with one exception that is the section's finding: **L5's write into L6's
attention watch-list costs +0.030** — 50× the random floor and 11× the
MLP-watch-list cost at the same edge, locally inverting registered (a). The
L5→L6-attention edge is the only concentrated attention-mediated consumption
found anywhere in the tail; everywhere else, causal load spreads below the
resolution of any 8-dim institutional channel. A fitting close for the motif:
the institutions are real (geometry §§130–132, causality here), and inside them
the model remains what it has always been — diffuse.

## 134. The L5→L6 edge routes values, not patterns

Files: `bilin18_l5l6_edge.py` (v1 control broken — the no-op arm ran with the
deletion active; recorded), `bilin18_l5l6_edge2.py` (control exact at +0.0000).
With L6's patterns clamped to their clean values, deleting L5's watched span
still costs +0.0257 of the full +0.0297 — the tail's one concentrated attention
edge is **87% value-side**: L5's content rides through L6's attention as routed
values; only 13% of the effect comes from pattern steering (registered (a),
pattern-side ≥60%, failed decisively). An ironic close: the span was found via
the *score-filter* watch-list, but what makes it causally valuable is that the
same directions are what L6's attention picks up and moves. Watch-lists locate
the traffic; the traffic is cargo, not steering.

## 135. Gauge audit and defect survey (from balanced_gauge_spec.md)

The user pointed at `basis_aligned/balanced_gauge_spec.md`; the bilin18-relevant
part is the gauge-freedom audit and the defect survey (`bilin18_gauge_defect.py`).

**Audit of the program's published statistics: gauge-safe.** Everything headline
is built from the interaction tensor T (coupling matrices, functional vocabulary
and its eff-ranks/LORO, QK/OV comparisons), from activations (spans, PCA bases,
all CE work), or from the function (linearization fits). The input-mode Gram is
invariant by cancellation (checked algebraically), and the |DW| direct-write
predictor uses embeddings dotted with activation-space directions, not raw unit
norms. No result needs correction.

**Defect survey: bilin18 is NOT near-balanced.** Every layer's m-weighted mean
defect sits at **0.28–0.40** (registered near-balanced ≤0.15 failed; the ≥0.5
arbitrary-gauge flag not tripped; zero dead units; m_i sanity held). The three
factor norms of a typical unit disagree by roughly e^0.35 ≈ 1.4–1.9×, uniformly
across depth — consistent with training without (effective) weight decay. Two
consequences: (1) bilin18's raw per-matrix statistics (stable ranks of Left or
Right alone, unit rankings by a single norm, any Hessian/curvature reading) are
gauge-contaminated at the ~2× level and must be taken at the balanced point —
**standing rule adopted: any future per-matrix weight statistic calls the
balance step first**; (2) our luck was discipline, not chance — the program
happened to work at the T/activation/function level throughout, which the spec
identifies as the invariant classes.

## 136. Unit masses are maximally flat: 98–99% of units active at every layer

File: `bilin18_unit_mass.py` (both bars held). The gauge-invariant unit-mass
spectrum is as flat as it can be: effective active units 98–99% of 4608 at every
layer (depth ratio 1.02). No hidden unit anywhere carries outsized mass — the
strongest possible statement of the program's diffuseness theme, now at the
finest structural grain the architecture has (individual rank-1 terms of T).
bilin18 has no "important neurons" at the weight level, at all.

## 137. The attention profile: front-loaded, non-energetic, and L14's is net harmful

File: `bilin18_attention_profile.py` — the last unmeasured component/operator cell.

    L0 +0.242 | L1 +0.302 | L2 +0.205 | L3 +0.105 | L4 +0.178 | L5 +0.100 | L6 +0.073
    L7-L13: +0.04-0.07 (L10 -0.004) | L14 -0.036 | L15 -0.001 | L16 -0.012 | L17 +0.011

- **Front-loaded** (registered (c) held): context assembly is early — L1's
  attention is the single most important in the model (+0.302), and the front
  three dwarf everything after L5. The same front-loading as functional
  nonlinearity (§107): the model does its real work — nonlinear computation *and*
  context gathering — in the first few layers.
- **Not energetic** (registered (a) failed, Spearman 0.31): attention damage does
  not track write energy. L2–L4 write at enormous magnitude (10⁷–10⁸ energy —
  the ±4 value-mixing λs of §111's table live exactly there) yet cost only
  +0.1–0.2; the dilution law is an MLP-write phenomenon, not a general one.
- **The L6 cargo edge confirms independently** (registered (b) held, 20× its
  share expectation) — §§133–134's edge, found again by a different operator.
- **Four late attentions are net harmful or null** (L10, L14, L15, L16 ≤ 0), and
  **L14's entire attention output improves CE by 0.036 when deleted** — the
  largest deletion-benefit found anywhere in the program, at whole-component
  scale. Per §97's hard-won lesson this is a frozen-point property (the four
  constraint-release refutations pre-empt any spare-capacity reading), but the
  magnitude demands replication before further interpretation — queued
  (registered: sign holds on disjoint rows, magnitude ≥ 40%).

## 138. Three late attentions are reliably net harmful

File: `bilin18_l14_attention_replicate.py` — full replication on disjoint rows:
L14 −0.0348 (96% of the original −0.0363), L10 −0.0129, L16 −0.0127 (both
negative in both measurements), L15 ~0, and the L13 positive control intact
(+0.0315). A consistent sub-family: **the attention blocks of L10, L14, and L16
subtract value on held-out data at the frozen operating point** — the
truncation-as-regularization phenomenon at whole-component scale, an order of
magnitude larger than the span-level cases (§96). Per the four constraint-release
refutations, no spare-capacity claim follows; the open question is whether the
harm has the §98 signature (sharpening easy tokens, overshooting hard ones) —
queued (registered: (a) deletion benefit ≥60% concentrated in the base model's
hardest-token quartile; (b) easy quartile hurt or flat; (c) L13-deletion control
shows the opposite pattern — broad harm).

## 139. All late attention sharpens; L14 just overdoes it

File: `bilin18_l14_signature.py`. Registered (a) and (b) held strongly — L14's
deletion benefit is a redistribution even more extreme than §98's spans (hard
quartile relieved by −0.270 per token, easy hurt +0.022, hard-share 1.94). The
(c) "control violation" is the finding: **L13's attention — net helpful — shows
the same shape** (hard −0.003, easy +0.020 under deletion). Late attention
components uniformly sharpen easy predictions and press on hard ones; they
differ only in where the trade nets out. L14 is not a different kind of
component — it is the same kind, tuned past the break-even point on this
distribution.

Queued, the depth-functional dichotomy this implies: early components should
show the *opposite* deletion profile (hurting hard tokens most — they compute
content), late components the sharpening profile (easy hurt, hard relieved).
Registered: (a) all four late components tested (L13/L14 attention, L14/L16
MLP-spans) show easy+ / hard− under deletion; (b) both early controls (L2
attention, L2 MLP-span) show hard-mean > easy-mean under deletion (content
loss lands on hard tokens); (c) the dichotomy is monotone-ish: hard-minus-easy
delta decreases with depth across all six.

## 140. CORRECTION: the "sharpening" shape is generic damage, not component function

File: `bilin18_dichotomy.py`. All three registered bars failed, and the pattern
that failed them corrects §§98 and 139's interpretation:

    L2attn: easy +0.278 hard -0.124 | L2mlp8: +0.030/-0.012 | L13attn: +0.020/-0.003
    L14attn: +0.022/-0.270 | L14mlp8: +0.003/-0.003 | L16mlp8: easy +0.232 hard +0.357

The easy-hurt/hard-relieved deletion profile appears at **every depth**, early
components included — it is not a "late sharpening function." The likely
mechanism: damage flattens the output distribution; under a convex loss,
flattening *helps* wherever the model was confidently wrong (which populates the
hard quartile) and hurts where it was confidently right (easy). §98's random
control already showed the same signed shape at small magnitude, supporting the
generic reading. What *is* component-specific: (1) the net balance (L14's
attention is the one whose hard-relief outweighs everything — still real, still
replicated); (2) **hard-token harm**, which only L16's span shows (+0.357) — the
signature of true content whose loss no flattening can compensate.

> **Correction to §§98/139.** "The spans/late-attention sharpen easy tokens and
> overshoot hard ones" over-attributed a generic damage signature to component
> function. Corrected reading: deletion-benefit components are those whose
> content value is smaller than the generic flattening relief on this
> distribution; true content components announce themselves by hurting hard
> tokens. Propagated to the report (ledger #11). Deciding control queued:
> random spans at matched energy must reproduce the easy+/hard− shape
> (registered), else the generic-damage story fails too.

## 141. The generic-damage control: confirmed where it matters

File: `bilin18_generic_damage.py`. Registered (a) failed formally (2/3) and holds
substantively: the large random damage (256-dim span at L9) reproduces the
easy+/hard− shape decisively (easy +0.011, hard **−0.044** — from a *random*
span), L13's random-96 does too (+0.003/−0.006), and only the smallest damage
(L5 random-96) is too weak to flatten anything (both quartiles +0.003–0.006).
The §140 correction stands with a magnitude qualifier: **at sufficient damage,
flattening relief on confidently-wrong tokens is generic**; small damages sit
below the effect. The content criterion survives: a component certifies real
content by *hurting* hard tokens under deletion — flattening cannot fake that
direction. Queued: re-score the tail profile by the content criterion
(hard-quartile mean under span deletion); registered: (a) L16's span is the only
one with hard-mean ≥ +0.05; (b) the content ranking reorders the net-CE ranking
(Kendall tau ≤ 0.7).

## 142. Content scores, first pass: the gain channel can fake content too

File: `bilin18_content_scores.py`. Both bars failed, and the failure teaches the
criterion its final lesson. The per-layer hard-quartile deltas under span
deletion (free final norm):

    L5 +0.086 | L6 -0.050 | L7 +0.028 | L8 -0.047 | L9 -0.128 | L10 +0.002
    L11 +0.016 | L12 -0.033 | L13 -0.007 | L14 -0.003 | L15 -0.092 | L16 +0.357 | L17 +1.674

- **L17's +1.67 is mostly the gain channel running in reverse**: deleting a
  huge-energy span shrinks the final RMS, the surviving logits get *amplified*,
  and confident errors get much worse — the mirror image of the flattening
  relief. Removing a loud span doesn't flatten; it sharpens. So hard-token harm
  certifies content **only under a frozen gain** — the §141 criterion inherits
  the §117 instrument requirement.
- The regularizer spans re-announce themselves (L9 −0.128, L15 −0.092 hard
  relief), and L5/L7/L11 emerge as moderate hard-token-content spans alongside
  L16 (+0.357).

Queued, the criterion done right: gain-frozen content scores. Registered:
(a) L17's hard-mean drops ≥70% when the gain is frozen (it was the channel);
(b) L16 stays ≥ +0.2 (true content at both levels); (c) L5's +0.086 persists
at ≥50% (genuine front-of-tail content).

## 143. The difficulty-decomposition arc closes: no tail span carries hard-token content

File: `bilin18_content_scores2.py`. Registered (a) held spectacularly — L17's
hard-token harm collapses from +1.674 to **−0.058** under the frozen gain,
certifying it as pure gain-channel amplification. But (b) and (c) failed, and
that ends the criterion: with the gain controlled, **every** tail span's deletion
relieves hard tokens and hurts easy ones — including L16 (hard +0.023, easy
+0.154). L16's genuine content value (§120's +0.148 net) lives on *easy* tokens:
its span keeps the model confident where it is right. The arc's conclusion,
stated honestly: token-difficulty decompositions cannot cleanly separate content
from generic damage in this model — flattening dominates the hard quartile and
easy-harm is itself generic — and at the 8-dim-span grain no tail span carries
concentrated hard-token-critical content. What stands from the whole chain
(§§137–143): the attention profile and its front-loading, the replicated
late-attention negatives, the L6 cargo edge, the gain channel's two faces
(masking interactions, §117; faking content, §142), and L16 as the largest net
content span with its value on the easy side.

## 144. Dilution law: ratio side replicates exactly (effects side queued)

File: `bilin18_dilution_replicate.py`. With fresh statistics rows the
write-to-stream ratios are nearly identical (0.253 → 0.043, zero inversions) and
the law holds at Spearman +0.79. Caveat stated plainly: this script re-measures
only the *ratio* side — the transplant effects were read from the original run's
cache. The effects-side replication (fresh base/source rows for all ten
transplants) is queued; registered: (a) new effects correlate with old at ≥ 0.8;
(b) the law holds jointly on the all-fresh pair at ≥ 0.7.

## 145. Dilution law fully replicated

File: `bilin18_effects_replicate.py`. Fresh base/source rows reproduce all ten
transplant effects (Spearman +0.89 vs original), and the all-fresh law —
new ratios against new effects — holds at **+0.83**, slightly better than the
original +0.79. The tail-routing headline (§93) is now replicated end-to-end on
independent data: writes unaimed, edge strength = the writer's share of the
stream. Queued: a combined replication sweep of three remaining headline
numbers on fresh rows — score-rank ~4.6 (registered: fresh median within ±2),
within-vs-matched-null watch-list gap +0.36 (registered: ≥ +0.25), and L1's
linearization cost +0.28 (registered: within ±30%).

## 146. Headline sweep: three for three

File: `bilin18_headline_sweep.py`, all registered bars held on fresh rows:

- Score-rank median **4.3** (original 4.6) — heads are matched filters, robust.
- Watch-list gap at L9: within 0.73 vs matched null 0.26 = **+0.48** (the pooled
  original was +0.36) — the attention lexicon is, if anything, stronger.
- L1 linearization cost **+0.289** (original +0.282) — the crown stands.

With §§144–145, every load-bearing headline of the recent arcs is replicated on
independent data. Queued: the one striking older claim never independently
replicated — §84's below-chance QK reconstruction (−0.26 vs random +0.37).
Registered: fresh-rows QK median ≤ 0.0, random control ≥ +0.25, gap ≥ 0.3.

## 147. The QK disjointness replicates with an identical gap

File: `bilin18_qk_replicate.py` (log prints carry the old labels; the JSON bars
are the registered ones, all held). On fresh rows: QK couplings reconstruct from
the MLP basis at median R² **−0.08**, random matrices at **+0.55** — a
below-chance gap of 0.63, *identical* to the original's (−0.26 vs +0.37). The
cross-type code disjointness is a stable property of the model, not of the rows.
Replication status across the program's recent arcs: dilution (both sides),
score-rank, watch-list gap, L1 linearization cost, L14 attention negativity,
L9/L15 regularizer spans, the L9×L16 interaction, and QK disjointness — all
replicated on independent data.

## 148. The matched filters are trained

File: `bilin18_scorerank_null.py`, registered bar held: row-shuffled weights give
median score-rank **18.7** and gaussian weights **36.1**, against the trained
**4.3**. Combined with §127's isotropic-input null (44.4), the origin bracket is
complete: the data's covariance concentrates any weights' score functions
somewhat (44 → 19–36), and **training compresses a further 4–8×** down to ~5
matched filters per factor. The attention heads' selectivity is learned
structure, like the functional vocabulary's compression (§62) and unlike the
density/orthogonality that came free (§§60-era origin arc). Queued: the same
origin question for the shared watch-list (§130) — is the within-layer filter
*alignment* trained, or do shuffled weights' filters also cluster? Registered:
shuffled within-layer alignment ≤ matched-null + 0.1 (the lexicon is trained);
alternative: generic clustering would scope-note §130.

## 149. The lexicon is one-third trained — a scope note for §130

File: `bilin18_lexicon_null.py`. Registered (a) failed at 0.47 vs the null+0.1
bar, and the number decomposes the claim: within-layer filter alignment is

    0.71 (trained) = 0.36 (covariance concentration) + 0.11 (generic clustering
    of shuffled weights' filters) + 0.24 (trained sharing)

Shuffled weights' filters cluster somewhat above the matched null — random-ish
projections through shared marginal statistics land closer together than clean
random subspaces do — so §130's "+0.36 genuine sharing" overstated training's
role. Corrected attribution: **training contributes +0.24 of alignment**, about
a third of the raw number and two-thirds of the above-null part. The lexicon is
real and partly trained, sitting exactly in the pattern the origin arc found for
the MLP vocabulary: geometry comes cheap, sharing is what training adds. Queued:
the origin of the *separation* (§§131–132, two watch-lists and two write
channels per layer) — registered skeptical: shuffled weights separate just as
well (separation is the default state; training never had to fight it);
alternative: trained avoidance would be new.

## 150. Separation is free; the attention origin accounting closes

File: `bilin18_separation_null.py` (after a scope fix). Registered skeptical (a)
held 5/5, emphatically: shuffled-weights watch-lists separate at cos 0.06–0.10 —
*below* both the matched null (0.35) and the trained values (0.13–0.40). The
multiplexed bus's type separation is the default state of any two reading
mechanisms; **training did not create it, and in fact added the small overlap
that exists**. Origin accounting of the attention structures, complete:

| structure | trained share |
|---|---|
| matched filters (score-rank 4.3 vs shuffled 18.7) | strongly trained (4–8×) |
| shared lexicon (0.71 = 0.36 conc. + 0.11 generic + 0.24 trained) | one-third |
| type separation (shuffled 0.07 vs trained 0.13–0.40) | free — training *reduced* it |

The same moral as the §§60–62 origin arc, now for attention: geometry and
separation come free; what training buys is selectivity (the filters) and
sharing (the lexicon).

## 151. Front attention replicates; the profile is certified

File: `bilin18_attnfront_replicate.py`, all bars held: L0 +0.232, L1 +0.358,
L2 +0.229 on fresh rows (originals 0.242/0.302/0.205; ordering L1 > L0 > L2
preserved; all within 30%). The attention profile's positive headline — L1's
attention as the model's single most important — is certified alongside its
negative ones (§138). Queued: the last unreplicated headline, §109's spectral
split of L1 (interface 12% of linearization cost at 59% of energy; complement
53%) on fresh eval rows. Registered: (a) span-share stays ≤ 20%; (b)
complement-share ≥ 40%; (c) full cost within ±30% of +0.282.

## 152. The replication campaign closes: every headline stands

File: `bilin18_split_replicate.py` — the spectral split replicates precisely
(span share 12%, identical to the original; complement 58%; full +0.289, within
3% of +0.282). Campaign summary — replicated on independent data: dilution law
(both sides), transplant effects, score-rank, watch-list gap, L1 linearization,
spectral split, front attention profile, late-attention negatives, regularizer
spans, L9×L16 interaction, QK disjointness. Every load-bearing number in the
program has now survived a fresh-data test; nothing required a correction at
replication (the corrections all came from *conceptual* re-examination — nulls,
instruments, decompositions — never from fragile numbers).

Queued: the one empty cell in the layer maps — **L0's MLP** (every linearity and
linearization sweep started at L1 or later, yet L0 is the "true reset" writing
at RMS 1436 into a stream of 6). Registered: (a) linearization cost ≥ 0.15
(front-loading extends to the front-most layer); (b) its write's linear R² ≤
0.75 (real quadratic computation on raw embeddings).

## 153. The L0 cell — and an accidental exact replication

File: `bilin18_l0_cell.py`, all bars held. L0's write is 74% linearly
predictable and costs **+0.1767** to linearize — agreeing with the §107 front
map's +0.176 to three decimal places on different eval rows (the map had
measured L0's cost; the §101 linearity sweep was the actually-empty cell, now
filled at R² 0.742). The functional-nonlinearity map is complete end to end:

    L0 +0.18 | L1 +0.28 (peak) | L2 +0.11 | L3 +0.09 | L4 +0.05 | L5-L15 ~0.03 | L16 +0.03 | L17 +0.10

Program state at §153: every layer mapped by both instruments, every component
family profiled, every headline replicated, every origin question answered, and
eleven corrections on the ledger — all caught by the program's own controls.

## 154. UNIVERSALITY: all three laws hold in bilin12

File: `bilin12_universality.py` (v1 failed on an architecture difference worth
recording: bilin12 uses a *single squared score* (q·k)² where bilin18 uses a
two-score product; v2 uses module-level attention). All three registered bars
held on the sibling checkpoint (12 layers, 6 heads, 768 dims):

- **Dilution**: tail write-to-stream ratios decline monotonically (1 inversion;
  bilin18 had 0). Notably the *front* ratios exceed 1 (L3's write is 8.3× the
  stream it joins) — the front-writes-loud motif is even stronger at small scale.
- **Front-loaded functional nonlinearity**: costs L0 +0.265, **L1 +0.317
  (peak)**, L2 +0.277, vs mid +0.10–0.14 — same shape, same L1 crown, though the
  front/mid ratio is 2.6× (bilin18: ~9×) — the smaller model keeps relatively
  more functional nonlinearity in its middle.
- **Matched filters**: score-rank median 4.8 (of 128) vs shuffled 16.3 — nearly
  identical numbers to bilin18's 4.3/18.7.
- The λ-table motif repeats: embedding re-injection at weight ~4–8 everywhere.

The program's structural laws are **family properties of bilinear transformers**,
not bilin18 idiosyncrasies.

## 155. The replacement ladder: the mid-tail needs nothing, the end needs rank-8, the front needs everything

File: `bilin18_replacement_ladder.py`. Registered (a) and (b) held; (c) failed in
the direction that answers the user's question at its strongest:

| stand-in | L1 | L9 | L16 |
|---|---|---|---|
| constant mean (rank 0) | +6.76 | **+0.031** | +0.241 |
| rank-1 linear | +6.25 | +0.023 | +0.137 |
| rank-8 linear | +4.25 | +0.039 | +0.059 |
| rank-64 linear | +1.11 | +0.040 | +0.036 |
| full linear | +0.290 | +0.039 | +0.029 |

- **L9 needs literally nothing**: replacing its MLP with its *constant average
  output* costs +0.031 — the same as the full linear map. For the mid-tail the
  user's "just keep the scale" reading is not just right but understated: even
  the input-dependence of the linear map buys nothing.
- **L16 needs a rank-8 map** (+0.059) — a genuinely low-dimensional but
  input-dependent stand-in.
- **L1 needs everything**: rank-64 still costs +1.11, and the full 1152-rank
  linear map still loses +0.29 to the real quadratic. The front is the model.

This table is the reference fidelity-vs-complexity Pareto for the benchmark
design (see RESULTS note): three regimes — constant, low-rank linear, full
quadratic — and the program's maps say exactly where each applies.

## 156. CORRECTION: not two channels — a relay. Attention reads what MLPs write

File: `bilin18_channel_test.py` (prompted by the user's "what's reading what?").
Registered (a) failed 0/3, decisively: at every reader layer tested, the
attention watch-list aligns far more with upstream **MLP** writes (0.62–0.68)
than with upstream attention writes (0.20–0.43). (b) held 3/3 but weakly
(MLP-watch vs MLP-write 0.17–0.23, barely above the 0.16–0.22 null).

> **Correction to §132's "multiplexed bus."** The two component types are
> separated *locally* (different read apertures, different write directions at
> each layer) but the end-to-end flow is **cross-type**: attention's read
> directions are precisely where upstream MLPs deposit. With §134 (attention
> moves content as values/cargo), the corrected picture is a **relay**:
> MLPs write → attention reads MLP-written content and transports it across
> positions → MLPs consume. There is no attention-to-attention channel.
> Ledger #12; report updated.

bilin12 identity check (user request): base CE 4.23 vs bilin18's 4.00 on the
same rows — consistent with a half-depth, two-thirds-width sibling; config
confirms `squared_attn: True`. Right model.

## 157. The benchmark's first Pareto points

File: `bilin18_reference_submission.py`. Both registered bars failed, and the
numbers are the point:

- Only **4** mid-tail layers qualify for constant stand-ins at the 0.05 bar
  (L8, L9, L14, L15 at +0.031–0.048 alone); the §155 sample (L9) was the
  cheapest, not typical — mid-tail rank-0 costs run +0.03–0.10.
- The knowledge assignment (4 constants + 8 rank-8 maps, layers 5–16) costs
  **+2.68 nats jointly at 0.15M stand-in params**; all-full-linear on the same
  layers costs **+1.26 at 15.9M params**. First two points of the
  fidelity-complexity Pareto: 1% of the parameters buys you 2.1× the damage.
- Clean-instrument note: the naive joint of full-linear 5–16 is +1.26 — far
  below §104's contaminated +5.49 and below even the sequential-refit +1.56
  (which included L17). With the λ-mixing bug gone, composition drift is modest;
  §104's superadditivity was mostly instrument.

The benchmark now has a measured reference curve: (0.15M, +2.68),
(15.9M, +1.26), plus §155's per-layer ladder. BENCHMARK.md updated.

## 158. Reference v2: sequential refit moves the frontier

File: `bilin18_reference_v2.py`, both bars held. The same architecture as v1's
assignment (4 constants + 8 low-rank maps), but each stand-in fit on the model
with upstream stand-ins already installed and ranks raised to 16:

    Pareto so far: (0.15M, +2.68 naive r8) -> (0.29M, +1.66 REFIT r16) -> (15.9M, +1.26 full linear)

Sequential refit is the frontier lever: it bought 36% of the cost (2.61 → 1.66
at the same architecture) for free — the stand-ins absorb each other's errors
when fit in sequence. At 1.8% of the full-linear parameter budget the refit
assignment is within 0.4 nats of it. BENCHMARK.md's reference table updated;
the frontier-tracing rank sweep is queued (registered: (a) refit-rank-64 joint
≤ +1.40; (b) the refit curve dominates the naive curve at every rank tested).

## 159. The Pareto frontier is flat: rank barely matters

File: `bilin18_pareto_curve.py`. (b) and (c) held — refit dominates naive by
~1.0 nat at every rank, both monotone — and (a)'s failure is the shape of the
curve:

    refit: rank 4 = +1.81 (0.07M) | rank 16 = +1.66 (0.29M) | rank 64 = +1.54 (1.18M)
    floor: full linear = +1.26 (15.9M)

Seventeen times the parameters buys 0.27 nats. The mid-tail's replaceable
function is captured by a *handful* of refit directions plus four constants;
everything beyond rank ~4 chases a residual that even the full 1152-rank linear
maps only reduce to +1.26. Benchmark lesson one: the competitive region is below
0.1M params, and closing the last ~1.3 nats requires a different computation
class, not more rank. Queued, the ladder's upper rung: does a compact *quadratic*
stand-in help where linear saturates? Registered skeptical per §113 (the
residues are diffuse in every natural basis): a rank-32-input quadratic stand-in
for L16 beats the rank-32 linear by < 0.05 nats; alternative would reopen the
compressed-quadratic door at the stand-in level.

## 160. The quadratic rung is dead; the reference ladder is final

File: `bilin18_quadratic_rung.py`, both skeptical bars held. A 528-feature
quadratic correction on the top-32 refit directions buys **nothing**: −0.009 at
L16 (held-out overfit), ±0.000 at L9. §113's basis-diffuseness verdict holds at
the stand-in level: between low-rank linear and the full quadratic component
there is no useful intermediate class in any compact basis. The benchmark's
reference ladder is final:

    constant mean  ->  low-rank REFIT linear (rank ~4-16; the flat frontier)
    ->  [compact quadratic: measured dead]  ->  full component

With §§155–159 this closes the reference-instrument arc: the benchmark has a
measured Pareto (0.07M/+1.81 → 15.9M/+1.26), a known frontier lever (sequential
refit), a known dead rung, and the §104-softening note (composition drift modest
on clean instruments). BENCHMARK.md finalized accordingly.

## 161. The fingerprint dataset: distinguishable, deterministic, and context-specific

File: `bilin18_fingerprints.py`; dataset saved to `bilin18_fingerprints.pt`
(12 components × per-token CE deltas on rows 384–448, with base losses and
manifest). Registered (b) held decisively — median pairwise Spearman between
components' fingerprints is **0.04**: the twelve are nearly orthogonal, so
component-specific causal scoring (Track 1) is well-posed. Registered (a)
failed because the bar was ill-posed, recorded plainly: the deltas are
deterministic (the model has no sampling noise), so "split-half stability" has
no noise to test; the even/odd-position proxy instead measured whether
*adjacent tokens* share fingerprint structure — they do not (0.06), which is
the program's context-specificity theme at the finest grain: what a component's
ablation does to a token depends on that token's context, not its neighbors'.
Benchmark consequence, added to the spec: Track-1 scoring runs on a *fixed*
held-out set where fingerprints are exact; generalization across sets is the
explanation's burden, not the dataset's.

## 162. The confound floor is low: Track-1 is well-posed end to end

File: `bilin18_fingerprint_floor.py`. Registered (a) failed in the favorable
direction: the base-loss floor is only **0.13** median |Spearman| (bar 0.2) —
the flattening confound that invalidated aggregate difficulty-quartile
statistics (§§140–142) barely structures per-token *ranks*. Position contributes
nothing (0.01), and base-loss-residualized fingerprints stay near-orthogonal
(0.06). Track-1 scoring rules, now fully grounded: score = Spearman between the
explanation-predicted and measured fingerprint on the fixed held-out set;
publish the 0.13 base-loss floor beside every score; anything above ~0.2 is
real signal; component identity is never in doubt (pairwise 0.04). The
benchmark's both tracks now have measured reference instruments, floors, and
known traps. BENCHMARK.md updated.

## 163. Cross-model analog transfer: the two models place function on the same text

File: `bilin12_fingerprints.py`, all three bars held; dataset saved
(`bilin12_fingerprints.pt`). The bilin12 fingerprints are distinguishable
(pairwise 0.12) with a low floor (0.11) — and the registered cross-model
question landed decisively: **analogous components' fingerprints correlate at
0.34 across models, versus 0.05 for non-analog pairs.** The two
independently-trained bilinear transformers, ablated at their analogous
components (front attention, mid MLP), lose accuracy on the *same held-out
tokens*. Function placement is convergent across the family — universality
(§154) extended from structural laws to token-level causal responsibility. For
the benchmark this is the ideal transfer split: an explanation of bilin18's
front attention should partially predict bilin12's fingerprint (expected ~0.34
ceiling), and beating the 0.05 non-analog floor across models is the
generalization test Track 1 needed.

## 164. Correspondence follows depth fraction

File: `bilin18_depth_scaling.py`, all three bars held:

    attn6->attn9 (relative) 0.368 > attn6->attn6 (absolute) 0.344
    mlp5->mlp7  (relative) 0.227 > mlp5->mlp5  (absolute) 0.171
    front pairs: attn1 0.388, attn2 0.417 (schemes coincide there)

The two models place token-level causal responsibility at **proportional depths**:
bilin12's mid-stack components match bilin18's mid-stack, not its same-numbered
layers. Margins are modest but consistent, and everything sits far above the
0.05 non-analog floor. Queued, the definitive form: the full correspondence
matrix — bilin18 attention fingerprints at every other layer (1–17) against
bilin12's three attention fingerprints, tracing each component's best-matching
depth. Registered: (a) each bilin12 component's best-matching bilin18 depth
fraction is within ±0.15 of its own fraction; (b) the correspondence curve is
unimodal around the match for ≥2 of 3 components.

## 165. The correspondence matrix: exact depth-fraction matching

File: `bilin18_correspondence_matrix.py` (after a stale-block fix). Registered
(a) held **3/3 with exact matches**:

    bilin12 attn1 (fraction 0.08) -> best bilin18 match L1 (0.06)
    bilin12 attn2 (fraction 0.17) -> best bilin18 match L3 (0.17)
    bilin12 attn6 (fraction 0.50) -> best bilin18 match L9 (0.50)

The correspondence curves peak at the proportional depth and collapse past
bilin18 fraction ~0.65 (all three bilin12 components correlate near zero with
bilin18's late attention — consistent with late attention being calibration
machinery in both models, matched by nothing in the other's front/mid).
Registered (b) failed only on the strict 0.03 wiggle tolerance; the curves are
broadly single-peaked. Statement for the record: **two independently trained
bilinear transformers of different sizes assign token-level causal
responsibility to attention components at the same fractional depths** — the
strongest universality result in the program, obtained entirely from ablation
fingerprints on shared text.

## 166. Six for six: the depth-fraction correspondence is complete

File: `bilin18_mlp_correspondence.py`, both bars held. The MLP side matches like
the attention side did:

    mlp1 (0.08) -> L1 (0.06) | mlp5 (0.42) -> L7 (0.39) | mlp8 (0.67) -> L11 (0.61)

With §165's attention triple, **every one of six components tested — both
component types, two independently trained models of different sizes —
best-matches the other model at its own fractional depth**, with correspondence
curves single-peaked around the match. Two bilinear transformers trained
separately on the same distribution develop causally interchangeable depth
programs: what fraction-x-of-the-stack does to a given token in one model, the
fraction-x components do in the other. This is the program's capstone
universality result, and the benchmark's cross-model split now has calibrated
expectations at every depth.

## 167. Third model: correspondence crosses the MLP family at the front; mid-depth scaling wavers

File: `sqrd12_correspondence.py`. Identity held (base CE 4.011 — better than
bilin12's 4.225, sensible for conventional MLPs). Registered (b) failed 2/3 by
a whisker: attn1 → fraction 0.06 (own 0.08, hit), attn2 → 0.11 (own 0.17, hit),
but attn6 best-matches bilin18's **attn6** (fraction 0.33) rather than the
fractional attn9 (0.50) — 0.02 beyond the ±0.15 bar. All matches sit 3×+ above
the non-analog floor and land on the same-named component (c held).

Reading: token-level causal correspondence **crosses the MLP-architecture
family** — a conventional-MLP model's attention fingerprints still match the
bilinear model's, strongly at the front — but the *fractional-depth scaling* of
the mid-stack may be a bilinear-family property; for sqrd12 the mid component
pairs at absolute index instead (or the curve is flat between the two — the
margin was not resolved). A fitting final note for the wake: the front of these
models is universal across all three checkpoints tested; the middle is where
families begin to differ.

## 168. Tie-break: depth-fraction scaling is a within-family law

File: `sqrd12_tiebreak.py`, both bars held. The curves resolve §167 cleanly:

    sqrd12-attn6 : L5 0.203 | L6 0.229 | L7 0.259 | L9 0.169 | L11 0.179  (peak L7)
    bilin12-attn6: L5 0.316 | L6 0.344 | L7 0.301 | L9 0.368 | L11 0.285  (peak L9)

The bilinear sibling scales **fractionally** (0.50 → L9); the conventional-MLP
model's mid-attention corresponds to bilin18's fraction ~0.36–0.39 — genuinely
front-shifted, not a flat tie. Final form of the correspondence law: **within
the bilinear family, models assign token-level causal responsibility at equal
depth fractions (six for six, exact); across MLP families the correspondence
survives (same components, 3×+ floor) but the depth mapping warps** — a
conventional-MLP model's middle does what the bilinear model does earlier in
its stack. The bilinear MLPs evidently stretch the early program deeper.

## 169. Fourth model: front-loading universal, dilution scope-noted, correspondence warps by family not depth

File: `swiglu18_test.py` (swiglu-gated MLPs, same 18-layer depth as bilin18;
base CE 4.17, identity fine). Three findings:

- **Front-loaded linearization holds at every checkpoint tested** (peak L1
  +0.213 vs mid +0.053) — now bilin18, bilin12, swiglu18. A family-wide law.
- **Dilution monotonicity FAILED (5 inversions) — and the failure scope-notes
  §93 rather than contradicting it.** swiglu18's ratios decline through the
  mid-tail then *rise* at L15–16 — and checking bilin18's own norm table
  (§112), its ratios would rise too past edge 14→15 (L16's write is RMS 938
  into a 1567 stream). §93's "monotone decline" was measured on edges 5→6
  through 14→15 and is correct *in that range*; the general law is: **decline
  through the mid-tail, rise again at the output-preparation end** — in every
  model. Also logged: swiglu18's λ₁ turns *negative* at L16/17 (−2.0) — it
  subtracts the embedding at the end.
- **Correspondence warps by family even at equal depth**: front hits exactly
  (attn1→attn1 0.37, attn2→attn2 0.39) but swiglu18's attn6 best-matches
  bilin18's attn2, and its attn9 matches attn6 — the gated-MLP models
  (swiglu18, sqrd12) are consistently *front-shifted* relative to pure-bilinear
  bilin18, independent of depth ratio. The §168 law finalizes: front
  correspondence is universal; mid-stack correspondence is exact within the
  pure-bilinear family and front-warped across MLP families.

## 170. Interface ladder graded; edges get types

**Interface ladder** (`bilin18_interface_ladder.py`, L5→L6 cargo edge): the
effect grows gradually — k=2 carries 26%, k=8 carries 72%, k=32 gives +0.0415
(random controls at floor throughout). Registered (a)/(c) failed at their exact
bars; the honest shape is *small-interface-dominant with a fat tail*: most of
the edge flows through ~8 dims, the rest is spread.

**Typed edges** (user design point, folded into BENCHMARK.md): edge complexity
must be the description length of the dependency's *functional form*, not its
wire count. A dense connection that only transmits a norm is simple. The
program has measured all three edge types this model actually uses:
- **summary-typed** (dense wires, one variable): L17's span dependence is 82%
  "energy → final gain" (§116); the dilution law says generic tail edges are
  share-of-stream typed (§93).
- **coordinate-typed** (few specific dims): the L5→L6 cargo edge (~8 dims
  dominant, this section).
- **opaque/full**: the front (L0–L2), where nothing simple substitutes.
The edge algebra for submissions: {k coordinates | summary statistic (norm,
mean) over a declared set | low-rank map | opaque}, priced by description
length. This is the user's "semantics to an extent": an edge label like "reads
the stream's energy" is a one-variable mini-explanation, and typed edges give
the compositional, cleanly-separated structure of good code — nodes as
functions, edges as typed arguments.

## 171. The gain channel has a sign: it amplifies MLP-span damage and cushions attention damage

File: `bilin18_attn_norm_share.py`. All attention norm-shares came out
**negative** (L1 −21%, L2 −16%, L6 −32%, L14 −22%): gain-freezing makes
attention ablations *worse*, the mirror image of the MLP-span result (§117:
L17 82% positive). Reading: deleting an MLP span removes specific loud
directions — the rms rescale then over-amplifies the distorted survivors
(damage adds); mean-ablating a whole attention output shrinks the residual
uniformly — the rescale restores the scale of the remaining, still largely
correct content (damage is cushioned, by ~20–30%). Consequences:

- **Attention edges are content-typed at every depth** — registered (a)
  (front norm-typed) refuted; even L1's +0.32 is structure, not energy.
- The §137 attention profile's raw numbers were, if anything, *understated* by
  the gain compensation.
- For the benchmark's typed-edge audits: the gain-frozen instrument must be run
  in both directions and its **sign reported** — a positive share says the edge
  was energy; a negative share says the component was content whose loss the
  norm partially papered over.

## 172. Keep-only: interfaces are real (14×) and partial (28%)

File: `bilin18_keep_only.py`. Deleting all of L5's write costs +0.069; keeping
only the 8 watched dims reduces that to +0.050 — the interface preserves **28%**
of the layer's downstream value, against 2% for a random 8 (registered (b)
held; (a)'s 40% bar failed; (c)'s 0.08 sanity missed on row variation, 0.069 vs
the other row-set's 0.095). The typed-edge audit's two directions now agree on
the L5 interface: cutting the watched channel removes a concentrated ~0.03
(§170), keeping only it preserves a concentrated ~0.02 of 0.07 — a real,
14×-enriched, minority channel. The program's oldest lesson holds at the
newest instrument: **structure is real and never total** — every concentrated
channel found in this model carries a well-measured minority of the function,
and the majority stays diffuse.

## 173. The full atlas: 36 fingerprints, depth-smooth, type-marked

File: `bilin18_fingerprint_atlas.py`; asset saved
(`bilin18_fingerprint_atlas.pt`, every layer's MLP span and attention). All
three bars held:

- **Distinguishable at scale**: median pairwise |Spearman| 0.07 across 36
  components.
- **Depth-smooth, 35/36**: nearly every component's most-similar same-type
  fingerprint is an adjacent layer's (±2). Token-level causal responsibility
  varies smoothly along the stack — the structural fact beneath the
  single-peaked correspondence curves of §§165–168.
- **Type-marked**: within-type similarity (0.09) exceeds cross-type (0.06) —
  the relay's two stages leave systematically different marks on tokens.

The atlas completes the benchmark's ground-truth layer: every component of
bilin18 now has an exact causal fingerprint, the sibling models have theirs,
and the whole set is mutually distinguishable, smooth in depth, and typed.

## 174. Seriation: the stack's order is written in its causal marks

File: `bilin18_seriation.py`, all bars held. Spectral ordering of the
fingerprint similarity graph recovers the true layer order at |ρ| **0.91**
(MLP components) and **0.90** (attention), with shuffled-token nulls at
0.23–0.28. Combined with depth-smoothness (§173) and the cross-model
correspondence (§§163–168), the closing picture of the atlas arc: **a model's
per-token ablation deltas, unlabeled, encode its architecture's depth order,
vary smoothly along it, mark component type, and align across independently
trained models at family-lawful depths.** The causal fingerprint is not noise
around a net number — it is a structured coordinate system for the model's
computation, and the benchmark's ground-truth layer inherits all of that
structure.

## 175. The sibling atlas: same structure, second model

File: `bilin12_atlas.py`; asset `bilin12_atlas.pt` (24 components, every layer,
both types). All three atlas bars replicate: distinguishable (pairwise 0.09),
depth-smooth (22/24), type-marked (0.13 within vs 0.08 cross). The fingerprint
atlas's structural character — near-orthogonal, smooth along depth, marked by
component type — is now a property of the family, not one model. Assets in
place for next: joint cross-model seriation (do the two stacks embed into one
ordered causal coordinate system, with the family-lawful depth warp visible as
the embedding's alignment?).

## 176. One causal depth coordinate for the family

File: `bilin_joint_seriation.py`. The registered headline held: embedding all 30
MLP fingerprints from both models in a single spectral ordering yields a
coordinate that tracks **depth fraction at |ρ| = 0.85** across models (shuffled
null 0.30). Within-model orders inside the joint embedding: bilin18 0.85,
bilin12 0.74 (the (a) bar of 0.8 missed narrowly on the smaller model — the
joint axis costs it a little fine order). The fingerprint arc's capstone
statement: **the bilinear family shares one causal depth coordinate.** Unlabeled
per-token ablation deltas from two different models, pooled, self-organize onto
a single axis — and that axis is fractional depth. Everything the program
measured layer-by-layer (front-loading, dilution-then-amplification, the relay,
the correspondence law) lives along this one recoverable coordinate.

## 177. The axis transcends the family; the warp is attention-specific

File: `sqrd12_join_axis.py`. Registered (a) and (c) held, (b) failed
informatively: the three-model joint embedding still tracks depth fraction at
**0.79** (null 0.04) — the causal depth coordinate spans even the cross-family
model — but sqrd12's MLP components show **no front-shift** (3/12), sitting at
their nominal fractions. Since §169's front-shift was measured on *attention*
fingerprints, the refinement is clean: **MLP causal placement follows depth
fraction universally, across all three architectures tested; the cross-family
warp lives in the attention stream.** A satisfying division: the component type
that is per-layer, contextual, and transport-typed (attention) is also where
architectural families express their differences; the component type that
carries the shared vocabulary (MLPs) places its work identically everywhere.
The fingerprint arc ends here: one axis, three models, and the family
differences localized to a single component type.

## 178. CORRECTION: the attention warp's direction is instrument-dependent

File: `attn_warp_confirm.py`. Registered (b) failed by *inversion*: on the
joint-axis instrument, sqrd12's attention components sit at **higher**
axis-implied fractions than nominal (−0.03/−0.17/−0.11), while §169's direct
best-match instrument placed them **lower** (attn6 → bilin18 L6–7). The axis
itself is sound (bilinear pair at 0.88, null 0.04).

> **Correction to §§169/177.** What is established: sqrd12's attention
> components do not sit at their nominal depth fractions on either instrument
> (displacements ~0.1–0.17 both times) — the cross-family attention warp is
> real. What is NOT established: its direction, which flips between the direct
> best-match and the spectral-axis placements (plausibly because sqrd12's
> attention fingerprints are weaker overall — peak cross-correlations 0.26 vs
> the bilinear pair's 0.34–0.42 — so aggregate placement is noisy).
> "Front-shifted" is withdrawn; "displaced, direction unresolved" stands. The
> MLP-side conclusion of §177 (no displacement, universal fractional placement)
> is unaffected — it held on both instruments.

Ledger #13. A fitting late reminder of the program's oldest rule: a claim's
direction needs two instruments to agree before it earns a verb.

## 179. Ledger #13 resolved: there is no warp — the fraction law is fully universal

File: `sqrd12_full_attn.py`. With all 12 sqrd12 attention fingerprints, the two
instruments agree at ρ = **0.90** (registered (a) held) and the median
displacement is **+0.000** (registered (b) failed in the best possible way):
sqrd12's attention places causal responsibility at its nominal depth fractions,
tracked closely by both the direct best-match and the axis instruments across
the whole stack. §169's "front-shift" and §178's "back-shift" were both
three-sample fluctuations in opposite directions — the honest ledger-#13 entry
("direction unresolved") was the correct intermediate state, and its resolution
is that the displacement itself was noise.

Final form of the universality law, now without exceptions among models tested:
**every component of every checkpoint — MLP and attention, bilinear and
conventional, 12 and 18 layers — assigns token-level causal responsibility at
its fractional depth.** One causal depth coordinate, three architectures, both
component types.

## 180. swiglu18: perfect atlas, doesn't join the axis — verdict withheld pending the second instrument

File: `swiglu18_atlas.py`; asset `swiglu18_atlas.pt` (36 components). The atlas
bars all hold (pairwise 0.08, depth-smooth 36/36, type-marked 0.11 vs 0.07),
but the fourth-model law test failed on the axis instrument: joint-embedding
fraction tracking drops to 0.55 and swiglu18's MLP components sit at median
displacement **0.333** from their fractions. Two readings: (i) the fraction law
has found its boundary — the gated-bilinear model places function differently;
(ii) the axis instrument fails when cross-model correlations are weak (exactly
§178's failure mode, which §179 resolved only by adding the direct instrument).
Per the ledger-#13 lesson, **no verdict until the direct best-match instrument
reports** — queued (registered: (a) instruments agree ≥0.5 across swiglu18's 18
MLP components; (b) if they agree, the direct median displacement decides the
law's boundary at the 0.08 bar).

## 181. Resolution: the law's fourth model passes on the reliable instrument

File: `swiglu18_direct.py` (print-label bug noted: the "nominal" column printed
li/12; computations used li/18). Registered (a) failed — instruments disagree
(ρ −0.22) — which triggers the pre-registered fallback: the **axis instrument is
declared unreliable for weak cross-model fingerprints** (second demonstration;
§178 was the first), and the direct best-match reading stands. That reading:
swiglu18's MLP components' implied fractions rise monotonically 0.06 → 0.94 and
track their nominal fractions at median |displacement| ≈ 0.06 — **inside the
0.08 law bar**. §180's boundary threat dissolves: the depth-fraction placement
law holds at its fourth model. One bounded anomaly logged: swiglu18's L2–L5 all
best-match bilin18's L1 (a front plateau — its early stack compresses onto the
bilinear model's crown layer).

Benchmark rule added: spectral/joint-embedding placements require a minimum
cross-correlation strength to be admissible; below it, use direct best-match.
The final tally: **four models, two component types, one law — token-level
causal responsibility sits at fractional depth** — earned twice over through
the two-instrument discipline that caught its own instruments failing.

## 182. The plateau dissolves: zero anomalies outstanding

File: `swiglu18_plateau.py`. All four plateau layers' L1 wins are near-ties
(margins 0.023–0.039, verdict pre-registered as argmax noise), and the
runners-up are the correct fractional neighbors (L2, L0, L2, L3) — the
underlying similarity structure points at the right depths; only the argmax
wobbled. §181's "bounded anomaly" is withdrawn as noise. The universality arc
ends with no exceptions and no open anomalies: four independently trained
models, three MLP architectures, two component types, thirty-six to
seventy-eight fingerprints each — all placing token-level causal responsibility
at fractional depth.

## 183. Leverage is text-borne: the fingerprint matrix factorizes

File: `bilin18_token_leverage.py`, all three bars held. Summing |delta| over all
36 components per token: leverage concentrates on hard tokens (0.65 vs base
loss) but is *more* strongly shared across models than explained by difficulty —
bilin18's and bilin12's per-token leverage profiles correlate at **0.78**. Which
tokens the machinery bears on is a property of the language, not the model. The
closing symmetry of the atlas work: the fingerprint matrix's **row structure
follows depth** (one causal coordinate, fraction-lawful across four models) and
its **column structure follows the text** (leverage shared at 0.78 across
independently trained models). Where in the stack × where in the language — the
two axes of this family's computation, both now measured.

## 184. Leverage universality: all four models, one text profile

File: `leverage_universality.py`. Registered (a) held — all six model-pair
leverage correlations ≥ 0.54 — and (b)'s failure is a coverage artifact stated
plainly: every low pair involves sqrd12, whose saved fingerprint set holds only
3 components against the others' 24–36, so its leverage sum is undersampled.
The coverage-matched comparison is decisive the other way: **bilin18–swiglu18
(36 components each, different MLP families) share the text-leverage profile at
0.78–0.83 — the highest pair measured.** Which tokens a transformer's machinery
bears on is a property of the language, shared across every architecture in the
registry. A last line for the program's ledger of themes: the models differ in
how they compute; they agree on what is hard, where in the stack to work on it,
and which words the work lands on.

## 185. Coverage closed: one text profile, four full atlases, minimum pair 0.74

File: `sqrd12_atlas.py`; asset `sqrd12_atlas.pt` (24 components). All three bars
held: the fourth atlas has the family structure (pairwise 0.10, depth-smooth
22/24, type-marked), and with coverage matched the §184 caveat resolves —
sqrd12's leverage correlations rise to 0.74–0.84, making the **four-model
minimum pair 0.74**. Final form of the leverage law: every pair of the four
independently trained models, spanning three MLP architectures and two depths,
agrees on the per-token leverage profile at 0.74–0.84. All four fingerprint
atlases are now complete, saved, structurally uniform, and mutually aligned on
both axes — depth (the fraction law) and text (the leverage profile). The
program's assets: four atlases, one benchmark spec with measured references and
instrument rules, a 13-entry resolved corrections ledger, and a findings report
current at a single URL.

## 186. Track-1 pilot: one pass, two benchmark-hardening lessons

File: `bilin18_track1_pilot.py`. Registered (a) and (b) failed, (c) held (3/4
signs) — and each failure teaches the track something concrete:

- **attn14 passed** (+0.183, above floor+0.05): the one explanation derived from
  per-token evidence ("net-harmful; deleting it relieves confidently-wrong
  tokens") predicts its fingerprint.
- **mlp16 scored −0.135 — a regime mismatch, not necessarily a wrong story**:
  its explanation describes the gain-frozen content level (value on easy
  tokens, §143), but the fingerprints are free-norm, where §142 showed its
  deletion lands on hard tokens via gain amplification. **Lesson 1: explanations
  must be scored in the measurement regime they describe** — the fingerprint
  assets should ship both free-norm and gain-frozen variants, and explanations
  declare their regime.
- **The matching null couldn't discriminate** (correct 0.107 vs shuffled 0.109)
  because three of four predictors degenerated to ±base-loss. **Lesson 2:
  difficulty-shaped explanations are unfalsifiable against each other; Track-1
  explanations must compile to distinctive per-token predictions to be
  scoreable** — a submission rule, now in the spec.
- attn1's position story scored ~0: early attention's damage is not
  position-graded — a small negative finding logged against any "more context,
  more damage" intuition.

## 187. Regime re-scoring: one explanation certified, one floor-grade, the asset shipped

File: `bilin18_frozen_fingerprints.py`; asset `bilin18_frozen_fingerprints.pt`
(gain-frozen fingerprints for the pilot components — the regime-paired data the
spec now requires). Results:

- **attn14's explanation is certified**: +0.277 in the frozen regime (up from
  +0.183 free) — "net-harmful late attention whose deletion relieves
  confidently-wrong tokens" predicts its fingerprint in both regimes, the
  Track-1 pilot's one full pass.
- **mlp16: regime mechanism confirmed, explanation floor-grade.** The score
  flips sign exactly as diagnosed (−0.135 free → +0.053 frozen; largest regime
  shift of the four, cross-regime corr 0.67, registered (c) held) — but +0.053
  is below the bar. The "value on easy tokens" story is directionally right at
  content level and too weak to count as understanding. An honest grade for the
  program's own explanation.
- The pilot ledger closes: 1 certified (attn14), 1 floor-grade (mlp16),
  1 weak (mlp9), 1 refuted (attn1's position story). The benchmark's semantic
  track is now demonstrated end to end — assets, floors, regime rules, matching
  nulls, and a first scored submission with honest mixed results.

## 188. Round 2: the first certified edge explanation

File: `bilin18_track1_round2.py`. Registered (a) and (c) held, (b) fell short:

- **E1 certified — the edge-explanation class works.** "attn6 transports L5's
  MLP-written content" compiles to fingerprint kinship, and attn6's fingerprint
  resembles mlp5's at 0.223 versus 0.087 for the median other MLP (2.6×, past
  the +0.05 bar). The §§133–134 cargo edge is thereby re-confirmed by a third
  independent instrument (span deletion, pattern-clamping, and now fingerprint
  kinship), and Track-1 gains a scoreable relation type: edge claims compile to
  "these two components mark the same tokens."
- E2 (mlp9, confident-error refinement): improved +0.092 → +0.129 but under the
  +0.05 gain bar — directionally supportive of the overshoot story, not
  certified.
- E3 (attn14): the confident-error framing ties its plain-difficulty score
  (0.171 vs 0.183, within tolerance) — the plain story already carries the
  content.

Track-1 scoreboard after two rounds: **two certified** (attn14's component
story, attn6's edge story), two floor-to-weak, one refuted — with regime rules,
distinctiveness rules, and now a working edge-explanation class.

## 189. The kinship map recovers the relay's direction; global ranking is front-dominated

File: `bilin18_kinship_map.py`. Registered (a) and (c) held: **16 of 18
attention components' top MLP kinship partner sits at their own layer or
upstream** (token-shuffled null: 11/18) — the relay's directionality is
recoverable from fingerprint resemblance alone. Registered (b) failed
instructively: the cargo edge ranks 19/324 globally because the map is
**front-dominated** — most components' top partner is mlp0/mlp1, the crown
layers whose fingerprints are strongest and correlate with everything.
Instrument rule (added to the spec): kinship edge claims are scored in the
per-component *relative* form (§188's certified design — partner vs that
component's median-other), never by global ranking, which conflates "shares a
channel" with "both touched by the dominant front signal." Also visible in the
map: attn17's top partner is mlp16 (0.21) — the 16→17 interchange surfacing in
a fourth independent instrument.

## 190. The interchange certifies by kinship: five instruments, one edge

File: `bilin18_interchange_kinship.py`, all three bars held. "attn17 reads what
mlp16 writes" certifies at kinship **0.212 vs 0.026 median-other** (8×
enrichment — the strongest relative kinship measured), with directionality
(attn17~mlp16 0.212 > attn17~mlp17 0.170) and MLP-side symmetry (mlp17's top
same-type partner is mlp16). The 16→17 interchange now stands on **five
independent instruments**: the composition excess and product law, the
cut-and-finetune load (+0.067), the quadratic-skin and norm accounting, and
fingerprint kinship. Track-1 scoreboard: **three certified explanations** (one
component, two edges), and the kinship class has produced both a certification
and its own scoring rule in two rounds.

## 191. The interchange is a deep-model feature: family edges split by depth, not architecture

File: `family_edge_kinship.py`. Registered (a) held — **relay directionality is
universal** (upstream-partner fractions 12/12, 12/12, 11/18 across the three
siblings; nulls ~7/12, 9/18): everywhere in the family, attention components
mark the same tokens as upstream MLPs. Registered (b) failed with the
informative pattern:

    bilin12  (12L, bilinear):      attn11~mlp10 = 0.012 vs 0.032 — NOTHING
    sqrd12   (12L, conventional):  attn11~mlp10 = 0.069 vs 0.036 — sub-bar
    swiglu18 (18L, gated):         attn17~mlp16 = 0.255 vs 0.021 — CERTIFIED, 12×

The output-stage interchange exists in **both 18-layer models** (swiglu18's is
even stronger than bilin18's 0.212/0.026) and in **neither 12-layer model** —
it splits by depth, not MLP architecture. The model family's strongest local
structure is an emergent feature of deep stacks: given enough layers, the last
attention and the second-to-last MLP form a dedicated hand-off that shallow
siblings never build. A genuinely comparative finding that none of the
single-model instruments could have seen.

## 192. CORRECTION: "the interchange is a depth feature" withdrawn — the two signatures dissociate

File: `family_interchange_causal.py` (control clean at +0.0009). The causal
instrument inverts §191's kinship pattern:

    swiglu18 L16/L17: d 0.016/0.669, excess +0.012  (kinship said CERT, 12×)
    bilin12  L10/L11: d 0.064/0.314, excess +0.091  (kinship said nothing)

And my registered bars ignored the program's own product law — excess scales
with d₁·d₂, so raw excess is confounded by damage sizes. Normalized coupling
c = excess/(d₁·d₂): **bilin18 ≈ 1.4, swiglu18 ≈ 1.1, bilin12 ≈ 4.5** — positive
everywhere, largest in the shallow bilinear model.

> **Ledger #14.** §191's "the interchange is a depth feature" is withdrawn.
> What stands: the *token-marking* signature (kinship — the last attention and
> second-to-last MLP marking the same tokens) appears only in the 18-layer
> models; the *causal coupling* signature (normalized composition excess
> between the last MLP pair) exists in every model tested and is strongest in
> bilin12. The two signatures measure different things — shared token territory
> versus interaction strength per unit damage — and they dissociate across the
> family. Neither alone defines "the interchange," and any future claim about
> it must state which signature it means.

The discipline note: this overclaim lived for one wake before its own protocol
caught it — the two-instrument rule is doing its job.

## 193. The coupling constant is not a model-level scalar

File: `family_coupling_stability.py`. Registered (a) and (c) failed, (b) held:

    bilin18  : c = 0.38 (k=4) -> 1.65 (k=8)
    swiglu18 : c = -3.91 (k=4) -> 1.09 (k=8)   [negative: compensation regime]
    bilin12  : c = 2.59 (k=4) -> 4.53 (k=8)

Within one model, c moves severalfold with span size, and swiglu18's flips sign
at small damage — the compensation-inversion regime §123 found in bilin18
appearing in a sibling. What survives all sizes: **bilin12's coupling exceeds
swiglu18's** (held at both k), and §192's dissociation lesson stands unchanged.
Final scoping rule (added to the benchmark's interaction-fidelity axis):
interchange and coupling comparisons are only meaningful per
(signature, damage-family, damage-size) triple — there is no single number
called "the interchange strength," within a model or across them. The
comparative arc closes properly humbled: directionality of the relay is
universal; everything quantitative about interactions is conditional.

## 194. The sign flip replicates: swiglu18's output pair actively compensates at small damage

File: `signflip_replicate.py`, all three bars held. On disjoint rows, swiglu18's
k=4 interchange excess is **−0.100** (original −0.070; 143% of magnitude, c =
−5.8), while bilin12's positive control replicates (+0.046, c = 2.6). The
§193 cell stands as a real phenomenon: at small damage, swiglu18's L16/L17 pair
is strongly *sub*additive — its output-stage machinery absorbs joint damage so
well that breaking both spans together costs less than breaking them
separately. The compensation motif (bilin18's §§115, 119, 121) is thus not
only present in the gated sibling but *dominant* at its output pair in the
small-damage regime — the family's output stages all stabilize, and they differ
in where the compensation-vs-coupling balance tips. Recorded with its
replication; the comparative arc's ledger is clean.

## 195. Hillclimb round 1: allocation doesn't pay — the frontier is flat in every direction tried

File: `bilin18_hillclimb1.py` (457s, the session's longest run). Greedy
per-layer rank allocation with sequential refit and joint scoring **matches but
does not beat** the uniform-rank reference at every budget:

    0.1M: greedy +1.808 vs uniform +1.807
    0.3M: greedy +1.682 vs uniform +1.660
    1.0M: greedy +1.574 vs uniform +1.541

Registered (a)/(b) failed; (c) held emphatically — the learned allocations are
wildly nonuniform (rank spread 32×, e.g. L5 at 64 while L9/L13/L16 sit at 2) —
and yet the nonuniformity buys nothing. §159's flat frontier extends to the
allocation dimension: in this model, *where* rank is spent matters far less
than *that* the stand-ins are sequentially refit. Benchmark lesson recorded:
blind allocation search is not a frontier lever here; the standing levers
remain refit and (untested at scale) different computation classes. The
strategy-2 fight (reader-aligned vs variance truncation) is next on the queue
and now carries the interesting question alone.

## 196. Round 1 verdict: at this grain, understanding loses to variance — twice

File: `bilin18_circuit_assign.py`. The reader-aligned rule ("write what your
reader reads") **loses to blind variance truncation 11/12 individually and by
0.34 nats jointly** (A +1.708 vs B +2.049) — the registered alternative, and
the §109 diffuseness verdict extending to allocation: the measured watch-lists
are genuine but *minority* channels (28% of a layer's value, §172), so
spending all eight ranks on them discards the diffuse majority the reader also
consumes.

Round-1 synthesis, stated plainly for the hillclimbing program: **at the
subspace-allocation grain, circuit knowledge does not beat blind variance in
this model — in either form tried** (greedy rank allocation matched uniform,
§195; reader-alignment lost outright). Where understanding *has* paid on this
benchmark: choosing the computation class per layer (constants where §155
showed nothing is needed), the sequential-refit protocol, and knowing the norm
regimes. Hypothesis for round 2: the wins live at the class-and-protocol level
and at *cross-layer sharing* (one basis amortized across the diffuse tail),
not at per-layer subspace choice.

## 197. The scale harness ships: self-tests green

Files: `HARNESS.md` (spec) and `harness_skeleton.py` (working reference,
drafted by a parallel subagent, verified end-to-end): an architecture-agnostic
replacement harness with the single-traced-forward contract, five mandatory
per-model self-tests (identity, no-op vs an independent reference
implementation, mean-ablation cross-check, gain-freeze-free-at-zero-damage,
fit cross-check — all GREEN on the bilin18 adapter), sequential refit and
joint-only scoring built in, both norm regimes (the frozen regime got its own
dual forward, validated by the zero-damage test), greedy allocation with
staleness-keyed map caching, and balanced-gauge parameter accounting. The
model-specific surface is ~100 lines of traced forward per architecture;
everything else is generic. This was the identified blocker for
"throw GPUs at larger models" — it now has a reference implementation.

## 198. Hillclimb rounds 1–2 synthesis: the Pareto curve is parameterization-invariant

File: `bilin18_shared_basis.py`. The shared-basis assembly (one 64-dim basis
pair for all twelve tail stand-ins, 0.20M params) scores **+1.734 — exactly on
the uniform curve's interpolation at 0.2M** (neither better nor worse at
matched params); held-out-layer bases cost +0.12 more (partial generalization).
Combined with §§195–196, four parameterization schemes have now been tried at
matched budgets:

    uniform ranks        -> on the curve (the reference)
    greedy allocation    -> on the curve (matches, never beats)
    reader-aligned       -> below the curve (loses 0.34)
    cross-layer shared   -> on the curve (param-neutral)

**The fidelity-vs-parameter curve of this model's tail appears
parameterization-invariant**: any r-parameter linear summary, however
structured, captures the same amount — the strongest measured form of the
diffuseness thesis, now as Pareto-invariance. The levers that actually moved
the frontier this session: sequential refit (+36%), computation-class selection
(constants where §155 licensed them), and the norm-regime accounting. Scale
recommendation recorded in BENCHMARK.md: invest engineering in class selection,
refit protocol, and the harness self-tests — not in allocation search.

## 199. Round 3: an optimization-limited null, not a class verdict

File: `bilin18_hillclimb3.py`. Both class bars failed, but the failure is
attributable to the fit, not (yet) the class: L16's outputs carry variance
~2.5×10¹² (it writes at RMS 1851), and a cold-started Adam run leaves the
stand-in at MSE 1.2×10⁹ — the +13.1 CE is a non-converged fit, not a measured
class limit. L1's fit converged 45× and still lost to rank-64 linear (+2.11 vs
+1.11), suggestive but unclean for the same reason. Recorded as
**optimization-limited**; the class lever gets one proper attempt (round 3b,
queued): fit the closed-form refit linear first, then a narrow factored
bilinear on the *residual* in output-normalized space — warm-started,
scale-sane. Registered: (a) L1 linear+bilinear-residual (width 64) ≤ +0.90
(beats rank-64 linear at comparable params); (b) L16 variant ≤ +0.05;
alternative: a clean class negative, which would close the ladder for good.

## 200. The class lever works: the hillclimb's first real frontier move

File: `bilin18_hillclimb3b.py`, all three bars held. Warm-started (closed-form
linear base + narrow factored bilinear on the normalized residual):

    L1  width-64: +0.330 at 0.22M   (vs rank-64 linear +1.11 at 0.15M; FULL
                                     linear +0.29 at 1.33M — matched at 1/6 size)
    L16 width-16: +0.031 at 0.055M  (vs linear r8 +0.059 at 0.15M)

**Refinement of §198**: the Pareto invariance is a *within-class* law — no
linear parameterization beats another — but **changing the computation class
moves the frontier**, by 3.4× at the front. And the class needed is exactly
what the program's maps said (§§101, 107, 110): genuinely quadratic at the
front where the real computation lives, nearly anything at the tail. The
hillclimb has its lever, and it is understanding-shaped after all — not
*which subspace* (that lost twice) but *which computation class where*. Scale
recipe, final form: per-layer class selection from the maps (constant /
low-rank linear / narrow warm-started quadratic), sequential refit, joint
scoring, harness self-tests. BENCHMARK.md Pareto references updated.

## 201. CORRECTION (ledger #15): §200's class-lever claim had a parameter-accounting error

File: `bilin18_champion.py`. The champion assembly's honest parameter count
exposed the bug: each warm-started bilinear combo carries a **full-rank D²
linear base (1.33M params) that §200's accounting omitted**. Corrected:

- §200's L1 combo: +0.330 at ~**1.55M** (not 0.22M) — versus full linear's
  +0.29 at 1.33M. Slightly worse at slightly more params: **no Pareto win**.
- The champion (class-selected 5–16): +1.619 at an honest **2.82M** — behind
  the linear frontier (+1.541 at 1.18M). Registered (a)/(b) failed on the
  honest count; (c) held (the bilinear rungs do buy 0.19 over rank-4 linear at
  their positions — the *class* helps, the *cost* was misstated).

> **Ledger #15.** "The class lever moves the frontier 3.4× at 1/6 size" is
> withdrawn — the warm base was uncounted. What §200's numbers actually show:
> linear + narrow quadratic residual ≈ full linear performance. Whether the
> class lever wins *net* depends on whether the warm base can be made cheap —
> queued: rank-32 linear base + width-64 bilinear at L1 (honest 0.30M).
> Registered: (a) ≤ +0.60 individually — then the lever stands, honestly; a
> failure closes the class question at this scale for good. The benchmark's
> own accounting rules caught this within one run — which is the system
> working, and the reason the parameter ledger exists.

## 202. The class question closes: the hillclimb's final verdict

File: `bilin18_hillclimb3c.py`. At honest accounting, both class bars failed:
L1's rank-32-base + width-64 bilinear scores +0.889 at 0.29M — below rank-64
linear (+1.11 at 0.15M) but consistent with the within-class curve at that
budget, i.e. **no class advantage**; L16's variant ties linear-r8 at 40% of
the params (+0.060 vs +0.059) — a marginal saving under its registered bar.
Per the pre-registration, the class question closes.

**Hillclimb arc, final summary (§§195–202):**

| lever tried | verdict |
|---|---|
| greedy rank allocation | matches uniform, never beats (§195) |
| reader-aligned subspaces | loses 0.34 (§196) |
| cross-layer shared basis | exactly param-neutral (§198) |
| narrow quadratic class (honest cost) | on the curve; marginal at best (§202) |
| sequential refit | **+36%, the one real lever** |
| constants where maps license | **free params, the other real lever** |

The honest conclusion for the benchmark: this model's replaceable function
defines a Pareto curve that *nothing structural shifts* — not allocation, not
alignment, not sharing, not computation class at honest cost. The only wins are
protocol (refit) and knowing where nothing is needed (constants). One
accounting error tried to say otherwise and lasted two hours (ledger #15).
This is itself the benchmark's calibration: on a model this diffuse, the
fidelity-complexity frontier is close to an information-theoretic given, and
submissions should be graded on *reaching* it cheaply, not on mythical beats.

## 203. Portability demo: the protocol transfers; replaceability scales with size

File: `bilin12_recipe.py` — the closed recipe run end-to-end on bilin12.
Registered (c) held: **sequential refit transfers** (21% improvement over
naive, 3.572 → 2.808 — the protocol lever is model-general). Registered (a)/(b)
failed for a substantive reason: **bilin12 licenses zero constants** — its tail
rank-0 costs run +0.12 to +0.54 against bilin18's +0.03–0.10, and its rank-4
joint lands at +2.81 where bilin18's comparable assignment sat near +1.8.

The comparative finding: **the smaller model packs more irreplaceable function
per layer — diffuse, compressible slack is a property of the larger model.**
For the scale program this is the right direction: replaceability appears to
grow with model size, so larger targets should have proportionally more cheap
tail, not less. Workflow lesson confirmed: thresholds and class licenses are
per-model empirical (the recipe's rank-0 scan is mandatory, not skippable);
only the protocol — scan, refit, joint-score, self-test — is universal.

## 204. The 2×2: size ordering survives, magnitude is architecture-modulated

File: `family_size_scan.py`. Registered (b) held cleanly — sqrd12 licenses
**zero** constants (rank-0 costs +0.12–0.37), matching bilin12's zero — but (a)
failed: swiglu18 licenses only **one** (L15), under the ≥2 bar. The full
licensing table:

    18 layers:  bilin18 = 4 constants | swiglu18 = 1
    12 layers:  bilin12 = 0           | sqrd12   = 0

§203's claim scopes accordingly: **replaceability increases with depth/size as
an ordering** (every 18-layer model out-licenses every 12-layer model; the 12L
row is uniformly zero) — but the *amount* is architecture-modulated, with the
pure-bilinear 18L model far more replaceable than the gated one. "Scales with
size" survives as a monotone ordering on this 2×2, not as a rate law; the real
test remains a genuinely larger checkpoint, and the prediction stands in that
scoped form.

## 205. The modulation is deep: gated models spread function, bilinear models concentrate it

File: `swiglu18_rank4_scan.py`. The registered alternative fired at **0/11** —
no swiglu18 tail layer drops below 0.05 even at rank-4 (costs +0.052–0.162) —
so its lower replaceability is not a missing zeroth rung but a property of
every rung. The shape is the explanation: swiglu18's tail costs are nearly
**uniform** (+0.05–0.09 across L7–15), while bilin18's ranged from ~0.02 slack
layers to concentrated load-bearers. The two architectures at equal depth make
opposite distributional choices: **the pure-bilinear model concentrates
function and leaves diffuse slack (which replacement exploits); the gated
model spreads function evenly and leaves slack nowhere.** This closes §204's
open cell and gives the scale prediction its refined form: what grows with
model size is the *slack*, and how much slack an architecture accumulates is a
family trait — worth measuring early on any new target, since it bounds what
any replacement submission can achieve before the first stand-in is fit.

## 206. Closing observation: the slack layers contain the regularizers

No new run — a connection already present in the recorded numbers. bilin18's
licensed-constant (slack) layers are {8, 9, 14, 15} (§157's rank-0 scan); its
deletion-*improves* spans at content level sit at {6, 9, 12, 15} (§120). The
overlap {9, 15} is exactly the pair whose regularizer character was replicated
and dissected (§§96–98, 122): the layers whose removal *helps* the frozen model
are a subset of the layers whose entire function a constant can carry. The
slack the replacement recipe exploits and the overshoot-trimming the
regularizer arc characterized are two views of the same low-content tail — and
the gated sibling, which spreads function uniformly (§205), correspondingly
showed no deletion-improves layers in any scan. The program's oldest
observation (truncation-as-regularization, first seen as a shifted-corpus
curiosity in §37) and its newest (slack as an architecture trait) close as one
phenomenon.

## 207. The slack-regularizer identity verified: exact in all three models

File: `family_regularizer_scan.py` — the verification §206 owed. Both bars
held, with an exactness the prediction didn't dare register:

    bilin18 : slack {8,9,14,15} ⊇ regularizers {9,15}
    bilin12 : slack ∅            = regularizers ∅
    swiglu18: slack {15}         = regularizers {15}  (deletion −0.030)

The no-slack model has no deletion-improves spans anywhere; the one-slack
model's lone improving span is exactly its lone slack layer. The identity is
now grounded cross-model instead of asserted: **a layer whose function a
constant can carry is the same kind of layer whose principal span can hurt
to keep** — slack and truncation-as-regularization are one property, and it
appears wherever an architecture concentrates function enough to leave any.
§206's overreach is retroactively licensed by its own verification run — but
the lesson stands recorded: the assertion went in a section before the scan
existed, and the discipline caught it one wake later.

## 208. The shared vocabulary is behavioral, not elementwise — and §61 replicates

Files: `bilin18_loro_replicate.py` + `bilin18_loro_fresh.py`, one mis-registration
between them, stated plainly. The replicate script's bar (a) FAILED (0.26 vs the
registered 0.55) — but the failure was in my registration, not in §61: I had it
measure **matrix-element** (Frobenius) reconstruction of the coupling matrices,
a stronger quantity §61 never claimed. The faithful rerun — §61's construction
verbatim, with the evaluation activations drawn from fresh rows never used for
the writer coordinates or reader spans — replicates the original:

    activation-weighted LORO, fresh rows : 0.637   (original 0.711; random −0.27)
    matrix-element LORO, fresh rows      : 0.261   (random +0.07; gap held)

The two numbers together sharpen what "shared vocabulary" means. The 80-basis
carries what a held-out reader's coupling matrices **do** to realized layer-1
activity (R² 0.64) far better than what they **are** as matrices (0.26): readers
share the behaviorally live components of their quadratic forms and diverge in
the components the activation distribution never excites. The vocabulary is a
shared way of acting on the same signals, not a shared set of matrix elements —
(§209 note: the further claim made here that sharing is specific to writer L1,
citing §114, was struck one wake later — behavioral sharing is writer-general.) No ledger entry:
§61's published claim was correct as stated and survives row-decontamination;
the wrongly-registered bar is recorded here as the instrument note it is.

## 209. Correction: vocabulary sharing is writer-general — §114's L1-specificity was a metric artifact (ledger #16)

File: `bilin18_behavioral_writers.py`. §208's lesson applied to §114, and §114
does not survive it. §114 measured cross-reader sharing for writers L0 and L9
with the matrix-element metric (0.12–0.25, bar failed) and concluded strong
sharing is L1-specific. But the matrix-element metric underestimates behavioral
sharing 2.5× (§208). On the activation-weighted instrument, fresh rows:

    writer L0: behavioral LORO 0.699   (random −0.20)
    writer L1: behavioral LORO 0.637   (random −0.27)
    writer L9: behavioral LORO 0.538   (random −0.14)

Registered bar (a) — both non-L1 writers under 0.45 — FAILED in the strongest
way: **L0's vocabulary is shared more strongly than L1's.** Ledger entry #16:
"strong sharing is L1-specific" corrects to **readers share a behavioral
vocabulary over every strong writer tested** — the shared code is a property of
how the reader population acts on upstream signals, not a special relationship
with one writer. §114's subsidiary finding survives (L11 is a normal reader of
L0/L9 — its dissidence remains L1-specific), and §208's closing sentence citing
§114 as consistent is struck. The pattern of the last three sections, stated
for the record: the elementwise metric systematically underestimates sharing,
and every conclusion built on it needed re-measurement. Both LORO instruments
now sit in the harness notes with their scopes: matrix-element for "what the
forms are," activation-weighted for "what the forms do" — only the second
supports vocabulary claims.

## 210. The sharing landscape: a private writer, a solitary reader, and three dead stories

Files: `bilin18_weak_writers.py`, `bilin18_causal_split_loro.py`,
`bilin18_random_v_loro.py`, `bilin18_communal_overlap.py` — one arc, four
runs, run in the wake of ledger #16. The corrected claim (§209: behavioral
vocabulary sharing is writer-general) needed its scope mapped, and the map has
real structure:

    behavioral LORO by writer:  L0 0.70   L1 0.64   L6 0.16   L9 0.54   L12 0.51
    per-fold, reader L17 only:      0.42      0.42     −0.31      0.03      0.11
    random-projection floor: 0.23

Two exceptions to writer-generality, both sharp. **L6 is a private writer**:
no reader reconstructs its couplings well — its 0.16 sits below the
random-projection floor, so readers agree on L6's coordinates *less* than on
arbitrary directions. **L17 is a solitary reader**: it shares no writer's
vocabulary (−0.31 to 0.42, worst fold for every writer) — consistent with its
§profile as the near-linear output head reading its own narrow code.

Three candidate explanations, each registered, each killed by its own run:

1. **Causality** (upstream readers can't see the writer, so their folds should
   be weak): refuted — acausal upstream folds pool at 0.51 vs downstream 0.63.
   Readers' forms agree even about signals they never receive.
2. **Global weight geometry** (the sharing is in the readers' weights,
   visible through any projection): refuted as the main story — random-V LORO
   is 0.23, far under writer-V's 0.64–0.70. Writer coordinates carry a real
   excess; the global floor exists but is small.
3. **Communal subspace** (shareability = overlap with the L0/L1 span the
   λ₁·x₀ re-injection keeps live): refuted, inverted even — L6 has the
   *highest* communal overlap among mid writers (0.39 vs 0.21/0.19) and the
   lowest shareability.

What stands: the reader population owns one shared way of acting on most
writers' coordinate systems — early writers most (0.64–0.70), mid writers
substantially (0.51–0.54), with exactly one writer whose code the population
collectively ignores (L6) and one reader that participates in none of it
(L17). The organizing principle behind those two exceptions is not causality,
not generic geometry, not communal-subspace membership; it is registered here
as open. L6's privacy is the sharper puzzle: the model routes a mid-depth
unit's output *around* the shared vocabulary while keeping it in the most
communal part of the stream.

## 211. A fourth dead story, killed free of charge — L6's privacy stays open

No new run: the fourth candidate explanation for L6's privacy dies on data
already in `bilin18_content_profiles_results.json`. The candidate: L6 is
private because its principal output span is a *regularizer* (content-level
deletion −0.016, one of §120's improves-set), so the reader population never
learned to consume it. The control kills it in the same table: **L9 is more
regularizer-flavored still (−0.023) and shares at 0.54.** Four stories now
dead — causality, global geometry, communal overlap, regularizer character —
each by a registered check, none replaced. L6's privacy is the program's
cleanest standing anomaly: a mid-depth writer sitting squarely in the
communal part of the stream, with a regularizer-shaped principal span like
its neighbors, whose 48-dimensional code the entire reader population
nonetheless declines to share a vocabulary over. Registered for whoever picks
this up (larger checkpoint or new instrument): the discriminating measurement
is per-coordinate — whether L6's *non-principal* coords (9–48, past the
regularizer span) are also unshared, which no current instrument isolates.

## 212. The anomaly narrows to eight directions: L6's privacy is its span, not its code

File: `bilin18_l6_tailcoords.py` — the discriminator §211 wrongly called
unbuildable, built by restricting the writer coordinates to SVD components
9–48. All three bars held:

    L6 coords 9-48: LORO 0.41  (full coords: 0.16)   random −0.17
    L9 coords 9-48: LORO 0.46  (full coords: 0.54)   random −0.20

Past its top-8 span, L6 is a normal writer — its tail code is shared at the
same level as the control's. The privacy that made L6 the program's cleanest
anomaly lives entirely in eight directions: the top-8 output span, which is
also its regularizer span (content-level deletion −0.016). The comparison
with L9 sharpens rather than kills the refined story: dropping the span
barely moves L9 (0.54 → 0.46) but transforms L6 (0.16 → 0.41) — L9's
regularizer span is *shared*, L6's is *private*. So "regularizer ⇒ private"
stays dead (§211), and what stands is narrower and stranger: one 8-dimensional
object in one mid-depth layer that carries most of the layer's output
variance, trims overshoot when deleted, and is the only code in the measured
model that the reader population declines to share a vocabulary over. The
standing anomaly is now localized: not layer 6 — span 6:1–8.

## 213. Span 6:1–8 is not unread — and an instrument confound, caught in-wake

File: `bilin18_span6_consumers.py`. Both registered outcomes technically
failed, and the run carries an instrument confound stated plainly before any
conclusion: the random-span controls were drawn from L6's components 9+,
which hold far less output variance than the top-8 span, so the raw response
ratios (5.5–11.2×, every downstream layer) are magnitude-inflated — the
registration should have demanded variance-matched controls. What survives
the confound is the *shape*, normalized by L7 — the immediately downstream
layer, whose response is the passive carry-through of whatever was deleted:

    response ratio / L7 baseline:
    L8-L14: 1.07-1.21     L15: 1.24     L16: 1.66     L17: 2.04

Two things stand. First, the (a) unread-cargo story is dead regardless of
normalization: span 6:1–8's content demonstrably propagates and every layer's
output moves with it — the model writes it, the stream carries it, downstream
computation transforms it. Deletion-improves therefore coexists with genuine
downstream consumption, as it did for the other regularizer spans. Second,
the depth profile of span-specific amplification rises toward the output end
and peaks at L17 at 2× the passive baseline — the direction the registered
long-shot (b) pointed, but compounding-through-depth is an unexcluded
alternative reading, so the "solitary reader reads the private span" story
remains unproven and is NOT claimed. The anomaly's standing description
after this wake: eight directions that the whole model responds to, that no
reader shares a vocabulary over, and that help when deleted. Certifying or
killing the L17 channel needs variance-matched controls plus a
freeze-intermediate-layers instrument, registered here as the requirement.

## 214. The honest channel: 87% of the peak was confound; the rest is real, direct, and masked by compensation

File: `bilin18_span6_channel.py` — the certifying instrument. Sanity exact
(frozen clean drift 0.0). With magnitude-matched controls the story shrinks
and sharpens at once:

    open (middle live):   L16 ratio 1.21   L17 ratio 1.42
    frozen (MLPs 8-15 pinned clean): L16 1.80   L17 1.62

First the deflation, stated plainly: §213's 11× peak was overwhelmingly the
magnitude confound — at equal injected energy, span content excites the
output end only 1.4× more than random content. The dramatic private-channel-
to-the-solitary-reader story dies at scale: bar (a) failed, and the frozen
excess is no larger at L17 than at L16, so nothing singles out the solitary
reader specifically.

Then the certification, which is small but clean: bar (b) held, and the
*pattern* is the interesting part. Freezing the middle MLPs — removing their
ability to respond — makes the span-specific ratio **rise** (1.2→1.8, 1.4→1.6).
The middle doesn't relay the span's signal to the output end; it partially
**absorbs** it, diluting span-specific damage with generic compensating
response, exactly the compensation regime the composition arc measured. What
reaches L16/L17 directly through the residual stream and attention carries a
genuine content signature (1.6–1.8× a matched-energy random perturbation).
So span 6:1–8's transport is now characterized: a modest direct
residual/attention channel to the output end, invisible at full strength
because the middle compensates over it. The anomaly rests here at earned
scope: eight private directions, written loudly, helping when deleted, read
weakly-but-specifically by the output end, and shared by no one.

## 215. The private writer is universal — and it sits at depth fraction one-third in both models

File: `bilin12_sharing_landscape.py`. The registered long-shot held exactly:

    bilin12 LORO by writer: L0 0.61  L1 0.56  L2 0.59  L3 0.32  L4 −0.08
                            L5 0.38  L6 0.31  L7 0.40  L8 0.48  L9 0.24
    random-V floor 0.22 | private writer L4, depth fraction 0.33
    bilin18 private writer L6, depth fraction 0.33

bilin12 reproduces the whole landscape shape: early writers shared most,
sharing declining with depth, and **one sharp privacy notch — below the
random-projection floor — at exactly the same depth fraction as bilin18's
(0.33 vs 0.33)**. Registered bars (b), (c), (d) all held; (a) failed at the
7-of-10 ≥ 0.35 bar (5 of 10) — overall sharing is weaker in the smaller
model, the same size trend the replaceability arc measured, recorded as a
scope note. The depth-fraction placement law, which put every component type
at its fraction across four models, now places a phenomenon discovered two
wakes ago: **wherever a bilinear stack is trained, the reader population
declines to share a vocabulary over one writer at one-third depth.** The
model-specific anomaly has become an architectural regularity — L6's private
span is not a quirk of bilin18 but the 18-layer instance of a fixed-fraction
structure. Scope stated plainly: the coupling-form instrument requires
bilinear MLPs, so this family test covers the two bilinear models only;
swiglu18/sqrd12 have no L⊙R forms to build the vocabulary from. Open,
sharpened: what computation lives at fraction 1/3 that wants a code nobody
else speaks — and does the fraction persist in a larger bilinear checkpoint
(the standing scale prediction gains its sharpest test yet).

## 216. The signature transfers whole — and sheds the regularizer coincidence

File: `bilin12_l4_tailcoords.py`, all bars held. bilin12-L4's privacy is
span-concentrated exactly like bilin18-L6's: past the top-8 span its code is
shared normally (0.46, control L8 at 0.56, randoms below 0.1). The universal
object is now fully specified: **in each bilinear model, one writer at depth
fraction one-third carries a top-8 output span the reader population declines
to share a vocabulary over, while the rest of its code participates
normally.** And the transfer subtracted a coincidence: bilin18-L6's private
span happened to be deletion-improving, but bilin12-L4's private span costs
+0.036 to delete — private and useful. Privacy is not regularizer character
(now dissociated cross-model, not just cross-layer); it is its own thing: a
loud, consumed, functional code that trained readers of the same stack
systematically fail to — or decline to — decode with their common vocabulary.
The arc rests with the phenomenon universal, localized, transport-measured,
and unexplained: the best-specified open question the program has produced.

## 217. The private computation is partially conserved — above every null, below the shared baseline

File: `family_private_fingerprint.py`. Registered bar (a) FAILED as written
(0.159 vs the 0.2 bar) and the honest statement keeps both halves:

    private pair (b18 span 6:1-8 ~ b12 span 4:1-8):  ρ = 0.159
    shared pair, fraction 0.5 (L9 ~ L6):             ρ = 0.229
    nine random-span cross-model pairs:              max 0.021, median 0.007

The two private spans damage substantially overlapping token sets on shared
text — 7.5× the strongest random pair, unambiguously related computations —
but the conservation is *weaker* than a generic matched-fraction shared span
(long-shot (c) also failed; control (b) held, revalidating the instrument at
span level). So the fraction-1/3 object is functionally the same kind of
thing in both models without being token-identical. One reading, offered as
interpretation and not claim: a code disciplined by a shared vocabulary is
pinned by its many readers and reproduces tightly across training runs; a
private code, read only weakly and directly (§214), is free to drift while
serving a related function — privacy and looser conservation would then be
two faces of missing consensus pressure. The measurable on-box facts about
the private writer are now: placed (fraction 1/3, both models), localized
(top-8 span), transported (direct to output end, compensation-masked),
dissociated (not regularizer character), and partially conserved (0.16 vs
0.23 shared baseline). What remains needs a bigger family.

## 218. The last reader ranks last everywhere; true solitude is 18-layer-deep

File: `bilin12_solitary_reader.py`. Bars (a) and (c) held, (b) failed, and
the split is the result. bilin12's fraction-0.94 reader L11 is the worst
LORO fold for 4 of 5 writers (the fifth's worst is the second-deepest
reader), so the ordinal signature — the output-end reader shares least — is
universal. But L11's median fold is 0.43: it still participates in the
common vocabulary, where bilin18's L17 genuinely secedes (median ≈ 0.11,
below zero for one writer). The pattern repeats §192's dissociation exactly:
in the shallow model the tendency is ordinal (last reader least shared); in
the deep model it sharpens into a categorical exception (last reader
solitary). Both landscape anomalies now have their universality statements:
the private writer is categorical in both models at fraction 1/3; the
solitary reader is ordinal everywhere and categorical only at 18 layers —
consistent with depth specializing the output head (bilin18's L17 is the
near-linear, 4-direction unit; bilin12's L11 retains more generic function).
The sharing landscape is complete at family scope.

## 219. Privacy is content, not address: the private code is a whisper under the loudest channels

File: `family_span_exclusivity.py`. Both registered bars failed in both
models, killing the dedicated-wire picture completely — and the refutation
reframes the anomaly better than the hypothesis would have:

    write energy into span 6:1-8 directions (bilin18): L0 91.5M, L17 45.1M,
    L16 35.4M, L3 19.9M ... owner L6: 0.44M — the owner is among the
    QUIETEST writers into its own span. bilin12 identical in shape
    (L0 22.4M vs owner L4 2.0M). Nearly every component in both models
    writes into these directions above its matched-random baseline.

So the private span's eight directions are not reserved real estate; they lie
in the busiest, most communal part of the stream (consistent with §-communal
overlap, where L6 scored *highest* among mid writers). What is private is not
the address but the *content*: the specific quadratic structure the
fraction-1/3 writer adds into directions everyone else also uses. The
corrected description of the anomaly, at family scope: **one mid-depth writer
whispers its own functional code underneath the loudest shared channels, the
reader population owns no common vocabulary for that contribution (though it
reads those same directions constantly), the middle absorbs its perturbations,
and a faint direct channel carries it to the output end.** Instrument note
recorded: absolute write energy compares differently-scaled layers — the
aiming measure (each component vs its own random baseline) is what shows the
directions are globally high-variance; both measures agree on the verdict.

## 220. The whisper is understood — in dialects

File: `bilin18_dialects.py`. The two main bars held decisively; one null was
mis-registered and is corrected in place. Over the private span 6:1–8, each
reader's OWN rank-18 basis reconstructs its held-out coupling forms far
better than the population's basis (median self 0.75 vs cross ~0.24; median
gap **+0.56**), while over the shared control writer L9 the population basis
is as good as your own (gap −0.10). So the hard-content story dies: the
private span's content is not unstructured — every reader compresses it
cleanly. What is missing is *agreement*: the readers hold mutually misaligned
private codes for the same signal. The whisper is heard by everyone,
understood by everyone, and shared by no one.

Two honest notes. First, the registered null (c) failed because the bar was
mis-set: forms over 8 coords live in a 36-dimensional space, so a random
rank-18 basis captures a substantial fraction by construction (measured
median ~0.17, one fold 0.26); the self-vs-cross comparison is internally
matched and unaffected, but the ≤0.1 registration was wrong for this space
and is recorded as such. Second, an unregistered observation worth its own
test: the dialect split is depth-graded — early readers L2/L3 actually do
*better* with the population basis (gaps −0.16, −0.38) while deep readers
carry huge positive gaps (L5 +0.57, L9 +0.56, L13 +0.35, L17 +1.02). Read at
face value: the early readers share a common code for the fraction-1/3
whisper, and it is the deep readers who go private. The anomaly's standing
description gains its last clause: the unshared vocabulary is specifically a
deep-reader phenomenon.

## 221. Correction: there is no early shared code — idiosyncrasy is total (ledger #17)

File: `bilin18_dialect_groups.py`. §220's unregistered depth-graded reading
("early readers share a code for the whisper, deep readers go private") went
into the published report and does not survive its registered test — ledger
entry #17. At pair resolution over the private span, NOTHING transfers above
the measured random floor:

    private span: early internal 0.31, deep internal 0.33, early→deep 0.03
                  measured random rank-18 floor: 0.31
    control (L9): early internal 0.62, deep internal 0.82, early→deep 0.81
                  random floor: 0.10

The control held (c), so the instrument distinguishes genuine sharing when it
exists. Over the private span there is no early code and no deep dialect —
**every reader's code for the whisper is fully idiosyncratic**. What §220
read as early-reader agreement was the 200-form population basis capturing
the dominant behavioral directions of easy test sets: the random floor on
exactly those test sets is 0.31 (vs 0.10 on the control), and pair-fit bases
collapse onto it. Both numeric bars (a) and (b) were also wrong-footed by
that floor — registered at 0.45/0.25 against an unmeasured baseline that
turned out to sit at 0.31; per rule (d) the conclusions here are stated
relative to the measured floor, and the floor-measurement-first rule now
joins the registration discipline. The whisper's final description loses its
last consoling clause: heard by all, understood by each alone, agreed by
none — at any depth.

## 222. The notch is one layer wide in both models

File: `bilin18_notch_profile.py`, all bars held (nulls measured per the
ledger-#17 rule: −0.26 to −0.14). bilin18's completed writer profile:

    L0 0.70  L1 0.64  L3 0.43  L5 0.43  [L6 0.16]  L7 0.52  L9 0.54  L12 0.51
    bilin12: L3 0.32  [L4 -0.08]  L5 0.38

Privacy does not shade in gradually — in both models the immediate neighbors
of the private writer share normally (L7 actually above the mid-depth trend),
and the notch is exactly one layer deep. Combined with the placement match
(fraction 0.33 in both), the object's structural specification is now fully
sharp: **one layer, top-8 span, fraction one-third, notch width one.** The
landscape profile is complete in both bilinear models, and everything the
program can measure about the private writer on this box has been measured.

## 223. The private writer is a private LAYER — attention at fraction 1/3 is unshared too

File: `bilin18_attn_landscape.py`. The attention half of the landscape, and
the registered "privacy is an MLP property" prediction (b) failed in the
sharpest way:

    attn writers: L1 0.49  L4 0.29  [L6 0.13]  L8 0.35  L12 0.47  L16 0.38
    (mlp profile:  L5 0.43  [L6 0.16]  L7 0.52; nulls measured, all clean)

Attention at layer 6 is the least-shared attention writer, at the same level
as its MLP (0.13 vs 0.16). The notch is not the MLP's quadratic content but
the **layer's entire output** — both component types at fraction 1/3 emit
low-consensus code. This kills the §219-era framing "privacy is the MLP's
content" in its narrow form: whatever layer 6 does, its attention transport
carries it too. Under the relay picture (attention transports upstream MLP
writes), two readings survive on-box: attn6 selects or transforms cargo into
the same private register, or attn6 predominantly transports the loud L6-span
content that saturates those directions. Distinguishing them needs cargo
attribution beyond this box's measured instruments. Scope notes recorded:
(a) narrowly failed — attention sharing overall runs slightly below the MLP
range (median 0.38 vs the 0.40 bar; attn4 also lowish at 0.29), so the
attention landscape is shallower-shared in general, and the notch claim rests
on attn6 being the clear minimum, not on the absolute bar. The private
object's final on-box name: **layer 6, whole output, fraction one-third,
notch width one, consensus zero.**

## 224. Correction: the layer-level privacy was borrowed — the private object is the span alone (ledger #18)

File: `bilin18_attn6_borrowed.py`. §223's headline ("privacy is a LAYER
property"), published in the report, lasted one wake — ledger entry #18.
With attn6's output coordinates orthogonalized against the mlp6 top-8 span,
its consensus recovers completely:

    attn6:  0.126 raw  →  0.452 orthogonalized   (normal attention range)
    attn12 control: 0.474 → 0.492 (unmoved)      nulls −0.26/−0.11

Layer 6's attention is a normal writer whose output happened to carry the
span's content — which is itself the expected behavior under the relay
picture: attention transports what MLPs write. The correction sharpens the
final description rather than weakening it: **the private object is the
single 8-dimensional span; mlp6 writes it, attn6 carries it, every reader
understands it idiosyncratically, none agree.** The fork was decidable
on-box after §223 called it off-box — the second time this program deferred
a measurement its own instruments could make (§212's lesson, repeated and
now generalized in the discipline: before deferring, write down the
orthogonalization or restriction that would decide it).

## 225. The carrier effect is 18-layer-specific; the family agreement is MLP-side only

File: `bilin12_attn_borrowed.py`. Bars (a), (c), (d) held; (b) failed, and
the failure scopes ledger-18's finding rather than undermining it:

    bilin12 attn4: raw 0.23 → orthogonalized 0.28   (recovery minimal)
    bilin18 attn6: raw 0.13 → orthogonalized 0.45   (recovery total)
    bilin12 control attn8: 0.40 → 0.39 (unmoved); nulls clean

In bilin18 the attention at the notch was fully explained as cargo — remove
the span's directions and it is a normal writer. In bilin12 the attention at
the notch is only mildly depressed to begin with (0.23, against an attention
control of 0.40) and stays depressed after orthogonalization: whatever
lowers it is not the 8-dim span. So the family symmetry of the anomaly is
**MLP-side only**: the private span itself matches across models in
placement, width, concentration, and consensus (exact), while the attention
story differs — total borrow at 18 layers, mild and unexplained depression
at 12. Scope note: only two bilin12 attention writers are measured, so
"mildly depressed" rests on one control; a full bilin12 attention profile
would firm it, and is noted as the obvious next filler run rather than a
claim. Ledger #18 stands as written — it was and is a bilin18 statement.

## 226. No attention notch at 12 layers — the anomaly is only ever the MLP span

File: `bilin12_attn_profile.py`. The full profile resolves §225's flagged
uncertainty by its registered alternative:

    attn0-5: 0.25 0.18 0.26 0.20 0.23 0.25   attn6-9: 0.38 0.39 0.40 0.39
    median 0.26, MAD 0.08; minimum attn1; attn4 within one MAD -- NO notch

§225's "mild depression at attn4" was sampling luck — its single control
(attn8, 0.40) came from what the full profile reveals as a distinct high
late-attention group. Bar (b) also failed informatively: bilin12's attention
median is 0.26, so attention sharing at 12 layers is shallow generally. The
unified cross-model statement, now at full scope: **the anomaly is only ever
the MLP span. Attention involvement is transport, never origin** — at 18
layers the carrier makes the span's content visible in attn6's output
(recoverable by orthogonalization), at 12 layers attention shows no notch
signature at all. Unregistered observation, recorded descriptively: bilin12's
attention sharing RISES with depth (0.18–0.26 early, 0.38–0.40 late) while
its MLP sharing falls — opposite gradients within one model; untested, noted
for any future profile work. The private-span dossier closes here: every
on-box thread is resolved or explicitly scope-noted, and the object's full
name is stable across twelve sections of adversarial testing.

## 227. The gradient story dies as a universal: bilin12's rising attention is its own trait

File: `bilin18_gradient_test.py`. Completed bilin18 attention profile (ten
writers, attn6 excluded as borrowed) and the three gradients:

    b18 MLP Spearman(depth)  −0.43   (falls, but missed the −0.5 bar — L7's
                                      0.52 bump softens the decline)
    b18 attn                 −0.13   (flat)
    b12 attn                 +0.82   (rises, strongly — the §226 observation
                                      confirmed within its own model)

The opposite-gradients pattern is 12-layer-specific, not a family trait —
registered alternative (c) held. What survives as family statements: MLP
sharing declines with depth in both models; attention sharing has no
universal depth shape. One number worth a scoping note rather than a claim:
attn17 is bilin18's least-shared attention writer (0.15), but every reader in
its fold set sits upstream of it and no MLP ever reads the final attention
output — for a last-layer attention writer the LORO instrument is measuring
purely acausal folds, so this is an instrument boundary, not a new anomaly.
With this, the profile work is exhausted on both models and both component
types; the arc's durable yield is the MLP decline (universal), the notch
(universal, MLP-only), and the seceding last reader (18L-only).

## 228. The notch is real but its floor-crossing is instrument-relative (ledger #19)

Files: `run_preregistration.py` (scorecard v1), `bilin12_notch_readerset.py`,
`bilin18_notch_readerset.py`. The scorecard's validation run on bilin12 did
exactly what validation is for: it failed to reproduce the known verdicts and
exposed an instrument sensitivity that the published claims had inherited.

The facts, across reader-ensemble sweeps and both aggregations:

    bilin18 L6:  0.16-0.26 pooled / 0.28-0.34 fold-median across 4 ensembles
                 control L9: 0.48-0.54 / 0.51-0.62 (stable, always ~2x L6)
    bilin12 L4:  -0.08 to +0.26 pooled across 5 ensembles (always below the
                 L8 control by >= 0.13) -- but +0.63 under fold-median
                 aggregation with the scorecard ensemble: the notch VANISHES
                 there even relative to its profile. Load-bearing readers
                 identified by single swaps: the {1,5,11} core.

Ledger entry #19, stated plainly: the categorical claims "below the
random-projection floor" (§§210, 215, and the report) are true in the
original construction but are NOT ensemble-independent facts about the
model — absolute notch depth moves with reader choice, and in bilin12 even
the relative notch is destroyed by fold-median aggregation. What survives
every construction tested is: **bilin18's L6 is depressed to ~half its
in-model control in all 8 instrument variants.** bilin12's notch is robust
across pooled constructions only. The universality statement (§215) is
therefore downgraded from "below-floor in both models" to "robust relative
depression at fraction 1/3 in bilin18; construction-dependent depression at
the matched fraction in bilin12." PREREGISTRATION P3 is rewritten
accordingly: pooled aggregation mandatory, minimum three reader ensembles,
and the bar is RELATIVE (notch <= 0.5x the in-model control median across
all ensembles), not absolute floor-crossing. Scorecard v1's other verdicts:
P4/P5 replicated robustly; P1 reported honestly; P2's zero-tolerance bar is
fragile near the ±0.01 boundary (L7's span flipped sign between span-fit
samples) and gains a replication requirement. The scorecard failing on its
first validation run is the system working — this is why the harness runs on
a known model before it ever sees a new one.

## 229. Concentration is robust in its invariant form — the sensitivity itself is span-localized

File: `family_concentration_sweep.py` (note: K=40 coords for matched full/
tail dimensionality, so absolute values differ slightly from the K=48 runs;
all comparisons internal). Bar (a) missed at one edge and the miss is the
finding:

    bilin18 L6:  full 0.17/0.28/0.31 across ensembles — tail 0.40/0.43/0.42
    bilin12 L4:  full 0.15/0.11               — tail 0.46/0.53

The tail-coords value is ensemble-STABLE in both models (spread ≤ 0.07)
while the full-coords value flexes by 0.14–0.19 — set D's ratio fell to 1.3
only because its full value rose. So §228's instrument sensitivity is not a
diffuse property of the LORO method: **it is localized to the span-dominated
measurement.** The non-span code shares at a constant normal level under
every construction tested; what the reader ensemble changes is only how
much the private span's irreconstructible content drags the pooled median.
This upgrades the concentration claim to its ensemble-invariant form —
"tail coords share stably at 0.40–0.53 in every construction; the span is
where both the privacy and the instrument sensitivity live" — and P3's tail
sub-bar is amended to the measured invariant (tail >= 0.35 absolute in
every ensemble, replacing the ratio form whose denominator flexes).

## 230. Dialects are construction-robust in the discriminative regime; the audit closes

File: `bilin18_dialects_sweep.py`, twelve variants. Bar (a) failed only at
rank 24, and the failure is a mapped instrument boundary, not a fragile
claim:

    rank 12: gaps +0.48/+0.51/+0.58/+0.54    rank 18: +0.56/+0.55/+0.46/+0.79
    rank 24: +0.13/+0.08/+0.26/+0.20         control: |gap| <= 0.10 in 12/12

Forms over 8 coordinates live in a 36-dimensional space; at rank 24 the
population basis spans two-thirds of it, so any content — shared or private —
reconstructs, and the self/cross distinction saturates away by construction.
In the discriminative regime (rank <= half the ambient dimension) the
dialects gap is large and stable across every split and ensemble tested,
and the control is flat in all twelve variants. §220's headline stands with
a scope boundary: **dialects are a rank-regime finding — keep the basis rank
at or below half the form-space dimension.** That rule (the rank analog of
ledger-17's floor rule) goes into PREREGISTRATION P7. With this, every
published headline of the sharing-landscape arc has been swept under the
ledger-19 discipline: the notch (rewritten to relative form), concentration
(upgraded to invariant form), the whisper's idiosyncrasy (floor-corrected at
#17), and dialects (rank-scoped here). The audit is complete; the arc's
claims now all carry construction-robustness statements.

## 231. The v2 validation wave: secession falls (ledger #20), L7 is a boundary case, P5 rewritten

Files: `run_preregistration2.py`, `bilin12_l7_span_replication.py`,
`bilin18_solitary_sweep.py`. Scorecard v2 (pooled, three ensembles,
relative bars, replicated fits) ran its bilin12 validation and exposed the
claims the ledger-19 audit had NOT yet swept — the bilin12-side landscape
and the solitary reader. Three resolutions:

**1. Ledger #20 — no construction-robust secession.** bilin18's L17, the
"solitary reader" (§§210, 218, published), is worst-fold 5/5 with median
0.14 under ensemble A — and 2/5 with median 0.44 under the mid-heavy
ensemble D. The secession is ensemble-dependent: L17 looks solitary against
early-heavy reader ensembles and normal against mid-heavy ones. With
bilin12's ordinal claim also failing its sweep (worst 1/10 in one v2
ensemble), §218's entire categorical-vs-ordinal contrast falls. Corrected
statement: last-reader depression exists in some constructions in both
models and is robust in none. The report is corrected accordingly.

**2. The bilin12 notch degrades further.** v2's mostly-early ensemble
(0,1,2,5,8,10) gives L4 = 0.52 POOLED — falsifying §228's "robust across
pooled constructions." bilin12's notch is construction-dependent, period.
The universality of the private writer now rests on: bilin18's relative
depression (8/8 constructions, solid) plus a bilin12 depression that
appears in ensembles containing its {1,5,11}-like core and not otherwise.
The fraction-0.33 coincidence stands only at that weakened scope.

**3. L7 is a boundary case, not a violation.** Four independent span fits:
rows 0-60 both improve, rows 60-120 both don't; the §207 instrument
reproduces its published +0.0305 exactly. The slack-regularizer identity
survives with a documented stats-row sensitivity at one layer; P2 now
requires four fits across the full stats window, all agreeing.

PREREGISTRATION updated: P5 rewritten (last-reader depression measured
across >= 3 ensembles, reported as a distribution, no categorical bar);
P2 four-fit rule; P3's bilin12-side corroboration note weakened. The
methodological arc of the last five wakes, stated once: every claim
measured under one construction was fragile until proven otherwise; the
four that survived their sweeps (relative notch depression at 18L, tail
stability, dialects-in-regime, MLP decline) are the arc's durable core.

## 232. Partial conservation is construction-robust — the audit truly closes

File: `conservation_sweep.py`, four variants (2 span-fit windows × 2 text
halves), both bars 4/4:

    private pair: 0.14-0.20   shared pair: 0.20-0.27   random max: 0.01-0.05

In every construction the two private spans' token-damage fingerprints
correlate well above every random pair and below the matched shared pair —
§217's "partially conserved" survives exactly as published, the only
landscape claim to pass its sweep without needing a single amendment. With
this, every published quantitative claim of the sharing-landscape arc has
now been construction-swept, and the final tally stands: four claims robust
as published (MLP decline, tail stability, dialects-in-regime, partial
conservation), two robust in amended form (notch → relative depression at
18L; concentration → absolute tail form), and three fallen or degraded
(below-floor categoricity, secession, the bilin12 notch's generality) —
ledger entries 19 and 20. The arc ends the way this program's arcs end:
smaller than it looked at its peak, sharper than it looked at its start,
and with every surviving number carrying its construction-robustness
statement.

## 233. The relay's directionality replicates on fresh text; the cargo edge's kinship rank does not

File: `bilin18_kinship_fresh.py` — all 36 fingerprints recomputed on text
rows 320–384 (never before used for fingerprints). Three-part verdict:

**Directionality: replicated, stronger.** 17/18 attention components' top
MLP kinship partner is upstream-or-same (94%; original 89%). The relay's
statistical leg survives fresh text.

**The cargo edge's kinship RANK: not replicated.** attn6~mlp5 ranks 23rd of
324 cross-type pairs (was top-5). On this window, kinship is dominated by
the loud early writers (mlp0/mlp1 are the top partner for 11 of 18
attention components). Stated precisely: the edge's CAUSAL certification
(interchange 0.223 vs 0.087, §-certified) is untouched — what weakened is
the statistical proxy's prominence, and the benchmark's edge-explanation
class should cite the causal number, not the kinship rank.

**The null band was mis-registered.** Token-shuffled fingerprints give
12/18 upstream — outside my 40–60% band, but the band was wrong, not the
null: with all partners at noise level, the top partner is uniform over 18
MLPs, and the structural expectation for upstream-or-same is ≈53% with a
±12-point binomial sd at n=18. The measured null (67%) is within 1.2σ of
that; the signal (94%) is 3.5σ above it. Lesson filed with the floor rule:
compute the NULL'S EXPECTED VALUE from the selection structure before
registering its band — a best-of-k selection over a causal ordering has a
built-in bias, and 40–60% was the band for a coin, not for this selector.

## 234. E1 re-certified on fresh text — rank was the wrong statistic, margin is the right one

File: `bilin18_e1_rescore.py`. The certified edge explanation ("attn6
transports L5's MLP-written content") survives its fresh-window re-score:

    fresh:    attn6~mlp5 0.193 vs median-other 0.092 (margin +0.101)
    original:            0.223 vs           0.087 (margin +0.136)

The §233 rank collapse (23rd of 324) and this margin survival are the same
data read through two statistics, and only one of them is right for edge
claims: RANK asks "is this pair the loudest," which the generically dominant
early writers always win on any window; MARGIN-over-median asks "does this
specific pair mark the same tokens more than a typical pair does," which is
what the explanation actually claims. The margin replicates at 2x the bar.
E1 stands as a three-instrument certification (span deletion, pattern
clamping, fingerprint kinship), and BENCHMARK.md's edge-explanation class
keeps kinship compilations with the margin statistic named as the score and
rank explicitly disallowed. With this, every certified Track-1 entry has a
fresh-data leg, and the replication backlog is empty.

## 235. attn14's score replicates fresh — the closing sentence is now exact

File: `bilin18_attn14_rescore.py`. The one certified Track-1 entry whose
SCORE (as opposed to phenomenon) lacked a fresh-data leg, re-scored on rows
320–384:

    frozen regime: +0.250 (original +0.277; bar +0.15)
    free regime:   +0.155 (original +0.183) — regime ordering preserved

Both bars held. The certification's regime structure reproduces too: the
gain-frozen score exceeds the free score on fresh text just as it did on
the original window, so the regime-declaration rule earns a replication of
its own. The report's closing sentence — every certified claim carries a
fresh-data leg — is now exact rather than defensible: attn14's story
(score +0.250 fresh), the cargo edge (margin +0.101 fresh), the relay
directionality (94% fresh), and all the earlier replicated headlines. The
record is closed at full strictness.

## 236. The circuit pipeline's first day: 147 structural, 27 semantic, and a measured DSL ceiling

Files: `circuit_atlas_big.py`, `circuit_certify.py`, `circuit_stageD_prep.py`,
`circuit_score_stories.py`, with 8 parallel story-writing agents between
stages D and E. The circuit-by-circuit track, restarted at scale per the
user's direction, produced in one pass:

- **Stage A**: all 36 component fingerprints over 54,272 fresh tokens in
  95 s (sign agreement with the original atlas 33/36); discovery and
  replication halves split at birth.
- **Stage B+C**: k=256 distinctive-ownership clustering; **147 of 154
  powered clusters certified as ownership circuits** by held-out
  replication (profile cosine ≥ 0.7, owner overlap, cohesion over a
  measured 0.027 floor), covering 76% of well-predicted tokens.
- **Stage D**: 8 fresh subagents wrote stories + mechanical membership
  rules for all 147 (134 non-null; 13 honest "unclear").
- **Stage E**: **27 semantically certified** on held-out data at
  precision ≥ max(5× base, 0.15) and recall ≥ 0.15 — median precision
  lift **81×** base rate, shuffled-null lift 0. Registered bar (a) of 40
  missed; (b)(c)(d) held.

Failure anatomy (the next rung's blueprint): 4 no-fire, 11 near-miss
(precise, too narrow), 92 low-precision — the first-order DSL
(target/prev/induction/position) cannot separate those clusters, i.e. the
ownership structure is real (held-out-replicated) but its trigger is
richer than one-token surface features. Scoreboard: 3 → 27 semantically
certified circuits in one session; the semantic ceiling is now a measured
property of the rule language, not of the model or the pipeline.

## 237. Red-team and confirmation: stories were document-bound; the circuits themselves are not

Files: `circuit_wave2_prep.py`, `circuit_score_wave2.py`,
`circuit_atlas_third.py`, `circuit_confirm.py`; 8 fresh agents (6 refine,
2 red-team). The full second cycle, with every registered bar and its
verdict:

**Red-team round** (user-directed): of 27 mechanically certified wave-1
stories, only **13 survived adversarial review** — 8 gerrymandered (token
grab-bags passing numerically: a month smuggled into a day-name list, a
"century" token welded onto a time rule), 6 story-rule mismatches. 12 of
the 14 proposed fixes re-passed mechanically. Refinement of the 107
failures with false-positive/missed-member evidence yielded 58 fixed
rules, 14 passing. Wave-2 provisional total: 39.

**Confirmation on the untouched third window** (rows 120-300): **6 of 39
confirm**. Failure anatomy: 0 cluster-gone, 22 rule-silent, 10
fires-wrong, 1 low-recall. And the transport fact that reframes
everything: **249 of 256 clusters keep healthy membership on the third
window** — the ownership structure (which components own which prediction
sites) is document-general, while the stories written for them were
document-bound: agents described the topical surface of one window's
documents (£ amounts, map labels, building names), and those topics simply
do not occur in the other window.

**Scoreboard, stated honestly**: 147 structural circuits (now with
cross-window transport evidence), 6 FINAL semantically certified, 33
provisional-failed with diagnosed modes. **Lessons for wave 3, applied**:
(1) evidence packs mix BOTH windows, forcing topic-invariant stories;
(2) class-level features first (token-class, induction, repetition), token
lists capped at 6 and only when the story names the lexical category;
(3) certification is cross-window BY CONSTRUCTION (rules built from window
A evidence, scored only on window C) — the adaptive-reuse problem
dissolves. The pipeline's second-day thesis: the model's circuits
generalize; our first two rounds of descriptions did not. That is a
statement about description discipline, not about the model — and it is
exactly what red-teaming was for.

## 238. The certification unit was wrong: supervised function circuits, and the first four causal certifications

Files: `circuit_score_wave3.py`, `supervised_circuits.py`. Wave 3 (141
topic-invariant stories, scored on a half-window no agent saw) passed **1 of
136** — class-level rules 0 of 111. The autopsy is a metric mismatch, not
bad stories: an honest functional description ("induction sites") is true of
its circuit but fires corpus-wide, while the bar demanded it identify one
micro-cluster among 256 — and one function spans MANY clusters (split by
ownership depth-band). Function stories cannot be cluster-precise; wave 1's
topical rules only passed because document vocabulary is accidentally
cluster-discriminative. The certification unit had to flip.

**The supervised track** (user-directed: the classic circuits): ten
mechanically defined function slices — induction/copy, repetition, digit
continuation, sentence-end punctuation, newline, quote-close, bracket-close,
subword continuation, name continuation, list comma — certified by
(i) cross-window ownership replication and (ii) causal damage concentration
of the top owners on-slice vs CE-matched controls. Instrument amendment
stated plainly: the registered top-3 set-overlap sub-bar proved brittle
(depth-adjacent owners swap ranks between windows even at profile cosine
0.98) and is replaced by cosine >= 0.90; original bars' failure recorded.

Results at amended criteria: **8 of 9 powered slices ownership-replicable
(cos 0.90-1.00 — function ownership is essentially window-invariant), and
four causally certified circuits**:

    digit continuation : attn8 + mlp15 (conc 2.3x / 2.5x)
    bracket closing    : attn13 (+attn4/7) (3.7x / 2.2x)
    subword completion : mlp16 + mlp15 (5.2x / 5.9x)
    name continuation  : attn1 + attn0 (5.3x / 3.5x)

Discoveries and honest negatives alongside: **induction is owned by
attn3-5** (registered guess attn1-2 FAILED — the early lexical attention is
the name-continuation circuit instead, which fits the §-era lexical-head
finding); induction, sentence-end, and comma are ownership-replicable but
NOT first-order concentrated (conc 0.5-0.8: redundant, distributed
computation — deleting any owner hurts matched control sites more);
**newline sites show negative damage** (deleting attn16/mlp17 improves
them — the flattening/regularizer channel surfacing inside a function
slice); and 79% of the unsupervised clusters' tokens fall inside these ten
functions — the 147 structural circuits are largely these functions,
subdivided by ownership. Scope note: IOI and addition have no natural
support in this web-text corpus; constructed-prompt windows are the
extension for task-style circuits.

## 239. Site-specificity: three circuits are local, and the name circuit works from the antecedent

File: `circuit_exclusivity.py` (instrument note: the attention module-hook
ablation agrees with the atlas's manual-forward instrument to four decimals,
+0.2416 = +0.2416). Positional ablation of each certified circuit's top-2
owners — everywhere vs on-slice-only vs off-slice-only, window C:

    digit   (attn8+mlp15): 94% of slice damage is site-local, 6% spillover
    subword (mlp16+mlp15): 102% site-local, 0% spillover — perfectly local
    bclose  (attn13+attn4): 64% local, 14% spillover
    name    (attn1+attn0):  18% local — 64% of slice damage comes from
                            ablating the owners at OTHER positions

Bars (a) and (b) held at 3/4, and the one "failure" is the best mechanism
discovery of the day: **the name-continuation circuit does its work at the
antecedent, not the prediction site.** Ablating attn0/attn1 away from name
sites destroys name predictions anyway, because those heads build the token
identity representations at the name's earlier mention that downstream
copying reads as its source — cross-position influence in this architecture
flows only through attention keys/values, so damage at the source position
arrives at the destination through the copy. attn1 is infrastructure (the
lexical representation everyone consumes), not a site-local name predictor —
consistent with its §-era "most important attention" role. Registered
moonlighting call (c) failed in the informative direction: mlp16+mlp15's
on-slice total cost (0.215) dwarfs their off-slice cost (0.039) — subword
completion is most of what those tail MLPs do, which squares with mlp16's
old profile as the syntax-bus consumer. The compression instruction from
this section: digit and subword circuits are site-local and slice-dominant,
so their owners are candidates for slice-conditioned replacement — the
first Track-2 targets nominated by certified semantics.

## 240. Slice-conditioned constants carry ~85% of the circuits — the first compression win from certified semantics

File: `circuit_replacement.py`. Rung 0 of slice-conditioned replacement,
fit on window A, applied on window C, and both registered directions were
beaten upward — my subword alternative (constants won't suffice) FAILED in
the informative way:

    digit   (attn8+mlp15): ablation +0.299 → const +0.055  (82% recovered)
    subword (mlp16+mlp15): ablation +1.182 → const +0.143  (88% recovered)
    random constants of matched norm: −105% / −323% (much worse than
    ablation — the content is specific, not just norm)

A fixed 8-number span coefficient per MLP owner and one fixed vector per
attention owner — conditioned ONLY on "this site is a digit/subword
continuation" — carries ~85% of what these components do at those sites.
The description cost is a one-line membership predicate plus a handful of
numbers, versus the rank-64 refit maps the flat layer-track needed. This is
the lever the layer-by-layer track structurally could not see: it replaced
components with input-generic functions, and the honest frontier said the
middle is high-rank. Slice-conditioning says: high-rank *unconditionally*,
nearly constant *given the site type*. The model's tail components look
like conditional-bias machines — "if subword site, write this vector" —
and the semantic track just paid for itself in the currency the user named:
ground-truth circuits buying better compression. Next rung queued: the
conditional-constant DICTIONARY — all ten function slices × all owner
components as one whole-model Track-2 submission priced at its true
description length.

## 241. Half the tail is a 640-number dictionary

File: `circuit_dictionary.py`. All eight tail MLP span components (mlp10-17)
replaced simultaneously by per-class constants — ten mutually-exclusive
function classes, fit on window A, applied on window C:

    joint span ablation: +0.675 nats     dictionary: +0.341 (recovery 49.96%
    — the registered 50% bar missed by rounding; recorded as failed)
    label-shuffled control: −17% (worse than ablation — assignments matter)
    description cost: 640 numbers + ten one-line predicates

Per-class (classes with near-zero ablation damage excluded from ratios —
their denominators blow up; floor note recorded): digit sites recover 102%,
subword 71% (bar (b) held), name 58%, the unclassified residual 26%, and
induction sites −9% — the dictionary is NO better than ablation where the
content must be token-specific (copying), exactly where constants should
fail. The honest summary of the submission: **half of everything the eight
tail span components do is "look up the site type, add the type's fixed
vector" — a ~0.7 KB description — and the unreplaced half concentrates
precisely in copying sites and the residual class.** The benchmark gains a
new frontier point purchased by semantics: the flat layer-track priced this
same content at rank-64 refit maps per layer. Next rung queued: per-class
linear maps for the two failing classes only (induction, other), priced
honestly against their parameter cost.

## 242. Rung 1: 95% of the tail spans replaced — with the oracle caveat stated before the applause

File: `circuit_dictionary_rung1.py`. Adding per-class rank-8 input-linear
maps for ONLY the two classes where constants failed (induction, residual):

    joint ablation +0.675 → dictionary +0.341 (50%) → rung 1 +0.035 (95%)
    induction class: −9% → 98%    (its stand-in reads the local stream,
    which carries the identity to be copied)
    per-parameter value: constants 52 nats/100K params, linear 0.21 —
    the semantic information is ~250x denser; the linear rung buys the
    second half at bulk rates. All three registered bars held.

**The caveat, which changes the claim's type**: the class predicates
condition on the TARGET token — this dictionary is ORACLE-CONDITIONED. As a
structural decomposition it stands and is strong: the eight tail span
components' content is, to 95%, a function of (target class, local input) —
ten fixed vectors plus two rank-8 reads. As a deployable Track-2
compression it does not yet count, because a runtime stand-in cannot see
the label. The registered next rung: an INPUT-ONLY classifier (site class
predicted from context — bracket-closing from an unclosed bracket, subword
from a mid-word fragment, induction from a matched prefix), with the
fidelity loss from classifier errors priced in. The gap between the
oracle-conditioned 95% and the input-only number will itself be a
measurement: how much of the tail's job is deciding WHICH site type it is
at, versus writing the type's content once decided.

## 243. Deciding the site type is real work: crude context rules hit 31%

File: `class_predictability_floor.py`. All three bars failed, and that is
the measurement: hand-written input-only rules predict the oracle site
class at 31% overall agreement (digit precision 0.05, subword 0.38, the
residual class 0.53). The strong signature classes have decent RECALL from
context (bclose 0.71, name 0.71) but terrible precision at their tiny base
rates. Verdict: the target-class information that made the §242 dictionary
work is not lying on the surface of the context — deciding WHICH site type
comes next is genuine computation. Which points the next rung at the
obvious place: the model itself has already made that decision by the time
the stream reaches mlp10. Queued: a linear probe on the tail's INPUT
stream predicting the site class (fit window A, applied window C), then
the probe-conditioned dictionary — fully input-only, honestly priced. The
gap from oracle-95% to the probe number will split the tail's job into
"deciding the type" (front-of-model work the tail reads) versus "writing
the type's content" (the dictionary).

## 244. The deployable dictionary: 75% input-only — and the tail's job splits into decide vs write

File: `circuit_probe_dictionary.py` (first run had a fit/apply interface
bug — linear maps fit on mlp10's input, applied to each component's own;
the λ-mixing bug class again, caught because the oracle arm failed to
reproduce §242's 95%. Fixed run reproduces it exactly, validating the
instrument.) The input-only rung:

    probe (linear, on the stream entering mlp10): 59% top-1 over ten
      classes (majority-class floor 36%) — bar (a) missed at 65%
    probe-conditioned dictionary: 75% of joint ablation recovered,
      fully input-only (oracle: 95%; shuffled probe: −67%)

The description is now deployable: one D×10 linear probe + 640 constants +
two rank-8 maps per component ≈ 160K params replacing 75% of what the
eight tail span components do. And the oracle-probe gap prices the split
the section-243 question asked for: of the tail spans' job, **~75 points
of recovery is "write the type's content" (dictionary), ~20 points is
"know the type better than a linear read of your input can"** — type
information that is either nonlinearly coded in the stream or computed by
the tail itself. That residual is the honest next frontier, and the
benchmark records the three-rung curve: 50% at 0.7KB (constants), 95% at
148K oracle-conditioned (decomposition), 75% at 160K input-only
(deployable).

## 245. Task circuits from constructed prompts: counting IS the digit circuit; IOI is executed by the "harmful" late attention

File: `task_circuits.py`. First constructed-prompt windows (the corpus has
no natural IOI/arithmetic support):

- **Counting** ("5, 6, 7, 8," → " 9"): the model is perfect (100% top-1)
  and ownership is the sharpest attribution the program has produced —
  **attn8 at +2.21 nats, ten times the runner-up**. The certified natural-
  corpus digit circuit (attn8+mlp15, §§238-240) IS the counting circuit on
  synthetic prompts. Cross-paradigm validation of a discovered circuit —
  registered bar (c) held.
- **IOI-analog** (58% top-1, bar (a) held): ownership at the answer is NOT
  the early copy band (bar (b) failed informatively) — it is the output
  end: attn14 +1.08, mlp17 +1.02, attn16 +0.97, mlp5 +0.83, attn13 +0.70.
  Combined with §239 (attn0/1 build the copy source at the antecedent),
  the task splits cleanly: early attention writes the source, LATE
  attention executes the retrieval at the answer. And the top executor is
  attn14 — the component whose certified Track-1 story is "net-harmful
  late attention whose deletion relieves confidently-wrong tokens" on
  natural text. Both are true: the component that hurts average prediction
  is the one that performs name retrieval when retrieval is the task. The
  strongest single dissociation between average-text and task-conditional
  importance in the record.
- **Addition**: 0% top-1 — the model cannot do single-digit addition;
  recorded as a capability fact per registration (d), ownership skipped.
- Shuffled-names control: 3% top-1, CE 5.99 (sane floor).

## 246. Twenty families: the amortization structure of 147 circuits

File: `owner_family_graph.py`. Linking circuits that share both majority
function class and a top owner: **147 structural circuits collapse to 20
families** (bar: ≤ 49). Multiplexed owners (≥3 function classes each)
include the mid-attention generalists (attn4, attn5) and late components;
subword and induction are split functions (≥3 distinct owner components
each — depth-multiplexed, matching why fine clusters looked function-
redundant in §238). The class×owner table is the amortization map the
benchmark's node-complexity idea wanted: twenty ownership families, ten
functions, and the certified-circuit list is the product of the two, not
147 independent facts.

## 247. Attention is two materials: type-constant at structure sites, irreducibly contextual at copy sites

File: `attn_dictionary.py`. All 18 attention components jointly replaced by
per-class constants (joint ablation +4.48 nats — deep-damage regime):

    recovery by class: bclose 85%  rep 83%  sentend 73%  comma 72%
                       digit 67%   name 41%  subword 23%  ind 3%  other −3%
    total 18% (registered band held); shuffled-label control 12%

Two honest notes before the reading. First, the shuffled control is high
(12% > the registered 5%) because in the deep-damage regime any structured
constant recovers generic norm — the aggregate dictionary signal is the
6-point margin, and the real evidence is the per-class dissociation, which
shuffling cannot produce. Second, newline's ablation is NEGATIVE (−1.28:
removing all attention IMPROVES newline sites — the regularizer channel
again), so its ratio is excluded per the floor rule and my "formatting"
bar (b) was mis-grouped; recorded as failed.

The reading: **attention output is two different materials.** At
structure sites — brackets, repetition, sentence ends, commas, digits —
its content is largely a function of site TYPE (67-85% constant-
replaceable): the "transport" at those sites carries a stereotyped signal.
At copy sites — induction 3%, subword 23%, the residual — it is
irreducibly contextual: what it transports IS the context. This cleanly
seconds the §239 antecedent finding and gives the context-conditional
replacement program its map: constants for the structural half, and for
the contextual half the stand-in must itself be a retrieval operation —
the registered next rung (a programmatic copy stand-in for the induction
band: attend to the matched prefix, emit the successor's value).

## 248. Label mining works; constants show diminishing returns — the residual is contextual, not under-labeled

File: `extended_dictionary.py`, fed by four label-mining agents (30
proposals from 320 decoded residual-class examples). Mechanically accepted:
24 labels (fires >= 100 residual sites, deduped) — listing delimiters,
capitals after clause punctuation, hyphenated-modifier heads, and the like.
The taxonomy grows 10 → 34 classes; the residual share of well-predicted
sites drops 36% → 22% on the held-out window.

The constants-only tail dictionary improves 50.0% → 55.0% recovery
(54.96% — the >= 55% bar missed by rounding; recorded failed). The
economics are the finding: the first ten classes bought 50 points of
recovery; twenty-four more classes bought 5. Label mining is cheap and
works (agents produce acceptable function-level labels at a 80% rate), but
the constant-dictionary channel is near saturation — the remaining
unrecovered content is CONTEXTUAL (token-specific), which is exactly what
the rank-8 input-linear rung already captures (95% oracle / 75% deployable,
§§242-244). Taxonomy refinement helps the deployable probe most (more
classes = sharper conditioning); further constants alone will not move the
frontier. The dictionary program's shape is now fully mapped: type-constant
content saturates near 55%, input-linear content reaches 95%, and the
20-point oracle/deployable gap is class-decision information.

## 249. Match-and-copy at the value level: refuted twice, honestly

File: `copy_standin.py` (two runs). Replacing the induction band's
(attn3/4/5) output at induction sites with an explicit retrieval — full
attention to the matched successor, its value through c_proj — makes
things far worse than ablation (raw −327%; per-layer scalar calibration
fit on window A: −337%). The random-target control is worse still (−390%),
so the matched position carries real signal, but the single-position
value-injection is wrong in DIRECTION: this product-attention architecture
evidently spreads the copy across many positions and the λ-mixed value
chain (my stand-in also approximated v1 by the layer's own values). The
board: constants 39% (the band's on-slice floor), naive copy refuted,
mechanism more distributed than the textbook story. A learned low-rank
channel from the matched-successor hidden state is the registered
next-if-pursued rung; for now the induction band keeps its honest label —
replicable ownership, causally concentrated at task level (counting), and
NOT yet mechanistically reduced.

## 250. The crown component is a lookup table: mlp1 folds at 79%

File: `fold_tables.py` (user-directed weights-first fold). Context-free
tables — every vocab token run alone through the model, each early MLP's
output captured — swapped in as full-output replacements on window C:

    mlp0: 15%   mlp1: **79%**   mlp2: −589%   mlp3: −1229%
    (ablation refs +0.51 / +5.79 / +0.55 / +0.55; shuffled table −350%)

The headline: **mlp1, the most important MLP in the model (+5.8 nats
ablation, the linearization crown, the vocabulary hub), is 79% a
token-indexed lookup table.** The single most load-bearing component
reduces to "look up the current token, emit its vector" for four-fifths of
its function — computed from weights+embeddings alone, no fitting. The
registered monotone-decay story failed informatively: mlp0 folds WORSE
(15%) than mlp1 despite sitting closer to the embeddings — its output is
evidently position/attention-sensitive in a way mlp1's is not — and at
mlp2/3 the context-free state is already so far from the running state
that table injection is catastrophic. Queued discriminator: EMPIRICAL
token-conditional mean tables (the best any token-lookup can do, fit on
window A) for mlp0-3 — separating "token-determined but not context-free-
foldable" from "genuinely context-dependent."

## 251. The front is a token-dictionary cascade

File: `empirical_tables.py`. Empirical token-conditional mean tables (the
ceiling for any token-lookup replacement) vs the §250 context-free folds:

    mlp0: 68% empirical (vs 15% fold)   mlp1: 85% (vs 79%)
    mlp2: 58%                            mlp3: 44%   (shuffled −54%)

The synthesis: **the model's front four MLPs are substantially
token-dictionaries**, decaying with depth as context mixes in — and they
differ in HOW their dictionary is stored. mlp1's is context-free: the
weights-only fold captures nearly the whole token channel (79 of 85) — you
can read its dictionary straight out of the parameters with one forward
per vocab entry. mlp0's is context-shifted: 68% token-determined but the
per-token mean moves with position/attention, so the naive fold misses it
(15%). mlp2/3 still carry 58%/44% token-determined content that only the
empirical tables see. For the benchmark's node-complexity language: the
front's natural interface unit is the vocabulary — per-token vectors are
the honest description currency there, and the user's fold conjecture is
confirmed where it was aimed (the early layers), with the mlp1 crown as
its sharpest instance: the most important MLP in the model is, to four-
fifths, a dictionary you can compute without ever running the model on
text.

## 252. Attention outputs are not token-tables — the dictionary cascade is an MLP phenomenon

File: `attn_tables.py`. Empirical token-conditional output tables for
attn0-5: recoveries 29% / 23% / 36% / −9% / 15% / −9% (shuffled −7%).
Registered bar (a) — the lexical pair attn0/attn1 at ≥50% — FAILED, and
the failure is an instrument-interface lesson rather than a refutation of
lexicality: an attention OUTPUT at position p is a mixture over the whole
prefix, so even a perfectly lexical head (token-determined keys/values)
has context-dependent outputs. The §239 infrastructure claim lives one
interface deeper: at the VALUE level. The dictionary cascade stands as an
MLP phenomenon (68/85/58/44% for mlp0-3 vs ≤36% for all attention), and
the queued discriminator moves the table to the right interface: replace
attn0/attn1's c_v outputs with per-token value tables at every position —
if the heads are lexical transport, value-tables should preserve what
output-tables cannot.

## 253. attn1's lexicality certified at the value level — and attn0's broadcast is more than lexical

File: `value_tables.py`. Per-token value tables (c_v output replaced by
the token's mean value vector at every position):

    attn1: 87% recovery  — but shuffled-vocab control 36%, so the honest
           TOKEN-SPECIFIC margin is ~51 points (the shuffle null was
           mis-chosen: it preserves the values' typical norm/mean
           structure; recorded per the null-design rule)
    attn4: 52% — the induction band's VALUES are half token-determined
    attn0: −223% — catastrophic

Three conclusions. First, the §239 lexical-infrastructure claim is
certified at its proper interface: **attn1 is a lexical value store** —
its values are token-determined (87%, margin 51 over a structure-
preserving null), its outputs vary only through the mixture weights, which
is why the output-level table (§252) failed at 23%. Second, attn4's 52%
reframes the §249 copy-refutation: the band's values are substantially
tabulable — what my match-and-copy stand-in got wrong was the pattern
side, not the value side. Third, attn0 is the genuine surprise: its value
role — which includes seeding the λ-mixed v1 broadcast that every layer
consumes — is NOT token-mean-approximable at all. The v1 broadcast carries
something beyond lexical identity (position structure, or per-context
normalization), and it now joins the private span as a named open object:
small interface, model-wide reach, not reducible to the obvious hypothesis.

## 254. Correction: there is no attn0 anomaly — its broadcast is EXACTLY lexical

Files: `attn0_broadcast.py` + the seen-prefix split. §253 named attn0's v1
broadcast a "new open object carrying more than lexical identity." One
wake later, the dissection kills that framing with an exact result:

    own-path arm: +0.018    broadcast arm: +0.389    (additive, sanity exact)
    broadcast damage at positions with fully window-A-seen prefixes: −0.0000
    at positions with ANY unseen prefix token:                        +0.395

The catastrophe was entirely the table's global-mean fallback for
vocabulary unseen in the fit window — and since every position's attention
reads all prefix values, one rare token contaminates the rest of its
sequence (45,348 of 46,080 positions had at least one). At seen tokens the
per-token table is EXACT, damage literally zero — which the architecture
in fact guarantees: at layer 0 the stream is the normalized embedding, so
c_v's output is a pure function of the token. **attn0's v1 broadcast is
the most lexical object in the model — exactly, not approximately** — and
the §253 open-object claim is withdrawn (in-record correction; the claim
never reached the published report). The practical consequence is pleasant:
v1 is trivially foldable — a weights-computable V×D table with zero
approximation error — and the layer-0 value system joins mlp1's dictionary
in the weights-readable column. Lesson filed: before naming an anomaly,
split the damage by the stand-in's COVERAGE — fallback cases masquerade as
mechanism.

## 255. The assembled dictionary model: +1.85 nats, sub-additive

File: `assembled_dictionary.py`. Every certified deployable stand-in
applied at once — attn0's values (weights-exact, solo cost 0.0000), attn1's
values (+0.032), mlp0/1/2/3 entirely (+0.162/+1.216/+0.228/+0.308, mlp1
from the weights-only fold), and all eight tail MLP spans via the
probe-conditioned dictionary (+0.167):

    sum of solo costs: +2.114     ASSEMBLED: +1.850   (0.88x — SUB-additive)
    base CE 3.535 → dictionary model 5.39; all solo sanities exact

The composition question resolved in the favorable direction: unlike raw
damage, which compounds multiplicatively (the §172-era product law),
dictionary REPLACEMENT errors partially absorb each other — the assembled
model is cheaper than the sum of its substitutions. The submission, stated
in benchmark terms: **thirteen of thirty-six components — including the
crown component and the entire model front — replaced by lookup tables,
one linear probe, and 640 class constants, at +1.85 nats**, with the attn0
table exact and the mlp1 table computed from weights without ever running
the model on text. This is the circuit program's Track-2 capstone: every
piece of it was nominated by a certified semantic story, and the flat
layer-track could express none of it.

## 256. The crown's ladder: linear wins, and the table's price is stated honestly

File: `mlp1_ladder.py`. The mlp1 stand-in ladder (sanity rungs exact):

    fold table +1.216   empirical table +0.840   table+rank-32 +0.564
    full ridge LINEAR: **+0.120**   (bar (a) missed by 0.014; (b) held big)

The crown component is ~97% linear in its input — the old "linearization
crown" title was still underselling it. The linear map subsumes the token
table because the stream entering mlp1 carries token identity linearly, so
a D×D map reads the dictionary AND the context corrections at once.

The honest-accounting note (ledger-#15 discipline): the token tables I have
been calling "cheap descriptions" cost V×D ≈ 58M raw numbers — more than
mlp1's own 16M parameters. Their value is INTERPRETIVE (the function IS
per-token vectors, derivable from weights with zero additional information
given the model) — but as standalone submissions they are expensive, and
the benchmark entry for mlp1 is the linear map: +0.12 nats at 1.3M params.
The two description levels are now stated separately everywhere: "what the
component is" (dictionary; semantic) vs "cheapest faithful substitute"
(linear; the Track-2 price). Assembled v2 queued with the crown's linear
rung swapped in.

## 257. Assembled v2 super-additive — the sequential-refit lesson crosses tracks

File: `assembled_v2.py`. Swapping the crown's stand-in from fold table
(solo +1.216) to ridge linear (solo +0.120) made the ASSEMBLED model worse:

    v1 (fold table):  solos sum +2.114 → assembled +1.850 (0.88x)
    v2 (linear):      solos sum +1.017 → assembled +2.690 (2.64x!)

Diagnosis: the linear map was fit on the CLEAN model's mlp1 inputs; with
mlp0 replaced upstream, mlp1's input distribution shifts and the map
misfires — while token tables, conditioning only on ids, are robust to
upstream substitution. The old layer-track bought this exact lesson as its
single biggest lever (sequential refit, +36%, §196-era), and the circuit
track just re-purchased it: **input-conditioned stand-ins must be fit with
their upstream replacements active.** v3 queued: sequential front-to-back
fitting of every fitted piece (a0 exact needs none; m0 table; m1 linear on
the a0+m0-substituted stream; m2/m3 tables likewise; a1 values; tail
probe+dictionary last). Registered: v3 ≤ +0.90 total.

## 258. The table-firewall effect: composition has a grammar

File: `assembled_v3.py`. Sequential fitting fixed exactly what it was
registered to fix — mlp1's in-context marginal cost is +0.124, matching
its clean solo (+0.120; bar (c) held) — yet the assembled total is +1.97,
still behind v1's +1.85. The decomposition explains it: the without-m1 arm
costs +1.85 BY ITSELF, though its parts' solos sum to ~0.9. The front's
token-table errors (m0/m2/m3, a1v) are transmitted and amplified by
whatever sits downstream that is input-faithful — including the REAL mlp1,
which being ~97% linear propagates its input's errors loyally. v1 scored
better not because its crown stand-in was good (+1.216 solo!) but because
an id-conditioned table CLAMPS its output to token-only values and thereby
absorbs upstream error — an accidental firewall.

The composition grammar, stated: **id-conditioned stand-ins are error
firewalls (lossy but stabilizing); input-conditioned stand-ins and
input-faithful real components are error conductors (accurate but
propagating).** An assembled model's cost is not a sum of solo costs plus
a generic interaction — it depends on the ORDER and TYPE of stand-ins
along the graph. Current standings: v1 +1.85 (all-table, firewalled),
v3 +1.97 (sequential, conductive), v2 +2.69 (naive). Registered next
rung (v4): keep the sequential linear crown but add rank-k input-linear
residual corrections to the m0/m2/m3 tables — fit sequentially, they
should absorb rather than transmit the table errors. The benchmark's
edge-typing vocabulary (coordinates vs summary vs opaque) was built for
exactly this distinction, and the assembled experiments are now measuring
it in composition.

## 259. Assembled v4: +1.39 — the composition grammar validated

File: `assembled_v4.py`. Tables plus sequentially-fit rank-32 residual
absorbers on m0/m2/m3 (each absorber reads the SUBSTITUTED input and
cancels the error its table would otherwise transmit):

    v1 +1.850 (all-table)   v2 +2.690 (naive linear)   v3 +1.971
    (sequential, conductive)   **v4 +1.392** — new frontier
    absorber residual-R²: 0.57/0.58/0.61; m1 in-context +0.088
    (bar (a) ≤1.20 missed; (b) held)

The §258 grammar is validated in the constructive direction: converting
each front table from pure firewall to firewall+absorber — clamp to the
token's vector, then add a learned low-rank read of the actual
(substituted) input — recovers half the residual and drops the assembled
cost 25% below the previous frontier. Description delta: three D×32×2
maps ≈ 222K parameters on top of the v1 inventory (tables priced per
§256's two-level rule). The assembled submission now stands at:
thirteen of thirty-six components replaced, +1.39 nats, with the
composition recipe itself — sequential fitting, firewall+absorb typing —
as much a product of the program as the numbers.

## 260. v5 takes the middle: 19 of 36 components at +2.10

File: `assembled_v5.py` (one heredoc-quoting stumble fixed in-wake).
Adding mlp4-9 as sequential firewall+absorber rungs:

    v5: 19/36 components replaced, total +2.095
    middle marginals (LOO, in-context): m4 +0.19, m5 +0.02, m6 +0.11,
    m7 +0.14, m8 +0.12, m9 +0.13 — sum +0.71, no component hit the
    pre-registered +0.4 drop rule; absorber residual-R² 0.61-0.71

The revision this makes to the program's oldest verdict deserves plain
statement: the flat layer-track concluded the middle was incompressible —
high-rank content, no cheap stand-ins, the "mezzanine" nobody could
linearize. That conclusion was true OF ITS DESCRIPTION CLASS
(input-generic replacements). Under the circuit-track's grammar —
token-table firewall plus a sequentially-fit rank-32 absorber that reads
the substituted stream — the middle costs about 0.12 nats per component
in composition, and mlp5 is nearly free. The frontier curve now has two
standing points: v4 = 13 components at +1.39, v5 = 19 components at
+2.10. Remaining unreplaced: attention 2-17 (the structure-class
constants of §247 are the obvious next rungs) and the mid attention
band's contextual transport.

## 261. Weight-only compression of the middle MLPs: the CP axis works, the subspace axis does not

Because the bilinear MLP is exactly `Down((Lx)*(Rx))`, compression hypotheses
can be computed from the weights with no data fitting. We tried two, solo
(swap one MLP, measure CE cost on the eval window), layers 4-9, with
registered predictions (`mlp_weight_rank.py`):

- **Input-subspace truncation** (project the input onto the top-r singular
  directions of the stacked read-weights [L;R]; control = random r-dim
  subspace): the weight directions are real -- at r=128 the random control is
  catastrophically worse (L6: 0.003 vs 0.715, and random projection there is
  worse than deleting the MLP outright) -- but prediction (b) FAILED 0/6:
  r=128 never reaches 25% of the r=16 cost. The read spectrum is BROAD; the
  middle MLPs do not read a small subspace. Prediction (a)'s 2x-at-r=64 bar
  also FAILED (3/6) -- at low rank both arms sit near the deletion floor.
- **CP truncation** (each hidden unit is a rank-1 quadratic scaled by its
  Down column; keep the top-k by `||down||*||l||*||r||`): prediction (c)
  HELD 6/6 -- k=1152 (25% of the 4608 hidden units) costs at most +0.07
  solo at every layer, recovering ~80% of the deletion cost at the worst
  layer (L4: 0.071 vs 0.357) and more elsewhere.

So the natural coordinate system of a bilinear MLP is its hidden units (a
sum of rank-1 quadratics), not an input subspace: a quarter of the terms
carry nearly the whole function, and that quarter is identified by a norm
product computed from weights alone.

## 262. v6 takes the tail attention: 27 of 36 components at +2.55

`assembled_v6.py` added attention 10-17 to the v5 assembly as dictionary
rungs, fit sequentially under the whole front: per-class output constants
for the six structure classes (the 247 material) and per-class linear reads
of the substituted stream for the contextual classes (newline, subword,
induction, other). Labels: oracle arm (target-token classifier) and a
deployable arm whose labels come from a probe fit at attn10's input
(accuracy 0.71 -- better than the 0.59 tail-stream probe).

Result: **v6 full = +2.548 oracle, +2.605 deployable, with 27/36 components
replaced.** ALL EIGHT attention rungs survived the pre-registered drop rule
-- leave-one-out marginals 0.04-0.13 each (a10 +0.05, a11 +0.06, a12 +0.05,
a13 +0.10, a14 +0.12, a15 +0.04, a16 +0.13, a17 +0.06). Predictions (a)
>=4/8 survive and (b) total <= +3.0 HELD; (c) FAILED -- a14, the tail
attention component whose deletion IMPROVES average text, still costs
+0.117 to replace with its dictionary, above the registered +0.10.
Harmful-on-average does not mean free-to-replace: in the assembled context
a14's class-conditional behavior carries real function. The deploy-oracle
gap is only +0.06 -- the label probe is no longer the bottleneck.

Frontier: 13@+1.39, 19@+2.10, **27@+2.55**.

## 263. Absorber variants: quadratic features pay, the interface subspace does not

`absorber_variants.py` re-ran the full v5 pipeline four times, changing only
how each MLP rung's absorber is built. Baseline reproduced (+2.0954,
prediction (a) HELD). The other two arms tested the two theory ideas:

- **QUAD** -- since the table residual is exactly quadratic in the input,
  add the 136 pairwise products of the layer's own top-16 weight directions
  as absorber features: total **+1.993**, better than baseline by 0.10, and
  every absorber R^2 rose (+0.03-0.06). Registered bar <= +1.95 FAILED by
  0.04 -- the direction is right, the effect at 16 directions too small.
- **READ** -- choose the absorber basis inside the 64-dim downstream-read
  subspace (summed trace-normalized Grams of all later layers' input
  weights + unembedding): total **+2.240 -- worse than baseline** -- and the
  random-subspace control (+2.261) is only 0.02 behind it. Prediction (c)
  FAILED cleanly: restricting correction to a low-dim "interface" hurts.

The READ failure and 261's subspace failure are the same fact measured two
ways: **bilin18's middle interfaces are not low-dimensional.** Downstream
layers read broadly, so no 64-dim summary of "what the wiring looks at"
captures where substitution error matters. The connection-SVD idea, in this
form, is refuted for this model; the exploitable weight structure is the
CP/hidden-unit axis (261) and the quadratic form itself (QUAD arm).

Registered next: v7 replaces the six middle table+absorber rungs with the
weights-only CP-truncated MLPs (k=1152; zero fitted parameters); prediction
(a) v7 <= +1.85 total, (b) middle in-context LOO sum <= +0.35 (half of
v5's +0.71, since faithful conductors should compose better than lossy
firewalls when their solo cost is this low), (c) k=2304 total also
reported. And quad-capacity ladder: 16/32/48 weight directions.

## 264. v7: the weights-only middle -- 19 components at +1.68

`assembled_v7.py` swapped the six middle table+absorber rungs for the
CP-truncated real MLPs of 261 (top hidden units by norm product, computed
from weights alone, zero fitted parameters). Registered bars: k=1152 total
<= +1.85 FAILED (+1.912 -- beats v5's +2.095 by 0.18 but not by the
registered margin); middle LOO sum <= +0.35 FAILED (+0.703, same sum as
v5's firewalls but differently distributed); k=2304 <= k=1152 HELD, and
decisively: **+1.678 total -- the new 19-component frontier point**, with
the whole middle described by "keep the loudest half of each layer's
rank-1 quadratics."

Two economics lessons in the marginals. First, conductor cost is
layer-specific: CP-mlp4/5 cost +0.26/+0.29 in-context (worse than their
firewalls) while CP-mlp6-9 cost +0.06/+0.11/-0.04/+0.03 (better). Early
middle layers amplify conducted upstream error; late middle layers
tolerate it. Second, **CP-mlp8's marginal is NEGATIVE**: with the
substituted front, the truncated mlp8 outperforms the real mlp8. The
truncation drops exactly the quiet units that react worst to
off-distribution inputs -- a lossy stand-in acting as a partial firewall.
The firewall/conductor distinction is a spectrum, not a binary.

Frontier: 13@+1.39, 19@+1.68 (weights-only middle, k=2304), 27@+2.55.

## 265. Quad ladder: the write-side rank binds, feature count saturates

`quad_ladder.py`: quad32 = +1.970 (prediction <= +1.94 FAILED -- doubling
the weight directions bought only 0.023 over quad16's +1.993); quad48 =
+1.973 (saturated, no blowup, HELD); quad32 with a rank-64 residual basis
= **+1.934** (HELD, the biggest single lever). So the absorbers are not
starved of input features -- 16-32 weight-derived quadratic directions
suffice -- they are starved of OUTPUT rank: the rank-32 basis P cannot
express enough of the residual. Consistent with 261/263: the middle's
interfaces are broad, so low-rank corrections saturate early on the write
side, not the read side.

Registered next (v8): merge the winners -- quad32/P64 absorbers on the
front tables, CP k=2304 middle, v6's attention dictionaries, tail.
Predictions: (a) v8 oracle <= +2.20 at 27/36; (b) deploy gap <= +0.10;
(c) all 8 attention rungs survive the drop rule. And cp_controls to make
261 airtight: random-k and bottom-complement nulls for the unit ranking,
plus a finer k ladder.

## 266. v8 merges the winners: 27 of 36 at +2.21

`assembled_v8.py` (quad32/P64 absorbers on the front tables, weights-only
CP-2304 middle, class-dictionary tail attention, tail MLP dictionary):
**+2.2121 oracle, +2.2544 deployable** -- the 27-component frontier drops
from +2.55 to +2.21. Registered (a) <= +2.20 FAILED by 0.012 (recorded as
failed; the direction is confirmed, the bar missed by a hair). (b) deploy
gap +0.042 <= 0.10 HELD (probe at a10 input: 0.73). (c) all 8 attention
rungs survive HELD -- but their marginals grew vs v6 (a14 +0.26, a16
+0.31, vs +0.12/+0.13 on the stronger v6 substrate): the cheaper the
middle stand-ins, the more the attention dictionaries cost in context.
Substitution costs are not additive across rungs; they share an error
budget.

## 267. LEDGER 21: the CP ranking was never doing anything -- redundancy was

`cp_controls.py` ran the selection nulls that 261 lacked, and both
registered nulls came back the WRONG way: random-1152 matches top-1152
(0/6 at the 2x bar; at L6 and L9 the random subset is BETTER than the
model, cost negative), and the bottom-3456 beats the top-1152 at all six
layers (0/6), also sometimes beating the full model. Only (c) held
(top-576 <= +0.15 everywhere -- but so would random-576, presumably).

**Correction (ledger 21):** 261's claim that the norm product
||down||*||l||*||r|| "identifies the important quarter" is withdrawn. The
truth is stronger and less flattering to the instrument: the middle MLPs
are so redundant that nearly ANY large-enough subset of hidden units
reproduces them solo, and deleting the loudest units can help -- the
loud tail contains slack (consistent with the deletion-improves anatomy
of the flat track). The CP-truncation FRONTIER RESULT (264's 19@+1.68)
stands -- the stand-in works -- but its mechanism is redundancy, not
weight-readable importance. Rule added to the ledger: **a selection rule
is not a finding until its selection null runs; "top-k works" is
meaningless without "random-k fails."** Open question queued: does the
ranking matter IN ASSEMBLY (where costs are 3-10x larger)? v7-rand
re-runs v7 with random-2304 middles, 3 seeds.

## 268. The in-assembly selection null: ranking buys reliability, not quality

`v7_rand_control.py` (registered branch decision): top-ranked middle
+1.692 (replicates 264); random-unit middles +1.646 / +2.034 / +1.719
across three seeds. Neither registered branch: one random seed BEATS the
ranking, one blows up by +0.34, spread 0.388. Refining ledger 21: the
norm-product ranking does not find a better-than-random subset (seed 0
beat it) -- it finds a reliably-adequate one. Selection value here is
VARIANCE REDUCTION. For the benchmark frontier the honest statement is:
"any half of the hidden units, checked once, works."

## 269. v9 takes the middle attention -- and the induction prediction fails

`assembled_v9.py`: attention 2-9 added as class-dictionary rungs, fit in
true block order. v9 full (all 16 attention rungs + everything else) =
+3.268 oracle; the drop rule removed only a8 (marginal +0.4002, a
whisker over the bar -- and a8 is the DIGIT owner, whose function is
genuinely site-conditional); **v9-best = 34 of 36 components at +2.868
oracle / +3.067 deploy** (only the pattern sides of attn0/attn1 remain
real, plus a8).

The registered mechanistic prediction FAILED in the most informative way:
the induction band (a3-a5) was predicted most expensive, but the top
costs were a8 (+0.40), a7 (+0.28), a6 (+0.23), and **a5's marginal is
NEGATIVE (-0.38)** -- replacing the certified induction owner attn5 with a
class dictionary makes the assembly substantially BETTER. Reading: on a
stream already carrying substitution error, the real induction machinery
misfires (matches the wrong antecedents) and injects noise; the
dictionary at least injects the class-typical signal. Same lesson as
CP-mlp8 (264) and the a14 result (262), now at its strongest: in-context
replaceability is a property of the ASSEMBLY, not the component. The
certified circuit facts (attn3-5 own induction on the REAL model) are
untouched; what failed is transferring them to the substituted context.

## 270. P-rank ladder: monotone, sublinear, not saturated

quad32 absorber totals by write-rank P: 32 -> +1.970, 64 -> +1.934,
96 -> +1.920, 128 -> +1.899. Registered (a) P96 <= +1.91 FAILED by 0.01;
(b) monotone HELD. Gains halve per step; the residual the absorbers
chase has a long flat spectrum -- consistent with the broad-interface
picture (263).

Frontier after this wave: 13@+1.39, 19@+1.68, 27@+2.21, **34@+2.87**.

## 271. The a5 anomaly resolved: some components are scaffolding, and the assembly wants their MEAN

`a5_anomaly.py`, four arms in the v9-best context: real attn5 +3.466,
class dictionary +2.868, **global mean output +2.722**, zero +4.746.
Registered (a) HELD (dict beats real by 0.6); (b) FAILED -- zeroing is
catastrophic, so attn5 is NOT slack-in-context; (c) FAILED -- the
class-blind mean BEATS the class dictionary. Reading: the assembly needs
attn5's average contribution (scaffolding -- a bias the downstream layers
are calibrated against), while its input-dependent deviations misfire on
substituted streams and its class-conditional structure adds nothing
there. The cheapest possible stand-in -- ONE VECTOR -- is the best one
found for this component. Follow-on running: greedy dict-vs-mean
selection over all 16 attention rungs (registered: >=2 prefer mean; a8
rescued as a mean; final <= +2.60).

v10 (P128 front absorbers, a8 pre-dropped): +2.853 oracle / +3.052
deploy -- only 0.015 under v9-best; registered <= +2.75 FAILED. The P128
gains measured on the 19-comp assembly (270) do not transfer to the
34-comp context; front absorber error is no longer the binding
constraint once the attention rungs are in.

## 272. Greedy stand-in selection: five attention components are scaffolding

`standin_select.py` swept all 16 attention rungs front-to-back, swapping
each class dictionary for its plain mean vector when that improved the
total. Five swaps adopted -- a2, a5, a9, a14, a16 -- landing at
**+2.7794 oracle / +2.9807 deploy at 34/36** (new frontier; v10 +2.853).
Registered (b) HELD (>=2 swaps); (a) FAILED (<=2.60 not reached); (c)
FAILED (a8 as a mean does not rescue the dropped rung -- the digit
owner's function really is input-dependent). The scaffolding phenomenon
(271) generalizes: for five of sixteen attention components, the best
known description in the assembled context is ONE VECTOR -- the
downstream layers need their average contribution and nothing else that
our dictionaries can currently supply. The description census after this
pass: 5 attention means, 10 attention class-dictionaries, 1 attention
dropped (a8), 2 attention value-tables (a0/a1), 4 front MLP
table+absorbers, 1 MLP linear (m1), 6 middle MLP weights-truncations,
8 tail MLP span dictionaries.

Module-relevance instrument (`module_relevance.py`, for the benchmark
figures): mean-ablation cost per component, all 36, plus the ceiling --
the embeddings-only model sits +11.95 nats above base, anchoring "0% of
the model's work." Current frontier retains 77% of the model's work with
34/36 components replaced; v4 retains 88% at 13/36.

## 273. Round-2 greedy: mlp8 joins the scaffolds -- 34/36 at +2.749

`standin_select2.py` extended the greedy sweep to the middle MLPs and
re-tried everything in the new context. One further swap adopted: **c8 ->
mean** -- the CP-truncated mlp8 (whose LOO marginal was already negative
in 264) is best described by a single vector in the assembled context.
Every other rung kept its richer stand-in (16 keeps), the greedy
converged, and the a8 rescue failed again (+3.57 -- its function is
genuinely input-dependent). Frontier: **34/36 at +2.7486 oracle /
+2.9526 deploy**. Registered (a) HELD, (b) FAILED (<= 2.70 not reached).
The census now: 6 attention/MLP means, 9 attention class dictionaries,
2 value tables, 4 front table+absorber MLPs, 1 linear MLP, 5 middle
CP truncations, 8 tail span dictionaries, 1 dropped rung (a8).

## 274. The front is solved: in-assembly marginals for the priced-out rungs

`front_tail_marginals.py` priced the rungs the figures were missing, LOO
in the round-2 best config (+2.7486): a0 +0.0000 (exact by
construction), m0 +0.0095, a1v +0.0070, **m1 +0.0060**, m2 +0.0142, m3
+0.0036; tail spans m10-m15 +0.011-0.019, m16 +0.031, m17 +0.082. All
three registered bars HELD, and (a) held with two orders of magnitude to
spare: the model's most important component (mlp1, solo relevance 5.32
nats) costs SIX THOUSANDTHS of a nat to replace in the assembly -- 99.9%
of its work captured by the token table + quadratic absorber. The entire
front (a0,m0,a1v,m1,m2,m3 -- which carries ~80% of total module
relevance) costs +0.046 combined. The assembly's remaining +2.75 nats
live almost entirely in the middle-attention dictionaries and the CP
middles' conducted error. Also: `deploy_probe2.py` FAILED both bars --
adding a current-token class prior moves probe accuracy only 0.686 ->
0.701 and deploy CE not at all; the class of the NEXT token is not
readable from the current token id, so the oracle/deploy gap (~0.2)
stays owned by the stream probe.

Addendum (a8 rescue refuted): `a8_digit_linear.py` -- adding a8 back as a
dictionary with a digit-class linear arm costs +0.62 in the current
config, indistinguishable from the plain dict (+0.617). Both bars
FAILED. attn8's digit function (contextual counting) is not a
same-position read in any form we have; it joins induction as
cross-position transport. a8 remains the one honestly-dead rung.

Correction to 274 (caught by user question): m1's in-assembly stand-in is
the fitted LINEAR map (the assembly's one linear rung, from v3 onward) --
NOT the token table. The token-table result for mlp1 (79% recoverable
from a weights-computed lookup, section 256) is its SOLO fold result. The
+0.006 marginal belongs to the linear stand-in. The table+absorber rungs
in the assembly are m0, m2, m3. The interpretive picture is consistent --
at layer 1 the stream is still nearly token-determined, so "linear map on
a token-determined input" and "token table" nearly coincide -- but the
census entry was mislabeled and the two descriptions have different
fitted-bits prices.

## 275. Linear arms are not low-rank; a8 stays dead

`linarm_rank.py`: truncating all dictionary linear arms + m1's map to
rank 128 costs +0.087 (bar +0.02 FAILED); rank 32 +0.138 (FAILED); rank
8 +0.20. The fitted-bits hogs cannot be shrunk for free -- the
contextual-class arms carry genuinely high-rank transport, consistent
with the broad-interface findings (263). The fitted-bits census stands
at its honest (large) value for those slots.

## 276. LEDGER 22: the eval window flattered us -- fresh-data audit

User asked whether results survive more data. `fresh_replication.py`
built a 180-row window from never-seen pile documents (dedup-checked
against FW) plus the saved fineweb window, and re-scored everything.

- **best-34 on fresh pile: +3.114 oracle / +3.208 deploy** vs +2.749 /
  +2.953 on the standard eval window. Registered +-0.30 band FAILED by
  0.07. The standard eval window (FW rows 120-300) comes from the same
  512-row corpus sample as the fit window (rows 300-512); table means and
  class constants transfer better within that sample than to fresh text.
  All frontier numbers are eval-window-optimistic by roughly +0.35.
- **Only 3 of 6 greedy mean-swaps replicate** (a5, a14, a16). Reverting
  a2 IMPROVES fresh CE by 0.163; a9 by 0.035; c8 by 0.011. The greedy's
  0.005 adoption threshold was fitting eval-window noise/particulars.
- a8 re-add: +0.249 on fresh (direction replicates, size halves; the
  >=0.3 bar FAILED).
- **All-means baseline: +7.04** (HELD >= 6) -- a 36-vector model retains
  only ~41% of the model's work vs the assembly's ~74% (fresh-honest
  number). The assembly content is real.
- Fineweb (out-of-domain): base 3.077, best-34 +3.449 -- degrades but
  works; the dictionaries are corpus-tilted, not corpus-locked.

**LEDGER 22:** a fit/eval split WITHIN one corpus sample does not control
sample-level correlation; greedy selection on a fixed eval window is
itself a fit. New standing rules: (i) frontier numbers are quoted with
their fresh-window value or not at all; (ii) any selection procedure
(greedy swaps, drop rules) must validate its choices on a window outside
the sample, with a third window for the final quote. The honest frontier
today: **34/36 at ~+3.11 oracle on fresh text (74% of the model's work),
all-means floor +7.04, ceiling +11.95.**

## 277. Validated selection: the honest frontier is 34/36 at +2.93 fresh

`select_validated.py` re-ran the greedy under ledger 22's rules: adopt a
swap only if it improves BOTH the standard window and a fresh-pile
selection half; quote on an untouched fresh validation half. It kept
exactly the three fresh-replicating swaps (a5, a14, a16 -- prediction (a)
HELD) and the quote is **+2.9252 oracle / +3.0172 deploy on data no
selection ever touched** (FR-sel agreement within 0.06 -- no leakage;
all bars HELD). Notably the validated config also beats the old greedy's
config on the standard window (+2.6345 vs +2.7486): round-1's extra
swaps were not just non-replicating, they were path-dependent noise
fits. THE quotable benchmark number: 34/36 components, +2.93 nats on
fresh text = 75% of the model's work (ceiling +11.95, all-means floor
+7.04, embeddings-only 0%).

## 278. Grounding the tables; the motif repertoire question

`table_semantics.py` (grounding the m0 token table): fold-table rows
(computed from weights, length-1 forwards, zero data) match empirical
rows at median cosine 0.917 -- but the registered shuffled-null bar
(<=0.05) FAILED for instrument reasons: unshuffled rows share a large
common mean direction, putting the null at 0.628. Signal over null is
real (+0.29) but the instrument needs mean-centering before the claim is
quotable. The semantic check HELD: table rows organize by token type
(within-class vs between-class cosine gap 12x the shuffled-label null;
digit rows tightest at 0.839). The table is a derived, semantically
organized object, not a memorized blob -- with the centered-null rerun
owed.

Motif repertoire (user direction): the flat track already certified the
MLP-side version -- section 58's shared functional basis, re-verified
this session: ~80 shared quadratic-form directions reconstruct any
reader's coupling matrix at leave-one-reader-out R^2 0.71 (random basis
~0; 8 directions get only 0.15). So bilinear layers DO draw from a
common mid-sized repertoire of interaction shapes, individually bound.
The counterweight is sections 219-221: the readers' OUTPUT codes are
idiosyncratic dialects -- shared shapes, private bindings, exactly the
"same function, different semantics" split the user proposed.
`attn_motifs.py` (running) does the attention side: bucket every head's
pattern mass into self/first/induction-target/match/offset motifs,
census the repertoire across all 162 heads, with token-shuffle nulls.

## 279. Attention motif census: prev-token and self ARE cross-layer motifs; induction hides below the argmax bar

Two instrument iterations (`attn_motifs.py`, `attn_motifs2.py`). v1
bucketed absolute pattern MASS and degenerated (155/162 heads 'other'):
in unnormalized squared attention the magnitude lives in a diffuse tail,
so mass fractions measure the noise floor. v2 bucketed the ARGMAX (where
each head looks hardest, per query) and found real repertoire structure:
**prev-token: 27 heads across 11 layers; self: 51 heads across 16
layers**; first-token: 2 heads (layer 5); diffuse: 82. The user's
conjecture is confirmed for these motifs -- "look back one token" is the
model's most repeated named pattern-function, appearing from layer 0 to
layer 17 with (per the dialect results) different bindings each time.
Registered bars still FAILED as written: only 2 named families cleared
the >=5-heads multi-layer bar (needed 3), and ZERO induction-target
heads appeared -- despite certified induction ownership at attn3-5.
Suspected dilution: induction-eligible queries (those with a real
earlier match) are a minority of positions, so a genuine induction head's
unconditional argmax fraction stays under 0.25. v3 (queued) scores
ind/match conditionally on eligible queries; registered: at least one
conditional-ind head in layers 3-5, else the pattern-level and
ownership-level pictures are in genuine tension.

## 280. The motif repertoire is real: conditioning finds the induction heads; the table grounding goes quotable

`attn_motifs3.py` (eligibility-conditioned): **9 induction-target heads
across layers 1-8** (top: L2h5 at 0.74, L3h8 at 0.63, L5h5 at 0.60
conditional argmax fraction), against a shuffle null of 0.042 vs real
0.531 -- a 12x margin. All three bars HELD: 3 of 9 sit in the certified
owner layers 3-5, and prev/self stay multi-layer. v2's zero was pure
dilution, as suspected. Note the spread: pattern-level induction
capability exists at seven layers while causal ownership concentrates at
3-5 -- redundancy at the pattern level mirroring the atlas finding that
induction is replicable but unconcentrated. Final census: self 47 heads
(16 layers), prev 27 (11 layers), ind 9 (7 layers), first 2, diffuse 77.
The attention pattern repertoire is FOUR named motifs covering 85 of 162
heads.

`table_semantics2.py` (centered instrument): fold-vs-empirical median
row cosine **0.841 with shuffled null -0.02** -- the m0 token table is
genuinely weights-derivable row by row (v1's 0.917/0.628 was mean-
contaminated; this is the quotable form). Class structure survives
centering (gap 0.083, null 0.000; digit rows tightest at 0.499). Both
grounding claims from the bits-metric discussion now stand on clean
instruments.

Registered next: exploit the census -- replace prev-motif heads'
patterns with the LITERAL one-hot previous-token pattern (one fitted
gain per head), and self-motif heads with one-hot self. If it holds,
the pattern side of 74 heads compresses to two sentences plus 74
numbers.

## 281. Motif pattern swap v1: both bars FAILED, and the control indicts the instrument

`motif_pattern_swap.py` replaced the pattern side of the 74 named-motif
heads with literal one-hot patterns (one fitted gain each): prev-swap
+0.385, self-swap +0.141, both +1.283 -- super-additive, the same
error-budget interaction the assembly shows everywhere. Registered (a)
<= +0.15 FAILED decisively. But the control is the real finding: giving
27 RANDOM non-prev heads the prev-one-hot pattern IMPROVED CE by 0.038,
so (b) failed vacuously and the instrument is under suspicion -- either
my recomputed-attention path (fp32 pattern recompute + c_proj replay)
does not reproduce the model's own arithmetic (a reconstruction null was
missing, an instrument-design error), or head patterns carry so much
slack that ANY tame pattern helps (consistent with the slack anatomy,
but unprovable without the null). v2 (queued) adds the reconstruction
null (replay with REAL patterns must cost ~0), per-layer recon offsets,
and a per-head greedy: which individual heads accept their motif
sentence for free? Registered there: |recon| <= 0.02; >=30/74 heads
swap at <= +0.01 corrected; adopted set <= +0.10 total.

## 282. Motif sentences are individually true, jointly conductive

`motif_swap2.py`: the reconstruction null is EXACTLY 0.0000 (the v1
instrument was valid all along -- v1's "control improved CE" was real
pattern slack: handing random heads a tame one-hot pattern slightly
helps). Per-head greedy: **71 of 74 named-motif heads accept their
literal motif pattern at <= +0.01 nats each** (prediction (b) HELD,
71 >> 30). But the adopted set jointly costs +0.451 (bar +0.10 FAILED):
the swaps are individually free and collectively expensive -- pattern
substitution errors conduct and compound exactly like the assembly's
component errors. The motif census is thus VALIDATED as description
("head L4h2 is a previous-token head" is true and per-head causally
cheap) while whole-repertoire replacement needs the same composition
machinery (sequential fitting, absorbers) the MLP side needed. The
pattern side now has its grammar problem, with a known playbook.

## 283. The fresh-window frontier ladder (the honest graph)

`fresh_frontier.py`, every rung on the untouched validation half:
front +1.984 (bar 1.90 FAILED by 0.08), +middle +2.021, +tail-attention
+2.485, validated-34 +2.925; monotone HELD. The striking number: the
CP-truncated middle costs only **+0.037 on fresh text** -- the weights-
only middle stand-ins are essentially free out-of-sample (they are
weights-derived, so they cannot overfit a window). The fitted rungs
(front tables +1.98, attention dictionaries +0.46 and +0.44 per band)
carry all the fresh-window cost. Weights-derived descriptions travel;
fitted descriptions pay a transfer tax.

## 284. The double QK circuit is a coincidence sharpener, not a selector-gate

`qk_factor.py` tested the registered factorization hypothesis (one score
set = positional selector, the other = content gate): **0 of 27 prev
heads factor; all three bars FAILED** -- and the per-head data says why,
which is the actual finding. For prev-token heads BOTH score sets peak
at offset-1 (s1 fractions 0.3-0.6, s2 0.3-0.8), each with top-1
concentration of only ~0.10; for induction heads both sets carry the
conditional induction signal (0.2-0.7 on both sides). The product of
two weak, broad versions of the SAME preference yields the sharp
argmax behavior of the census: bilin18's two QK circuits implement
soft-AND agreement -- coincidence detection that suppresses accidents
either set alone would make -- rather than a division of labor. This is
a genuine motif-algebra fact: the pattern quartic
(x_q^T A x_j)(x_q^T B x_j) is used with A ~ B in function, i.e. the
model squares its evidence. (Prediction failed, discovery recorded;
the selector-gate design remains available to models but this one does
not use it.)

## 285. OV has no shared library; the block handoff motif is front-loaded

`ov_motifs.py`: leave-one-head-out r=256 read energy 0.284 vs random
control 0.222; write side 0.288 vs 0.222; mean pairwise subspace overlap
0.116 vs the 0.111 random floor. **Both shared-library predictions
FAILED cleanly: the 162 heads' OV subspaces are spread essentially as
far apart as geometry allows.** The private-bindings prediction HELD
(within-family/across ratios: prev 1.13, self 1.01, ind 1.26). Combined
with 280/284: bilin18's attention repertoire is a PATTERN-side
phenomenon -- four repeated ways of choosing where to look -- while what
each head reads and writes is unshared, per-head content. Shared
function, private bindings, now measured on weights.

`block_motif.py` (the 18x18 wiring map): same-block coupling
(attn_i write subspace -> mlp_i read subspace) averages 0.117 vs
cross-block 0.063 vs random floor 0.056 -- about 2x -- but only 10/18
blocks clear the registered 1.3x bar (FAILED as registered): the
handoff motif is STRONG in the front half and fades to cross-block
levels in the tail (attn13/14/16 own-couplings 0.036-0.056), exactly
where attention is class-constant and the MLPs are span dictionaries.
attn0's broadcast row >= cross mean (positive control HELD); next-block
handoff loses to same-block everywhere. The composed unit "attention
selects, adjacent MLP transforms" is a repeated circuit motif of the
model's FRONT, not of the model.

## 286. Two instrument failures, recorded

`handoff_causal.py` v1 subtracted raw attention output from the MLP's
already-normalized input -- wrong interface, wrong scale: every arm
including the random control cost ~10-12 nats (the all-means range).
Void. The block computes mlp(rms_norm(x_mixed + attn_out)), so the
correct counterfactual is mlp(rms_norm(x_mixed)); v2 does that with a
norm-matched control at the same interface. `decomp_census.py` v1
crashed on Down's orientation and, worse, gave low-rank 2.5x the budget
on non-square matrices (rank set for square shapes) -- its one printed
result (Right: sparse 18/18) survives a fortiori (sparse won even
against over-budgeted low-rank) but the census is rerun at fair
budgets. Standing rule reinforced: scale/orientation sanity checks
(reconstruction null, budget audit) BEFORE the science bars.

## 287. Three critical handoffs; sparse sweeps the census (null owed)

`handoff_causal2.py` (corrected interface): the attn->own-mlp handoff is
causally concentrated in THREE junctions -- **attn0->mlp0 +0.83,
attn1->mlp1 +1.95, attn5->mlp5 +2.14** -- with blocks 2-4 at 0.03-0.07,
blocks 6-17 at zero (several slightly negative). Spearman with the
wiring-map diagonal 0.802 (HELD); front/tail ratio ~200x (HELD).
Prediction (c) failed only by bar construction: block 8's real cost is
-0.01 (nothing to halve); at the one substantive control site (block 2)
the random perturbation costs 11% of the real cut. The composed-block
motif is real but rarer and sharper than the subspace map suggested:
the model has three places where an MLP critically consumes its own
block's attention output -- and notably attn5->mlp5 sits exactly at the
private-writer anatomy (mlp5/6), and is invisible to the assembled
model, whose stand-ins bypass the junction entirely (a5 as a mean, m5
as a table+absorber, both cheap there). The real model and the
stand-in model achieve the same next-token behavior through genuinely
different internal traffic at this junction.

`decomp_census2.py`: at matched budget P=D^2/8, **top-P sparse beats
low-rank, block-diagonal, and diag+low-rank for all 126 weight
matrices** (7/7 families at 18/18; both registered bars HELD). Caveat
before this becomes a claim (ledger-21 pattern): sparse also beats
low-rank on iid GAUSSIAN matrices at this budget -- the census needs a
random-matrix null to show the model's sparse advantage EXCEEDS chance.
Census v3 queued with matched Gaussian baselines + per-family excess
kurtosis; plus a sparse-pattern experiment: are the hot entries
ALIGNED across layers (privileged stream coordinates -- the tail-coords
anatomy from the flat track predicts yes)?

## 288. The weights are entrywise Gaussian: structure lives in function space, not entry statistics

`decomp_census3.py` (the owed null): the sparse sweep of 287 was
entirely the Gaussian effect. Model sparse-error 0.470-0.798 vs
shape-matched Gaussian 0.498-0.800 -- within 1-5% of chance at every
family -- and entry kurtosis 0.03-0.52 (Left/Right at 0.03: as Gaussian
as matrices get). Both bars FAILED 0/7. (Instrument note: the per-matrix
"winner" column in v3's log is contaminated -- the diagnostic keys
leaked into the argmin -- but the verdict uses the explicit
sparse-vs-null comparison, which is unaffected.) `sparse_pattern.py`
agrees from the other side: top-entry masks overlap at chance (Jaccard
0.005-0.013 vs 0.005 floor; only c_proj marginally above), and
cross-family hot-coordinate correlation is 0 of 21 pairs. No privileged
stream coordinates exist at the raw-weight level.

Verdict, closing the decomposition-type direction at scope: **bilin18's
weight matrices are statistically featureless entrywise.** Every
structure this program has found -- the rank-80 functional basis, the
four pattern motifs, the three critical handoffs, the class
dictionaries, the token tables -- lives in SUBSPACES, FUNCTIONS, and
BEHAVIOR, never in entry statistics. This is 208's lesson (sharing is
behavioral, not elementwise) at full generality: elementwise views of
this model are noise; function-space views carry everything.

## 289. The big handoff is four directions wide; induction refuses linear reduction a third time

`handoff5_dissect.py`: cutting only the top-4 PCA directions of attn5's
contribution at mlp5's input costs +1.63 of the full cut's +2.14 (76%);
top-16 costs +2.09 (98%); sixteen RANDOM directions cost +0.0007.
Predictions (a),(b) HELD decisively ((c) failed on a 0.05 monotonicity
jitter between k=64 and k=256). **The model's biggest causal junction
is a four-to-sixteen-direction channel** -- attn5 hands mlp5 a narrow,
concentrated signal worth two nats, at the private-writer anatomy, and
it now has a shape small enough to name. Semantic dissection queued
(logit-lens on the four directions, class correlation, overlap with
the section-212 private span).

`induction_channel.py`: the registered matched-successor low-rank read
FAILED again -- channel R^2 0.26/0.15/0.31 on the three strongest
census heads, and the SAME-POSITION control is higher (0.59/0.46/0.32),
with shuffled-match nulls deeply negative (valid instrument) and
rank-32 lossless (what little the channel captures is low-rank). This
is the third refutation of a linear reduction of induction (naive copy
x2, now the learned channel). One caveat remains -- the v-mixing with
layer-0 values was approximated -- so an exact-mixing rerun is queued
under the three-strikes rule: if it fails with exact values, the
program closes the hypothesis class at scope and records induction
transport as irreducibly the head's own bilinear computation.

## 290. The anomaly finds its upstream: the handoff channel feeds the private span

`induction_channel2.py` (exact v-mixing): unchanged -- channel R^2
0.245/0.163/0.304, same-position control higher at all three heads.
**Strike three; the hypothesis class closes at scope.** Induction
transport in bilin18 is not a linear read of any single position
(naive copy x2, learned matched-successor channel x2); the head's own
pattern-times-values computation stands as its minimal description.

`handoff5_semantics.py`: both bars HELD. All four directions of the
attn5->mlp5 channel carry function-class information (class R^2 0.20 /
0.14 / 0.31 / 0.14; shuffled nulls exactly 0.0), and the logit lens
reads as a CLAUSE-BOUNDARY / syntax axis: dir1 separates punctuation
and connectives from sentence-starters, dir2 newline-and-function-words
from math/subword symbols, dir3 clause connectives (which / because /
however / where). And the closure: the channel's overlap with mlp5's
output span is 0.408 and with mlp6's PRIVATE span 0.376, against a
0.009 random floor -- 40-50x. **The model's biggest causal junction is
the upstream source of the private-writer anatomy**: attn5 aggregates
clause-structure evidence and hands it to mlp5/6, which write the
contested 8-dim code that every reader understands in a private
dialect (219-224). The two-hundred-section anomaly arc and the circuit
arc now meet in one mechanism: gather (attn5, 4 directions) -> encode
(mlp5/6, 8-dim span) -> idiosyncratic readout (every tail layer, no
consensus). Causal link test queued: cutting the 4 directions should
collapse the span code and concentrate CE damage in clause classes.

## 291. The channel is a regulator, not a source: both link predictions failed

`handoff_span_link.py`: cutting the four channel directions does NOT
silence mlp6's span code -- it makes it EXPLODE: span-coefficient
variance rises 4.6x (+358%) while the random-4 control moves it 3.8%.
And the CE damage concentrates in the CONTEXTUAL classes (subword 27%
vs 18% position share, induction 33% vs 24%, other 46% vs 36%), while
every clause class is at or below share and newline damage is NEGATIVE
(cutting the channel helps there). Both registered predictions FAILED.

Revision to 290's story: the attn5->mlp5 channel geometrically overlaps
the span and carries class information, but functionally it REGULATES
the span code rather than sourcing it -- deprived of the channel, the
quadratic writer's code becomes high-variance noise, and the cost lands
on the classes that need stable contextual integration downstream. The
logit-lens clause reading described what the directions write toward
the vocabulary, not what their removal breaks -- a lesson on logit-lens
semantics vs causal payload. This rhymes with the slack-regularizer
anatomy that has haunted this neighborhood since section 206: the
model's biggest handoff may be its biggest stabilizer. Queued:
span-specificity check (does the explosion live in the 8-dim span
specifically, or is it generic destabilization of mlp6's output?).

## 292. The regulator is aimed: the explosion lives in the span alone

`span_specificity.py`: under the 4-direction cut, mlp6's span
coefficients inflate 4.58x while the full output moves 1.65x and a
random 8-dim probe 1.66x -- and the NEXT eight PCA directions SHRINK to
0.39x. Both bars HELD. The channel is a targeted governor: with it, the
contested code runs tame and the adjacent directions carry variance;
without it, the span thrashes and the neighbors go quiet. The private
span (212-224) now has a complete causal profile: written by mlp5/6,
carried by attn6, read in private dialects by everyone, and RANGE-
REGULATED by a four-direction channel from attn5 -- the model's largest
single handoff exists to keep its most contested code in calibration.
The regulator arc rests at earned scope; remaining thread queued: does
the assembled model (which bypasses this junction entirely) reproduce
the internal regulation, or achieve the same behavior with different
internal statistics?

## 293. Why a mean vector suffices: the regulator regulates through its average

`assembly_span_check.py`: registered "different internal traffic" and
FAILED -- the validated assembly reproduces mlp6's output statistics
almost exactly in shape (span ratio 0.70, next-8 0.65, full 0.65: a
mild uniform damping, no differential span distortion). The failure
completes the mechanism: attn5's regulatory effect on the private span
is carried by its AVERAGE contribution, which is precisely what the
assembly's mean-vector stand-in supplies. That is WHY a5's best
stand-in is one vector (271): the channel's mean is the calibration
signal; its input-dependent fluctuations are noise the substituted
context is better off without. The arc closes coherent at every level:
geometric (channel overlaps span 40-50x), causal (cut -> targeted 4.6x
explosion; random -> nothing), functional (damage in contextual
classes), and compressive (the whole junction reduces to one vector in
the assembly at near-zero cost). Fresh-window replication queued
(ledger 22) before the arc is quotable: three-junction ranking, the
targeted explosion, both on never-seen text.

## 294. The regulator arc replicates fresh -- quotable

`regulator_fresh.py` (ledger 22 gate): on 120 never-seen pile rows the
three junctions replicate exactly -- blocks 0/1/5 at +0.958/+1.952/
+2.160 versus controls at +0.053 (block 3) and +0.004 (block 9) -- and
the targeted span explosion replicates at 4.33x (standard window:
4.58x). Both bars HELD. The arc (three junctions; a four-direction
clause-marked channel; targeted range-regulation of the private span;
mean-carried function explaining the one-vector stand-in) is now
certified on fresh data end to end and goes into the published report
at this boundary. Queued comparison: is block 1's +1.95 junction also
a narrow channel (the "junction = few directions" motif repeated), or
is narrowness special to the regulator?

## 295. Block 1 is a coherence junction, not a channel

`handoff1_dissect.py`: the rank ladder INVERTS at block 1 -- cutting
only the top-4 directions of attn1's contribution costs **+4.08, more
than double the full cut's +1.95**, and the cost FALLS monotonically
toward the full cut as more directions are removed (4.08 -> 4.04 ->
3.20 -> 2.13); random-16 is free. Bars (a),(b) HELD, (c) FAILED by
inversion -- and the inversion is the finding. Block 5's junction is a
narrow additive channel (top-4 = 76% of the cost, remainder benign);
block 1's is a COHERENCE junction: mlp1, the near-token-function crown,
handles its input at two operating points -- attention fully present or
fully absent (the clean absence is close to the pure-token regime it is
79-97% equivalent to) -- but a selectively edited signal is worse than
either. The composition grammar's deepest lesson (a coherent lossy
stand-in beats an accurate partial one) reappears at the scale of a
single junction. Junction typology so far: block 5 = narrow regulator,
block 1 = coherence-critical handoff; block 0's dissection queued to
complete the trilogy.

## 296. Three junctions, three architectures

`handoff0_dissect.py`: block 0's ladder rises smoothly -- top-4 +0.006,
top-16 +0.079 (10% of the +0.831 full cut; bar (a) FAILED as
registered), top-64 +0.335, k=256 +0.977 -- no narrow carrier set at
all. The lexical handoff (attn0's input is exactly rms(wte), so this
junction is token-conditioned by construction) is DISTRIBUTED across
hundreds of directions, matching its value-table character: a
full-vocabulary object has no four-direction summary.

The junction trilogy is complete, and the answer to "is junction a
repeated motif?" is precise: the model has exactly three critical
attn->own-mlp junctions and they use THREE DIFFERENT ARCHITECTURES --
**block 0: distributed lexical broadcast** (wide, smooth ladder),
**block 1: coherence-critical handoff** (partial edits cost double the
full cut), **block 5: narrow regulator** (four directions, mean-
carried, governs the private span). The wiring motif repeats; the
implementation does not -- the same shared-function-private-binding
signature the whole program keeps finding, now at the junction level.
Fresh-shape certification queued (the three ladder SHAPES on never-seen
text; the costs themselves already replicated in 294).

## 297. Junction typology: two types certified fresh, one amended

`junction_shapes_fresh.py`: block 5's narrowness (top-4 = 87% of full)
and block 1's coherence inversion (top-4 +4.30 vs full +1.95) both
REPLICATE on never-seen text -- those two architecture labels are
certified. Block 0's "distributed" label FAILED: with a fresh-refit
basis, top-4 carries 18% and top-16 carries 52% of the full cut, versus
1%/10% on the standard window. Amendment: block 0's shape is
WINDOW-DEPENDENT -- expected in hindsight for a junction that is
token-conditioned by construction (different corpora weight different
tokens, so "which directions matter" follows the token mix). The label
drops from "distributed" to "broad, content-following" pending the
queued basis-stability test: if the window-A top-16 basis transfers to
fresh text at under half the fresh-basis effect, the concentration is
carried by window-specific token content and both measurements are
right about their own windows.

## 298. b0: variance basis is not importance basis

`b0_basis_stability.py` refused both registered branches: the two
windows' top-16 delta subspaces agree well (overlap 0.676 vs floor
0.014) -- so the directions are NOT window-specific -- yet the window-A
basis cut on fresh text costs +0.067 vs the fresh basis's +0.495, a 7x
gap. The resolution must be that the CE-carrying directions sit in the
roughly one-third of the fresh basis OUTSIDE window-A's span: variance
concentration and importance concentration are different bases at this
junction (the program's margin-not-rank / instrument-relative lessons
in yet another costume). Decisive remainder test queued: cut only the
component of the fresh basis orthogonal to window-A's span (~5 dims);
registered: it carries >= 70% of the fresh-basis cut. Either way the
junction typology stands with b0 labeled "broad, importance-unstable"
-- an honest, replicated characterization.

## 299. b0 resolved: the cost is holistic -- and the junction typology unifies

`b0_orth_remainder.py`: both sub-cuts are nearly free (orthogonal
remainder +0.008, shared component +0.066) while the joint 16-direction
cut costs +0.495 -- six times the sum of its parts. Prediction (a)
FAILED and the failure is the resolution: at block 0 the handoff cost
has NO additive decomposition at rank <= 16. Every partial cut leaves a
coherent-enough signal; only sufficiently complete removal breaks the
downstream computation, and "sufficiently complete" differs by window
(which is everything 297-298 observed). The window-dependence and the
variance-vs-importance gap were both artifacts of asking an additive
question about a holistic quantity.

Final junction typology, unified: the two FRONT junctions (b0, b1 --
both token-conditioned, both feeding table-like MLPs) are
**coherence-holistic**: cost lives in joint removal, with b0 at the
harmless end (partials free) and b1 at the catastrophic end (partials
worse than everything). Only b5, the regulator, is **additive-narrow**
(four directions, mean-carried). The arc rests at earned scope: three
junctions, two architectures, every claim either fresh-certified or
explicitly typed by the experiment that refuted its first description.

## 300. The junction anatomy is a family trait

`junctions_bilin12.py` -- all four registered bars HELD. bilin12's
handoff profile: front junctions (block 0 +0.45, block 1 +0.73), a
mid-depth ridge rising through +0.52 / +1.35 to a peak at **block 5:
+3.52**, then a cliff -- blocks 7-11 all under +0.045. And the peak
junction carries the NARROW signature (top-4 directions = 85% of the
full cut): the additive-narrow regulator architecture transports across
the family, at a mid junction 3.5x larger relative to its model than
bilin18's.

One honest structural difference: bilin18's narrow junction (block 5)
sits one block BEFORE its private writer (mlp6) and range-regulates it;
bilin12's narrow junction (block 5) sits one block AFTER its private
writer (mlp4) -- so it cannot regulate the writer's output and, if the
mechanisms correspond, should CONSUME the private code rather than
govern it. In both models the family fact is: one narrow, high-value
mid-depth junction adjacent to the private-writer anatomy, front
coherence junctions, and a causally free tail. Link test queued for
bilin12: does its 4-direction channel geometrically overlap mlp4's
span (consumption), with per-class damage reported?

## 301. One conserved complex: the private span and the narrow junction are the same structure

`b12_regulator_link.py`, both bars HELD emphatically: bilin12's
four-direction channel overlaps mlp4's span at **0.656** (random 0.012,
floor 0.0104) -- the channel essentially lives inside the private code's
subspace -- and the per-class damage profile matches bilin18's junction
signature almost class for class (subword 33% vs 18% position share,
induction 29% vs 24%, newline NEGATIVE in both models, clause classes
at or below share in both).

Family conclusion: the private span and the narrow mid-depth junction
are two views of ONE conserved structure. In bilin18 the junction sits
one block upstream of the writer and RANGE-REGULATES the code's
writing; in bilin12 it sits one block downstream and CARRIES the code
forward; in both, a ~4-direction channel adjacent to the fraction-1/3
private writer serves contextual-integration classes and is slightly
harmful at newlines. The anomaly that consumed sections 210-232 --
placed, localized, transported, dissociated -- now has its family-level
identity: a mid-depth private code with a narrow service channel,
conserved across depths with the plumbing order as the only free
parameter. Cross-model arc rests at earned scope; the sharpest next
test remains off-box (the larger checkpoint, where PREREGISTRATION.md
predicts the writer at fraction 0.33 +/- 0.08 -- to which this arc adds
the registered expectation of an adjacent narrow junction).

## 302. The contrast: junctions are general, the narrow-channel complex is bilinear

`junctions_swiglu18.py`: swiglu18 (same depth/width/data, gated) has
LARGER front junctions than bilin18 (+2.17 / +1.72 at blocks 0/1 --
prediction (a) HELD: front handoffs are architecture-general) and, against
the registered contrast (b) FAILED, a real mid junction at block 7
(+0.65, 12x its median, fraction 0.39) with a free tail. But contrast
(e) HELD: its ladder is INTERMEDIATE (top-4 = 28%, top-16 = 45%) --
nothing like the bilinear models' four-direction channels -- and it is
3-5x smaller in absolute nats.

Final scoped statement of the junction anatomy across three models:
"the block's attention feeds its own MLP critically at the front and at
one mid-depth site, never in the tail" is TRANSFORMER-GENERAL (3/3
models). What is BILINEAR-SPECIFIC is the concentration: only the
bilinear models fuse the mid junction into a ~4-direction channel
carrying a private mid-depth code -- the same
concentrate-vs-spread architectural dichotomy the flat track certified
for replaceability (204-205), now visible in the wiring. The junction
program closes with its claims scoped to the evidence: anatomy general,
implementation concentrated only where the architecture concentrates.

## 303. The motif census pays: head-level hybrid beats the dictionary band by 0.27

`head_hybrid.py`, both bars HELD with a bonus sign: leaving attention
2-9 real and swapping the 38 named prev/self heads for their one-hot
patterns costs +2.3644 on window C -- 0.27 BETTER than the v9-best
all-dictionary band (+2.6345) -- and the motif swaps' joint marginal is
NEGATIVE (-0.078 vs band-real +2.4419): in the substituted context the
literal patterns are better than the real heads, the pattern-slack
effect from 282's control now working for us. New census at this
config: 26 full components + 38 of 72 attention heads in layers 2-9
described by one sentence and one gain each, with induction, first,
and diffuse heads honestly real. Fresh certification queued (ledger
22) before this replaces the quotable frontier; a8_symbolic requeued
after a Unicode-digit crash (superscript characters pass isdigit but
not int -- isdecimal is the correct predicate).

## 304. Fresh frontier: 26 components + 38 heads at +2.54; a8 closes irreducible

`head_hybrid_fresh.py`: the hybrid certifies on never-seen text at
**+2.5432** -- 0.32 better than the all-dictionary config's +2.866 --
bars (a),(b) HELD. Bar (c) FAILED honestly: the motif swaps' marginal
is +0.124 on fresh (window C had it negative at -0.078; the "swaps
help" sign was partly window luck, though 38 heads for an eighth of a
nat remains cheap). The quotable frontier is now **26 full components
plus 38 of 72 middle-attention heads, at +2.54 fresh** (79% of the
model's work), with per-head descriptions of one sentence and one gain.

`a8_symbolic.py`: the counting-feature rescue is REFUTED with unusual
cleanness -- the symbolic arm (+3.3707) is indistinguishable from the
stream-only arm (+3.3704); giving the digit-class read the exact
symbolic count changes nothing. attn8's in-context function is not
"a linear read that lacks the count"; it is irreducible to any
same-position map we have tried (constant, linear, symbolic-augmented
linear). a8 joins induction transport in the closed class: components
whose own computation is their minimal description. Next backlog rung
queued: the front transfer tax (+1.98 of the fresh total) attacked
with weights-derived FOLD tables, which travel tax-free.

## 305. Fold tables strictly dominate: the transfer tax was avoidable all along

`fold_front.py`, both bars HELD with margin: replacing the empirical
token tables (m0, m2, m3) with FOLD tables -- every vocabulary token
run alone through the real prefix, zero training data -- is better on
BOTH windows: window C +1.390 vs +1.481 (the weights-derived table
beats the fitted one even in-distribution, no trade), fresh +1.602 vs
+1.926 (a third of a nat of transfer tax gone). The 283 principle
("weights-derived stand-ins travel; fitted ones pay") is not just a
diagnosis but an optimization: the empirical tables were memorizing
window-A token statistics that the fold construction gets from the
weights, generalization included. Registered merge queued: fold tables
into the head-hybrid frontier config -- predicted new frontier <= +2.35
fresh (from +2.543).

## 306. The merge fails by our own grammar -- and gets rebuilt correctly

`hybrid_fold.py`: hot-swapping fold tables into the finished hybrid
cost +3.76 on both windows -- worse than either ingredient. The failure
is diagnostic, not mysterious: every downstream fitted piece (attention
dictionaries, tail spans, mean gains) was fit SEQUENTIALLY under the
empirical front's stream; swapping the front afterward violates the
composition grammar's first rule (257: fit each stand-in under its
actual upstream context), and the conducted mismatch compounds.
Instrument error of the self-inflicted kind, recorded. The corrected
merge -- fold tables installed inside the sequential fit chain, so the
entire downstream stack is fit under the fold front -- is queued with
the same registered bars (<= +2.35 fresh; beats +2.543 by >= 0.15).

## 307. Sequential is not enough: fits must match the eval context

`hybrid_fold2.py` (fold tables fit in-chain): +4.77, worse than the
hot-swap. Diagnosis sharpens 306: the v9 fit chain interleaves the
attention-dictionary fits, so the front absorbers are fit under
a2/a3-DICTIONARY streams -- but the hybrid config runs that band REAL.
The empirical hybrid tolerated this context mismatch (+2.54) because
empirical tables leave small residuals and their absorbers barely
matter; fold tables leave larger residuals, their rank-64 quadratic
absorbers carry real load, and a quadratic read misfiring off-context
is expensive. Grammar rule upgraded: SEQUENTIAL FITTING MEANS FITTING
UNDER THE EVAL CONFIGURATION'S OWN CONTEXT -- in-order fits under a
different downstream plan still conduct. The matched-context merge
(front fold fits with attention real, CP middles, tail refit under the
same, motif gains last, and an empirical-table twin refit identically
as the controlled comparison) is queued.

## 308. The grammar pays: 20 components at +1.46 fresh

`hybrid_fold3.py`, both bars HELD with margin. Built entirely under the
upgraded rule (fit every rung in the eval configuration's own context:
attention real throughout, tail refit under the exact stack), the fold
arm reads **+1.4565 fresh / +1.2505 window C at 20 components** (fold
front m0/m2/m3 + a1v + m1 + CP middles + tail spans). The matched
empirical twin reads +1.8613 fresh -- the fold advantage (0.40) not
only survives composition, it exceeds its front-only size. Context:
the old fresh ladder had the 20-component rung at +2.02, and the full
34-component frontier at +2.93 -- this construction beats the latter by
1.47 nats with 14 fewer components replaced. Two lessons compounded:
weights-derived tables travel (305), and context-matched refits recover
what mismatched chains leak (307). Queued: the matched-context ladder
back up -- +38 motif heads, then tail-attention dictionaries refit
under the stack, then the middle-attention band -- each rung fit under
its own eval context, with registered bars per stage.

## 309. Stream fidelity, not CE, is the currency of composition

`matched_ladder.py`, both bars FAILED, and the numbers teach the
economics: on the fold-front base (20 comps, +1.46 fresh), adding the
38 motif-head gains costs **+1.04** -- the identical swaps cost +0.12
on the empirical-front substrate. Adding the tail-attention
dictionaries costs +0.34 more, landing L2 at +2.84 fresh at ~28
components: the fold advantage (0.40 at the base) is fully consumed by
the higher price every later rung pays. Mechanism hypothesis, now
registered for direct test: the fold tables are CE-CHEAP but
STREAM-UNFAITHFUL (context-free rows cannot track in-context
variation), and downstream substitutions are priced by the fidelity of
the stream they read, not by the CE of the stand-ins that produced it.
If confirmed, the benchmark's high-coverage frontier is governed by a
quantity none of our per-rung metrics measured: intermediate stream
MSE. The coverage-fidelity curve is now visibly a genuine trade-off
with a mechanism, not an engineering backlog: **20 comps @ +1.46 /
28 @ +2.84 / 34 @ +2.93 fresh**, each point built under matched
grammar.

## 310. Pricing is local: rungs pay for stream error where they read

`stream_fidelity.py` REFUTED the registered global-fidelity hypothesis
-- fold streams are BETTER than the empirical twin's at blocks 10 and
14 (rel-MSE 1.10 vs 1.30, 0.66 vs 0.81) and worse only at block 7
(2.00 vs 1.69). But the pattern of costs matches a sharper account:
the motif heads read blocks 2-9, exactly where fold streams are worst,
and cost 1.04 there; the tail-attention dictionaries read blocks
10-17, where fold streams are best, and cost LESS on the fold base
(0.34) than historically (~0.44). Refined principle, registered for
the control test: **downstream rungs are priced by LOCAL stream error
at their read sites**, not by global fidelity or upstream CE. Two
side-facts worth their ink: (i) relative stream error around 100%
coexists with CE cost of +1.4 -- the model's behavior is extraordinarily
robust to stream substitution, so "the assembly works" never meant
"the streams are close"; (ii) the empirical twin is not globally more
faithful either -- both stand-in families trade local fidelity
differently. Control queued: motif gains refit and priced on the
empirical twin base under identical grammar -- local pricing predicts
a materially lower marginal than the fold base's +1.04.

## 311. Local pricing certified -- the composition economics are complete

`local_pricing.py`: the 38 motif-head gains on the empirical twin base
cost **+0.4258 fresh** against +1.04 on the fold base -- under the
registered 0.6x bar, general-substitution branch rejected. The
composition economics now stand certified end to end: (1) downstream
rungs are priced by LOCAL stream error at their read sites -- an 18%
local fidelity gap (block-7 rel-MSE 2.00 vs 1.69) amplified to a 2.4x
cost ratio, superlinear sensitivity; (2) stream fidelity and CE
dissociate completely (~100% relative stream error at +1.4 nats); (3)
the coverage frontier is therefore substrate-strategic: CE-cheap,
locally-unfaithful stand-ins (fold) win at low coverage and poison the
mid-band; faithful-where-it-matters substrates carry high coverage.
Updated fresh envelope: **20 comps @ +1.46 (fold base) / 20 comps + 38
heads @ +2.29 (empirical base -- dominates the old 26+38 @ +2.54) / 28
@ +2.84 / 34 @ +2.93.** The benchmark's next moves are now
principled: pick each rung's substrate by where downstream readers
look, and spend absorber capacity at read sites, not uniformly.

## 312. The pricing law predicts its second config: high coverage at +2.67

`empirical_L2.py`, both bars HELD: the substrate chosen BY the pricing
law (empirical base -- faithful in blocks 2-9 where the motif heads
read) carries the high-coverage stack to **+2.6735 fresh at 28
components + 38 heads** (effective coverage ~32 of 36), beating the
fold-L2 (+2.84) and the old 34-component frontier (+2.93). The
tail-attention increment came in at +0.386, inside the registered
[0.30, 0.55] band predicted from the substrate's late-stream fidelity
-- the local pricing law has now called two configurations in advance.
Updated fresh envelope: 20 @ +1.46 (fold base) / 24-eff @ +2.29 /
**32-eff @ +2.67** / 34 @ +2.93. The remaining coverage gap to full:
the 8 middle-attention components' non-motif heads and a8 -- both known
irreducible-or-expensive, which is now a measured statement about the
model, not a to-do item.

## 313. Mixing substrates doesn't beat choosing them

`mixed_front.py` (fold m0 + empirical m2/m3): base +1.7254 fresh (bar
1.65 FAILED), +38 heads +2.3191 (bar 2.20 FAILED -- 0.03 WORSE than the
pure-empirical stack). Only the marginal bar held (+0.594 <= 0.60), and
it holds in the most informative way: the motif marginal is now
measured on three substrates and orders exactly by mid-band stream
fidelity -- empirical 0.43 < mixed 0.59 < fold 1.04 -- the pricing
law's third consecutive directional confirmation. But the envelope is
unmoved: fold-m0's unfaithfulness leaks into the mid-band through the
refit chain despite empirical m2/m3, so within-front mixing buys
nothing over picking the pure substrate per coverage target. The
practical rule stands at its simpler form: fold front for low-coverage
points, empirical front when stacking the head band.

## 314. Damage modes v1: real structure, document-confounded instrument

`damage_modes.py` (the SLT-inspired data-dual): 108 causal probes x
12k tokens, factored. Registered (a) FAILED (top-8 energy 0.878 vs
0.808 shuffle null -- heavy-tailed CE deltas inflate both) and (b)
FAILED (no known-class mode above R^2 0.134); (c) held nominally but
is NOT claimable: the modes' sample tokens are dominated by a single
document's vocabulary (a travel guide), the even/odd token split
SHARES documents between halves (so the 0.83-1.00 replication does not
control the confound), and 48 rows cannot separate document effects
from function effects. The idea is validated in outline -- the modes
are stable, and the failure is the instrument's, not the direction's.
v2 queued: 3x the rows, per-column winsorization (|z|<=3) to tame the
CE tails, DOCUMENT-DISJOINT split halves (row blocks, not interleaved
tokens), and the same three bars re-registered on the fixed
instrument.

## 315. The data-dual works: five new co-dependence circuits, and the taxonomy misses them

`damage_modes2.py` (fixed instrument): structure certified at 3x the
shuffle null (0.312 vs 0.102); five modes replicate across
DOCUMENT-DISJOINT halves (0.70-0.99) with class-R^2 <= 0.15 -- bars (a)
and (c) HELD. Bar (b) FAILED informatively: NO mode aligns with the
10-class target-token taxonomy above R^2 0.149 -- causal co-dependence
organizes the data along different lines than next-token type, which
is precisely why the user's data-dual instrument adds information the
class dictionaries cannot see. The new labels, readable from their
probe loadings and sample tokens: mode 0 = the front lexical scaffold
(m0/m2/m3/a4 co-fail on digits and punctuation); mode 1 = THE TAIL MLP
BAND AS ONE UNIT (m10-m14 co-fail together -- the span-dictionary band
is a single causal object); mode 2 = an a5h7 + late-MLP newline-
adjacent complex (the regulator layer's h7 again); mode 4 = a mid-late
attention complex (a9, a11, a7h1, a9h8) whose dependent tokens are
NAME-FRAGMENTS ('ford', 'ane', ' John', 'rian') -- morpheme/surname
completion machinery the taxonomy has no class for. Certification step
queued: joint-ablate each mode's top probes on held-out rows and test
that damage concentrates on mode-scored tokens (>=3x median), with
random-probe controls.

## 316. The recursion works; the yield needs data

`circuit_tree.py` v1: **17 replicated leaves** (bar 40 FAILED), but
both structural bars HELD -- 12 of 17 leaves are invisible to the
10-class taxonomy, and the CHILD REPLICATION RATE IS 0.60: refined
probes (slice-conditioned MLP output-PCA blocks, per-head splits)
factored on a parent mode's own data yield sub-modes that replicate
across document-disjoint halves at 60% -- the user's recursive
splitting principle is validated as an algorithm. The yield bottleneck
is level 0: only ~6 of 16 root modes cleared the replication gate on
36k tokens (weaker modes need more documents to separate from noise).
v2 queued with the three scaling levers: all 212 window rows (54k
tokens), 24 root modes, and a SECOND recursion level (grandchildren,
with pca-block parents refined by block-halving on the child's slice).
Same bars re-registered.

## 317. Sixty-eight supervised circuits -- the census quadruples

`circuit_tree2.py`, all three bars HELD: **68 leaves pass the
document-disjoint replication gate** (from 17 in v1 -- one data/depth
scaling step quadrupled the census), **49 of the 68 are invisible to
the 10-class taxonomy** (class R^2 <= 0.15), and the descendant
replication rate held at 0.60 across both recursion levels. The
depth-2 gate behaved exactly as designed: grandchild slices of ~180
tokens mostly fail replication -- the algorithm refuses to
over-subdivide rather than hallucinating structure, which is the
honest failure mode the recursion was built around. Yield analysis:
leaves come predominantly from depths 0-1; the binding constraint on
further scale is DOCUMENTS (root modes and small slices both starve
for document diversity, not tokens). Next steps queued: (i) the pack
builder -- rerun the tree saving member-token indices and +/-10-token
context windows per leaf into naming packs; (ii) the naming wave --
fresh stateless subagents, one pack each, writing candidate names and
mechanistic one-liners for the 68, to be red-teamed per the standing
story protocol; (iii) the third data tranche (fresh pile documents)
targeting the 100-circuit mark.

## 318. The accidental harder gate: 35 circuits are corpus-general

`circuit_tree3.py`: 35 leaves (bar 90 FAILED) -- HALF of v2's 68,
despite 1.5x the data. Diagnosis: appending the 100 fresh pile rows
after the 212 window rows placed the corpus boundary at the
document-disjoint split median, so the replication gate silently
became CROSS-CORPUS: a mode had to reproduce its probe-loading vector
on travel-guide text AND on general pile text to survive. That is a
stricter, better test than intended, and its 35 survivors (27
taxonomy-invisible; child rate 0.60 again) are the census's
corpus-general tier, with v2's 68 as the within-corpus tier. Two-tier
census recorded: **68 within-corpus / 35 cross-corpus-stable
supervised circuits.** v4 queued with corpus-interleaved rows (both
halves balanced) to measure the intended same-gate scale-up; the
cross-corpus gate is kept as the tier-2 certification for the final
census.

## 319. Names with scores: 35 of 67 circuit names survive blind discrimination

The naming wave (4 stateless agents, 68 packs, 1 honest unclear) was
scored by a new mechanical protocol: BLIND DISCRIMINATION -- a fresh
grader receives only a circuit's name and mechanism sentence, plus 12
shuffled context snippets (6 true members, 6 from other circuits), and
must recover the members. Chance is 3/6; the wave scored **mean 4.45/6,
with 35 names passing at >=5/6**, 22 borderline at 4/6, and 10 failing
at <=3/6 (queued for renaming against the richer v4 packs). Passing
names are concrete functions: "second word of proper names" (6/6),
"entry-terminating newline" (6/6), "separators before station and date
anchors" (6/6), "connective before proper noun" (6/6). The census now
carries THREE certification tiers per circuit: structural
(document-disjoint mode replication), corpus-general (the accidental
cross-corpus gate of 318), and SEMANTIC (a name that transmits the
concept to a blind reader, quantitatively). The day ends with 68
structural circuits, 35 corpus-general, 35 blind-nameable -- and the
scaling levers identified and running (v4, corpus-interleaved).

## 320. One hundred eighteen supervised circuits

`circuit_tree4.py` (corpus-interleaved, overflow-hardened after two
instrument crashes -- fp16 capture overflow on the code-heavy fresh
documents, itself evidence the new tranche stresses regimes the travel
window never touched): **118 leaves pass the document-disjoint
replication gate on corpus-balanced data** -- past the 100-circuit
target -- with 91 of 118 taxonomy-invisible and the descendant
replication rate RISING to 0.72 (from 0.60): corpus balance helped the
recursion, not just the count. Census trajectory: 17 -> 68 -> 118 in
three instrument generations over one day, each scaling step driven by
a diagnosed constraint (data, then document diversity, then corpus
balance at the split). The naming wave for all 118 launches now
(stateless packs from circuit_tree4_packs.json), to be scored by the
blind-discrimination protocol of 319.

## 321. Forty-seven blind-nameable circuits; and modes are not output classes

The v4 naming wave scored: **47 of 115 names pass blind discrimination
at >=5/6** (mean 4.15/6, chance 3.0; 38 borderline, 30 fail). The
blind-nameable tier grew 35 -> 47 while the pass rate fell 61% -> 41%
-- the census is reaching subtler structure. Top passes read as crisp
functions ("mid-word subword continuation", "space before
opening-hours marker").

`mode_dictionary.py` FAILED both ways it could inform: +modes recovery
collapsed to 8.4% against the 10-class baseline's 55.6%, and the
shuffle control was numerically IDENTICAL to the real arm -- the
sixteen mode-constants are all near the global mean in span space, so
permuting them changes nothing. Two lessons, recorded as rules:
(i) **causal co-dependence does not imply output-value similarity** --
a damage mode says WHICH components matter WHERE, not WHAT the output
should be; modes are component-axis objects, and their benchmark value
is selecting stand-ins per site, not serving as output classes;
(ii) the override design was additionally wrong -- replacing the base
label on 54% of positions destroyed working classes; refinements must
be hierarchical (split WITHIN a base class), never overrides. v2
queued: split only the two contextual classes (ind, other -- where
constants already fail) by their top-4 modes, 18 labels total,
registered at >= +2 points with the same shuffle control.

## 322. The mode-to-output direction closes: orthogonality certified twice

`mode_dictionary2.py` (hierarchical splits of the two contextual
classes only): recovery 56.1% vs the 10-class 55.6% -- harmless, under
the +2 bar (FAILED), shuffle again identical. Together with 321 the
verdict is double-certified: **damage-mode structure is orthogonal to
output-value structure at the tail.** The modes carve which components
matter for which data; they do not carve what those components write.
The feedback loop from census to benchmark therefore runs through the
COMPONENT axis -- mode-conditioned stand-in selection (keep the
implicated components real or richly modeled on their mode's data,
spend cheap stand-ins elsewhere), a per-position extension of the
substrate-pricing law -- and that design goes to the backlog as the
next properly-registered rung rather than being improvised tonight.
The day's census stands: 118 structural circuits, 47 blind-nameable,
two instrument rules earned (no selection without nulls at every
level; hierarchies not overrides), and a benchmark frontier of
20 @ +1.46 / 24 @ +2.29 / 32 @ +2.67 fresh.

## 323. Red-team round one: three attacks repelled, one lands on the instrument

`census_redteam.py` (reviewer-2 audit, user-directed): the census
survives three of four attacks cleanly -- **118/118 leaves are DISTINCT
under Jaccard-0.5 dedup** (the nesting-inflation attack fails: children
select different top-members than their parents), **97/118 are
sign-mixed** (contrastive circuits, not damage-severity gradients), and
the confound medians are clean (base-CE correlation 0.19, top-token
share 0.26, position 0.03 -- members are not fragile tokens, single
token types, or position bands).

Attack D -- joint-causal certification -- FAILED at 3/20 as registered,
and the failure pattern indicts the TEST, not (necessarily) the
circuits: mode scores are SIGNED, members were selected by |score|, and
several leaves show huge NEGATIVE member effects under joint ablation
(r.7.0.0: members IMPROVE by 2.71 nats while matched controls move
+0.11 -- a 24x signed effect my damage-positive bar scored as failure).
A sign-blind bar cannot certify signed objects. Recorded as FAILED as
registered (rule: the registered bar stands); the sign-aware
re-certification runs next: members split by score sign, joint
ablation must move each group in its predicted direction with pooled
|effect| >= 2x matched controls and >=70% member-level sign agreement.
Registered: >=60% of the 20 pass sign-aware. If THAT fails, the
census's causal-unit claim is genuinely in trouble and the write-up
will say so.

## 324. The census downgraded honestly: dependence neighborhoods, not linear units

`census_redteam2.py`: the sign-aware certification FAILED at **4/20**
against the registered 60% bar, and per 323's pre-stated stakes the
claim is downgraded now. What the data shows precisely: pooled member
effects exceed base-CE-matched controls by >=2x in 14 of 20 leaves
(descriptive, post hoc -- see below), but member-level SIGN agreement
collapses on most leaves, several systematically ANTI-aligned (r.7.0.0
agree 0.03, r.0.2.0 agree 0.11): the direction a token moves under
JOINT ablation is not predicted by its single-probe score signs. This
is the model's composition physics appearing at circuit scale --
single-component fingerprints do not linearly compose, exactly as the
junction (295-299) and assembly (306-311) arcs found for stand-ins.
Four leaves DO pass the full sign-aware bar (r.0.0.0 spectacularly:
pos-members +6.4, neg-members -8.4, agreement 0.96, 4.5x controls) --
linear response units exist but are rare.

**Corrected claim, propagated to the report at this boundary:** the
118-leaf census is a replicated, distinct, contrastive, confound-clean
atlas of CAUSAL DEPENDENCE NEIGHBORHOODS -- data that specially depends
on specific components -- NOT an atlas of sign-predictable response
units. The selectivity observation (14/20 at 2x matched controls) is
post hoc and therefore now REGISTERED as its own bar on the NEXT
twenty untouched blind-nameable leaves: >=60% must clear
selectivity-2x. If that fails too, the census claim narrows further to
its replication tier alone.

## 325. The selectivity bar holds fresh: the census's final form is certified

`census_redteam3.py` (twenty untouched leaves, selectivity-only bar,
registered in 324): **15/20 pass at pooled member effect >= 2x
base-CE-matched controls** -- 75% against the 60% bar, HELD. The
census's scoped claim is now certified end to end rather than
narrowed: 118 replicated, distinct, contrastive, confound-clean,
CAUSALLY SELECTIVE dependence neighborhoods, 47 with blind-transmitting
names, with sign-predictable linear response a rare special property
rather than the norm. A bonus observation from the fresh set: sign
agreement runs higher there (four leaves at 0.83-0.96) than in the
round-2 top-named twenty -- the linear-unit subpopulation is real and
not confined to celebrated leaves; estimating its true fraction is a
registered follow-up for the next census pass. Reviewer 2's audit,
verdict: three attacks repelled, one landed and reshaped the claim
into what the evidence supports, and the reshaped claim then survived
its own fresh-data test. That is the shape of a robust result.

## 326. Mechanism bootstrap v1: the fold construction rediscovers glitch tokens

`mechanism_bootstrap.py` (the induction-style third leg, v1): 0 of 22
front-MLP leaves passed -- and every trigger set failed the same way,
topped by the SAME junk tokens (SourceFile, ModLoader, cffff,
zero-width space, mojibake fragments). These are GPT-2's known GLITCH
TOKENS -- vocabulary entries with anomalous, undertrained embeddings --
and the raw fold construction surfaces them from pure weights: their
single-token forwards are globally loud, projecting strongly into
EVERY direction, so they hijack any magnitude-ranked trigger set. Two
records from this: (i) the instrument lesson -- mechanism-derived
conditions must be ranked by the FRACTION of a token's output in the
circuit's directions, not the magnitude, over corpus-supported
vocabulary (v2, queued, both fixes member-blind so the anti-fake
discipline stands); (ii) a small independent finding -- the anomalous-
token pathology of GPT-2-family vocabularies is detectable in this
model by a weights-only, one-forward-per-token instrument, no behavior
needed. The null bar held (0.0x -- random trigger sets predict
nothing), so the test has teeth; v1's zero is a real miss by a real
instrument flaw, not a vacuous test.

## 327. Unigram mechanisms certified insufficient; the ladder goes to pairs

`mechanism_bootstrap2.py` (glitch-proofed): still 0/22 -- but now the
trigger sets are SANE and diagnostic: for the name-completion family
the top unigram triggers are name-INITIAL fragments (' W', ' H', ' T')
while the members sit on the SECOND token of names. Verdict, certified
twice with clean nulls: the front-MLP circuits are NOT unigram
functions -- their firing conditions are inherently at least PAIR
conditions ((name-initial, continuation)), which is itself a
mechanistic fact about what the front computes. v3 queued: the BIGRAM
FOLD -- all 60k corpus-occurring token pairs through the prefix,
T=2 forwards, weights-only -- with the pair-level trigger sets as the
mechanism-derived conditions. The mechanism ladder (unigram -> bigram
-> context features) is now itself an instrument: the rung at which a
circuit's condition becomes expressible IS its mechanistic complexity
class.

Overnight docket (user directive: keep running; main thrust = census
-> benchmark): fake_battery (running), program_names (fixed),
mechanism_bootstrap3, and the two benchmark cash-ins -- gated_assembly
(targeted real-component spending at census positions, random-gating
control) and slack_harvest (per-position ablation of circuit
components at negative-score positions, registered to IMPROVE the
untouched model, random-position control).

## 328. First cash-ins: the knowledge is real, the economics and signs need work

`gated_assembly.py`: FAILED its bars (gain +0.138 at 36% gated vs
registered 0.25 at <=25%) but the control certified the core claim --
random gating at matched fraction gains +0.045, so census positions
carry **3.1x the value per gated token**. The gate was too broad (union
of ten modes at the 92nd percentile) and the gated component set too
coarse. v2 queued: per-mode 97th-percentile gates (~10% fraction),
same bars scaled (gain >= 0.12 at <= 12%, random <= 40%).

`slack_harvest.py`: FAILED informatively -- the policy HURT (+0.58,
worse than random-position +0.33), and the diagnosis is a design bug,
not a refutation of circuit slack: **SVD mode signs are arbitrary**, so
"negative score = improves under ablation" was an unjustified global
convention; 324's sign result was purely relative (score sign predicts
EFFECT sign per leaf, either polarity). v2 queued with per-mode sign
CALIBRATION on the fit window (measure which polarity's members
improve under that mode's component ablation on fit rows -- a
fit-window statistic, deploy-consistent), then apply to fresh.

`fake_battery` crashed on its second candidate in a way that is itself
a scoring datum: the severity fake's members exhaust the top of the
difficulty distribution, so NO matched control set exists -- the
matching discipline rejects it STRUCTURALLY (before any measurement).
F1 (random members) was cleanly rejected before the crash (selectivity
0.93, specificity 0.18). The battery reruns with empty-band =
structural rejection and the dtype guard.

## 329. Sixty-four bits is not enough: the census sits above the surface-feature class

`program_names.py` (fixed): **6 of 71 unnamed and 8 of 47 blind-
nameable circuits are predictable by a <=64-bit token-predicate
program** (held-out balanced accuracy >= 0.75; median shuffled-label
null 0.49 -- the search cannot fake it). Both bars FAILED, and the
failure completes a three-way complexity measurement. The same
circuits that pass BLIND-NAME discrimination (a full reasoning agent
decoding a sentence like "next section heading recall") mostly cannot
be captured by shallow rule programs, and cannot be captured by
unigram weight mechanisms either (326-327). The description-language
hierarchy, now with measured populations at each rung:
  64-bit token programs (14 circuits) < unigram fold mechanisms (0)
  <= bigram mechanisms (running) < agent-decoded natural-language
  concepts (47) < replicated-but-undescribed (71 -> the census
  majority).
The census majority is certifiably real (replication + selectivity)
and certifiably NOT simple in any language tried yet -- which is the
honest, quantified version of "not yet understood," and the target
list for richer mechanism rungs (bigram in flight; context features
next).

## 330. First model improvement from circuits; and the battery earns its keep by breaking the battery

Three verdicts. **`slack_harvest2` HELD both bars: the sign-calibrated
per-position ablation policy IMPROVES the untouched model, Delta-CE
-0.048 on fresh text, while the random-position control HURTS
(+0.143).** Calibration found two harvestable modes (mode 8: a13/m17,
positive polarity, fit-window effect -0.183). This is the program's
first interpretability-guided edit that beats the real model --
circuit-level slack, harvested with controls.

`gated_assembly2`: absolute gain +0.078 at 8% gated (bar FAILED) but
the efficiency of census knowledge RISES with gate tightness: 3.1x
random per token at 36% coverage, **9.4x at 8%**. The knowledge is
increasingly pure at the top of the score distribution; the absolute
ceiling is set by how much error the gated components carry at those
positions.

`fake_battery` (clean run) did precisely what it was built for --
THREE hits on the certification design itself:
(i) DESIGN FLAW: cyclic foreign-set assignment paired sibling leaves
(r.1.1.0 tested against r.1.1.2 -- same subtree, shared machinery), so
real circuits "failed" specificity against their own relatives; foreign
sets must be family-disjoint. Only r.8.0.0 passed as scored (1/5).
(ii) HOLE: the Frankenstein passes selectivity (4.17) because half its
members are genuine -- selectivity cannot detect composition; the sign-
coherence dimension (which crashed to 0.03-0.11 on exactly such
mixtures in 324) must join the per-candidate battery.
(iii) HOLE: the adversarial statistic-selected fake passes BOTH
dimensions (sel 3.99, spec 13.72) -- my prediction that fragile tokens
respond to any strong probe set was wrong; selection on a specific
probe set produces set-specific members. What still distinguishes it:
its "loading" is an all-positive severity cone (the sign-mixedness
test from round one). F2's nan is the structural-rejection case
surfacing as arithmetic. Battery v3 queued: family-disjoint foreign
sets, per-candidate sign-mixedness, nan==structural-reject -- and the
adversarial fake MUST fail sign-mixedness or the hole stands.

## 331. Two circuits reach induction grade: weights-derived pair mechanisms at 33x precision

`mechanism_bootstrap3.py` (bigram fold): the 40% bar FAILED (2 of 22),
but the two passes are the program's first full induction-grade
certifications by the user's standard -- mechanism derived from
component structure alone, predicting the member extension out of
sample. **r.0.1.2: precision 32.6x base rate at recall 0.31, null
0.0x, trigger pairs ' vert|ig', ' hem|isp', ' amph|ithe', ' char|cut'
-- the circuit fires on the second piece of rare multi-token words**,
read directly off m0's direction blocks by running token pairs through
the fold. r.0.0.1 (35.5x): boilerplate fragment continuations
('www|.', 'HOT|EL', 'S|ights'). Structure -> extension, zero member
data in the condition. The description hierarchy update: 64-bit
programs 14, unigram mechanisms 0, BIGRAM MECHANISMS 2 (at 33-36x
precision when expressible), agent-decoded names 47. Twenty front
leaves sit above the bigram rung; v4 queued extends the fold capture
to layers 0-9 so the middle-MLP leaves get their rung measured too
(registered: >=6 pass; depth-vs-complexity-class reported).

## 332. The battery closes: five fakes, five rejections -- and specificity learns its final form

`fake_battery3.py`: **every constructed fake is now rejected on at
least one dimension.** F1 by selectivity (0.93); F2 by the structural
nan (unmatchable members -- the accounting maps nan to structural
rejection in v4); F3 by specificity; F4 by selectivity (1.08) and
sign-minority (0.10); and F5 -- the adversarial statistic-selected
fake that passed everything in v2 -- **rejected by sign-minority 0.00
against the reals' 0.33-0.44**, closing the discovered hole exactly as
registered: a candidate selected on raw damage is an all-positive
severity cone and cannot fake contrastive sign structure. The
remaining defect is the instrument's, identified by the reals' scores:
raw-effect specificity conflates COMPONENT STRENGTH (weak-component
circuits score anti-specific against any strong foreign set). Final
form, v4: specificity = selectivity(own probes) / selectivity(foreign
probes), each measured against matched controls -- strength cancels.
Registered: >=4/5 reals pass v4 with all five fakes still rejected.

`mechanism_bootstrap4.py`: extending the fold to layers 0-9 added NO
eligible leaves -- the deeper census is attention-probed, so its
mechanism rung requires pattern-side constructions (motif conditions
composed with value reads), not deeper MLP folds. The bigram
population stands at 2; the ladder's next rung is architectural, and
goes to the backlog with a design note rather than an overnight
improvisation.

## 333. The battery converges: three hard dimensions, one structural lesson

Final accounting across five battery versions. The converged
per-candidate battery is THREE dimensions: **selectivity** (>=2x vs
difficulty-matched controls), **sign-mixedness** (>=0.15 -- severity
cones and statistic-selected fakes score ~0), and **structural
rejection** (unmatchable members). These reject four of the five fake
classes cleanly and stably across every version: random members (sel
0.93), pure severity (structural nan), topic clusters (sel 1.08, smix
0.10), and the adversarial statistic-selected candidate (smix 0.00 --
the hole it exposed stayed closed once found). The fifth class -- the
Frankenstein, a mislabeled UNION of two real circuits -- is provably
uncatchable by per-candidate statistics in a hierarchical census:
member-overlap cannot flag it because SIBLING leaves intrinsically
share 0.62-0.81 of members (overlapping extreme tokens of one parent
slice), so the mixture sits inside the reals' own overlap
distribution. The scoped conclusion, recorded as the battery's final
form: fabricated, confounded, and adversarial circuits are caught
per-candidate; COMPOSITION of real circuits is governed structurally
-- by the census tree and its overlap bookkeeping -- not statistically.
Reals pass the converged three-dimension battery 5/5. The
certification pipeline is calibrated, its blind spot named and
bounded, and the battery rests.

## 334. The pattern-side rung certifies two more: four induction-grade circuits

`mechanism_bootstrap5.py` (attention bigram fold -- the 2-token
sequence makes the pattern-composed value read exact for prev-motif
paths): 2 of 63 attention-probed leaves pass (bar 4 FAILED), at 7.7x
and 7.9x precision with recall ~0.40 and clean nulls. The two passes
are **r.3.0 and r.8.0** -- triggers are (place-name-suffix, newline)
pairs: ' Dover|\\n', 'bury|\\n', 'ington|\\n' -- and r.8.0 is the PARENT
of r.8.0.0 ("next section heading recall"), so the heading-recall
complex now carries a weights-derived mechanism at two tree levels,
coherent with its '#### Place\\n' document structure. Induction-grade
population: 4 (two MLP-side, two attention-side). The remaining 61
attention leaves sit above the 2-token rung -- their patterns need
longer context than a bigram can express, which the motif census
predicts (self and induction motifs read beyond the previous token).
The mechanism ladder stands measured at every rung the instruments can
currently reach; longer-context folds are a combinatorial design
problem for the backlog, not an overnight run.

## 335. The linear-unit fraction: one in six

`linear_fraction.py` (batch 3, leaves 40-60): 1/20 sign-aware linear
units, inside the registered [1,8] band. **Pooled over sixty leaves:
10/60 = 16.7% (Wilson 95%: 9.3%-28.0%).** The census's linearity
statement is now a measured population parameter: about one circuit in
six responds to joint ablation in the directions its single-probe
fingerprint predicts; the remaining five-sixths are interaction-
dominated, the same composition physics certified at the junction,
assembly, and battery levels. A trend worth a line but not a claim:
linearity falls with name-score rank across batches (4/20, 5/20,
1/20) -- the circuits easiest to name may also be the most linear,
which if it survives a registered test would tie the description
hierarchy to response geometry. The census's headline numbers now
stand complete: 118 replicated / 35 corpus-general / 47 blind-nameable
/ 14 programmable / 4 induction-grade / ~17% linear -- every adjective
a measurement.

## 336. Registered: does nameability predict linearity? (analysis of existing data)

The batch trend (4/20, 5/20, 1/20 by descending name-score rank)
suggests easier-to-name circuits are more linear. REGISTERED before
computing: across the 60 sign-aware-tested leaves, the point-biserial
association between blind-name score (true positives out of 6) and
linear-unit outcome is positive, with one-sided permutation p < 0.05
(10,000 shuffles). If it fails, the batch trend was rank-batching
noise, recorded as such.

Result: **FAILED as registered.** Across the 60 tested leaves, linear
units average name-score 5.10 vs 4.98 for the rest (diff +0.12,
one-sided permutation p=0.39). The batch trend was rank-batching
noise, exactly as the failure branch anticipated: nameability and
response linearity are UNRELATED properties of the census at this
sample size -- how easy a circuit is to describe says nothing about
whether it responds linearly. The description hierarchy and the
response-geometry axis are independent measurements, which is itself a
useful fact: a circuit atlas needs both columns.

## 337. The deploy gap for gating, priced: 9.4x oracle vs 1.5x programs

`deploy_gated.py` (fixed): gating on the 14 certified circuit programs
-- fully input-only, 19.6% of positions -- gains +0.014 against
random-matched +0.010: **efficiency 1.5x, versus the oracle
mode-score gates' 9.4x.** Both bars FAILED, and the failure is the
measurement: the 64-bit programs transmit circuit NAMES but not
gate-quality position selection -- they describe classes of position
(mid-word, early-after-newline) while the causal scores pick the
specific dependent tokens. The ratio 9.4/1.5 prices what the census
knows beyond surface features, in nats-per-gated-token, and matches
329's hierarchy verdict from the other side. Closing the gating arc's
ledger: census knowledge is worth 9.4x random when you can afford
causal labels, 1.5x when restricted to 64-bit surface programs, and
the gap is the deployment cost of the census majority living above
the surface-feature class. Sharper deployable labelers = the richer
mechanism rungs, already the backlog's design item. The benchmark and
census arcs both now rest at fully-measured states.

## 338. The compositionality dividend: fifty circuits crack bottom-up

`compositional_ladder.py`, all three bars HELD. With the certified-
feature library (surface predicates + mechanical class labels + the
four induction-grade pair conditions + SHIFT compositions licensed by
attn0's exact lexicality and the 27 certified prev-motif heads):
**36 of 104 previously-unprogrammable circuits became predictable at
held-out >= 0.75** (surface baseline: 6), median null 0.51. Iteration
two -- every pass's program joining the library -- **added 14 more**,
and the iteration-2 programs realize the user's compositional thesis
literally: circuits described as conjunctions of OTHER circuits
(r.8.0.0 = circ_r.0.0.1 AND circ_r.0.1.0; r.8.2.0 = circ_r.8.1.2 AND
seen_before). The census is collapsing bottom-up into a COMPOSITIONAL
DEPENDENCY GRAPH of functions, at ~50 newly-described circuits for the
cost of one search over certified pieces -- no k-token enumeration
anywhere. Description hierarchy update: programmable population 14 ->
~64 of 118. Run-until-dry continuation queued (iterate library growth
until a round adds fewer than 3; registered: converged programmable
total >= 70, null stays <= 0.6).

## 339. Until-dry convergence: 57 of 118 are programmable, and that is the number

compositional_ladder2 reran the whole ladder from a deterministic tree
rebuild and iterated the compounding loop to exhaustion: base library
(~90 features: surface predicates, mechanical class labels, trigger
pairs, and prev1_/prev2_ shifted copies) programs 35 leaves at the
doc-disjoint held-out >=0.75 bar; then each round adds every passing
circuit as a `circ_<tag>` predicate and re-searches the failures.
Rounds added +14, +4, +4, +0 -- converged in three rounds at **57/118
programmable**. Median shuffled-label null 0.509 (bar <=0.6 HELD);
compounding contribution 22 (bar >=3 HELD). The registered headline
bar -- converged total >=70 -- **FAILED**. §338's "~64" was an
overcount from the non-converged v1 pass on a different tree draw;
57 is the disciplined number and replaces it. The compounding story
survives with a sharper shape: the library DOES crack previously
unprogrammable leaves (44/104 got candidate rules, 22 survived
held-out), and the passes are genuinely circuits-of-circuits
(r.8.0.0 = circ_r_0_0_1 AND circ_r_0_1_0; r.0.2.2 = circ_r_3_0_0 AND
circ_r_3_0_1), but growth is strongly sub-geometric: each round's
catch is built from the previous round's, and the well runs dry once
the correlated-context families are exhausted. Bottom-up composition
buys one multiplicative step (35 -> 57, +63%), not a cascade.

## 340. The circuit call-graph is flat: composition does not follow depth

dependency_graph built the DAG over the 57 converged programs (edge
A -> B when B's program cites circ_A) and tested two registered
structural predictions. Both **FAILED**, and the failures are the
finding. (a) Layering: only 14/29 edges point from a shallower
circuit (mean layer of its top causal probes) into a deeper user --
48%, a coin flip, against the registered >=75%. (b) Hubs: zero
circuits are cited by >=4 others (registered >=3 such hubs); the
best are r.6.0.1, r.8.1.2, r.0.1.0 at degree 3. Reading: when a
circuit's program cites another circuit, it is borrowing a
*correlated context family*, not calling a *subroutine that runs
earlier in the stack*. The functional composition graph does not
mirror physical depth, and no circuit is a universal primitive --
reuse is broad (16 circuits cited at least once, 29 edges over 57
nodes) but shallow. This matches the motif census from the weights
side: bilin18 shares pattern-side *types* across heads but builds no
shared OV library, and now the same shape appears in data space --
overlapping context families without a layered call hierarchy.

## 341. Scaling the program set 4x: gain x12, efficiency still 1.5x

deploy_gated2 gated the empirical_L2 assembly with all 57 converged
programs (input-only features plus fixed-point resolution of circ_
references). Gated fraction of fresh positions: 65.7%. The assembly
improves +2.6735 -> +2.5029, a **+0.171 nat gain -- 12x the
14-program version (+0.014) and the largest deploy-legal benchmark
move so far**. Both registered bars still FAILED: random gating at
the matched fraction recovers +0.111 (65% of the gain; bar <=40%),
so efficiency is 1.54x random -- the same ratio as the 14-program
gate and far under the 9.4x causal-label oracle (§337). The deploy
gap is therefore a property of the *description language*, not of
program count: input-only surface programs, no matter how many,
know roughly WHERE circuit-dense text is but not WHICH positions
carry the causal load; quadrupling the vocabulary scales coverage
(and absolute gain) without sharpening per-position targeting.
Follow-on registered and queued (probe_gate.py): a ridge probe
reading the residual stream after block 2 -- available at inference
because every frontier config keeps the cheap lexical rungs real --
fitted to the fit-window oracle labels. Bars: AUC >=0.75, efficiency
>=2.5x random, >=50% of oracle gain at matched fraction. That places
the stream-readable middle ground between 1.5x (surface) and 9.4x
(oracle). Ops note: census_cache crashed at final serialization
(keys popped before the save that read them); fixed, requeued, now
running -- once census_state.pt lands, census-lineage scripts stop
paying the ~10-min tree rebuild.

## 342. The stream probe breaks the surface ceiling: 3.8x random

probe_gate fit a ridge probe on the residual stream after block 2
(deploy-legal: every frontier config keeps the cheap lexical rungs
real) to the fit window's oracle causal labels, then gated fresh rows
at the oracle-matched fraction (17.25%). Results against the three
registered bars: AUC 0.621 (bar 0.75 FAILED); **efficiency 3.8x
random (bar 2.5x HELD)** -- the first deploy-legal gate above the
1.5x surface-program ceiling; 42% of the oracle gain recovered (bar
50% FAILED, close). Gains: oracle +0.078, probe +0.033, random
+0.009 at matched fraction. The description-language ladder for
gating now has three measured rungs: input-only surface programs
1.5x, stream probe 3.8x, causal labels 9.4x. Reading: roughly half
of what the oracle knows about where circuits carry load is linearly
readable off the early stream; the remainder needs either deeper
reads or nonlinear features. v2 queued (probe_gate2): quadratic
features in mlp3's read directions (the model is bilinear -- give
the probe its native basis) + direct regression of the 10 mode
scores with any-mode gating, bars AUC >=0.70 / eff >=3x / >=60%
oracle.

## 343. Induction heads are the LEAST compressible heads (prediction inverted)

head_lowrank truncated per-head score projections (c_q,c_k,c_q2,c_k2)
vs value machinery (c_v rows + c_proj cols) to rank 8/16/32 of 128,
per motif class, matched 9 heads each, fresh CE. All three registered
bars FAILED because the prediction was exactly inverted: induction
heads cost +0.254 (QK, rank 16) and +0.172 (value) -- an order of
magnitude MORE than any other class (self +0.005/+0.008, prev
+0.024/+0.067, diffuse +0.014/+0.026). Self heads are nearly free at
rank 8 on both sides. The lesson replaces the registered story:
motif "simplicity" is about pattern SHAPE, not weight rank. Self and
prev heads have simple patterns AND low-rank implementations;
induction heads have a simple pattern RULE (match-and-copy) whose
implementation is intrinsically high-rank -- distinguishing which of
50k tokens to match requires high-dimensional keys, and the model
spends full head width on it. This explains, from the weights side,
why induction heads resisted every stand-in in the assembly arc and
closed as irreducible: they are the densest computation per
parameter in the attention stack. It also means backlog item 5
(learned low-rank bilinear stand-in for induction heads) is now
measured as unpromising at the weight level and closes.

## 344. First standalone circuit explainer: the line-break circuit (r.0.0.1)

User asked for one census circuit explained end to end with
non-cherry-picked examples. circuit_explainer reproduced the exact
census causal operation on census_state.pt leaf r.0.0.1 (184
members; machinery = slice-conditioned output-PCA blocks m0(12-16),
m0(48-64), m3(16-24), m3(24-32)) and reported 5 top-|score| + 5
seed-0 random member positions. Both registered bars FAILED
informatively: the circuit is a TWO-SIGNED policy, not a uniform
helper. Members split 96 positive-score (dCE +1.01 when ablated;
examples +3.5..+5.2 on single tokens) vs 88 negative-score (dCE
-2.73; ablation HELPS where the pushed line-break expectation was
wrong). |dCE| concentrates on members: 2.67 nats vs 1.51 same-doc
non-members vs 0.35 corpus background (7.7x). All ten examples are
guidebook listing layouts predicting newline/heading tokens at
genuinely hard positions (base CE 5-26 nats) -- the circuit is the
model's line-break policy in list structures, computed in mlp0/mlp3
direction bundles. Published as a standalone artifact page with the
full 184-member scatter and the ten examples verbatim. Caveats on
record: fit-window corpus (one travel guide dominates); tags are
tree-instance-relative (this leaf is the grid-reference/venue end of
the certified place-suffix|newline family r.3.0/r.8.0, not the same
tag across builds); ops note -- census_cache's state is the 212-row
v1 tree, not cl2's 312-row tree, so cross-tree matching must go
through member overlap.

## 345. Registered: fold-basis features (early layers as basis for later circuits)

User direction: the early layers are the best-understood part of the
model (weight folds from the embedding); use them as a basis.
fold_basis.py (running) folds mlp0-3 over the vocabulary
(mlp(rms_norm(wte))), takes top-8 PCA directions of each fold table,
and adds 32 above-median score predicates (+64 shifted copies free)
to the ladder library, then reruns until-dry. REGISTERED: (a)
converged >=65/118 (+8 over cl2's 57); (b) null <=0.6; (c) >=half
the gain cites a fold_ predicate.

## 346. Fold features substitute but do not extend: the programmable frontier is feature-robust

fold_basis added 32 early-layer weight-fold predicates (top-8 PCA
directions of mlp0-3 folded over the vocabulary, +64 shifted copies)
to the ladder library and reran until dry. Converged total: **58/118
(+1 over cl2's 57)** -- the registered >=65 bar FAILED. But the fold
predicates were cited by 19 passing programs (bar (c) HELD): they
SUBSTITUTE for hand-named surface features inside programs that
already passed, without cracking new leaves. Convergent evidence
with 340: the ~60 unprogrammable leaves are not waiting for a better
input-token vocabulary -- any reasonable basis (hand-named, class,
trigger, weight-fold) programs the same ~58, and the remainder needs
context the input token stream does not carry position-locally
(attention-transported content, longer-range structure). The
programmable frontier is a property of the description language
class, mirroring the deploy-gap result (341).

## 347. Probe v2 overfits: richer basis WORSE than plain ridge (negative result)

probe_gate2 replaced v1's linear ridge -> binary label with
quadratic features in mlp3's read directions (1680 dims) + direct
regression of the 10 oracle mode scores + any-mode gating. All three
bars FAILED, and v2 is strictly worse than v1: AUC 0.551 (v1 0.621),
probe gain -0.005 (v1 +0.033). The quadratic per-mode probe
overfits the fit window and the any-mode gate construction amplifies
per-mode noise. v1's simple binary ridge at 3.8x random stands as
the record. Lesson recorded: at a fixed early read point, richer
feature bases overfit before they help; the AUC ceiling (~0.62)
needs information not linearly-or-quadratically present at block 2
-- later read points or context aggregation, not fancier features.

## 348. Sign-mixed policy structure is UNIVERSAL: 50/50 leaves two-signed

explainer_batch ran the 344 treatment over every pca-probe census
leaf (50 tested, 18 skipped as roots/small). Both registered bars
HELD emphatically: **96% of leaves concentrate |dCE| >=3x over
corpus background, and 100% are two-signed** (minority sign share
>=0.15 -- typical minority share is near half). r.0.0.1 is not a
quirk: EVERY census circuit is a two-signed policy whose machinery
pushes a decision that is right at some members and wrong at others.
Mechanical examples + base-CE stratification (member mean, frac<3)
now recorded for all 50 -- the raw material for the circuit
registry. This retires the registered-bars-assume-uniform-damage
error class (344) permanently: uniform-damage predictions are now
known-wrong a priori in this model.

## 349. The line-break circuit is a push-brake pair; first tension edges measured

bundle_split ablated r.0.0.1's four bundles singly (+pairs+joint).
(a) DISSOCIATION HELD: bundle damage profiles are not
interchangeable (pairwise r from -0.48 to +0.35); b0 (m0 dirs
12-16) and b1 (m3 dirs 24-32) are ANTI-correlated (-0.48). (b) As
registered ("wings governed by different bundles") FAILED -- both
wings' largest driver is b0 -- but the structure found is sharper:
**b0 is a break-PUSH and the m3 bundles are a break-BRAKE**. Pos
wing: b0 +3.17, b1 -2.03 (removing the brake HELPS where the push
is right). At the Westminster-Abbey example: b0 alone +8.07, b1
-5.99, b3 -6.48, m0-pair +0.41, m3-pair -5.11, joint -6.79 --
(d) HELD, the improvement is 95% attributable to the brake side.
The user's composition hypothesis is confirmed in structure: the
circuit is a SUBTRACTION of two simpler parts (mlp0 pushes
line-breaks in list layouts; mlp3 opposes it), and the two-signed
membership is the visible shadow of that antagonism. (c) TENSION
HELD: 11 other leaves' members IMPROVE by <=-0.3 mean when this
machinery is removed (r.1.0.0 -0.54, r.1.1.0 -0.75, r.3.0.0 -0.53,
r.3.1.0 -0.55, r.6.2.0 -0.45, ...): the first measured tension
edges, now first-class relations in the circuit schema. Follow-on
queued: interchange.py (Geiger-style) -- set the push channel to
its break-state value at non-break positions and predict the
newline logit follows.

## 350. Interchange null: the channel's value does nothing LOCALLY (read is elsewhere)

interchange set the push channel (b0 = m0 dirs 12-16) to its
break-state value at 128 non-break positions and to its rest-state
at the push wing. Both registered bars FAILED at near-zero effect
(+/-0.02 and -0.15 mean newline-logit shift) -- while PROJECTING OUT
the same directions moves member CE by multiple nats. The
asymmetry: projection-removal acted at ALL positions; the patch was
position-local. Inference: the channel's causal read is NOT local
-- downstream attention reads m0's b0 content from CONTEXT positions
(the preceding list entries), and/or the effect is bilinear
(product with partner factors, dead when set without them), and/or
the mean-donor value washes out. v2 queued (interchange2) separates
these: local vs full-prefix vs prefix-only vs random-subspace-prefix
patching with a single sampled donor value. Registered: prefix >=5x
local; prefix_only >=70% of prefix; random <=40%. Whatever wins, the
read path of the line-break circuit gets localized.

## 351. The registry goes live (50 records) and the IOI window OPENS at 99%

sop_populate wrote 50 schema-v1 circuit records (causal stats,
sign splits, base-CE stratification, mechanical examples,
certification verdicts) into circuits/ and regenerated
circuits.html; r.0.0.1 carries the push-brake bundle structure and
its 11 tension edges. Program pass rate at the strict bar was 2/16
(r.3.1.0 0.788, r.1.1.2 0.773) -- bar (b) FAILED because
census_lib's library lacked the 10 mechanical class labels that
carry most cl2 programs; ported in, so SOP reruns will search the
full base. And the capability gate paid off immediately:
**bilin18 does IOI at 99% pair accuracy (margin +2.41, shuffled-name
control -0.01)** on 96 constructed prompts -- the first constructed-
prompt task window to OPEN (addition closed at 0%). ioi_circuit
queued: 36-component mean-ablation margin drops + head-level
deletion in the top attention owner. Registered: <=6 components for
70% of the drop; >=1 mid/late attention layer in top-3; top-2 heads
>=50% of their layer's drop.

## 352. Value-transplant interventions fail everywhere on the push channel

interchange2 separated the candidate explanations for 350's null:
local, full-prefix, prefix-only-context, and random-subspace-prefix
patches of the b0 channel to a single sampled donor's break-state
value. Result: NOTHING moves the newline logit meaningfully (local
-0.03, prefix -0.06, prefix-only -0.04, random -0.004). (a) and (b)
FAILED; only subspace-specificity (c) held. Conclusion recorded:
on this circuit, SETTING the channel's value -- at any position
pattern tried -- is causally inert, while DELETING the subspace
moves member CE by nats. This is a substantive finding about
bilinear circuits: the census's dependence neighborhoods are
established by subtraction, and the deleted directions do not
behave like classical additive features (a fixed transplanted value
is off-manifold; downstream products with varying partner factors
average it away). Classical activation-patching intuitions from
ReLU-transformer work (IOI name-movers etc.) do NOT transfer
as-is. Motivates DAS (learned aligned basis instead of PCA blocks)
as the next instrument class -- PCA rotation plausibly mixes the
true variables. Value-transplant arc on r.0.0.1 closes here.

## 353. IOI localization: five owners, and zero-deletion is the wrong head instrument

ioi_circuit mean-ablated all 36 components on the 96 IOI prompts.
All three bars HELD: 5 components account for 70% of margin damage
-- **m1 (+2.82, more than the whole +2.41 margin), a14 (+1.65), a5
(+1.47), m2 (+1.28), m0 (+0.91)** -- and two mid/late attention
layers sit in the top-3. a5 contains the induction head (5,5) and
both first-motif heads; a14 was a surprise owner. But the
head-level leg produced a striking artifact-shaped result: deleting
ANY single head of a14 costs the same ~1.64, all nine heads within
0.005 of each other. The recompute path was verified EXACT
(empty-deletion delta +0.0000), so the uniformity is real -- and the
reading is that ZEROING a head is an off-manifold magnitude shock
to c_proj's input (echoing 352: subtraction vs value-change
asymmetry), so it measures scale sensitivity, not head content.
ioi_heads2 queued: within-prompt MEAN ablation per head (content
killed, magnitude kept) on a14 and a5. Registered: differentiation
max/min >=3; top-2 heads >=60% of layer; a5's top head is the
induction or a first head. sop_programs2 also queued: step-3 rerun
with the enriched 65-feature library (bars: >=8/16 pass).

## 354. The IOI circuit resolves: a name-mover head exists (a14.h4)

ioi_heads2 replaced zero-deletion with within-prompt mean ablation
(content killed, magnitude kept). All three bars HELD and the
uniform-drop artifact vanished: **a14 head 4 carries 0.93 of the
layer's 1.35 margin drop** (top-2 = 81%); a5's top head is (5,7) --
a FIRST-motif head -- with the induction head (5,5) third at 0.12.
The IOI circuit in bilin18 now reads: early token tables (m1, m2,
m0) + sentence-start anchoring (5,7) + duplicate detection (5,5
band) + one dominant late mover head (14,4). The zeroing-vs-mean
contrast (353 vs 354) is now a measured methods rule: in this
architecture, zero-ablation of a head is a magnitude shock that
reads uniform across heads; mean-ablation recovers head content.
Same lesson as 352 from the other side.

## 355. Enriched library: 5/16 programs pass (bar missed), compounding visible

sop_programs2 reran step-3 with the 65-feature library: 5/16 pass
(vs 2/16), median null 0.52 (HELD), class predicates cited (HELD),
registered >=8 FAILED. Notable: circ_r_1_1_2 (a passing program)
already appears inside three other leaves' best programs --
registry-mediated compounding works exactly as designed, but the
212-row grid's leaves are harder than cl2's 312-row leaves (denser,
smaller n). The swarm SOP inherits the honest bar as-is.

## 356. Swarm pipeline validated: three Sonnet agents, three honest records, three fix rounds

Dry run for the Thu-Sun swarm: three fresh Sonnet-class agents each
ran CIRCUIT_SOP.md on an unclaimed leaf (r.6.1.1, r.6.3.0, r.1.2.0),
GPU shared with the queue. VERDICT: the verification-driven design
works. All three reproduced the recorded causal numbers exactly
(concentrations 6.91-6.94x, HELD), ran every gate honestly (0/3
program passes at the strict bar -- bacc 0.54-0.68 -- and none
falsely registered a feature), wrote schema-valid records, and one
agent, on discovering a mid-run SOP edit, re-ran its red-team under
the corrected wording and DOWNGRADED its own story to weak (0/3
causal-direction hits) rather than let an optimistic verdict stand.
Cheap models cannot certify junk through this pipeline; they can
only fail to find things. The dry run surfaced five real defects,
all fixed and pushed in three rounds: (1) write_circuit shallow-
update could silently destroy append-only certification history ->
deep-merge with dedup, enforced in code; (2) no partial-record
branch in the SOP -> resume clause; (3) step-5 red-team wording was
literally untestable ("predict membership" of guaranteed members) ->
causal-direction test; (4) MY OWN directory-wide `git add circuits/`
swept a concurrent agent's uncommitted work into a pushed commit ->
consolidator model: agents never commit, the wake loop commits all
records with explicit paths; (5) registry read-modify-write race
under concurrency -> registry now rebuilt-from-scan under flock,
features.json appends locked. Provenance now records the library
git rev at task start so mid-wave infra edits are detectable (one
agent experienced version skew and caught it itself). The swarm SOP
is now concurrency-hardened by construction, not by hope.

## 357. Exact 2-token context ruled out: the programmable frontier is 55-58, full stop

fold_pair_basis added the exact 2-token contextual fold (real model
run on every corpus token pair, mlp0/mlp1 captured at position 1 --
attn0 composition and bilinear cross terms included) as 16
predicates. Converged total: **55/118 -- three LOWER than
fold_basis's 58** (bar >=63 FAILED; null HELD; pairfold cited in 15
programs, HELD). Two lessons. (1) The programmable frontier is
feature-robust at 55-58 across four vocabularies (hand-named, class,
unigram fold, exact pair fold): 2-token exact context is now RULED
OUT as the missing ingredient, so the ~60 unprogrammable leaves need
longer-range, attention-transported structure -- fourth and
strongest convergent leg. (2) The total went DOWN when features were
added: greedy rule search with a richer vocabulary overfits its
fit-half and pays at held-out -- the same richer-basis-overfits
lesson as probe_gate2 (347), now at the program rung. Run-to-run
band: treat 55-58 as one number with +/-3 search noise.

## 358. DAS half-resolves the fork: values CAN steer, specificity NOT established

das_line_break learned a 4-dim basis + value in m0-output space by
gradient ascent on the newline logit under prefix patching.
(a) HELD emphatically: held-out recipients shift **+4.42**
newline-logit (PCA basis: -0.03). So 352's "subtraction-defined"
conclusion was PCA-SPECIFIC, not class-level: a learned basis
supports value interventions. But (b) FAILED: the
shuffled-objective control -- same optimization aimed at arbitrary
off-slice positions -- also reaches +2.70 held-out (ratio 1.6x, bar
2x). Reading: gradient-learned patches are confounded by
ADVERSARIAL STEERING CAPACITY -- m0 feeds everything, and the
optimizer can find directions that pump any logit anywhere; only
~40% of the DAS effect is slice-specific. The fork therefore
resolves to a third branch nobody registered: value semantics exist
but optimization-based instruments cannot certify them without a
naturalness constraint. das2_natural queued: reuse the learned
basis but transplant NATURAL donor coordinates (no optimized
value). If natural values steer (>=+0.5), the basis captures a real
circuit variable; if only the optimized value works, it is an
adversarial direction and gets recorded as such.

## 359. The IOI circuit is a PARALLEL SUM, not a chain

ioi_chain tested serial structure by joint mean-ablation. Both
serial predictions FAILED in the cleanest possible way: every joint
is 97-99% of the sum of singles (h7+h4: 1.154 vs 1.190; h5+h4:
1.034 vs 1.046; h7+h5: 0.386 vs 0.384; all three: 1.266 vs 1.310;
near-zero control pair additive, HELD). The IOI margin is an
ADDITIVE composition of independent contributors -- first-head
anchoring, induction-band duplicate signal, and the a14.h4 mover
each feed the answer through their own path, with no measurable
shared-bottleneck dependency at this grain. This matches the
model's signature everywhere else: flat call-graph (340), no shared
OV library, additive-narrow junctions -- bilin18 composes by
summing independent evidence streams, not by pipelining. The
"mover" is not the end of a chain; it is the largest of several
parallel voters.

## 360. The DAS fork closes: "steering" was learned DELETION. Values never mattered.

das_natural transplanted natural donor coordinates in the learned
basis: mean +0.16 held-out (individual donors scatter -0.58..+0.77)
-- (a) and (b) FAILED, verdict (c) recorded. And the decisive
number nobody registered: **setting the learned coordinates to ZERO
gives +4.421 -- identical to the optimized value's +4.421.** The
optimizer never learned WHAT to write; it learned WHICH directions
to delete. The gradient found a subspace whose natural variation
suppresses the newline logit at these positions, and any constant
overwrite (optimized, zero, whatever) removes that variation
equally. So the class-level conclusion survives DAS and is now
sharper than 352's version: in this architecture the intervention
algebra is PROJECTION, not ASSIGNMENT -- circuits respond to
variance removal; written values are causally void across every
instrument tried (PCA blocks, learned bases, natural donor values,
optimized values). Value-transplant/patching methodology from ReLU
interpretability does not port to bilinear models; deletion-based
and variance-based instruments are the native tools. Intervention-
algebra arc CLOSED with a one-line law: only subtraction bites.

## 361. The "name-mover" is task-contextual, not a corpus specialist

mover_profile measured a14.h4's natural-text footprint: damage
top-5% share 38% (bar 50% FAILED -- moderately diffuse), modal
class of top-1% positions is 'other' then ind/newline/subword, with
name a distant fifth (bar FAILED); one census leaf implicated
(r.2.0.0, +0.33; bar HELD). Honest reading: a14.h4 moves NAMES in
IOI prompts because names are the salient repeated entity there,
but on natural text it is a general repeated-structure head. The
IOI "mover" role is a CONTEXT-INDUCED specialization of broader
machinery -- consistent with the parallel-sum picture (359): task
circuits in bilin18 are assembled on the fly from general-purpose
additive voters, not dedicated task modules.

## 362. The projection law, refined on 10 circuits: subtraction dominates, offsets tax

projection_law_batch generalized 360 beyond n=1. Registered (a)
("the written constant is causally void") **FAILED** as stated:
median agreement 0.626 vs the 0.75 bar. But the table shows a
lawful refinement, not chaos. Across all 10 leaves: (1) every
constant assignment -- zero, slice-mean, natural donor -- produces
large member damage, 7-30x the matched-dim random-subspace control
(median 13x, (b) HELD): VARIANCE REMOVAL is the first-order term,
confirming the subtraction picture at population level. (2) But
deletion-to-zero consistently costs ~1.6x more than assigning the
slice MEAN (delete > mean in 9/10 leaves): zeroing removes the mean
too, and that off-manifold offset carries its own tax -- the same
magnitude-shock seen in the IOI zeroing artifact (353), now
quantified. (3) Crucially, mean vs natural-donor constants are
similar everywhere: among on-manifold constants, WHICH value you
write still does not matter -- consistent with das_natural's
zero==optimized identity. Refined law on record: **effect =
variance-removal (dominant) + distance-of-constant-from-mean
(offset tax); value semantics nil.** One violator flagged per (c):
r.6.0.0 (agree 0.32, delete 1.64 vs mean 0.67) -- the strongest
candidate for a circuit where something beyond variance matters;
earmarked for individual treatment. Methods rule for all future
ablations: mean-ablate, never zero-ablate, unless the offset term
is itself the object of study.

## 363. The plateau survives match transport: no token-rule describes the other half

transported_features added induction-rule transports computable from
the raw token stream (has_match, after_match_F = "F held right
after the last occurrence of the current token", at_match_F,
distance buckets) -- features reaching arbitrarily far back, which
shifts and pair folds cannot. Converged: **55/118** (bar >=64
FAILED; null HELD; match features cited in 6 passes, HELD). This is
the FIFTH and final convergent leg: the programmable frontier sits
at 55-58 against every input-computable feature class -- surface
predicates, mechanical classes, trigger pairs, 1-2-3-token shifts,
unigram weight folds, exact 2-token pair folds, and now unbounded-
range match transports. Conclusion promoted to a named result: for
roughly half the census, MEMBERSHIP IS NOT TOKEN-DEFINABLE -- no
rule over the input string, however far it reaches, identifies
where these circuits act. The two standing hypotheses are now
stream-level: (i) the defining information exists in the model's
own early activations (consistent with the stream probe's 3.8x vs
surface 1.5x), or (ii) the leaves are activation-space objects with
no position-set description at all. stream_features queued: block-2
stream projections as ladder predicates, same >=64 bar -- the
direct test between (i) and (ii).

## 364. Pattern dictionary v1: right idea, wrong metric (magnitude swamps shape)

pattern_dictionary fit realized attention patterns as linear
combinations of 6 archetype masks. R^2 >= 0.7 for **1/162 heads**
(bar 120 FAILED) -- yet dominant-atom labels still agree with the
motif census 59/85 (69%), and we know from 282 that 71/74 motif
heads accept literal one-hot pattern swaps at <=+0.01 nats. The
resolution: bilin18's patterns are UNNORMALIZED squared products
(no softmax), so raw values carry enormous query-to-query magnitude
variation that a fixed mask cannot explain; the functional swaps
worked because fitted per-head gains absorbed exactly that
magnitude. The v1 metric measured magnitude variance, not shape.
Registered bars failed and stay on record; v2 queued with per-query
shape normalization (pattern rows normalized over keys), bars
>=100/162 at R^2>=0.5 and motif agreement >=80%. Methods note for
the swarm: in this architecture, any pattern-level claim must
separate SHAPE (where attention goes) from GAIN (how hard) -- they
are different objects with different owners (mask vs fitted
scalar).

## 365. The textbook induction story is REFUTED as code in bilin18

mech_replicate implemented the canonical induction mechanism as
executable code -- z_h(q) = alpha_h * v(j+1) at match positions,
silent elsewhere -- and substituted it for all 9 census induction
heads. Every registered bar FAILED decisively: match-position CE
+9.8 nats (bar +0.05), continuation-logit correlation 0.03 (bar
0.8), and the fitted alphas themselves are the tell: 0.05-0.29 in
magnitude with MIXED SIGNS -- the heads' real outputs are nearly
orthogonal to the literal induction read. bilin18's "induction
heads" do not compute match-and-copy in the textbook form. This
coheres with two prior results that now snap into place: these
heads are the LEAST rank-compressible in the model (343: full-width
keys), and their IOI contribution is one additive vote among
several, not a chain stage (359). The census 'ind' label describes
a pattern-shape preference, not a functional read. mech_diag
queued: per-head sparsity of the actual read (top-1 pattern share),
location of the top key (match region vs local vs other), so the
TRUE story is built from measurement. Also this session: the first
computational-grade attempt on a novel circuit (suffix_code,
context-freeness of r.3.0 as a 2-token lookup table) crashed on a
probe-kind mismatch -- census_lib now has leaf_hooks covering all
three probe kinds (pca/comp/head), requeued. Ops: the cleanup's
directory move broke the canary import chain (bilin18_canary +
bilin18_pipe_refit moved back to top level; import-graph scan now
part of any future move).

## 366. Stream predicates leave the plateau at 58: the description arc closes

stream_features added 16 block-2 residual-stream projections as
ladder predicates (stream-legal, same stance as the 3.8x probe).
Converged: **58/118** -- gain vs baseline +0 (bar >=64 FAILED; null
HELD; stream predicates cited in 17 programs, HELD -- substitution
yet again). The plateau now spans SIX feature classes: surface,
mechanical class, trigger pairs, weight folds, exact pair folds,
unbounded match transports, and block-2 stream projections. Verdict
recorded: for the ~60 residual leaves, membership has no position-
set description in any language tried -- they are ACTIVATION-SPACE
OBJECTS at this grain. Caveats that keep the door ajar: block-2
only, 16 linear directions, median thresholds; the probe AUC
ceiling (0.62) suggests deeper or nonlinear reads hold some more.
But the arc closes with the honest headline: the census describes
WHERE dependence concentrates; for half of it, "where" cannot be
compressed into a rule. Programs 55-58/118 is the number, and it is
a property of the model, not of our vocabulary.

## 367. Optimality audit: 10/10 circuits are corpus-wide useful

net_utility: every one of the 10 highest-concentration leaves is
globally net-positive-damage under its own ablation (both bars
HELD; member-tail/global ratios 17-66x). No net-harmful circuit
found; the member-level improvements that confused the layout-brake
reading are pure selection effects. The optimality argument holds
leaf-by-leaf, and the registry pages now carry this framing.

## 368. Novel-circuit code v1: context-freeness refuted at 91.5%

suffix_code tested the executable claim "r.3.0's action = TABLE
[prev,cur]" (2-token weights-derived lookup). Result: corr 0.294
(bar 0.6 FAILED), explained variance **8.5%** -- the context gap is
91.5% (shuffled control clean at -0.06, so the 8.5% is real signal).
The circuit's per-position contribution in full documents is
overwhelmingly context-modulated: the 2-token core exists but is a
small minority of the computation. Same lesson as 365 from the MLP
side: bilin18's circuits are dense contextual computations; sparse
context-free code approximations capture <10%.

## 369. The real induction story: opportunistic part-time match-readers

mech_diag measured what the 9 'ind' heads actually compute at match
positions. Both registered bars HELD: 6/9 heads are SPARSE readers
(top-1 pattern key carries 46-65% of output variance) and 5/9 are
MATCH-SEEKING (top key lands in {j, j+1} at 26-42% of match
positions -- far above the ~1/q chance floor but far below the
textbook's 100%). The functional picture: these heads split their
reads between the match region (~1/3), local positions (~1/4), and
elsewhere (~2/5) -- a mixed policy in which match-reading is one
mode, not the identity. This explains 365's refutation exactly: the
textbook code forced match-reads always and captured nothing. The
minimal faithful code therefore must INCLUDE the pattern
computation itself -- which in this architecture is closed-form
(double-QK bilinear scores are polynomial in the stream), so
"pattern-as-code" is legitimate weights-derived code, not a copout.
sparse_read_code queued: replace each head's output by ONLY its
top-1 read (pattern from the real QK weights, value from the real
mixed values). Bars: (a) match-position CE cost <=25% of full head
deletion; (b) IOI margin >=60% retained under all-9 sparse coding;
(c) shuffled-top-key control breaks it. head_read_census queued
alongside: the same functional-read metrics for ALL 162 heads --
the successor to the failed pattern-matrix dictionary, measuring
function (where the read mass actually lands) instead of form.

## 370. FIRST COMPUTATIONAL-GRADE PASS: sparse pattern-as-code carries 77%

sparse_read_code replaced each induction head's output with ONE
LINE OF CODE -- z_h(q) = pat(q,k*) * vm(k*), the head's top
coincidence-scored read, pattern from the real double-QK weights --
and ALL THREE registered bars HELD. Match-position CE cost +0.138
vs +0.601 for deleting the same heads: the code carries **77% of
the heads' function**. IOI margin 62% retained under all-9 coding
(bar 60%). Shuffled-top-key control +0.674 (>=3x, specific). The
induction story therefore lands in its true form: each of these
heads computes a SINGLE SPARSE READ chosen by its double-QK
coincidence score; the top read is most of the function; the
residual ~23% is the diffuse tail plus off-match modes (369). This
is the program's first circuit-grain executable mechanism to pass
registered replication bars -- with the honest caveat that
"the code" includes the pattern computation, which is legitimate
here because the double-QK score is closed-form in the stream (the
tensor-native standard, per user direction).

## 371. The functional read dictionary: sparse reads are a minority idiom (69/162)

head_read_census extended the same metrics to all 162 heads. Both
bars FAILED informatively: 69/162 heads have top-1 share >=0.4
(bar 100 -- sparse reading is a substantial minority, not the norm)
and functional-modal agreement with motif labels is 50/76 = 66%
(bar 70% -- close but under). Scope statement recorded: pattern-as-
code applies to the ~43% sparse-reader minority; the diffuse
majority needs the full pattern-weighted sum (which is still
closed-form, just not one read). The per-head functional profiles
(top-1 share + read-location distribution) are written to the
results JSON -- the functional dictionary that the pattern-matrix
dictionary (364) failed to be. Follow-ons queued: sparse_code_all
(replace ALL 69 sparse heads with one-read code; bars: global fresh
CE <=+0.15, IOI >=70%, diffuse-head matched control >=3x -- the
benchmark cash-in of 370) and violator_probe (r.6.0.0, the one
value-sensitive candidate from 362: which bundle's MEAN carries
function).

## 372. One-read code at scale: +0.49 for 69 heads, and an instructive inversion

sparse_code_all replaced all 69 sparse-reader heads with one-read
code. Bars (a),(c),(d) FAILED, (b) HELD: grid cost +0.486 (bar
0.15), IOI 62% (bar 70), fresh transfer clean (+0.462, travels).
The instructive part is the CONTROL INVERSION: the 69 most-diffuse
heads cost only +0.232 under the same code -- LESS than the sparse
set. Reading: top-1 share measures how much of a head's output one
read carries, not how much the model needs that head; the sparse
readers are largely the motif heads, which are the load-bearing
ones. So "sparse" ranks compressibility of form, not fitness for
substitution. The usable numbers stand anyway: 138/162 heads (both
sets) can run on one-line code for a combined ~+0.7 nats, against
the mid-attention band's ~0.9-nat benchmark weight -- head-grain
readable code is now a priced benchmark rung. Follow-on queued
(head_code_frontier): per-head one-read substitution cost for ALL
162 heads, then the cheapest-first coverage curve -- the
coverage-vs-fidelity frontier at head grain with readable code,
bars: >=60 heads within +0.15 total; fresh <=1.5x; full curve
reported.

## 373. The projection-law violator dissolves: offset tax up to 2.8x, no new channel type

violator_probe decomposed r.6.0.0 per bundle: the two a9 bundles
show delete/mean-assign ratios of 2.72 and 2.82 (just under the
registered 3x for "mean-carried"), the other two are variance-
carried (1.13, 1.65) -- (a) FAILED, (b) HELD. Verdict: r.6.0.0 is
not a qualitatively different channel; it is the projection law's
offset-tax term at its observed maximum (~2.8x on specific attention
bundles). The law's final form absorbs it: variance-removal
dominant everywhere; offset tax ranges ~1.1-2.8x by bundle, largest
on attention-side bundles. Per-bundle table merged into the r.6.0.0
circuit record. No value-sensitive circuit remains on the books.

## 374. MILESTONE: 80 heads run on one-line code at NEGATIVE cost

head_code_frontier measured every head's individual one-read
substitution cost and built the cheapest-first curve. The headline:
**substituting the 40 cheapest heads IMPROVES the model (-0.077
nats on the census grid), and 80 heads -- half the attention stack
-- still nets -0.043.** Fresh leg at 80 heads: +0.064 (the improve-
ment does not fully travel, but half the stack on readable code for
+0.06 fresh is a benchmark rung in itself; registered bar (c) was
ill-posed for negative grid costs and is recorded as such). At 120
heads the joint cost is +0.168 vs -0.014 sum-of-singles -- bar (b)
FAILED: interaction penalties appear only at high coverage, the
first measured departure from this model's otherwise relentless
additivity. Sum-of-singles predicts 153/162 heads within +0.15.
The negative costs are slack-harvest by another route: one-read
truncation removes diffuse-tail noise from cheap heads. And the
induction heads' position on this curve answers where the LOAD is:
5 of 9 rank BEYOND 120th cheapest -- they are among the hardest
heads in the model to compress, which is why the program kept
tripping over them.

## 375. Novel-circuit code verdict (r.3.0): 59% mean / typical-member good / weak key-specificity

circuit_code_r30 replaced r.3.0's two named heads (16.8, 16.2) with
one-read code. Mixed verdict, all on record: mean member |dCE| 0.209
vs 0.505 for deletion -- the code carries 59% of the machinery's
function (bar wanted 70%, FAILED); the MEDIAN member is well-
replicated (bar HELD -- the mean is dragged by a heavy tail of
badly-coded members); and the shuffled-key control only costs 1.7x
the true code (bar 2.5x FAILED): for these heads, WHICH key is read
matters less than that something is read with the right magnitude
-- weaker key-specificity than the induction band. Honest grade for
r.3.0: partially computational -- typical members replicate under
executable code, the tail and the key-specificity gap are named
residuals. Both feed the same question as 370's +0.138, now being
anatomized (induction_residual, running): multi-match reads,
context corruption from off-member substitution, and tail
composition.

## 376. THE INDUCTION STORY CLOSES AT COMPUTATIONAL GRADE: four reads, 99% replication

induction_residual anatomized 370's +0.138 residual. The top-k
curve collapses immediately: **top-1 +0.129, top-2 +0.024, top-4
+0.0073, top-8 +0.0041** against +0.601 for deletion -- (a) HELD
with 7x room. Two reads recover 96% of the heads' function; four
reads 98.8%, passing even mech_replicate's original strict bar
(+0.05) six-fold over. The other two hypotheses FAILED informatively:
the extra reads are NOT more match occurrences (only 19.8% of the
dropped pattern mass sits on the match family) and there is no
context-corruption effect (match-only substitution is slightly
WORSE than uniform code, +0.147 vs +0.129 -- downstream prefers a
consistent head over a mode-switching one). Final story, executable
and validated: **each induction head computes its top 2-4
double-QK coincidence reads; the top read carries 77%, the second
most of the rest; the reads beyond the first are the head's next-
priority coincidences wherever they land, not additional match
copies.** The census induction band is hereby the program's FIRST
COMPUTATIONAL-GRADE circuit: code = "sum the 4 largest
coincidence-scored reads", replication 98.8%, shuffled control
broken, all preregistered. Recorded in the registry as ind_band
with mechanism_level=computational. Queued: topk4_stack -- the same
4-read code applied to ALL 162 heads (registered: grid <=+0.10,
fresh <=+0.15, IOI >=85%, r.3.0 members <=15% of deletion). If that
holds, the entire attention stack of bilin18 runs on four lines of
readable code.

## 377. Whole-stack code: +0.22, and the cheapest-80 frontier stands as the optimum

topk4_stack applied 4-read code to all 162 heads: grid +0.221 (bar
0.10 FAILED), fresh +0.343 (FAILED), IOI actually IMPROVED (+2.574
vs +2.408, HELD), and r.3.0's members cost 1.37 -- 2.7x its own
heads' deletion (FAILED): the diffuse heads its bundles depend on
do not survive 4-read truncation. Standing benchmark statement:
readable 4-read code covers the cheapest ~80 heads at ~zero cost
(374) and the whole stack at +0.22; the last ~40 heads carry
interaction structure that 4 reads cannot hold. A cheapest-N-at-k=4
sweep is queued to find the largest head set within +0.10.

## 378. CENSUS SCALE WARNING: the partition is not corpus-stable (9% identity)

census_scale rebuilt the tree at 2x corpus (212 fit + 212 fresh
rows). Both substantive bars FAILED loudly: **29 leaves** (vs 68 at
1x -- the recipe's thresholds do not scale) and **cross-instance
identity 6/68 = 9%** at J>=0.5 on the shared grid. Reading: the
leaf-level objects are largely WINDOW-RELATIVE -- the old tree's
clusters reflect the fit window's document family (one travel
guide) as much as the model's structure; mixing in diverse text
dissolves most of them. This vindicates every corpus-generality
caveat on record and REVISES THE SWARM PLAN: naively scaling the
census does not multiply certified circuits; it produces a
different, coarser partition. Before mass production, the census
recipe needs a stability diagnosis: census_stability queued --
rebuild on the fresh 212 rows ALONE (same size, disjoint corpus)
and compare (a) leaf count at matched size (was it size-scaling or
corpus mixing that collapsed the tree? registered: >=50 leaves if
size was the confound) and (b) identity between the fresh-alone
tree and the 424-tree restricted to the same fresh rows (same
data, different clustering context; registered fork: >=40% = recipe
stable / below = clustering itself is context-dependent). The
swarm's unit of production may need to become "window-replicated
leaf" rather than "leaf".

## 379. Head-code frontier final: 120 heads at NEGATIVE cost, 140 within +0.08

topk4_frontier swept cheapest-N at k=4: N=100 -0.034, N=120
**-0.026** (still improves the model), N=140 +0.076, N=162 +0.221.
Bar (a) HELD with room: 140 of 162 heads -- 86% of the attention
stack -- run on four-line readable code within +0.08 on the grid.
Fresh at 140: +0.169 (bar was 2x grid = 0.152, FAILED by 0.017 --
recorded, and honestly close). Standing benchmark statement:
**bilin18's attention is now describable as: 120 heads = readable
4-read code at zero-or-negative cost; the last 40 heads (including
most of the induction band and the motif-head core) carry the real
attention computation.** The interpretive frontier of attention has
been reduced from 162 heads to ~40.

## 380. Second computational-grade circuit: r.3.0's heads close at k=8

circuit_code_r30_k8: ALL BARS HELD. The 8-read code for heads
(16,8),(16,2) costs 0.107 on r.3.0's members vs 0.505 for deletion
(79% of function, bar 70%), median-member replication held, and the
shuffled-key control costs 3.3x (bar 2.5x) -- key choice now
demonstrably matters, resolving 375's specificity worry (at k=1 the
single read was too impoverished for the control to separate). The
novel circuit's attention machinery joins ind_band at computational
grade: same code template, k=8. Emerging law of the code ladder:
k is a per-mechanism capacity dial -- cheap diffuse heads at k=4
gain, induction at k=4 replicates 98.8%, r.3.0's heads need k=8.
Registry updated.

## 381. Stability fork resolves on the harsh branch: clustering is context-dependent

census_stability (safe_svd rerun): fresh-212-alone yields **77
leaves** ((a) HELD -- the 2x-corpus collapse was corpus MIXING, not
size), but same-data identity vs the 424-tree is **4/77 = 5%**
((b) FAILED decisively). Even on identical rows, the tree carves
different clusters depending on what other data sits in the build.
Fairness caveat recorded: the 424 tree is much coarser (29 leaves)
and Jaccard punishes subset-containment; a containment metric is a
registered follow-up. But the operational conclusion is already
forced and now stands in the SOP: leaf member-sets are one sample
from a family of valid partitions, NOT stable objects. Circuit
records must anchor on MACHINERY + PROGRAM + CAUSAL PROFILE;
member-sets are evidence, not identity. The swarm's production
unit is revised accordingly: a certified record requires its
machinery's causal profile to replicate on a disjoint window (the
Ledger-22 discipline promoted from benchmark to census).

## 382. The MLP k-dial: 11% of quadratic units per position, and it travels

mlp_topk: ALL BARS HELD. Keeping the top-512 of 4608 rank-1
quadratic units per position across all six mid MLPs costs +0.055
on the grid and **+0.044 on fresh data** (travels perfectly);
random-512 costs +1.21 (22x -- the selection is doing everything).
Curve: k=32 +0.48, 128 +0.20, 512 +0.055, 1024 +0.021. The sparse-
active-set idiom now covers BOTH halves of the architecture:
attention = top-4..8 coincidence reads per head, MLPs = top-11% of
quadratic units per position. In both cases the "code" includes
computing the selection scores (closed-form in the stream), and the
readable payoff is the same: at any position the model's actual
computation is an ENUMERABLE LIST -- which reads, which quadratic
units -- rather than a dense mixture. Queued: mlp_topk_all (all 18
MLPs) and the COMBINED readable configuration (120 coded heads +
top-512 MLPs; registered <=+0.25 grid / <=+0.35 fresh) -- the
whole-model sparse-enumerable form. Also queued: unit-usage census
(is the top-512 set per-position dynamic or is there a static ~512
subset that nearly suffices -- dynamic sparsity vs a fixed circuit
subset, a sharp interpretive fork).

## 383. Containment verdict: re-carving is real (40%)

census_containment: only 31/77 fresh-alone leaves are >=0.7-
contained in a 424-tree leaf (median containment 0.59) -- bar
FAILED, so the 381 instability is NOT mere coarse-graining: trees
built in different contexts genuinely re-carve the same data. The
SOP identity revision stands at full strength: machinery + program
+ causal profile is the circuit; member-sets are evidence. The
census's role in the program is now firmly the targeting system for
code-grade mechanism work (which is prospering: two computational-
grade circuits, both architecture halves on sparse code), not a
stable ontology of parts.

## 384. The combined readable model HOLDS: +0.22 fresh, whole model enumerable

combined_readable: top-512 quadratic units in all 18 MLPs + 4-read
code on the 120 cheapest heads = **+0.216 grid, +0.224 fresh, IOI
IMPROVED** (bars b,c,d HELD; the all-18-MLP-alone bar failed at
+0.241 -- front/tail MLPs are less sparse than the mid band -- yet
the combined config costs LESS than MLPs alone: the coded heads
interact favorably). Standing statement: bilin18 now runs in a form
where EVERY position's computation is an enumerable list -- <=4
reads for 120 heads, dense attention for the load-bearing 42,
<=512 active quadratic units per MLP -- at +0.22 nats on never-seen
text. This is a different kind of benchmark object than the
assembly (weights kept, computation paths sparsified, no fitting),
and it is the program's strongest whole-model interpretive form.

## 385. Routing is real: no static subnetwork explains the MLPs

unit_usage: the 512 most-used units capture only 18% of top-512
slots, and substituting the STATIC most-used 512 costs +0.516 vs
+0.055 dynamic (9x) -- both bars FAILED in the direction that
matters: mid-MLP computation is genuinely PER-POSITION ROUTED among
thousands of units, not a fixed circuit subset. (Reconciled with
the older half-units result: a static HALF (2304) suffices; a
static 512 does not -- the routing pool is wide.) Interpretive
consequence: MLP-side semantics must be sought in the ROUTING
FUNCTION (which contexts activate which units -- the census's
slice-conditioned directions were exactly this) rather than in a
small fixed unit set.

## 386. Eval-data provenance corrected: training data is FINEWEB; Pile fresh legs were mildly OOD

User correction, verified on-box: bilin18 was trained on FineWeb.
The fit window (FW = bilin18_eval_tokens_large.pt) is FineWeb-
derived (sibling qk_mdl scripts load data_fineweb_tokens.npy under
the same FW name; the source npy no longer exists on this recycled
box). Every FRESH leg to date used NeelNanda/pile-10k -- i.e., the
program's transfer tests were run mildly OUT-OF-DISTRIBUTION.
Measured magnitudes: FineWeb-fresh base CE 3.23, Pile-fresh 3.33
(+0.10 OOD tax), and the census fit window is the real outlier at
4.01 -- the travel-guide document is hard text even in-distribution.
Consequences, stated plainly: (1) transfer results STAND and are
conservative -- anything that traveled to Pile passed a harder test
than intended; (2) absolute fresh numbers carry a ~+0.1
distribution component; (3) the in-distribution fresh standard is
now census_lib.fineweb_rows() (streaming, dedup'd), pile fresh_rows
stays as the deliberately-harder OOD leg, and every fresh number
must say which distribution it used (LESSONS rule added); (4)
census_diverse (queued, not yet started) patched to build its
1000-row corpus from FineWeb.

## 387. The induction score's semantic input found: mlp0 writes the matching code

qk_writer_decomp decomposed each induction head's score factor into
exact (query-writer x key-writer) pairs at its top reads. Registered
bar (a) held only for the early band (1.4 at 1.00, 2.5 at 0.98, 3.5
at 0.86, 3.8 at 0.80; the deep heads 5.5-8.4 sit at 0.36-0.40 --
FAILED as registered, mixed context inputs). But the discovery is
the DOMINANT PAIR, unanimous across all nine heads: **m0|m0**. The
coincidence score compares mlp0's output at the query position with
mlp0's output at the key position -- the matching key is not the raw
token embedding but mlp0's token-enrichment of it. Deep heads add
m2|m2, m3|m0, m0|a4 side terms (context creeping into the match
code with depth). Why this matters: mlp0 is an exactly-foldable
token table (the front-of-model result), so for the early band the
match trigger should be EXACTLY computable from token pairs alone:
score ~ (Wq . m0fold(t_q)) . (Wk . m0fold(t_k)) x (same for QK2).
fold_score_test queued: (a) fold-only score's argmax predicts the
real top-read location >=40% at match positions for heads (1,4),
(2,5),(3,5); (b) fold-pattern/real-pattern correlation >=0.4 there;
(c) deep heads reported (expected lower). If (a) holds, the early
induction circuit is COMPLETE under the user's definition:
understood input variables (m0's fold table) + understood
computation (double-QK bilinear coincidence + top-4 reads) +
validated code (376). mlp0's role in the model sharpens: it is not
just the biggest token table -- it is the model's IDENTITY-CODE
GENERATOR, writing the representation that all match machinery
compares.

## 388. Fold-only trigger: partial (head 2.5 at 45%), rotary omission named as the gap

fold_score_test predicted head reads from m0-fold codes + QK
weights alone, WITHOUT rotary. Bars FAILED, but far from null:
head 2.5 hits the real top read 45% of the time (chance 0.8%),
1.4 19%, 3.5 7.5%, deep heads ~1%. The design dropped the rotary
rotation entirely -- and rotary is position-deterministic, hence
exactly computable per pair: the omission is a fixable gap, not a
fundamental one. fold_score_test2 queued: same construction with
rotary applied at actual positions (still weights+tokens+positions
only, no model forward). Registered: early band hit >=40%, corr
>=0.4 with rotary included. Deep heads expected to stay low
(context terms in the match code, per 387).

## 389. Ops: diverse census outgrew the runner timeout; runner upgraded

census_diverse hit the 90-min runner cap at root r.22 (exit 124,
state unsaved) -- the 1000-row corpus exposes far MORE damage
structure than the old window (22+ root modes vs 9), which is
exactly why it timed out and exactly why it is worth rerunning.
Runner timeout raised to 4h and restarted while idle, which also
activates the queue dup-guard (the phantom re-append bug's
mitigation, pending since 19:3x). census_diverse requeued
unchanged.

## 390. THE DIVERSE CENSUS LANDS: 311 leaves, document dominance broken

census_diverse completed on the 4h runner (105 min). The curated
corpus: 1000 rows / 513k tokens from FineWeb (the training
distribution), at most 2 rows per document -- ~5x the old window,
and built specifically because the old window was dominated by one
travel guide. All registered bars HELD:
- (a) 311 leaves (bar: >=100). The old same-size-window tree had
  ~100; diversity does not thin the structure, it multiplies it --
  22+ root damage modes vs the old 9.
- (b) median leaf's top-single-document member share is 1% (bar:
  <=50%). No leaf is a disguised document detector anymore; the
  old guide-dominance is fully broken. This retro-validates the
  identity-rule revision (381): what survives corpus change is
  machinery, not member lists.
- (c/d) state saved (census_state_diverse.pt, packs in
  circuit_tree_packs_diverse.json), every pack >=12 contexts.
Child replication rate 0.85 within-tree. The A/B disjoint-half
machinery-replication leg was deferred at registration time;
census_ab_replication is now queued as its own registered
experiment (below). Class labels attach weakly at this scale
(class_r2 mostly <0.05) -- consistent with 381: token-class is a
profile feature, not the partition principle.

## 391. Fold trigger v2: rotary was MOST of the gap, but the bars still fail

fold_score_test2 added the deterministic rotary rotation to the
weights+tokens fold score. Registered bars: (a) FAILED -- but
marginally and unevenly: head 1.4 jumped 19%->62%, 2.5 45%->52%,
3.5 7.5%->39.1% (the bar was >=40% on all three; 3.5 misses by
0.9 points). Chance is 0.8%. (b) FAILED: rank corr 0.28/0.14/0.24,
all under 0.4 -- the fold score finds the argmax read half the
time but does not reproduce the full pattern shape. (c) held: deep
heads 1.8%/3.3% (context lives in their match code, 387).
Post-hoc diagnosis (registered forward in v3): v2's fold code was
mlp0's write ALONE. The residual entering early attention also
carries the token embedding's direct path and the lambda skip --
all token-computable. fold_score_test3 queued: full attention-free
forward (wte + every mlp fold + lambda mixes, rotary at real
positions), registered (a) all three early heads >=40% hit, (b)
corr >=0.4 on >=2/3, (c) strict improvement over v2 on all three,
(d) deep heads stay <10%.

## 392. Fold v3: the token-computable score is exhausted at ~2/3

fold_score_test3 used the full attention-free forward (wte direct
path + every mlp fold + lambda mixes + rotary). All four registered
bars FAILED: 1.4 62->66.5%, 2.5 52->53.1%, 3.5 DROPPED 39.1->35.7%
(higher mlp folds evaluated off-distribution add noise -- more
evidence that m0 is THE identity code and later folds are poor
token approximations), and deep head 8.4 rose to 11.8% (above the
10% bar). Verdict: tokens+weights+positions cap at roughly 2/3 of
early-band trigger prediction. The missing information is
contextual by elimination -- and the only contextual quantity
below layer 1 is attention 0. fold_gap_locate queued with an arm
ladder to locate it exactly.

## 393. LAYER-1 INDUCTION TRIGGER CLOSED EXACTLY: the gap was m0's contextual input

fold_gap_locate ran three arms (fold baseline / real-m0+wte with
a0's residual write dropped / real residual through block 0):
  1.4:  arm1 66.5% -> arm2 99.8% (corr 0.999) -> arm3 100%
  2.5:  arm1 53.1% -> arm2 57.8% -> arm3 57.9%
  3.5:  arm1 35.7% -> arm2 43.3% -> arm3 43.3%
Registered (a)/(b) FAILED as written (only 1.4 crossed 80%/0.6);
(c) monotone ladder HELD; (d) arm3 sanity HELD. But the arm
pattern is the discovery:
- For the layer-1 head, arm2 is ESSENTIALLY EXACT. The whole v3
  gap was m0's contextual INPUT (m0 really reads rms(E+a0), not
  rms(E)); a0's residual write adds nothing to the pattern
  (arm2=arm3). The layer-1 induction trigger is now fully
  accounted mechanistically: match code = wte + m0(identity code,
  locally contextualized by a0) -> double-QK coincidence -> top-4
  reads -> validated executable code (376). This completes the
  input side of the early induction circuit under the user's
  standard: known input variables, known computation, verified
  code.
- For layers 2-3, arm2=arm3 means block-0 context is NOT the
  binding gap: their match codes accrete writes from layers 1-2
  (the m2|m2, m3|m0 side terms of 387). The trigger's context
  depth grows with head depth -- same gradient the deep band
  (5.5, 8.4 at ~0 fold hits) extends.

## 394. Census machinery replicates across corpus halves -- at coarse grain

census_ab_replication ran the deferred identity-rule leg on the
diverse tree: 45 leaves sampled (16/depth), each leaf's 4-probe
member-mean dCE profile computed independently on corpus halves A
(rows 0-499) and B (500-999), sampled-row capped.
- (a) HELD: 77% of leaves replicate at profile cosine >=0.7 (bar
  60%).
- (b) HELD: matched median 0.955 vs mismatched-leaf null 0.544 --
  the profiles are leaf-specific, not a generic all-positive
  artifact.
- (c) FAILED (registered fork, informative branch): depth<=1
  replicates at 88%, depth>=2 at 56%. Identity is COARSE-GRAIN
  WEIGHTED: root/depth-1 machinery is corpus-stable; depth-2/3
  probes are partly window-fit. Matches 381's context-dependence.
  Consequence for the swarm: the A/B cosine is a computable
  per-leaf certification gate. census_ab_full queued (all 311
  leaves, registered a-c) to produce the certified production
  shortlist.

## 395. Full-tree A/B certification: 165 leaves certified; the tree is depth-heavy and fine grain replicates at 46%

census_ab_full ran the A/B gate over all eligible leaves. The
full-tree replication rate is 0.53 -- (a) FAILED against the
sampled 0.77, and the reason is composition, not contradiction:
the per-depth sample (16/depth) overweighted shallow leaves, and
the full tree is mostly deep. Depth split confirmed at scale ((c)
HELD): 72% at depth<=1 vs 46% at depth>=2. Specificity stays
strong (matched median 0.736 vs mismatched 0.271). The production
number: **165 leaves certified at cos>=0.7** ((b) HELD) -- the
swarm's shortlist. Standing rule for circuit claims on the diverse
tree: claim certified leaves first; an uncertified leaf's record
must carry its A/B cosine as a warning label.

## 396. The 2.5 trigger fork resolves M1-MEDIATED; the MLP-ladder hypothesis is on the table

fold_gap_locate2, per-component closure of the layer-2/3 triggers:
  2.5: real-b1-input+m1_fold 0.579 | +a1 write 0.585 | +m1 real
       0.858 | full real 1.0
  3.5: real-b2-input+m2_fold 0.590 | +a2 write 0.668 | +m2 real
       0.779 | full real 1.0
Sanity and monotonicity HELD; the single-component bar (0.9)
narrowly FAILED at 0.858. The fork answer is unambiguous:
**m1-mediated**. Attention's residual write adds ~nothing to the
match code at either depth (0.579->0.585 at 2.5); what closes the
trigger is the MLP's real write. Together with 393 (a0's write
irrelevant at 1.4), the generalization writes itself: the
induction match code is a CUMULATIVE MLP LADDER -- wte enriched by
m0, m1, m2... with attention contextualizing the MLPs' inputs but
never carrying the code in the residual. mlp_ladder_code queued:
rebuild the residual with all attention writes deleted and all
real MLP writes kept, plus the inverse control (attention writes
only). Registered: 2.5>=0.85, 3.5>=0.75, deep-reach fork at 0.5
for 5.5/8.4, control <0.2, and 1.4>=0.99 for 393 consistency.

## 397. The early induction match code IS the MLP ladder; the deep code is not

mlp_ladder_code rebuilt the residual with every attention residual
write deleted and every real MLP write kept (attention still
contextualizes MLP inputs), then predicted each head's pattern:
  1.4: 0.998 (corr 0.999)   2.5: 0.859 (corr 0.77)
  3.5: 0.741 (corr 0.651)   5.5: 0.362   8.4: 0.504
Scored honestly: (d) HELD (393 consistency); (a) FAILED by 0.009
(3.5 at 0.741 vs the 0.75 bar; 2.5 cleared its 0.85); (b) deep
reach FAILED (5.5 well under 0.5); (c) control FAILED as
REGISTERED but the registration was mis-set: the "attention-only"
arm retains wte through the lambda mixes, and wte itself is
identity signal -- it scores 0.36-0.59, not <0.2. The correct
control statement: at fixed wte base, swapping MLP writes for
attention writes drops 1.4 from 0.998 to 0.594 and 2.5 from 0.859
to 0.479 -- the MLP writes are the code carrier, decisively, but
the <0.2 bar was wrong to register against a wte-bearing arm.
Standing summary: **for the early band, the induction match code
is wte progressively enriched by the MLP chain (rank corr 0.77 at
2.5 vs 0.18 for token-only folds); attention residual writes
carry none of it. The deep band (5.5, 8.4) reads match content
that attention DOES write into the residual.** deep_trigger_source
queued: add one attention layer's real write at a time to the
ladder and localize which layer writes the deep match content
(registered: a4 leads for 8.4, per 387's m0|a4 term;
concentration fork; 2.5 as no-lift control).

## 398. Deep trigger sources: 5.5 is ladder+a4 (one layer, 78% of the gap); 8.4 is diffuse; a1 finishes 2.5

deep_trigger_source added one attention layer's real write at a
time to the MLP ladder:
  2.5: ladder 0.859; +a1 -> 1.000 (+0.141); +a0 -> nothing
  5.5: ladder 0.362; +a4 -> 0.860 (+0.498; 78% of the gap);
       +a3 +0.11, everything earlier ~0
  8.4: ladder 0.504; diffuse: +a6 +0.124, +a7 +0.123, +a5 +0.100
Scored: (b) HELD -- 5.5 has a CONCENTRATED attention carrier, a4.
(a) FAILED -- 387's m0|a4 term belongs to 5.5, not 8.4; 8.4's
match content accretes diffusely over a5-a7. (c) FAILED
informatively: with m1's real write present, a1's write closes
2.5 COMPLETELY (locate2's B-arm hid this by pairing a1 with
fold-m1: the two writes matter jointly, not separately). Revised
picture: every induction head matches on the MLP-ladder code plus
a head-specific attention side-channel whose weight grows with
depth -- 1.4 pure ladder, 2.5 ladder+a1 (14%), 5.5 ladder+a4
(58%), 8.4 ladder+diffuse-mid-attention. Next question queued
(deep_code_content): is a4's contribution NEW content or the
ladder code RELAYED from other positions? Arms: a4 with real
patterns but ladder values (relay test), per-head restriction of
a4's write, sanity. If the relay arm holds, the story closes as
ONE code -- built by MLPs, moved by attention.

## 399. THE RELAY RESULT: a4's channel into 5.5 is the ladder code, moved

deep_code_content tested whether a4 writes new match content or
relays existing code. Arms at head 5.5: ladder 0.362; ladder +
a4-real 0.860 (sanity HELD); ladder + a4 with real patterns but
VALUES READ FROM THE LADDER RESIDUAL: 0.837 -- the relay arm
recovers 95% of a4's lift ((c) HELD). (b) FAILED: no single a4
head carries the relay (h7 0.168, h5 0.160, h0 0.143, h3 0.105 --
four heads share it). Conclusion, now covering heads 1.4, 2.5 (up
to its a1 term), and 5.5: **the induction machinery of bilin18
compares ONE identity code -- written by the embedding,
progressively enriched by the MLP chain, and moved between
positions by attention. Attention contributes no code content of
its own at 5.5: its channel is 95% relay.** relay_closure queued
to close the last two heads under the same story: a1's channel
into 2.5 (relay arm), 8.4's diffuse a5-a7 channel (combined real,
one-shot relay, ITERATED relay -- nested moves of the same code --
and a shuffled-value null), registered a-d.

## 400. RELAY CLOSURE: ALL FOUR BARS HELD -- the induction match code is one code

relay_closure, all registered bars HELD:
  2.5: ladder 0.859 | a1 real 1.000 | a1 RELAY 0.999
  8.4: ladder 0.504 | a5-7 real 0.859 | one-shot relay 0.617 |
       ITERATED relay 0.841 | shuffled-value null 0.400
(a) 2.5's channel is pure relay (0.999). (b) a5-a7 jointly close
8.4 (0.859). (c) the iterated relay recovers 95% of the lift --
8.4's channel is a NESTED relay: attention moving code that was
itself already moved (one-shot relay only reaches 0.617, so the
nesting is real and measurable). (d) shuffled values land BELOW
the ladder baseline: the relay carries position-specific code,
not statistics. Standing conclusion across every head tested
(1.4, 2.5, 5.5, 8.4): **the induction machinery of bilin18
compares ONE identity code -- written by the embedding, enriched
layer-by-layer by the MLP chain, and moved between positions by
attention (deep heads read relayed, even doubly-relayed copies)
-- via double-QK coincidence, executing its top-4 reads.** Queued:
ladder_census -- the maximal claim on all NINE band heads (3.8,
6.5, 7.3, 8.3 still untested): every attention write below every
head replaced by its iterated relay; registered early >=0.90,
deep >=0.75, shuffled null <25% of lift. After that, the causal
cash-in (ladder-computed patterns driving the live model at match
positions) closes the arc end-to-end.

## 401. INSTRUMENT FAILURE: ladder_census's iter_all arm was the identity in disguise

ladder_census returned iter_all = 1.000, corr = 1.000, on all
nine heads -- and that perfection is the tell. With relay writes
inserted at EVERY block and values drawn from the growing chain,
the reconstruction equals the real residual by induction (real
patterns + values-from-real-so-far = the real attention write,
block after block). Bars (a)/(b) held VACUOUSLY; the run is void
as a test of the one-code claim (the shuffled arm remains
meaningful: shuf tracks the ladder baseline, so patterns alone
carry little without matched values). Recorded per the standing
red-team rule: an arm that cannot fail is not evidence. The
honest maximal claim bounds RELAY DEPTH instead: chain_k allows
values only from chain_{k-1} (k moves of the code), chain_0 =
pure MLP ladder. ladder_depth queued with registered depth
predictions: early band closes at k=1; 5.5 at k=1; 8.4 needs
k=2 (nesting real); all nine >=0.85 by k=3 (bounded-depth fork);
shuffled-value null at k=1.

## 402. RELAY DEPTH MEASURED: every induction head closes by three moves of one code

ladder_depth, the corrected bounded-depth instrument: ALL FIVE
BARS HELD.
  head   k0     k1     k2     k3     k1-shuffled
  1.4   0.998  1.000  1.000  1.000  0.998
  2.5   0.859  0.999  1.000  1.000  0.923
  3.5   0.741  0.942  1.000  1.000  0.749
  3.8   0.793  0.941  1.000  1.000  0.774
  5.5   0.362  0.893  0.960  0.994  0.348
  6.5   0.535  0.827  0.956  0.989  0.452
  7.3   0.415  0.683  0.938  0.986  0.422
  8.3   0.538  0.694  0.926  0.982  0.469
  8.4   0.504  0.623  0.929  0.976  0.393
Relay depth is a measured, small integer that grows with head
depth: the early band closes at ONE move of the MLP-ladder code,
layers 5-6 at one-to-two, layers 7-8 at TWO, and by three moves
all nine heads sit at >=0.976. The shuffled-value null never
leaves the ladder baseline. This completes the input-side
anatomy of the induction band: match code = MLP-ladder identity
code, moved at most ~2-3 times by attention, compared by
double-QK coincidence. ladder_causal queued -- the causal
cash-in: the LIVE model runs with all nine band heads' patterns
computed from the k=2 reconstruction (real values), priced in CE
at match positions against the 4-read code, a shuffled-content
control, and deletion. Registered a-d.

## 403. CAUSAL CLOSE: the live model runs on computed induction triggers at zero cost

ladder_causal replaced all nine band heads' patterns in the LIVE
model with patterns computed from the k=2 reconstruction (wte +
MLP chain + two attention moves of the same code; values real):
  ladder-trigger: match -0.0019, off-match +0.0007
  shuffled-content control: match +0.0731
  deletion anchor: match +0.5008
All four bars HELD -- (a) far under 0.05, (b) under 10% of
deletion trivially, (d) surgical off-match. One arithmetic
footnote: bar (c) (shuffled >= 6x ladder) degenerated because the
ladder cost is NEGATIVE; the substantive contrast (0.073 vs
-0.002 vs 0.50) is unambiguous. THE INDUCTION ARC IS CLOSED
END-TO-END: output side = 4-read code (376, 98.8%); input side =
one identity code, MLP-built, attention-moved at most twice
(397-402); and the model actually RUNS on the computed triggers
(this section). Report updated with the causal number. Next arc
opened: sop_batch_certified queued -- does the A/B certification
gate (395) predict the SOP concentration gate? Top-24 certified
vs 24 uncertified leaves, SOP steps 1-2 batch-computed, packs
written for passers (registered a-c).

## 404. GATE INVERSION: A/B-certified leaves FAIL the selectivity gate (17% vs 88%)

sop_batch_certified asked whether the A/B stability gate (395)
predicts the SOP concentration gate. It ANTI-predicts it: top-24
certified leaves pass concentration>=3 at 17%; 24 uncertified at
88%. Both registered bars failed in the inverted direction, and
25 partial packs were written (all from passers, mostly
uncertified -- (c) held). Diagnosis: the cosine of a RAW dCE
4-vector is trivially stable whenever one component is huge (the
top-certified leaves are exactly the whole-component root probes,
e.g. r.4's profile [8.08, 0.05, 0.00, 0.02]) -- so the 395 gate
certifies BIG GLOBAL damage, which is anti-selective by
construction, while selective leaves' small profiles have noisy
cosines. STANDING CORRECTION to 395's framing: raw-damage-cosine
certification is a STABILITY-OF-MAGNITUDE gate, not an identity
gate; the 165-leaf list must NOT be used as a swarm shortlist
as-is. Fix queued (gate_reconcile): replicate SELECTIVITY across
halves instead -- pass iff concentration>=3 on BOTH corpus halves
independently; registered agreement with the SOP gate,
production-pool size on 48 fresh leaves, both-halves stability of
SOP passers, and a label-shuffled null.

## 405. The selectivity gate reconciles: 94% SOP agreement; the pool is big

gate_reconcile scored the fixed certification (concentration>=3
on BOTH corpus halves independently):
- (a) HELD: 94% agreement with the full-corpus SOP gate on 403's
  48 leaves -- the inversion is resolved; the both-halves gate
  and the SOP gate measure the same thing.
- (b) HELD: 79% of 48 fresh leaves pass -- the production pool is
  far larger than the voided raw-cosine list implied (projected
  ~245 of 311).
- (c) HELD: 88% of SOP passers hold on each half separately --
  selectivity is not a corpus-average artifact.
- (d) FAILED by one point: label-shuffled null passes 6% vs the
  registered 5% bar (shuffled concentrations collapse to ~1.0;
  the 6% is a small tail, but the bar is the bar -- recorded as
  failed and the full-tree rerun's bar recalibrated to <=8%
  with the observed value stated here for transparency).
gate_full queued: the both-halves gate over ALL 311 leaves,
writing swarm_shortlist.json -- the production list that replaces
395's. Registered: rate within 15 points of 0.79; >=180 pass;
DEPTH fork (selectivity may be depth-uniform, unlike raw
stability); null <=8%.

## 406. THE SWARM SHORTLIST EXISTS: 199 leaves, and selectivity favors the FINE grain

gate_full ran the both-halves selectivity gate over all 311
diverse-tree leaves: 199 pass (64%; swarm_shortlist.json
written -- (b) HELD). (a) FAILED at the literal boundary (|0.64 -
0.79| = 0.150 vs <=0.15, lost to rounding; recorded as failed).
(d) HELD (shuffled null 6%). The scientific result is (c),
FAILED in the OPPOSITE direction from the raw-stability gate:
depth<=1 passes at 44%, depth>=2 at 72%. The two gates are
mirror images -- 395's raw-damage cosine favored coarse
whole-component probes (big, global, trivially stable,
anti-selective), while selectivity favors the FINE leaves. Read
together with 383 (re-carving) and 395: the census's fine
structure is real, locally selective, corpus-stable machinery;
what is unstable at fine grain is only the raw damage MAGNITUDE
profile. Caveat noted from the stream: depth-0 leaves have no
off-slice positions inside their own rows (conc is
floor-inflated, e.g. r.6 at 2.5e4); they are few and
production work should treat depth-0 concentration as
ill-defined rather than passing. The swarm now has its
production list: 199 certified-selective leaves. relay_heads
running next: who does the relaying (per-head relay lifts vs the
head census's prev-token shares).

## 407. THE RELAY HEADS ARE NAMED: a6.h3 moves the code for the whole deep trio

relay_heads localized each band head's top relay layer, then its
top relay HEAD, and cross-referenced the independent head-read
census:
  5.5 <- a4.h7 (lift 0.174; census modal PREV, prev share 0.765)
  7.3 <- a6.h3 (lift 0.135; modal PREV, prev 0.468)
  8.3 <- a6.h3 (lift 0.100; same head)
  8.4 <- a6.h3 (lift 0.118; same head)
ONE previous-token head, a6.h3, relays the identity code for all
three deepest induction heads; a4.h7 does it for 5.5. This is the
classic prev-token -> induction composition, recovered in
exact-code form with named parts. Scored honestly: (b) HELD (4/8
top relay heads prev-modal); (c) HELD (no 'first' enrichment,
rho -0.27); (a) FAILED (pooled rho 0.249 < 0.4) -- the pooled
correlation is diluted by the early band, whose relay lifts are
small and spread (2.5's biggest head lift is 0.054; its trigger
barely needs relays). The claim that survives: DEEP relays are
concentrated in named prev-token heads; early relays are diffuse
dribbles. relay_edge_causal queued: zero a6.h3 / a4.h7 in the
live model vs matched same-layer controls; registered selective
pattern damage on the deep trio / 5.5, and match-position dCE
ratios.

## 408. RELAY EDGES CAUSALLY VERIFIED: deleting a6.h3 breaks exactly the deep trio

relay_edge_causal zeroed the named relay heads in the LIVE model
vs matched same-layer controls. ALL THREE BARS HELD:
  delete a6.h3: top-read shift 29/21/19% on 7.3/8.3/8.4, and
    EXACTLY 0% on 1.4/2.5/5.5; match dCE +0.051 vs control
    a6.h0's -0.0003 (>150x)
  delete a4.h7: shifts 5.5 at 18.6% (plus 12-17% on the deep
    trio -- a4.h7 feeds the chain a6.h3 reads, consistent with
    nested relay), 0% on the early band; control a4.h2 about
    half its effect
The prev-token -> induction composition is now interventionally
verified with named parts: a6.h3 is the deep band's code courier;
cutting it selectively re-aims the three deepest induction heads
and costs CE at match positions, while its layer-mate control
does neither. The induction circuit record is complete at every
level the program defines: variables, computation, executable
code, trigger reconstruction, courier identity, and live causal
verification of each.

## 409. The gate transfers at 98%; pack inventory at 72; census_lib goes diverse

sop_batch_shortlist ran SOP steps 1-2 over 96 seeded shortlist
leaves (depth>=1; depth-0 excluded per the 406 caveat): 98% pass
the full-corpus concentration gate ((a) HELD, bar 80%) and 47 new
packs were written ((b) HELD; inventory now 72). (c) FAILED at
the ceiling -- deep 98% vs shallow 100%, a one-leaf difference;
the registered inversion prediction was unnecessary at
saturation. The both-halves selectivity gate and the SOP gate are
now interchangeable in practice. Infra shipped alongside:
census_lib gained use_state() and a parameterized grid (additive;
default behavior byte-identical), so every SOP function -- leaf
ablation, features, rule_search, programs -- now runs on the
diverse tree. sop_program_batch queued: SOP step 3 over all 72
packed leaves with STRICT doc-disjoint splits (docid parity, not
row parity -- rows of one document are adjacent in this corpus);
registered >=40% earn programs, median null <=0.6, programs
written for passers.

## 410. Surface programs fail on the diverse tree (1/72) -- and the standard reorients to mechanism

sop_program_batch ran SOP step 3 over all 72 packed leaves with
STRICT doc-disjoint splits (docid parity): only 1 leaf earns a
surface program at bacc>=0.75 ((a) FAILED at 1%; median null
0.504, honest). On the old tree, row-parity splits and a
guide-dominated corpus made token-class programs look workable;
on a diverse corpus with true doc-disjointness, leaf membership is
NOT surface-predictable. This lands together with a USER DIRECTIVE
(recorded to memory, was given before and lost in a context
break): compression facts -- the k-dial law, top-N unit counts --
and surface programs are not the goal. The standard is the EXACT
COMPUTATIONAL MECHANISM stated interpretably: named variables,
named writers, named couriers, causally tested -- the induction
record (393-408) is the template. Consequences shipped this turn:
CIRCUIT_SOP v2 (diverse tree setup, both-halves identity gate,
step-3 demoted, mechanism templates named as the actual goal),
census_lib diverse mode, memory entries, and a Sonnet dry-run
agent launched on r.0.3.0 to validate the swarm pipeline
end-to-end before Fable credits run out. Queue re-pointed at
mechanism work: r30_writer_decomp (what does the novel circuit
r.3.0 COMPARE -- registered: concentration, shared dominant pair
across its two heads, and the context-vs-identity-code fork).

## 411. r.3.0 COMPARES THE SAME IDENTITY CODE: m0|m0 at layer 16

r30_writer_decomp, the mechanism-first pass on the novel circuit:
both heads' dominant writer-pair is **m0|m0** -- near-unanimous
per read (16.8: 495 of ~500 top reads; 16.2: 446 of ~460; runner-
up m0|m3 in single digits). (b) HELD (shared dominant pair --
the census's bundling of the two heads reflects a shared score
input). (c) FAILED on the registered context branch, which is the
interesting outcome: r.3.0 does NOT compare mid-layer context
summaries -- it compares mlp0's identity code, sixteen layers
after it was written. (a) FAILED informatively: top-5 pair mass
is only ~0.30 (vs 0.6+ for the early induction band) -- at layer
16 the residual carries far more than the code, so m0's SHARE is
diluted even though it wins the argmax essentially always.
Emerging picture: the m0 identity code is the model's universal
comparison substrate -- the induction band (layers 1-8) and the
late novel circuit r.3.0 (layer 16) both score coincidences on
it. This replaces "r.3.0 closes at k=8" with an actual mechanism
sentence. ladder16 queued: does the bounded-relay reconstruction
(402) predict 16.8/16.2's patterns -- registered k3>=0.70 both,
monotone, shuffled null, and a seen-vs-fresh position split (are
they seen-before detectors?).

## 412. Swarm pipeline VALIDATED (first diverse-tree record) + ladder16's bars disqualified by their own null

Two results this cycle.

SONNET DRY RUN: a fresh Sonnet agent ran SOP v2 end-to-end on
r.0.3.0 and merged the FIRST diverse-tree circuit record
(circuits/r_0_3_0.json). Concentration reproduced (pack 6.66 vs
rerun 6.61; both-halves 6.60/6.61); story red-teamed 2/3; no tool
failures; git hygiene clean. Its four friction findings are all
fixed and pushed: (1) pack causal numbers are a 60-row subsample
-- SOP now says only concentration is expected to reproduce; (2)
pack examples are non-canonical -- SOP now says always regenerate
with cl.examples; (3) write_circuit stamped '212row-v1' on diverse
records -- census_lib now derives tree.instance from the active
state; (4) step-1 timing corrected to ~45s. The Sonnet handoff
surface is operational: SOP v2 + census_lib diverse + 199-leaf
shortlist + 72 packs + this validated loop.

LADDER16: bars (a)/(b) technically held (k3 = 0.98/0.976,
monotone) BUT the run's own null disqualifies them as content
claims: the shuffled-value arm at k1 recovers ~103% of the true
k1 lift (for 16.8 shuffled BEATS true). At layer-16 generic
positions, a relay write with real patterns and WRONG-ROW content
predicts the argmax as well as right content -- the instrument is
measuring write-magnitude/position structure there, not code
content. The k3 numbers are therefore NOT evidence for
one-code-at-16; that claim currently rests on the writer
decomposition (411) alone. ladder16b queued with shuffled arms at
EVERY k (registered content fork: k3shuf <= k3-0.25, else the
instrument is declared powerless at this depth and the claim
stays at 411's level).

## 413. Layer-16 content fork resolves PARTIAL; swarm runbook + reviewer-two shipped

ladder16b (shuffled arms at every k): (a) FAILED as registered --
content separates at depth for 16.2 (k3 0.976 vs k3shuf 0.651,
gap 0.33) but not for 16.8 (0.98 vs 0.806, gap 0.17 < 0.25 bar).
(b)/(c) held. Verdict recorded per the fork: at layer-16 generic
positions the ladder instrument's signal is majority-structural
with a real content increment that differs per head; the
one-code-at-16 claim stays at the writer-decomposition level
(411). Queued r30_read_semantics: the token-level mechanism
question (do 16.8/16.2's top reads land on tokens identical to
the query -- coincidence semantics -- with a frequency-matched
null and a seen/fresh split).

SWARM HARDENING (for the model swap): SWARM_RUNBOOK.md written --
new-session bootstrap (cron is session-only and must be
recreated; bqrunner is not), the wave loop (author waves capped
at 4 concurrent on the 32GB card, driver is the only committer),
standards that do not relax, author + REVIEWER-TWO prompt
templates, health checks, and the union of every trap hit so
far. Reviewer-two is an adversarial fresh agent per record:
recompute the gate, test the story on fresh seed-11 examples the
author never saw, hunt gerrymanders, verdict
CONFIRM/WEAKEN/REFUTE appended to the record's certification. A
Sonnet reviewer dry run was launched on r_0_3_0 to validate the
loop live.

## 414. Reviewer-two VALIDATED -- and its first objection becomes a rule; r.3.0's heads are local comparators

REVIEWER DRY RUN (r.0.3.0): verdict CONFIRM under the v1 rules --
gate reproduced byte-for-byte (concentration 6.61), fresh
seed-11 examples 4/5. But the reviewer's gerrymander hunt landed
the real finding: the story's SPECIFIC claim (helps rare/name
tokens) was never exercised by the fresh draw -- every hit came
from the catch-all "hurts ordinary words" branch, which is the
base-rate direction -- and the record's own program-heldout test
had already failed. Verdict rules upgraded to v2 in
SWARM_RUNBOOK: a story whose specific claim went untested caps at
WEAKEN unless a class-targeted draw passes 4/5; a failed
program_bacc raises the same requirement. The r.0.3.0 story is
flagged weak by the driver accordingly. Also fixed from the
reviewer's friction notes: certification dedup key
(test,source,date) must be populated or later reviews silently
drop; cl.examples needs d= passed. Both loops (author, reviewer)
are now validated live -- the swarm can run.

R30 READ SEMANTICS: all three registered bars FAILED,
informatively. At generic positions, 16.8's top reads are LOCAL
(offset -1 carries 549/1008; 76% within 2 tokens) while 16.2's
are diffuse mid-range; same-token reads are enriched 5-9x over
the frequency-matched null (7-11% vs 1.2-1.6%) but are a small
minority; the two heads disagree. (One metric was vacuous by
construction and is void: 'read lands in query history' is
trivially true for any causal read -- design flaw, stated.)
Combined with 411 (m0|m0 dominant): r.3.0's machinery compares
identity codes at SHORT RANGE -- same substrate as induction,
different range regime; 16.8 looks like a local identity-code
comparator (adjacent-token relations), not a match head. Queued:
stack_writer_decomp -- the 8 costliest code-resistant heads' score
inputs (registered diffuse-input hypothesis + universal-substrate
count + the mechanism worklist table for the swarm).

## 415. The code-resistant heads compare the SAME code: m0|m0 dominates 7 of 8

stack_writer_decomp decomposed the score inputs of the 8 heads
costliest under 4-read truncation (1.1, 12.6, 5.7, 1.8, 11.6,
10.5, 3.8, 9.8). The registered diffuse-input hypothesis is
WRONG: (a) FAILED -- only 3/8 have top-5 pair mass under 0.5, and
heads 1.1/1.8 are at 1.00, pure m0|m0. (b) HELD overwhelmingly:
7/8 dominant pairs are m0|m0 (the eighth, 5.7, is m0|m4 -- m0
still on the query side; ALL EIGHT read m0 as the query input).
Combined with 411 (r.3.0) and 387 (induction band), every
attention head examined so far scores its reads by comparing m0
identity codes. Mechanism reframe for "code-resistance": these
heads are not resistant because their score has many inputs --
the comparison substrate is the same one code -- but because
their FUNCTION integrates many reads (breadth of the read set,
not breadth of the score inputs). The next mechanism question is
value-side: what do wide integrators accumulate? Queue stocked
for the model swap with three registered experiments
(m0_code_geometry: substrate compactness + shared read subspaces;
local_bigram_score: is 16.8's offset -1 score a bigram-
plausibility signal; courier_centrality: is a6.h3 a specific
courier or a hub -- all with nulls/controls; note their
docstrings self-number 415-417, one behind this ledger).

## 416. Three registered results: compact code read universally; weak signed bigram split; a6.h3 is a HUB and a courier

m0_code_geometry: (a) HELD -- the identity code's effective rank
is 71 of 1152 (very compact). (b)/(c) FAILED informatively: the
band heads' 16-dim read subspaces overlap at 0.62 -- but so do
RANDOM heads' (null 0.68). With a rank-71 substrate, any 128-dim
head projection reads most of the code: the sharing is universal,
not band-specific, and 16 dims capture only ~55% per head. The
substrate is small; the reads are broad.

local_bigram_score: (a)/(c) FAILED, (b) held (shuffled-table null
clean). 16.8's offset -1 score correlates with bigram frequency
at only +0.16; 16.2 ANTI-correlates at -0.20 (prefers rare
adjacencies). A real signed split, but the bigram-plausibility
story is not supported at registered strength -- recorded as a
lead, not a claim.

courier_centrality: ALL BARS FAILED in the direction that
matters: deleting a6.h3 shifts EVERY downstream head (99/99 over
5%, median ~13%, up to 41% for 7.3) while the a6.h0 control
shifts nothing (90.9% under 5%). REFINEMENT OF 407/408 (stated
plainly): a6.h3 is not a narrowly-specific courier -- it is a
HUB whose write conditions generic read positioning across
layers 7-17, AND the match-content courier whose deletion
selectively re-aims the deep trio at match positions (408's
selectivity was measured at match positions; the global picture
is hub-like). courier_mean_split queued to separate the two
roles: mean-ablation (operator-C) should keep the bus load and
kill the content -- registered (a) mean halves the generic
shift, (b) trio match shift survives >=15%, (c) control clean.

## 417. The hub is not a bias: a6.h3's broad influence is position-specific content

courier_mean_split separated bus-load from content: (a) FAILED --
replacing a6.h3's write with its mean leaves the downstream
generic shift at 0.107 (vs 0.122 zeroed; the a6.h0 floor is
0.037), so the hub effect is NOT carried by the mean. (b)/(c)
HELD: the deep trio's match shifts survive mean-ablation
(0.18-0.33) and the control is clean. a6.h3 is a hub of CONTENT:
everything downstream reads its position-specific write. Together
with 399 (its channel is ~95% relayed ladder code at 8.4), the
closing hypothesis is that its entire downstream role is carried
by RELAYED IDENTITY-CODE VALUES. courier_content_id queued: live
model with a6.h3's values replaced by the pure-ladder
reconstruction (patterns real), vs a row-shuffled null and the
zeroed reference -- registered (a) generic shift collapses to the
control floor, (b) courier function preserved, (c) shuffled
values as bad as nothing.

## 418. The courier's payload is NOT just relayed code (all bars failed)

courier_content_id substituted a6.h3's values with the pure
MLP-ladder reconstruction in the live model. All three registered
bars FAILED, and the null is the reason: ladder values leave a
median generic shift of 0.071 (zeroing 0.122, control floor
0.037) -- only ~40% of the gap recovered -- while ROW-SHUFFLED
values recover ~22% (0.095). The content increment over the null
is real but small, and the trio's match shifts stay at 0.11-0.18
rather than collapsing. Correction to the closing hypothesis of
417, stated plainly: 399's "95% relay" result was about the
TRIGGER PATTERN of one downstream head (8.4), and it does not
generalize to a6.h3's whole downstream payload. Its broadcast is
mixed content, not the identity code alone. payload_decomp queued
to name the mixture exactly (writer-share decomposition of the
head's write, with the a6.h0 control).

## 419. WAVE-2 SWARM RESULT: 4 records, 4 reviews, and a standard change

Four Sonnet authors and four adversarial reviewers ran the full
loop. Gates: all four reproduced (6.45/6.57/6.67/6.84, several
exactly; both corpus halves independently above 3). Reviews: ONE
CONFIRM (r.11.3.1, borderline) and THREE WEAKEN (r.1.3.1,
r.4.1.0, r.23.2.1). No REFUTE -- the machinery is real in every
case; what failed was the STORIES:
- r.4.1.0: "helps concrete nouns" scored 3/5 on a class-targeted
  draw against a 48.6% base rate -- statistically indistinguishable
  from chance; two plain concrete nouns were hurt.
- r.1.3.1: the "clean word vs sub-word fragment" story is
  definitionally the tokenizer's leading-space bit; the trivial
  rule scores identically (6/10), and 2/5 on the named class.
- r.23.2.1: the flagged counterexample was not isolated -- a
  second clean case suggests continuation-of-a-started-name, a
  different mechanism than the claimed generic-vs-proper-noun.
Consequence (SOP v3, shipped): the swarm's deliverable is now the
MECHANISM step -- leaf_input_decomp.py decomposes the residual
entering each machinery component into exact writer contributions,
member vs off-slice, with ENRICHED/BEATS_NULL bars and an honest
negative branch. Behavioral stories are optional and must clear
cl.story_test's base-rate binomial on a mechanically filtered
class draw (cl.examples_filtered) -- both helpers added because
all three WEAKEN reviewers independently hand-built class draws
and flagged the variance. Wave 3 launched on the new deliverable.

## 420. The courier's payload is a LAYER property, not a head property

payload_decomp split a6.h3's write into exact writer
contributions: m0 0.271, the first-layer value stream (v1 lambda
term) 0.266, m3 0.139, m4 0.095, m2 0.076, a4 0.037. (a) HELD --
m0 is the single largest writer, so the relay is real; (b) HELD --
the payload is a MIXTURE (m0 well under 60%, six writers above
5%), which quantitatively explains 418: substituting pure ladder
code recovers ~40% of the effect because m0 is ~27% of what the
head carries (plus v1's early-value stream). (c) FAILED and it is
the most informative line: the CONTROL head a6.h0 carries almost
the same mixture (m0 0.361, v1 0.294, m3 0.109). Every head at
layer 6 reads the same residual, so payload composition cannot be
what makes a6.h3 the courier -- its PATTERN must be. Registered
crossover queued (pattern_payload_swap): give h3 the control
head's pattern with its own values, and its own pattern with the
control's values; predictions (a) pattern swap keeps the deep
trio's damage >=0.15, (b) value swap keeps it <=0.10, (c) neither
exceeds deletion. Also of note for the value-side program: a
quarter of a layer-6 head's write is the FIRST-LAYER value stream
carried forward by the lambda term -- an under-examined channel.

## 421. Crossover: BOTH swaps hurt as much as deletion -- instrument audit ordered before any conclusion

pattern_payload_swap gave h3 the control head's pattern with its
own values (patswap) and its own pattern with the control's
values (valswap). Neither arm behaved as registered: patswap
0.330/0.202/0.219 on the deep trio, valswap 0.336/0.199/0.190 --
both statistically the same as ZEROING (0.336/0.191/0.185). (a)
HELD only trivially, (b) FAILED, (c) FAILED (patswap even edges
above deletion on the generic median, 0.154 vs 0.122).
Two readings, and honesty requires testing the second before
publishing the first: either the courier role needs the exact
(pattern x values) product, or the top-read-shift metric
SATURATES under any perturbation of this magnitude -- in which
case every shift number in 416-421 measures "a perturbation the
size of h3's write" rather than content. shift_metric_audit
queued with the controls that decide it: a POSITION-PERMUTED
write (same norm, same values, wrong alignment) and a
NORM-MATCHED GAUSSIAN write, scored on three metrics (top-read
shift, graceful rank correlation of the whole read distribution,
and dCE at match positions). Registered: (a) permute also
>=0.15 -> the metric is non-specific and the caveat goes into the
ledger AND the published report; (b) CE still separates zeroing
from a random write by >=0.05 nats; (c) rank correlation
separates patswap from random by >=0.10. No claim about a6.h3's
pattern-vs-payload division is being recorded until this returns.

## 422. INSTRUMENT CORRECTION: top-read shift is not content attribution

shift_metric_audit settled 421 against my own headline. Controls
at a6.h3, deep trio (7.3/8.3/8.4):
  arm        top-read shift        rank corr      dCE(match)
  zero       0.311/0.205/0.183     0.70/0.78/0.79   +0.057
  patswap    0.332/0.214/0.219     0.67/0.73/0.74   -0.001
  valswap    0.317/0.217/0.193     0.67/0.76/0.77   +0.062
  permute    0.420/0.252/0.235     0.57/0.70/0.71   +0.042
  gauss      0.594/0.360/0.366     0.36/0.53/0.53   +0.143
(a) HELD: the position-permuted control shifts MORE than any real
arm, and a norm-matched Gaussian write shifts twice as much. The
argmax-shift metric measures perturbation SIZE, not content.
STANDING CORRECTION: every cross-arm shift comparison in 416-421
must be read as "a perturbation the size of this write", not as
content attribution. What survives untouched is WITHIN-arm
dissociation -- 408's finding that deleting a6.h3 shifts the deep
trio 19-29% while leaving the early band at exactly 0% is one arm
measured on different heads, so it stands. (b) FAILED (a random
write costs MORE than deletion: 0.143 vs 0.057 -- noise is worse
than absence). (c) HELD: rank correlation degrades gracefully and
separates real arms from noise. Instruments going forward: dCE
for function, rank correlation for pattern similarity; argmax
shift only within a single arm.

## 423. FUNCTIONAL DISSOCIATION: what a head broadcasts matters, where it reads does not

The audit's dCE column contained the real result, and
value_vs_pattern_ce replicated it at n=32 rows across four heads:
  a6.h3  zero +0.051 | pattern-swap -0.002 | value-swap +0.058
  a4.h7  zero -0.004 | pattern-swap -0.022 | value-swap +0.002
  a6.h5  zero +0.018 | pattern-swap +0.002 | value-swap +0.032
  a4.h1  zero +0.013 | pattern-swap +0.019 | value-swap +0.014
(a) HELD for a6.h3: giving it a sibling head's READ PATTERN while
keeping its own values is FREE (-0.002), while keeping its own
pattern with the sibling's VALUES costs as much as deleting it.
(b) HELD: the other named courier a4.h7 orders the same way. (c)
FAILED as a general law: a6.h5 follows the pattern but a4.h1 does
not -- so this is common, not universal. (d) FAILED: deleting
a4.h7 costs nothing at match positions (-0.004) even though 407
measured it re-aiming 5.5's reads by 18.6% -- another instance of
read-shifts not being function.
Emerging statement, with its own caveat queued: for these heads
the functional payload is WHICH value-subspace they broadcast,
not where they read from. pattern_necessity queued to decide
whether that is real or an artifact of sibling patterns being
similar: uniform-over-prefix, reversed, and cross-row patterns
(all definitely wrong), plus the h3-h0 pattern correlation as a
diagnostic. Registered: uniform and cross-row arms <= 0.02 nats
if positions truly do not matter.

## 424. Pattern necessity: the SHAPE matters, the ALIGNMENT does not

pattern_necessity replaced a6.h3's read pattern with patterns
that are definitely wrong (values always its own):
  zero      +0.0512      patswap (sibling)  -0.0021
  uniform   +0.0233      reversed           +0.0465
  cross-row +0.0168      sibling pattern corr with h3: 0.226
(c) diagnostic settles 423's caveat: the sibling patterns are only
weakly correlated (0.226), so the free pattern swap was NOT a weak
perturbation -- the dissociation is real. (b) HELD: a pattern
taken from a DIFFERENT ROW -- wrong context entirely -- costs
0.017, a third of deletion. (a) FAILED, and that failure is the
finding: a UNIFORM pattern costs 0.023 and a REVERSED one costs
0.047, nearly full deletion. So the read positions are not
irrelevant in general; what a6.h3 needs is a pattern with the
right SHAPE STATISTICS (concentrated, forward-decaying like a real
attention pattern) rather than one aimed at the right content.
Corrected statement, replacing 423's provisional wording: this
head's downstream function is carried by WHICH value-subspace it
broadcasts plus HOW CONCENTRATED its read is -- not by which
positions it reads. Reversal (same concentration, mirrored
targets) is the expensive perturbation; a real pattern from
unrelated text is cheap.

## 425. The swarm reviews the driver: two methodology catches, both shipped

Wave-3 reviewers came back with catches against MY instruments,
not just the records.
(1) r.18.2.0 (WEAKEN): the reviewer showed the ROBUST behavioral
gate I wrote is UNDERPOWERED BY CONSTRUCTION -- its seed leg
requires 5/5 draws on >=60% of seeds, but a true 84% effect
produces 5/5 only ~41% of the time. The punctuation claim it
demoted has whole-population support of 36/43 vs a 0.528 base
rate, p~0. Fixed: ROBUST_V2 gates on the population test over
EVERY member of the class with n>=10 (no draw noise), Bonferroni
option added (alpha = 0.10/n_tests) after a second reviewer showed
multi-direction scanning can still cross threshold. The claim is
reinstated as real-but-small in the record, with the WEAKEN
verdict preserved as the contemporaneous judgment.
(2) r.3.0.2 (WEAKEN): the a14 -> a15 enrichment (2.85x) REPRODUCES
on an unrelated leaf from a different family (2.44x), so for the
a15 leg it is a generic adjacent-layer property, not this
circuit's mechanism; the a17/a16 legs remain specific. Fixed:
leaf_input_decomp gained a --baseline mode that names peer leaves
sharing each component, and SOP v3 now REQUIRES a cross-leaf
specificity check before any writer is claimed as a circuit's
mechanism. Both records' gates and tables reproduced exactly
(6.1/6.2/6.02 and 5.96/5.92/5.99), so the verification layer is
doing its job in both directions: it certifies machinery and it
refuses interpretation.

## 426. TOOL CORRECTION: the mechanism tool needed a bootstrap; one swarm claim retracted

The r.2.0.2 reviewer ran the decisive check I had not: it varied
the ROW SUBSAMPLE. leaf_input_decomp drew one fixed 24-row sample
(seed 5), so its ratios carried no sampling error, and its own
null used the same draw. Resampled, r.2.0.2's headline a0
enrichment collapses: 1.464 -> the range 0.99-1.46 (mean 1.18).
Fixed the same hour: the tool now bootstraps over five row draws
and reports mean/min/max per writer, with ENRICHED_STABLE
requiring the MINIMUM across draws to clear 1.3 and beat the null.
Re-run under the fix:
  r.2.0.2  a0 -> a8 1.178 [0.991-1.464], a6 1.190 [1.000-1.475]
           ENRICHED_STABLE=False  -> CLAIM RETRACTED (record note
           appended; the leaf stands as gate-only)
  r.3.0.2  a14 -> a15 2.37 [2.06-2.85], a17 2.431 [2.09-2.96],
           a16 2.362 [2.05-2.84]  ENRICHED_STABLE=True -> claim
           survives, at the reduced scope its reviewer established
           (the a15 leg reproduces on an unrelated leaf, so that
           leg is an adjacent-layer property; a17/a16 stay
           family-specific)
Wave-3 verdicts complete: CONFIRM 1 (r.5.0.1, a reproduced
negative plus a new lead -- its downstream consumers are as
diffuse as its upstream writers), WEAKEN 3. Zero fabricated
claims survived contact with review, and every gate reproduced
exactly. The r.5.0.1 reviewer also raised the sharpest open
objection of the wave: members run high base CE, so concentration
may partly measure FRAGILITY rather than selectivity.
gate_specificity queued to settle it (rank-matched random
subspace ablation in the same components, registered a-c).

## 427. Head role map: position-sensitivity is the norm, and the induction band is the clearest case

head_role_map classified 63 heads across seven layers by whether
swapping their read pattern or their values costs more.
(a) FAILED, informatively: only 38% are payload-dominant, so the
a6.h3 result (423) is a minority regime, not the architecture's
rule -- most heads' function depends on WHERE they read. The
layer profile is non-monotone (22% payload at layers 1, 2 and 16;
44% at 4, 6 and 12; 67% at layer 8).
(b) HELD: all three sampled induction heads are position-
sensitive (1.4, 2.5, 8.4), exactly as their mechanism demands --
for a match head, the read target IS the function.
(c) FAILED: only 48 of 63 heads cost anything to delete (15 have
dCE <= 0), consistent with the program's older finding that a
large minority of heads are free or net-harmful on average text.
Costliest deletions in the sample: 1.1 (+0.089), 12.6 (+0.073),
6.3 (+0.047) -- note 1.1 and 12.6 were also the two costliest
under 4-read truncation (415), and 415 showed both compare m0|m0.

## 428. The stack's costliest heads are PREVIOUS-TOKEN readers, not identity matchers

costly_head_semantics located the top reads of the three heads
that cost the most to delete (1.1 +0.089, 12.6 +0.073, 6.3
+0.047), all three of which compare m0|m0 (415):
  1.1   offsets: -1 in 994 of 1008 reads, -2 in 14. Nothing else.
        A pure previous-token head. (a) HELD at the ceiling.
  6.3   -1 in 659, -2 in 161, tail to -6. Local, previous-token
        dominant -- consistent with its census 'prev' profile and
        with its courier role (407).
  12.6  -1 in 100, -2 in 76, -3 in 71, -4 in 53, decaying: a
        LOCAL WINDOW reader, not a point reader. Its 'previous
        token is the same as the query' rate is 31%.
(b) FAILED: 12.6's same-token read rate is 6% against a 1.3%
null -- enriched 4.6x but nowhere near the 20% bar. So the
model's most expensive heads are NOT identity matchers even
though their scores compare the identity code: they use m0's
code to decide WHICH RECENT TOKEN to read, not to find repeats.
That is the cleanest reconciliation yet of 415 (everything
compares m0|m0) with 427 (most heads are position-sensitive):
the identity code is the model's universal comparison currency,
and most heads spend it on local structure rather than on
long-range matching. The induction band is the specialist
minority that spends it on repeats.
Ops: gate_specificity crashed on leaves whose probe bundles
contain comp/head entries (census_lib.proj_hooks only handles
pca probes) -- fixed by filtering to all-pca bundles with the
skip count recorded, and requeued. head_cost_map (all 162 heads
under the corrected dCE metric) is queued behind it.

## 429. Complete head cost map (corrected metric): one head dominates the stack

head_cost_map deleted each of the 162 heads in turn under the
functional metric (dCE, per the 422 correction).
(a) HELD: 39 of 162 heads (24%) cost nothing or help when
deleted. (b) FAILED: the free set is NOT concentrated late --
median layer 8, and it is spread almost uniformly (5 free heads
at layer 0, 2-3 at most middle layers, 4 at layer 16, 3 at 17).
Being free is a property of individual heads, not of depth.
(c) HELD: the costliest ten are 5.7 (+0.916!), 0.3 (+0.112),
1.1 (+0.088), 12.6 (+0.073), 6.3 (+0.047), 3.8, 2.6, 1.4, 11.6,
10.5.
The striking number is head 5.7 at +0.916 nats -- an order of
magnitude above the next head and larger than most whole-LAYER
ablations in the program's depth map. 415 already flagged 5.7 as
the one code-resistant head whose score does NOT compare m0|m0
on both sides (it is m0|m4, the only such head found). The
single most important attention head in bilin18 is therefore
also the one that reads a different key-side variable from
everything else. Registered follow-up will target it directly.

## 430. GATE SPECIFICITY (partial, 5 of 12 leaves): the census gate is substantially a FRAGILITY detector

gate_specificity compares each leaf's own probe-bundle
concentration against a RANK-MATCHED RANDOM SUBSPACE ablation in
the same components. The run was killed twice by GPU pressure
from concurrent swarm agents (no traceback; now resumable with
per-leaf try/except and incremental saves, and requeued), but
five leaves completed and they already carry the message:
  r.12.0.1  own 4.01  random 2.65  ratio 1.51
  r.12.1.3  own 3.69  random 2.70  ratio 1.37
  r.8.1.2   own 3.93  random 2.38  ratio 1.65
  r.1.2.2   own 3.75  random 2.71  ratio 1.38
  r.2.0.0   own 8.28  random 2.56  ratio 3.23
A RANDOM subspace of matched rank, ablated in the same
components, already produces concentration 2.4-2.7 -- just under
the >=3 gate the whole census uses. Four of five leaves sit below
the registered 2x bar. PROVISIONAL CORRECTION, to be finalized
when the full run lands: leaf concentration is only about 1.4-1.6x
what an arbitrary same-rank ablation achieves, so the census gate
measures selectivity ON TOP OF a large fragility baseline, and
"concentration 5-6" should not be read as "these directions are
specifically responsible". The r.5.0.1 reviewer who proposed this
control was right to. Records already merged keep their numbers;
what changes is the interpretation, and the swarm SOP will carry
the ratio-to-random alongside raw concentration once the full run
is in.

## 431. Wave 4 and a meta-finding: input-writer composition is the wrong lever

Wave-4 authors (SOP v3 with bootstrap + Bonferroni):
  r.5.3.1  gate 5.17 (halves 5.16/5.17) | mechanism NEGATIVE
           (a2, a4 both ENRICHED_STABLE=False) | best behavioral
           candidate punct p=0.034 vs required 0.0083 -> correctly
           REJECTED by the agent itself
  r.13.2.1 gate 5.59 (5.63/5.56) | mechanism NEGATIVE (a7/a6/a3)
           | behavioral claim KEPT: helps at punctuation targets,
           39/49 = 80% vs 51% base, population p ~ 0,
           ROBUST_V2 with n_tests=12
  r.2.0.1  gate 5.40 (5.09/5.64) | mechanism NEGATIVE (a6/a8) |
           closest behavioral candidate digit-hurt p=0.0105 vs
           0.0083 -> REJECTED. Notably a0 NEVER APPEARED here
           (ratio ~0.96, range 0.75-1.09), independently
           confirming that its sibling r.2.0.2's retracted a0
           claim was subsample noise.
META-FINDING: the input-side mechanism tool has now returned
ENRICHED_STABLE=False on FIVE of seven leaves, and its one
survivor (r.3.0.2's a14) proved partly an adjacent-layer
property. Leaf selectivity is not explained by which writers feed
the machinery. This is a real negative about the method, not
about the model, and it redirects the swarm: leaf_output_decomp
queued -- for each leaf, ablate its bundle and measure which
DOWNSTREAM components' inputs change at member positions versus
off-slice, i.e. who CONSUMES the machinery. The 430 fragility
lesson is built into the tool from the start: every leaf is
scored against a rank-matched random-subspace ablation, and a
"consumer" only counts if the real bundle beats the random one.
Also queued: punct_generality -- two independent leaves now carry
the same punctuation claim, so the same random-subspace control
decides whether punctuation-specific pushing is a shared function
or a fragility artifact (registered a-c).
Ops: two wave-4 agents parked waiting on background jobs under
shared-GPU load; both were resumed by message and finished. The
runbook now carries a never-park rule for authors and a
resilience rule for queue scripts (try/except per item,
incremental saves, resume support) after gate_specificity was
killed twice.

## 432. THE MODEL'S COSTLIEST HEAD IS AN ATTENTION SINK

head_5_7_reads returned a suspicious histogram -- exactly 16
counts at every offset -8, -12, -16, -20... which is the
signature of a FIXED absolute key position sampled at every
fourth query, not a relative-offset preference. (Instrument note:
the read scan records relative offsets only; for sink-like heads
that is misleading and absolute positions must be checked. Done
here directly.) Verified:
  head 5.7 reads position 0 for 99.8% of queries
  neighbour head 5.6 reads position 0 for 5.3%
  position 0's value norm 730 vs 197 elsewhere
So the single most expensive head in bilin18 (+0.916 nats to
delete, eight times the next head, 429) is an ATTENTION SINK
locked onto the first position, where the model parks a
high-norm value.
head_5_7_role adds the functional shape: deletion costs 0.785 at
match positions, and BOTH swaps cost MORE than deletion
(sibling-pattern 0.98, sibling-values 1.11) -- for a sink, any
substitution injects a wrong constant, which is worse than
removing the constant altogether. Registered bars (a) and (b)
both HELD (the head is position-sensitive, trivially so: its
position is the mechanism).
This architecture has NO SOFTMAX, so the usual explanation for
sinks (absorbing normalisation pressure) cannot apply. What a
sink does here is add nearly the same vector at every position --
a learned bias. sink_bias_test queued: replace 5.7's write with
its own mean, with a mean taken from other rows, and with a
per-row mean. Registered: the mean arm costs <= 0.10 nats against
deletion's 0.92, and the cross-row mean <= 0.15. If those hold,
the model's most important attention head is 1152 numbers.

## 433. Wave 4 closes: four records, four honest negatives, one real claim

r.8.1.0 completed the wave: gate 5.68 (halves 5.73/5.62),
mechanism NEGATIVE on all three components -- and a textbook
demonstration of why the bootstrap was added: its a9 leg scored
ENRICHED=True on the single draw at 1.396 but min 1.216 across
draws, so ENRICHED_STABLE correctly rejected it. Best behavioral
candidate punct p=0.0455 against the corrected 0.0083 -- rejected
by the agent itself. Wave-4 totals: four gates all reproduced and
stable across corpus halves, four mechanism negatives, three
self-rejected behavioral claims, one KEPT claim (r.13.2.1's
punctuation effect, now under adversarial review with a
random-subspace attack). Zero unsupported claims entered the
record set.
Friction fixed from the wave: SOP timing estimates now say
"minutes under swarm load", agents are told to THREAD the dCE
vector through the task rather than recompute it (one agent burned
a full GPU pass), to capture the git rev at task START (it drifts
under concurrent commits), and the step-1 gate now carries the
430 calibration -- a random rank-matched subspace already scores
2.4-2.7, so a bare pass at 3 means little.

## 434. Third instrument catch: the enrichment gate had no power, and the negatives were overstated

The r.5.3.1 reviewer (verdict WEAKEN) did the analysis I should
have: it dumped the per-seed NULL values behind the enrichment
gate. The gate compared the bootstrap minimum against the MAXIMUM
of five null draws -- an extreme-value statistic. On that leaf the
null max is 1.333 against a 1.3 bar: 2.5% headroom, with the
null's own spread (1.01-1.33) as wide as the entire margin being
tested. So ENRICHED_STABLE could not distinguish a weak-to-
moderate true effect (ratio 1.1-1.25) from noise, and every
"no writer enriches" negative in the swarm was overstated.
Fixed and verified the same hour: ENRICHED_STABLE2 gates the
bootstrap minimum against the null's MEAN + 2 SD.
  r.5.3.1  a2 threshold 1.447 vs top-min 1.246 -> False
           a4 threshold 1.366 vs top-min 1.082 -> False
  r.3.0.2  a15 threshold 1.338 vs top-min 2.061 -> True
           a17 threshold 1.891 vs top-min 2.093 -> True
           a16 threshold 1.702 vs top-min 2.051 -> True
The real positive survives a properly calibrated threshold with
room to spare; the negatives stay negative but are now correctly
scoped. SOP v3 wording changed accordingly: a failing leaf
records "no STRONG single-writer mechanism (top ratio r,
threshold t)" and must quote what the test could not have
detected -- never a blanket absence. Record note appended to
r.5.3.1.
Tally of the swarm auditing its driver: three instrument flaws
found by reviewers in one night (the underpowered behavioral
seed-gate, the single-subsample enrichment ratio, and now the
extreme-value null), all fixed, all with the affected claims
re-scored. This is the verification layer earning its cost: the
records it produces are thinner than the first drafts, and the
ones that survive are worth more.

## 435. THE COSTLIEST HEAD IN THE MODEL IS 1152 NUMBERS

sink_bias_test, all three bars HELD, and the numbers are not
marginal:
  delete head 5.7                       +0.9154 nats
  replace its write with its OWN MEAN   -0.0053
  replace it with a mean from OTHER ROWS -0.0034
  replace it with a per-row mean        -0.0039
The model's single most expensive attention head -- eight times
costlier than the next, larger than most whole-layer ablations --
is EXACTLY a constant bias adder. Every position gets essentially
the same vector, the vector does not depend on the text (a mean
computed on different rows works just as well), and substituting
it is not merely cheap but very slightly BETTER than the intact
head. 429 measured its importance, 432 found the mechanism (a
sink locked on position 0, where the value norm is 730 vs 197),
and this closes it: the sink exists to add a learned constant,
and 1152 numbers replace it.
This also explains 431's crossover oddity, where both swaps cost
MORE than deletion: substituting a sibling's pattern or values
replaces the right constant with a wrong one, which is worse than
having no constant at all.
head_bias_sweep queued: how many of the 162 heads are pure bias
adders (deletion >= 0.02 but mean-replacement <= 0.005), what the
layer profile is, and how many nats the whole set recovers.

## 436. Three verdicts from the queue: consumers found, punctuation real, gate confirmed weak

leaf_output_decomp (the pivot after input composition failed on
5 of 7 leaves): BOTH bars HELD. Every leaf has a downstream
consumer whose input changes more at member positions than
off-slice, and every one beats its rank-matched random-subspace
control:
  r.3.0.2 in_a17 2.014 (random 1.471)   r.5.3.1 in_a17 1.603 (1.280)
  r.18.2.0 in_a17 1.357 (1.100)         r.2.0.1 in_a17 1.356 (1.235)
  r.5.0.1 in_a15 1.283 (1.205)          r.13.2.1 in_a17 1.188 (1.057)
The output side is the productive direction -- but the honest
caveat is that in_a17 is the top consumer for five of six leaves,
so "who consumes this" is answered mostly at the LAYER level (the
last attention layer reads what these bundles write); only the
margins over random differ per leaf.

punct_generality: the swarm's first fully verified behavioral
claim. r.13.2.1's punctuation effect is REAL and leaf-specific --
own bundle 39/49 hits, p ~ 0; rank-matched random subspace in the
same components 21/49, p = 0.73. The three control leaves show no
punct effect under their own probes (p = 0.016-0.040, none
clearing the corrected 0.0083). So punctuation-specific pushing
is a genuine function of that bundle, not the fragility artifact
430 warned about.

gate_specificity (complete, 12 leaves): BOTH bars FAILED, and 430's
provisional correction is now FINAL. A rank-matched random
subspace scores median concentration 2.62 -- just under the census
gate of 3 -- and only 4 of 12 leaves reach twice their random
baseline (ratios 1.26 to 3.23, median ~1.68). Census concentration
therefore measures selectivity sitting on a large fragility
baseline; a leaf at 5-6 is roughly 1.7x an arbitrary same-rank
ablation, not 5-6x "nothing". Every concentration number in the
program keeps its value and loses its old interpretation. The SOP
already carries this calibration for the swarm.

## 437. Bias-adding is not a role: head 5.7 is a special case of one

head_bias_sweep swept all 162 heads for deletion cost versus
mean-replacement cost. (a) FAILED decisively: exactly ONE head
(11.6) meets the registered bias-adder definition, and the sweep's
own bar excluded 5.7 by a hair (its mean arm scored -0.0074 at
NR=16 against a |0.005| threshold -- the criterion was too tight
for its own headline case, stated here rather than quietly
adjusted). (b) HELD: 5.7 owns the largest deletion-minus-mean gap
in the model at 0.923, an order of magnitude above anything else.
The informative statistic is the distribution: of the twelve heads
costing >= 0.02 nats to delete, the median share of that cost
explained by a constant is 39%:
  5.7  zero 0.916  mean -0.007  constant explains 101%
  0.3  zero 0.112  mean  0.104  7%
  1.1  zero 0.089  mean  0.055  37%
  12.6 zero 0.073  mean  0.057  21%
  6.3  zero 0.047  mean  0.028  41%
  1.4  zero 0.029  mean  0.011  63%
So most heads carry a partial constant component and a real
contextual remainder; 5.7 is the singular case where the constant
IS the head. Bias-adding is one enormous special case sitting on
an otherwise contextual stack, not a widespread role.
sink_source queued: decompose position 0's residual at layer 5
into exact writer contributions and check the chain end to end
(registered: one writer carries >= 0.40, it is wte or m0, and the
routed position-0 value matches the head's mean write at cosine
>= 0.9).

## 438. Punctuation claim CONFIRMED under attack; held-out test queued

The r.13.2.1 reviewer returned CONFIRM after three attacks, and
the confound analysis is worth recording in full because it cuts
the other way: punctuation members do have much lower base CE
(4.08 vs 6.02), which is exactly the profile of a regression
artifact -- but conditioning on base CE REFUTES that explanation.
Members in the punct-matched low-CE band show only 40% help, and
help-rate RISES with base CE across quartiles (37.5% -> 66.7%),
the opposite of what an artifact requires. The random rank-matched
subspace scored 43-63% across five seeds against the real
bundle's 80%, never reaching the bar.
Its standing objection is the right one: the test lives on the
corpus the leaf was discovered in, and the alpha corrects only
within-leaf sub-tests, not the search across leaves and classes.
punct_heldout queued to settle it on text the census never saw --
fresh FineWeb rows, the claim in generalized form (does ablating
this bundle lower CE at punctuation targets relative to
non-punctuation targets?), permutation-tested, with random
rank-matched subspaces as the control.
Report updated at this phase boundary and republished: a new
section on head 5.7 (the costliest head is a constant), and the
census paragraph now carries the fragility calibration -- a
rank-matched random subspace scores median 2.62 against the gate
of 3, so a leaf at 5 is about 1.7x an arbitrary ablation.

## 439. The bias vector is built by mlp4, and the chain closes at cosine 0.999

sink_source decomposed position 0's residual at layer 5 into
exact writer contributions. (b) FAILED as registered -- I
predicted wte or m0 on the reasoning that position 0 has no
context, but the answer is **mlp4** at projection share 0.626,
then m3 0.159, m0 0.119. (a) HELD (one writer well over 0.40).
(c) HELD emphatically: routing position 0's value through head
5.7's own projection reproduces the head's actual mean write at
**cosine 0.999** -- the chain is closed arithmetically, not
approximately.
Standing mechanism, four named parts:
  something at position 0 -> mlp4 writes a large vector there ->
  head 5.7 reads position 0 for 99.8% of queries -> that vector
  is added to every position as a constant (deleting it costs
  0.92 nats; replacing it with any equal constant is free).
sink_origin queued to characterise the source: is mlp4's
position-0 write a fixed learned vector (stable direction across
rows) or a function of the first token? Registered: (a) its norm
at position 0 is >= 3x elsewhere, (b) mean pairwise cosine across
rows >= 0.9, (c) split by first-token identity, (d) which writer
dominates mlp4's own input there.

## 440. The punctuation claim GENERALIZES to unseen text

punct_heldout took r.13.2.1's claim to 64 fresh FineWeb rows the
census never saw, in its generalized form (does ablating this
bundle lower CE at punctuation targets relative to others?).
  own bundle: punct -0.0078, non-punct +0.0158,
              difference -0.0236, permutation p = 0.000
  random rank-matched subspaces: +0.0009 (p 0.30),
              +0.0018 (p 0.16), -0.0064 (p 0.0025)
(a) HELD: the effect generalizes, and it is a genuine dissociation
-- ablating the bundle HELPS at punctuation while HURTING
elsewhere in the same forward passes.
(b) FAILED honestly: one of three random subspaces also produced
a significant same-direction difference, though at 3.7x smaller
magnitude. So a small part of the punctuation effect is generic
to ablating anything in those components; the bundle-specific
part is the large remainder.
(c) FAILED: the random subspaces' overall damage (0.002-0.005
nats) is well under the 0.01 threshold I registered as
"non-trivial", so they are not magnitude-matched to the real
bundle (0.0126) -- the control is conservative in specificity but
weak in magnitude, and a magnitude-matched control is the right
next version.
Verdict as recorded: r.13.2.1's punctuation function is real,
survives adversarial review, and generalizes off-corpus, with a
measured generic component that a future control should isolate
properly. This is the swarm's first behavioral claim to clear
every bar the program has.

## 441. THE BIAS CIRCUIT, COMPLETE: a fixed vector manufactured at position 0

sink_origin: ALL FOUR BARS HELD, with margins that leave no
interpretive room.
  mlp4's write at position 0: norm 155,009 vs 14,803 elsewhere
                              (ratio 10.5)
  direction across 32 documents: mean pairwise cosine 0.9978
  between documents with DIFFERENT first tokens: 0.9978
  (same first token: 1.000)
  mlp4's own input at position 0: m0 0.441, m3 0.298, m2 0.175,
                                  m1 0.055, a4 0.018
So the vector is not a function of the first token -- texts
beginning with a newline, with " in", with " and" all produce the
same direction to three decimal places. The MLP chain manufactures
it, and attention 4 contributes almost nothing (0.018), which is
what "position 0 has no context" should look like.
THE COMPLETE CIRCUIT, every part named, no approximation between
them:
  m0 -> m2 -> m3 -> mlp4 build a fixed vector at position 0
  (10x normal norm, direction constant across texts)
    -> head 5.7 reads position 0 for 99.8% of queries
      -> that vector is added at every position as a constant
        -> deleting it costs 0.92 nats, the largest single-head
           cost in the model
        -> replacing it with any equal constant is free (-0.005)
This is the program's most completely specified novel circuit:
not a literature motif, but an architecture-specific
bias-generation mechanism, closed from manufacture to broadcast
to price. Report updated and republished with the full chain.
Open question queued (bias_semantics): what is the bias FOR?
Read the constant through the unembedding -- which tokens it
pushes, against a norm-matched random vector, plus a causal check
that deleting the head lowers exactly those tokens' logits.

## 442. What the bias pushes -- and a sign paradox worth its own experiment

bias_semantics read the constant through the unembedding.
(b) HELD (reported for naming): the bias pushes
  -  and  ,  (  or  all  in  so  at  on  just  to  ...  for
  back  post  set  over  /  only
and suppresses rare fragments and oddities (' Berserker',
' Replay', 'ngth', 'perature', ' CARD'). That IS a nameable
class: high-frequency function words, connectives and
punctuation on the push side; low-frequency subword debris on the
suppress side. In one line, the constant is a UNIGRAM-FREQUENCY
PRIOR.
(a) FAILED, informatively: the profile is DIFFUSE, less
concentrated than a norm-matched random vector (top-20 share
0.0013 vs 0.0019 null). A frequency prior should look exactly
like that -- it nudges thousands of tokens a little, rather than
a few tokens a lot.
(c) HELD but with the sign INVERTED against expectation, which
is the interesting part: deleting head 5.7 RAISES the logits of
precisely the tokens the bias appears to push (+0.211 vs +0.074
for a random control, 2.8x). So the direct unembedding read and
the head's total causal effect DISAGREE IN SIGN. Either the
readout is measuring a path that does not dominate, or the bias's
downstream consumers invert it.
This is a caution about a very common practice (project a vector
through the unembedding and name it), so it gets its own
experiment rather than a hedge in prose. bias_path_split queued:
inject the constant only at the final residual (direct path,
bypassing layers 6-17) versus back into layer 6 (the real path),
and price both. Registered: (a) the direct path recovers under
30% of the head's 0.92 nats, (b) the indirect path recovers 80%+,
(c) under direct-only injection the pushed tokens move UP,
confirming the readout measures the direct path only.

## 443. INSTRUMENT BUG: the injected constant was 16384x too large; path-split run VOID

bias_path_split returned direct +11.54 and indirect +11.54 nats
against deletion's 0.92 -- both arms catastrophic, and bars (a)
and (c) "held" only because the arms exploded. Recorded as VOID
per the standing rule that an arm which cannot fail (or which
fails by blowing up) is not evidence.
Diagnosis, measured rather than guessed: the residual entering
attention is rms-normed to norm ~34, the raw final residual is
~85,091, and the constant I injected had norm 111,064,552 -- more
than a thousand times the whole residual stream. Cause: my
single-head score helper omitted the model's /128 normalisation on
EACH of the two QK factors, so every reconstructed pattern (and
hence the constant derived from it) was 128 x 128 = 16,384x too
large.
Scope of the error, stated precisely: 442's findings are NOT
affected -- the token lists and the concentration comparison pass
the vector through rms_norm first (scale-invariant), and the
causal sign test used real model ablations, never the
reconstructed constant. Everything that used the constant's
MAGNITUDE is void: the path-split arms, and nothing else. The
earlier sink results used the model's own mean of z (not a
reconstruction), so 435/439/441 are untouched.
Fixed in both scripts with the divisions restored and a comment
naming the bug; bias_path_split requeued. Standing lesson for the
program's scripts: whenever a reconstructed quantity is INJECTED
rather than compared, check its norm against the residual it is
being added to before trusting the run -- a scale error is
invisible in cosine-based checks and fatal in causal ones.

## 444. The sign paradox resolved: the bias is consumed immediately, and inverted downstream

bias_path_split, rerun with the scale bug fixed:
  delete the head                       +0.9154
  inject the constant at block 6        +1.0166
  inject it at the final residual       +1.3746
  (reference, 435: replace the head's write with the same
   constant IN PLACE inside layer 5)    -0.0053
(a) HELD: the direct path recovers nothing -- it is worse than
deletion. (b) FAILED, and the failure is the finding: my
"indirect" injection point was one sublayer too late. The head
writes immediately after attention 5, so mlp5 normally sees the
bias in its input; injecting at block 6 skips mlp5, and doing so
is WORSE than never adding the bias at all. (c) HELD, and it
resolves 442's paradox quantitatively:
  mean logit of the bias's top-20 pushed tokens
    intact model            4.4016
    head deleted            4.6191   (deleting RAISES them)
    constant injected direct 6.4746  (direct path PUSHES them hard)
So the direct path does exactly what the unembedding read says --
injecting the constant straight into the logits lifts those tokens
by +2.07 -- while the head's TOTAL effect is to lower them by
0.22. The downstream consumers invert the direct effect roughly
tenfold. A vector's unembedding projection described its direct
path correctly and its actual function backwards; that is the
caution 442 flagged, now measured.
bias_injection_depth queued to pin the consumption point: inject
the same constant in place, after mlp5, at blocks 6/9/13, and at
the final residual. Registered: cost rises monotonically with
depth, and the after-mlp5 arm already costs >= 0.5 nats -- i.e.
most of the bias's value is consumed by layer 5's own MLP.

## 445. Injection depth is a CLIFF, not a decay

bias_injection_depth priced the constant delivered at six points:
  in place (inside layer 5, before mlp5)  -0.0053
  after mlp5                              +1.0166
  block 6 input                           +1.0166
  block 9 input                           +1.3206
  block 13 input                          +1.1796
  final residual                          +1.3746
  (reference: delete the head)            +0.9154
(a) FAILED: the cost does not decay monotonically -- block 13 is
cheaper than block 9, and the differences past block 6 are noise
around a plateau. (b) and (c) HELD. Two design notes recorded:
"after mlp5" and "block 6 input" are the SAME point in this
architecture (adding to mlp5's output is adding to block 6's
input), which the identical numbers confirm -- a redundant arm,
not a coincidence; and my "monotone decay" framing was simply the
wrong model of the phenomenon.
The real shape is a CLIFF: before mlp5 the constant is free,
anywhere after it the constant is worse than nothing, and depth
past that point barely matters. Two readings remain, and they
have different consequences, so bias_consumer is queued to
separate them: either mlp5 is the consumer that must see the
bias, or adding any constant late is generically harmful. Its
arms ask the question directly by SUBTRACTION with the head left
intact (mlp5 sees the bias, nobody later does) plus norm-matched
random-vector controls in both directions. Registered: (a)
after_blind <= 0.20 nats if mlp5 is the consumer, (b) the junk
control quantifies how much of the late-injection penalty is
generic damage, (c) sanity arms.

## 446. MLP5 IS THE CONSUMER (74%) -- plus a fourth self-caught instrument bug

First the bug, because the first run's numbers were meaningless:
bias_consumer initially returned after_blind = 0.0000 and
junk_sub = 0.0000, EXACTLY zero -- the signature of a hook that
never fired. Cause: I nested the injection hook inside the
head-ablation branch, so the two arms that leave the head intact
registered nothing at all. Bar (a) "held" vacuously. Fixed by
de-nesting, and a permanent guard added: the script now records
which arms produced exactly-zero effect and prints
"ARMS THAT NEVER FIRED ... run is VOID for those arms". Every
future multi-arm script in this program should carry that check;
this is the fourth instrument fault of the night (underpowered
seed gate, single-subsample ratio, extreme-value null, dead arm),
and three of the four were only visible because something forced
the arms to be compared against each other.
Corrected results, all arms live:
  delete the head                                  +0.9154
  head zeroed, constant re-added after mlp5        +1.0166
  head INTACT, constant removed after mlp5         +0.2398
  norm-matched random vector added after mlp5      +1.3344
  norm-matched random vector removed after mlp5    +3.2122
(a) FAILED by 0.04 against a 0.20 bar, but the substantive claim
holds with room: once mlp5 has seen the bias, stripping it from
the entire rest of the stack costs 0.24 against 0.92 for full
deletion -- **mlp5 delivers about 74% of the bias's value**.
(b) HELD: late injection of junk costs 1.33, so much of the
1.02 penalty for delivering the real constant late is generic
damage rather than missing consumption.
(c) FAILED, and the pairing was my error: junk_add starts from a
bias-less stream while junk_sub starts from the intact one, so
they are not matched. The comparison that IS matched is
after_blind versus junk_sub -- both subtract a same-norm vector
from the intact stream at the same point -- and it is decisive:
removing the true bias costs 0.24, removing a random direction of
identical magnitude costs 3.21, THIRTEEN TIMES more. The
downstream stack tolerates losing this particular vector and
nothing else of its size.
Next, from theory rather than from more ablations. In a BILINEAR
layer a constant input offset is not just an offset:
  Down[(L(x+c))*(R(x+c))] = Down[Lx*Rx] + Down[Lx*Rc]
                          + Down[Lc*Rx] + Down[Lc*Rc]
and the two cross terms are LINEAR in x. So the bias may exist to
give the model a linear pathway through an otherwise purely
quadratic layer. bias_linearizes queued: decompose mlp5's output
into those four exact terms and price each, with an exactness
check that they reconstruct the real output to <1e-3.

## 447. LINEARIZATION REFUTED -- and the bias's value is strongly non-additive

First, v1 of this experiment was VOID and is recorded as such: it
subtracted the RAW constant from mlp5's RMS-NORMALISED input
(norm ~6800 against ~34) and omitted the layer's Down_bias, so
every arm exploded to 14-16 nats and the exactness check failed at
1.6e-2. That is the same scale-mismatch class as 443, two cycles
after I wrote the lesson down -- so the lesson is now a REGISTERED
GATE in the script itself: exactness is prediction (a), and if it
fails no arm is interpreted.
v2, with the constant's share of the normalised input computed
per position from the true raw residual and Down_bias included,
reconstructs mlp5's real output to a relative error of
**1.253e-06** -- the four-term bilinear decomposition is exact, as
the algebra says it must be:
  Down[(L(x+c))*(R(x+c))] + b = Down[Lx*Rx] + Down[Lx*Rc]
                              + Down[Lc*Rx] + Down[Lc*Rc] + b
And with that exactness established, the hypothesis dies:
  remove both cross terms (the LINEAR pathway)   +0.0203
  remove the pure constant term                  +0.0045
  remove both                                    +0.0154
  (full bias deletion, for scale)                +0.9154
(b) FAILED by a factor of 23. The bias does NOT earn its 0.92
nats by opening a linear pathway through mlp5; mlp5's entire
processing of the constant is worth two hundredths of a nat.
Combined with 446 this leaves a sharp accounting problem, stated
plainly rather than smoothed: mlp5's processing 0.02, downstream
stream presence 0.24, everything 0.92. The parts sum to about a
quarter of the whole. The bias's value is overwhelmingly an
INTERACTION -- it is worth almost nothing wherever you look for it
locally, and 0.92 nats when it is absent everywhere.
A hypothesis that fits a network with no activation functions:
every downstream rms_norm divides by the residual norm, so a
large constant added at layer 5 sets the operating SCALE of the
whole rest of the stack -- a global effect no local ablation
would find. bias_norm_vs_direction queued to decide it: replace
the head's write in place with a random direction of the same
norm, and with the true constant scaled by 0.5x and 2x.
Registered as a fork -- (a) random same-norm <= 0.20 means the
bias is a scale device, (b) >= 0.5 means direction is the point.

## 448. NOT a scale device: the bias is a specific direction the stack is calibrated to

bias_norm_vs_direction, all arms in place inside the head where a
constant is otherwise free:
  the true constant                 -0.0053
  0.5x the true constant            +0.0457
  2.0x the true constant            +0.1393
  a RANDOM direction, same norm     +6.3037
  delete the head entirely          +0.9154
(a) FAILED: the scale hypothesis is dead. (b) HELD, and the
number is the story: substituting a random direction of identical
magnitude costs SEVEN TIMES more than having no constant at all.
(c) FAILED: halving costs 0.046 and doubling 0.139, both under
the 0.10 bar for half -- magnitude is only loosely tuned.
So the constant is a specific learned direction, tolerated across
a factor of four in magnitude, and catastrophic if rotated. Read
with 447's accounting problem (local ablations find a quarter of
its value), the picture is that every later layer is calibrated
around this particular offset: absent, the stack degrades
gracefully by 0.92 nats; replaced by something else of the same
size, it is thrown off its operating point entirely.
bias_stream_geometry queued to test that directly and without
ablation: measure the cosine between the constant and the
residual stream's own mean direction and top principal direction
at every layer 5-17, plus the share of residual norm it accounts
for, against a random-vector null. Registered: (a) |cos| >= 0.5
at a majority of layers 6-17, (b) at least 10x the null, (c)
norm shares reported per layer.

## 449. THE CONSTANT IS THE STREAM: 62-72% of the residual is one fixed vector

bias_stream_geometry, no ablation involved -- pure geometry
against the residual at every layer:
  layer  cos(mean resid dir)  cos(PC1)  ||resid||  const/||resid||
   5          +0.869            0.213     32,414        0.209
   6          +0.993            0.744     11,018        0.615
   7          +0.990            0.927      9,468        0.716
   8          +0.985            0.857     10,001        0.678
   9          +0.981            0.772     15,478        0.438
  10-15    +0.973 -> +0.878   0.73->0.62  17k->34k   0.39->0.20
  16          +0.743            0.518     39,724        0.171
  17          +0.268            0.075     50,819        0.133
  random-vector null: 0.012-0.037
ALL THREE BARS HELD: aligned at 92% of layers 6-17, mean |cos|
0.880 against a null of 0.0165 -- a fifty-three-fold separation.
The sink constant IS the residual stream's mean direction through
the middle of the network, and at layers 6-8 it accounts for
62-72% of the residual's entire magnitude. After layer 5, the
stream is mostly this one vector; everything the text says rides
on top as a smaller perturbation. The alignment decays through
the last two layers (0.74, then 0.27) as the network turns toward
producing logits.
This explains every earlier measurement at once. Rotating the
constant costs 6.3 nats (448) because it moves the centre the
whole stack is calibrated around. Halving or doubling it is
nearly free because rms_norm removes overall scale. And no local
ablation finds its value (447: parts sum to a quarter of the
whole) because being the baseline is not a local property.
Report updated and republished with the completed story.
IMMEDIATE CONSEQUENCE FOR THIS PROGRAM'S OWN TOOLING, queued
rather than assumed: leaf_input_decomp measures each writer's
PROJECTION SHARE onto the total residual -- which after layer 5
is 60-70% the bias. Writer shares at mid and late layers may
therefore be partly measuring alignment with the bias.
mech_tool_recenter recomputes the tables with the bias axis
projected out for the one CONFIRMED positive (r.3.0.2's a14) and
two confirmed negatives. Registered: (a) the top writer changes
for at least one leaf, (b) r.3.0.2's a14 survives at >= 1.5 --
and if it does not, the program's only confirmed mechanism claim
was a bias-alignment artifact and gets retracted.

## 450. The instrument survives the stream finding -- and the reason is worth stating

mech_tool_recenter recomputed the mechanism tables with the bias
axis projected out. ALL THREE BARS HELD:
  r.3.0.2  a17: a14 3.075 -> 3.019   a16: a14 2.963 -> 2.938
  r.13.2.1 a7/a6/a3: a0 1.25 -> 1.18 (still no enrichment)
  r.2.0.1  a8: top writer m6 (1.038) -> a4 (1.071)
(b) HELD: the program's one CONFIRMED mechanism claim survives
recentering essentially unchanged (3.02 against 3.08) -- a14's
enrichment into r.3.0.2's late-attention machinery is not a
bias-alignment artifact, and does not get retracted.
(a) HELD technically -- the top writer changed for one leaf --
but the honest reading is the opposite of alarming: the change
happened only where NOTHING was enriched (r.2.0.1's ratios are
1.03-1.07, a reshuffle among near-ties).
The reason the instrument is robust is worth recording, because
it was not obvious before the run: enrichment RATIOS compare
member positions against off-slice positions, and the bias is
position-independent, so it inflates both sides equally and
CANCELS. Ratio statistics are immune to a constant offset;
ABSOLUTE share statistics are not. So the program's ratio-based
mechanism claims stand as measured, while any absolute
writer-share number quoted at layers 6+ (for instance "m0
dominates at 0.42") is inflated by the stream centre and should
be read as alignment with it, not as contribution. That
distinction is now in the swarm's instructions.
Wave 5 launched on three fresh leaves with the hardened SOP, and
sink_census queued: sweep all 162 heads for position-0 locking to
ask whether 5.7 is alone or the extreme member of a class
(registered: >=5 heads over 50%, sinks costlier to delete, 5.7
the most extreme).

## 451. Sink census: only TWO in 162 heads, both in layer 5

sink_census swept every head for position-0 locking.
  5.7   99.7% of top reads at position 0, deletion cost 0.916
  5.2   67.6%,                            deletion cost 0.018
  5.4    8.5%   5.6 6.1%   5.1 6.0%   6.1 5.8%   rest under 2%
(a) FAILED against a bar of five: there are exactly TWO sinks in
the model, and both live in layer 5. (b) HELD: median deletion
cost 0.916 for sinks against 0.003 for non-sinks. (c) HELD: 5.7
is the extreme.
Two readings the data supports and one it kills. Sinks are not a
CLASS in this architecture -- they are a pair, co-located in one
layer, which fits a model with no softmax where nothing forces
attention mass to go somewhere. And being a sink does not make a
head important: 5.2 reads the same position two-thirds of the
time and costs fifty times less to delete than 5.7. sink_pair
queued to ask what separates them: whether 5.2 also broadcasts a
near-constant, whether its constant lies on the same axis as
5.7's, and whether deleting both is superadditive.

## 452. Wave 5: three records, three mechanism negatives, two behavioral claims

  r.11.1.2  gate 5.82 (6.02/5.65) | mechanism NEGATIVE on a8/a3/a4
            (top ratios 1.11-1.15, thresholds 1.30, headroom
            -0.22 to -0.25) | KEPT: punctuation, 36/51 = 71% vs
            47% base, p=0.0007, ROBUST_V2 at n_tests=12, with a
            direction check (punct dCE -0.209, non-punct +0.177)
  r.23.2.3  gate 5.51 (5.55/5.46) | mechanism NEGATIVE on a8
            (1.077 vs threshold 1.30) | punctuation TESTED AND
            FAILED here (p=0.107 help, 0.947 hurt) -- the first
            leaf to reject the sibling hypothesis, which is what
            makes the other three worth believing
  r.7.1.1   gate 5.08 (4.95/5.21) | mechanism NEGATIVE on a7
            (1.078 vs threshold 1.433) | KEPT, and a NEW class:
            capitalized-initial targets, 90/137 vs 72.3 expected,
            p=0.0015, ROBUST_V2 at n_tests=12
So the swarm's behavioral yield is now three punctuation claims,
one capitalized claim, and one explicit rejection -- and every
mechanism table has come back negative except r.3.0.2's. The
agents are also auditing the tooling unprompted: one flagged that
ROBUST_V2 can pass with seed_pass_frac 0.0 (correct by design
after 434, but counterintuitive), and another raised the
Bonferroni counting question, now answered in the SOP -- n_tests
is every (class, direction) pair you actually evaluate, and if
you eyeballed the data first you must count all the pairs you
could have chosen from.
punct_shared is queued to decide whether the three punctuation
claims are one effect or three (joint versus individual ablation,
plus random-subspace controls on all three).

## 453. The three punctuation claims are ONE effect

punct_shared ablated each bundle alone, all three jointly, and
rank-matched random subspaces per leaf.
  leaf       own (hits/n, p)        random subspace     excess
  r.18.2.0   36/43, p=0.000         25/43, p=0.266       0.309
  r.13.2.1   39/49, p=0.000         30/49, p=0.041       0.286
  r.11.1.2   36/51, p=0.0007        26/51, p=0.480       0.233
  joint ablation of all three, scored on each leaf:
             33/43 p=0.0008 | 38/49 p=0.0001 | 34/51 p=0.004
             excesses 0.246 / 0.273 / 0.195
(a) HELD, and decisively in the informative direction: the joint
ablation produces LESS excess than any single bundle alone. Three
independent effects would add; these SATURATE, and slightly
interfere. The three leaves are three views of one shared
punctuation effect.
(c) HELD at the corrected threshold, with one number worth
stating rather than hiding: r.13.2.1's random-subspace control
reached p=0.041 -- clean against the Bonferroni bar of 0.0083 but
not against a naive 0.05. The generic component 440 measured is
visible here too, and it is small.
punct_carrier queued to find what carries the shared effect. The
three bundles overlap in components (a7 in two, a3 in two, a8 in
two), so single whole-component mean-ablations are scored on all
three leaves' punctuation populations, with two components in no
bundle as controls. Registered: (a) some single component
reproduces excess >= 0.15 on at least two leaves, (b) it is one
of the shared components, (c) the controls stay clean.

## 454. The two sinks share an axis and back each other up

sink_pair compared the model's only two position-0 heads.
  delete 5.7            +0.9154
  delete 5.2            +0.0154
  delete both           +1.2136   (sum of individuals: 0.9308)
  5.7's write vs its own mean: cosine 0.998
  5.2's write vs its own mean: cosine 0.883
  5.7's constant vs 5.2's constant: cosine +0.853
(a) FAILED narrowly -- 5.2 is near-constant at 0.883 against a
0.9 bar, so it is a broadcaster with more position-dependence
than 5.7, not a pure one. (b) HELD: the two constants sit on
essentially the same axis (+0.853). (c) HELD: deleting both costs
0.283 nats MORE than the sum of deleting each.
Read together: layer 5 maintains the stream centre with two
heads on one axis, a dominant one and a partial backup, and they
are superadditive -- removing 5.7 alone leaves 5.2 holding part
of the centre, which is why 5.2 looks nearly free on its own
(0.015) and expensive once its partner is gone. That is a
redundancy structure, and it is the first clean instance of one
in this program's causal graph.

## 455. SCOPE CORRECTION: the punctuation effect is damage-general, not leaf-specific

punct_carrier mean-ablated whole components and scored the same
punctuation populations:
  component  r.18.2.0        r.13.2.1        r.11.1.2
  a7         exc 0.263 ***   exc 0.246 ***   exc 0.100
  a3         (carrier)       (carrier)       --
  a6         exc 0.022       exc 0.157       exc 0.177
  a8         exc 0.116       exc 0.203 **    exc 0.112
  a4         exc 0.175       exc 0.141       exc 0.149
  a9         exc 0.060       exc 0.143       exc 0.008
  a12 CTRL   exc -0.135      exc -0.022      exc -0.267  (clean)
  m7  CTRL   exc 0.254 ***   exc 0.208 **    exc -0.003  (NOT
                                                         clean)
(a) and (b) HELD -- carriers exist and include shared components
-- but (c) FAILED, and that failure outranks them: m7 belongs to
NONE of the three bundles and reproduces the effect at p=0.0005.
CORRECTION, stated plainly and propagated to all three circuit
records: the punctuation effect is NOT specific to these leaves'
machinery. Several different whole-component ablations produce
it; only a12 is clean. What remains true and measured is (i) each
leaf's own bundle produces it, (ii) 16-dimensional random
subspaces inside the same components do NOT (453), and (iii) it
generalizes to fresh FineWeb text (440). So it is not "any
damage" -- small random damage does nothing -- but it is not this
circuit's private function either.
One explanation covers every one of those facts and ties the arc
back to 442, where the sink constant read out as a UNIGRAM-
FREQUENCY PRIOR: sufficiently large damage makes the model fall
back toward that prior, which helps wherever the true next token
is high-frequency -- and punctuation is the most frequent class
in the corpus. frequency_fallback queued to test it directly:
(a) does help-rate rise monotonically across unigram-frequency
quartiles, (b) does punctuation retain excess AFTER conditioning
on frequency, (c) does ablation move the model's predictions
measurably toward the unigram distribution (KL).
If (a) and (c) hold and (b) shows no residual excess, the
program's most-tested behavioral claim resolves into a general
property of damaged prediction rather than a circuit function --
and the swarm's behavioral bar needs a frequency control added
before any future class claim is kept.

## 456. Frequency does NOT explain the punctuation effect either

frequency_fallback (after a vocab-padding crash on the first
attempt -- lm_head is 50304 wide, the tokenizer 50257; fixed and
rerun, the pre-crash frequency analysis was unaffected):
  help-rate by unigram-frequency quartile of the target
    Q1 0.502 (n=245)  Q2 0.553 (190)  Q3 0.479 (215)  Q4 0.593 (214)
  within the TOP frequency quartile
    punctuation targets   0.778 help (n=45)
    everything else       0.544 help (n=169)
  KL(prediction || unigram):  intact 4.957 -> ablated 4.751
(a) FAILED: the ladder is flat-to-noisy, not monotone -- it dips
at Q3 and rises only from 0.50 to 0.59 overall.
(b) FAILED decisively: conditioning on frequency leaves
punctuation with a 23-point excess (0.778 against 0.544). The
frequency-fallback account, which fit the earlier facts so
neatly, is REFUTED as an explanation of this effect.
(c) HELD but small: damage does move predictions toward the
unigram prior, by about 4% of the KL. Real, and nowhere near
enough to carry the punctuation result.
So after 440, 453, 455 and now 456 the punctuation effect is:
reproducible, off-corpus generalizing, robust to random-subspace
controls, NOT specific to any one leaf's machinery, NOT explained
by target frequency, and only marginally related to prior
fallback. It is a property of this model that many ablations
expose, and its cause is still open.
The next hypothesis is about the MODEL rather than the metric: at
punctuation targets the intact network may be systematically
over-confident in a wrong continuation -- it keeps the phrase
going -- and damaging almost any machinery lets the punctuation
win. That predicts an identifiable competitor token.
punct_competitor queued: at punctuation positions where ablation
helps, record the intact top-1 and how its probability moves,
against non-punctuation helped positions as the control.
Registered: (a) the intact top-1 is a non-punctuation
continuation in >= 60% of cases, (b) ablation suppresses that
competitor at least 3x more than a random token, (c) no reverse
asymmetry in the control class.

## 457. THE PUNCTUATION EFFECT IS A MODEL DEFICIENCY: over-continuation at phrase boundaries

punct_competitor ran twice. v1's arms all pointed the right way
but on n=5 helped punctuation positions -- too few to record, so
it was rerun with all member rows pooled across the three
punctuation leaves and a POWER GATE as prediction (a): interpret
nothing unless at least 40 sites are scored. v2, n=100:
  (a) POWER GATE HELD: 100 helped punctuation positions
  (b) HELD: at 75% of them the intact model's top-1 is a
      NON-punctuation token -- the model wants to keep the phrase
      going where the text actually ends it
  (c) HELD, and this is the number that matters: ablation lowers
      THAT competitor's probability by -0.108 while a random
      token moves by +1.24e-06 -- five orders of magnitude of
      selectivity, not diffuse damage
  (d) CONTROL HELD: at the 1194 helped NON-punctuation positions
      the intact top-1 is punctuation only 12.6% of the time, so
      the asymmetry runs one way
This resolves the arc that 440-456 kept failing to explain. The
punctuation effect is not a circuit function, not a frequency
artifact, and not generic damage. It is a systematic BIAS IN THE
TRAINED MODEL -- over-confidence in continuing a phrase at
positions where punctuation is correct -- which many different
ablations partially relieve. Ablating machinery "helps" there in
the same sense that removing a thumb from a scale helps.
That reframes the earlier corrections rather than erasing them:
455 was right that the effect is not leaf-specific, 456 was right
that frequency does not explain it, and both were looking for a
circuit where the answer was a model-level bias.
punct_overconf_source queued to ask which machinery creates the
over-confidence: at these sites, do the five helping components
(a3, a6, a7, a8, m7) push the competitor token in logit space,
and do they share a direction, with the clean component a12 as
control? Registered a-c.

## 458. The over-confidence has a location: five components push the wrong continuation

punct_overconf_source, at the 100 helped punctuation sites, took
each component's actual write and read its logit contribution to
the competitor the model wrongly prefers versus the true
punctuation target:
  component  ->competitor   ->true target   pushes competitor?
  a3            7.219          4.674            yes (+2.55)
  a8           14.297          7.014            yes (+7.28)
  m7           15.518         11.943            yes (+3.58)
  a7           14.332         13.087            yes (+1.25)
  a6           11.816         11.427            yes (+0.39)
  a12 CTRL      9.547          9.996            NO  (-0.45)
(a) HELD at 5 of 5, and the control fails to push -- exactly the
split that 455's ablation scan predicted from the other side.
The deficiency now has a location: these five components write
toward continuation at precisely the positions where the text
ends the phrase, and a12, whose ablation does not help, does not.
(b) HELD as registered but I am recording it as CONTAMINATED
rather than banking it. The helpers' mean pairwise cosine is
0.712 against a random null of 0.032 -- but the CONTROL sits at
0.650 with them, nearly as high. That is the stream-centre effect
from 449/450: every write at these depths is partly aligned with
the layer-5 bias axis, so raw cosines between component writes
are inflated and cannot discriminate helpers from non-helpers.
The geometry leg tells us nothing until the centre is projected
out; the LOGIT asymmetry, which is a difference of two numbers
computed the same way, is unaffected by that inflation and is
what carries the result.
punct_overconf_recentered queued: redo the geometry with the bias
axis projected out of every write and re-check the logit
asymmetry under the same projection. Registered: (a) recentered
cosine >= 0.30 AND at least 0.15 above the control's, (b) verdict
recorded either way, (c) the logit asymmetry survives.

## 459. Recentered: the geometry is dead, the logit asymmetry is 100x -- and a bar-design lesson

punct_overconf_recentered projected the layer-5 stream-centre axis
out of every component write and redid both legs.
  component  ->competitor  ->target   margin
  a3            4.926        1.781    +3.15
  a6            4.090       -3.520    +7.61
  a7           10.048        3.101    +6.95
  a8           11.726        2.816    +8.91
  m7           13.041        8.145    +4.90
  a12 CTRL      5.996        5.957    +0.04
(a) FAILED, and it settles the question: after recentering the
helpers' mean pairwise cosine falls to 0.294 while the CONTROL
sits at 0.349 -- HIGHER than the helpers. The five components do
NOT share a special direction. 458's 0.712 was the stream-centre
inflation and nothing else; the geometry leg is now closed as
uninformative rather than left as a soft positive.
(c) FAILED AS WRITTEN, and the failure is my bar's fault, not the
result's: I required the control to not push the competitor AT
ALL, and after recentering a12's margin is +0.039 -- positive by
four hundredths. Every helper sits between +3.15 and +8.91. The
discrimination is a factor of ~100 in MAGNITUDE and the binary
sign test threw it away. Lesson recorded for the program's bar
design: when the quantity is continuous and the effect is a
ratio, register a margin, never a sign.
Standing statement, unchanged by both failures: five components
push the wrong continuation at these sites by 3 to 9 logits, the
one component whose ablation does not help pushes it by 0.04, and
the effect is not carried by a shared direction. The deficiency
is distributed across components that each independently favour
continuation.
Report updated and republished with the whole arc as a new
section: "A bias the model actually has".
punct_repair queued -- the strongest test available for a claimed
deficiency. Fit the over-continuation direction on census sites,
subtract a scaled multiple of it from the residual at EVERY
position on FRESH FineWeb rows, and see whether the model gets
BETTER. Registered: (a) CE falls at the best scale, (b) the gain
concentrates at punctuation by >= 3x, (c) a random direction of
the same norm at the same scales does not help.

## 460. The repair FAILS -- and the failure says what kind of thing the bias is

punct_repair fitted the over-continuation direction on census
sites (the five helping components' mean write at helped
punctuation positions minus their mean write elsewhere, norm
1357) and subtracted scaled multiples of it from the residual at
every position on 48 FRESH FineWeb rows:
  scale   repair: dCE punct / non-punct    random: punct / non-punct
  0.05      +0.057  /  +0.003               +0.009 / +0.006
  0.10      +0.130  /  +0.028               +0.022 / +0.028
  0.20      +0.345  /  +0.175               +0.059 / +0.141
  0.40      +1.343  /  +1.202               +0.299 / +1.089
(a) FAILED at every scale -- the correction makes the model WORSE,
never better. (b) FAILED. (c) HELD (the random control also
hurts).
The informative part is the comparison, not the failure: at the
smallest scale the fitted direction hurts punctuation SIX TIMES
more than a random direction of the same norm (+0.057 against
+0.009). Subtracting what those components write at boundary
positions does not remove a bias -- it removes signal they are
also carrying, and it removes it precisely where it matters most.
WHAT THIS SETTLES: the over-continuation bias is NOT a fixed
additive vector. That is a real constraint, and it separates this
deficiency sharply from the layer-5 sink constant, which IS a
fixed vector and can be replaced by one for free (435). This
model contains both kinds of bias -- one constant and removable,
one input-dependent and not -- and the same experimental move
(subtract the mean direction) succeeds on one and fails on the
other. 447's accounting problem was the first hint that these
components' contributions are not additive; this is the second,
from the opposite direction.
punct_oracle_ceiling queued to measure headroom rather than guess
another fix: mean-ablate the five components ONLY at positions
whose true next token is punctuation (an oracle upper bound on
what any detector could buy), and compare against the same gate
driven by a purely CAUSAL cue available at inference. Registered:
(a) the oracle lowers CE, (b) its gain at punctuation is >= 0.05
nats, (c) the causal proxy captures >= 30% of it -- and if not,
the honest finding is that the deficiency is real but not cheaply
fixable.

## 461. THE ORACLE COMES BACK BACKWARDS -- and exposes selection in my own chain

punct_oracle_ceiling mean-ablated the five helping components
ONLY at positions whose true next token is punctuation, on fresh
FineWeb rows:
  baseline overall CE      3.3536
  oracle gate              +0.0270 overall, +0.1168 at punctuation
  causal proxy gate        +0.3285 overall
ALL THREE BARS FAILED, and not narrowly -- the oracle arm, which
should have been an upper bound on benefit, makes the model WORSE
exactly where it was aimed.
That forced me to re-examine the chain rather than the run, and
the problem is mine: 457's competitor statistics were computed at
positions selected BECAUSE ablation helped there (d < 0). Reporting
that ablation helps at positions chosen for being helped is
circular, and the striking "75% of the time" figure -- which I put
in the published report -- inherits that conditioning.
WHAT IS ACTUALLY ESTABLISHED, separated by whether selection was
involved:
  UNSELECTED and still standing -- 440's held-out test: ablating
  r.13.2.1's 16-dimensional probe BUNDLE on fresh FineWeb rows
  lowers CE at punctuation targets (-0.008) while RAISING it
  elsewhere in the same forward passes (+0.016), permutation
  p = 0.000.
  SELECTED and therefore weaker than reported -- 457's 75%
  over-continuation rate, 458/459's competitor-margin table
  (those sites were chosen as helped), and the report paragraph
  built on them.
  NEWLY IN CONFLICT -- whole-component ablation and 16-dim bundle
  ablation point in OPPOSITE directions at punctuation on fresh
  text. The arc treated them as the same intervention from 455
  onward. They are not.
Corrections propagated immediately: the report section now carries
a "Correction in progress" paragraph stating the circularity, the
backwards oracle result, and what survives unselected; the artifact
is republished.
punct_unselected queued to settle it without any conditioning: on
fresh rows, measure the intact top-1 at ALL punctuation targets,
and price bundle-ablation against component-ablation on the same
rows. Registered: (a) unselected over-continuation >= 60% or the
claim was an artifact of conditioning, (b) the bundle dissociation
replicates, (c) the two interventions differ in direction.

## 462. THE OVER-CONTINUATION CLAIM IS REFUTED -- and the survivor is much smaller

punct_unselected, no conditioning anywhere, 48 fresh FineWeb rows:
  intact top-1 at ALL 1602 punctuation targets:
      non-punctuation only 23.5% of the time
(a) FAILED against a 60% bar, and it is not close. Unselected, the
model gets punctuation RIGHT three times in four. 457's 75%
over-continuation rate was entirely an artifact of conditioning on
positions where ablation helped. THE OVER-CONTINUATION CLAIM IS
WITHDRAWN, including from the published report, which now carries
the refutation in place of the claim (section retitled "Three
readings that died").
(b) HELD: the bundle dissociation replicates cleanly --
  bundle:      punct -0.0098   non-punct +0.0153   (-0.0251)
(c) FAILED, and it dissolves 461's supposed conflict:
  components:  punct +0.3246   non-punct +0.7462   (-0.4216)
Both interventions SPARE punctuation; they differ only in overall
severity. 461's oracle looked backwards because of my design, not
the model's behaviour -- gating ablation to punctuation positions
compares "damage only there" against "no damage", which is bound
to be worse at those positions; the meaningful comparison is
against damaging everywhere, and under that comparison punctuation
is spared under both interventions.
FINAL STATE OF THIS ARC, honestly graded. Three readings of the
same measurement died: a CIRCUIT claim (not leaf-specific, 455), a
FREQUENCY claim (frequency does not explain it, 456), and a BIAS
claim (selection did, 462). What survived all three is the
smallest of them and it is real: ablation damages punctuation
predictions less than others, reproducibly, on unseen text, under
two very different interventions.
punct_confidence queued to finish it: punctuation may be spared
simply because it is PREDICTABLE, and confident predictions
survive damage. Registered: (a) after matching positions on the
intact model's confidence the sparing shrinks below 25% of its
unmatched value, (b) damage falls monotonically with confidence
decile, (c) if (a) fails something class-specific remains.
If (a) holds, the whole arc reduces to "robust predictions are
robust", the correct and unglamorous end -- and the program keeps
the methodological yield instead: never compute a rate at
positions selected by the outcome you are measuring.

## 463. Predictability does NOT explain the sparing -- something class-specific survives

punct_confidence matched punctuation and non-punctuation
positions on the intact model's top-1 probability, decile by
decile, on fresh rows:
  unmatched punctuation sparing        -0.0251
  confidence-matched sparing           -0.0229  (91.4% of it)
  damage by confidence decile: 0.009, 0.014, 0.028, 0.014,
    0.011, 0.017, 0.012, 0.011, 0.006, -0.002
(a) FAILED: 91% of the sparing survives matching -- predictability
does not explain it. (b) FAILED too, and it is the reason: damage
is essentially FLAT across confidence deciles rather than falling,
so the premise that confident predictions are more robust is
itself wrong in this model (only the top decile dips).
So the deflationary ending I expected does not arrive. After a
circuit reading died (455), a frequency reading died (456), and a
selection artifact was exposed and the headline withdrawn (462),
what remains is small, unselected, replicated on unseen text under
two interventions -- and now also NOT reducible to the target
simply being easy to predict. Something class-specific is real.
The honest arc summary at this point: five hypotheses tested,
four killed, one standing.
class_sparing queued to scope the survivor: is punctuation
special, or one member of a structural/format family? The same
bundle ablation, priced on fresh rows for six mechanical target
classes (punct, newline, digit, subword, space_word,
capitalized), each against everything else. Registered: (a) at
least one other class is also spared, (b) content classes take
EXTRA damage, (c) punctuation is the most spared or within 0.005
of it.

## 464. The survivor scoped: a FORMAT-TOKEN family, and content pays

class_sparing priced the same bundle ablation on fresh rows for
six mechanical target classes, each against everything else:
  class         n      dCE in    dCE out   dissociation
  punct       1602    -0.0097    +0.0154     -0.0251
  digit        223    -0.0059    +0.0124     -0.0183
  subword     1496    +0.0083    +0.0126     -0.0044
  newline      350    +0.0383    +0.0113     +0.0269
  space_word  8531    +0.0165    +0.0022     +0.0143
  capitalized 1235    +0.0174    +0.0115     +0.0059
ALL THREE BARS HELD. (a) punctuation is not alone -- digits are
spared too, and subwords marginally. (b) content classes pay
extra: space-initial words and capitalised words take MORE damage
than average. (c) punctuation is the most spared.
One prediction inside (a) was wrong in an interesting way: I named
NEWLINE as the natural second member of a format family, and
newline is the single WORST-damaged class (+0.0269). So the split
is not "format versus content". It is closer to
short-closed-class-token versus everything else: punctuation and
digits are spared, ordinary words and capitalised words pay, and
newlines -- which in this corpus mark document structure rather
than in-sentence syntax -- pay most of all.
Standing statement for the arc: ablating this bundle shifts the
model's competence AWAY from content words and TOWARD punctuation
and digits. Four explanations are dead (circuit-specific,
frequency, selection, predictability) and the survivor now has
measured boundaries.

## 465. The free heads are genuinely free -- no redundancy pool

free_head_redundancy tested whether the 39 individually-free heads
are spare capacity or a redundancy pool, after the sink pair
showed that "free" can mean "covered by a partner" (454).
  individual costs, summed              -0.3372
  all 39 deleted jointly                -0.0296
  random subsets of 20 (3 draws)        -0.076, -0.074, -0.050
  random subsets of 10 (3 draws)        -0.063, -0.039, -0.162
  control: the 39 COSTLIEST heads       +2.9016
(a) FAILED and (b) FAILED, both in the same direction: deleting
all thirty-nine at once is still slightly BENEFICIAL (-0.030), and
subsets are no cheaper than the whole. There is no superadditive
blow-up, no hidden redundancy, no partner effect at the population
level. (c) HELD emphatically -- the same-sized costly set costs
+2.90, a hundred-fold difference.
So the sink pair's mutual cover (454) is a LOCAL structure, not a
general property of cheap heads. Nearly a quarter of this model's
attention heads can be removed together, on the corpus it was
trained for, for free. That is a fact about trained-model slack
worth having as a plain number, and it sharpens the earlier
finding: the 39 are not a redundancy pool waiting to be exercised,
they are simply not doing much.

## 466. Wave 6, and the swarm rediscovering a known effect for the fourth time

  r.11.1.1  gate 5.70 (5.88/5.52) | mechanism NEGATIVE on
            a4/a3/a8 (top ratios 1.09-1.14, headroom -0.23 to
            -0.32) | punctuation claim ROBUST_V2, 44/63 = 69.8%
            vs 48% base, p=0.0004, margin +21.8 points -- and,
            following 462's rule, computed on a CLASS-defined
            position set, not an outcome-selected one
  r.4.1.1   gate 4.83 (5.00/4.66) | mechanism NEGATIVE on a12/m1
            | downstream consumer a17 at 1.109 beats its random
            control (0.97) but misses the 1.3 bar | no behavioral
            claim survives (best was capitalized p=0.114)
r.11.1.1 is the FOURTH leaf to independently surface the
punctuation effect. Four independent discoveries of the same
population-level phenomenon is a pipeline problem, not four
findings, so CIRCUIT_SOP now carries a "KNOWN GENERAL EFFECTS"
section: punctuation and digit sparing is documented as
non-leaf-specific with its population value (-0.025), and an
agent claiming anything leaf-specific must show its leaf EXCEEDS
that value by a stated margin. Otherwise the step-5 budget goes
to the other classes. This is the swarm's own throughput being
protected from a real result.
Two tooling fixes from the wave, both from agent reports:
 - story_test_class now returns `margin` directly (the 459 rule
   asked agents to report margins and then made them compute it
   by hand).
 - leaf_output_decomp.py overwrote its shared results file
   wholesale, and silently dropped a concurrent agent's entry;
   the agent noticed, recovered the lost entry from git, and
   hand-merged. Now fixed to read-merge-write like write_circuit.
   That is the second concurrency hazard the swarm has found in
   the driver's tooling, and both were found by agents rather
   than by me.
damage_signature queued -- the deepest available deflation of the
surviving effect: run the identical class breakdown for a random
rank-matched subspace, a single head deletion and a single MLP
mean-ablation. Registered: if all reproduce the profile (punct and
digit spared, newline worst), the survivor is a UNIVERSAL DAMAGE
SIGNATURE of this model rather than anything about a circuit.

## 467. NOT a universal damage signature -- the arc closes with a narrow true statement

damage_signature ran the identical class breakdown under four
interventions on the same fresh rows:
  arm      punct   newline   digit   subword  space_wd  capital
  bundle  -0.0251  +0.0269  -0.0183  -0.0044  +0.0143  +0.0059
  random  +0.0068  -0.0021  +0.0255  +0.0028  -0.0063  +0.0011
  head    -0.0082  -0.0012  +0.0142  +0.0054  +0.0009  +0.0055
  mlp     -0.0615  -0.0258  +0.0146  +0.0719  -0.0020  -0.0005
(a) FAILED and (b) FAILED -- so the deflation does not land, and
that is the informative outcome. A random rank-matched subspace
produces the OPPOSITE sign at punctuation (+0.007) and at digits
(+0.026). Real machinery of three different kinds all spare
punctuation (-0.025 bundle, -0.008 head, -0.062 MLP), but their
FINER profiles disagree: newline is the worst-damaged class under
the bundle (+0.027) and a SPARED class under the MLP (-0.026);
digits are spared only by the bundle.
FINAL STATE OF THE ARC, and it closes here. Ablating real
machinery in this model shifts competence away from ordinary words
and toward punctuation; arbitrary perturbations of the same size
do not. The finer class profile depends on which machinery is
removed, so there is no single "damage signature" to name. Five
readings were tested and four died: a CIRCUIT claim (455), a
FREQUENCY claim (456), a BIAS claim built on selected positions
(462, withdrawn from the published report), and a PREDICTABILITY
claim (463). The survivor is narrow and true, which is all this
thread supports.
The lasting yield is methodological and is now enforced in the
swarm's instructions: never compute a rate at positions selected
by the outcome being measured (462), register a margin rather
than a sign (459), and check a reconstructed quantity's norm
against the residual before injecting it (443/447).
Report updated and republished with the closed-out section.
Next, back to mechanism: a14_pathway queued to escalate the
program's ONE confirmed enrichment (a14 into r.3.0.2's late
attention machinery, bootstrap-stable and recentering-robust)
from a correlational statement about input composition to a
causal one. Registered: (a) ablating a14 damages that leaf's
members selectively at concentration >= 2, (b) an adjacent-depth
control does not, (c) a14 and the leaf's own bundle are
subadditive, i.e. one pathway rather than two.

## 468. a14's enrichment IS causal -- but it is not unique, and it is a separate pathway

a14_pathway mean-ablated components and scored r.3.0.2's own
member set:
  arm            concentration   member |dCE|   off-slice |dCE|
  a14 alone           5.76           0.589          0.102
  a13 control         3.40           0.333          0.098
  the leaf's bundle   6.10           0.329          0.054
  a14 + bundle        7.23           0.892          0.124
(a) HELD: ablating a14 -- the writer the mechanism table flagged
as enriched -- damages this leaf's members at concentration 5.76,
essentially matching the leaf's OWN probes (6.10). The correlational
enrichment statement is now backed by a causal one.
(b) FAILED: the adjacent-depth control a13 reaches 3.40, well
above the 1.5 bar. Selectivity for this leaf is a GRADIENT over
nearby components, not a property of a14 alone. a14 is the
strongest by a clear margin (5.76 against 3.40) but the claim
"a14 specifically" is too strong as I wrote it.
(c) FAILED, and the arithmetic is worth stating: joint member
damage is 0.892 against 0.589 + 0.329 = 0.918 summed -- almost
exactly ADDITIVE. I predicted subadditivity (one shared pathway);
the data say a14 and the leaf's own bundle damage these members
through largely INDEPENDENT routes. Two pathways, not one.
This is the program's first confirmed enrichment escalated to a
causal claim, with both of its accompanying predictions refuted.
The honest record: the mechanism table pointed at a component
whose removal really does hit this leaf hardest, and the two
things it does not tell you are whether nearby components do the
same (they partly do) and whether the writer and the leaf's
machinery form one circuit (they do not).
enrichment_predicts queued to test the tool itself rather than
one pair: across twelve components spanning the writer table --
top-enriched, mid-table, bottom-table, and components absent from
it -- does the enrichment ratio predict ablation selectivity?
Registered: (a) Spearman >= 0.5, (b) a14 tops both measures, (c)
absent components are less selective than the top-five enriched.
If (a) fails, the swarm's central instrument does not predict
causal importance and that has to be said plainly.

## 469. THE INSTRUMENT IS VALIDATED: enrichment predicts causal selectivity at rho 0.84

enrichment_predicts mean-ablated ten components spanning
r.3.0.2's writer table and correlated each one's enrichment ratio
against the concentration of damage it causes on that leaf's
members:
  component  enrichment   concentration
  a14           2.431          5.76
  a15           2.334          5.40
  a16           2.171          8.56
  a13           1.774          3.40
  a12           1.488          2.61
  a7            1.064          1.97
  m0            1.035          1.48
  a1            1.021          1.84
  m5            0.840          2.12
  a0            0.737          1.68
  Spearman = 0.842
(a) HELD with room, and this is the result the swarm's whole
method rests on: a cheap correlational table computed from
forward passes alone predicts which component's REMOVAL will hurt
a circuit's positions, at rank correlation 0.84. (b) HELD -- a14
is top-two on both measures. The ordering is not perfect (a16 has
the third-highest enrichment and the highest concentration at
8.56), so the table ranks rather than measures.
(c) FAILED, and the bar was unevaluable rather than false: I
registered a control of components ABSENT from the writer table,
but for a leaf whose machinery sits at layers 15-17 the table
covers essentially every earlier component, so the absent set was
empty. Recorded as a design error, not a negative result.
Report updated with an instrument-check paragraph and
republished.
enrichment_generalize queued to ask the harder version: does the
ordering survive on leaves whose table is NEGATIVE everywhere
(every ratio near 1.0)? If it does, negative tables still carry
usable signal below their own threshold; if it does not, a
negative table means exactly nothing and the swarm should stop
reporting the ratios in that case. Registered: (a) Spearman >=
0.5 on at least one negative leaf, (b) negative leaves show a
flatter concentration range than r.3.0.2's 7.08, (c) pooled
Spearman across all three leaves >= 0.5.

## 470. Generalization, and a CONFOUND I put in my own validation

enrichment_generalize ran the same correlation on two leaves whose
tables are NEGATIVE everywhere, with the positive leaf as control:
  leaf        Spearman   concentration range
  r.5.3.1       0.905          1.87
  r.13.2.1      0.119          2.02
  r.3.0.2       0.783          7.08
  pooled        0.521
All three bars HELD as written -- but only just, and the split is
the real content. Ordering survives on ONE negative leaf (0.905)
and collapses on the other (0.119), which across n=2 is a coin
flip, and on both the concentration range is about 2 against the
positive leaf's 7.1. So on negative leaves the tool's ranking is
unreliable AND the thing being ranked barely varies. SOP updated
accordingly: quote and chase ratios when ENRICHED_STABLE2 is true
somewhere; when it is false everywhere, report the flat profile
and the threshold and stop -- do not rank writers by ratio.
CORRECTION to 469, found while reading the component list rather
than by any run: three of the ten components in that validation
(a15, a16, a17) ARE r.3.0.2's own machinery, and two of them were
in the sample. Ablating a component that CONTAINS the leaf's probe
directions is close to ablating the leaf itself, so their
concentrations (5.40 and 8.56) are partly definitional and inflate
the correlation. Recomputed with the leaf's own machinery
excluded:
  rho = 0.762 on the eight remaining components (was 0.842)
The instrument's validation stands -- 0.76 is still strong, and
the top-ranked non-machinery writer a14 is still the most damaging
non-machinery component -- but the headline number moves down and
the published report's "0.84" needs the same treatment at the next
boundary update.
mech_map_all queued to use the validated tool at census scale: run
the table over 60 shortlist leaves and aggregate. Registered: (a)
fewer than 25% of leaves carry any stable enrichment, (b) the
positive pairs concentrate on a few recurring writers, (c) those
writers are within two layers of what they feed at least 60% of
the time -- i.e. the tool mostly finds local structure, which is
what 425 warned about.

## 471. CENSUS-SCALE MECHANISM MAP: positives are rare, local, and concentrated

mech_map_all ran the validated mechanism table over 60 shortlist
leaves (615 s):
  60 leaves scanned
  3 leaves carry any ENRICHED_STABLE2 positive -- 5.0%
  3 positive (leaf, component, writer) pairs in total:
     r.6.2.0   m15 -> m17   min ratio 2.232 (threshold 1.98)
     r.1.2.2   m14 -> m15   min ratio 2.333 (threshold 1.739)
     r.1.2.0   m14 -> m15   min ratio 2.277 (threshold 1.412)
  top writers: m14 (2), m15 (1)
  adjacency fraction: 1.00
ALL THREE BARS HELD, and the first one held by a wide margin: I
registered "fewer than 25% of leaves carry a positive" and the
answer is FIVE PERCENT. After the tool was validated against
causal ablation (469/470), its verdict at scale is that this
model's damage-clusters almost never have a single dominant
upstream writer. Fifty-seven of sixty leaves are input-diffuse.
(b) held trivially at n=3, and (c) held at 1.00 -- every positive
is an ADJACENT-LAYER MLP pair, m14 into m15 twice and m15 into
m17 once. That is precisely the shape the wave-3 reviewer warned
about (425), so the positives are exactly the cases most likely to
be layer properties rather than circuit facts.
Two things follow, and the second is the one that matters for the
swarm's economics. First, the honest census statement: in this
model, "which component feeds this circuit" has a concentrated
answer about one time in twenty. Second, the swarm has been
spending a mechanism step on every leaf to find a positive 5% of
the time -- so the SOP's step-3M should be understood as a cheap
SCREEN whose usual outcome is a scoped negative, not as the
deliverable it was promoted to in v3. That is a real cost finding
about the pipeline, from the pipeline's own output.
mech_map_specificity is running on all three positives: for each,
read the same (component, writer) ratio from peer leaves that
share the component and ask whether the positive leaf stands out
by at least 0.5. Registered: (a) >= 30% of positives are
leaf-specific, (b) adjacent pairs -- which here means all of them
-- are specific less often than distant ones.

## 472. ONE REAL MECHANISM IN SIXTY LEAVES -- and it is shared by two siblings

mech_map_specificity checked each of the map's three positives
against peer leaves that share the same component:
  r.6.2.0  m15 -> m17   2.232 vs peers [1.333, 2.691, 1.739]
                        median 1.739, gap 0.49 -> LAYER PROPERTY
                        (one peer actually scores HIGHER at 2.691)
  r.1.2.2  m14 -> m15   2.333 vs peers [0.934, 1.167, 0.691]
                        median 0.934, gap 1.40 -> LEAF-SPECIFIC
  r.1.2.0  m14 -> m15   2.277 vs the same peers -> LEAF-SPECIFIC
(a) HELD: 2 of 3 positives are leaf-specific. (b) FAILED and was
UNEVALUABLE -- I registered a comparison against distant pairs
when the map had already reported an adjacency fraction of 1.00,
so the distant category was guaranteed empty. My design error, the
third of this kind; the rule I keep relearning is to check that a
registered comparison class can actually be populated by the data
that will exist when the bar is scored.
THE CENSUS BOTTOM LINE, which is now quotable: in 60 certified
leaves this model has ONE genuinely leaf-specific input mechanism
-- m14 into m15 -- and it is carried by two SIBLING leaves
(r.1.2.2 and r.1.2.0, both under r.1.2), which is what two views
of one circuit should look like. Everything else is either
input-diffuse (57 leaves) or an adjacent-layer property (1).
That is a modest yield and an honest one: the program's mechanism
tooling, validated against causal ablation at rho 0.76, says this
model mostly does not have the kind of structure the tool looks
for.
SOP updated: step 3M is now documented as a SCREEN whose usual and
correct outcome is a scoped negative, with the 5% census figure
quoted so agents stop treating a negative as a failed task.
m14_pathway queued to escalate the survivor, with bar design
corrected by 468's failures -- no absolute control threshold
(adjacent components are partly selective everywhere) and no
assumption of a single shared pathway. Registered: (a) m14
ablation is selective on BOTH siblings at concentration >= 2, (b)
it beats the m13 control on both (a RELATIVE bar), (c) the two
siblings' concentrations agree within 25%, as two views of one
circuit should.

## 473. THE LAST MECHANISM FAILS ITS NEIGHBOUR CONTROL -- census yield is zero writers

m14_pathway escalated the census's only surviving leaf-specific
enrichment to a causal test on both sibling leaves:
  leaf       m14 (enriched)   m13 (control)   bundle   m14+bundle
  r.1.2.2    conc 4.33        conc 4.44       3.75      3.74
  r.1.2.0    conc 4.29        conc 4.89       3.49      4.07
(a) HELD: m14 is genuinely selective on both siblings. (c) HELD:
the two siblings agree to within 1% (4.33 against 4.29), which is
exactly what two views of one circuit should look like and is the
cleanest evidence yet that the census's sibling structure is real.
(b) FAILED, and it is decisive: the ADJACENT component m13, which
the enrichment table does NOT flag (peers ~0.9, m13 not in the
positive list), damages these leaves' members MORE than m14 does,
on both siblings. The same pattern appeared for a14 versus a13
(468). So the one input mechanism that survived a 60-leaf screen
and a peer-specificity check does not survive a neighbour control.
FULL CENSUS YIELD, stated plainly: 60 certified leaves -> 3 screen
positives (5%) -> 2 leaf-specific after peer comparison -> 0 that
are causally distinguishable from an adjacent unflagged component.
Input-composition analysis in this model RANKS how much a
component matters (rho 0.76 against causal ablation) and CANNOT
separate a writer from its neighbours. That is a real limitation
of the method as applied here, not a shortfall of effort, and it
is now the headline of the report's instrument section and a
mandatory neighbour control in the SOP.
band_unit queued to take the obvious next step rather than
mourn: if selectivity is smooth across adjacent layers, the unit
is a BAND, not a writer. Registered: (a) the band m13+m14 damages
members more than either alone on both siblings, (b) extending to
m12 adds under 20% -- a sharp boundary, (c) a distant band of the
same width does less than half the damage.

## 474. Bands do not isolate either -- the input-composition thread closes

band_unit tested whether a contiguous BAND is the right unit once
single writers failed their neighbour control:
  arm              r.1.2.2 conc / member    r.1.2.0 conc / member
  m14                4.33 / 0.675             4.29 / 0.670
  m13                4.44 / 0.773             4.89 / 0.851
  band m13+m14       4.37 / 1.330             4.89 / 1.487
  band m12+m13+m14   3.72 / 1.879             4.29 / 2.171
  distant m8+m9      2.71 / 1.138             3.22 / 1.351
(a) HELD but only trivially: the band damages more than either
part -- and almost exactly the SUM of them (0.675 + 0.773 = 1.448
against 1.330 measured). Additive, not synergistic, so "band"
buys no explanatory power over "two components".
(b) FAILED: adding m12 adds 41-46% more damage. There is no sharp
boundary; damage keeps accumulating as layers are added.
(c) FAILED: the DISTANT band m8+m9 causes 86-91% as much member
damage as the adjacent one. Absolute damage is barely local at
all.
The one place locality does show is SELECTIVITY: concentration is
4.4-4.9 for the adjacent band against 2.7-3.2 for the distant one.
So nearby components damage these members more SPECIFICALLY, while
distant ones damage them nearly as much in absolute terms by
damaging everything.
THREAD CLOSED. Across 474 and its predecessors: leaf damage
profiles in this model are not attributable to single writers
(473, neighbour control), not to bounded bands (474, no boundary),
and are only mildly local (selectivity gradient, not a magnitude
one). What input-composition analysis delivers here is a
magnitude RANKING (rho 0.76) over a smoothly accumulating
dependence -- and the program should stop asking it for culprits.
Pivot, and it is the pivot the evidence supports: this program's
two COMPLETE circuits -- induction and the position-0 bias -- both
came from chasing a specific anomaly, not from the census. So take
the next anomaly. head_0_3_fold queued on head 0.3, the second
costliest head in the model (+0.112, behind only the sink). Layer
0 is special: attention there reads only token embeddings, so the
head's pattern is a pure function of tokens and rotary and should
be EXACTLY foldable with no forward pass. Registered: (a) one
offset carries >= 60% of its reads, (b) the token+rotary fold
reproduces the real top read >= 95% -- at layer 0 a miss means a
bug, not a finding -- and (c) replacing its value output with a
per-read-token table costs <= 0.02 nats.

## 475. Head 0.3 is a previous-token head, and its pattern folds EXACTLY

head_0_3_fold on the model's second costliest head (+0.112 nats,
behind only the sink at +0.916):
  offset -1   663 reads   65.8%
  offset  0   283         28.1%   (self)
  offset -2    55          5.5%
  same-token rate 0.282 | fold match 1.000
(a) HELD: a single offset carries 66% -- head 0.3 is a
previous-token head with a substantial self component.
(b) HELD exactly: predicting its top read from WEIGHTS, TOKENS AND
ROTARY ALONE, with no residual stream, reproduces the real read
1.000 of the time. At layer 0 that is what the architecture
demands (attention there reads only token embeddings), so the
value of this arm is as a bug check on the machinery, and the
machinery passes cleanly.
(c) FAILED, and the fault is visible in my own code rather than in
the model: the per-read-token value table was accumulated only
from tokens that appeared as TOP READS at sampled query positions,
so most positions in the replacement pass hit an empty slot and
contributed zero. The +0.054 is table coverage, not head
behaviour.
The fix makes the test stronger rather than weaker. At layer 0 the
value is a pure function of the token, v = c_v(rms_norm(wte(t))),
so the correct table is computable from WEIGHTS ALONE over the
entire 50304-token vocabulary, with no data and no holes.
head_0_3_exact queued with three arms: the weights-only value
table, a FULL fold replacing both pattern and values (the whole
head as a lookup), and a token-shuffled table as the null.
Registered: (a) the value table costs <= 0.005 nats (an identity
up to numerics), (b) the full fold costs <= 0.01, (c) the shuffled
null costs >= 0.05.
If (b) holds, the model's second most expensive attention head is
exactly a table -- which would join the sink (a constant) and the
induction band (four reads of an identity code) as the third
component of this model reduced to something you can write down.

## 476. THE SECOND COSTLIEST HEAD IS EXACTLY A LOOKUP

head_0_3_exact, all three bars HELD and two of them at the floor:
  weights-only per-token value table          dCE -0.0
  FULL fold (pattern AND values, weights-only) dCE -0.0
  token-shuffled table (null)                  dCE +0.14675
So head 0.3 -- +0.112 nats to delete, the model's second most
expensive attention head -- can be replaced ENTIRELY by two
weights-only lookups, its pattern rebuilt from tokens and rotary
and its values read from a 50304-entry table, at zero measurable
cost. The shuffled null at +0.147 confirms the table's CONTENT is
what matters, so this is not a vacuous substitution.
The honest framing matters here. At layer 0 attention reads only
token embeddings, so exact foldability is guaranteed by the
architecture rather than discovered in this head. What the run
delivers is threefold: the machinery is verified exact (a strong
instrument check after a night of scale and coverage bugs), the
head's SHAPE is named (previous-token at 66% with a 28% self
component), and the practical consequence is real -- an expensive
head needs no attention computation at runtime at all.
That makes three components of this model now reduced to something
writable: the sink (a constant vector), the induction band (four
reads of an identity code), and head 0.3 (a token-pair lookup).
layer0_fold queued to ask the whole-layer version with its
boundary: fold ALL NINE layer-0 heads at once, with a
token-shuffled null, and apply the identical construction to layer
1 -- which reads layer-0 outputs rather than raw tokens and should
therefore FAIL. Registered: (a) the whole layer folds at <= 0.01
nats, (b) the null costs >= 0.20, (c) layer 1 costs >= 0.10. If
all three land, "the first attention layer of this model is a
bigram table" becomes a measured statement with a measured
boundary.

## 477. THE FIRST ATTENTION LAYER IS A BIGRAM TABLE -- with a sharp boundary at layer 1

layer0_fold, all three bars HELD:
  all nine layer-0 heads folded to weights-only
    token-pair patterns + per-token value tables   dCE  -0.00000
  the same tables with token identities shuffled   dCE  +0.23687
  the IDENTICAL construction applied to layer 1    dCE  +1.47026
The entire first attention layer of this model can be replaced by
lookups computed from weights alone -- no forward pass, no
residual stream -- at zero measurable cost, and the replacement is
not vacuous: shuffling the token identities in those same tables
costs 0.24 nats.
The boundary is the part worth keeping. Applying exactly the same
construction one layer up costs +1.47 -- a hundred and fifty times
more -- because layer 1 reads layer-0 OUTPUTS rather than raw
tokens. So "this attention layer is a bigram table" is true of
layer 0, false of layer 1, and the transition is measurable to two
decimal places rather than argued.
Report updated at this phase boundary and republished.
Note on what this is and is not: layer-0 tableability follows from
the architecture, so the finding is not that the model learned
something surprising -- it is that the front of this model has an
exactly writable form, verified end to end, and that the writable
region stops immediately. Combined with 476 (head 0.3's shape),
435-441 (the sink is a constant) and 376-408 (induction is four
reads of an identity code), the tally of this model's components
reduced to something you can write down now includes an entire
layer.
mlp_table_ladder queued to find where per-token TABLEABILITY ends
on the other side of the front. mlp0's input is rms_norm(E +
attn0_out), not the token alone -- 393 showed that contextual term
matters for the induction trigger -- so the question is what
ignoring it costs, and how fast that cost grows with depth.
Registered: (a) a per-token table for m0 costs <= 0.10 (the
identity-code generator is nearly a table), (b) cost rises
monotonically m0 < m1 < m2, (c) the shuffled m0 table costs
>= 0.50.

## 478. CORRECTION: the front MLPs are NOT per-token tables

mlp_table_ladder replaced each of the first three MLPs with a
per-token table computed from weights alone,
T[t] = mlp_i(rms_norm(wte(t))):
  m0  dCE +1.018      m1  dCE +1.775      m2  dCE +0.744
  m0 with shuffled token identities: +3.188
(a) FAILED by a factor of ten: m0 costs 1.02 nats as a per-token
table, not the <= 0.10 I registered. (b) FAILED: the ladder is not
monotone -- m1 is the most context-dependent of the three and m2
the least. (c) HELD: the shuffled null at 3.19 confirms the
tables' content matters, so the failures are about CONTEXT, not
about broken machinery.
CORRECTION to a framing this program has been carrying since 387:
mlp0 is NOT an exactly-foldable token table. Its input is
rms_norm(wte(t) + attn0_out), and ignoring the attention term
costs a full nat. 393 already showed the token-only fold of m0
predicted only 66% of head 1.4's reads against the real m0's
99.8%; this puts the price on it in loss. The published report's
"token-dictionary cascade" figures came from a variance-explained
measure, which is compatible -- 68-85% of variance can be
token-determined while the residual costs a nat -- so the report
now carries both numbers side by side and says which is which.
Republished.
What the failure points at is specific rather than vague. attn0 is
a bigram table (477), dominated by the previous token (476: head
0.3 reads offset -1 at 66%), so m0's true input is close to a
function of the TOKEN PAIR. mlp_bigram_table queued: recompute m0
on rms_norm(wte(t) + attn0-with-all-weight-on-the-previous-token),
making it a pure function of (t, t_prev), against the per-token
table as reference and a shuffled-previous-token null.
Registered: (a) the pair form costs <= 0.30, (b) it beats the
unigram table by >= 0.50, (c) the null costs >= 1.00.

## 479. The token PAIR is not the answer either -- and a self-inflicted recursion bug

mlp_bigram_table first crashed with a RecursionError: my forward
hook on mlp0 called mlp0 inside itself. Fixed by computing the
bilinear forward manually from the weights (Left, Right, Down,
Down_bias) -- recorded because it is the fifth self-inflicted
instrument fault of this run, and like the others it was caught by
the run failing loudly rather than by returning a plausible
number.
With the fix:
  m0 as a per-token table (reference, 478)        +1.0180
  m0 recomputed with ALL of attn0's weight on the
    previous token, i.e. a function of (t, t_prev) +1.2554
  the same with the previous token shuffled        +1.5459
(a) and (b) FAILED, and in the informative direction: the token
PAIR form is WORSE than ignoring attn0 entirely. Concentrating all
nine layer-0 heads onto offset -1 -- because ONE of them (head
0.3) reads there 66% of the time -- produces something worse than
no attention contribution at all. The other eight heads read
elsewhere, and forcing them to the previous token actively
misinforms m0. (c) HELD: shuffling the previous token costs 1.55,
so the machinery works.
So the front of the model is exactly computable from tokens (attn0
is a bigram table, 477) but m0's context is NOT a
previous-token relationship. The open quantity is how WIDE a
prefix m0 actually needs, and that is directly measurable.
m0_context_window queued: rebuild attn0's pattern from weights and
tokens (the exact fold), truncate its reads to the last k
positions, and sweep k = full, 16, 8, 4, 2, 1. Registered: (a) the
untruncated fold is exact at <= 0.02 (a miss means a bug), (b)
k = 4 costs <= 0.20 -- m0's context need is local, (c) cost is
monotone in k with k = 1 >= 1.00, consistent with the +1.255
measured here.

## 480. THE FRONT OF THE MODEL IS A BIGRAM FUNCTION -- and 479 is RETRACTED

m0_context_window's first run failed its own sanity bar exactly as
registered ("a miss means a bug"): the untruncated fold cost
+0.5526 when it should have been an identity. The bug, found by
checking the block-0 lambdas rather than guessing: the residual
entering block 0's MLP is (lam0 + lam1) * E + attn_out, and
lam0 + lam1 = 12.1875 in this model. I had used 1.0 * E, so the
embedding was under-weighted TWELVEFOLD in every reconstruction
that fed m0.
Rerun with the correct mix:
  full (untruncated fold)   dCE +0.0000   <- exact, bar (a) HELD
  k = 16                    dCE -0.0105
  k = 8                     dCE -0.0268
  k = 4                     dCE -0.0420   <- bar (b) HELD
  k = 2                     dCE +0.0041
  k = 1 (self only)         dCE +0.5369
(c) FAILED as written: the sequence is not monotone (a four-token
window is slightly BETTER than the full one) and k = 1 costs 0.54
rather than the >= 1.00 I predicted.
THE RESULT: the front of this model -- the entire first attention
layer plus the first MLP -- is a BIGRAM FUNCTION. The current
token plus ONE previous position costs 0.004 nats against the full
model; four positions is free; only cutting to the current token
alone hurts, and even then by half what dropping attention
entirely costs (478's 1.018).
RETRACTION of 479, stated plainly: its headline -- "the token PAIR
form is worse than ignoring attn0 entirely" -- was produced by the
same twelvefold lambda bug and is WRONG. The corrected measurement
says the opposite: the token-pair form is essentially free. 479's
recursion-bug note stands; its scientific conclusion does not, and
the ledger and the report now carry the corrected version.
Report updated and republished with the bigram-front result and
the exactness anchor.
block1_window queued to measure how fast the required context
widens with depth: the same window sweep at block 1, whose
attention reads block-0 outputs. Registered: (a) untruncated is
exact, (b) k = 2 costs >= 0.30 there against +0.004 at block 0 --
layer 1 is where context widens, (c) k = 16 costs <= 0.10, so it
is still bounded.

## 481. Block 1 is local too -- context does NOT widen at layer 1

block1_window applied the 480 sweep one block up, restricting
attn1's reads to the last k positions:
  full  +0.0000  <- exact, sanity HELD
  k=32  +0.0025    k=16  +0.0110    k=8  +0.0105
  k=4   +0.0138    k=2   +0.0803    k=1  +0.7939
(a) HELD, (c) HELD, (b) FAILED: I predicted k = 2 would cost
>= 0.30 at block 1 against +0.004 at block 0, on the theory that
layer 1 is where context widens. It costs +0.080 -- twenty times
cheaper than predicted, and a four-token window is nearly free at
+0.014.
So the first TWO blocks of this model are both essentially local:
block 0 is a bigram function, block 1 needs about four positions.
Whatever makes this model more than an n-gram machine happens
later than layer 1, and the natural question -- where? -- is
directly measurable rather than speculative.
window_by_depth queued: restrict ONE attention layer at a time to
a 4-token window, all others intact, and sweep all eighteen. The
layer whose restriction is expensive is the layer that genuinely
needs distant reads. Registered: (a) layers 0-2 each cost <= 0.05,
(b) some layer costs >= 0.30 (a transition exists at all), (c) the
worst layer falls in 5-8, where the induction band lives and
long-range matching is this model's documented long-range
function.
If (c) holds, the depth profile of context requirement will line
up with the one circuit this program has closed end to end, which
would be a satisfying convergence of two very different
measurements. If it fails, the model's long-range dependence lives
somewhere the circuit work has not looked.

## 482. ONE LAYER CARRIES THE MODEL'S NON-LOCAL READS -- and an obvious confound

window_by_depth restricted each attention layer in turn to a
4-token read window, all others intact:
  L0 -0.042   L1 +0.014   L2 +0.021   L3 +0.039   L4 -0.062
  L5 +1.112   L6 +0.018   L7 +0.006   L8 +0.038   L9 -0.002
  L10 -0.018  L11 +0.022  L12 +0.086  L13 +0.027  L14 +0.007
  L15 +0.003  L16 +0.012  L17 +0.008
ALL THREE BARS HELD -- (a) the front is local, (b) a transition
exists, (c) the worst layer is 5, inside the 5-8 induction band I
registered. Layer 5 costs +1.112 while the next-worst layer costs
+0.086, a thirteen-fold gap. Seventeen of eighteen attention
layers in this model can be restricted to a four-token window at
a cost under a tenth of a nat.
I am NOT claiming the convergence with the induction band yet,
because there is a confound I can see in my own result. Head 5.7
lives in layer 5 and is an attention SINK that reads POSITION 0
for 99.8% of queries (432) -- and position 0 is precisely what a
sliding 4-token window cuts off. Layer 5's cost may therefore be
the sink losing its constant rather than any long-range content
read, in which case the tidy story ("the depth profile lands on
the induction band") would be a coincidence and must be retracted
before anyone repeats it.
layer5_window_source queued to settle it with three arms: the same
window but with position 0 always allowed; the window applied ONLY
to head 5.7; and the window applied to every layer-5 head EXCEPT
5.7. Registered: (a) allowing position 0 drops the cost to <= 0.15,
(b) head 5.7 alone reproduces at least half of the +1.112, (c) the
other eight heads are local at <= 0.15. If all three hold, 482's
real headline is "this model's only non-local read is a constant
fetch", which is a stranger and more useful statement than the one
I would have published.

## 483. THE CONFOUND WAS REAL: the model's only non-local read is a constant fetch

layer5_window_source, all three bars HELD:
  4-token window on layer 5 (reference)     +1.1121
  the same window PLUS position 0 allowed   +0.0766   (-93%)
  window applied ONLY to head 5.7           +0.5967
  window applied to the other eight heads   +0.0515
So layer 5's apparent long-range dependence is the SINK. Let
position 0 through and layer 5 becomes as local as everything
else; window head 5.7 alone and you recover half the damage;
window the other eight and it costs a twentieth of a nat.
CORRECTED HEADLINE, and it is better than the one I nearly
published: on average text, EVERY GENUINE CONTENT READ IN THIS
EIGHTEEN-LAYER MODEL FITS INSIDE FOUR TOKENS, plus a single
constant fetched from position 0. Seventeen layers under 0.09
nats, the eighteenth explained by the constant.
RETRACTION OF AN INTERPRETATION, not of a number: 482's bar (c)
predicted the worst layer would fall in 5-8, the induction band,
and it did -- but for the wrong reason. A prediction that holds
for the wrong reason is not a confirmation, and the "depth profile
converges with the induction circuit" reading is withdrawn.
One number worth keeping for the composition ledger: the arms do
not add. Windowing the sink alone costs 0.597 and the other eight
0.052, summing to 0.649 against 1.112 measured together --
superadditive by 0.46, the same non-additivity 447 found for the
bias's value.
Report updated and republished with the four-token result.
That leaves a genuine puzzle, and it is the right next
experiment. Induction REQUIRES a distant read, and this program
closed that circuit end to end across layers 1-8 -- yet windowing
those layers costs 0.014 to 0.038 on average. The reconciliation
is presumably that induction barely moves AVERAGE loss.
window_at_match queued to check exactly that: rerun the whole
sweep scored at MATCH positions. Registered: (a) some band layer
costs >= 0.20 there, (b) the worst is a documented induction
layer, (c) unlike the average case, allowing position 0 does NOT
rescue layer 5 at match positions.

## 484. At match positions too, layer 5 is the sink -- and layer 12 is the real outlier

window_at_match reran the depth sweep scored separately at match
and non-match positions:
  layer   match      non-match        layer   match      non-match
    0    -0.1388     +0.0092            9    -0.0246     +0.0102
    1    -0.0586     +0.0517           10    -0.0212     -0.0159
    2    -0.0533     +0.0600           11    +0.0300     +0.0176
    3    +0.0082     +0.0548           12    +0.2094     +0.0207
    4    -0.2299     +0.0257           13    +0.0450     +0.0177
    5    +0.9964     +1.1727           14-17  all under +0.04
    6    -0.0176     +0.0370
    7    -0.0403     +0.0307
    8    -0.0163     +0.0662
  layer 5 at match, with position 0 allowed: +0.0703
(c) FAILED: allowing position 0 rescues layer 5 at MATCH positions
too (+0.996 -> +0.070). Bars (a) and (b) technically HELD, but
only through layer 5 -- the same wrong-reason pass as 482(c), and
I am scoring them as uninformative rather than banking them twice.
Two real findings in the table. First, LAYER 12 is the only layer
with a genuinely match-specific long-range cost: +0.209 at match
against +0.021 elsewhere, a tenfold ratio, and it is nowhere near
the induction band. Second, windowing several EARLY layers HELPS
at match positions -- layer 4 at -0.230, layer 0 at -0.139 --
so restricting attention to four tokens improves prediction at
repeat positions.
That sharpens a tension with this program's flagship result rather
than resolving it. Deleting the nine induction-band heads costs
+0.601 at match positions (376), but restricting their layers to
four tokens costs between -0.06 and +0.01 at those same positions.
Induction is DEFINED by reading a distant earlier occurrence. If
those heads cannot see past four tokens and nothing happens, then
either their value at match positions is carried by LOCAL reads,
or the deletion cost measures something other than the match read.
induction_window queued to test the heads themselves rather than
their layers: window the nine band heads to four tokens, delete
them as a sanity check against 376's 0.601, and window nine random
non-band heads as control -- all scored at match positions.
Registered as a decisive fork: >= 0.30 means the distant reads
carry the function, <= 0.10 means induction's value at match
positions is LOCAL, and either outcome is a substantive finding
about the program's most-cited claim.

## 485. THE FORK RESOLVES FOR THE FLAGSHIP: induction's distant reads carry it

induction_window tested the nine band heads directly rather than
their layers, scored at match positions:
  window the nine band heads to 4 tokens   match +0.3182
  delete the nine outright                 match +0.4258
  window nine random non-band heads        match +0.1798
ALL THREE BARS HELD, and the registered fork resolves in the
direction that supports this program's most-cited result:
windowing the band recovers 75% of the damage that DELETING it
causes (0.318 against 0.426), and costs 1.8x what windowing nine
arbitrary heads costs. Induction's value at match positions is
carried by its DISTANT reads, exactly as the circuit work
(376-408) says. The tension raised in 484 is resolved in favour of
the claim, not against it.
Honest calibration alongside it: nine random heads windowed
already cost +0.180 at match, so part of the band's 0.318 is the
generic price of windowing nine heads. The band-specific excess is
0.138, and the cleanest single statement is the within-heads one
-- window versus delete for the SAME nine heads, 75%.
What that leaves is a genuine structural question rather than a
worry. Windowing whole LAYERS containing band heads cost between
-0.06 and +0.01 at match (484), yet windowing the band's heads
together costs +0.318. Each of those layers holds one band head
and eight others, so the natural reading is REDUNDANCY ACROSS THE
BAND: one head's distant reads going missing is covered by the
rest, and only removing them together bites.
induction_redundancy queued to measure that curve directly --
window the band heads cumulatively, one at a time, against the
same curve over nine random heads. Registered: (a) all nine cost
>= 3x the mean of the singles (superlinear), (b) at least seven of
nine singles cost <= 0.05 alone, (c) the control curve is closer
to linear.

## 486. THE INDUCTION BAND IS MUTUALLY COVERING: 108x superlinear

induction_redundancy windowed the band heads cumulatively:
  singles (each head alone, at match positions)
    -0.0123  +0.0231  -0.0393  +0.0349  +0.0088
    +0.0262  -0.0051  -0.0218  +0.0119     mean +0.0029
  cumulative
    -0.012, +0.035, +0.012, +0.085, +0.141,
    +0.262, +0.299, +0.327, +0.318
  ratio all-nine / singles-mean:  108.5
  same ratio for nine random control heads:  10.0
(a) HELD enormously and (b) HELD: every single band head, windowed
alone, costs essentially nothing at match positions -- the mean is
three thousandths of a nat and FOUR of the nine are slightly
HELPFUL -- while all nine together cost 0.318.
(c) FAILED, and the failure is worth keeping: the control curve is
ALSO superlinear, at 10x. Multi-head ablation is superadditive in
this model generally, so the band's 108x is not a category
difference from arbitrary heads but an order-of-magnitude one. I
registered "closer to linear" and the honest statement is "eleven
times less superlinear".
THE FINDING: the induction band is a MUTUALLY COVERING circuit. No
individual head's distant reads are necessary; each one's loss is
absorbed by the other eight. Only the collective loss bites, and
then by 75% of what full deletion costs (485). This qualifies the
program's flagship result in a way worth publishing rather than
burying: "four reads of an identity code" describes what the band
computes COLLECTIVELY, not any head's private necessity, and it
explains cleanly why windowing whole layers looked free (484) --
each layer holds one band head, and one is always covered.
Report updated and republished with the qualification.
layer12_match queued on the one long-range signal still
unexplained: layer 12 costs +0.209 at match against +0.021
elsewhere, has nothing to do with the sink, and sits outside the
band. Registered: (a) one head carries >= 50% of it, (b) that head
reads the SAME TOKEN as the query at >= 30% of match positions
against a frequency-matched null under 5% -- which would make it
an induction-like head the band list missed -- and (c) the median
layer-12 head carries under 0.05.

## 487. Layer 12's long-range work is ONE head -- and my probe for it was wrong

layer12_match windowed each layer-12 head individually at match
positions:
  12.6  match +0.1770   non-match +0.0147   <- 84.5% of the layer
  12.2  match +0.0133   12.4 +0.0067   12.3 +0.0036
  the other five all under +0.004
  layer total at match +0.2094 | median head +0.0032
(a) HELD decisively: a single head carries 84.5% of layer 12's
match-specific cost, and its own match/non-match ratio is
twelvefold. (c) HELD.
(b) FAILED -- and the failure is a bad probe, not a result. I
asked whether 12.6 reads the SAME TOKEN as the query (6.0%
against a 3.3% null) but that is not what an induction-style head
does. Such a head reads the token that FOLLOWED the repeat last
time, position p+1, not the repeat itself; this program's own head
census even names the motif "induction-target". I tested for the
wrong object and would have recorded 12.6 as "not induction-like"
on the strength of it.
So the honest state: layer 12 contains one head doing genuine
long-range work specifically at repeat positions, its function is
unaccounted for by any circuit this program has closed, and its
read semantics are still unmeasured because my first attempt
measured the wrong thing.
head_12_6_reads queued with the corrected classification: at match
positions, is 12.6's top read the SUCCESSOR of a previous
occurrence, the occurrence itself, local, position 0, or other --
each against a frequency-matched null, with a median layer-12 head
as control. Registered: (a) successor-reading >= 25% against a
null under 5% makes it an induction-target head the band list
missed, (b) the histogram is reported either way, (c) the control
head stays under 10%.

## 488. Head 12.6 is a DIFFUSE LONG-RANGE reader -- not induction, not local

head_12_6_reads, with the corrected probe:
  head 12.6 at match positions (n=1408)
    successor of a repeat (induction target)  9.3%  (null 1.1%)
    the repeat itself                         3.3%
    local, offset >= -4                      15.5%
    position 0                                0.5%
    other (distant, scattered)               71.4%
    top six offsets -1,-2,-4,-3,-6,-9 cover only 22% of reads
  control head 12.3
    successor 3.9% | local 43.5% | other 50.8%
(a) FAILED: 9.3% successor-reading is 8.5x its null and clearly
non-random, but nowhere near the 25% that would make 12.6 an
induction-target head the band list missed. (b) HELD (histogram
reported), (c) HELD (the control is not a successor reader).
So the corrected probe rules out the identity I expected, and the
head that carries 84.5% of the model's only match-specific
long-range cost is something this program has not catalogued: a
DIFFUSE long-range reader. It is markedly less local than its
layer-mate (15.5% against 43.5%), spreads its reads thinly over
distant positions rather than concentrating on any offset, and
damages the model specifically at repeat positions.
head_12_6_targets queued to characterise it by WHAT it reads
rather than where: at match positions, take the token at its top
read and compare the class distribution against the corpus base
rate, with a rarity split and head 12.3 as control. Registered:
(a) some class is enriched >= 2x, (b) RARE tokens are the most
enriched -- the salience hypothesis, a long-range head that
ignores position and seeks informative tokens -- and (c) 12.6 is
more selective than the control.

## 489. Head 12.6 reads STRUCTURE, not salience -- and its layer-mate is its mirror image

head_12_6_targets classified the token at 12.6's top read at match
positions, against corpus base rates:
  class         12.6 enrichment    12.3 (control)
  punctuation       2.33x              0.34x
  capitalised       1.79x              3.72x
  digit             1.33x              0.33x
  newline           1.27x              0.09x
  space-word        0.68x              1.27x
  subword           0.38x              1.40x
  RARE tokens       0.69x              1.01x
(a) HELD: punctuation is enriched 2.33x. (b) FAILED and the
failure kills my hypothesis cleanly -- rare tokens are DEPLETED at
0.69x, so 12.6 is not a salience detector seeking informative
tokens; it seeks COMMON structural ones. (c) FAILED: the control
head is more selective on its own top class (3.72x) than 12.6 is
on its (2.33x), so "selectivity" does not distinguish them.
What distinguishes them is WHICH classes, and the two heads are
near mirror images. 12.6 reads punctuation, capitals, digits and
newlines -- the anchors of layout and clause boundaries -- and
avoids prose content (subword 0.38x). 12.3 does the opposite:
capitals and word-content, almost never punctuation (0.34x) or
newlines (0.09x). One layer holds a long-range STRUCTURE reader
and a local CONTENT reader side by side.
That is a characterisation, not yet a function, so
head_12_6_structure is queued to test it where it should bite: if
12.6 tracks layout, its contribution should scale with how
structured the text is. Split fresh FineWeb rows into quartiles by
punctuation-and-newline density and measure 12.6's window damage
at match positions in each, with 12.3 as control. Registered: (a)
a >= 2x gradient from bottom to top quartile for 12.6, (b) no
gradient for 12.3, (c) per-quartile numbers reported either way.
