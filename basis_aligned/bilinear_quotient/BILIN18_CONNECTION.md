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
not the top. There is no low-dimensional causal core to find at layer 1; the
direction-level program's <10% coverage was not a sampling failure but the geometry.

**(b) failed, inverted: PCA dominates G_lam for cumulative coverage at every k on both
layers** — the exact opposite of the leader-identity results (§23/§30). The Λ-Gram is
the better *leader* predictor and the worse *coverage* ordering; variance ordering wins
for bulk. Two instruments, two jobs, and my assumption that the first generalised to
the second is now a recorded failed prediction. Whatever ordering is causally optimal
is neither — the mid-spectrum peak says both orderings misplace the heavy directions.
