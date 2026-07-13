# Per-Datapoint Mechanism Decomposition

**Purpose:** a general, unsupervised method for decomposing a model's weights into
interpretable, ablatable rank-1 *mechanisms*, discovered SAE-style from data, but
decomposing the *map* (W, or a Jacobian) rather than the activations. Components
should (a) be found without supervision, (b) be excisable from the weights with
predictable behavioral effect, and (c) compose across layers by tensor contraction,
giving circuit discovery for free.

This file is a self-contained spec: theory summary, tiered experiment plan
(least → most ambitious), algorithm details, metrics, and explicit falsification
tests. **Tier 0 exists to catch either coding bugs or conceptual confusion before
anything expensive runs. Do it first and do not skip tests.**

---

## 1. Theory summary (all the context you need)

### 1.1 Closed-form low-rank data-conditioned solution

For a linear layer W ∈ R^{d_out × d_in} and data matrix X ∈ R^{d_in × N} with
rank(X) = d_in (assumption A1), the rank-r problem

    min_{A,B} || A B X − Y ||_F^2,   A ∈ R^{d_out × r}, B ∈ R^{r × d_in}

has global minimizer M* = [Y X† X]_r X†, where [·]_r is SVD truncation and
X† = Xᵀ(XXᵀ)^{-1}. With Y = WX (Problem 1, "functional core"), this reduces to
M* = [WX]_r X†. With Y = W X̃ where X̃ replaces target datapoints with
counterfactuals (Problem 2, "edit"), same formula. Optimal value =
Σ_{j>r} σ_j²(YX†X) + ||Y(I − X†X)||_F². Proof: project onto row space of X
(Pythagoras), isometric change of variables G = M U_X Σ_X, Eckart–Young–Mirsky.
This theorem is the exact M-step of the training loop in §3, and the
verification/surgery tool in every tier.

### 1.2 Per-datapoint solutions and why naive clustering fails

For a single datapoint, the min-norm rank-1 solution to Mx = Wx is

    M_x = (W x) xᵀ / ||x||².

These are the objects one might cluster by cosine similarity. **Key negative
result (verify numerically in Tier 0):** with orthonormal features {e_j},
outputs u_j = W e_j, and x = Σ_{j∈S} e_j with |S| active features,

    M_x = (1/|S|) Σ_{j,k∈S} u_j e_kᵀ.

Only the |S| diagonal terms correspond to anything W does; the fraction of
Frobenius mass on the diagonal is 1/|S|, and cos-sim between M_x and the true
mechanism W P_S = Σ_{j∈S} u_j e_jᵀ is ≈ 1/√|S|. At 20 active features, 95% of
each vector is instance-specific junk. Min-norm cannot distinguish "W restricted
to the active subspace" from "the one map sending this x to the right place."
Cross terms are artifacts *for linear W*; they become real signal only for
Jacobians of nonlinear layers (Tier 3).

Note also: cos-sim between two flattened rank-1 matrices factorizes as
(input-side cos) × (output-side cos), so e.g. a Dog+Sunglasses point sits at
~0.5 × 0.5 from the Dog atom even in the friendly orthogonal case. Two reasons
naive clustering fails; the mass-fraction one is the killer.

### 1.3 The fix: masked-projector objective (the actual method)

At the mechanism level, linear superposition holds: W P_S = Σ_{j∈S} W e_j e_jᵀ
is additive over features. So parameterize components as rank-1 mechanisms
pinned to a learned dictionary D = [d_1, …, d_m] and train

    L(D, C) = Σ_i || (Σ_j c_ij · W d_j d_jᵀ) x_i − W x_i ||² + λ Σ_i ||c_i||_1,

with c_i ≥ 0 optional (try both). This is fully unsupervised: shared dictionary +
sparsity = same discovery mechanism as an SAE, one level up. The support
{i : c_ij ≠ 0} of each atom is the "cluster," an output not an input. The
per-datapoint M_i's from §1.2 never need to be constructed.

Comparison points:
- vs SAE: an SAE can learn features W treats as noise; this objective has no
  incentive to represent directions whose image under W is negligible. Feature
  discovery weighted by mechanistic relevance to this layer.
- vs APD/SPD: components are pinned to the parameterization W d_j d_jᵀ rather
  than free, which removes the "invent components that mimic the circuit without
  doing its computation" cheating channel that forced SPD's stochastic masking.
  No stochastic masking needed here (verify this claim empirically in Tier 1:
  look for cheating-like solutions and confirm absence).
- Ablatability: subtract W d_j d_jᵀ from W (equivalently project out d_j on the
  input side) and behavior on feature-j data should be removed, elsewhere
  preserved. This is the property APD buys with global faithfulness; here it
  comes from the parameterization.

### 1.4 Identifiability and the correlated-features stance (design decision)

Two features are separable iff they either (a) occur separately somewhere in
the data, or (b) are treated differently by W (the mechanism side can rescue
identifiability when co-occurrence statistics cannot; this is a genuine
advantage over activation SAEs and should be demonstrated in Tier 1).

**Accepted limitation, by design:** features that always co-occur AND are
treated isotropically by W will merge into one atom. This is fine. The desired
behavior is that when such features later *decorrelate* (test-time distribution
shift), reconstruction error on those datapoints spikes, giving a built-in
novelty signal: "these specific features were not seen (separately) before."
Tier 1 must include an explicit test of this: train on perfectly correlated
pair, evaluate reconstruction on decorrelated points, confirm the error spike
is large and localized (i.e. usable as a detector, not just nonzero).

### 1.5 Tensor network structure: why this is cheap and exactly solvable

The loss is a closed multilinear tensor network. Consequences to exploit:

- **Contraction order:** never materialize any d_out × d_in matrix per point.
  Compute a_ij = d_jᵀ x_i (cost m·d_in), then Σ_j c_ij a_ij (W d_j) with cached
  WD (cost m·d_out). O(m(d_in + d_out)) per point.
- **Hadamard Gram trick for the sparse-coding step:** the per-point design
  matrix has columns a_ij · (W d_j), so the Gram factorizes:

      G_i = (a_i a_iᵀ) ⊙ G^{(W)},   G^{(W)}_{jk} = (W d_j)ᵀ (W d_k),

  with G^{(W)} precomputed once per dictionary update. Per-point lasso setup
  drops from O(m² d_out) to O(m d_in + m²). Implement this; do not use a
  generic autodiff lasso per point.
- **Multilinearity ⇒ exact block updates:** codes fixed → dictionary update is
  least squares / the §1.1 machinery; dictionary fixed → per-point lasso.
  Alternating minimization with closed-form inner solves (§3).
- **Composition:** components have open legs. Across layers,
  (W² d_k² d_k²ᵀ)(W¹ d_j¹ d_j¹ᵀ) = (d_k²ᵀ W¹ d_j¹) · W² d_k² d_j¹ᵀ, so circuit
  strength between layer-1 atom j and layer-2 atom k is the scalar
  d_k²ᵀ W¹ d_j¹. Circuit discovery = reading a matrix (Tier 3).

Known TN disease to guard against: CP-style degeneracy (components growing while
nearly canceling). Constrain ||d_j|| = 1 (absorb scale into codes) and monitor
Σ_j |c_ij| ||W d_j|| vs ||W x_i||; add a norm penalty or ODT-style
canonicalization if degeneracy appears.

---

## 2. Tiered experiment plan

Each tier has explicit pass/fail criteria. A tier failing its falsification test
means stop and diagnose (bug vs conceptual error) before proceeding.

### Tier 0: Ground-truth orthonormal toy (least ambitious; DO FIRST)

Setup: d_in = d_out = 64. m_true = 10 orthonormal feature directions {e_j}
(random orthonormal basis subset). W: random Gaussian (also try a structured W,
e.g. distinct singular values per feature, for the identifiability tests later).
Data: N = 10k points, x_i = Σ_{j∈S_i} α_ij e_j, |S_i| ~ Uniform{1..5},
α_ij ~ Uniform[0.5, 1.5], plus small isotropic noise (σ = 0.01; note noise is
also what makes rank(X) = d_in hold, per A1).

Tests, in order:

1. **Theorem verification.** Implement M* = [YX†X]_r X† for Problems 1 and 2.
   Check (i) achieved loss matches the closed-form optimal value expression to
   numerical precision, (ii) no rank-r matrix found by direct optimization
   (Adam on A,B from many inits) beats it, (iii) Problem 1 with r = m_true
   recovers W P_S exactly on data spanning the feature subspace.
2. **Negative result verification.** Compute per-point M_x; confirm diagonal
   mass fraction ≈ 1/|S| and cos(M_x, W P_S) ≈ 1/√|S| as functions of |S|.
   Run the naive cos-sim clustering, show it degrades with |S| as predicted,
   and show the Dog+Sunglasses-style compositional point fails to join either
   parent cluster. (This is the motivation section's figure.)
3. **Recovery.** Train the masked-projector objective with m = 40 atoms (4×
   overcomplete). Pass: max cos-sim of each true e_j to some learned d_j > 0.99
   after sign fixing; support F1 vs true S_i > 0.95; mean-matched-cos (MCC-style)
   reported. Sweep λ; report sensitivity.
4. **Ablation.** For each recovered atom, edit W ← W − W d_j d_jᵀ. Pass:
   ||W' x − W x|| large exactly on datapoints with feature j active, ≈ 0
   elsewhere (report the two distributions).

**Falsification:** if test 3 or 4 fails on this maximally friendly setup after
reasonable hyperparameter search, the method or its implementation is wrong.
Stop and diagnose.

### Tier 1: Adversarial toys (identifiability, correlation, superposition)

Same infrastructure, harder data-generating processes. Each is a targeted test
of a specific theoretical claim from §1.

1. **Correlated pair, isotropic W (the accepted-merge case).** Two features
   with co-occurrence probability 1.0 during training; W treats them with equal
   singular values. Expect: one merged atom spanning both. Then evaluate on
   decorrelated test data: reconstruction error must spike on those points,
   cleanly separable from in-distribution error (report ROC/AUC of the error as
   a novelty detector). This validates the design stance in §1.4.
2. **Correlated pair, anisotropic W (the rescue case).** Same co-occurrence,
   but W acts with very different gain/output direction on the two features.
   Claim to test: mechanism-side information separates them even though
   co-occurrence statistics cannot. Pass: two distinct atoms recovered. Compare
   against a vanilla SAE on activations, which should merge them. This is the
   headline "advantage over SAEs" result if it works.
3. **Superposition.** m_true = 100 > d_in = 64, features random unit vectors,
   sparse activation. Measure recovery quality vs feature count, interference,
   and how it degrades relative to a matched SAE on x and a matched SAE on Wx.
4. **Cheating check.** Search for pathological solutions: does the optimizer
   ever produce atoms whose supports match the data but whose d_j are not
   feature-aligned (the APD failure mode)? Quantify over many seeds. Expected:
   absent, because components are pinned; verify.
5. **Hierarchy (optional).** Nested features (animal → dog → breed). Matryoshka
   variant: prefix-ordered code groups must reconstruct alone. Expect coarse
   atom W a aᵀ before refinements.
6. **Rank-k mechanisms (optional).** Replace d_j with D_j ∈ R^{d_in × k}
   (feature manifolds / multidimensional features). Component = W D_j D_jᵀ
   (D_j orthonormal). Only pursue if a toy with genuine k-dim feature subspaces
   breaks the rank-1 version.

### Tier 2: Real model, single linear layer

Target: Gemma-2-2B, one linear map. Default: MLP down-projection at a middle
layer (alternatives: attention OV per head, which is naturally low-rank).
Data: residual/hidden activations from The Pile, N = 100k–1M samples (start
100k), stored in fp16.

- Scale: d ~ 2304–9216, m = 8k–16k atoms. The §1.5 contraction tricks are
  mandatory at this scale. Minibatch the alternation (codes per batch;
  dictionary update via running sufficient statistics or periodic full M-step).
- Evaluation: (i) reconstruction R² of Wx and downstream CE-loss delta when
  splicing the reconstruction into the forward pass; (ii) L0 of codes;
  (iii) comparison with Gemmascope SAE features at the same site: max cos-sim
  matrices in both directions, and qualitative auto-interp of top-activating
  datapoints per atom; (iv) ablation: remove single atoms from W, measure loss
  delta concentrated on the atom's top-activating contexts vs random contexts;
  (v) the SAE-disagreement analysis: atoms with no SAE match and SAE features
  with no atom match are the interesting objects (predicted: SAE features that
  W treats as noise appear only on the SAE side).
- Pass criteria are softer here; the goal is "competitive reconstruction with
  interpretable, ablatable atoms and an interesting disagreement story," not
  beating SAEs on their own metric.

### Tier 3: Nonlinear layers, Jacobians, circuits (most ambitious)

1. **Jacobian version.** For a nonlinear layer f, per-point object is J_i =
   Df(x_i). Cross terms are now real (gating/curvature). Decoder becomes
   bilinear: code enters twice, Ĵ_i = Σ_{jk} c_ij c_ik (or a learned core
   T_jk) · u_j v_kᵀ with u, v learned or tied through the layer. Start with a
   toy nonlinear layer with known gating structure. Connection: compare
   recovered Jacobian subspaces with J-space from the J-lens paper (overlap
   with the existing steering-vector/J-space experiment).
2. **Cross-layer circuits.** Decompose two consecutive layers; compute the
   contraction-coefficient matrix d_k^{(2)ᵀ} W^{(1)} d_j^{(1)} (for the linear
   case; Jacobian analogue for nonlinear). Validate on a task with known
   circuits (induction heads, or an algorithmic task with ground-truth circuit,
   e.g. modular arithmetic checkpoints already on hand).
3. **Whole-model ambition.** Every layer decomposed, circuits read off
   contraction matrices, edits performed by the §1.1 Problem-2 machinery on
   discovered supports. This is the "general method for decomposing a model"
   endpoint; do not start here.

---

## 3. Algorithm (reference implementation spec)

Alternating minimization:

- **E-step (codes):** for each datapoint (batched), solve nonneg lasso /
  lasso with Gram G_i = (a_i a_iᵀ) ⊙ G^(W), linear term b_i = (a_i) ⊙
  ((WD)ᵀ W x_i). Coordinate descent or FISTA; warm-start from previous codes.
- **M-step (dictionary):** with codes fixed, update D. Options, in order of
  preference: (a) exact per-atom update holding others fixed (residual fit,
  closed form via the §1.1 machinery restricted to the atom's support);
  (b) joint gradient step with unit-norm projection. Renormalize ||d_j|| = 1,
  absorb scale into codes, after every update.
- **Init:** (a) random unit vectors; (b) top right-singular vectors of WX;
  (c) run k-means on activations, then the supervised closed-form extractor
  per cluster, then refine. Compare all three in Tier 0; (c) predicted best.
- **Dead atom handling:** reinit atoms with support size 0 to
  high-residual datapoints' input directions (standard SAE resampling).
- **Degeneracy monitor:** track max_j ||c_·j||·||W d_j|| / ||W x||; alarm on
  growth (CP degeneracy, §1.5).
- **Hyperparameters:** λ swept log-scale; m overcomplete 4× (toys) / 8–16k
  (Tier 2); report L0–R² frontier rather than single points.

Repo shape suggestion:

    mechdecomp/
      closed_form.py      # §1.1 theorem, Problems 1 & 2, unit-tested vs SVD
      perpoint.py         # M_x construction + mass-fraction analysis (Tier 0.2)
      objective.py        # masked-projector loss, contraction-ordered
      estep.py            # Hadamard-Gram lasso
      mstep.py            # dictionary updates (exact + gradient variants)
      toys.py             # DGPs for Tiers 0–1 (orthonormal, correlated,
                          #   anisotropic-W, superposition, hierarchy)
      metrics.py          # MCC/max-cos, support F1, ablation deltas,
                          #   novelty-detector ROC
      tier0.py … tier3/   # experiment scripts, one per test above
    tests/                # theorem verification is a unit test, not a script

---

## 4. Metrics glossary

- **Reconstruction:** R² of Ŵx vs Wx per point; CE-loss delta when spliced
  (Tier 2).
- **Recovery (toys):** max cos-sim per true feature (after sign fix), mean
  matched cosine, permutation-matched; support F1 per feature.
- **Sparsity:** L0, L1 of codes; dead-atom count.
- **Ablation faithfulness:** distribution of ||ΔWx|| on atom-active vs
  atom-inactive points; overlap statistic (e.g. AUROC).
- **Novelty detection:** AUROC of per-point reconstruction error separating
  decorrelated-OOD from in-distribution (Tier 1.1).
- **Circuit (Tier 3):** correlation of contraction-coefficient matrix with
  ground-truth circuit adjacency.

---

## 5. Open decisions (defaults chosen; flag if changing)

1. **Rank-1 vs rank-k components.** Default rank-1; escalate per Tier 1.6 only
   on demonstrated need.
2. **Nonnegative codes?** Default yes (mechanisms fire or don't); ablate.
3. **Global weight faithfulness.** Deliberately NOT imposing Σ_j W d_j d_jᵀ ≈ W.
   Atoms may overlap/double-count in W; ablatability is the operative
   faithfulness notion here. Optionally *report* || Σ_j W d_j d_jᵀ P_data − W
   P_data ||_F as a diagnostic, never as a constraint (constraining it re-opens
   APD-style problems).
4. **Bias:** fold into W via homogeneous coordinate, or ignore (toys have none).
   Default: ignore in toys, homogeneous coordinate in Tier 2.
5. **Tier 2 site:** Gemma-2-2B mid-layer MLP down-proj, Pile data, to line up
   with existing Gemmascope comparison infrastructure. Change if a different
   site has better SAE baselines on hand.
