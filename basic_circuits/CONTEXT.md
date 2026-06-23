# Bilinear superposition + tensor-network decomposition: session context

Working notes for continuing this in Claude Code. Everything is pure numpy
(manual Adam, manual backprop) so it runs anywhere without torch. Inputs are
sparse boolean unless noted. Index 0 is reserved as a constant coordinate
(x0 = 1) wherever a "folded residual" is used.

## The through-line

Question we started from: does *weight* superposition happen in bilinear
layers, and if so what is it actually storing? Answer that emerged: linear
readouts (ridge / FVU) say "no superposition," but the computation is real and
sits below a thresholded (sigmoid) readout. So FVU was measuring the linear
interface, not the computation. From there the work turned into: given a
bilinear (or stacked-bilinear) net, find the *right decomposition* of each
output's polynomial form, in feature space, that separates signal / inhibition
/ interference into interpretable pieces, ideally without ever instantiating
the full high-order tensor.

## Files, in dependency order

### Single bilinear layer, Universal-AND
- `train_uand.py` — trains a 1-layer biasless bilinear net
  h = (W1 x) ⊙ (W2 x) on the Universal-AND task. m=32 boolean features, 3-hot
  inputs embedded in d0=16 dims (so inputs are already in superposition for
  m>16), n_hidden=64, T = C(32,2) = 496 AND targets, sigmoid + BCE head.
  3 seeds. Saves `uand_seed{0,1,2}.npz` (W1, W2, Wo, bo, E).
  Result: FVU(sigmoid) ≈ 0.002, FVU(ridge-on-hidden) ≈ 0.84 vs ceiling
  1 - n/T = 0.87. ~8 gates/neuron, invisible to a linear probe.

- `pullback.py` — loads the saved seeds, recomputes eval FVU, then does the
  *pullback*: each output t is an exact quadratic form in the 32-dim feature
  basis, Qf[t] = E^T Q_t E, built in one einsum
  `Qf = einsum('tk,ki,kj->tij', Wo, W1@E, W2@E)` then symmetrized.
  Splits Qf into signal (the (a,b) cross term), diagonal (acts as a linear
  term because x_i^2 = x_i on booleans), and off-diagonal interference.
  Saves `pullback_seed2.npz` (Qf, sig, diag, off, bo).
  Result: signal ≈ +38; diagonal = structured inhibition (≈ -4 at the
  target's own two indices, ≈ -16 at all others); interference mean 0 std ~11,
  carries most of the squared mass but is tolerated, not cancelled.

- `structure.py` — conditions interference on index overlap with the target,
  decomposes logits by case (positive / hardest negative / easy negative),
  checks per-target eigenspectrum (NOT rank-2 + noise: top-2 eigvecs carry
  ~35% and barely align with span{e_a,e_b}), and cross-target interference
  correlation (~0.41 -> shared structure).

- `factorize.py` — the interference factorization + the figures.
  Builds C = 2*Qf at the off-diagonal pairs (T x T coefficient matrix),
  X = C with the diagonal (signal) zeroed = pure interference.
  Verifies the exact neuron factorization C = Wo @ Mcross (machine precision).
  SVD of X: effective rank ~5.5, top component 41.5% of interference variance,
  removing top-1 drops cross-target correlation 0.41 -> ~0.  The rank-1 mode is
  identified as embedding crosstalk: its pair-side vector correlates -0.91 with
  the embedding gram overlaps G[i,j] = (E^T E)[i,j], and its target-side vector
  is ~constant. So dominant interference is INHERITED from the non-orthogonal
  embedding E, not learned; the learned defense is the diagonal inhibition
  ladder, not cancellation.
  Writes fig1_qform.png ... fig4_factor.png.

  Final single-layer ansatz:
    Qf = signal  ⊕  inhibitory diagonal  ⊕  rank-1 embedding-geometry crosstalk
         ⊕ small residual.

### Two stacked bilinear layers
- `decomp_exact.py` — NO training. Builds the exact symmetric degree-4 tensor
  for a 4-way-AND monomial x_a x_b x_c x_d, matricizes to (ij)(kl), and shows
  that the ODT / orthogonal (eigh) canonical form is forced to return
  (e_ab ± e_cd)/sqrt2 style *mixtures of complementary pairings* with equal
  eigenvalue magnitudes (so SVD can't even rank/select among the 3 pairings).
  Contrast: any single pairing is an exact 1-term product decomposition
  (3 equivalent interpretable choices). Then the mixed-degree case: a degree-2
  target x_a x_b lifted to x_a x_b x0 x0 entangles the genuine detector with
  const-routed edges e_{0a}, e_{0b}, e_{00}.
  This is the core "orthogonal isn't the right structure" demonstration.

- `train2layer.py` — trains a 2-layer bilinear net on 4-way AND (m=8 features +
  const coord, 5-hot, T=70). Builds the factored representation
  {Acheck_p, Bcheck_p, Wo} where Acheck_p = sum_k W2a[p,k] Q1f[k] etc.
  (Q1f = layer-1 forms in input space). Verifies the factored quartic
  reproduces the forward pass to ~6e-14 (this is the "instantiate the
  5th-order tensor only as a sanity check" step — the factored form is the
  scalable object, size 2*n_h2*m^2, never m^4). Then runs eigh on a per-output
  quartic and shows the same pairing-mix pathology in a *learned* model.

- `mixed.py` — 2-layer bilinear with the **folded residual** (const coordinate,
  h0 pinned to 1 so layer 2 can route lower-degree terms through the const
  slot; mathematically identical to a skip connection). Mixed targets: 28
  degree-2 ANDs + 28 degree-4 ANDs. Trains to 100% TPR/TNR on both strata.
  Folds to x-space (const slot -> Q1f[0] = e0 e0^T), verifies fold to ~3.6e-14,
  then decomposes one target of each degree. Finding: degree-2 targets entangle
  the real detector with const-routed edges (const mass > real mass), AND the
  optimizer routes opportunistically through the const even for degree-4
  targets that didn't need it, so mixed-degree is strictly harder for
  orthogonal canonicalization than either pure case.

### One layer + residual (isolating the residual stream)
- `residual1.py` — shows that on BOOLEAN inputs a 1-layer residual bilinear
  CANNOT isolate the residual, because the skip weight s_t[a] and the bilinear
  diagonal Qf[t,a,a] are gauge-equivalent: effective linear coef = s + diag,
  and you can move delta between them (boolean output unchanged, verified to
  1e-9; continuous output changes by 2.8). Trained boolean nets put only ~32%
  of each linear target in the skip and ~68% in the self-square diagonal.
  Continuous (Gaussian) inputs break the degeneracy: skip fraction -> 1.00,
  zero cross-seed variance, because x_a^2 != x_a so the self-square is the wrong
  basis function for a linear target.

### Boolean square-free canonicalization (executes open threads #1 + #2)
- `hollow.py` — NO new training for Part A; retrains the mixed 2-layer net (~3s)
  for Part B. Implements the boolean x_i^2 = x_i collapse as a *canonical form*
  at both depths, and tests whether it rescues the signal. It does not — both
  fixes are exact and necessary, but interference, not the gauge artifacts they
  remove, is the dominant obstruction at both depths.
  - Part A (single layer): hollows the seed-2 Qf -> H (diag zeroed) + explicit
    linear vector lin = diag(Qf). This is the unique residual1.py gauge fix made
    canonical. Verified boolean-identical (1e-13), continuous output changes
    (max diff ~1539, gauge is boolean-only). lin reproduces the inhibition
    ladder (-4.06 at the target's own 2 indices, -15.99 elsewhere). Re-running
    structure.py's per-target eigenspectrum on H: top-2 alignment with
    span{e_a,e_b} barely moves (0.276 -> 0.345; 2nd 0.029 -> 0.034) and the
    top-2 |eig| share DROPS 0.350 -> 0.231. Conclusion: the diagonal was not
    what buried the signal edge; off-diagonal embedding crosstalk (factorize.py)
    still dominates.
  - Part B (two layers): square-free reduction of each per-output quartic,
    grouping the m^4 monomials by their distinct-index SET, then folding the
    const coordinate (x0=1 drops index 0) so const-routed monomials land in
    their real lower-degree stratum (thread #2, const as a distinguished mode).
    The folded multilinear polynomial reproduces the forward pass to 5.7e-14
    and is the scalable object (size = #subsets up to degree 4, never m^4).
    ~0.35-0.39 of the raw quartic mass touches the const index and is what gets
    relocated. After folding, the genuine AND coefficient is the largest in its
    own degree-2 stratum but only 1.12x the runner-up, and for degree-4 it is
    badly dominated (23 same-stratum competitors larger; genuine/competitor
    0.26x). So 'const mass > real mass' (mixed.py) was a representation artifact
    now removed, but the genuine detector still does not dominate its stratum:
    the degree-4 analog of Part A's interference. Stratification is necessary,
    not sufficient.

### Interference / bias / sparsity follow-ups (write-ups in results/*.md)
- `factorize.py` — now also writes results/ figures fig3b (ablation: drop
  interference vs drop inhibition), fig3c (interference rebuilt from top-k SVD
  modes; TPR non-monotonic), fig3d (full interference minus the dominant mode).
  See results/factorize.md.
- `couplings.py` — results/couplings.md. (Q1) the "interference helps positives"
  effect is a CALIBRATION artifact: no-interference logits are already separable
  (AUC ~1.0); a single threshold shift 0->-13 recovers TPR 99.9%. (Q2) couplings
  split by target-overlap: shares-1 mean +0.69, shares-0 mean -0.68; the big
  shares-0 ones track the embedding gram (corr -0.62, inherited crosstalk).
  (A) the bias bo is near-constant ~-3.94 (a global operating point).
- `factorized_sparsity.py` — results/factorized_sparsity.md. Iterative magnitude
  pruning (L1 + prune 10%/round, persistent mask, prune only active weights) from
  the dense seed. Two configs side by side: A = prune W1,W2,Wo; B = also prune the
  embedding E + an L1 penalty on the pullback Qf itself. Then an L1-FREE recovery
  fine-tune at the chosen sparsity. All accuracy is the recalibrated metric (Q1).
  Findings: ~47% of weights removable for free after recovery (BCE 0.00007, TPR
  1.0); the pruned-but-not-recovered TPR collapse is the Q1 calibration effect
  again (TNR stays ~1). Qf is ~85% magnitude-compressible even dense; L1-on-Qf
  drives the pullback to 75% near-zero (vs 7%) and extends the compressibility tail
  to ~92-98%, BUT the L1-free recovery re-densifies Qf -> sparse-Qf and recovered-CE
  trade off. Figures: fig_sparsity_curve, fig_sparsity_ladder (recal ladders),
  fig_qf_frontier, fig_sparse_pullback. Writes uand_seed2_sparse.npz.

## Conventions / gotchas
- Column order of the (T x T) coefficient matrix follows
  np.triu_indices(m, 1), which matches itertools.combinations(range(m), 2) and
  pair_idx. C[t,t] is target t's own signal. Assert this if you reimplement.
- "Pullback" = fold the embedding E into the weights FIRST (W1@E, W2@E), then
  einsum over the neuron index. Don't build per-neuron matrices in a loop.
- Boolean idempotence x_i^2 = x_i is load-bearing everywhere: it's why the
  diagonal is an effective linear term, why degree-2 collapses through squares,
  and why the residual is non-identifiable at 1 layer.
- Saved .npz weights: uand_seed{0,1,2}.npz (single layer),
  pullback_seed2.npz (Qf and decomposition pieces for seed 2).

## Open threads / next steps discussed
1. **Hollowing / canonicalization step**: DONE in `hollow.py` Part A. Pushed the
   diagonal into an explicit linear vector (canonical, diag(H)=0, boolean-exact).
   Finding: it isolates the inhibition ladder cleanly but does NOT surface the
   signal edge in the eigenspectrum -- interference still dominates. Remaining:
   apply the same hollowing inside the depth-2 fold before bond canonicalization
   (#4).
2. **Degree-stratified Tucker**: partly DONE in `hollow.py` Part B -- the const
   mode is folded into its own channel via the square-free reduction, giving
   exact homogeneous strata. Finding: stratification is necessary but not
   sufficient; within the degree-4 stratum the genuine detector is dominated by
   cross-target interference. Remaining: an actual Tucker/CP *within* a stratum
   (the square-free reduction only gives the coefficient form, not a factorized
   decomposition of it).
3. **Non-orthogonal / sparse pursuit**: prefer arms with disjoint, const-free
   support to select genuine conjunctions over const-padded shadows. CP-style
   with a tie-break for the 3-fold pairing symmetry.
4. **Bond canonicalization for depth 2**: the n_h1 and n_h2 bond indices are
   gauge freedoms; "finding the right decomposition" = picking the canonical
   gauge. Acheck_p is already invariant to the bond-1 gauge; the live gauge is
   bond-2 GL(n_h2) plus per-p A<->B swap and scaling. Map onto ODT
   canonicalization (Thomas's machinery) but with the caveat from
   decomp_exact.py that orthogonality mixes pairings.
5. Residual factorization higher modes: is the post-rank-1 single-layer
   interference also embedding-explainable at higher SVD modes (higher-order
   structure of G)?

## Reproduce
    python train_uand.py        # ~2 min, writes uand_seed*.npz
    python pullback.py          # writes pullback_seed2.npz
    python structure.py
    python factorize.py         # writes fig*.png
    python decomp_exact.py      # no training, fast
    python train2layer.py       # ~few sec after vectorization
    python mixed.py             # ~5 sec
    python residual1.py         # trains 6 small nets, ~30-60 sec
    python hollow.py            # Part A loads seeds; Part B retrains mixed, ~5 sec
    python couplings.py         # threshold/coupling/bias analysis, writes results/fig5-7
    python factorized_sparsity.py  # iterative magnitude pruning, ~10 min, writes results/
