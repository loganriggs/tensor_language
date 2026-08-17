# Plan: causal verification + red team of the layer-1 story, then the other layers

User directive (2026-08-17): causally verify the register-leader story and the
compression claim (replacement must win on fidelity AND on MDL), red-team it, causally
verify the semantic hypotheses at the text level, then give the other layers the same
depth of treatment, starting at layer 0.

## Phase A — layer 1 leader: verify, replace, red-team
- A1 semantic causal test: ablate ONLY the leader direction; damage must concentrate
  where the hypothesis says (whitespace-heavy contexts), measured per-position with
  contexts binned by layout-token fraction. Includes the reverse prediction: prose
  contexts nearly unharmed.
- A2 causal replacement: swap the leader coefficient c_0 = xhat^T M xhat for the
  compressed surrogate the story implies (a single squared projection of the head-4
  component, then ladder: +rank-2, +full attn1 restriction). Score each rung on
  (i) CE fidelity vs the intact model, (ii) parameter count -> honest MDL ladder.
- A3 red team: (i) matched-size surrogate on a random direction; (ii) surrogate fit on
  document-shuffled targets; (iii) transfer to held-out document rows (the §16
  heterogeneity is the obvious failure mode); (iv) text-level intervention: inject
  layout tokens into prose, verify the leader moves and downstream CE responds as the
  story predicts; remove layout from markup-heavy text, verify the reverse.
- A4 wrap into BILIN18_CONNECTION §19 + commit.

## Phase B — layer 0, same depth
- B1 Shapley (big-data basis from the start; 20 perms) -> concentration verdict.
- B2 writer folding (writers: emb + attn0 only) -> which pairs drive the leaders.
- B3 data structure (spectrum / kurtosis / ICC / hierarchy).
- B4 naming: excitation + emb-curvature vocab naming; unfold attn0 by head if
  attention-driven.
- B5 causal check of its leader, A1-style.
- Commit as §20.

## Phase A' — strengthen the causal claims (user directive, 2026-08-17 second pass)
The register-semantics failure is acceptable; the requirement going forward is
STRONGER, testable causal claims in the causal-abstraction sense (Geiger): a proposed
circuit is a set of VARIABLES and COMPUTATIONS between them, and every hypothesis must
say which upstream variable affects which downstream one, tested by interchange
interventions (patch variable values across inputs, not just ablate). Concretely for
layer 1's verified surrogate: the candidate abstraction is
    z := (u . xhat)          [scalar variable, computed by head-4-dominated attn1]
    c0 := a z^2 + b          [the leader coefficient]
    write := c0 * d0         [what downstream reads]
Interchange tests: swap z between two inputs -> the downstream effect must match
swapping c0; verify z's own upstream dependence (which keys/values move z, tested by
patching attention inputs, not correlational attribution). Same template for every
future leader. Bottom-up discipline: because we start at layer 0, the input side must
be fully understood first -- layer 0's variables are functions of (token, attn0
context) only, and their input dependence is exactly characterizable.

## Phase D — theory pass (after analysis + red team of each layer)
Re-derive the empirical findings from the weights where possible, with linear algebra
and tensor-network structure doing the work:
- The bilinear layer is a third-order tensor T = sum_j Down_j (x) Left_j (x) Right_j;
  per-direction forms M_d are its contractions. Ask: do the measured leaders coincide
  with the top components of a weight-side decomposition (HOSVD / CP of T, computed in
  the Lambda-weighted metric)? The validated whitening says WHICH metric makes
  weight-side SVD meaningful -- that is the bridge between the weight basis (60% of
  causal effect) and the data basis.
- Attn1-head-4's dominance should be visible in the weights: the folded operator
  P_h^T M P_h per head, its spectral norm against the measured head shares.
- The surrogate's u is the top whitened eigenvector -- check how much of this entire
  pipeline could have been PREDICTED from weights + input second moment alone, i.e.
  a weights-only protocol with one data statistic (S) as input. That is the honest
  'make the most of the weight-based part'.
- Tensor-network view of the two-layer composition (emb -> attn0 -> mlp0 -> attn1 ->
  mlp1): each leader's surrogate is a small tensor network; write it down explicitly
  and count its contraction cost vs the model's.

## Phase C — remaining layers, triaged
- Layer 16 next (the compressible one, R=9: tractable), then 17 tail directions.
- Layers 2-15: §10 showed individual ablations understate them (2.87x superadditive);
  the per-layer treatment must therefore use Shapley-style attribution from the start,
  and a cheaper variant (fewer perms, coarser basis) is acceptable. Do as budget
  allows, deepest-first by delete cost: 3, 2, 4, 15, ...
- Keep the report updated at phase boundaries; correction-first writing throughout.
