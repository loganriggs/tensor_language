# TN-gauge / code-propagation program — goals, ladder, state

**GOAL.** Reduce a bilinear model to a *faithful, legible tensor network of
mechanisms in a shared overcomplete code basis* — Logan's construction: one learned
dictionary Φ (d×m, m>d) per residual stream; every layer expressed as a small
quadratic form over sparse codes; codes manufactured by propagation from the token
boundary; the sparse solver demoted to a faithfulness *auditor*. The model stays one
fixed tensor network; each input runs a small legible sub-network of it. Verified by
end-to-end ΔCE (binding), staying a tensor network throughout (no nonlinear SAEs).

**Method.** Toys first for fast iteration (`runs_lm/block1..block4`, TinyStories,
d=128); port the winning reduction to bilin18 (546M) only once it clears the gates.

## The construction (Logan 2026-07-20), as testable pieces

Layer `y = D(Lx ⊙ Rx)`, bond dictionary Φ (d×m), code `x ≈ Φc`, support `s=|c|₀≪d`.
- Step 1 code: c from (a) sparse solve, (b) amortized encoder, or **(c) inherited
  from upstream** (the target regime — no per-input inference).
- Step 2 layer in code coords: `y = Σ_{j≤k∈s} (2−δ) c_j c_k u_jk`, `u_jk=½D(ℓ_j⊙r_k+ℓ_k⊙r_j)`,
  `ℓ_j=LΦ_j, r_k=RΦ_k`. Precomputed; forward = lookup+combine over the live pairs.
- Step 3 two sparsities multiply: input sparsity |s| × core sparsity (few `u_jk` matter).
- Step 4 closure/propagation: if each `u_jk ≈ Φ w_jk` (sparse), then
  `c_out = c_in + Σ (2−δ) c_j c_k w_jk`, truncate to stop support growth. Codes flow
  by arithmetic; solver only audits `‖x_ℓ − Φc_ℓ‖`.
- Step 5 cost of superposition: (a) code well-definedness = local Gram conditioning
  (near-parallel active atoms → credit split is gauge, report don't interpret);
  (b) residual `ε=x−Φc` amplifies `y=[code]+2T(Φc,ε)+T(ε,ε)` → ~2× rel-error per
  bilinear layer → deep claims degrade with distance from boundaries (LIB mechanism).

Learned object: **Φ only**. Derived (contractions of Φ with frozen weights):
`ℓ_j,r_k,u_jk,w_jk`. Only non-tensor object: the discrete support field `s(x)`.

## Experiment ladder (gate each before the next)

- **[0 DONE] Gauge primitives** (`toy_gauge_probe.py`). Residual bond is pinned by the
  embedding/unembedding boundaries (no interior sweep); real freedoms are per-layer
  private: OV = full O(d_head); QK = RoPE-constrained (input-anchored); MLP hidden =
  pinned by ⊙ (perm+scale only). Weight-only cross-layer DAG is uniform → need
  data-contrastive scoring. → `PLAN.md`.
- **[1 DONE] Shared-Φ code-propagation gate** (`toy_code_propagation.py`). One Φ
  (m=512) coding EVERY bond. **NEGATIVE at this size**: end-to-end ΔCE +0.59 even at
  k=64 (FVU ~0.03–0.08); faithful code-coordinates need far higher fidelity than a
  small shared dictionary gives cheaply. FVU rises with depth (0.07→0.23 at k=16);
  amplification 1.0×(shallow MLP)→1.4×(deep MLP), below the 2× bound but
  depth-increasing (Step-5 mechanism supported, magnitude looser). → findings below.
- **[2 NEXT] Fidelity/bits floor.** How rich must the code be for end-to-end
  ΔCE<0.05? Sweep m∈{512,2k,8k}, k, and **shared-Φ vs per-bond Φ** (Step-4 claims the
  stream forces one Φ; G1 shows one Φ weakens with depth — is per-bond needed?).
  Report bits (structural + estimation) at the ΔCE<0.05 crossing.
- **[3] QK-measure propagation** (Logan's direct ask, not yet done). Reduce layer-0
  QK to an alphabet; propagate the attention-weighted co-occurrence measure
  Pr[s attended from t] to weight layer-1's pair domain; show the effective pair
  count ≪ V² and ≪ k². Tests "alphabets forward, cores at bonds, monomials never."
- **[4] Closure test** (Step 4). Are the live `u_jk` sparse in Φ? If not, codes
  can't propagate without re-solving → regime (c) fails, fall back to (a)/(b).
- **[5] Core sparsity** (Step 3). Fraction of live pairs carrying the output norm.
- **[6] OV L1-gauge sparsity floor vs depth** on block4 (8 layers): does the
  achievable OV rotation sparsity worsen with distance from the boundary (LIB)?
- **[7] bilin18 port** of whatever clears gates 2–6.

## Findings so far

### F1 — the residual bond is pinned; freedoms are per-layer private (tick, PLAN.md)
Verified exactly on block2. Reframes the DMRG-sweep plan into independent per-layer
gauges; no deep-layer SAE; stays a tensor network.

### F2 — naive shared-dictionary code coordinates are too lossy end-to-end
`toy_code_propagation.py`, block2, real TinyStories, baseline CE 1.729.
- **G1** shared Φ (m=512), FVU per bond (LS-refit coeffs, monotone in k):
  k=16 → 0.067/0.104/0.142/0.229 (bonds 0–3); depth-increasing.
- **G2** every bond coded, ΔCE: +2.71/+2.05/+1.52/+1.12/+0.59 at k=4/8/16/32/64.
  Even the richest code costs +0.59 nats. **The propagation regime does not cheaply
  preserve the model with a small shared dictionary** — sets up gate 2 (how big must
  Φ be, shared vs per-bond).
- **G3** MLP error amplification: 1.0× (bond1, shallow) → 1.4× (bond3, deep). Below
  the 2× worst-case bound (input error not aligned with the amplifying directions),
  but depth-increasing — the Step-5 mechanism holds directionally.
- Figure: `fig_code_propagation.png`.
