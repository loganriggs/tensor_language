# Basis-aligned bilinear networks

Conceptual clarity program: when a bilinear layer sits between an embedding and an
unembedding, the bilinear layer's own input space is **not** a privileged basis — the
embedding chose it. But you can **fold** the embedding into the bilinear layer and the
unembedding into the down-projection:

```
unfolded:  y = U · D · ( (L·E·x) ⊙ (R·E·x) )
folded:    y = D̃ · ( (L̃·x) ⊙ (R̃·x) )      with  L̃ = L·E,  R̃ = R·E,  D̃ = U·D
```

The folded weights live in the *token/input* basis (rows of L̃, R̃) and the *class/output*
basis (rows of D̃), which ARE privileged. Everything here probes the gap between
**weight sparsity** (basis-dependent, breaks under an inserted rotation) and
**functional sparsity** (the per-class interaction form `B_c = Σ_k D̃[c,k] L̃[k]ᵀR̃[k]`,
invariant to any rotation inserted at the embedding/unembedding interfaces).

## Thread 1 — block-sparse toy + rotation + iterated sparsification (e1, e2)

Task: 8 inputs in 4 fixed pairs (blocks of 2×2 interactions); each pair's product is one
of 4 output classes; exactly one block active per sample. `y_c = x_{2c}·x_{2c+1}`.

- **e1 (hand-coded, no training):** build the exact sparse solution; insert a random
  rotation Q into the embedding and Q⁻¹ into L,R (and Q₂ between D and U). Same function,
  same folded weights, same B_c — but unfolded weights go fully dense. This is the
  "why folding" picture.
- **e2 (trained + sparsified):** iterated sparsification protocol (user-specified):
  L1-penalty training → prune bottom fraction by magnitude → repeat; when val error
  degrades past threshold, revert to previous iterate and finetune WITHOUT L1.
  Arms: (a) no sparsification control; (b) L1+prune on L,R,D only (E,U dense/free);
  (c) L1+prune on everything incl. E,U; (d) start from the *rotated hand-coded* model
  and sparsify everything (does it learn to undo the rotation?).
  Readouts: unfolded vs folded weight sparsity, block-structure score of B_c
  (note: cross-block entries of B_c are OFF-distribution — never probed by the task —
  so trained models may carry invisible junk there).

## Thread 2 — computation in superposition for squares (e3)

(Ref: Vaintrob/Mendel/Hänni, "Toward a Mathematical Framework for Computation in
Superposition", LW 2023.) No embedding/unembedding. Inputs x ∈ R^m sparse (each feature
active w.p. p), target y = x² elementwise, bilinear net with d_h < m hidden units.
- Show trained loss beats the "dedicated" baseline (compute d_h squares exactly, ignore
  the rest) ⇒ more squares computed than hidden dims.
- Count per-feature fidelity (activate feature alone, measure relative error) — is
  #computed > d_h?
- Then run the same iterated sparsification protocol on L,R,D: how sparse can the weights
  get before degradation, and does forced sparsity collapse superposition back to ~d_h
  dedicated features?

## Thread 3 — structure in real LLM embedding matrices (later)

The embedding of a real LLM is itself compressible: the linear down-projection could be
represented with fewer "objects" than vocab size (extreme case: all 50k tokens pointing
the same direction = 1 object). User will specify details when we get there.

## Conventions

- `common.py`: model (dict-of-tensors), fold, interaction tensor, sparsity metrics
  (Hoyer, near-zero fraction), the iterated sparsify protocol, data generators.
- Degradation thresholds in FVU (MSE / Var[y]) so they transfer across tasks.
- Figures via root `palette.py`; results in `RESULTS.md` with embedded figures.
