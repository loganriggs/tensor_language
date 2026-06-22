# Sparse Universal-AND: what does optimizing for sparse weights change?

`python factorized_sparsity.py` (~10 min). Takes the trained dense seed-2 model
and runs **iterative magnitude pruning** on the pullback weights `W1, W2` and the
decoder `Wo` (the embedding `E` and bias `bo` are not pruned):

```
repeat for 18 rounds:
   (i)  fine-tune 600 steps on   CE + λ·‖W‖₁      (λ = 3e-5)
   (ii) prune the smallest 10% of the still-active weights
```

Gotchas handled (as specified):
- **persistent mask** — pruned weights, *and their Adam moments*, are re-zeroed
  every step, so nothing "retrains from 0" / resurrects.
- **prune only among active weights** — each round ranks by magnitude over
  `mask == 1`, so a 10% round removes 10% *new* weights, never re-counting
  already-zeroed ones.

Every round is checkpointed; we report two operating points and analyze the
sparsest still-separable one (saved to `uand_seed2_sparse.npz`).

---

## The sparsity / accuracy tradeoff

![Sparsity vs CE and accuracy](./fig_sparsity_curve.png)

| sparsity | BCE | TPR @ thr 0 | TNR @ thr 0 | AUC | bal-acc @ best thr |
|---|---|---|---|---|---|
| 0% (dense) | 0.00011 | 0.998 | 1.000 | 1.000 | 0.9998 |
| 10% | 0.00012 | 0.999 | 1.000 | 1.000 | 0.9999 |
| **19%** | 0.00027 | 0.997 | 1.000 | 1.000 | 0.9997 |
| 27% | 0.00065 | 0.986 | 1.000 | 1.000 | 0.9985 |
| 41% | 0.00235 | 0.939 | 1.000 | 0.9998 | 0.9957 |
| **52%** | 0.00622 | 0.898 | 0.999 | 0.9994 | 0.9921 |
| 65% | 0.01484 | 0.799 | 0.997 | 0.9969 | 0.9768 |
| 85% | 0.02122 | 0.226 | 0.999 | 0.9624 | 0.8981 |

Two readings, and the gap between them is the whole point:

- **At the model's own threshold** (logit > 0): weights prune nearly free to
  **~19%** (BCE essentially unchanged, TPR 0.997), and the TPR then collapses —
  down to 0.90 by 52% and 0.23 by 85%.
- **Recalibrated** (best global threshold): the model stays essentially perfectly
  separable far longer — **balanced accuracy ≥ 99% out to ~52% sparsity**, AUC ≈
  1.0 out to ~40%.

The fixed-threshold collapse is **the Q1 calibration effect again** (see
[`couplings.md`](./couplings.md)): the L1 penalty shrinks the signal (mean
+37.9 → +29.0), so positives slide under the *fixed* zero threshold while
**TNR stays ≈ 1.0 the whole way** — a calibration loss, not lost computation.
Re-centering the threshold recovers it. So the honest answer to "how much can you
prune?" is **~half the weights** with the AND function intact, provided you also
recalibrate; ~20% if you insist on the original bias.

---

## How does sparsification change the pullback?

Comparing the dense model with the 52%-sparse checkpoint:

| metric | dense | sparse (52%) |
|---|---|---|
| signal (mean `2·Qf[t,a,b]`) | 37.94 | 29.04 |
| diagonal inhibition (mean) | −15.24 | −11.08 |
| interference std | 11.17 | 9.58 |
| **top-1 interference SVD mode** | **41.5%** | **32.7%** |
| neurons used / target (`Wo` row nnz) | 64.0 | 30.6 |
| targets / neuron (`Wo` col nnz) | 496.0 | 237.3 |
| **Qf exact zeros** | **0.0%** | **0.0%** |
| Qf entries with \|·\| < 0.5 | 7.2% | 9.7% |

![Dense vs sparse pullback Qf](./fig_sparse_pullback.png)

Three things change, one pointedly does **not**:

1. **Everything shrinks together** (signal, inhibition, interference all ~−20%):
   L1 just scales the whole quadratic form down. This is why the fixed-threshold
   TPR falls — the form is the same shape, lower amplitude, against a fixed bias.
2. **Each target now reads ~31 of 64 neurons** instead of all 64 (and each neuron
   serves ~237 targets instead of 496) — genuinely fewer gates per neuron, the
   one real interpretability win.
3. **The dominant interference mode drops 41.5% → 32.7%** — pruning preferentially
   thins the diffuse, inherited embedding-crosstalk (it has no single weight to
   defend), so the interference becomes a touch less rank-1-dominated.
4. **But the pullback `Qf` does not become sparse at all** — still **0.0% exact
   zeros**, and the fraction of near-zero entries barely moves (7.2% → 9.7%). A
   52%-sparse set of weights produces a fully dense feature-space quadratic form,
   because `Qf = Wo·(W1E)·(W2E)` and the frozen **non-orthogonal embedding `E`
   re-mixes everything**. Weight sparsity ≠ representation sparsity here.

**Takeaway.** Optimizing for sparse weights buys a leaner decoder (≈half the
weights, ~31 gates/neuron) with the AND computation fully intact *after
recalibration*, and slightly de-emphasizes the inherited interference mode — but
it does **not** sparsify the pullback. To get a sparse feature-space form you would
have to attack `Qf` (or `E`) directly, not the raw weights; that connects to the
non-orthogonal/sparse-pursuit thread (#3) in `../CONTEXT.md`.
