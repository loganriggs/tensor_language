# Sparse-core decomposition of bilinear attention (two QK sets)

Handoff spec. Self-contained: notation, the three-stage pipeline, a joint-training variant, pitfalls, sanity checks, and a verification protocol against supervised ground truth. Everything is head-local unless stated otherwise.

## 1. Setup and notation

Bilinear attention head with two independent QK circuits and one OV circuit. No softmax; scores enter multiplicatively.

- `E ∈ R^{n×d}`: token embeddings, rows `x_j`. `n` = vocab (or token-occurrence set), `d` = residual stream dim.
- `W1 = Q1 K1ᵀ`, `W2 = Q2 K2ᵀ ∈ R^{d×d}`, each rank ≤ `d_h` (head dim). Keep them factored; never materialize `d×d`.
- `W_OV = V O ∈ R^{d×d_out}`, `V ∈ R^{d×d_h}`, rank ≤ `d_h`.
- Head output for destination token `i`:

```
Out_i = Σ_j p_j (x_iᵀ W1 x_j)(x_iᵀ W2 x_j) · W_OVᵀ x_j
```

with `p_j` a token weight (uniform, or unigram frequency; see §3). In matrix form: `Out = (E W1 Eᵀ ⊙ E W2 Eᵀ) diag(p) E W_OV`.

The source-token dependence is cubic, so the data-independent head object is a third-order tensor:

```
C = Σ_j p_j (W1 x_j) ⊗ (W2 x_j) ⊗ (W_OVᵀ x_j)   ∈ R^{d × d × d_out}
Out_i = C(x_i, x_i, ·)      # contract modes 1,2 with the destination embedding
```

`C` is a bilinear layer whose interaction tensor is the (leg-dressed) third moment of the source tokens.

**Gauge freedoms (canonicalize before any analysis):**

1. Swap `(W1, W2) → (W2, W1)` and rescale `(αW1, α⁻¹W2)`: fix by `‖W1‖_F = ‖W2‖_F` and a deterministic ordering rule.
2. Only the mode-1,2 *symmetrization* of `C` is behaviorally observable, because both modes receive the same `x_i`. Any per-output-direction matrix `Σ_j p_j ⟨b_j,u⟩ (W1 x_j)(W2 x_j)ᵀ` must be symmetrized before eigendecomposition; the antisymmetric part is gauge.
3. Diagnostic worth computing once per head: principal angles between the row spaces of `K1ᵀ` and `K2ᵀ` (and `Q1`,`Q2`). Near-aligned ⇒ head degenerates to squared-salience attention; well-separated ⇒ genuine conjunctive (AND-gated) attention. Report this in every run.

## 2. The decomposition target

Factor the embeddings through a sparse overcomplete dictionary:

```
E ≈ S D,   S ∈ R^{n×m} (rows k-sparse, nonneg optional),   D ∈ R^{m×d} (rows = atoms, m > d)
```

Substituting `x_j = Dᵀ s_j` into `C`:

```
C ≈ M ×1 A1 ×2 A2 ×3 B
M_abc = Σ_j p_j s_ja s_jb s_jc          # sparse core, m×m×m, FULLY symmetric
A1 = W1 Dᵀ ∈ R^{d×m},  A2 = W2 Dᵀ,  B = W_OVᵀ Dᵀ ∈ R^{d_out×m}
```

Read: `M_abc` is the third-order co-occurrence of features on source tokens (data-only, head-independent). The head-specific structure lives entirely in the leg maps: column `a` of `A1`/`A2` is the destination-direction that triggers attention to source-feature `a` through each QK circuit; column `c` of `B` is the payload feature `c` writes when attended. Because one shared dictionary dresses all three legs, `M` is symmetric under all permutations of `(a,b,c)`; the attendability/payload role split comes from the legs, not from `M`. Do not break this symmetry in storage or in the Stage-3 factorization.

Sparsity: each token touches ≤ `k³` entries, so `nnz(M) ≤ Σ_j k³ = n k³`, typically far less after accumulation. Store COO on `a ≤ b ≤ c` with multiplicity bookkeeping.

## 3. Stage 1 — dictionary under the head-induced metric

**Why not a vanilla SAE on `E`:** the head reads `x_j` only through `K1ᵀ, K2ᵀ, Vᵀ` (source side). Reconstruction error in their common kernel is free; error along amplified directions is expensive; and the third-moment contraction *cubes* error that correlates with feature activations. Vanilla L2 on the residual stream optimizes the wrong metric.

**Well-posedness trap:** the pulled-back metric `G = W1ᵀW1 + W2ᵀW2 + W_OV W_OVᵀ` has rank ≤ `3 d_h ≪ d`. Fitting atoms in `R^d` under `G` alone leaves them undetermined up to `ker(G)`. Two clean options:

- **(1a, recommended)** Fit the SAE in concatenated source head space: `y_j = [K1ᵀ x_j ; K2ᵀ x_j ; Vᵀ x_j] ∈ R^{3 d_h}`. Standard TopK SAE on `{y_j}` with weights `p_j`. Atoms live in `R^{3d_h}`; their three blocks are directly the columns of `A1, A2, B` (up to the Q-side factors). Cheap, well-posed, nothing wasted. Note this weights the three blocks equally, whereas the full metric `G = W1ᵀW1 + W2ᵀW2 + W_OV W_OVᵀ` also carries the Q-side / O-side scaling; optionally rescale each block by the corresponding `Q`/`O` singular values (`y_j = [Σ_Q1 K1ᵀ x_j; ...]`) to match `G` exactly. Config flag; default off, but check 5 (behavioral reconstruction) is the arbiter if they disagree.
- **(1b)** Keep atoms in embedding space but add an isotropic anchor: loss `‖G^{1/2}(x_j − Dᵀs_j)‖² + ε‖x_j − Dᵀs_j‖²`. Use only if you need atoms comparable across heads in a shared residual basis (multi-head runs: replace `G` by `Σ_h G_h`).

Sparsity mechanism: TopK (fixed `k`) preferred over L1 to avoid shrinkage bias; shrinkage directly biases `M` (activations enter cubed). Track dead features; resample or use auxiliary loss.

**Stage-1 exit criteria:** (i) reconstruction R² in the fitted metric > target; (ii) **moment-level check**: estimate `‖T − T̂‖ / ‖T‖` where `T = Σ_j p_j y_j^{⊗3}` and `T̂` uses reconstructions — via random contractions, not materialization (see §7 check 4). Per-token L2 can look fine while cross-terms (error correlated with activations) wreck the third moment; this check is the one that matters.

## 4. Stage 2 — build the sparse core

```
for j in tokens:  # or stream over corpus occurrences with p_j implicit
    idx = nonzero(s_j)                     # ≤ k entries
    for (a,b,c) in combinations_with_replacement(sorted(idx), 3):
        M[a,b,c] += p_j * s_ja * s_jb * s_jc * mult(a,b,c)
```

where `mult` accounts for storing only the sorted representative (6 / 3 / 1 for distinct / one-pair / triple). Accumulate in a hash map; convert to sorted COO.

Choices to fix explicitly in config:

- `p_j`: uniform over vocab vs unigram frequency vs task-distribution. Frequency weighting makes `M` an honest data moment (and makes `C` additive across corpus mixtures — a feature worth keeping); uniform over vocab answers "what can the head do" rather than "what does it do on this distribution". Support both.
- Diagonal handling: entries with repeated indices (`a=b`, `a=b=c`) encode single-feature salience, not co-occurrence. Keep them, but report on/off-diagonal mass separately; a core dominated by `M_aaa` means the dictionary already did all the work and Stage 3 is uninteresting.
- Optional normalization to correlation-like scale: `M_abc / (σ_a σ_b σ_c)` with `σ_a² = Σ_j p_j s_ja²`. Store both raw and normalized; factorize raw, inspect normalized.

## 5. Stage 3 — factorize the core

`M` is sparse and fully symmetric. Two model classes, by assumption strength:

- **Symmetric CP**: `M ≈ Σ_r π_r μ_r^{⊗3}`, `μ_r ∈ R^m`. Assumes tokens behave as a mixture of `R` archetypes; `μ_r` is archetype `r`'s feature loading. This is exactly method-of-moments topic modeling on the sparse codes. Uniqueness is generic (Kruskal) without any orthogonality assumption — that is the point of working at order 3.
- **Symmetric LL1 / block terms** if archetypes need internal degrees of freedom: each term `(U_r Σ_r U_rᵀ)`-style with an `L`-dim subspace per mechanism. Start with CP; move to LL1 only if CP residual plateaus with atoms that look like unions.

Fitting: ALS on sparse COO (only touch nnz entries), symmetric-constrained (single factor matrix), nonnegativity optional (nonneg helps interpretability and avoids CP degeneracies/swamps; the codes are nonneg if the SAE is, so `M ≥ 0` entrywise and nonneg CP is natural). Small ridge on factors for stability. Multiple restarts; report best-of and stability across seeds (atom match rate between restarts is itself a health metric).

**MDL / rank selection:** three description-length knobs, penalized together:
`k` (features per token, Stage 1), `m_eff` (features ever used), `R` (mechanisms). Simple scoring: `bits(residual) + β1·k·n + β2·m_eff·cost(atom) + β3·R·cost(μ)`; sweep `R`, take the knee, confirm with the stability metric. Do not tune `R` on the verification tasks of §8.

## 6. Joint training (single objective, optional but the interesting endgame)

Motivation: the dictionary minimizing per-token reconstruction is generally not the dictionary making `M` low-rank. Jointly optimize `D` (or encoder+decoder), and CP parameters `{π_r, μ_r}`:

```
L = Σ_j p_j ‖y_j − Dᵀ s_j‖²                        # reconstruction anchor (in head space, 1a)
  + γ · E_{u,v,w} [ ( m3(u,v,w) − cp3(u,v,w) )² ]   # sketched moment matching
  + MDL penalties on k (TopK schedule), feature usage (group lasso on D rows), R (shrinkage on π)
```

where the moment matching is done by **random contractions** rather than materializing `M` inside the loop:

```
m3(u,v,w)  = Σ_{j∈batch} p_j (uᵀ s_j)(vᵀ s_j)(wᵀ s_j)        # minibatch third moment sketch
cp3(u,v,w) = Σ_r π_r (uᵀ μ_r)(vᵀ μ_r)(wᵀ μ_r)
```

with fresh Gaussian or Rademacher `u,v,w` per step (a handful per batch). Minibatch third moments are high-variance: use large batches for this term and/or an EMA of sketch values keyed to a fixed probe set of `(u,v,w)` triples.

Practical protocol: **always warm-start from the stagewise solution.** Ramp `γ` from 0. The known collapse mode: the encoder can shrink effective feature diversity to make `cp3` trivially matchable (e.g., every token routed to few features) — the reconstruction anchor is what prevents this, so never let its weight decay below the point where Stage-1 exit criteria still hold. Log Stage-1 metrics continuously during joint training; if reconstruction degrades while moment loss improves, γ is too high.

## 7. Sanity checks (run in this order; each is cheap)

1. **Planted-orthogonal unit test.** `d = m`, `D_true` orthogonal, `k` small, no noise: Stage 1 must recover `D_true` to permutation/sign (mean max |cosine| > 0.99), Stage 2 must reproduce the analytic `M` exactly.
2. **Squared-attention limit.** Set `W2 = W1`: the whole pipeline must reproduce the single-QK symmetric analysis; the `K1/K2` principal-angle diagnostic must read ~0. Any asymmetry appearing in results is a bug or a gauge leak.
3. **Permutation null for `M`.** Independently permute each feature's activation column of `S` across tokens (preserves marginals, destroys co-occurrence). Rebuild `M`. Real off-diagonal mass and CP fit quality should collapse relative to the true `M`; if they don't, your "co-occurrence structure" is a marginal artifact.
4. **Moment residual by sketching.** `rel_err = E_{u,v,w}[(T(u,v,w) − T̂(u,v,w))²] / E[(T(u,v,w))²]` with `T` from raw `y_j` and `T̂` from `(M, D)` (or CP thereof). This is the check that catches correlated reconstruction error that per-token loss misses. Gate every stage on it.
5. **Behavioral reconstruction.** On held-out destination tokens, compare `Out_i` from the true head vs `Ĉ(x_i, x_i, ·)` from `(M or CP, A1, A2, B)`. Report relative error and, if a downstream loss exists, loss-recovered.
6. **Ablation monotonicity.** Zero the smallest-|value| entries of `M` (or smallest-π CP atoms) cumulatively; behavioral error should rise smoothly. A cliff at tiny entries ⇒ mass is misallocated (often the frequency-weighting choice).
7. **Gauge audit.** Re-run analysis after random invertible re-gauging `(αW1, α⁻¹W2)` and swap: all reported quantities must be invariant. Anything that moves is reading gauge.
8. **Restart stability.** Stage-3 atom matching across ≥5 seeds (Hungarian match on |cos(μ_r, μ_r')|); unstable atoms are not findings.

## 8. Verification against known co-occurrence ("colors" protocol)

Goal: confirm the unsupervised pipeline rediscovers co-occurrence structure that supervised analysis already established.

**(A) Synthetic ground truth (do this first).**
DGP: `m0` true atoms partitioned into attribute groups (e.g., COLOR, SHAPE, TEXTURE). Each synthetic token samples one feature per group, with a planted joint distribution (e.g., P(red, circle) high, P(red, square) low). `Z ∈ R^{n×m0}` = codes, `E = Z D_true + σ ε`, and construct the head weights either randomly or adversarially (e.g., `ker` of `W`'s overlapping some atoms — good stress test for the metric argument in §3).
Metrics, computed after Hungarian matching of learned features to true atoms by cosine:
- Dictionary recovery: mean max-cosine (MCC-style), fraction matched above 0.9.
- Triple recovery: precision@K between top-K entries of matched-`M` and top-K of the ground-truth `M* = Σ_j p_j z_j^{⊗3}`; Spearman correlation on the union support.
- Mechanism recovery: CP atoms of `M` vs planted joint-distribution modes (each planted correlated pair/triple should appear as an atom loading on exactly those features).
- Sweep: noise `σ`, overcompleteness `m/m0`, `k` mis-specification, and the `W`-kernel adversarial case. The claim to falsify: recovery survives `W`-kernel overlap when fitting in head space (1a) and fails with a vanilla residual-space SAE.

**(B) Supervised cross-check on real data.**
Given token-level labels for attribute groups `g` (e.g., color words, or any labeling from the prior supervised analysis):
1. Supervised co-occurrence tensor over groups: `N_{g g' g''} = Σ_j p_j 1[j∈g] 1[j∈g'] 1[j∈g'']` (tokens can carry multiple group memberships).
2. Assign each learned feature to groups by activation/label association (AUC of `s_ja` for membership in `g`; threshold, allow none/multiple).
3. Aggregate: `M↓_{g g' g''} = Σ_{a∈g, b∈g', c∈g''} M_abc`.
4. Compare `M↓` to `N`: rank correlation over group triples, and specifically check that every supervised finding ("colors co-occur with X") appears as excess mass in `M↓` relative to the permutation null of check 7.3.
Report both hits and *misses in both directions*: supervised triples absent from `M↓` (pipeline failure or supervision artifact) and large `M↓` triples with no supervised counterpart (candidate discoveries — the actual point of the exercise).

## 9. Pitfalls (condensed)

- **Metric rank-deficiency** (§3): never fit atoms in `R^d` under `G` without an anchor; default to head-space fitting.
- **Error cubing**: per-token reconstruction loss is not a valid gate; sketch the third moment (check 4).
- **Shrinkage bias**: L1 SAEs bias `s` low, cubed in `M`; use TopK/JumpReLU or debias activations before building `M`.
- **Frequency weighting silently changes the question**; make `p_j` a config choice and report which.
- **Diagonal dominance of `M`** means Stage 3 is factoring salience, not interaction; report the split.
- **CP degeneracies** (swamps, diverging paired atoms): nonneg constraints + ridge + restarts; prefer LL1 over cranking `R` when atoms pair up.
- **Symmetry leaks**: `M` is fully symmetric; the two-QK asymmetry lives only in `A1 ≠ A2`. An asymmetric core fit is modeling head structure with data parameters.
- **Gauge leaks** (§1): symmetrize mode-1,2 quantities; fix the `(α, swap)` gauge; run check 7.
- **Joint-training collapse**: moment-matching term without a strong reconstruction anchor lets the encoder degenerate; warm-start and ramp γ.
- **Memory**: `n·k³` COO accumulation can spike; stream with a bounded hash map and spill/merge.

## 10. Suggested build order

1. Synthetic DGP + planted-orthogonal unit test (checks 1–3) — no real model needed.
2. Stage 1 in head space on the real head; gate on check 4.
3. Stage 2 + nulls (check 3), diagonal/off-diagonal report.
4. Stage 3 symmetric nonneg CP + MDL sweep + stability (check 8).
5. Behavioral reconstruction and ablations (checks 5–6), gauge audit (check 7).
6. Verification protocol §8(A), then §8(B).
7. Only then: joint training §6, warm-started, with continuous Stage-1 gating.
