# Spec: Balanced canonical forms for bilinear MLPs, and gauge-honest statistics and nulls

Handoff document for implementation. Two work packages. WP1 is closed-form and should be quick; WP2 builds on it. Background and motivation are included so this is self-contained; skim the math, implement the deliverables, run the listed sanity tests before any experiments.

## Background

A bilinear MLP layer computes

```
y = C ((A x) ⊙ (B x))
```

with `A, B ∈ R^{h×d}`, `C ∈ R^{o×h}`, and ⊙ elementwise product over the hidden index. Write `a_i, b_i` for the i-th rows of A, B and `c_i` for the i-th column of C. The layer's function is determined by the third-order interaction tensor

```
T = Σ_i  c_i ⊗ a_i ⊗ b_i        (a CP / rank-h decomposition)
```

**Gauge freedom (per-unit torus).** For any nonzero scalars α_i, β_i, the substitution

```
a_i → α_i a_i,    b_i → β_i b_i,    c_i → c_i / (α_i β_i)
```

leaves every rank-1 term, hence T, hence the function, exactly unchanged. Additionally, permuting hidden units (applying the same permutation to rows of A, rows of B, columns of C) leaves the function unchanged.

Consequence: any statistic computed from the raw entries or norms of A, B, or C **individually** is not a property of the function. Example: the stable rank of A, the distribution of row norms of A, or "which units have large c_i" can each be set almost arbitrarily by choosing gauge, without changing what the network computes. Statistics built only from the per-unit products (e.g. `‖a_i‖‖b_i‖‖c_i‖`, or anything computed from T itself, including the recursive-Frobenius tensor similarity metric) are gauge-invariant and unaffected by any of this.

**Balanced (canonical) gauge.** On each gauge orbit there is a distinguished representative: the one minimizing the total Frobenius norm `‖A‖_F² + ‖B‖_F² + ‖C‖_F²`. For the torus action this has a closed form. Minimizing `α²‖a‖² + β²‖b‖² + ‖c‖²/(αβ)²` over α, β > 0 gives, at the optimum,

```
‖a_i‖ = ‖b_i‖ = ‖c_i‖ = m_i := (‖a_i‖ ‖b_i‖ ‖c_i‖)^{1/3}
```

i.e. rescale each of the three vectors of unit i to have norm equal to the geometric mean of the three original norms. This is exact and one-shot; no iteration. Theory context (not needed for implementation): this is the Kempf-Ness / minimum-norm condition; gradient flow conserves the balancedness defect, and L2 regularization drives it to zero exponentially, so trained-with-weight-decay networks should already be near-balanced. That last sentence is an empirical prediction we will test.

**Residual symmetry after balancing.** Balancing kills the noncompact scale freedom. What survives per unit is the sign choice `(α_i, β_i) ∈ {±1}²` (four elements; note (−1,−1) flips a_i and b_i and leaves c_i fixed since αβ = 1), plus global permutation of units. So the canonical form is unique up to signed permutations, generically. Non-generic degeneracies (two units with proportional structure) allow more mixing; do not worry about handling this specially, but see the uniqueness test below.

---

## WP1: Balancing and gauge-honest statistics

### Deliverable 1.1: `balance_bilinear(A, B, C) -> (A', B', C', info)`

Per unit i:

1. Compute `na = ‖a_i‖, nb = ‖b_i‖, nc = ‖c_i‖`.
2. If any of the three is ≤ eps (say 1e-12 relative to layer scale): the rank-1 term `c_i ⊗ a_i ⊗ b_i` is (numerically) zero, so the unit contributes nothing. Set all three vectors of that unit to exactly zero (this is the minimum-norm representative of a dead unit) and record it in `info['dead_units']`.
3. Otherwise set `m = (na * nb * nc)^{1/3}` and rescale: `a_i *= m/na`, `b_i *= m/nb`, `c_i *= m/nc`.

Optionally canonicalize signs (flag `fix_signs=True`): e.g. flip (a_i, b_i) jointly so that the largest-magnitude entry of a_i is positive, and flip (a_i, c_i)... careful: allowed sign moves are exactly (α, β) ∈ {±1}² acting as above. A simple convention: use (−1,−1) to make the max-|entry| coordinate of a_i positive; then use (−1, +1) (which flips a_i and c_i's sign via αβ = −1... verify against the action definition when implementing) to make the max-|entry| of b_i positive. Any consistent convention is fine; document the one chosen. Default the flag off; nothing downstream requires it except the uniqueness test.

`info` should contain: per-unit m_i, the balancedness defect before/after (see 1.2), dead unit indices.

### Deliverable 1.2: Balancedness defect metric

Define per unit

```
δ_i = std( log ‖a_i‖, log ‖b_i‖, log ‖c_i‖ )
```

(std of the three log-norms; zero iff balanced; scale-appropriate since gauge acts multiplicatively). Layer defect: report both `max_i δ_i` and the m_i-weighted mean of δ_i (weight by m_i³ so dead/tiny units don't dominate). Also implement the additive defect from the paper for cross-checking: `Δ_i = (‖a_i‖² − ‖b_i‖², ‖b_i‖² − ‖c_i‖²)`.

### Deliverable 1.3: Tests (must pass before anything else)

1. **Function preservation.** For random (A, B, C) and random inputs x, `‖f_before(x) − f_after(x)‖ / ‖f_before(x)‖ < 1e-6` (float32) after balancing. Also compare the materialized interaction tensors T directly for small dims.
2. **Idempotence.** `balance(balance(θ)) = balance(θ)` exactly (up to fp noise).
3. **Gauge invariance of the canonical form.** Apply a random torus gauge with log α, log β ~ N(0, σ²), σ up to 3, then balance; result must equal `balance(original)` up to signed permutation of units (with `fix_signs=True` and a canonical unit ordering, e.g. sort by m_i descending with a deterministic tiebreak, it should be exactly equal; report max abs deviation).
4. **Invariant quantities untouched.** `‖a_i‖‖b_i‖‖c_i‖` per unit, and T itself, identical before and after (they are gauge-invariant; this is a consistency check on the implementation).

### Deliverable 1.4: The demonstration experiment (gauge artifacts in weight statistics)

Purpose: produce the figure/table showing that common weight statistics are gauge artifacts and that balancing fixes them.

Take a trained bilinear model (any of the existing toy checkpoints; a freshly trained small model on the toy task is also fine). For k = 200 random torus gauges (log-normal scales, σ ∈ {0.5, 1, 2}), compute at each gauged point:

- stable rank of A and of B: `‖M‖_F² / ‖M‖_2²`
- singular value entropy of A (effective rank `exp(H(σ²/Σσ²))`)
- Gini coefficient / top-k mass of the column norms of C ("which units look important if you only look at C")
- top-10 unit ranking by `‖c_i‖` alone

Report the spread of each statistic across gauges (it should be large, and ranking-by-‖c_i‖ should be unstable), then the single value at the balanced point, and note that ranking by m_i (invariant) coincides with ranking by any single norm at the balanced point. This is the "before/after" evidence for the writeup.

### Deliverable 1.5: Defect on real checkpoints

For every available trained checkpoint (toy language models, modular arithmetic runs, any bilinear-transformer variants), measure the layer defect of 1.2. Group by whether the run used weight decay and by optimizer. Prediction from theory: weight-decay runs sit near zero defect; no-weight-decay runs may not; AdamW breaks the exact conservation law so treat the prediction as qualitative. If defect is large anywhere, that itself is a finding worth flagging (it means raw-weight readings of those checkpoints were in an arbitrary gauge).

### Note on where balancing is REQUIRED downstream

Not needed for: the recursive-Frobenius similarity metric, or any statistic of T (already invariant). Required for: per-matrix effective ranks, norm thresholds for pruning/sparsity, the linearize-vs-prune-vs-keep decision when the threshold is applied to individual factor norms, and any curvature/Hessian/flatness measurement at a parameter point (Hessian spectra vary along the orbit; measure them at the balanced representative or they are meaningless). Any such measurement in downstream code should call `balance_bilinear` first and assert defect < tol.

---

## WP2: Canonical null baselines

### The problem being fixed

The existing "gauge-scrambled" null applies random gauge transformations to a trained model and recomputes statistics, treating the scrambled ensemble as a reference distribution. For gauge-**invariant** statistics this null is degenerate (the statistic never moves), which is fine and even diagnostic. For gauge-**dependent** statistics it is ill-posed: the torus is noncompact, so there is no natural distribution to scramble from; log-normal with σ=1 vs σ=2 gives different nulls and there is no principled way to pick σ. Conclusions of the form "trained value exceeds the scrambled distribution" then depend on an arbitrary choice.

### The fix: project to canonical form instead of averaging over the orbit

Replace orbit-sampling with orbit-projection. Every model (trained, random-init, task-shuffled) gets mapped to its balanced canonical point before any gauge-dependent statistic is computed. Comparisons are then canonical-point vs canonical-point, i.e. genuinely orbit vs orbit. Concretely, the null pipeline becomes:

1. `balance` the trained model.
2. `balance` each null model (random init with matched shapes/init-scale; task-shuffled retrain; whatever nulls are already in use).
3. Compute the statistic on both; compare distributions.

The old scramble still has a role as a **diagnostic**, not a null: for any statistic S claimed to be meaningful, verify `S(balance(gauge_g(θ))) = S(balance(θ))` for random g. Implement this as a reusable assertion (`gauge_robustness_check(statistic_fn, θ, n_gauges)`), and run it on every statistic used in a writeup.

### Residual-group scrambling (where a nontrivial canonical null exists)

After balancing, the surviving symmetry group differs by architecture, and this determines whether a residual-scrambling null is interesting:

- **Bilinear / CP (frozen elementwise product):** residual group = signed permutations of units. Discrete. Scrambling over it moves nothing interesting (all sane statistics are permutation/sign invariant). So for bilinear MLPs, the canonical null is just the balanced-null-model comparison above; there is no continuous residual scramble. Implement the signed-permutation scramble anyway as a cheap sanity check (statistics should be exactly invariant).
- **Trainable-core / Tucker-style bond edges (tensor-transformer variants with a trainable mixing tensor on an internal edge of bond dim r):** the gauge group on that edge is GL(r), and after norm-minimization the residual group is O(r), which is compact and carries Haar measure. Here residual scrambling IS a well-defined nontrivial null: sample Q ~ Haar(O(r)) (QR of a Gaussian matrix with sign-fixed R diagonal), apply Q to one side of the edge and Qᵀ to the other, recompute the statistic. This gives a canonical finite-volume reference distribution for bond-basis-dependent quantities (e.g. sparsity or alignment of individual bond directions). Deliverable: `haar_orthogonal_scramble(edge_params, n_samples)`.

### Deliverable 2.1: `balance_ttn_edge` (needed for the Tucker-style case)

For an internal edge between node tensors W_u, W_v with bond dimension r, matricize along the edge: `M_u = mat_e(W_u)` (bond as columns), `M_v = mat_e(W_v)` (bond as rows). The one-edge minimum-norm representative comes from an SVD of the contraction: let `P = M_u M_v` (shape: u's other modes × v's other modes), take thin SVD `P = U Σ Vᵀ` keeping only σ_j > eps, and set

```
M_u' = U Σ^{1/2},    M_v' = Σ^{1/2} Vᵀ
```

then fold back into tensor shape (zero-padding the bond index back to r if rank dropped). This simultaneously (a) balances the edge (`M_u'ᵀ M_u' = M_v' M_v'ᵀ = Σ`), (b) deletes dormant junk (components of either tensor annihilated by the other), and (c) preserves the contraction exactly. For a network with multiple internal edges, sweep edges repeatedly applying this projection; total parameter norm is nonincreasing and bounded below, so the sweep converges; stop when every edge defect `‖M_uᵀM_u − M_vM_vᵀ‖_F / ‖Σ‖_F < tol`. (Caution: the fold/unfold index bookkeeping is the entire difficulty here; write shape-asserting tests with tiny dims first.)

Tests: contraction preserved to fp tolerance after each single-edge projection and after full sweeps; sweep convergence in a bounded number of iterations on random networks; rank-junk removal (construct a network with planted dormant components as in the paper's Appendix F counterexample style: nonzero slices of one tensor lying in the kernel of the other; verify the sweep zeroes them and strictly reduces total norm while preserving the function).

### Deliverable 2.2: Uniqueness / multi-orbit check (empirical, important caveat)

The theorem "unique balanced point per function, up to compact residual" is proven for tree networks with trainable cores. For bilinear/CP (frozen diagonal core) it is NOT guaranteed; the fiber of a given function may in principle contain multiple balanced points not related by signed permutation. Test empirically: for each model studied, apply n=20 random gauges, balance each, and check pairwise agreement up to signed permutation (after canonical sort). Report the failure rate. Torus gauges cannot move between distinct balanced points, so also test with stronger fiber-preserving perturbations where available (e.g. for models with duplicate/degenerate units, mix them and re-balance). If disagreements appear, flag loudly; it bounds how much weight the "canonical" interpretation can bear for that model.

### Deliverable 2.3: Integration into the existing null-baseline suite

Wire the above into the existing baseline code (alongside Marchenko-Pastur and task-shuffled nulls): every model entering a weight-statistic comparison passes through `balance_bilinear` / TTN sweep first; gauge-dependent statistics additionally get the `gauge_robustness_check`; Tucker-style models additionally get the Haar-O(r) residual null. Keep the raw-gauge path available behind a flag for before/after comparisons.

---

## Order of work

1. WP1 deliverables 1.1-1.3 (closed form + tests). Small.
2. 1.4 demonstration experiment, 1.5 checkpoint defect survey.
3. 2.1 TTN edge balancing with tests (fiddly bookkeeping; budget time).
4. 2.2 uniqueness check, 2.3 integration.

## Assumptions made in this spec (flag if wrong)

- Bilinear layer convention `C((Ax) ⊙ (Bx))`; adapt row/column conventions to the actual codebase.
- Biases, layernorm, and residual streams are outside the gauge action described here; balance only the (A, B, C) triple per bilinear block. If the codebase folds biases in as an extra input dimension, the same formulas apply with the augmented x.
- Frobenius norm / L2 is the norm of record (it is what weight decay and the conservation law privilege). If a differently-weighted inner product is wanted later, m_i generalizes to a weighted geometric mean; not in scope now.
