# The codebook methods: code + intuition + one comparison graph

## 0. The object everything acts on: "factors"

For a layer-0 head, the query input is a deterministic function of the token alone:
embedding row → rms-norm → W_q → take this head's 128-dim slice → per-head rms-norm.
Doing this for every token in the vocab gives a matrix

```python
def branch_factors(model, branch):           # tier2_folding.py, exact (gate 1e-15)
    E  = model.wte.weight                    # (V, 1152)   V = 50304 tokens
    h  = F.rms_norm(E, (1152,))              # what the attention actually reads
    Q  = (h @ W_q.T).view(V, n_head, 128)    # per-token, per-head query vectors
    q_hat = F.rms_norm(Q, (128,))            # the model's per-head QK-norm
    ...                                      # same for k_hat with W_k
    return q_hat, k_hat                      # each (V, n_head, head_dim=128)
```

**"factors" = these per-token vectors.** For one (head, branch) you get a pair of
matrices q̂, k̂ of shape **(V, d_head) = (50304, 128)**: row t is token t's query (or
key) vector *before* rotary. They generate every score the head can ever produce:

```python
score(t_q @ pos i, t_k @ pos j) = ⟨R_i q_hat[t_q], R_j k_hat[t_k]⟩ / d_head
                                = Σ_f cos(ω_f(i−j))·C_f + sin(ω_f(i−j))·S_f
```

so compressing the head = compressing (q̂, k̂). Full DL per (head, branch) = 2·V·128
floats. Shared helper used everywhere below:

```python
def kmeans(X, k, iters=12, seed=0):          # plain Lloyd, chunked distances
    C = X[randperm(len(X))[:k]].clone()
    for _ in range(iters):
        assign = ((X**2).sum(1,keepdim=True) - 2*X@C.T + (C**2).sum(1)).argmin(1)
        C = scatter_mean(X, assign, k)       # dead clusters keep old centroid
    return C, assign
```

---

## 1. Truncated SVD — "few directions, dense codes"

```python
U, S, Vt = torch.linalg.svd(q_hat, full_matrices=False)    # (V,128) → V×128 SVD
q_hat_r  = U[:, :r] @ torch.diag(S[:r]) @ Vt[:r]           # rank-r approximation
# DL = r·(V + 128 + 1) floats per matrix; every token keeps its own r-dim code
```
The spectral prior: the head's token-dependence lives in r global directions.
**Wins:** low-rank plants (at exactly true DL); tiny-model layer-0 (svd16 free — shallow
models are rank-structured); joint svd on the 546M is respectable (+0.0045 at 12.5% DL,
negative at 50%). **Loses per bit** to vq on the 546M (~20× more DL at matched ΔCE);
blind to blocky/positional structure (45–1000× true DL on those plants).

## 2. Token-VQ / bicluster — "few objects"

Your restatement is exactly right: concatenate the head's q̂ and k̂ (so a class must be
consistent for the token in BOTH roles), k-means over TOKENS, then map every token to
its class centroid:

```python
C, assign = kmeans(torch.cat([q_hat, k_hat], 1), k)   # cluster the V tokens
q_c = C[assign][:, :128]                              # token t → its class's q-vector
k_c = C[assign][:, 128:]
# DL = k·256 floats (centroids) + V·log2(k) bits (which class each token is)
```
Afterwards the head genuinely computes only class×class interactions: the effective
score table has k×k distinct token-pair values (per Δ). The battery's 2-D cousin
(`fit_bicluster`) partitions rows and cols separately with block means — spectral init
mandatory (random init lost its own plant to SVD, twice):

```python
U, S, Vt = svd(M);  rows = kmeans(U[:, :k]*S[:k], k);  cols = kmeans(Vt[:k].T*S[:k], k)
for _ in range(30):                                   # alternate to a local optimum
    B = block_means(M, rows, cols)                    # k×k table
    rows = argmin_a Σ_j (M[i,j] − B[a, cols[j]])²     # reassign rows, then cols
```
**Wins:** the 546M's QK — selection is a ~256-class computation; classes are readable.
**Loses:** tiny-model QK; OV content everywhere.

## 3. Band sparsity — "few RoPE frequencies"

```python
qa, qb = q_hat[:, :64], q_hat[:, 64:]                 # the two rotary half-planes
mass   = (qa**2).sum(0) + (qb**2).sum(0)              # energy per frequency band
keep   = mass.argsort(descending=True)[:m]            # keep m of 64 bands
q_hat_m = q_hat * band_mask(keep)                     # zero the rest (both planes)
# DL = 2·V·2m floats + 64-bit mask
```
**Never wins** on real heads (546M needs 48/64 bands jointly) despite visible mid-band
concentration — energy ≠ behavioral necessity.

## 4. Toeplitz / positional — "scores depend only on distance"

```python
c = mean of scores over each diagonal Δ               # from data, or from C_f/S_f
score_hat[i, j] = c[i − j]                            # DL = (2T−1) floats, or
C = torch.fft.rfft(c); keep the top modes             # a few Fourier coefficients
```
**Wins its plant at exactly true DL. Cleanly falsified on every real head** (0/32
branches; the prev-token head *attends* at Δ=1 but its score *magnitudes* are
token-dependent — pattern-positionality ≠ score-positionality). Joint positional on the
546M caps at +1.47 (vs zero's +2.50): ~40% of layer-0 QK's contribution is positional.

## 5. Conjunction — "AND of two cheap structures" (expanded)

Setting: a matrix M that is (approximately) an elementwise PRODUCT of a token-structured
part and a position-structured part, M[i,j] ≈ B-part[i,j] · c(i−j) — e.g. "attend to
same-class tokens, but only nearby". A flat codebook must pay for every (class, distance)
combination (k₁·k₂); the conjunction pays k₁+k₂. You cannot read the two parts off M
directly (neither is observable alone), so we alternate least squares, holding one part
fixed and solving the other in closed form:

```python
# init: gate from the row-shifted energy profile; blocks from the gate-whitened matrix
c0 = sqrt(diag_mean(M**2)); c0 /= c0.mean()          # positive-gate estimate |c(Δ)|
rows, cols = spectral_init(M / c0[d_idx])            # divide the gate OUT, then cluster

for _ in range(8):
    T = c[d_idx]                                     # gate as a full matrix, T[i,j]=c(i−j)
    # (a) block means, WEIGHTED least squares:  min_B Σ (M − B[rows,cols]·T)²
    B = scatter(M*T, rows, cols) / scatter(T*T, rows, cols)
    # (b) reassign rows (then cols) under the same weighted objective:
    #     cost[i,a] = Σ_j (M[i,j] − B[a,cols[j]]·T[i,j])²  — expand, argmin over a
    P = (M*T) @ onehot(cols);  W = (T*T) @ onehot(cols)
    rows = (-2*P@B.T + W@(B**2).T).argmin(1)
    # (c) re-solve the gate per diagonal:  c(Δ) = Σ_Δ M·M1 / Σ_Δ M1²,  M1 = B[rows][:,cols]
    c = diag_ratio(M * M1, M1 * M1)
# DL = k² block means + V·log2(k)·2 partitions + Fourier(c) + 1 scale float
```
Each step is exact weighted LS given the other parts, so the objective is monotone;
the battery's conjunction plant is won 33× over SVD. Identifiability caveat: from the
product ALONE a sign-oscillating gate is unrecoverable (per-diagonal signs can't be
absorbed into block-constant B) — irrelevant for bilinear attention, where the two
branches are given separately by the architecture; there "conjunction analysis" means
analyzing each branch's structure and testing them causally (which is how the induction
circuit's identity ∧ positional factorization was established).

## 6. Conditional-mean lookup (path-folded) — "0th order in context"

```python
# accumulate over induction batches: what key vector does a position advertise,
# as a function of its PREVIOUS token (the L0H1-transported identity)?
kbar[t] = mean{ k_vector(pos j) : token[j−1] == t }        # (V, 128) table
k(pos j) := kbar[tokens[j−1]]                              # replace live computation
```
Wins as a STRUCTURE metric (identity conjunct at 2200× chance, exactly matching causal
ablations where generic weights point at the wrong L0 head). Loses as a COMPUTATION
(−0.62…−0.74 P(copy) when substituted): the circuit consumes context-dependent
components (norm scales, actual pattern weights, within-condition variance) that a mean
discards. Structure-visible ≠ computation-sufficient.

## 7. CE/KL-trained codebooks — "same structure, right objective"

```python
# assignments FROZEN (the discrete structure is fixed); centroids become the only
# trainable parameters; gradients flow through the score computation into the tables
q_tab = nn.Parameter(centroids_q)                     # (n_head, k, 128) per branch
s1 = scores_from_factors(q_tab[assign[tokens]], k_tab[assign[tokens]], ...)
loss = F.cross_entropy(model_forward_with(s1, s2), targets)   # or teacher-CE for KL
# bf16 + grad clipping (fp16 backward diverges); train chunks disjoint from audit
```
**Wins:** QK everywhere — vq64 CE-trained beats the original model (−0.032; KL-faithful
−0.007, so it's compression, not just adaptation). **Partial on OV** (~38% recovery):
content genuinely resists hard classing → next step is top-k sparse coding.

---

![methods compare](fig_methods_compare.png)

One graph, one object (546M layer-0 QK, joint over all 9 heads × 2 branches). Reading:
**vq dominates per bit** (vq256 = +0.008 at 0.6% DL; CE-trained negative at 0.08–0.6%);
**svd respectable at moderate ratios** (+0.0045 at 12.5%; negative at 50%) but ~20× more
DL than vq at matched ΔCE; **band** needs 48/64 bands; **positional** caps at +1.47.
Decomposition: of layer-0 QK's ~2.5-nat contribution, ~1.0 nat is positional, ~1.5
token-selective — and token classes capture nearly all of the selective part.

---

## FAQ (Logan's questions)

**Q: In fig_pattern_display the axes are tokens, but shouldn't they be clusters?**
The axes are *sequence positions* of a real text snippet, labeled with the token at each
position. The ENTRIES are computed from class centroids: token → class → centroid
factors → RoPE(position) → score. Two same-class tokens have identical pre-rotary
factors but sit at different positions, so their rows still differ (through Δ). The
effective token-pair table behind the display has only 256×256 distinct values per Δ.
The updated figure annotates each token with its class id (`token·c17`) so you can spot
positions that share entries.

**Q: How does the compressed QK interface with OV in forward passes? Isn't there an
embedding mismatch?** No shared reduction is performed: the codebook replaces only the
scores s₁, s₂; the value path still reads the full embedding (v(t) = W_v ê_t, full
precision), and the pattern multiplies those full-precision values. Since the scores are
the *only* thing QK feeds downstream, nothing else needs to change — the compressed
model is "class-precision selection × full-precision content." DL accounting is
correspondingly circuit-scoped (layer-0 QK only).

**Q: Could you reduce the Embedding itself in one class structure so OV inherits it?**
That's the shared-registry question — tested in `shared_registry.json`: OV with its own
256 classes, OV forced onto QK's classes, and a single global 256/4096-class embedding
driving both circuits ("layer 0 sees k effective tokens"). See the numbers there (and
LOG tick 12): selection tolerates coarse classes, content does not, and a global
registry inherits the worse of the two.
