# Tensor-network embeddings, explained (e8 / e8b)

The class-4 representation from the 4-class map: store the embedding matrix as a **tensor
network** instead of a lookup table. This file explains every moving part, what the
hyperparameters mean, and collects all results so far.

## 1. The object

`E` is a lookup table: token id → row vector in R^1024 (V = 50304 rows; padded to
65536 = 16⁴ so the index factorizes; pad rows = row-mean, i.e. zero after centering).

Write each token's **position** (see "ordering" below — position ≠ token id in general)
in base 16 with 4 digits:

```
position p  =  d₁·16³ + d₂·16² + d₃·16 + d₄        dᵢ ∈ {0..15}
```

Now "look up row p of E" is the same as "index a 5-way tensor T[d₁,d₂,d₃,d₄,:] of shape
(16,16,16,16,1024)". A tensor network approximates T as a contraction of small **cores**,
so the table is stored implicitly.

## 2. What actually happens when you index token 90 (worked example)

Say we're in the BPE ordering, so token 90 sits at position 90:

```
90 = 0·4096 + 0·256 + 5·16 + 10   →   digits (d₁,d₂,d₃,d₄) = (0, 0, 5, 10)
```

(Under the semantic ordering you'd first look up the token's position p = π(90) in the
permutation, then take p's digits. That indirection is stored as 4×4 = 16 bits per token.)

The tensor-train (TT) has 5 cores:

```
G₁: (16, r₁)        — 16 row-vectors, one per value of digit 1
G₂: (r₁, 16, r₂)    — 16 matrices of shape r₁×r₂, one per value of digit 2
G₃: (r₂, 16, r₃)
G₄: (r₃, 16, r₄)
G₅: (r₄, 1024)      — one final matrix shared by all tokens
```

The lookup is a chain of tiny matrix products, one per digit — each digit **selects which
matrix to multiply by next**:

```
v = G₁[0]            # r₁-dim row vector        (digit 1 = 0)
v = v @ G₂[:, 0, :]  # now r₂-dim               (digit 2 = 0)
v = v @ G₃[:, 5, :]  # now r₃-dim               (digit 3 = 5)
v = v @ G₄[:,10, :]  # now r₄-dim               (digit 4 = 10)
embedding = v @ G₅ + rowmean   # 1024-dim
```

So yes — "indexing does some indexing thing that ends up computing the right vector":
each digit picks one small matrix out of that core's stack of 16, and the product of the
four picked matrices (times G₅) *is* the token's embedding. The whole point: **two tokens
with the same digit prefix share the partial product** — all 4096 tokens with d₁=0 share
`G₁[0]`; all 256 tokens with prefix (0,0) share `v` after two steps. Shared prefixes =
shared "partial meaning". That's why the format is a hierarchy prior, and why it only
pays off if prefix-mates actually have something in common — which is what the ordering
controls. (Cost: a lookup is ~4 small matvecs instead of free; TT trades FLOPs for bytes.)

## 3. The hyperparameters (all sweepable)

**(a) Ordering π** — which token sits at which position. The ONLY knob that injects
semantics. Tested: `bpe` (identity), `random`, `semantic` (balanced recursive 16-way
k-means, so digit-blocks = cluster tree). Not tested: orderings optimized jointly with
the decomposition (our semantic ordering is a lower bound on the best achievable).

**(b) Digit factorization — "number of bits and which base".** Any (b₁,…,b_p) with
∏bᵢ ≥ V works; you need log₂∏bᵢ ≥ log₂V ≈ 15.6 bits to specify a vocab item. Ours:
4 digits × 4 bits (16⁴). Alternatives: 16 binary digits (2¹⁶, deepest hierarchy, block
sizes 2,4,8,…), 8 digits of 4, 2 digits of 256 (shallow). More digits = more sharing
scales but more cores and more rank bottlenecks. **Untested beyond 16⁴.**

**(c) Tree topology** — how the digits are wired:
- **chain (TT)**: d₁–d₂–d₃–d₄–d_model. What e8 ran (TT = HT with a linear tree).
- **balanced tree (HT proper)**: leaves (d₁,d₂)→node, (d₃,d₄)→node, nodes→root, root
  carries the 1024-dim mode. e8b ran one config: leaf matrices A₁..A₄ (16×r), pair cores
  B₁₂,B₃₄ (r,r,r₂), root C (r₂,r₂,1024).
- **star (Tucker)**, **TT-matrix** (also factorize the 1024 embedding dims into digits —
  the Khrulkov et al. format used for real embedding-layer compression; more
  param-efficient). **Untested.**
- Where the d_model mode attaches (we always put it at the end/root). **Unswept.**

**(d) Ranks (bond dimensions) — what rmax means.** Every edge of the tree carries a rank
r_e: the width of the intermediate vector passed along that edge — equivalently, the rank
of the matrix you get by flattening T with "digits on one side of the cut" as rows and
"everything on the other side" as columns. Interpretation: **r_e = how many independent
"partial meanings" can cross that boundary.** If the 4096 tokens in a d₁-block only ever
need 64 independent directions-worth of shared content, rank 64 suffices on that edge.
`rmax` is simply a uniform cap applied to every edge (natural ceilings apply anyway:
r₁ ≤ 16, r₂ ≤ 256, r₃ ≤ 4096, r₄ ≤ 1024 — so "rmax=256" really means ranks
(16, 256, 256, 256)). Per-edge rank allocation instead of one cap: **unswept.**

**(e) Solver & objective.**
- `TT-SVD`: deterministic sequential SVDs; quasi-optimal for Frobenius (within √#modes of
  the best TT at those ranks).
- gradient fit (Adam on relative Frobenius error = 1 − cos² with optimal scale — the
  tensor generalization of cosine similarity, scale included).
- CE-through-the-frozen-model on the cores (the behavioral metric; e7b machinery).

**Parameter count** (the bytes side of the MDL graph):
`TT: 16r₁ + 16r₁r₂ + 16r₂r₃ + 16r₃r₄ + 1024r₄` floats + 16 index bits/token (only if
the ordering isn't the identity). rmax=256 → 2.43M floats = **4.7% of E**.

## 4. Results so far

**Ordering gap, TT-SVD (e8)** — FVU (weight-metric; 1.0 = predict every token as the
mean, 0 = perfect). Same matrix, same solver, only the row-ordering differs:

| rmax | params (% of E) | random | BPE | semantic |
|---|---|---|---|---|
| 32 | 0.1% | 0.994 | 0.991 | **0.960** |
| 64 | 0.4% | 0.983 | 0.980 | **0.949** |
| 128 | 1.3% | 0.950 | 0.948 | **0.919** |
| 256 | 4.7% | 0.875 | 0.874 | **0.848** |
| 384 | 8.5% | 0.812 | — | **0.787** |
| 512 | 13.4% | 0.745 | — | **0.723** |
| 768 | 26.1% | 0.605 | — | **0.586** |
| 1024 | 42.9% | 0.467 | — | **0.450** |

- Semantic beats random by a stable ~0.02–0.03 FVU at every size → the vocab is
  **measurably but weakly hierarchical**; BPE ordering ≈ random (token-id adjacency
  carries no block structure).
- "Isn't 0.85 bad?" Yes — but it's 4.7% of the bytes on a matrix where even SVD needs
  51% for FVU 0.32 (e6). Increasing rmax closes it (table above); plain SVD at matched
  bytes stays slightly ahead throughout (e.g. 0.556 @ 25% vs TT 0.586 @ 26%).
  **Correction logged in FINDING 10:** an earlier draft claimed TT beats SVD per param —
  wrong (arithmetic error, 2.4% vs 4.7%).

**Weights-only gradient fits (e8b)** — "optimize the tensor net to match the matrix,
no data, no model", at matched ~2.4M params:

| method | ordering | FVU |
|---|---|---|
| TT-SVD (rmax=256) | semantic | 0.848 |
| TT gradient fit (rmax=256) | semantic | **0.839** |
| TT gradient fit | random | 0.867 |
| balanced HT gradient fit (r=16, r₂=48) | semantic | 0.900 |
| balanced HT gradient fit | random | 0.949 |
| SVD reference (r=50, 2.57M params, e6) | — | 0.836 |

- Gradient fit improves on TT-SVD by ~0.009 — sequential SVD was already near-optimal.
- The one balanced-HT config tried **underperforms the chain** at matched params (its
  root core hogs the budget: r₂²·1024 floats), BUT its **ordering gap is ~2× larger**
  (0.05 vs 0.028) — the more the topology leans on block structure, the more the
  semantic ordering matters. A proper topology×rank sweep might find HT shapes that win.

**Behavioral (e8):** raw TT-SVD rmax=256 swap-in costs +4.9 nats (all orderings — at
4.7% budget the weight-metric fit is nowhere near good enough); CE-finetuning the
semantic cores through the frozen model recovers **+4.91 → +2.01 nats at ~4.7 MiB**,
landing on the MDL Pareto envelope at that size (see `figures/e7_mdl.png`).

## 5. The sweep we haven't run

The full grid is {ordering} × {digit base/count} × {topology, d-mode placement} ×
{per-edge ranks} × {solver ∈ Frobenius-SVD, Frobenius-grad, CE-grad}. Most informative
next cells, in order: (i) 2¹⁶ binary digits with a matched-params chain (does a deeper
hierarchy widen the semantic-vs-random gap?); (ii) TT-matrix format (split the 1024 dims
too — the literature's efficient version); (iii) CE-in-the-loop training of the whole
sweep's best cell, since e7 says the weight metric undersells every representation;
(iv) ordering optimized against the decomposition rather than fixed from k-means.
