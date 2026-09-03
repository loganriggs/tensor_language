# Three-hourly mathematical review — 2026-09-03 10:06 UTC (Claude)

Grounded in the live pivot: attention8 DAS closed (§2671, planted falsifier 0/15), and Codex pivoted to the exact
MLP0 decomposition — rung525 (§2672) showed the exact token-context operator does NOT group across tokens
(strong null); rung526 (downstream-conditioned operator grouping) is building. Sign convention §2135 (not used;
weight-space rank statistics).

## The object and the key mathematical observation

For the bilinear MLP0(z) = D[(Lz) * (Rz)] + bias (L,R: 4608x1152; D: 1152x4608), the token-conditioned linear
operator on a context deviation delta is `K_t delta = D[diag(L x_t)(R delta) + diag(R x_t)(L delta)]`. This is
LINEAR in the token embedding x_t: `K_t = sum_k x_t[k] G_k`. Therefore the entire 50,257-token operator FAMILY
lies in a subspace of operator space of dimension <= d_model = 1152. So rung525's finding "the operators do not
GROUP (are not equal)" is unsurprising — the real, and untested, compression question is the family's EFFECTIVE
RANK: do the 50k operators live near a LOW-dimensional subspace (compressible: store the subspace + per-token
coordinates), even though no two are equal?

This reduces to an exact 1152x1152 eigenproblem: family covariance nonzero spectrum = eig of
`Sig^{1/2} Gram Sig^{1/2}`, Sig = embedding 2nd moment, `Gram[k,l]=tr(G_k^T G_l)` = `L^T(P*QRR)L + R^T(P*QLL)R +
L^T(P*QRL)R + transpose`, P=D^TD, QRR=RR^T, QLL=LL^T, QRL=RL^T. Exact, noise-free — escapes the §2668 ceiling.

## Ranked moves

### 1. Operator-family effective rank — EXECUTED (CPU, 2.1 s), §2673
Frozen threshold before compute: effective rank < 288 (a quarter of the 1152 bound) = low-dim/compressible.
Result: effective rank (entropy) = 438; 90% of energy needs 611 of 1152 dimensions, 99% needs 1050; top-1
energy 0.06 (no dominant direction). Verdict: HIGH-RANK, NOT compressible. The MLP0 token-context operator family
genuinely spans most of the available 1152 dimensions. This EXACTLY explains rung525 (operators do not group
because nearly every token has a materially different operator) and answers my §2671/09:31 caution with an exact
number: the noise-free pivot object LACKS a small shared operator vocabulary. (Caveat: intrinsic rank from the
bilinear weights, token-embedding x_t, uniform vocab; robust to Codex's exact centering/gauge/attention0 basis,
which are near-orthogonal and do not change rank.)

### 2. Context-only and token-only branch ranks — PROPOSE/next (CPU)
Apply the same exact-Gram method to MLP0's OTHER exact branches: the context-context quadratic (Lδ)*(Rδ) and the
token-only part. rung396 gave the degree-1 token rank curve; the context-context quadratic map's rank is
untested. If those are ALSO high-rank, MLP0 block 0 is genuinely high-complexity with no exploitable low-dim
structure — a decisive program-level conclusion. CPU, same weights; I can execute next or hand to Codex.

### 3. Downstream bisimulation quotient (Codex rung526) — the causal-abstraction framing
Codex's "group operators by stable downstream use" IS a bisimulation quotient (operators equivalent iff
interchanging them leaves all downstream circuit effects invariant, Beckers-Halpern/Geiger). Note: move-1 bounds
its payoff — if the operator family is high-rank, downstream-use grouping can only help if MANY high-rank
operators map to the same downstream effect (a genuine abstraction collapse), which the §2668 low causal leverage
makes unlikely. Codex's active lane; framing offered.

## Pruned
Operator EQUALITY grouping (rung525, done); effect-variance metrics (§2666, superseded by §2668 MDL); gauge
quotient of the CP tensor (parameterisation dof, not storage); Hankel/RMSNorm (non-composable). Cites: standard
tensor-mode rank; Beckers-Halpern 2019; Rissanen MDL (§2668).

## Top three
1. Operator-family effective rank — DONE (§2673): HIGH-RANK, pivot object not compressible.
2. Context/token branch ranks — next CPU (does the rest of MLP0 also lack low-dim structure?).
3. Downstream bisimulation quotient — Codex's rung526; payoff bounded by move-1's high rank.

## Executed
Move 1: exact operator-family effective rank (ops/mlp0_operator_family_rank.py, result committed, ledger §2673).
The pivot object is high-rank/not-compressible — a decisive, exact, noise-free answer that sharpens the pivot's
prospects.
