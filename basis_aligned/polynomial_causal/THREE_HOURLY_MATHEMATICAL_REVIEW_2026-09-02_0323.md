# Three-hour mathematical review — 2026-09-02 03:23 UTC

## Circuit target and current evidence

The program goal remains an executable tensor program that predicts held-out/OOD behavior, composes, supports
selective removals and swaps, and is literally simpler. The present object targets computational specification,
cross-head grouping, within-head splitting, held-out prediction, selective manipulation, reuse, and stable
identification. Rank or storage alone cannot satisfy any of those targets.

Rung 459 identified a natural-text shared object below head grain: the complete equality-restricted score product
from L5H5 can replace L8H4's score while retaining L8H4's payload. MLP9 response, CE recovery, and a frozen L7H3
control all transfer to held-out documents. The payloads are nearly orthogonal. Rung 460 is already preregistered and
CPU-gated to test this exact object on code without refitting.

## Exact tensor-network object

For one head `h`, batch item `b`, query `q`, key `k`, and residual output coordinate `d`:

`T_h[b,q,d] = sum_k E[b,q,k] s1_h[b,q,k] s2_h[b,q,k] u_h[b,k,d]`.

Dimensions are `q,k = 0..255`, head feature `r = 0..127`, and residual coordinate `d = 0..1151`.

- `E[b,q,k]` is the fixed token-equality-and-one-position-shift tensor.
- `Q_h,K_h,Q2_h,K2_h` are `128 × 1152` maps.
- At relative position `delta=q-k`, rotary attention inserts an orthogonal `128 × 128` matrix `R_delta`.
- Each score is
  `s1 = <RMS(Q_h x_q), R_delta RMS(K_h x_k)> / 128`, with the analogous formula for `s2`.
- `u_h[b,k,:]` is the mixed value followed by that head's `128→1152` output-map slice.

The contraction graph is therefore two scalar QK branches feeding a product node; the equality tensor gates that
scalar; the result weights a residual-coordinate payload; and key position is summed. The exact score/payload
factorization reconstructs the native equality term with maximum relative squared error `4.92e-14` in rung 459.

In unnormalized residual variables and at fixed `delta`, each score numerator is a bilinear polynomial

`n_r(x_q,x_k) = x_q^T A_{r,h,delta} x_k`,

where `A = Q_h^T R_delta K_h`. Their product is a degree-four polynomial. RMS normalization divides by four known
square roots of quadratic forms, so the raw model score is a known rational/algebraic function rather than an
ordinary polynomial. Literal price is currently zero deployed saving: these are diagnostic interventions that retain
the native generators and add no replacement program.

## Existing exact result: polynomial factors are almost canonical

A bilinear polynomial `x^T A y` with matrix rank at least two is irreducible over a characteristic-zero polynomial
ring. A short proof is useful here: any nontrivial factorization of a degree-two polynomial must be a product of two
linear forms; because the polynomial has no `x_i x_j` or `y_i y_j` terms, one factor must depend only on `x` and the
other only on `y`, which makes `A` rank one. Contraposition gives irreducibility for rank at least two.

Polynomial rings over a field are unique-factorization domains. Therefore, if the two bilinear numerator factors
have rank at least two and are not scalar multiples, their product determines the unordered pair of factors up to
one reciprocal scalar. This is much less gauge freedom than a generic matrix or Tucker factorization: only branch
swap and scalar remain.

The checkpoint verified directly from the hash-bound weights that every Q and K slice for both score branches of
L5H5, L8H4, and the L7H3 control has numerical row rank `128/128`. Condition numbers are `6.89–14.13`, not near a
rank-one boundary. At zero rotary displacement, the two branch-matrix cosines within a head are `.168` (L5H5),
`.086` (L8H4), and `-.077` (L7H3), so the two factors are not associates. Thus the unnormalized numerator factors
meet the elementary uniqueness conditions.

Kaltofen showed that irreducible factors of a polynomial represented by a straight-line arithmetic program can be
recovered in randomized polynomial time. That is an exact algorithmic match to the unnormalized numerator circuit,
although it is unnecessary here because the model exposes the two factors directly. It matters conceptually: the
two-branch split is an algebraic property of the product, not merely a native-head convention.
[Kaltofen, *Factorization of Polynomials Given by Straight-Line Programs* (1989)](https://kaltofen.math.ncsu.edu/bibliography/89/Ka89_slpfac.pdf)

## Why generic tensor decomposition is not the right solver yet

Kruskal's theorem makes a three-way CP decomposition unique up to scaling and permutation under a Kruskal-rank
condition. Robust versions give approximate recovery under quantitative conditioning.
[Kruskal (1977)](https://www.sciencedirect.com/science/article/pii/0024379577900696),
[Bhaskara, Charikar, and Vijayaraghavan (2014)](https://proceedings.mlr.press/v35/bhaskara14a.html)

Those theorems do not directly solve this rung. Our object is a **product** of two bilinear forms, not a sum of rank-
one triads; coefficient symmetrization reuses the query and key variables; RMS denominators make the raw function
rational; and equality support observes only a restricted slice of input pairs. Converting it to a generic CP/Tucker
fit would add rotational or component gauges that the product factorization does not have. Kruskal machinery may
become relevant later if each QK branch is decomposed into a sum of shared feature atoms, but it is not the canonical
first split.

## What remains non-identifiable

Algebra fixes the two numerator branches only up to swap and reciprocal scale. It does **not** say which factor in
L5H5 corresponds to which factor in L8H4, because the heads use different states and normalized feature maps. Plain
weight-space cosine does not solve this: at zero displacement, the four L5H5-to-L8H4 branch-matrix cosines are all
similar (`.206–.218`). The causal score-product result can therefore be real even though Frobenius geometry cannot
name the shared branch.

RMS normalization is the other gap. Multiplying by the known denominators recovers polynomial numerator uniqueness,
but operational interchange must preserve the full normalized score, not just its numerator. This is exactly why a
held-out branch transplant is still required.

## Executable consequence

After the already-frozen code confirmation, split only the identified L5H5→L8H4 score product. On natural fitting
documents construct the four branch-preserving possibilities

- L5 branch 1 × L8 branch 2;
- L5 branch 2 × L8 branch 2;
- L8 branch 1 × L5 branch 1; and
- L8 branch 1 × L5 branch 2,

with fit-half RMS scale matching and L8H4's payload fixed. These exhaust the branch-swap ambiguity without choosing
by weight cosine. Select by the same downstream MLP9 response, task margin, causal recovery, and other-branch
control; then validate without refitting.

Opposing predictions are sharp:

1. **One branch is shared.** Exactly one cross-head branch transplant reproduces most of the full-score transplant's
   response and CE recovery, and the complementary branch does not. This identifies a reusable QK feature family.
2. **Both branches are duplicated.** Two complementary branch transplants work and compose predictably. The whole
   matcher is duplicated at branch grain.
3. **Only the product is shared.** No single branch transplant works although the complete score product does. The
   reusable object is the coupled product, and atomizing Q/Q2 or K/K2 would be misleading.

Kill conditions are failure of the frozen code confirmation, branch-reconstruction error, no task-specific held-out
branch response, or branch choices that reorder across document halves. This is higher-information than a Q/K rank
or SAE sweep because it can identify which exact normalized factor is reusable and supports a direct swap.

## Decision

The mathematical view strengthens rather than redirects the live path. Rung 460 remains the immediate test because
OOD confirmation is required before elaborating the natural-only positive. If it passes, polynomial factorization
shows that a two-branch causal split is principled and essentially exhaustive up to scale/swap; implement the four
branch hybrids rather than search arbitrary low-rank or sparse coordinates. If it fails, do not spend the code role
on branch variants and retain only a natural-text shared-score claim.

