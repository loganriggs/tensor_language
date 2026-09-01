# Three-hour mathematical review — 2026-09-01 21:23 UTC

## Goal and decision

The goal is a smaller, independently executable tensor program for bilin18 that remains predictive on fresh and
shifted text, composes with other replacements, and supports selective causal edits. Literal storage is one cost;
it is not by itself interpretability. A proposed representation earns the stronger simplicity claim only when its
ordering predicts held-out OOD, extraction, intervention, or composition consequences.

Rung 431 established that a six-mode *readout* of native attention0 scores is near-lossless, while a half-price
linear-bilinear *generator* of those modes fails catastrophically. The previous explanation named the per-head RMS
normalizers as the missing nonlinear node. This review makes that node exact and changes the immediate decision:
test its causal necessity before fitting it. The relevant existing mathematics is partially symmetric CP/INDSCAL
and simultaneous diagonalization by congruence, not an ordinary matrix SVD.

## 1. Exact attention0 object

Let

- real token vocabulary size `V=50,257`;
- residual width `D=1,152`;
- heads `h=1,...,9`;
- head width `H=128`;
- score branches `b in {1,2}`;
- `x_t in R^1152`, the exact RMS-normalized state presented to attention0 for token `t`.

At layer 0, `x_t` depends only on token identity. This finite-input fact does not generalize to later attention
layers, whose states depend on context.

For query and key maps `W^q_bh,W^k_bh in R^(128 x 1152)`, define

`A^q_bh = (W^q_bh)^T W^q_bh / 128`,

`A^k_bh = (W^k_bh)^T W^k_bh / 128`,

`B_bh(delta) = (W^q_bh)^T R_delta W^k_bh / 128`,

where `R_delta` is the fixed rotary relative-offset operator. The exact branch kernel, including the implementation
epsilon, is

`kappa_bh(t,u,delta) = x_t^T B_bh(delta) x_u`

` / sqrt((x_t^T A^q_bh x_t + eps)(x_u^T A^k_bh x_u + eps))`.

The head's attention pattern is

`p_h(t,u,delta) = kappa_1h(t,u,delta) kappa_2h(t,u,delta)`,

and its output-projected payload is

`v_h(u) = O_h V_h x_u in R^1152`.

The complete attention0 write at query token `t` is the causal-prefix contraction

`a_0(t) = sum_(u in prefix) sum_h p_h(t,u,delta_tu) v_h(u)`.

### Contraction graph and degree

Each branch numerator is bilinear in `(x_t,x_u)`. Each squared denominator is quadratic in one token state and is
followed by reciprocal square root. The two branches are multiplied and then contracted with a linear payload.
Before division, the pattern numerator has degree two in the query state and degree two in the source state. The
physical kernel is therefore an algebraic/rational homogeneous function, not a bilinear tensor. Rung 431 tried to
move the sum over heads through four different token-dependent square-root denominators; that move is not an
identity.

The tied objects are the four dense projection maps, reused over all tokens and positions. The outputs that a valid
replacement must preserve are not merely the 36 norms: they are both branch kernels, their product, the full write,
the first-value broadcast, named block1 consumers, suffix logits/CE, and registered interventions.

## 2. Symmetries and legitimate coordinates

An orthogonal change of a head's private 128-dimensional coordinates leaves `W^T W` unchanged. Query/key changes
must additionally respect the rotary family and the cross-form `B(delta)` to preserve scores. Positive rescaling of
a whole slice cancels between numerator and RMS denominator up to the implementation epsilon. Therefore raw Q/K
rows and their norms are not individually canonical.

The joint family

`{A^q_bh, A^k_bh, B_bh(delta)}`

is the safer score-level object. Trace- or data-normalized `A` slices remove the near-scale gauge for structural
comparison, while physical evaluation restores the exact scale and epsilon. Any CP form still has sign, scale, and
permutation freedoms in its factors; only the reconstructed forms, factor spans, and held-out functions are
gauge-safe without further assumptions.

## 3. Literal price

The native four layer-0 Q/K maps contain

`4*1152*1152 = 5,308,416` scalar values.

The 36 exact denominator values on all real token types would contain

`50,257*36 = 1,809,252` scalar values,

but this is only a denominator table. It does not produce any numerator vector or rotary cross-form. A shared
rank-`R` squared-feature representation

`A_j approximately sum_(r=1)^R c_jr u_r u_r^T`, `c_jr >= 0`,

costs

`R*1152 + 36*R = 1,188R`

values for the forms; at `R=256` that is `304,128`. It computes `R` shared linear reads, squares them, mixes them
into 36 positive scalars, and takes reciprocal square roots. Numerator, rotary, decoder, precision, and index costs
remain additional. In particular, adding this number to rung 431's mode-map bill would not make rung 431 valid:
the numerator must remain head-indexed through normalization or be replaced by an equivalent rational kernel.

## 4. Exact literature match: partially symmetric CP / INDSCAL

Stack the 36 symmetric matrices into

`A in R^(36 x 1152 x 1152)`, with `A[j,i,k]=A[j,k,i]`.

The shared-square model above is a partially symmetric canonical polyadic decomposition, commonly called an
INDSCAL structure:

`A = sum_r c_r tensor u_r tensor u_r`.

This is an exact object-to-object mapping, not an analogy. De Lathauwer showed that, under a deterministic rank and
independence condition and with tensor rank below the greatest dimension, canonical components can be recovered
constructively by simultaneous matrix diagonalization by congruence
([SIAM J. Matrix Analysis, 2006](https://doi.org/10.1137/040608830)). Stegeman gives checkable uniqueness conditions
for CP with partial symmetry ([SIAM J. Matrix Analysis, 2011](https://doi.org/10.1137/100814615)); Domanov and De
Lathauwer derive generic uniqueness bounds specifically including INDSCAL
([paper](https://arxiv.org/abs/1405.6238)).

For the square, undercomplete case, take a positive-definite combination

`M = sum_j alpha_j A_j`

and whiten it. If `A_j=U diag(c_j) U^T` with an invertible common factor on the retained support, then

`C_j = M^(-1/2) A_j M^(-1/2)`

are simultaneously orthogonally diagonalizable and therefore commute. Conversely, commuting real symmetric
whitened slices have a common orthogonal eigenbasis. This gives an exact falsifier before optimization: large
whitened commutators reject a common-congruence dictionary on that support.

For an approximately commuting family, randomized joint diagonalization diagonalizes a random linear combination.
He and Kressner prove recovery error of order the distance to a commuting family, with high probability, under their
near-commuting assumptions ([SIAM J. Matrix Analysis, 2024](https://doi.org/10.1137/22M1541265)). Thus random-combination
eigendecomposition is a principled initialization and diagnostic, not merely another neural optimizer.

Semidefinite moment methods can decide and recover certain positive partially symmetric CP decompositions
([Ni and Li, 2022](https://doi.org/10.1287/moor.2021.1231)). Their lifted polynomial optimization is not a practical
next algorithm at dimension 1,152; it is useful mainly as an exact small-toy check.

## 5. Why the theorems do not yet solve attention0

Several required assumptions are unknown or false:

1. The known exact private-row decomposition has `36*128=4,608` squared atoms, above `D=1,152`; the desired shared
   rank is an empirical hypothesis, not a theorem.
2. We need a low-error approximation, while the clean uniqueness results concern an exact minimal decomposition and
   rank/independence or genericity conditions that have not been checked.
3. The physical score also contains the nonsymmetric rotary cross-forms `B_bh(delta)`. Recovering the `A` tensor does
   not recover the relative query/key orientation needed for any numerator.
4. The data requirement may be weaker than weight-space equality: at attention0 we only need the forms on 50,257
   token states. Distinct quadratic forms can agree on that finite set. Such a token-relative quotient may be useful
   here but does not identify the weights or generalize to continuous later-layer states.
5. Approximate CP factors can remain unstable even when the reconstructed norm functions are stable. The uniqueness
   hypotheses must be checked before naming atoms.
6. No Frobenius or commutator theorem implies low CE, composition, or selective edit collateral through the nonlinear
   suffix. Those remain physical consequence tests.

## 6. Executable consequence: causal necessity before factorization

The highest-information first test is cheaper than INDSCAL fitting. Keep every native numerator, rotary map,
payload, and downstream computation exact. Replace only each token-dependent denominator

`d_j(t)=sqrt(x_t^T A_j x_t+eps)`

by its uniform-FIT-token root-mean-square constant. If score products and CE barely change, the denominator cannot
explain rung 431 and the mathematical factorization is irrelevant. If the change is large, the nonlinear node is
causally necessary and worth compressing.

Rung 433 was preregistered and queued with this exact discriminator. It also measures the complete
`50,257 x 36` log-denominator matrix using token-ID-mod-5 FIT/SELECT, a rank-8 map-space hypothesis, split-map token
subspace transport, and independently token-permuted controls. Its strong reversal condition is constant-denominator
product error at most `.05` and CE damage at most `+.002` nat.

The first managed receipt is **instrument-invalid and publishes no content**. The exact-table score/product/write
replay errors were around `1e-13` and CE differed by `3.7e-9` nat, but the maximum suffix-logit difference was
`6.15e-5`, above the frozen `2e-5` bar. This is numerical amplification of a minute upstream replay difference, not
license to relax the threshold. The same-rung mechanical repair will preserve the raw/table comparisons and close
the exact endpoint explicitly before rescoring the unchanged constant arm. Until that lands, the apparent constant
arm and table-spectrum values are withheld from scientific conclusions.

## 7. If necessity holds: the next exact algorithm and falsifier

Only after a valid rung-433 pass on necessity:

1. Form all 36 `A_j` exactly and normalize their near-scale gauge.
2. Compute the rank and spectrum of the union of their ranges. This is a lower bound on the number of shared squared
   directions required by any undercomplete positive dictionary in weight space.
3. On a frozen token-covariance support, whiten a positive aggregate and measure pairwise commutator action with
   deterministic probes. Compare with independently conjugated/permuted slice controls that preserve each spectrum.
4. If commutator excess is small, run randomized joint diagonalization and measure off-diagonal energy, FIT/SELECT
   norm-function error, and split stability. If it is null-like, do not optimize a common-diagonal model.
5. Compare one shared nonnegative `R=256` squared-feature program with an equal-price private-form baseline and a
   permuted-token control. The shared program must improve held-out inverse-norm and physical score/product errors,
   not merely weight Frobenius error.
6. Reconnect any admitted denominator program to a head-indexed numerator representation and score full writes, CE,
   shifted text, extraction, selective removal, and composition. The already successful shared-QK program is the
   required physical baseline.

This sequence separates three hypotheses cleanly:

- **weight-global structure:** common congruence/INDSCAL works on the forms themselves;
- **token-manifold-only structure:** weight commutators fail but held-token norm functions compress;
- **head-private normalization:** both fail, so exact tables or shared-QK are preferable to invented common atoms.

## 8. Relation to learned simplicity

Even a small, stable denominator dictionary is only a screen. Its candidacy for useful simplicity is tested by the
new sealed-family protocol: a measure learned from solved circuits may guide the search, but the rule and resulting
program are frozen before unseen circuit families, data distributions, intervention kinds, and compositions are
opened. This normalizer route supplies a particularly good adversarial case: low table rank, low quadratic rank,
low bytes, and low CE may disagree. The measure that prospectively predicts which representation preserves causal
consequences—not the one with the smallest retrospective number—is the one the project should retain.

## Protected decision

Repair only rung 433's exact endpoint without changing its constant arm, token roles, structure test, or bars. Score
the receipt exactly. Do not fit common quadratic atoms from the invalid preview. If denominator necessity holds but
the frozen rank-8/shared-overlap hypothesis fails, report “necessary but more distributed than predicted” and run
the commutator/RJD falsifier before selecting a higher rank. If necessity fails, abandon the denominator route and
return to numerator parameterization or the existing shared-QK generator.
