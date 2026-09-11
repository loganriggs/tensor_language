# Exact routing-contrast mode factorization

The full centered source/value pullback has43.3904%double-antisymmetric
coefficient energy, essentially matching source-permuted43.4491%. Do not call
that learned circuit structure. Instead test whether that whole component
shares a few routing contrasts across outputs and source features.

Its head-antisymmetric mode has choose(9,2)=36 coordinates. Form the exact36×36
Gram of their complete output/source coefficient functions, with no output-rank
truncation. Unit head-wedge basis is(E_hk-E_kh)/sqrt2; the corresponding source
coefficient is sqrt2*skew(B_hk), so the Gram trace must equal the already-measured
full antisymmetric energy. The hidden-product contraction is controlled against
a dense unfolding/SVD on a toy problem.

Eigendecomposition exactly solves this routing-mode low-rank approximation,
leaving arbitrary full source/output functions in each mode. It is not a CP
factorization or automatically a cheaper circuit. Each eigenvector is a9×9
antisymmetric head matrix. Its best single alternating pair (rank2) capture is
the leading two squared singular values divided by total squared singular values.
This allows linear combinations across native heads, not only native head pairs.

- A: dense covariance/SVD controls pass, native trace matches full-source
  antisymmetric energy and covariance eigen-replay, relative errors<=1e-10;
  no negative eigenvalue fraction beyond1e-10.
- B: top4 routing-contrast modes retain>=50%of the full antisymmetric coefficient
  energy. This is an exact optimum in this unfolding, not an optimization timeout.
- C: each leading4head matrix admits one alternating pair capturing>=90%of its
  own squared coefficient norm. Preserve B/C as separate claims.

FP64, managed GPU,900second alarm; no body forwards or data. Up to~14GB of
temporary head-to-head hidden Grams; persist36×36Gram/eigenvectors and compact
head matrices in JSON. Native QK/value/U/L/R/D, all other tensor components and
normalizations remain charged. Do not assume singular modes are semantic units.

A failed B excludes a four-dimensional linear routing-contrast description
at the50%energy bar for this full coefficient tensor. It does not exclude
nonlinear/shared source-factor programs, different physical weighting, or
useful low-energy circuits. A passed C identifies simple head combinations
only algebraically. The multisource contrast is a_p^T R a_q paired with source
contrasts, not a nonzero antisymmetric quadratic in a single routing vector.
