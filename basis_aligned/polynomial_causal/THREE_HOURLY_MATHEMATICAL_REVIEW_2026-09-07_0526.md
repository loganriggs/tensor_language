# Three-hour mathematical tensor-network review — 2026-09-07 05:26 UTC

## Decision

The newly selective object is a family of **response projectors**, not a unique coordinate basis.
Its meaningful gauge is `Q -> QO` for `O in O(8)`, so stability must first compare projectors or
principal angles and then compare their exact weight-induced maps. Coordinate equality is the wrong
test. The immediate executable consequence is the already preregistered cross-fit weight-interface
audit, corrected before outcomes to the actual four attention/four MLP site split. A pass licenses
literal projected-weight interfaces; low principal overlap with equivalent downstream writes would
instead identify only an operational-equivalence class.

## Exact object

The fixed native-site set is

\[
S=\{\mathrm{MLP1,L8H1,L9H1,L9H4,MLP3,MLP4,MLP6,L11H3}\}.
\]

For an attention site, its patched response vector is the corresponding pre-`c_proj` head slice in
`R^128`; for an MLP site it is the post-`Down` response in `R^1152`. For split `h in {0,1}`, stack
all valid-token donor-minus-base responses from matched controls into `C_s^h` and from target rows
into `T_s^h`. Let `U_s^h in R^(p_s x 8)` be the top control right-singular vectors and define

\[
E_s^h=T_s^h(I-U_s^hU_s^{h\top}),\qquad
Q_s^h=\operatorname{topR}_{8}(E_s^h),\qquad P_s^h=Q_s^hQ_s^{h\top}.
\]

The intervention at every valid token is

\[
r_{s,\mathrm{base}}+(r_{s,\mathrm{donor}}-r_{s,\mathrm{base}})P_s^h.
\]

`p_s=128` for four heads and `p_s=1152` for four MLPs. The empirical basis payload is therefore
`4*128*8 + 4*1152*8 = 40,960` floats per split. The basis has an orthogonal gauge; the projector is
unique only when the eighth/ninth singular values are separated. Token order within the stacked
matrix is irrelevant to `P`. Head-label and site permutations only reorder independent factors.

The downstream model remains nonlinear through RMS normalization, attention softmax, subsequent
bilinear MLPs, and the output soft cap. The projector intervention is linear in the cached local
response, while each native ungated MLP is degree two in its normalized input. Its approximation
norm is not merely Frobenius response error: adoption requires the registered six-cell causal norm
(two tasks times behavior/two physical modes), signed behavior projection, and same-answer control
movement. The executed cross-fits give worst target residual `.0338/.0418`, behavior projection
`.825-.882`, and controls `.0048-.0306` of target scale.

## Exact weight contractions

For attention head `s`, let `W^O_s in R^(1152 x 128)` be its exact column block of `c_proj.weight`.
Then

\[
\Delta r_sP_sW_s^{O\top}
=(\Delta r_sQ_s)(W_s^OQ_s)^\top.
\]

Thus `W_s^O Q_s` is the literal residual-write map for the eight response coordinates. For MLP
site `s`, with hidden bilinear product
`z_s=(L_sx_s) odot (R_sx_s) in R^4608` and
`W_s^D in R^(1152 x 4608)`,

\[
(\Delta z_s W_s^{D\top})P_s
=(\Delta z_s)(W_s^{D\top}Q_s)Q_s^\top.
\]

`W_s^{D\top}Q_s in R^(4608 x 8)` states exactly which bilinear hidden combinations write the
identified response coordinates. These identities are algebraic and should close at float-relative
error near `1e-13`; failure indicates an implementation error, bias term, or boundary mismatch.
They do not yet compile attention patterns/value generation or the upstream normalized MLP input.

## Theorems and applicability

The top-right-singular-vector step is the empirical Frobenius-optimal rank-eight approximation by
Eckart--Young--Mirsky, but only for `E_s^h`; it provides no guarantee for nonlinear downstream
causal loss or control selectivity. The target-versus-control construction is adjacent to canonical
correlation: Hotelling's formulation finds linear variates maximizing correlation between two
variable sets ([Hotelling 1936](https://academic.oup.com/biomet/article-abstract/28/3-4/321/220073)).
Our matrices are unpaired target and control samples, so ordinary CCA does not apply object-for-
object. A generalized Rayleigh ratio of target covariance to regularized control covariance would
be a new estimator, not a theorem validating the present PCA sequence; `p >> n` also makes the
unregularized control covariance singular.

Wedin-style singular-subspace perturbation and Davis--Kahan sin-theta bounds relate covariance or
data perturbation to principal-angle error through a spectral gap. The original Wedin result is the
relevant SVD formulation, summarized in the primary SIAM record and later references
([Wedin 1972](https://epubs.siam.org/doi/10.1137/0709056)); the generalized singular-value literature
also explicitly treats paired matrix subspaces
([Sun 1983](https://epubs.siam.org/doi/pdf/10.1137/0720041)). The assumptions we would need are a
nontrivial eighth/ninth gap and a bound on the split covariance perturbation. We have neither yet;
therefore no population recoverability or uniqueness claim follows from the two behavioral passes.
Directly measuring all eight principal cosines and aligned weight maps is stronger evidence for
this finite cross-fit, but still not a population theorem.

Nonlinear sufficient-dimension-reduction theory defines a central class through conditional
independence and completeness, with spectral estimators under assumptions on the joint law
([Lee, Li, and Chiaromonte 2013](https://arxiv.org/abs/1304.0580)). Our response intervention is
deterministic, finite, and selected by causal outcomes rather than sampled from the required
predictor/response distribution, so those recovery results do not certify it. They do motivate the
fallback named in the hourly review: if linear projectors are unstable, define equivalence by the
full downstream causal-response map rather than by Euclidean activation coordinates.

## Executable consequence and route comparison

For each site and both split fits, the next audit will:

1. compute singular values of `Q_s^{0T}Q_s^1` and the mean squared principal cosine;
2. hash both projectors and their derived attention/MLP weight maps;
3. verify the two exact contractions above on the opposite target half with relative squared error
   at most `1e-10`;
4. optimally align the two gauges by orthogonal Procrustes and compare the induced weight maps.

This changes computational specification, within-module splitting, stable identification, and
weight-readable grouping. It is not a rank/compression sweep: rank eight is already frozen, and a
failure forbids treating the coordinates as a stable weight circuit. Empirical principal overlap
plus weight-map agreement is more informative and cheaper than introducing CCA/ridge choices now.
If it passes, fresh/OOD causal testing dominates further mathematics. If principal overlap fails but
held-out writes agree, the correct object is the quotient under downstream operational equivalence.
If both fail, redirect to a conditional/nonlinear gate rather than increase rank.

Next mathematical review due around **2026-09-07 08:26 UTC**.
