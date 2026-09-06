# Three-hour mathematical tensor-network review — 2026-09-06 20:26 UTC

## Decision

Treat the confirmed H3-Q8 interface as a finite causal realization with an eight-dimensional
internal state, then identify its *reachable upstream row space* and *observable downstream
column space* separately. The recent subject-only failure says source coordinates are not stable
semantic units; it does not refute the stable observable subspace. The next experiment should
construct the complete source-region x attention-factor x Q8 tensor on v10. After that, cross-task
reuse should be tested as a block-Hankel rank and held-out matrix-completion question, not by
comparing independently fitted DAS axes.

## Exact current object

The checkpoint model has residual width `d=1152`, `h=9` attention heads of width `d_h=128`,
18 blocks, and vocabulary readout width 50,257 (with 50,304 admitted embedding rows). For row
`i`, H3 at block 11 and answer query `t_i` has pre-output-projection value

\[
 z_i=\sum_{s\le t_i}P_{i,s}V_{i,s}\in\mathbb R^{128},\qquad
 c_i=Q^\top\Delta z_i\in\mathbb R^8,
\]

where `Q in R^{128 x 8}` is the frozen orthonormal tensor-derived basis. For source group `g`
and attention factor `f in {pattern, value, interaction}`, define

\[
 A_{i,g,f,k}=q_k^\top\Delta z_{i,g,f}.
\]

The exact source-factor identity is

\[
 \Delta z_i=\sum_g\left[(P_1-P_0)V_0+P_0(V_1-V_0)
 +(P_1-P_0)(V_1-V_0)\right]_g.
\]

At the value branch, `V_1-V_0` is itself a literal weight contraction:

\[
 \Delta V_{i,s,3}=(1-\lambda_{11})W^{V}_{11,3}
 \left(\operatorname{rms}(x^1_{i,s})-\operatorname{rms}(x^0_{i,s})\right).
\]

The downstream residual modes are

\[
 M=\alpha\,W^O_{11}E_3Q\in\mathbb R^{1152\times8},\qquad
 \alpha=\prod_{\ell=12}^{17}\lambda_{\ell,0}=1.5136314812480123,
\]

where `E_3` embeds H3 coordinates into the concatenated head vector. For native final residual
`x_i`, answer `a_i`, and foil `b_i`, the analytic reader is

\[
 r_i=M^\top\nabla_x\left[
 30\tanh(w_{a_i}^\top\operatorname{rms}(x)/30)
 -30\tanh(w_{b_i}^\top\operatorname{rms}(x)/30)
 \right]_{x=x_i}\in\mathbb R^8.
\]

The compiled first-order causal effect is `c_i^T r_i`. Empirically it predicts the exact finite
residual intervention at cosine 0.99999985/0.99999991 and relative RMSE 0.056%/0.047%.

The contraction graph is therefore

`L8H1 cue write -> source-position residual states -> rms -> W_V11,H3 -> P11,H3 -> Q8
coordinates -> W_O11 -> residual skip -> final rms -> unembedding/tanh`.

The fixed-pattern value part is linear in normalized source-state changes; the writer-reader
interface is bilinear as `c^T r`. The full native network is not a polynomial: RMS normalization,
softmax/normalized attention, and the final tanh are analytic or rational operations. QK score
numerators are quadratic (and quartic for product/squared variants), while bilinear MLPs are
quadratic before normalization. Consequently, polynomial tensor-rank theorems apply exactly only
to declared frozen-pattern or local-Jacobian restrictions, not to the whole transformer.

## Symmetry, gauge, inputs, norms, and price

For any orthogonal `G in O(8)`, `Q -> QG`, `c_i -> G^T c_i`, `M -> MG`, and `r_i -> G^T r_i`
leave `c_i^T r_i` invariant. With nonorthogonal factorizations the gauge expands to `GL(8)` with
contragredient writer/reader transforms. Individual Q8 coordinates are therefore not identified
unless a canonical reachability/observability gauge is imposed; the eight-dimensional causal
subspace and its contracted effects are identified.

Current allowed inputs are the sealed v8-v10 temporal construction families and their declared
base/donor interventions. General text and arbitrary task suffixes remain unproved. Outputs to
preserve are full-vocabulary causal effects when selecting the subspace, answer/foil causal
margins for the compiled reader, exact complement interventions, and unrelated P/C controls.
Norms are centered full-vocabulary KL/effect cosine for selection and per-row causal margin
cosine/relative RMSE for the executable path.

Literal persistent state is eight floats per active row. Storing Q costs 1,024 floats; the 9,216
residual-mode floats need not be stored because they are derived from native `W_O`. Reader vectors
need not be stored because they are derived from final weights and the live residual. The path
uses approximately `sources x 128 x 8` value contractions plus `1152 x 8` output contractions,
versus executing the native suffix. This is already simpler locally, but whole-model adoption
requires source completeness, cross-task reuse, composition, and a full price ledger.

## Realization-theory mapping

Construct a causal response matrix `H` whose rows index reachable upstream interventions or
prefix/source states `u`, whose columns index downstream suffix/readout conditions `v`, and whose
entries are exact centered causal effects:

\[
 H_{u,v}=F(uv)-F_0(uv).
\]

If one shared interface is valid, `H=C R^T` with row coordinates `C_u=c(u)` and observable
reader coordinates `R_v=r(v)`, hence `rank(H)<=8`. A full-rank factorization is unique only up to
the `GL(8)` gauge. If row and column spans are both complete, rank is also a lower bound: no exact
linear interface with fewer states can reproduce the same intervention table.

This maps directly to Petreczky's bilinear switched-system realization theorem: for a family with
a generalized Fliess-series expansion, finite Hankel rank is equivalent to a finite-dimensional
bilinear realization, minimal dimension equals Hankel rank when the realization is reachable and
observable, and minimal realizations are unique up to similarity. The paper also covers regular-
language switching constraints and gives a construction from Hankel columns
([Petreczky 2011, Theorems 2.3, 2.6, 2.7](https://www.numdam.org/item/10.1051/cocv/2010015.pdf)).

The exact theorem does **not** yet solve Theseus: our empirical `H` is finite; token-conditioned
attention and RMS/tanh do not presently have a proved Fliess-series realization over the declared
alphabet; available interventions do not establish arbitrary reachability; and the row-specific
answer/foil reader changes with `v`. What survives exactly on the measured family is the finite-
matrix rank lower bound and gauge statement. The infinite-Hankel minimality claim remains an
unmet assumption, not a conclusion.

Weighted-automaton singular-value canonical forms similarly balance forward and backward spaces
and provide approximate minimization bounds for rational series
([Balle et al. 2015](https://arxiv.org/abs/1501.06841)). Applied here, an SVD of the *causal
upstream-by-downstream matrix*, rather than activation covariance, supplies a canonical gauge and
orders states by joint reachability/observability. The rational-series assumption is again
unproved, so the finite causal matrix is the legal restriction.

## Tensor-train mapping

After the source atlas, form

\[
 \mathcal A[\text{construction},\text{source group},\text{factor},
             \text{Q8 mode},\text{readout task}].
\]

Oseledets' TT theorem states that ranks of successive unfoldings bound achievable exact TT ranks,
and TT-SVD constructs such a representation; truncation error is controlled by discarded
unfolding singular values
([Oseledets 2011, Theorem 2.1 and TT-SVD](https://epubs.siam.org/doi/10.1137/090752286)).
This mapping is exact for the measured tensor. It does not confer causal meaning on factors or
guarantee transfer to unmeasured text. TT compression is therefore a post-identification storage
tool, not the next discovery step.

## Executable consequence and opposing predictions

1. Run the already licensed complete source-region x `{pattern,value,interaction}` atlas in Q8
   and causal-margin space on v10. Exact Möbius/factor closure must hold. This identifies which
   source-factor union is reachable under each construction. Kill the simple shared-writer account
   if no common union retains at least 90% of Q8 response and behavior across A1/A2.
2. Using independently capable temporal/number tasks, build a finite causal matrix with upstream
   interventions as rows and downstream answer/foil or full-logit probes as columns. Freeze a FIT
   block, obtain its SVD gauge, and predict a held-out construction x task block without refitting.
   Shared-state prediction: numerical rank <=8, stable row/column subspaces, and low held-out block
   error. Task-specific-coordinate prediction: apparent rank grows with tasks or held-out cross
   blocks fail despite within-task fits.
3. Only if (2) succeeds, use balanced singular coordinates to test exact state deletion and joint
   cross-task composition. The smallest retained dimension must pass causal behavior and complement
   gates; singular values alone cannot license deletion.

The immediate source atlas has higher information value than constructing a broad Hankel table
now, because the failed subject-only assumption means the upstream row coordinate is not yet
correctly specified. Once it closes, the finite Hankel/SVD test dominates another DAS fit: it
directly measures shared realization dimension, fixes the gauge, predicts held-out cross blocks,
and provides a certified lower bound on the measured family.

Next mathematical review due around **2026-09-06 23:26 UTC**.
