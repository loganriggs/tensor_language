# Three-hour mathematical review: compile the cancellation, not its factors

**Time:** 2026-08-29 00:57 UTC

**Status:** repository-specific mathematical review, sealed-artifact CPU diagnostic,
exact algebraic proof kernel, and prospective experimental design. No new GPU/model
outcome was opened. The independently owned `iso_cost_rank.py` job currently occupies
the GPU and was not touched.

## Conclusion

The new MLP-only suffix result changes the mathematical object. Late attention and
late MLPs cannot be simplified as independent interfaces: attention cancels most of
the early-prefix/late-MLP interaction. The next compiler should group one attention
write, the intervening residual/RMSNorm, and the following bilinear MLP as a typed
supernode.

This is unusually favorable algebraically. The supernode admits an **exact** five-term
polarization, including RMSNorm, rather than a Taylor approximation. The primary new
move is to use those physical terms as the shared-factorization grammar and ask whether
their downstream response has a smaller executable realization.

The safe CPU work completed in this review supplies two pieces of evidence:

1. an exact RMS-typed polarization implementation replays random asymmetric bilinear
   blocks to relative error below (2\times10^{-15}), survives gate-scale gauge replay
   term by term, and passes 5/5 proof tests;
2. the now-complete five-action intervention cube was transformed into an orthonormal
   Walsh/functional-ANOVA basis. Five transform/order/transfer tests pass, exact
   reconstruction and Parseval replay, 1,000 document bootstraps per role were run,
   and direct cross-role coefficient transfer was measured.

These are mathematical/diagnostic results, not whole-model ledger credit. Structural
coverage remains 36/36, certified storage reduction 5.3481%, strict named causal CE
recovery 10.923%, and final extraction/removal/OOD actions 0/68.

## Evidence that fixes the new target

The sealed de-alias result added the missing MLP3--8-only corner to the previous
early-prefix by suffix grid. Its receipt/result hashes are `51377280...` and
`dc8edddf...`. The frozen attention-invariance prediction failed on both roles:

- NRE 0.8813/0.8841 and (R^2=-4.5227/-3.9284);
- directed-transfer NRE 0.8909/0.8738;
- three-way error (Q) versus the true MLP-only interaction has cosine
  (-0.9983/-0.9982);
- scalar/affine correction restores only the mean scale, with (R^2\le0.010).

Thus the failure is stable structure, not insufficient data or a missing free bias.
At the same time, Claude's independent context-free frontier now has rank-256
second-class replications across disjoint builds; its current iso-cost test is a
different question and does not explain the contextual attention/MLP cancellation.

## Rank 1 — exact RMS-typed polarization and grouped block factorization

### Exact bilin18 object

At any native block, let (h) be the residual immediately before attention,
(a=A(\operatorname{rms}(h))) the native attention write, and

$$
Q(z)=D\big[(Lz)\odot(Rz)\big]+b
$$

the following bilinear MLP. Define

$$
\gamma(v)=\left(\frac1d\lVert v\rVert^2+\varepsilon\right)^{-1/2},
\qquad x=\gamma(h)h,
$$

$$
\alpha=\frac{\gamma(h+a)}{\gamma(h)},
\qquad \beta=\gamma(h+a).
$$

Then the exact post-attention normalized state is

$$
\operatorname{rms}(h+a)=\alpha x+\beta a.
$$

If (B(u,v)=D[(Lu)\odot(Rv)]), bilinearity gives

$$
Q(\operatorname{rms}(h+a))
=\alpha^2 B(x,x)
+\alpha\beta B(x,a)
+\alpha\beta B(a,x)
+\beta^2B(a,a)+b.
$$

This identity carries the exact RMS scalar. It needs both asymmetric cross terms;
bilin18 does not assume (L=R).

### Theorem or operational simplicity definition

The displayed equation is the polarization identity for the native bilinear map.
It turns the formerly opaque “attention×MLP cancellation” into five typed physical
outputs:

1. scaled base MLP quadratic;
2. residual-left/attention-right cross term;
3. attention-left/residual-right cross term;
4. attention-square term;
5. Down bias.

A grouped compiler is simpler only if a shared realization of these terms has fewer
stored values and executed projections/products than the native block **and** preserves
held-out downstream consequences. Expanding one native product bank into four full
banks would be more expensive and earns no simplicity credit. The opportunity is
simultaneous factorization after restricting the attention port to its empirical
causal subspace, sharing (L,R,D), and preserving the scalar ports ((\alpha,\beta)).

The typed terms are invariant under each physical gate's scale gauge
(L_g\mapsto tL_g, R_g\mapsto R_g/t), as the proof tests verify. Gate permutation is
also only a relabeling. Sparsity that disappears under these gauges is not admissible.

### Assumptions that may fail

- The exact identity does not imply any term is low-rank or sparse.
- The attention-write trajectory may occupy most residual directions.
- Computing all four quadratic/cross banks separately can cost more than native MLP;
  only a genuinely shared/restricted factorization can win.
- Low response rank on natural rows can miss rare contexts, finite edits, or OOD
  attention writes.
- Compressing each block independently may again fail composition; a shared port and
  downstream vector test are mandatory.

### Predicted consequence beyond reconstruction

A passing grouped realization predicts the effect of attention removal, MLP removal,
their joint replacement, and continuous attention-write edits from the same small
state. It should preserve the observed cancellation when blocks compose, identify
which typed term carries an editable circuit, and expose the exact executed
multiplication/storage cost. Because the decomposition is exact before truncation,
termwise approximation errors can be propagated and certified.

### Cheapest falsifier

On one fresh block (start at block 3), collect (h,a,\alpha,\beta), the four typed
term outputs, and registered residual/logit response vectors on source-disjoint rows.
Fit one shared downstream-weighted factorization at matched multiplication counts.
Reject immediately if:

- cross/attention-square response spectra have no stable knee under data doubling;
- a grouped rank/support does not transfer between document roles;
- it fails held-out single, joint, and continuous attention-write edits;
- or its actual projection/product/storage cost is not below native and below
  independently compressed attention plus MLP.

### CPU action executed

`rms_bilinear_polarization.py` implements the identity, typed terms, RMS scalars, and
relative replay error. Five proof tests cover exact replay, zero-attention reduction,
gate-scale gauge invariance, asymmetric cross-term necessity, and multiple RMS epsilons.
All pass; maximum registered synthetic relative error is below (2\times10^{-15}).

## Rank 2 — orthogonal action-Fourier complexity and sparse query design

### Exact bilin18 object

The completed response is a real set function on five physical intervention bits:

$$
f:\{0,1\}^5\to\mathbb R,
$$

with bits MLP0, MLP1, MLP2, attention3--8, and MLP3--8. Its 32 values exist for CE
and top-1 on both sealed document populations. The proposed extension is the
36-site physical program-action function, not a Fourier transform of weights or
activations.

### Theorem or operational simplicity definition

In the uniform Walsh basis,

$$
f(x)=\sum_{S\subseteq[5]}\widehat f(S)\chi_S(x),
\qquad
\mathbb E[(f-\mathbb Ef)^2]=\sum_{S\ne\varnothing}\widehat f(S)^2.
$$

Parseval makes squared coefficients an exact variance decomposition here. This fixes
the earlier warning that squared coefficients in the nonorthogonal anchored Möbius
basis were not “energy.” Keeping a support (K) has exact uniform-mask squared error

$$
\lVert f-f_K\rVert_{L^2}^2=\sum_{S\notin K}\widehat f(S)^2.
$$

Sparse Fourier set functions can, under their stated sparsity/support assumptions, be
recovered from far fewer queries than the full cube; Stobbe and Krause give an exact
random-query result for (k)-sparse set functions
([primary paper](https://proceedings.mlr.press/v22/stobbe12.html)). For this project,
query reduction is only credited after prospective held-mask prediction.

### New CPU result

For CE, the degree-energy fractions are extremely stable:

| role | degree 1 | degree 2 | degree 3 | degree 4 | degree 5 |
|---|---:|---:|---:|---:|---:|
| skip7000 | 71.119% | 26.866% | 1.814% | 0.194% | 0.0076% |
| skip11000 | 70.739% | 26.864% | 2.164% | 0.230% | 0.0027% |

The two role coefficient vectors correlate 0.99945. The attention×MLP-containing
coefficients alone carry 24.95%/24.63% of total CE variance. Eight terms reconstruct
the observed cube at NRE 0.187/0.203 and (R^2=0.965/0.959); direct source-value
transfer has NRE 0.188/0.204. At 16 terms, direct transfer NRE is 0.0877/0.0946.
Eight CE terms are selected in at least 80% of 1,000 document bootstraps on both roles.

Top-1 is materially more complex: degree at least 3 carries 12.65%/12.95%, versus
only 1.99%/2.40% for CE; eight-term NRE is 0.476/0.478 and 16-term NRE about 0.256.
The top-1 coefficient vectors nevertheless correlate 0.99952 across roles.

The central lesson is subtle. The **global CE action function** is mostly degree at
most two, but the small high-degree tail is exactly what controls the conditional
early-prefix/MLP circuit after the dominant main and pair effects are removed. A 95%
global-energy criterion would delete the scientific object we just learned matters.
Simplicity must therefore be conditional on the consequence or circuit being queried.

### Assumptions that may fail

- Uniform intervention masks are a declared experimental distribution, not natural
  language frequency.
- A five-bit low-degree spectrum need not scale to 36 action bits.
- Scalar CE can be low-degree while residual/logit response vectors require a larger
  shared support.
- Current support and term count were inspected after all 32 cells; they have no
  prospective credit.
- Fourier sparsity predicts the intervention response function, not a zero-native-call
  tensor implementation by itself.

### Consequence beyond reconstruction

If a support frozen at one cut predicts held masks and response coordinates at an
adjacent cut, it is an executable decision program for extraction/removal experiments:
one can forecast which combinations interact before running every model. Parseval
supplies an exact average-case error certificate under the registered mask measure,
and sparse-query theory supplies a principled experimental budget.

### Cheapest falsifier

At a fresh adjacent boundary, freeze degree at most two plus the eight structural term
types selected here. Measure a source-chosen random/cross set of masks and reserve
complete held masks on both roles. Reject if the frozen support fails CE NRE below
0.25, response-vector error below the independent-basis control, or direct role
transfer. Do not choose support after opening the adjacent cube.

Code/result: `dealiased_boolean_spectrum.py`, 5/5 tests, and
`dealiased_boolean_spectrum_results.json` (SHA256 `d3c5c95c...b624415f`).

## Rank 3 — local incremental-quadratic certificates for composed truncation

### Exact bilin18 object

This applies after a grouped block candidate exists. Let (e_k) be its residual-port
error and (T_k) the exact native continuation through RMSNorm, attention, and
bilinear MLP. We need a bound on

$$
\Delta_{k+1}=T_k(x_k+e_k)-T_k(x_k)
$$

over a registered ellipsoidal tube of natural states and edits, not a product of
global spectral norms.

### Theorem or operational definition

An incremental quadratic constraint certifies

$$
\begin{bmatrix}\delta x\\ \delta T\end{bmatrix}^{\!\top}
M_k
\begin{bmatrix}\delta x\\ \delta T\end{bmatrix}\ge0
$$

on a declared region. A quadratic storage function (V_k(e)=e^\top P_ke), combined
with compatible block constraints, yields an LMI certificate for contraction or a
finite-horizon error bound. Convex IQC programs can certify local Lipschitz,
one-sided-Lipschitz, invertibility, and contraction properties of neural maps
([Hashemi, Ruths, and Fazlyab 2021](https://proceedings.mlr.press/v144/hashemi21a.html)).

Here the exact polarization should be used to derive tighter bilinear cross/remainder
bounds, while the RMS scale is bounded on the measured tube. This is a certificate
layer, not a representation learner.

### Assumptions that may fail

- The needed ellipsoid may cover too little of the OOD/edit domain.
- Attention's cross-position Jacobian and near-small RMS denominators may make the LMI
  infeasible or vacuous.
- A quadratic storage function can miss a valid nonquadratic certificate.
- Empirical tube construction is not itself a global guarantee; coverage must be
  explicit.

### Consequence beyond reconstruction

A nonvacuous certificate turns local truncation errors into a bound on downstream
residual/logit change, enabling certified component removal and composition. It also
provides a principled rejection when two individually accurate simplifications can
amplify each other—one of the project's recurring failures.

### Cheapest falsifier

For one grouped block and one held edit family, solve the local LMI using exact RMS
and bilinear sector bounds. Compare the certified bound with observed residual/logit
errors. Reject the route if the bound is infeasible, exceeds measured error by more
than 10×, or fails any held trajectory in the declared tube. Do not attempt an
18-block global certificate first.

## Pruning ledger

| Mathematical family | Decision after current evidence |
|---|---|
| Tensor rank / HOSVD / CP | Do not refactor the same folded third-order MLP tensor. Its coefficient HOSVD is dense and scalar gauge balancing cannot change its spectrum. Use tensor structure only inside the exact typed supernode and price executed contractions. |
| Arithmetic-circuit rank / algebraic complexity | Promote executed projection/product count as a hard price for rank 1. The five-term identity is not automatically simpler; sharing or restriction must reduce the circuit. Border rank and a dense Tucker core remain uninterpretable and potentially ill-conditioned. |
| Simultaneous factorization / shared dictionaries | Promote only across the five typed polarization ports under a downstream response metric. Generic shared dictionaries on raw layer coordinates repeat the failed local-composition problem. |
| Polynomial invariant theory / gauge quotients | Keep as a validity constraint. Typed terms pass the native gate scale gauge term by term; gate permutations and residual basis changes must not create new “atoms.” Computing a full invariant ring is not the next experiment. |
| System identification / minimal realizations | Defer generic local/tangent realization: independent tangent frames already failed. Bilinear/generalized Hankel theory does support finite realization when the appropriate physical edit-series Hankel has finite rank ([Arbib and Manes 1980](https://doi.org/10.1016/0022-0000(80)90012-4)), but typed supernode port data must exist first. |
| Hankel / automata | Keep token-splice Hankel pruned because it was strongly OOD. A future action-sequence Hankel is a different object, but ranks below the typed port dimension must be earned prospectively. |
| MDL / prequential coding | Use only after two grouped executable candidates pass causal tests. Charge typed support, coefficient precision, decoder, scalar ports, and exceptions; description length cannot validate semantics by itself. |
| Causal abstraction / bisimulation | The grouped supernode is now the candidate abstraction. It earns equivalence only through interchange, joint replacement, continuous write edits, and closure under the next block—not by similar coefficients. |
| Information bottleneck | Still pruned as a representation finder. Mutual information depends on arbitrary noise/quantization and need not preserve rare edits or executable arithmetic. |
| Sparse program synthesis | Restrict to the physical typed grammar and frozen action-Fourier support. Unconstrained synthesis over the revealed cube is post-outcome search. |
| Approximation certificates | Parseval is exact for average error over the declared action cube; local IQCs are the best new route to composition bounds. Neither licenses global natural-language/OOD claims without coverage tests. |
| Hierarchical/DAG codes | Successive-refinement regret remains the right test but is not repeated here. It requires vector-valued downstream distortion from the grouped interface; current scalar CE is insufficient. |

## Ranked decision

1. **Exact RMS-typed polarization plus grouped downstream factorization** is the best
   new move because it directly explains the newly observed cancellation, respects the
   tensor/polynomial architecture, crosses RMSNorm exactly, and defines an executable
   price and edit interface.
2. **Orthogonal action-Fourier complexity** is the best experimental-design and
   consequence-prediction move. It already reveals a stable low-degree CE structure,
   while proving why global energy alone can erase a conditional circuit.
3. **Local incremental-quadratic composition certificates** are the best certification
   move once a grouped candidate exists; they target the exact failure mode in which
   good local replacements amplify or compensate downstream.

The next GPU experiment should not be another independent MLP rank sweep. After the
current independently owned iso-cost job finishes, the useful next collection is one
fresh block's typed polarization outputs and downstream response vectors, source-split
and priced against native arithmetic.
