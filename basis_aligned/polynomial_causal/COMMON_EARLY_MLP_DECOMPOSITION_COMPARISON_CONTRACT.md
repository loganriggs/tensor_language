# Common comparison contract for early-MLP tensor decompositions

Date: 2026-08-28

Status: prospective mathematical contract only. This document authorizes no row
role, model forward, fitting run, finite intervention, result interpretation, or
promotion. It is separate from the active MLP1 and MLP2 namespaces.

## 1. Common physical object

At site \(s\in\{0,1,2\}\), between input RMS normalization and residual addition,
write the bias-separated quadratic map as

\[
 F_s(x)_o-b_{s,o}
 =\sum_{g=1}^{h_s}D^{(s)}_{og}(L^{(s)}_g x)(R^{(s)}_g x)
 =\sum_{ij}T^{(s)}_{oij}x_i x_j ,
\]

where

\[
 T^{(s)}_{oij}=\sum_gD^{(s)}_{og}
 \frac{L^{(s)}_{gi}R^{(s)}_{gj}+R^{(s)}_{gi}L^{(s)}_{gj}}2
 \in Y_s\otimes\operatorname{Sym}^2(X_s^*).
\]

Every decomposition must target this folded, partially symmetric physical tensor,
not the ordered tensor \(D\otimes L\otimes R\). Antisymmetric input coefficients are
functionally null on \(x\otimes x\). Bias is exact and separately priced. RMSNorm,
residual mixing, attention, and the suffix remain explicit program nodes.

Keep two evaluation modes distinct:

- **Teacher-input local:** every candidate receives the native \(x_s\) and is scored
  against the same \(F_s\). This diagnoses the local decomposition only.
- **Live deployed:** upstream candidates generate the state, no replaced native MLP
  is called, and suffix/final behavior is scored. Only this tests composition.

No teacher-input result earns deployed or compositional credit.

## 2. Five non-substitutable notions

### 2.1 Scalar gate balancing

A displayed product gate has reciprocal gauge

\[
 (\ell_g,r_g,d_g)\mapsto
 (a_g\ell_g,b_g r_g,d_g/(a_gb_g)),\qquad a_gb_g\ne0.
\]

For nonzero factors, minimizing their total squared norm balances all three squared
contributions at

\[
 t_g=(\|\ell_g\|_2^2\|r_g\|_2^2\|d_g\|_2^2)^{1/3}.
\]

Thus \(|a_g|^2=t_g/\|\ell_g\|^2\) and
\(|b_g|^2=t_g/\|r_g\|^2\). Deterministic joint signs, left/right swap, and gate
permutation conventions are still required. Zero terms are removed, not divided by.

This conditions one given product decomposition. It does not choose a unique
decomposition of \(T_s\), lower gate count, or identify semantics. The exact folded
tensor is already invariant to scalar gauges and permutations. Scalar-balance then
HOSVD of the exact folded tensor must therefore reproduce the same HOSVD subspaces
and spectra up to numerical tolerance.

### 2.2 Full folded-tensor/network gauge

An invertible change on a contracted latent bond is a gauge only when its inverse is
installed on every adjacent map. Under physical coordinate changes \(x'=Gx\),
\(y'=Hy\),

\[
 T'=H\,T\times_2G^{-1}\times_3G^{-1}.
\]

This is not a free standalone rewrite when \(x,y\) are residual-stream coordinates:
all transports and consumers belong in the executable graph and price. A full
network gauge is a set of inverse pairs on declared internal bonds that leaves the
complete physical program unchanged.

Exact multilinear ranks are invariant under invertible basis changes. Euclidean
singular values, truncation errors, sparsity, coefficient norms, and quantized bytes
are generally invariant only under orthogonal gauges. A nonorthogonal comparison
must transform the declared metrics, whiten them, or use a prospectively frozen
minimum-norm orbit representative while pricing every transport.

A valid gauge test replays the physical polynomial, deployed suffix response,
selection, and total price. Replaying only a local tensor while omitting neighbors is
not a full-gauge test.

### 2.3 HOSVD multilinear rank

For \(T\in Y\otimes X_1^*\otimes X_2^*\),

\[
 (r_o,r_1,r_2)=
 (\operatorname{rank}T_{[Y]},
  \operatorname{rank}T_{[X_1]},
  \operatorname{rank}T_{[X_2]}).
\]

Partial input symmetry implies \(r_1=r_2\) for the exact tensor under a common
tolerance. HOSVD supplies orthogonal mode subspaces and a core

\[
 T\approx G\times_1U_o\times_2U_1\times_3U_2 .
\]

These are mode-subspace dimensions, not multiplication counts. A dense core with
separate input factors may require \(r_1r_2\) cross-products; a shared symmetric
basis may require \(r_x(r_x+1)/2\) independent monomials. The core and execution
must be explicit and priced.

Approximate HOSVD also needs a norm. Coefficient Frobenius error,
activation-weighted write error, Fisher response error, suffix KL, and CE are
different currencies.

### 2.4 CP/product-gate and block-term rank

The executable product rank in the repository grammar is

\[
 r_{\rm prod}(T)=\min\left\{q:
 T=\sum_{a=1}^q c_a\otimes
 \operatorname{sym}(u_a\otimes v_a)\right\}.
\]

It counts shared products \((u_a^Tx)(v_a^Tx)\), each decoded to every output by
\(c_a\). It is not unrestricted CP rank, symmetric Waring rank, an unfolding rank,
or the sum of scalar-output ranks. The native factorization is an explicit
\(h_s\)-gate upper bound. Output/input flattenings and contracted scalar inertia give
the gauge-invariant lower bounds

\[
 q\ge \operatorname{rank}T_{[Y]},\qquad
 q\ge \left\lceil\frac{\operatorname{rank}T_{[X]}}2\right\rceil,\qquad
 q\ge \max_{\lambda\in\Lambda}
 \max\{p(\lambda\mathbin{\lrcorner}T),n(\lambda\mathbin{\lrcorner}T)\},
\]

where \(p,n\) are positive/negative inertia and \(\Lambda\) is the frozen bank of
output contractions. Equality between a constructive upper bound and a valid lower
bound is required before calling a gate count minimal.

A block-term program is a sum of Tucker-like blocks. Block count is not execution
cost. Each block pays its factors, core, independent core monomials, decoder, and
routing. A dense block pays its dense product count unless an explicit CP or sparse
core supplies a cheaper executable certificate.

Approximate CP may exhibit border-rank degeneracy: error improves while factors
diverge and cancel, with no best finite decomposition. Every approximate candidate
therefore reports factor norms, conditioning, perturbation behavior, and actual
precision. A divergent limit earns no finite-program credit.

### 2.5 Down rank

For \(D\in\mathbb R^{m\times h}\), \(D=UV\) gives

\[
 F(x)=U(Vh(x))+b,\qquad h_g(x)=(\ell_g^Tx)(r_g^Tx).
\]

This compresses final mixing but still computes all \(h\) native products and retains
both factor banks. Down rank is neither HOSVD rank nor product rank. It is a strong
write-map/storage baseline and a weak arithmetic simplification. Cancellation may
make the output unfolding of \(T\) lower rank than \(D\); low Down rank alone says
nothing about the quadratic feature bank.

## 3. Minimum-norm then HOSVD

This pipeline is admissible only if:

1. **Correct orbit:** every transformation is a function-preserving gauge of the
   complete declared graph, including neighboring transports.
2. **Finite minimum:** the orbit contains a finite minimum; null-cone and
   orbit-closure limits are detected.
3. **Declared metric:** the norm is tied to physical coordinates or frozen
   input/output covariance metrics.
4. **Optimizer honesty:** a minimum is certified or labeled as the output of a fixed
   optimizer/restart budget; independent starts agree.
5. **Residual compact gauge:** repeated singular values license only an invariant
   subspace, not stable individual axes.
6. **Complete object:** symmetrization, bias, precision, and linear terms are handled;
   nonpolynomial boundaries are not silently folded into \(T\).

Under these assumptions, minimum-norm balancing improves conditioning and HOSVD is a
reproducible Tucker diagnostic. Truncated HOSVD gives a coefficient-norm
multilinear-rank approximation guarantee, not a best CP/gate program.

The pipeline fails as a simplicity claim if:

- scalar gate balancing is called a canonical tensor decomposition;
- truncation curves move under a registered full-gauge replay;
- Tucker ranks are priced without core products and decoder;
- minimum norm is equated with product rank, sparsity, MDL, semantics, or editability;
- coefficient error is promoted to suffix/CE fidelity without sensitivity bounds;
- approximation depends on divergent cancellation or increasing precision; or
- any producer, basis, router, index table, or inverse transport is omitted from price.

## 4. Matched executable cost

Every candidate reports a vector

\[
 C=(B_{\rm standalone},B_{\rm amortized},P_{\rm stored},
 M_{\rm linear},M_{\rm product},\text{depth},\text{memory},
 \text{precision},\kappa).
\]

Primary comparisons use standalone serialized bytes at executed precision and product
multiplications per token. Parameter count is a same-precision structural proxy, not
literal MDL. Gauge-quotient dimension is separate and cannot be deducted from bytes
unless a canonical codec actually removes it.

For input \(n\), output \(m\), and \(q\) products, a dense product program stores

\[
 P_{\rm CP}(q)=q(2n+m)+m
\]

reals plus indices/metadata and executes \(q\) product gates. At \(n=m=1152\), this
is \(3456q+1152\).

For a retained \(h\)-product bank with rank-\(r_D\) factored Down,

\[
 P_D(h,r_D)=2nh+r_Dh+mr_D+m,
\]

and execution still pays \(h\) product gates. If native \(L,R\) are an independently
admitted shared library, report the amortized-new count \(r_Dh+mr_D+m\) separately;
never substitute it for standalone cost.

For dense Tucker ranks \((r_o,r_1,r_2)\), a direct stored upper bound is

\[
 mr_o+nr_1+nr_2+r_o r_1 r_2+m,
\]

plus transports and metadata. Direct execution pays \(r_1r_2\) projected products,
or \(r_x(r_x+1)/2\) only when a shared symmetric basis and core enforce it. Block
prices sum complete factor/core costs; shared factors count once only as actual common
DAG nodes.

Matched cost means one prospectively frozen rule:

- coordinatewise no-greater cost in every promoted coordinate;
- equal standalone bytes within a fixed tolerance and no greater product count; or
- a frozen scalarization of the full vector.

Matching decoder columns while omitting producers is forbidden. Standalone,
amortized, and runtime Pareto fronts remain separate. Claimed replacements require
exactly zero native calls.

## 5. Common within-site assay

At each site compare factor-complete versions of:

1. native gate subset/product-response selection;
2. Down-only SVD;
3. HOSVD/Tucker with dense or explicitly sparse core;
4. CP/product-gate or block-term execution;
5. matched random/deranged subspaces or factors; and
6. a generic linear/quadratic baseline at the same standalone cost.

Selection and coefficients use fit rows only. Validation cannot change rank, support,
gauge, basis, core sparsity, precision, or cost rung. Score common rows and
denominators in separate currencies:

- weights-only coefficient error;
- native-input local write error;
- frozen-suffix causal/Fisher response error; and
- live deployed suffix KL and final CE.

Only the latter two support causal or compositional language. A local frontier point
must beat matched structured and random controls, pass gauge/perturbation replay, and
respect the registered pointwise collateral margin.

## 6. Eight-cell composition and Möbius interactions

Let \(N_s\) be baseline and \(P_s\) an independently fitted, frozen, zero-native-call
candidate at site \(s\). Evaluate all eight subsets on identical documents, positions,
targets, and suffix realization:

\[
 v(S),\qquad S\subseteq\{0,1,2\}.
\]

Sites in \(S\) use \(P_s\); others use \(N_s\). Lower \(v\) is better. Transform CE,
KL, and causal-response distortion separately.

Define

\[
 m(T)=\sum_{U\subseteq T}(-1)^{|T|-|U|}v(U).
\]

Then

\[
\begin{aligned}
m_i&=v_i-v_\varnothing,\\
m_{ij}&=v_{ij}-v_i-v_j+v_\varnothing,\\
m_{012}&=v_{012}-v_{01}-v_{02}-v_{12}
          +v_0+v_1+v_2-v_\varnothing.
\end{aligned}
\]

Closure requires \(v(S)=\sum_{T\subseteq S}m(T)\). Positive pair/triple terms are
superadditive harm in a loss currency; negative terms are compensation. Do not clip.

Report:

1. singleton additive error
   \[
   e_{\rm add}=v_{012}-(v_\varnothing+m_0+m_1+m_2)
   =m_{01}+m_{02}+m_{12}+m_{012};
   \]
2. pair-informed triple residual \(e_3=m_{012}\);
3. interaction mass
   \[
   \rho=\frac{|m_{01}|+|m_{02}|+|m_{12}|+|m_{012}|}
   {\max(|v_{012}-v_\varnothing|,\epsilon_{\rm denom})},
   \]
   alongside raw signed terms and a frozen denominator floor; and
4. conditional increments
   \[
   \delta_i(B)=v(B\cup\{i\})-v(B),
   \quad B\subseteq\{0,1,2\}\setminus\{i\},
   \]
   as no-free-rider/retention checks in every deployed background.

Independent simplicity composes only if the all-candidate arm meets the absolute
suffix/CE gate, every required conditional increment respects its harm margin, and
the registered additive or pair-informed prediction interval contains the joint
outcome. Good \(P_0P_1P_2\) with large negative interactions is a jointly compensating
behavioral program, not evidence of independently equivalent interfaces. Failed joint
behavior with good singletons localizes incompatibility; it does not falsify each local
approximation.

Use one shared document-cluster bootstrap across all cells, metrics, strata, and
contrasts. Every replicate recomputes pooled numerators/denominators, Möbius terms,
nonlinear maxima, \(\rho\), and conditional increments. Never subtract independently
bootstrapped cell intervals. Familywise bands cover all promotive interactions and
no-free-rider contrasts.

## 7. Cheapest falsifiers

### A. Weights-only algebra and price

For each site, without natural rows:

1. verify folded-factor evaluation equals native \(F_s(x)-b_s\) on deterministic
   random vectors;
2. compute implicit unfolding Gram/HOSVD spectra with independent numerical sketches
   and exact gauge/orthogonal replays;
3. report Down spectra, Tucker spectra, product-rank lower bounds, and native upper
   bounds at common tolerance; and
4. price dense Tucker, CP/block-term, Down, and native-support programs completely.

Stop a path if no stable knee exists, complete price does not beat the simpler baseline
at the same tolerance, or the knee moves under gauge. This is the cheapest screen.

### B. Small activation-weighted transfer

On disjoint fit/validation documents, fit then freeze the candidate decomposition or
sparse core. Compare local write and suffix-response error against matched Down,
random, and generic controls with nested fit sizes and two probe halves. Stop if
support/subspace stability fails, error does not improve with data, or a simpler
matched baseline dominates.

### C. Four-cell incompatible-edge screen

Before the full cube, cross the two least certain adjacent candidates in
\(NN,PN,NP,PP\). The interaction \(v_{PP}-v_{PN}-v_{NP}+v_{NN}\) cheaply detects an
incompatible edge. It may prune a pair but cannot establish three-way composition;
promotion still needs a fresh complete cube.

## 8. Data scaling

Weights-only identities and algebraic bounds need no corpus. Distributional and
causal claims treat source document, not token position or probe, as the independent
unit.

A 16--32 fit / 16--32 validation document assay is only a cheap falsifier. Survivors
need a prospectively fixed nested doubling ladder such as \(32,64,128,256\) documents
with unchanged compiler budget and rungs. At every doubling report:

- support/subspace distances and principal angles;
- spectrum/effective-rank drift;
- local, causal-response, KL, and CE estimates;
- factor norms and condition numbers; and
- matched-control margins and interval widths.

Promotion requires an untouched final wave and a second corpus/domain. Freeze the
maximum sample before outcomes. Given target margin \(\delta\), document-level
contrast deviation \(\sigma\), family size \(J\), type-I level \(\alpha\), and power
\(1-\beta\) from an external/nonpromotive pilot, a conservative planning rule is

\[
 N\ge\left[
 \frac{(z_{1-\alpha/(2J)}+z_{1-\beta})\sigma}{\delta}
 \right]^2,
\]

rounded up to whole documents and batch size. Otherwise run the full frozen doubling
ladder; never stop at first significance.

Final CE claims need hundreds of thousands of scored tokens across hundreds of
documents, document-cluster inference, and a distinct domain. Power cube interactions
on document-level Möbius or conditional-increment contrasts, whose variance may exceed
single-cell loss variance. Extra probes reduce conditional Monte Carlo error; they do
not add natural-text sampling units.

## 9. Permitted conclusions

- Stable Down rank: compressible output mixing while retaining all products.
- Stable HOSVD ranks: compressible mode subspaces in the declared metric, not a small
  arithmetic program.
- Admitted CP/block-term point: an executable product-rank upper bound; it is minimal
  only if a lower-bound certificate matches.
- Good independent candidates but failed cube: local simplicity without a composable
  interface.
- Good cube with small interactions: evidence for modular composition on tested
  distributions/interventions.
- Good cube with large compensation: behavioral joint compression, not independent
  causal equivalence.

None alone establishes semantics, selective editability, OOD generalization, or
literal MDL. Those require separate held-out consequence tests under the repository's
conditional-simplicity contract.
