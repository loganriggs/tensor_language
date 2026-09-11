# Three-hour mathematical review — 11 September 10:51 UTC

The target is still a simpler executable explanation with OOD prediction,
extraction, selective removal/interchange and composition/reuse. This cycle
asks what would identify reusable units inside a well-fitting weight program.
The answer is not supplied by reconstruction alone, or by importing a generic
tensor uniqueness theorem whose hypotheses have not been checked.

## Current object and price

The local native bilinear map has input dimension d=1152, r=4608 products,
output dimension1152 and vocabulary size50304. Its weight-folded tensor is

$$
T_{vij}=\sum_{k=1}^{r}(UD)_{vk}
\frac{L_{ki}R_{kj}+R_{ki}L_{kj}}2.
$$

The current candidate uses a K=2304 shared input dictionary B and sparse code
matrices C_L,C_R, each with128 fixed positions per row:

$$
f=Bx,\quad a=C_LB,\quad b=C_RB,\quad
\widehat q(x)=W[(ax)\odot(bx)]+\mathrm{bias}.
$$

The contraction graph is input → shared features → two sparse linear maps →
paired products → output map → unembedding. These are degree-two polynomials
in the normalized local input x, symmetric in the two input tensor indices.
Native residual paths, attention, final RMS normalization and tanh remain
outside this polynomial fit. Folding through them does not make the whole
model a fixed quadratic tensor. Native product pairing/supports remain tied;
input features, sparse values and W are learned.

The discovery domain is all local real inputs under coefficient Frobenius
error, not a sampled text distribution. The output metric is M=U^T U, handled
through an exact square root. FineWeb remains frozen in-distribution validation;
no activation statistics or text weights enter this fit.

Literal program price is9,142,272 matrix coefficients,1,179,648support indices
and1152bias values, plus runtime sparse indices, U and native background.
The linear maps cost approximately Kd+2r*128+dr multiply-accumulates per input,
plus r bilinear products and the retained unembedding/normalization work.
Gauge freedom does not erase stored coefficients or index costs.

The new variable-projection fit reuses the exact conditional output solve from
the07:51 review. It is now running on both original repaired starts, with
3600softseconds per start and full gradient checks. A preceding one-step test
improved full capture from53.6% to55.96/55.94%, with about1second per gradient.
At the inspected first-start iteration225, capture was64.68%, but relative
stationarity remained9.36e-3. This is intermediate, not converged identification.

## Which mathematical results apply

**Sparse dictionary identification.** The observable in ordinary dictionary
learning is Y=B^T C^T, whereas the new fit observes only the composed output
quadratic tensor and also changes W. Local dictionary-identification theorems
require conditions on the generating coefficients, sparsity/coherence, sample
distribution and closeness of initialization. They are not global recovery
guarantees for arbitrary trained weights. Schnass gives local overcomplete
identification results for a particular criterion and an O(dKN)-per-iteration
algorithm; that criterion and its generative assumptions are not our current
folded loss. [Schnass](https://arxiv.org/abs/1401.6354),
[Gribonval and Schnass](https://arxiv.org/abs/0904.4774).

A directly checkable, weaker result concerns exact sparse codes in a *fixed*
unit-atom dictionary. With mutual coherence

$$
\mu=\max_{i\ne j}|\langle B_i,B_j\rangle|,
\qquad k<\frac12(1+\mu^{-1}),
$$

the standard sufficient bound certifies unique sparsest exact codes and their
basis-pursuit recovery. It does not identify B itself or prove anything about
Lasso with a nonzero penalty. The CPU audit on the two frozen repaired
dictionaries finds mu=.7284/.6927, giving strict thresholds1.186/1.222 versus
actual maximum support128. This sufficient certificate is unavailable.
It does **not** prove nonuniqueness: a single close atom pair can make a global
coherence bound pessimistic for other supports. Median nearest-atom cosines
are only about.112. For example, adding a nearly duplicate e1/e2-direction atom
to an identity dictionary spoils the global bound while e3+e4 still has a unique
two-atom representation. Computing coherence costs O(K²d), storage O(K²).
[Donoho and Elad](https://pmc.ncbi.nlm.nih.gov/articles/PMC153464/),
[executed native audit](READER_COHERENCE_CERTIFICATE_V1_AUDIT.json).

**CP tensor identifiability.** Rank-one CP uniqueness is the correct neighboring
problem, but our symmetrized products expand into two CP terms with tied output
columns. Ordinary Kruskal certificates cannot simply treat those as independent
generic factors. Even ignoring the ties, each factor's column rank is limited
by1152 through the output/input maps. Failure of a sufficient Kruskal bound is
not failure of identifiability. Robust recovery results require robust rank
conditions; stronger generic algebraic results concern general complex tensors,
or specific decompositions satisfying additional nonsingularity conditions.
Neither condition has been verified here. [Bhaskara et al.](https://proceedings.mlr.press/v35/bhaskara14a.html),
[Chiantini et al.](https://arxiv.org/abs/1403.4157).
Constructing a full parameter-to-coefficient Jacobian here would involve
hundreds of millions of output coefficients and millions of variables; that
is not a practical native certificate. Structure-aware directional tests are
the usable consequence, not a blanket claim of uniqueness.

**Tensor trains and graph-width contraction.** TT-SVD gives controlled
approximation through unfolding ranks once a tensor ordering is fixed. It
does not prove that its coordinates are circuit units. The current local
order-three tensor is already evaluated implicitly; splitting indices into
more artificial modes adds an ordering assumption, not a new observed
independence. Small graph width can make contraction efficient, but does not
find a simpler graph or identify shared computation. These results can improve
execution if suitable structure is demonstrated. They do not replace the
current identification test. [Oseledets](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf),
[Markov and Shi](https://arxiv.org/abs/quant-ph/0511069).

**Hankel minimal realization and arithmetic circuits.** Finite Hankel rank
characterizes finite weighted-automaton realizations; spectral algorithms need
appropriate complete prefix/suffix blocks and linear/bilinear recurrent state
structure. The actual normalized attention/residual model has no established
finite linear realization of that form. Earlier finite-horizon tangent tools
are approximations with their own scope, not a theorem for the whole network.
Tensor rank measures bilinear multiplication complexity, while arbitrary
linear-map reuse and semantic editability require additional accounting.
Different arithmetic circuits can implement the same map.
[Li et al.](https://arxiv.org/abs/2010.10029),
[Bläser's bilinear complexity survey](https://theoryofcomputing.org/articles/gs005/).

## Executable consequence: quotient feature changes by output compensation

Prior `quotient_jacobian_minimality.py` already states the general principle:
compare physical-function changes after accounting for known gauges. This
cycle specializes it to the current quadratic product bank, reusing the
existing CP inner products and exact output solve.

Let h_k=sym(a_k b_k^T), with writers already transformed by the output metric.
A feature/reader perturbation induces

$$
\delta T=\sum_k w_k\otimes
\operatorname{sym}(\delta a_k b_k^\top+a_k\delta b_k^\top).
$$

Form the feature Gram matrix and tangent cross matrix,

$$
G_{jk}=\langle h_j,h_k\rangle_F,
\qquad K_{ok}=\langle\delta T_o,h_k\rangle_F.
$$

The output change best able to cancel this tangent is

$$
\delta W_*=-KG^+,
\qquad
\|\delta T_\perp\|_F^2
=\|\delta T\|_F^2-\operatorname{tr}(KG^+K^\top).
$$

This tests whether the physical first-order change can be absorbed by output
reweighting. It is **not** the complete variable-projection residual Jacobian
or Hessian at a nonzero target residual. A null tangent alone does not prove
an exact finite symmetry or global nonidentifiability.

The new [tool](writer_compensated_tangent_v1.py) reuses the spectral writer solve.
For r products its dense Gram solve remains O(r³), with O(r²d) contractions and
O(r²) storage. Multiple directions can share a future factorization; no full
native Jacobian is required. [Executed controls](WRITER_COMPENSATED_TANGENT_V1_CONTROL.json)
compare against independent dense coefficient tensors:

- Rescaling readers is canceled exactly by output compensation.
- Rotating the complete2D quadratic block {x1²,x1*x2,x2²} leaves zero tangent
  residual. A finite.3radian rotation with refitted writers preserves the
  complete function to5.32e-32 relative error.
- Rotating an isolated x1² feature toward x2 retains100% of the tangent energy
  after optimal output compensation. The corresponding finite rotation leaves
  relative error.1670.

All registered bars held. This suggests an operational distinction: an
identifiable computation may be a whole interacting block, while its individual
feature axes remain interchangeable. The result is a small known-answer tool,
not evidence that a particular native block has been identified. It informs
future grouping and removal semantics after the native fit stabilizes.

## Another consequence: compare gradients in a declared gauge

With each product unchanged under

$$
a_k\mapsto s_k a_k,\qquad b_k\mapsto b_k/s_k,
$$

the current global code statistic ||grad C||_F ||C||_F / loss can change even
though the function is identical. For nonzero rows, the rowwise statistic

$$
\frac{\sqrt{\sum_i\|\nabla_{C_i}\mathcal L\|^2\|C_i\|^2}}
{\max(|\mathcal L|,10^{-12})}
$$

is invariant to independent row rescaling. It is still not a global optimum
certificate, and zero rows need separate treatment. The
[executed control](PROJECTED_READER_SCALE_GAUGE_V1_CONTROL.json) scales Left
codes by1000 and Right codes by1/1000. Loss is unchanged; writers and dictionary
gradients replay within2.72e-14. The global code statistic inflates379,588times,
while the rowwise statistic agrees to1.58e-15 relative error.

Do not change the running source or retrospectively replace its convergence
criterion. Its dictionary gradient also remains nonzero, so this observation
does not rescue native convergence. Use the result to interpret future
conditioning and to compare equivalent parameterizations honestly.

## Decision and continuation

Keep the sustained full-tensor fit running under its frozen protocol. It has
made materially larger gains than the reader proxy, and its allocated joint
convergence budget is not exhausted. Do not infer native units from generic
dictionary/CP theorems or from raw parameter similarity.

The completed tangent tool is the highest-information new mathematical
consequence for stable identification and cross-boundary grouping: use it to
distinguish feature coordinates from irreducible interacting blocks, then test
finite edits and the four behavioral properties. It must not become a large
catalog of gauge controls detached from actual candidates. The coherence and
gradient checks above settle their narrow assumptions; do not repeat them.

Concrete continuation is the live managed projected fit, plus the actually
executed polynomial tangent and scale-gauge controls. No text/data-guided
discovery is added. Next mathematical review13:51; hourly11:22.
