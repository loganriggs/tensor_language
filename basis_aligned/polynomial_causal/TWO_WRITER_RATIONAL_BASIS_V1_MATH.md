# Two different writers: an exact 19-term interaction bank

13 September2026. The shared-denominator simplification extends beyond two scalar edits along the same writer. This is an algebraic generalization, with actual-weight CPU checks; native behavioral validation remains outstanding.

Let the perturbation remove $a_1w_1+a_2w_2$ from a pristine pre-MLP state z. Define

$$
\beta_i=\frac{z^\top w_i}{d},\qquad
\Gamma_{ij}=\frac{w_i^\top w_j}{d},\qquad
\rho(a)=\rho_0-2\beta^\top a+a^\top\Gamma a,
$$

where $d=1152$, $\rho_0=\|z\|^2/d+\epsilon$, and $m_0=B_9(z)/\rho_0$ is the bias-free pristine MLP output. Let $K_9(x,y)$ be its symmetric bilinear map and define

$$
p_i=K_9(w_i,z)-2\beta_i m_0,\qquad
q_{ij}=K_9(w_i,w_j)-2\Gamma_{ij}m_0.
$$

The exact residual-plus-MLP response, before multiplying by the next block's learned re-entry coefficient, is

$$
r(a)=-\sum_i a_iw_i-\frac{\sum_i a_ip_i}{\rho(a)}
+\frac{\tfrac12\sum_{i,j}a_i a_jq_{ij}}{\rho(a)}.
$$

There are seven vector directions: two w, two p and three symmetric q. A generic symmetric downstream product on these vectors would store $7\cdot8/2=28$ output vectors. But its scalar coefficients are constrained by the shared denominator.

## Why only19 are needed

Write $r=X+(P+Q)/\rho$, where X and P are homogeneous degree-one vector polynomials in $(a_1,a_2)$, and Q has degree two. For the next bilinear layer's symmetric map K,

$$
K(r,r)=K(X,X)+\frac{2K(X,Q)}{\rho}
+\frac{2\rho K(X,P)+K(P,P)+2K(P,Q)+K(Q,Q)}{\rho^2}.
$$

This keeps:

- Three degree-two monomials without a denominator.
- Four degree-three monomials divided by $\rho$.
- Twelve monomials of degrees two, three and four divided by $\rho^2$: $3+4+5=12$.

Hence19scalar functions suffice. The reduction moves the $K(X,P)/\rho$ contribution into the shared numerator; it does not discard any interaction. Multiplying every scalar function by $\rho^2$ also explains independence: the first group has independent degree-six leading polynomials, the second adds degree-five leading polynomials, and the remaining twelve fill degrees two through four. This assumes the quadratic part of $\rho$ is nonzero. It concerns scalar-function space, not proof that all19physical output vectors are independent for this checkpoint.

The same counting argument gives

$$
2\binom{k+1}{2}+2\binom{k+2}{3}+\binom{k+3}{4}
$$

functions for k fixed writers. It recovers five for k=1 and19for k=2. This is the same denominator-elimination principle used earlier, not a claim of a new general tensor-decomposition theorem.

## Executed checks

[The implementation](two_writer_rational_basis_v1.py) constructs the coefficient systems, solves the19-by-28 bank transformation, and evaluates the resulting scalar functions. [The CPU control](TWO_WRITER_RATIONAL_BASIS_V1_CONTROL.json) checks an exact rational polynomial instance with symbolic arithmetic: both the original28-function design and chosen19-function design have rank19.

**Provenance correction:** the two stored directions are head8.2's first-value writer and head9.8's current-value writer, as documented in the regional producer package. The original control and earlier board entry incorrectly called them two head9 writers. That receipt is preserved. This test transplants both directions to a common MLP9 input; it does not compose their original circuit paths, which would require executing the intervening computation.

The test normalizes these directions to unit RMS and uses actual MLP9/MLP10 weights with four synthetic pristine contexts. The writers have cosine0.899. Across16signed strength pairs per context, exact response error is at most $1.27\times10^{-14}$ and downstream self-product error at most $1.30\times10^{-13}$. Combining coefficient products before Down agrees with combining completed output vectors within $2.88\times10^{-15}$.

The bank therefore drops28output vectors to19, a32.1%reduction in this local vector count. The prototype also stores a19-by-28 transformation and still prepares the source projections and context. Transforming hidden products before Down permits19rather than28Down writes, but complete preparation/runtime pricing and global sharing are not yet measured. Native weights, normalization, attention changes and suffix dependencies remain outside this local count.

## Boundary: self-products versus independent cross-products

The reduction is for $K(r(a_1,a_2),r(a_1,a_2))$ within one combined branch. It can therefore be used in each branch of a two-writer intervention before forming the full finite interaction. It does not automatically reduce $K(r(a),r(b))$ when a and b are independently chosen two-dimensional strength vectors.

[The independent-cross countercheck](TWO_WRITER_RATIONAL_DOMAIN_V1_CONTROL.json) confirms response-function rank7, same-strength self-product rank19, but independently chosen left/right strength product rank28. Treating every cross-product as19-dimensional would be incorrect.

This opens an exact compression route for multi-writer composed interactions. It does not yet establish native task prediction, selective removal, semantic reuse, fresh/OOD transfer or a faster full-model implementation for two writers.

## Direct compilation and measured cost

[The direct compiler](two_writer_direct_bank_v1.py) expands the polynomial coefficients in the displayed identity before applying Down. It projects the seven response vectors through Left and Right once, combines their hidden products into nineteen coefficients, and performs nineteen Down writes. It needs no per-context least-squares solve or stored 19-by-28 transformation.

[The actual-weight control](TWO_WRITER_DIRECT_BANK_V1_CONTROL.json) agrees with the original transformed bank within $2.94\times10^{-15}$ relative error and preserves the self-product within $1.72\times10^{-13}$. The provenance limitation above still applies.

[Matched preparation timing](TWO_WRITER_DIRECT_BANK_V1_PRICE.json) compares against an optimized 28-product implementation that also projects the seven vectors only once. Seven interleaved FP64, two-thread CPU trials give median times of 9.28 versus 8.69 ms for one context (1.07×), and 38.24 versus 26.39 ms for eight (1.45×). The declared requirement of at least 1.1× at both batch sizes **fails**. These are local preparation measurements on synthetic vector banks with actual MLP10 weights; they exclude generating the response vectors, attention, and the suffix.

The output bank has 21,888 rather than 32,256 scalars per context, a 32.1% reduction. This is a valid local representation saving, not a measured reduction of total circuit state or whole-model latency.

## Precision: compress in independent writer coordinates

[An FP32 stress test](TWO_WRITER_PRECISION_V1_CONTROL.json) exposes cancellation when nearly identical writers receive opposite amplitudes. With writer cosine 0.999999, the nineteen-bank relative error reaches 1.29–4.28, while direct response-product evaluation stays around 0.00010–0.00018. The declared 0.001 bar fails. These are synthetic contexts and writer pairs using actual MLP weights, not failures measured on native text.

The failure is avoidable in this screen. For unit-RMS orthogonal directions $u,v$, write

$$
w_1=u,\qquad w_2=c u+\sqrt{1-c^2}v,\qquad
a w_1+b w_2=(a+cb)u+b\sqrt{1-c^2}v.
$$

Compile the bank in $(u,v)$ coordinates and transform amplitudes accordingly. This preserves the input edit, but avoids expanding a small difference in two nearly identical writer coordinates. [The executed countercheck](TWO_WRITER_ORTHOGONAL_PRECISION_V1_CONTROL.json) lowers the maximum relative error on material cases to $4.77\times10^{-7}$ and gives zero computed response for exactly cancelling identical writers. The tiny nonzero FP64 reference there is rounding residue. The original failure remains recorded.

This suggests orthogonalizing a writer span before compiling its interaction bank; orthogonal coordinates are computational choices, not newly identified semantic circuits. The screen constructs its orthogonal directions explicitly. A general rank-revealing compiler for arbitrary stored writers, its coordinate-map cost, native BF16 execution and behavioral validation remain outstanding.

### General writer-span utility

[writer_span_coordinates_v1.py](writer_span_coordinates_v1.py) now computes a thin FP64 SVD of arbitrary writer rows $W$ and returns unit-RMS orthogonal rows $V$ and an amplitude map $C$, with $W\approx CV$. An edit $aW$ becomes $(aC)V$. The reported residual $E=W-CV$ bounds edit error by $\|a\|_2\|E\|_2$; numerical rank truncation is explicit. Amplitude transformation stays in FP64 to preserve cancelling edits before conversion to execution precision.

[The utility control](WRITER_SPAN_COORDINATES_V1_CONTROL.json) checks the stored writer pair, a nearly duplicate pair, exact duplicates and a rescaled pair. All edit reconstruction errors are below $2.87\times10^{-15}$ relative to input amplitude/weight scale. Near duplicates retain rank two; exact duplicates reduce to rank one. Two independent writers require 2,304 basis scalars plus four map scalars, versus 2,304 original scalars. Exact duplicates require 1,152 plus two. These counts exclude diagnostic residual storage, which callers need not retain for execution.

[The generic SVD-basis precision check](TWO_WRITER_SVD_PRECISION_V1_CONTROL.json), replacing the hand-specified basis above, passes the same material-case bar with maximum relative error $4.90\times10^{-7}$. Its rank-one case is padded with a zero second direction for compatibility with the nineteen-bank test. It does not yet dispatch to the cheaper five-bank implementation. Native precision and behavioral tests, and integration across the original two circuit paths, remain outstanding.
