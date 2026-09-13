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

It then uses the two existing head9 writer directions from the regional producer package, normalized to unit RMS for the test, plus actual MLP9/MLP10 weights and four synthetic pristine contexts. The writers have cosine0.899: they are independent directions, but are not established here as distinct semantic circuits. Across16signed strength pairs per context, exact response error is at most $1.27\times10^{-14}$ and downstream self-product error at most $1.30\times10^{-13}$. Combining coefficient products before Down agrees with combining completed output vectors within $2.88\times10^{-15}$.

The bank therefore drops28output vectors to19, a32.1%reduction in this local vector count. The prototype also stores a19-by-28 transformation and still prepares the source projections and context. Transforming hidden products before Down permits19rather than28Down writes, but complete preparation/runtime pricing and global sharing are not yet measured. Native weights, normalization, attention changes and suffix dependencies remain outside this local count.

## Boundary: self-products versus independent cross-products

The reduction is for $K(r(a_1,a_2),r(a_1,a_2))$ within one combined branch. It can therefore be used in each branch of a two-writer intervention before forming the full finite interaction. It does not automatically reduce $K(r(a),r(b))$ when a and b are independently chosen two-dimensional strength vectors.

[The independent-cross countercheck](TWO_WRITER_RATIONAL_DOMAIN_V1_CONTROL.json) confirms response-function rank7, same-strength self-product rank19, but independently chosen left/right strength product rank28. Treating every cross-product as19-dimensional would be incorrect.

This opens an exact compression route for multi-writer composed interactions. It does not yet establish native task prediction, selective removal, semantic reuse, fresh/OOD transfer or a faster full-model implementation for two writers.
