# Store the discarded subspace, rather than a large retained basis

13September2026. The earlier dense-adapter cost was a property of that implementation, not an unavoidable price. A compact complement construction now implements the same optimal input projection with positive local storage savings. It is a modest representation result, not a new identified circuit.

## Exact construction

Unfold the retained mixed interaction as $M\in\mathbb R^{1152\times1536}$, where1536combines twelve output readers and128head-write coordinates. Suppose $N\in\mathbb R^{1152\times k}$ contains the orthonormal lowest-energy input directions to discard. The optimal input-projected coefficient matrix is

$$
\widehat M=(I-NN^T)M.
$$

Instead of storing1152times$(1152-k)$retained-reader coefficients, compute a QR factorization of the thin matrix $N$. Its orthogonal completion is a product of $k$Householder reflections,

$$
Q=H_0H_1\cdots H_{k-1},\qquad H_j=I-\tau_jv_jv_j^T.
$$

Each $v_j$ has known leading zeros and a leading one; only its trailing values and $\tau_j$ must be stored. The first$k$columns of $Q$ span$N$. Let $S$ select the remaining coordinates. Then

$$
C=S Q^T M,\qquad \widehat M=Q S^T C.
$$

For an input$x$, compute $u=S Q^T x$ by applying the reflectors, then contract $C^Tu$ with the head-write input. A full square$Q$is never stored. The executable CPU control uses implicit reflector application and independently compares against$(I-NN^T)M$and its bilinear output.

## Literal price and controls

The reflector adapter stores

$$
\sum_{j=0}^{k-1}(1152-j)=1152k-\frac{k(k-1)}2
$$

scalars, including the$k$reflection coefficients. The dense reduced core stores$(1152-k)1536$scalars. Relative to the original matrix, the saving in scalar count is therefore

$$
384k+\frac{k(k-1)}2.
$$

This is a general storage identity, not a model-specific theorem. The model determines how many directions can be discarded at the desired error.

| Coefficient error ceiling | Discarded / retained input directions | Adapter scalars | Net storage saving |
|---|---:|---:|---:|
|2%|47 /1105|53,063|1.08%|
|5%|144 /1008|155,592|3.71%|
|10%|283 /869|286,113|8.40%|

The three actual errors are1.994%,4.979%and9.999%. Projection replay errors are below1.8e-15; bilinear execution errors below2.4e-15. Reconstructing the adapter using only charged reflector tails and coefficients gives identical results. All registered criteria pass. The actual-weight CPU calculation took0.90seconds.

These are scalar-count ratios validated with FP64 algebra, not a serialized FP32 artifact or native behavioral test. Input dimension reduction does not mean the whole graph has fewer nodes: all1152native coordinates are still read, and47/144/283reflection operations are introduced. The dense reduced core has fewer readers, but adapters and their work remain part of the graph. Runtime and native regional/FineWeb fidelity remain unmeasured for this representation.

## Executed generic countercheck

The method works for any matrix, so three fixed Gaussian matrices of the same1152×1536shape were tested at identical relative tolerances and prices. They discard18,69–70and160directions, giving0.40%,1.63–1.66%and4.19%storage savings. The native interaction permits47,144and283directions and roughly twice or more the savings.

This establishes greater spectral compressibility than those generic controls. It does not identify semantic structure or compare against nulls preserving native tensor marginals. The controls also show that a nonzero saving alone is not evidence of a special circuit mechanism.

The irregular sparse-entry representation still saves more storage at10%error. The useful new point is that a smaller core need not pay for a dense retained basis. Combining this compact adapter with edge sparsity is a distinct next comparison; it must share an explicit total error budget and include both support and adapter costs. No claim is made that this construction beats all earlier representations.

[Executable construction](interaction_complement_adapter_v1.py) · [Actual-weight receipt](INTERACTION_COMPLEMENT_ADAPTER_V1_RESULT.json) · [Generic control code](interaction_complement_generic_v1.py) · [Generic results](INTERACTION_COMPLEMENT_GENERIC_V1_RESULT.json) · [Earlier node bounds and dense prices](INTERACTION_BLOCK_SPARSITY_V1_MATH.md).
