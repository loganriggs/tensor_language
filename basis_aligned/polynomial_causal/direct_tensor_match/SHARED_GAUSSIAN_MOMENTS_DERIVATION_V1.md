Exact moments while retaining shared quadratic computations

Write each whitened feature as q_i(z)=z^T A_i z+l_i^T z+c_i, with symmetric A_i and z~N(0,I). For input affine factors, A_i=Sym(sum_k U_ik V_ik^T), l_i=sum_k(b_ik V_ik+d_ik U_ik), c_i=sum_k b_ik d_ik. Its mean is tr(A_i)+c_i. The executable polynomial is unchanged by whitening; these quantities describe only the fitting metric.

Let A(t)=sum_i t_i A_i, l(t)=sum_i t_i l_i and c(t)=sum_i t_i c_i. Around t=0 the Gaussian cumulant generating function is

$$
\log\mathbb E\exp\left(\sum_i t_iq_i\right)
=c(t)-\tfrac12\log\det(I-2A(t))
+\tfrac12 l(t)^\top(I-2A(t))^{-1}l(t).
$$

Differentiation gives the first cumulant above. For k>=2 distinct labelled slots (their underlying feature indices may repeat),

$$
\kappa(q_{i_1},\ldots,q_{i_k})=
\frac{2^{k-1}}{k}\sum_{\pi\in S_k}
\operatorname{tr}(A_{i_{\pi(1)}}\cdots A_{i_{\pi(k)}})
+2^{k-3}\sum_{\pi\in S_k}
l_{i_{\pi(1)}}^\top A_{i_{\pi(2)}}\cdots
A_{i_{\pi(k-1)}}l_{i_{\pi(k)}}.
$$

For k=2 the second term is l_i^T l_j, with no intervening A. Cyclic and reversal identities for symmetric forms reduce the trace sum to onecycle fork=3 andthreecycles fork=4. Reversal reduces the linear-chain sum aswell. The implementation checks these multiplicities independently; repeated slots are still labelled during differentiation.

The rootfeatureGram G_ab=E[q_i q_j q_k q_l] follows from the15setpartitions of four labelled slots: for each partition multiply the cumulant of each block, then sum. This incorporates constant,linear,quadratic andhigherGaussianchaos contributions without confusing them with coefficient Frobenius innerproducts.

To avoid dense1152-by1152matrices for144features, factor A_i=E_i^T F_i with E_i=[U_i;V_i], F_i=[V_i;U_i]/2 (rank<=8). Cache F_i E_j^T, E_j l_i and F_i l_j. Trace cycles become products of8-by8blocks. Process64leftrootpairs at a time. The cross with the native teacher uses the existing exact affine-CP cross and expands only a bounded batch ofrootpairs; the entire8192-term student is never materialized as a selfGram.

[Implementation](shared_gaussian_moments.py) · [Independent quadrature and gradient checks](SHARED_GAUSSIAN_MOMENTS_CONTROLS_V1.json). This is an algebraic fitting instrument, not evidence that the Gaussian law equals actual activation statistics or that its features are semantic units.
