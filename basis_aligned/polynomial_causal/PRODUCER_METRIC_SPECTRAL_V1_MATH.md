# Factor after folding the producer metric into the outer quadratic

12 September2026. The frozen outer32 approximation passed developmental effects but missed progressive swaps on the fresh panel at10.54%. All fresh removals and write errors passed. This note proposes a different weight-only objective, not a fit to those examples. It concerns the same two-output pure MLP16→MLP17 path; full-unembedding and attention-path discovery remain separate work.

Let $A_a=\operatorname{sym}(l_a r_a^T)$ be the4608 native MLP16 neuron-product matrices. The bias-free producer, including its learned residual scale, is

$$
p_i(x)=x^TP_ix,\qquad P_i=\lambda\sum_aD_{ia}A_a.
$$

For one frozen downstream output direction, the exact composed numerator is

$$
f(x)=p(x)^TS p(x).
$$

The earlier baseline diagonalized $S$ in ordinary producer-output coordinates. That assumes all those directions should receive equal approximation weight. Instead form the exact quadratic-function Gram

$$
H_{ij}=\langle P_i,P_j\rangle_F,\qquad H=\lambda^2DGD^T,
$$

where the4608-by4608 neuron Gram $G$ uses the exact formula in the producer-projection note. Computing $H$ needs no text, empirical activation covariance, or Gaussian-output assumption.

To identify exactly which objective this solves, let $\mathcal P$ have rows consisting of Frobenius-isometric vectorizations of the symmetric $P_i$. Its dimensions are1152-by664128. Preserve the ordered grouping of the four input slots into two pairs. The corresponding symmetric coefficient **matrix** is

$$
T_{\rm pair}=\mathcal P^TS\mathcal P,\qquad H=\mathcal P\mathcal P^T.
$$

When $H$ is positive definite, $O=H^{-1/2}\mathcal P$ has orthonormal rows. Therefore

$$
T_{\rm pair}=O^T(H^{1/2}SH^{1/2})O.
$$

The nonzero spectrum can be obtained from the1152-by1152 middle matrix, without constructing the enormous paired matrix. If its largest-magnitude eigenpairs are $(\mu_j,v_j)$, the rank-$k$ spectral approximation executes as

$$
\widehat f(x)=\sum_{j=1}^k\mu_j(a_j^Tp(x))^2,
\qquad a_j=H^{-1/2}v_j.
$$

This minimizes the paired coefficient matrix's Frobenius error among matrices of rank at most $k$. It is **not** a proof of optimality for the fully symmetrized four-slot tensor, repeated-input error, native behavior, arbitrary arithmetic circuits, or shared factors across the two output modes. Full symmetrization can expose equivalent Gram representations, as earlier controls demonstrated. Preserving the pairing retains the known composition graph as an inductive assumption. The resulting inner quadratics stay inside the native producer span rather than becoming unconstrained canonical eigenmatrices.

If $H$ is singular or numerically truncated, whitening applies only on its retained range. The [helper](producer_metric_spectral_v1.py) uses an explicit1e-12relative eigenvalue cutoff and reports retained rank and condition. A native comparison must disclose that truncation, not claim unrestricted exactness. Native parent weights and normalization dependencies remain fully charged.

[The executed control](PRODUCER_METRIC_SPECTRAL_V1_CONTROL.json) constructs four symmetric5Dproducer quadratics and an explicit25-by25paired coefficient matrix. It verifies the spectral error optimum, full reconstruction and $H$-orthogonal readers within1.1e-15. It does not test a new model fit or establish fully symmetrized quartic optimality.

## Registered next native comparison

Use the same frozen output writers, exact MLP16 producer and32terms per output. Bind the native checkpoint, source program, helper and control before managed execution. Compare with the ordinary outer32 baseline at equal reader/weight count. The recent fresh panel is now inspected and may only serve as a labelled subsequent diagnostic for a new candidate; a new clean confirmation panel would be needed for a new generalization claim.

- A: all1152producer metric directions survive the fixed cutoff; ordinary baseline write error replays within1e-8; new readers satisfy $a^THa=I$ within1e-8 relative Frobenius error; paired-error formula agrees with direct1152-dimensional computation within1e-8.
- B: paired coefficient error is at least10% lower than the ordinary outer32 approximation for each output mode.
- C: every original developmental family passes swap relative error<=0.10, sign>=0.90 and at least4live pairs.
- D: every original family passes removal CE disagreement<=0.02nats and write relative error<=0.05.

The null is that respecting the paired producer geometry improves its stated coefficient metric but not intervention fidelity. Do not replace it with a fully symmetrized-metric claim. If the numerical restriction fails, report an invalid or restricted comparison before interpreting behavioral numbers. No language-data fitting, rank sweep or label-conditioned selection. Current status: mathematical kernel implemented and controlled; native runner is the next implementation task, not yet queued.
