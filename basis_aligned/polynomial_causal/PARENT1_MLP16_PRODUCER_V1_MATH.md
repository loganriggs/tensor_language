# The preceding bilinear layer's contribution to the three branch inputs

The generic attention OV pullback was already implemented in SHARED_READER_UPSTREAM_PORTS_V1_AUDIT.json. This study applies the existing producer-function viewpoint to the **different, frozen parent1 ports**: its shared reader and two partner readers. It does not introduce another attention-folding method.

The conditional native MLP16 fold is exact, but the three resulting quadratic functions are not close to one common scalar producer or to a small bank of square terms. The leading output mode captures **40.71%** of coefficient energy, or **40.62%** after separating the radial term. The three trace-removed output modes need **481,507,502 signed squares** for90%coefficient capture. Sixteen squares capture20.66%,11.18%,12.96%; the best single real products capture9.77%,3.28%,4.91%. These are weight-coefficient statements, not natural-input or behavioral fidelity measurements.

## Exact conditional fold

Let $x$ be the normalized MLP16 input and

$$
z(x)=(L_{16}x)\odot(R_{16}x).
$$

Let $F$ have rows $u^\top,p_0^\top,p_1^\top$, the three frozen input readers of the downstream shared node. Choose an orthonormal basis $B$ for their span and retain the exact adapter $FB$. The block17 residual re-entry coefficient is $\beta=1.0234375$. The three producer quadratics are

$$
Q_j=\beta\,\operatorname{sym}\left[L_{16}^\top\operatorname{diag}(B_j^\top D_{16})R_{16}\right].
$$

Thus $x^\top Q_jx+\beta B_j^\top b_{16}$ is the MLP16 contribution to port $j$ in these orthonormal coordinates. Map back through $FB$ and add the other contributions. If $b$ denotes the intervening attention/residual contribution held as an explicit background,

$$
h_{17}=b+\beta[D_{16}z(x)+b_{16}],\qquad
F\,\operatorname{RMS}(h_{17})
=\frac{Fb+(FB)\,[x^\top Q_jx+\beta B_j^\top b_{16}]_j}{\rho_{17}}.
$$

The branch outputs then multiply the shared port by each partner port and write through their original writers. The calculation retains the actual RMS divisor and bias. The attention contribution inside $b$ can itself depend on MLP16; treating it as an interface is conditional extraction, not an independent producer or a claim that the whole causal path is fixed.

Direct native-factor versus dense-quadratic, adapter, bias/radial and downstream-branch replays hold within2.61e-15. Only three1152-by-1152matrices are materialized. [Code](parent1_mlp16_producer_v1.py) · [Receipt](PARENT1_MLP16_PRODUCER_V1.json).

## Separating the radial term

Write $Q_j=\widetilde Q_j+\alpha_jI$, where $\alpha_j=\operatorname{tr}(Q_j)/1152$. Its contribution is exactly $\alpha_j\|x\|^2$. This is constant on an exact fixed-radius sphere; native RMS with epsilon makes the radius slightly input-dependent. We retain the actual radius, rather than silently dropping this term. Radial energy accounts for only0.208%of this producer tensor, so it does not explain the observed complexity.

## Could different output mixtures be much simpler?

To check that objection, Frobenius-orthonormalize the three trace-removed matrices into $A_1,A_2,A_3$. Every unit-energy function in their span is $A(c)=\sum_i c_iA_i$ with $\|c\|=1$. Define

$$
M=\sum_i A_i^2.
$$

For every vector $v$, Cauchy–Schwarz gives

$$
\|A(c)v\|^2\le\sum_i\|A_iv\|^2,
\qquad A(c)^2\preceq M.
$$

For any rank-$r$ orthogonal projector, the trace against $A(c)^2$ is therefore bounded by its trace against $M$. Maximizing over those projectors gives the uniform bound

$$
\text{best rank-}r\text{ captured energy of }A(c)
\le\min\left(1,\sum_{j=1}^{r}\lambda_j(M)\right).
$$

This holds for **every output mixture**, not just the three displayed singular modes. A sum of$r$signed squares has matrix rank at most$r$, so it inherits the bound. A single real bilinear product has symmetric matrix rank at most2, so the rank2bound is also a valid, potentially loose upper bound for it.

The rank2upper bound is **12.86%**, rank16is **35.55%**, and rank31is **48.73%**. At least **93signed squares** are necessary for90%coefficient capture for any nonzero function in this three-dimensional trace-removed span. This lower bound need not be tight. Gram normalization and four independent mixture checks pass within5.56e-15. These are FP64 numerical bounds, not interval-arithmetic certificates. [Code](parent1_producer_mixture_bound_v1.py) · [Receipt](PARENT1_PRODUCER_MIXTURE_BOUND_V1.json).

Changing the output basis alone cannot rescue a16-square producer in this span. This does not rule out different input readers, shared computations in a larger program, cancellation after full composition, or a representation accurate on naturally reachable model states. It also does not invalidate the exact conditional interface. Earlier MLP16 producer studies found related complexity for other reader sets; these results are new measurements of the current three ports, not a novel general conclusion about MLP16.
