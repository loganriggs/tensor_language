# Simple products inside fixed shared producer spans

The leading common functions from the midpoint QK/OV pair were dense. Their
selection optimized alignment, not arithmetic simplicity. This tests whether
other combinations in the same function spans are better. No data, new head,
new source frame, new unembedding component or tradeoff-weight sweep is selected.
Prior authority: explanation_2026-09-11_0433, producer scalar audit, MLP17/readout
dossiers and the explicit user weights-first instruction.

For17 Frobenius-orthonormal symmetric coefficient matrices H_i per head, select
unit c and Q(c)=sum_i c_i H_i. Optimize its exact best one-real-product capture:

$$
F(c)=\max(\lambda_{\max}(Q(c)),0)^2
+\min(\lambda_{\min}(Q(c)),0)^2,\qquad \|c\|_2=1.
$$

The gradient uses the two extreme eigenvectors. It is smooth when active extreme
eigenvalues are simple; report gaps rather than assuming differentiability at
ties. Use existing Pymanopt2.2.1 Sphere/Polak–Ribiere conjugate gradient with
analytic derivative. Four starts/head: leading canonical pair, best individual
basis function, random seeds1471+100h and1483+100h. Limits15seconds/start,
1000iterations,10000evaluations, gradient threshold1e-7, minimum step1e-18.
Time/step/iteration limits are not convergence. Save all final points so that a
same-objective restart repair is possible. No undocumented objective change.

For S=sum_i H_i^2, every unit c satisfies Q(c)^2<=S in positive-semidefinite
order, by the vector Cauchy–Schwarz inequality. A k-product matrix has rank<=2k,
so its capture is at most min(1,sum of the top2k eigenvalues of S). This is a
whole-span upper bound; it may be loose. Record k=1,4,16. Thus a failed local
fit can be distinguished from a useful analytic exclusion when the bound is low.

- pred_a: normalized Gram, eigenspectrum residual, constructed-product energy,
  common-reader replay, and upper-bound violations all <=1e-8; controlled analytic
  gradient, basis invariance and planted-product recovery pass before enqueue.
- pred_b: all36 starts reach the original gradient threshold1e-7.
- pred_c: at least6of9 heads have >=2x leading-pair one-product capture, >=.05
  absolute capture gain, and associated QK/OV function cosine>=.5.

Null: arbitrary mixtures of these common functions remain expensive, or their
arithmetic gain loses the parent computation similarity. Preserve every miss.
These are coefficient-space screens, not behavioral claims. The parent's
canonical similarity need not be retained when choosing new combinations.

Common basis i is (u_i+v_i)/sqrt(2(1+cos_i)). For fitted c, its associated QK and
OV functions have coefficients c_i/sqrt(2(1+cos_i)) in u_i and v_i respectively.
Their sum exactly reproduces the common candidate. Measure their actual cosine.
Native residual/bias/RMS/base streams and complete downstream background remain;
normalization is not replaced by a polynomial surrogate.

Literal price: one approximate product has2304 input coefficients and one scale
per head, plus the explicit common/private adapters and retained residual. A
single scalar target's small price does not price the complete circuit. The run
uses0body forwards/0sequences and one head's17 dense1152-square matrices at a
time (~180MB FP64); native products remain implicit until this bounded slice.

Method mapping: Nakatsukasa, Soma and Uschmajew study low-rank elements/bases in
a matrix subspace using singular-value shrinkage and alternating projections.
Our span is real symmetric and one product permits only one eigenvalue of each
sign, which is stricter than generic rank2. We optimize the exact restricted
approximation objective with sphere CG, not their complete algorithm, and claim
no global recovery guarantee. [Paper](https://arxiv.org/abs/1503.08601),
[Pymanopt manifolds](https://pymanopt.org/docs/stable/manifolds.html).
