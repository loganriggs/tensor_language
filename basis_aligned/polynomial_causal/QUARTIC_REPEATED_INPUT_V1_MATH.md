# A weight-only metric for repeated-input quartics

The existing coefficient objective is mathematically valid, but it is not the same as averaging the evaluated polynomial over directions. This distinction matters because the native model puts the same normalized input in all four slots. It offers a different structural assumption to test without collecting text or fitting activation statistics.

For symmetric quartic coefficient tensors $T,S$, define their partial traces

$$
A_{ij}=\sum_kT_{ijkk},\qquad B_{ij}=\sum_kS_{ijkk}.
$$

Independent isotropic vectors in the four slots give

$$
\mathbb E\left[T(x_1,x_2,x_3,x_4)S(x_1,x_2,x_3,x_4)\right]=\langle T,S\rangle_F.
$$

For the same standard Gaussian vector $x$ in every slot, the answer is instead

$$
\mathbb E\left[T(x,x,x,x)S(x,x,x,x)\right]
=24\langle T,S\rangle_F+72\langle A,B\rangle_F
+9\operatorname{tr}(A)\operatorname{tr}(B).
$$

This follows from [Isserlis' pairing theorem](https://www.unige.ch/~vilmart/paper_isserlis.pdf): the105 pairings of eight slots split into24 with four cross-tensor pairs,72 with two cross pairs, and9 with none. The numerical coefficients are our quartic specialization of that theorem. No sampling is required to evaluate the resulting Gram and target-cross matrices.

Uniformly distributed directions on the radius$\sqrt d$ sphere give the same objective times

$$
\frac{d^4}{d(d+2)(d+4)(d+6)}.
$$

Indeed, a Gaussian's direction and radius are independent and its eighth radial moment is the denominator. The common positive factor does not change a least-squares optimizer. This correspondence applies to homogeneous quartic numerators; native input denominators, residual routes and capped final logits are not silently included. Actual model inputs need not be uniformly distributed or Gaussian, so this remains an isotropic modeling assumption, not a universally correct behavioral loss.

## Exact feature and native-network traces

For a squared quadratic $f(x)=(x^\top Qx)^2$, the partial trace is

$$
\operatorname{Tr}F=\frac{(\operatorname{tr}Q)Q+2Q^2}{3}.
$$

For the native target, first fold the output reader into the last MLP, obtaining its symmetric matrix $M$. Let the previous MLP's rank-two product matrices be

$$
A_\alpha=\frac{l_\alpha r_\alpha^\top+r_\alpha l_\alpha^\top}{2},
\qquad H=\lambda^2D_{16}^\top M D_{16}.
$$

Then the native homogeneous quartic is $\sum_{\alpha\beta}H_{\alpha\beta}(x^\top A_\alpha x)(x^\top A_\beta x)$, with partial trace

$$
\operatorname{Tr}T=\frac13\left[
\sum_\alpha A_\alpha\sum_\beta H_{\alpha\beta}\operatorname{tr}(A_\beta)
+2\sum_{\alpha\beta}H_{\alpha\beta}A_\alpha A_\beta
\right].
$$

The second sum is symmetric because $H$ is symmetric. Products of the native left/right Gram matrices evaluate it without materializing every residual-output quadratic or the full quartic tensor. The resulting two target trace matrices contain about21MB of doubles; these are analysis artifacts, not additional candidate parameters.

[Controls](QUARTIC_REPEATED_INPUT_V1_CONTROL.json) compare the native trace contraction to a fully materialized small network tensor, and the metric to exact five-node-per-coordinate Gauss–Hermite quadrature. All errors are below5e-16, including signed quadratic features. This also checks the distinction from simply discarding quartic trace terms.

## Discriminating native experiment

Keep the initial seed11511's32 quadratics and two centered-unembedding output directions fixed. Replace the coefficient Gram and target contractions by the repeated-input versions, then solve only the output mixing. This is a different weight-only objective at the same592,704-float candidate price. It does not change the live coefficient-objective nonlinear run.

The native check requires: partial-trace replay against direct basis contractions within1e-8; a normal-equation residual within1e-8 and nonincreasing new objective; halved native write error relative to the old coefficient-optimal writer; and native write error at most10%. Two fixed probe vectors test the partial trace by summing over all1152 coordinate basis vectors. Cached native states are used only after fitting for validation. No text-weighted objective or covariance estimate enters discovery.

If the fixed-bank comparison passes, the next question is whether the trace-aware objective improves nonlinear extraction and its interventions. If it fails, distinguish insufficient fixed features from the isotropic assumption; neither outcome licenses a circuit claim. The earlier per-inner-quadratic isotropic correction was a different operation and its failure remains preserved.
