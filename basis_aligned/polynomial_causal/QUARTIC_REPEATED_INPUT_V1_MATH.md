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

## An exact radial and harmonic decomposition

The trace also gives a canonical structural split, rather than merely a different fitting loss. Write $r^2=\|x\|^2$. The homogeneous polynomial decomposition into harmonic components is described in [Kazdan's notes, chapter5 section6](https://www2.math.upenn.edu/~kazdan/425S11/Australia/AMSI-PDEnotes-08/AMSI-2up.pdf). For our quartic, specializing that decomposition gives

$$
f(x)=H_4(x)+r^2H_2(x)+r^4H_0,
$$

$$
H_0=\frac{3\operatorname{tr}(A)}{d(d+2)},
\qquad
H_2(x)=\frac{6}{d+4}x^\top\left(A-\frac{\operatorname{tr}(A)}{d}I\right)x.
$$

“Harmonic” here means zero Laplacian: $\Delta H_4=\Delta H_2=0$. It is a mathematical angular decomposition, not a claim that a feature is periodic or semantically meaningful. The formulas follow from $\Delta f=12x^\top Ax$, $\Delta^2 f=24\operatorname{tr}(A)$, $\Delta(r^2H_2)=2(d+4)H_2$, and $\Delta(r^4)=4(d+2)r^2$.

On a fixed-radius sphere, the first two displayed lower components act as a constant and a quadratic. Native normalization is not an excuse to replace the radius or downstream denominator silently: execution keeps $r^2$ and the actual MLP17 input denominator. The remaining harmonic quartic remains an explicitly accounted-for function.

[Controls](QUARTIC_HARMONIC_V1_CONTROL.json) verify vanishing traces and exact quadrature orthogonality within5.9e-16. The planned CPU native check uses the queued exact trace artifact and asks whether the radial-plus-quadratic part approximates both write levels and paired input changes within10%. No lower-rank fit is attempted unless this exact lower component first passes fidelity. Spherical orthogonality does not imply orthogonality on text; native component norm ratios must not be reported as additive variance shares.

## Native result and executed harmonic diagnostic

The [fixed-bank repeated-input solve](QUARTIC_REPEATED_INPUT_NATIVE_V1_RESULT.json) passes exact partial-trace replay at1.7e-15 and its normal equations at6.2e-16. It improves its own isotropic objective, but native write error worsens from56.91% to96.59%. The improvement and10% fidelity predictions fail. This is not an unconverged writer solve; the changed metric is insufficient for native extraction in the current fixed feature bank.

The [exact harmonic split](QUARTIC_HARMONIC_NATIVE_V1.json) explains another limitation of the simple isotropic picture. Keeping only the radial and quadratic components gives148.7% native write error and114–202% error in paired changes. The higher harmonic remainder is essential and cancels other components on these inputs. Reconstruction holds within9.1e-17 and the harmonic quadratic's trace is zero within4.6e-16. Component norms are not additive variance shares on the native distribution. Per the preregistration, no rank fit to the inadequate lower component is pursued. The generic harmonic decomposition is still exact; its lower-degree truncation is what failed.

## Fit the harmonic remainder and restore exact lower terms, 12 September 01:14

The lower-only truncation failed, but that does not test a representation retaining an approximate harmonic quartic alongside exact lower terms. We have now executed that distinct test on the frozen V2 and LBFGS V1 banks. If $A=\operatorname{Tr}T$ and $B=\operatorname{Tr}S$ are partial traces of fully symmetric quartic tensors in dimension $d$, their trace-free projections satisfy

$$
\langle H_4(T),H_4(S)\rangle
=\langle T,S\rangle-\frac{6}{d+4}\langle A,B\rangle
+\frac{3}{(d+4)(d+2)}\operatorname{tr}(A)\operatorname{tr}(B).
$$

This follows by subtracting the orthogonal projection onto tensors containing an identity-matrix factor, using the earlier harmonic split. The [metric implementation](quartic_harmonic_metric_v1.py) agrees with explicit symmetrization and projection within5.7e-16; projected traces vanish within8.9e-16 in the [dense control](QUARTIC_HARMONIC_METRIC_V1_CONTROL.json).

The experiment solves output coefficients for the harmonic-projected features and target, then evaluates the fitted quartic plus the exact lower component of the target-minus-fit. The native radius and downstream denominator remain explicit. The diagnostic charges3,246,912 floats: the592,704-float bank plus two dense1152-by1152 target-trace matrices. It is not a claimed compact replacement. Output coefficients are fitted using weights only; native inputs are loaded afterward for validation.

[Native results](QUARTIC_HARMONIC_FIT_V1_RESULT.json) pass replay and solve checks but fail improvement and10% fidelity. V2 error changes from22.81% to **67.79%**; LBFGS V1 changes from26.65% to **68.16%**. Correction-write norms are46.3% and42.9% of reference-write norm. These norms are not variance shares. Exact lower components do not repair the approximation of the harmonic remainder on the native distribution; the combined error worsens through cancellation. This is not a contradiction of the coefficient-space orthogonality identity.

The negative covers these fixed banks and this exact linear harmonic fit. Jointly learning harmonic factors remains untested. We will not compress the unsuccessful dense correction or treat it as an identified circuit. This also distinguishes the result from the previous isotropic solve, which forced the same output mixture to account for all harmonic degrees.

## Red-team using learned input banks, 12 September 01:08

The initial-bank repeated-input refit failed native fidelity. The [new learned-bank test](QUARTIC_LEARNED_REPEATED_V1_RESULT.json) applies the same exact isotropic objective to the frozen V2 and LBFGS V1 readers. This directly tests whether the initial spectral bank caused that failure. The saved native target traces and fixed output writers are identical; coefficient target correlations are reconstructed as C=KA from each saved coefficient-optimal mixing. This identity applies at the fixed bank only, and must not be differentiated as a target formula when moving readers.

A low-rank trace implementation avoids materializing32 dense1152-by1152 feature matrices. In an orthonormal eigenframe B with eigenweights n, the partial trace of the fully symmetric tensor for the squared quadratic has the same frame and eigenweights (n times sum(n) +2n squared)/3. The [implementation](quartic_repeated_lowrank_v1.py) uses this form for exact trace inner products. Dense controls agree within the registered1e-10 tolerance. Saved objective and native-write replays agree within1.6e-15; normal residuals are below1.9e-15.

Both behavioral predictions fail. V2 native write error changes from22.81% to **41.90%**; LBFGS V1 changes from26.65% to **88.33%**. These are weight-only linear mixing fits followed by validation on the reused native cache, with no native data in the objective. Their own isotropic objectives improve substantially, so the miss is not an unfinished linear solve.

The earlier negative was not confined to the original spectral bank. It now covers these two learned frozen banks as well. It still does not exclude learning different nonlinear factors under the repeated-input metric, another input-space prior, or other decomposition families. The isotropic law was a synthetic discovery assumption, not a claim that actual normalized language activations have that distribution. The long coefficient-only fit and residual-scored replacement remain separate managed experiments.
