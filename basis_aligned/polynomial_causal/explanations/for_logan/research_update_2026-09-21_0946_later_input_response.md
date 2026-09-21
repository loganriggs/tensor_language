# Checking the graph's response to its later native input

21 September 2026, 09:46 UTC. **The smaller graph also loses fidelity in how the third component responds to the supplied later state.** This complements the earlier-input derivative tests; it does not replace the stronger fresh intervention results or establish a standalone circuit.

The six-arm empirical-source direction fit is still running. Its first lambda-one arm reaches 14.44% third-component error, above the 13.14% relative-baseline allowance. That is an interim result, not the selected primary or final verdict. The independent diagnostic below concerns the already frozen earlier programs.

**Which input is being varied?**

The conditional extracted program takes two native model states: the earlier MLP input $z$, from which it computes quadratic measurements, and a later residual state $h$. Previous Jacobian tests varied $z$ while holding $h$ fixed. Here we vary $h$ while holding $z$ and its measurements fixed.

For one component, write

$$
s(h)=\sqrt{\frac{h^\top h}{d}+\epsilon},\qquad
t=h^\top a-\frac12q_a(z),
$$

$$
A=\frac{t}{s}-\alpha,\qquad
B=\frac{q_b(z)}{s}-\beta,\qquad
\phi=AB.
$$

Here $a$ is a fixed later-state reader, $q_a,q_b$ are the source measurements, and $\alpha,\beta$ are fixed offsets. The exact conditional derivative is

$$
\nabla_h\phi
=\frac{B}{s}a
-\frac{tB+Aq_b(z)}{d\,s^3}h.
$$

The second term includes the change in RMS normalization; it must not be dropped.

**Completed CPU result**

Relative derivative error is the Euclidean norm of the student-minus-original derivative, divided by the original derivative norm, aggregated across 448 previously examined states and all later-state coordinates.

| Program | Component 1 | Component 2 | Component 3 |
|---|---:|---:|---:|
| Earlier shared graph, 512 products | 3.30% | 6.47% | 12.79% |
| Compact graph, 399 products | 3.20% | 7.04% | 18.51% |

The third component's sensitivity loss is therefore visible at both supplied-input interfaces, not only in its scalar value. This is descriptive evidence; no new acceptance threshold was introduced after seeing these numbers.

Five seeded small examples compare the analytic derivative to automatic differentiation. Relative discrepancies are below $1.7\times10^{-16}$. The native value calculations reproduce the established component-error figures. Both derivative evaluations use float64 and two CPU threads; no new native-model forward or GPU process was launched.

**Limits and next decision**

In the actual model, $z$ and $h$ are connected by upstream computations. These two conditional derivative tests do not supply the missing chain rule through that computation, and should not be combined into an end-to-end claim. The earlier native removal and donor-swap tests remain the stronger behavioral evidence.

The running direction fit will be audited against its original component and coefficient requirements before any fresh validation. If source fitting improves while native component behavior remains deficient, that supports testing a loss or structure that explicitly accounts for the later component operation; it does not justify relaxing the existing baseline.

Evidence: [CPU diagnostic](../../direct_tensor_match/LATER_INPUT_RESPONSE_V1.json), [executable derivation and autodiff checks](../../direct_tensor_match/audit_later_input_response.py), and [registered direction fit](../../direct_tensor_match/EMPIRICAL_SOURCE_DIRECTIONS_PLAN_V1.json).
