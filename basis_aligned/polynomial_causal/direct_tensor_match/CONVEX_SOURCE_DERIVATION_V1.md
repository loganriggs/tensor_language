# Conditional convex source correction

Hold first read and h fixed. For a correction direction v, the mean/affine-corrected square is

$$\psi_v(z)=(v^\top(z-\mu))^2-v^\top\Sigma v.$$

Writing the current first factor as a(z,h), second-read coefficients c change the component by

$$\Delta\phi=a(z,h)\sum_k c_k\psi_{v_k}(z)/s(h).$$

Its source derivative is also linear in c:

$$\Delta\nabla_z\phi=\sum_k c_k\left[\frac{a}{s}2v_kv_k^\top(z-\mu)-\frac{\psi_{v_k}}{2s^2}\nabla_z q_a\right].$$

Each reconstruction criterion is therefore a squared norm of an affine function of c. Coefficient geometry uses signed rank-one matrices; component and Jacobian geometry use the explicit responses above. Dividing by each original squared acceptance bar gives convex quadratics. Their maximum can be minimized without changing graph directions or product count. This isolates coefficient choice from direction choice and from joint nonlinear optimization.

The mean and covariance here are the existing affine-correction statistics, not a claim that second moments determine the complete functional metric. Functional residual rows use the actual448 opened states. Native first-read linear correction must be included in its gradient. No denominator differentiation with respect to z: h is held independently fixed, matching the existing source-Jacobian convention.

Solver controls pass ten known compatible/incompatible cases. Native quadratic construction and independent executable validation are the next implementation step. A converged numerical solution with small KKT residual supports this finite convex diagnostic; it does not certify the infeasibility of other graphs or directions.
