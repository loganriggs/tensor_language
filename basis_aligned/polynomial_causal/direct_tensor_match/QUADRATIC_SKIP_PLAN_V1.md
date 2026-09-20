# Add an output path from existing quadratic nodes

2026-09-20 21:52 UTC. The conditional-reader test estimates calibration-Gaussian
floors7.5%/8.6% versus actual12.3%/13.0% errors. Input information alone does not
explain the full gap. Test a graph edit that reuses existing intermediate nodes.

The frozen ten-product program computes six quadratic products p, four quartic
products h and output yhat. Add W(p-Ep) to its output. This permits quadratic
nodes to skip the quartic stage, adds no scalar product nodes, and preserves the
Gaussian output mean. It is a small instance of the proposed shared DAG search.
Input directions, quartic coefficients and canonical interpretation stay frozen.

Fit W to the native-minus-student residual using EXACT noncentral Gaussian
moments, no text outputs and no Monte Carlo fitting. Let K=Cov(F-yhat,p),
G=Cov(p,p). The full readout is W=K G^{-1}. For output ranks1/2/4/6, truncate
K G^{-1/2} by SVD before multiplying G^{-1/2}. This minimizes Gaussian residual
MSE subject to the specified readout rank, with all other computations fixed.
Do not confuse output rank with number of existing product nodes.

For p=(a^Tx)(b^Tx), mean mu and covariance M, write Phi(mu)=E F(mu+noise).
The exact cross moment is

$$
\operatorname{Cov}(F,p)=(a^T\mu)D_{Mb}\Phi
 +(b^T\mu)D_{Ma}\Phi+D_{Ma}D_{Mb}\Phi.
$$

The implemented nested directional-derivative method passes independent
Gaussian quadrature at9e-16. Subtract the same covariance for the frozen student.
The constant has zero covariance; its mean-preserving adjustment is explicit.

Registered fit/screen predictions:
- pred_a_exact: covariance solve relative residual<1e-8, finite positive G,
  and analytic nonnegative Gaussian MSE reduction for all ranks.
- pred_b_gaussian: full-rank skip reduces independent artificial-probe MSE by
  at least10% relative to the frozen ten-product program, on4096 pairs of fresh
  calibration-Gaussian draws (seed260930), no fitting to probes.
- pred_c_transfer: on reused FineWeb64/256 and code panels, report full-branch
  and fixed-mode1 metrics; rank2 skip improves code mode1 removal relative error
  from.480565 to<.40 without increasing FineWeb whole-branch CE above.025.
Primary rank2 is fixed before native skip outcomes. All ranks exported/reported;
code is diagnostic after reuse, not new OOD confirmation. No semantic claim.

Literal cost: original19632coefficients/10products plus1152r+6r coefficients
for factorized readout (rank6 can use direct6912 instead), and additional linear
operations. Reuse primitive p nodes; fold -W Ep into existing output constant.
Full model instrumentation still retains native background and subtraction cost.
If this fails, preserve the result rather than refit on code or change thresholds.
