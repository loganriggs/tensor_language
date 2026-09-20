# Reuse linear and quadratic nodes at matched output rank

2026-09-20 22:00 UTC. Quadratic-only skip improves overall fidelity but fails
registered code feature1error<.40. Test whether excluding the already computed
linear nodes restricts its benefit. No new input readers or product nodes.

Use phi=[A x; B x; (A x)*(B x)] with18 entries instead of just6 quadratic
products. Fit a rank1/2/4/6 direct readout of phi-Ephi to the SAME frozen
base residual using exact calibration-Gaussian moments. Rank2 is primary;
its extra coefficients versus quadratic-only rank2 are exactly24, and both
have10 products. Total costs19632+1170r, with factorized readouts for all ranks.

Reuse archived exact native-minus-base quadratic cross moments. Compute the
12 new linear cross moments via directional derivatives of the exact Gaussian
mean; include nonzero linear/quadratic cross-covariance in G. Do not fit the
stages independently or assume Gaussian polynomial features are orthogonal
without centering and Hermite conversion. CPU covariance quadrature is implemented.

Registered before native mixed-skip outcomes:
- pred_a_exact: full covariance solve relative residual<1e-8, positive G,
  rank2 analytic Gaussian gain >= quadratic-only rank2 gain minus1e-8 relative.
- pred_b_gaussian: primaryrank2 fresh artificial-probe MSE <= quadratic-only
  rank2 MSE on same8192 draws(seed260930); no fitting to probes.
- pred_c_transfer: rank2 code feature1 native-removal error<.40 and reused
  FineWeb whole-branch CE<.025. All ranks/laws reported with no rank reselection.

The same reused text panels and canonical output definitions remain fixed.
Synthetic Gaussian gains are not actual-text guarantees. A further failure
will be retained and should motivate examining metric mismatch or adding new
reader directions, rather than increasing the same readout indefinitely.
