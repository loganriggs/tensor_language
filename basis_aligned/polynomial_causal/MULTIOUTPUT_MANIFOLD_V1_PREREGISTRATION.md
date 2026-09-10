# Overlapping quadratic blocks: manifold optimization

Registered 10 September 2026 before native execution. User weight-first priority.
The old 540-second fit remains unconverged. This changes optimizer coordinates,
not the 16 blocks × 16 shared readers × 4 symmetric quadratic outputs per block,
the full-unembedding coefficient norm, or the explicit component-energy penalty
lambda=0.01. Old objective0.914566897645157; raw error0.9137697876842366.

Within each block, represent the row readers by an orthonormal frame E and
the four quadratic cores by unit Frobenius symmetric matrices C. Dense cores
absorb the exact QR change of coordinates; different blocks may overlap.
The saved gauge transport has function and penalty errors below3e-15.
No feature-family restriction is added for full-rank blocks; rank-deficient
blocks can use an orthogonal completion and rank-deficient cores.

Minimize on the product of row-Stiefel manifolds and core spheres, with PR+
conjugate gradients, projected transport, beta clipped to[0,10], and restart
unless the direction satisfies slope<=-0.1||gradient||². QR/sphere retraction,
Armijo coefficient1e-4, at most25 halvings, next step hint1.5×accepted step;
initial hint0.01×capture/(-slope), capped by step1e6 and displacement norm1.
Solve the conditional output writers exactly at every evaluation. First-order
envelope gradients are valid; no detached-writer reduced-Hessian claim.

Each feature has coefficient norm1, hence Gram diagonal1. With64 features,
the regularized Gram condition is bounded by(64+0.01)/0.01=6401. This is a
conditioning safeguard, not a convergence theorem. Original condition~32.

CPU controls: retraction finite derivative, tangent constraints, nonlinear toy
objective descent, and40-step vs20+20 resumed state identity. Native initial and
final objective replay, manifold constraints, monotonicity. No body forwards,
token labels or data access. Native240-second fit,900-second overall alarm;
save model, writer and resumable CG state, about8MB. Original377344 floats:
294912 readers+8704 core+73728 writer. Inherited540.148-second fit is charged.

Predictions scored without revision:

- A: initial objective/raw bridges, final replay, manifold/Gram-diagonal error
  and maximum objective increase all<=1e-10.
- B: A plus projected max gradient<=1e-7 and relative stationarity<=1e-4,
  plus relative objective change over five diagnostics<=1e-5. Relative
  stationarity is max(||g_E||sqrt(16×16),||g_C||sqrt(16×4))/raw capture.
  Diagnose every5 accepted steps. This is local stationarity in new coordinates,
  not numerical equivalence to the old raw-gauge stationarity statistic.
- C: A plus objective improvement>=1e-6 and raw capture no more than1e-6
  below the old0.0862302123157634.

Budget or line-search failure is unfinished optimization, not absent structure.
If B fails, inspect accepted-step progress, gradient trend and line-search
cost before any extension. If B passes, decompose common vs centered capture
and inspect within-block output function use. Restart stability and all four
circuit properties remain untested for this family.
