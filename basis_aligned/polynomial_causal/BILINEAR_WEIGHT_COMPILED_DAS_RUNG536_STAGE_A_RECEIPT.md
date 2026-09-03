# Rung 536 Stage-A receipt: product-space DAS compiles exactly into quadratic weights

**Completed:** 2026-09-03 13:29 UTC

All four registered CPU toy checks pass. In a deterministic float64 bilinear MLP with input dimension 12, product
dimension 24, output dimension 17, and a planted three-dimensional rotated circuit:

- direct projected product-space output and the compiled quadratic implementation agree to `8.53e-14` maximum
  absolute error;
- direct and compiled donor/base interchange agree to `9.95e-14`;
- an orthogonal rotation within the learned subspace changes output by at most `8.53e-14`;
- gradient-based alignment recovers the planted projector with overlap `0.9999999999999997`;
- held-out projected-interchange relative error is `4.64e-16`.

The result is byte-identical on a second run. This validates the algebra and toy optimizer only. It does not
authorize a real-model fit. Stage B still requires a larger-document split-half power gate because the completed
MLP0 49-term probe had only `0.106` cross-half stability and localized 0/32 circuits.

- preregistration: `aa610d826a36cc2a73e73165691f171a05a210f65ffd97edd6a1e3300f6d1c49`;
- source: `bc4d2048907ed63ee69cf49a0484134c95956a7aa827f7c6b5ab590cc0f0bc77`;
- result: `6c215fdd7d471e3eb530109e1fb3e5c2391a596c720a33861a07a7604c9b9226`;
- new model forwards: `0`.
