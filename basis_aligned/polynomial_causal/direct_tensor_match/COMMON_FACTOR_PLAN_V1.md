# Common quadratic factor hypothesis — 2026-09-20 16:02 UTC

The planted shared quartic admits f_v=q*r_v, one common quadratic times output-specific quadratics. A star-shaped sparse root permits arbitrary mixing of its noncentral features, so those feature identities are not unique. The common factor may be stable even when the rest of the bank is not.

Test this narrower, interpretable structural hypothesis on the known planted positive control first, then all16 native quartic first-row targets. Each native student has15 common-factor coefficients plus4*15 quotient coefficients =75 values, close to the76-value four-product shared-bank export. The families differ; neither is assumed to contain the other.

Randomly initialize the common factor and solve quotient coefficients at each step. Normalize the common factor to unit Gaussian quadratic norm. Tiny1e-10 relative ridge ensures differentiable solves; optimize actual reconstruction error.136 fits:17 targets, Adam/Muon, rates0.01/0.05, two restarts,600steps. Primary loss is exact isotropic Gaussian function norm; report coefficient Frobenius too. No native data samples used in training.

Predictions before execution: planted best relative error<1e-3; at least8/16 native contexts have a best error<10%; all fits finite. Null: a single reused quadratic factor is too restrictive for native targets. Compare against same-context four-product errors at nearly matched storage; do not infer cross-context transfer from independent fits. This test follows a failed feature-identity agreement audit, not a new causal screen.
