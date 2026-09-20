# Sparse quartic stability audit — 2026-09-20 15:59 UTC

Audit all32 four-product exports from the16-context replication. Compare direct float32 execution on2,048 fixed Gaussian points against float64 canonical polynomial execution; also compare coefficients in exact Gaussian metric. Apply independent Gaussian perturbations at relative RMS1e-6/1e-4/1e-2 to the bank and root, three seeds each; measure function change relative to the original program. Normalize perturbations by each factor's overall RMS, not per-entry division near zeros. Record cancellation via sum of product-term function norms divided by total function norm.

Compare quadratic features from the two gauge-search restarts in each context by best sign/permutation Gaussian correlation. Both start from the same feature span; this is a restrictive agreement test, not a general identifiability theorem.

Predictions before execution: every float32 program error<1e-5; every1e-4 factor-noise function change<1e-3; every matched feature correlation>.99. Preserve failed predictions. This distinguishes numerical fragility from residual basis freedom; neither stability nor agreement establishes semantics or causal utility.
