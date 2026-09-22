# Weight-derived output-specific correction screen

22 September 2026, 02:13 UTC. After matched output balancing failed, test a different structural hypothesis: missing effects in output coordinates 4–15 can be represented by separate low-degree corrections without changing coordinates 0–3. Frozen CP512 parents seeds1001/1002; selected native purequartic16outputpath unchanged.

Use existing calibration Gaussian x=mu+Sz. Compute exact mean, linear Hermite and quadratic Hermite coefficients of native-minus-parent from weights, with no text labels fitted. For each of outputs4–15 retain rank8/16/32 signed eigenterms of its symmetric quadratic residual, ordered by eigenvalue magnitude. Add exact residual mean and linear terms. Evaluate allthree fixed ranks, plus affine-only and fullquadratic diagnostic ceilings; do not select a candidate on evaluated text.

For z standard Gaussian, correction is m+l^Tz+sum_j eigenvalue_j[(v_j^Tz)^2-1]. Fold S inverse and mu into original-coordinate affine nodes; a rankr correction has12r square products,12(r+1) linear readers, and12 output biases. Correction writes zero into coordinates0–3 exactly. This protects those selected numerator coordinates, not full-model effects after nonlinear normalization. Nonhomogeneous corrections approximate the local data-informed law; native evenness/homogeneity are not claimed.

Controls: five affine-CP toy families, independent tensor-product Gaussian quadrature verifies Hermite projections. Native coefficient symmetries, finite predictions, exact zero correction on0–3; compiled original-coordinate replay. CPU only, no queued helpers modified.

Screen criterion: fixed primary rank16 reduces RMS relative value AND matched-response errors over4–15 by>=15% for BOTH parents on the original opened2048state panel/all-output existing30pairs. Larger16384state/2494matchedpair panel is secondary opened diagnostic. Failure at fullquadratic ceiling rejects this correction family underthismeasure; lowrank-onlyfailure suggests rankcost problem. Report storage/productcost and allranks. No semantic, OOD or native-removal adoption follows; those need separate tests if this screen passes.
