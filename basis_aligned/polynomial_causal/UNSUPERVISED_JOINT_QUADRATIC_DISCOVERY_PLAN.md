# User-directed unsupervised joint quadratic discovery

Both active routes: unsupervised unembedding/bilinear factor discovery, and jointQK input-product separation. The user explicitly favors discovering factors from all weights before looking at token labels. Four selected words tested the source file's example only; no further hand-selected token family is the main search.

Target T_vij=sum_k (UD)_vk sym(l_k r_k^T)_ij. Search T_hat_vij=sum_j (UW)_vj sym(a_j b_j^T)_ij. This learns shared input products and their entire vocabulary consumer profiles jointly. Native4608-product representation is the exact reference, not a new discovery. The native remainder, normalization, residual and all arbitrary constants stay charged.

Implemented and CPU-verified exact implicit Frobenius loss in joint_quadratic_fit_v1.py. For H(a,b), cross Gram equals [(l.a)(r.b)+(l.b)(r.a)]/2. Conditional least-squares writers W=D G_cross pinv(G_candidate) reproduce dense full-output fitting. EntireU enters the input-factor optimization through U^TU. This avoids allocating50304x1152x1152 (~267GB float32). No token-label selection is needed.

Next native implementation: a bounded32-product discovery block with all-vocabulary objective, native-weight-informed and fixed random initializations, frozen optimizer budget before execution. This nominates a shared component with explicit remainder; it cannot represent the full tensor to10%error, given already measured379–391-product necessary bounds for four complete token slices. Do not call a fitted low-dimensional component an identified circuit. Inspect token consumer profiles only after freezing factors; test stability and actual predicted removals/interchanges on new inputs before promotion.

The fit should alternate/reduce output coefficients by linear least squares and optimize shared a/b factors against the exact implicit loss. Normalize scale/sign gauges; retain real signed coefficients. Compare discovery against the existing native-product basis at the same component budget. Store complete reader/writer constants. Broader joint DAG sharing is a hypothesis, not guaranteed unique or globally optimal decomposition.

Other source-file ideas remain available: domain-restricted affine/quadratic behavior and simultaneous block structure. A failed primitive pair, three-product example or globally simple square basis would not reject all of them. Avoid rank sweeps whose only outcome is a nicer reconstruction curve.

QK continuation: first test position-corrected joint input spaces before interpreting the A1-to-A2 failure as semantic. V1 fit after rotary maps at query position5; A2 position8 changes that representation. Fit before those maps or transport the bases using actual rounded transforms, preserving all mixed products and native head-normalization factors. V1 remains a valid post-position product-port result, not raw-input identification.

## Addition from unembedding_factors_how.md

Read the user-provided file in full. Core folded tensor and joint product fit match the existing plan. New explicit requirements: report signed factor-to-token usage, output connection cost, and shared-linear-reader internals as separate graphs. Token support overlap does not prove one factor computes another. Consider biclustering/formal-concept organization only after stable factors; thresholded incidence is approximate and no full lattice is required.

The first fit remains unconstrained least-squares in output writers, without a sparsity penalty. Report that honestly. Add per-factor effective token participation and counts covering90%/99%of squared loadings, signed consumers, plus normalized cross-factor reader overlap. For comparisons across factors, normalize product-function scale; an empirical function RMS must identify its input sample explicitly. No claim that thresholding establishes exact sparsity.

The file proposes a stronger exact identity with factor/connection cost. Our32-product block is partial discovery with the native remainder retained; it does not solve that exact objective. For residual-realizable writes A=UW, arbitrary sparsification of A can leave the allowed output space and change intervention semantics. Any later sparse variant must either retain that constraint or explicitly price and validate a different output interface.
