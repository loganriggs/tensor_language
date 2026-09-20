# Native channel-selection and exact-output-refit baseline — 2026-09-20

Question: is high full-tensor fitting error evidence against compact structure, or partly optimization failure? Compare random-start fits against width-matched students initialized with selected teacher channels. This is a baseline, not the requested random-start discovery algorithm.

Same full MLP17 numerator and exact reduced input/output frames as prior native studies. Normalize channel input vectors and absorb scales into output weights. Build exact channel quadratic Gram matrices for both coefficient Frobenius and isotropic Gaussian metrics. For each selected channel set S, optimize its output weights globally by solving the small convex linear least-squares problem:

D = C K[:,S] inverse(K[S,S]).

Use a float64 symmetric eigensolve with relative1e-10 cutoff, recording eigenvalues, rank and residual; check the result against independent implicit tensor contractions. Compare unchanged selected teacher output weights to optimal refits. Selection policies: largest individual channel energy (ignores cancellation), random seed0, random seed1. Widths128/512/1024. Eighteen fits across two metrics. Selection scores depend on the metric; therefore compare errors under both metrics and literal identical channel counts.

Predictions: exact refit never worsens its objective beyond1e-8 relative squared tolerance; Gram and implicit scores agree within1e-8 relative squared tolerance; at width1024 at least one teacher-channel Frobenius refit beats the existing best random-start error0.9548. Null: original-channel selection/refitting cannot improve that result at equal channel count. A positive result diagnoses an optimization/initialization gap, not a minimal-rank theorem or interpretable circuit. Include separate frame cost. No native forwards or checkpoint changes.
