# Learned output-mixture LL1 fit for setting2

Registered before fitting. Twelve individual token unembedding rows folded throughMLP17 ontohead17.2W; mixed tensor12×1152×128. Rank32 per output-linked matrix block, orthogonal12×12outputbasis. All block input factors solved by SVD at every step. Normalize tensor globally only; no text fitting or behavior-based weighting.

Ten starts: identity, outputGram eigenvectors, eight seeded random orthogonal bases. Each gets15 L-BFGS-B iterations; bestthree get up tothree80-iteration cycles with output-basis recentering. Report all starts, statuses, losses and intrinsic gradients; no claim all starts converged. Intrinsic gradient<=1e-6 defines local stationarity; parameter-chart stopping alone is insufficient. Budget550fitseconds/600alarmseconds. Zero model forwards or text tokens. Save best mixed-tensor factors,491664scalars; quadratic/residual/norm/readout dependencies remain external.

- pred_a: GPU objective replay against CPU control absolute squared-error difference<=1e-8 (dimensionless normalized loss).
- pred_b: at leasttwo ofthree promoted fits reach intrinsicgradient<=1e-6.
- pred_c: best squared coefficient error <=90% of the best initial error at the identical block capacity.

Null: orthogonal output mixtures offer little compression improvement or fits remain locally unconverged. Cross-start reconstructed-function cosines are descriptive, not a semantic identification certificate. Preserve all misses; exact coefficient controls do not establish native behavioral preservation. General overlapping output blocks and producer-constrained targets are outside this restricted family.
