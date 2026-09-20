# Sampled coefficient-loss optimization controls — 2026-09-20 16:15 UTC

Compare exact coefficient Frobenius loss with unbiased uniform ordered-entry and equal-collision-stratum loss estimators on three representable quartic targets: planted tree, planted shared DAG, and one-coordinate fourth power. The latter is a sparse sampling adversary.

36 fits:3targets ×3loss modes ×Adam/Muon ×2seeds;1200steps, lr0.05 with cosine decay,128 queried ordered entries per sampled step. Initialization paired across loss modes; fixed exact teacher norm used as denominator to isolate estimator noise. Native denominators would need separate estimation. Report final exact Frobenius error and number of sampled batches containing zero teacher coefficient energy. No best-checkpoint selection using exact losses.

Toy code materializes small canonical coefficient vectors for evaluation; it is an optimization-quality control, not a scalability test. Uniform samples weight by d^4/batch; stratified samples weight each stratum by exact population count/sample count. Entries divide canonical polynomial coefficients by permutation multiplicity; this is coefficient Frobenius, not Gaussian function loss. Sampling distributions have full support and may still miss sparse mass in finite budgets.

Hypotheses: sampled loss can obstruct recovery of sparse diagonal targets; collision stratification should improve that specific case, not necessarily dense or all-distinct structure. Compare measured recovery against exact-loss runs; do not treat failure under any one optimizer/rate as a representational lower bound.
