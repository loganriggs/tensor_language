# Residual structure transfers, but a corrective computation remains unidentified

22 September 2026, 01:00 UTC. CPU follow-up to CANDIDATE_RESIDUAL_DISTRIBUTION_V1.json. Uses the same frozen CP512 seeds1001/1002, 6144 calibration states and2048 previously opened evaluation states. No model coefficients changed or exported.

Question: are the remaining errors independent fitting noise, an easily corrected bias, or shared output directions? This changes the stable-identification and reusable-computation search: stable residual writers can focus a later search for missing scalar computations, but do not specify those computations.

| Measurement | Seed1001 | Seed1002 |
|---|---:|---:|
| Original evaluation error |6.322%|6.588%|
| Error after subtracting calibration residual mean (diagnostic)|6.263%|6.523%|
| Calibration/evaluation mean residual cosine|0.287|0.292|
| Evaluation error energy captured by 1 calibration residual direction|62.86%|60.65%|
| Captured by 4 directions|84.76%|84.15%|
| Captured by 8 directions|95.14%|95.18%|

Residual matrices from the two restarts have cosine0.916 on calibration and0.956 on evaluation. Averaging predictions yields6.384% evaluation error, worse than the better single candidate6.322%, while requiring both programs. This does not prove an optimization limit or rule out substantially different initialization/optimization; it demotes repeating these two similar fits or ensembling them as a simplicity route.

The calibration residual mean has norm0.505/0.566 versus3.170/3.259 on evaluation in the fixed normalized units. The earlier finding that evaluation mean accounts for27–28% of evaluation error therefore did not imply a transferable bias repair. Subtracting the calibration mean changes the functional class by adding a constant; the small gain does not warrant adopting it.

For the output-subspace diagnostic, let calibration residuals be R_cal and take V_r as its leading right singular vectors, without centering. Measure

$$
\frac{\|R_{eval}V_r\|_F^2}{\|R_{eval}\|_F^2}.
$$

The basis depends only on calibration, but R_eval V_r uses the unknown evaluation residual. It is an oracle amplitude calculation, not an executable correction or generalization score. In particular, the rank4 oracle remaining errors2.47/2.62% cannot be reported as achieved candidate errors. A candidate must learn scalar coefficient functions of the original input without evaluation labels and pay for their computation.

This calculation emphasizes high-energy residual directions. It does not repair or identify the low-energy output features with high relative error, and their separate ledger remains necessary. Because the same data-informed metric and similar CP parameterization were used in both restarts, shared residuals could reflect a common objective mismatch or shared model bias, not a unique native circuit.

Next discriminating action: interpret the already-running shared-direction fit and queued CP-parent refit before launching another fit with the same objective. If learned shared directions do not improve feature and intervention errors, prioritize explicit residual components or an output-balanced weight metric with actual native response checks. A fresh-prefix panel remains necessary before promotion; all measurements here use opened data.

Reproduction: audit_residual_subspace_transfer.py writes RESIDUAL_SUBSPACE_TRANSFER_V1.json. Both64-input-token and65-token-prefix hashes match caches/labels. Projection energy and orthogonal complement sums agree to1e-12. No GPU work;0.715 seconds CPU analysis in this run. Rank16 projection closes to numerical precision, verifying accounting but not validating a model.
