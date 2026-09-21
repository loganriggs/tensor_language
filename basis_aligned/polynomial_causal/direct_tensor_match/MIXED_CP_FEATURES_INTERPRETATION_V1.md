**Joint feature learning improves conditional prediction and the tested component response, but misses the two-start component criterion and does not recover the global polynomial.**

Exact coefficient-plus-shiftedGaussian objective, two archived CP starts,100Muonsteps each, same512terms/1536products/2385920coefficients. Runtime387.48seconds, peak6.49GB. No empirical output labels used in fitting; input mean/covariance and the16fixed output readers remain data-informed. No graph edit or price reduction.

|Metric|Seed1001 initial→final|Seed1002 initial→final|
|---|---:|---:|
|Opened text aggregate error|9.79→6.32%|10.03→6.59%|
|Root1 same-token response error|14.27→9.72%|16.29→11.78%|
|Root1 sensitivity-weighted error|23.11→10.00%|25.64→10.78%|
|Sampled coefficient error|99.60→98.27%|99.39→98.40%|
|StandardGaussian function error|98.76→99.45%|98.16→99.31%|

Integrity PASS. Registered learning gate FAIL: omitted-constant objective improves0.452%/0.512%, below the preregistered1%bar, although both text improvements are substantial. Do not rewrite this threshold after seeing results or equate the failed objective-gain gate with no learning. Registered component gate FAIL: seed1001passes its response<=10% and sensitivity<=15.993% criteria, seed1002fails response. The first sensitivity value is10.00495%, not below10%; the registered sensitivity bar was15.993%. No rounding to manufacture a pass.

The older384-product empirical graph has8.13%aggregate text error,15.32%root1same-token response and14.54%sensitivity error. New conditional fidelity is better in several measurements, but product cost is4times larger and coefficient storage about7.4times larger. These do not establish a better all-objective simplicity frontier. Whole16-output component fidelity, untouched OOD, native finite removals and composition remain unproven.

Both regularized coefficient objectives improve relative to the initial guarded fits (changes-.003543/-.003728). Lambda remains fixed, so this was not a maintained coefficient constraint. The sampled near98%errors and standardGaussian near99%errors still show poor global approximation. Exact native teacher self-norm is not available for a normalized full-Frobenius certificate.

**Restart redteam:** with the learned dictionaries, whole-function cosine under the calibrationGaussian rises to.99986 and relative difference falls to1.68%; centered functional difference is2.72%. Yet coefficient-space cosine falls from.814 to.651, with84.39%relative difference, and only3/512term contributions match above.9cosine. Gaussian term matches above.9 also fall to5, although50subspace canonical directions exceed.9. Thus the fits converge on similar conditional functions without converging on global polynomials or individual reusable units. This is candidate-candidate comparison, not teacher accuracy.

Scientific decision: the positive response change justifies an immediate frozen finite-removal check, using both starts and the existing original native projection/denominator and opened FineWeb/code panels. Retain the previous failed capitalization selectivity; changing the approximate executor cannot repair the native reference's semantic selectivity. Do not select onlyseed1001or rename the unit. A separate exact support-exchange implementation has passed five exhaustive small controls and is preserved as the next structural alternative; it has not been run on the native shared hierarchy.

[Native results](MIXED_CP_FEATURES_NATIVE_V1.json) · [Preregistered fit](MIXED_CP_FEATURES_NATIVE_PLAN_V1.md) · [Restart comparison](CP_RESTART_IDENTITY_MIXED_V1.json) · [Graph-edit controls](SPARSE_SUPPORT_EXCHANGE_CONTROLS_V1.json) · [Frozen removal plan](MIXED_CP_REMOVAL_PLAN_V1.md).
