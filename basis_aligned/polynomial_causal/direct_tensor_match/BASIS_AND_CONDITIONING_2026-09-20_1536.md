# Sparse basis recovery and covariance conditioning

Two follow-ups separate questions that a single tensor-loss score cannot answer.

## Sparse structure can be hidden by a learned basis

The previous Tucker toy matched its target function and input span accurately, but its learned core had 12 non-negligible coefficients rather than the four planted interactions. We froze that fitted function and optimized only invertible changes of its input-feature and output-feature bases.

For input basis change S and output basis change V, the leaf and writer factors become SP and WV. The core transforms with the corresponding inverses. We then normalize each leaf and writer, compensating in the core, and minimize the core's L1 norm. Function preservation is algebraic; no activation examples enter training. Matrix exponentials parameterize invertible changes. We compare Adam/Muon, two rates and three seeds: 12 searches of 3,000 steps.

At each final basis, retain exactly 3, 4, 6, 8 or 12 core entries and refit their coefficients by least squares in the exact Gaussian weight metric. This is hard support with measured reconstruction loss, not a threshold-only sparsity claim.

| Four-interaction program | Relative Gaussian error |
|---|---:|
| Original learned basis, then coefficient refit | 11.985% |
| Best optimized basis, then coefficient refit | 0.00629% |

The best search used Adam, learning rate 0.05, seed 2. Outcomes vary: four-entry errors across searches range from 0.00629% to 3.025%. Before pruning, all basis changes preserve the original fitted function within $5.23\times10^{-16}$ relative error. Maximum input/output transformation condition numbers are 3.62 and 1.67; extreme ill-conditioning does not explain the observed result.

The exported hard program stores 18 input-feature values, six output-writer values and four core coefficients: **28 floats plus support metadata**, versus 36 floats for the same factors and dense core. Its four interactions are explicitly evaluated once each. An independent native-index tensor contraction confirms its Gaussian error; direct sparse execution agrees with that tensor within $2.67\times10^{-15}$ absolute error on shifted inputs.

This finds a four-interaction representation of the planted function. It does not establish unique recovery of the original feature labels, nor a native-model semantic circuit. The important consequence is methodological: **function matching followed by exact-basis sparsification can succeed where pruning the initial fitted core fails**.

Evidence: SPARSE_BASIS_SWEEP_V1.json, SPARSE_TUCKER_PROGRAM_V1.json; implementations sparse_basis.py and export_sparse_tucker.py.

## Covariance loss and optimizer conditioning are separate

The prior covariance experiment changed both the objective and conditioning. The new paired experiment holds the physical objective and initial student function fixed. Write $x=Lz$, with $LL^\top=\Sigma$. Transform teacher and student weights exactly, then optimize either in the original coordinates with the covariance metric or in whitened coordinates with the isotropic Gaussian metric. These are mathematically the same loss. No sparsity penalty is applied.

There are 120 runs over five target structures, two optimizers, matched/wide variants and three seeds per coordinate system. Monomial models ignore the width argument, so their two width variants duplicate a condition; do not count those duplicates as independent evidence. Polynomial substitution, metric congruence, factor transforms and initial-loss equality are independently checked.

Selected median covariance errors, combining the declared matched/wide conditions:

| Structure / optimizer | Original coordinates | Whitened coordinates |
|---|---:|---:|
| Coordinate quadratic / Adam | 1.351% | 0.000087% |
| Bilinear CP / Muon | 0.678% | 0.149% |
| Sparse Tucker / Adam | 2.195% | 0.363% |
| Quartic tree / Muon | 1.552% | 4.678% |
| Shared quartic DAG / Muon | 1.436% | 7.159% |

Whitening helps several quadratic conditions but worsens these quartic conditions at the tested learning rate and budget. It is not a universal optimizer improvement. The parameterized landscape, initialization expressed in the new coordinates, and optimizer step geometry matter even though the physical initial function and loss agree. Follow-up rate tuning should be paired rather than used to silently favor one coordinate system.

The covariance here is still derived from synthetic calibration inputs. A separately preregistered managed job captures actual normalized MLP17 inputs on fixed calibration/evaluation FineWeb panels, saving means, centered covariances and second moments separately. That job and the full native objective/rate sweep remain queued behind a live shared GPU job at this report's cutoff. No native covariance outcome is claimed here.

Evidence: WHITENING_SWEEP_V1.json, COORDINATE_CHECK_V1.json and COVARIANCE_CONDITIONING_PLAN_V1.md. These findings refine the ongoing two-day decomposition study; they do not reopen the deferred data-based circuit agenda.
