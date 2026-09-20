# A shared quadratic output path helps, but feature transfer remains incomplete

2026-09-20 21:58 UTC

We implemented a genuine graph edit: the six existing quadratic product nodes
now feed a direct linear output path as well as the quartic stage. Their
computation is shared. Exact Gaussian moments from the weights determine the
new readout; no text-output fitting or optimizer search was used.

The preregistered rank-two path raises stored coefficients from19,632 to21,948
while retaining10 product nodes. Literal DAG evaluation matches the tensor
implementation within6e-16; its price includes20,774 additions. The common
fixed vocabulary mapping remains excluded from both program counts.

| Metric | Original ten-product program | Rank-two skip |
|---|---:|---:|
| Gaussian relative reconstruction error | 13.01% | 11.53% |
| FineWeb CE added above native | .02015 | .01403 |
| FineWeb KL from native | .01997 | .01384 |
| Code CE added above native | .01061 | .00122 |
| Code KL from native | .05907 | .04207 |
| Code feature1 removal-effect error | 48.06% | 42.83% |

The Gaussian squared-error reduction is21.5% for rank2 and25.6% for rank6.
The solve and export checks pass. However, the registered transfer prediction
requires feature1 error below40%; rank2 FAILS. Code prediction KL also remains
above the earlier .02 preservation bar. Better average CE is not sufficient.

The hyperparameter tradeoff is informative. Code feature1 errors for ranks
1/2/4/6 are40.18%,42.83%,43.48%,46.09%, respectively, even while overall KL
improves. A larger readout optimized for total output fidelity need not improve
a particular feature's intervention. Rank1 is not retrospectively promoted.

A follow-on paired CPU audit finds KL and logit MSE improve on every one of the
16 documents in each panel. Rank2 minus baseline CE differences have95%
bootstrap intervals[-.00915,-.00301] on FineWeb and[-.01542,-.00321] on code.
These panels have been reused and code files are related: this is diagnostic
support for the graph edit, not independent confirmation of generalization.

The circuit goal remains incomplete. This result supports explicit reuse and
native branch approximation; it does not supply semantic specificity, reliable
all-feature OOD transfer, or a cheaper complete model implementation.

[Fit](../../direct_tensor_match/QUADRATIC_SKIP_FIT_V1.json),
[native tests](../../direct_tensor_match/QUADRATIC_SKIP_NATIVE_V1.json),
[literal DAG](../../direct_tensor_match/QUADRATIC_SKIP_GRAPH_V1.json),
[paired audit](../../direct_tensor_match/QUADRATIC_SKIP_PAIRED_GAIN_V1.json).
