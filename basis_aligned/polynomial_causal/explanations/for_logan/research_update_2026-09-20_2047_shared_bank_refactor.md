# Shared products simplify the quartic; coefficient error can mislead

2026-09-20 20:47 UTC

The joint bank refactor fit all four quadratic features together, allowing them
to reuse primitive products. Sixteen registered CPU fits compared widths 8/12,
Adam/Muon, two rates and two seeds. Selection used weight-space loss only.

| Bank products | Total products | Coefficients | Fresh64 error | Fresh256 error |
|---|---:|---:|---:|---:|
| Original 16 | 26 | 47,312 | 17.44% | 17.90% |
| Shared 12 | 22 | 38,144 | 17.95% | 18.65% |
| Shared 8 | 18 | 28,912 | 25.86% | 26.99% |

The 12-product bank saves 19.4% of stored coefficients and four multiplications
with a small prediction cost. The 8-product bank fails the registered composed
prediction bar, despite retaining 99.77% of weighted bank coefficient energy.
The energy and cost predictions pass. Literal shared-DAG replay is below 4e-15.
Component energy ratios below 1 give no evidence of the earlier huge-cancellation
failure for these selected fits.

Muon at .03 won both widths here; Muon at .005 was worse than Adam at either
rate. Thus an optimizer label alone does not predict performance. All fits
used 500 steps and the same explicit component penalty.

The failure has a concrete mathematical explanation to test. A post-fit audit
measures exact Gaussian bank function error, including the nonzero input mean.
Width 8 has 4.78% coefficient error but 30.49% function error; width 12 has 1.32%
and 5.08%, respectively. Most of the missed function energy comes from the
internal features' means. Correcting the final output mean does not undo the
changes those internal means induce when features are multiplied together.

The next registered refinement changes to this exact noncentral Gaussian bank
metric, holding widths and root structure fixed. Its moment embedding is already
implemented and independently checked by quadrature. Composed-function tests
remain mandatory because the root-sensitivity weighting is still a surrogate.
These remain computational candidates, not identified semantic circuits.

[Refactor receipt](../../direct_tensor_match/QUARTIC_BANK_REFACTOR_V1.json),
[function metric audit](../../direct_tensor_match/BANK_FUNCTION_METRIC_AUDIT_V1.json),
[next metric correction](../../direct_tensor_match/BANK_NONCENTRAL_REFIT_PLAN_V1.md).
