# The small quartic program preserves much of a native branch's effect

2026-09-20 21:36 UTC

We installed the frozen polynomial approximation as a replacement for only
the pure MLP16-to-MLP17 quartic branch. Attention, residual cross terms, biases,
the actual RMSNorm denominator, final RMSNorm and softcap remain native.
These are 16 previously unused FineWeb documents, context128.

| Intervention | CE added above native (lower is better) | KL from native | Top-token agreement |
|---|---:|---:|---:|
| Exact branch replay | 0.000000015 | numerical zero | 100% |
| Remove branch | 0.127850 | 0.141877 | 87.16% |
| Ten-product replacement | 0.020146 | 0.019965 | 96.29% |
| 26-product replacement | 0.016752 | 0.016384 | 96.34% |

The registered CE<0.02 condition fails narrowly for the ten-product program.
Do not round this into a pass. Its document-bootstrap 95% interval is
[0.01276,0.02776]. Its post-softcap logit disturbance is 40.4% of removal's,
with bootstrap interval[35.0%,45.2%], passing the registered half-size bar.
The exact instrument replay error is1.06e-7; output-frame solve error1.77e-15.

This advances extraction evidence: a small executable approximation can replace
one algebraically defined branch and preserve much of its native effect. It is
not yet a whole-block speedup, OOD result, semantic circuit, or selective edit.
The subtraction instrument still computes the native branch.

The next test compares removal of each learned output-shared scalar feature
with removal of the corresponding native component, using the same background.
A CPU diagnostic already finds an important difference: the first two scalar
features have18–26% prediction error, whereas the fourth has53–54%. Stable
fitting directions do not by themselves imply accurately extracted features.

[Native result](../../direct_tensor_match/NATIVE_QUARTIC_BRANCH_V1.json),
[uncertainty](../../direct_tensor_match/NATIVE_QUARTIC_BRANCH_UNCERTAINTY_V1.json),
[next intervention](../../direct_tensor_match/NATIVE_MODE_INTERVENTION_PLAN_V1.md).
