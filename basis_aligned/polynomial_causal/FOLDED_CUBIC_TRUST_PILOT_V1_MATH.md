# Stronger optimization of the folded source-cubic bank

12 September 2026, through20:30 UTC. This is a routine experiment note, not another requested full report.

## What changed

The matrix-free exact-Newton trust-region pilot improves all four saved native fits. Stable secant coordinates improve captured coefficient energy22.75%/31.54%, versus21.70%/22.72% in the original product coordinates. All four runs reach their60-second time cap and fail unit-reader stationarity. Thus the original stalled fits were not limits on attainable capture, but this pilot still does not establish convergence or a recovered circuit.

| Parent / coordinates | Capture before | Capture after | Held-position capture after | Unit-reader gradient | HVP calls |
|---|---:|---:|---:|---:|---:|
| 0 raw | .434945 | .529338 | .291846 | .12938 | 631 |
| 0 secant | .434945 | .533897 | .292982 | 2.40898 | 354 |
| 1 raw | .291630 | .357897 | .188977 | .22479 | 644 |
| 1 secant | .291630 | .383599 | .202974 | .00777 | 352 |

Capture entries are squared coefficient norms projected into the fitted source bank, not percentages of the full target. The original estimated total target energy at the fit position was43.89, so the largest new capture is still only roughly1.22% on that denominator. This is an approximate conversion using the old coefficient-probe estimate, not a new native behavioral score.

A (instrument/nonworsening) and C (held-position gain for the training-selected secant arm) pass. B requires both10% gain and gradient<=1e-6; its convergence part fails. Total execution241.38seconds, peak PyTorch allocated GPU memory2,061,090,304bytes. This is an implementation measurement, not a ten-process memory guarantee. Best finite evaluated trial candidates were retained; they need not be accepted trust-region iterates. The two parents are old fits, not independent new discovery starts.

[Terminal receipt](FOLDED_CUBIC_TRUST_PILOT_V1_RESULT.json) · [Registered comparison](FOLDED_CUBIC_TRUST_PILOT_V1_PREREGISTRATION.md).

## Do the improved fits describe the same computation?

Not accurately enough. We compare their complete fitted coefficient functions, including private query/output maps and head identities, rather than matching source vectors alone.

If G_a and G_b are the two source feature Gram matrices, K_a and K_b their target-contraction Gram matrices, and G_ab/K_ab the corresponding cross blocks, the fitted energies and cross inner product are

$$
E_a=\operatorname{tr}(G_a^{-1}K_a),\qquad
E_b=\operatorname{tr}(G_b^{-1}K_b),
$$

$$
C_{ab}=\left\langle G_{ab},G_a^{-1}K_{ab}G_b^{-1}\right\rangle_F.
$$

These are summed over separately retained head labels. The symmetric relative function difference is

$$
\epsilon_{ab}=\sqrt{\frac{E_a+E_b-2C_{ab}}{(E_a+E_b)/2}}.
$$

Normalized Cholesky solves replace explicit inverses in the implementation. An independent dense coefficient construction agrees within5.3e-16; permutation/scaling of source factors leaves the full function unchanged within5.1e-16 squared-relative energy.

The two secant endpoints have cosine .84761 at the fit position and .83212 at the held position, but relative function differences **57.24% and60.28%**. A moderately high cosine is not a faithful replacement. Within parent0, raw versus secant have11.12% held disagreement despite cosine .99382. The complete functions also changed materially from their parents, so old behavioral claims cannot be assigned to them by source-reader similarity.

[Complete-function analysis](FOLDED_CUBIC_TRUST_PILOT_V1_ANALYSIS.json) · [Independent control](CUBIC_PROJECTED_FUNCTION_COMPARE_V1_CONTROL.json).

## Where did the gain go?

Most of the secant gain is in head13.0. Head8.2 and9.8, the earlier behaviorally validated producers, gain much less. At the held position, parent0's head8.2 coefficient capture even declines slightly. This is not a new failure of the existing extracted regional component: that component was not edited by these weight fits.

We re-scored the same frozen projected functions under the existing positive-definite4×4 downstream consumer metric

$$
H=\mathbb E[J_fF(q,f)^TJ_fF(q,f)],
$$

constructed previously from weights, independent synthetic moments and a uniform vocabulary first-state lookup. We normalize tr(H)=4 only to make the scale readable. Its normalized eigenvalues are .6993,.7285,.7969,1.7754. This is a polynomial consumer sensitivity metric, not an empirical language distribution or a theorem about finite normalized-model effects.

For a fixed source span, projection commutes with an invertible linear output transform H^(1/2). Therefore this audit reweights the same physical fitted functions; it does not refit source readers or choose a different physical output solution. The gain remains overwhelmingly concentrated in13.0. At the held position it accounts for96.55% and99.91% of the net gain under H. Simply changing to this already available metric is not evidence that the optimizer has started improving the desired8.2/9.8 generators.

[Consumer-metric audit](FOLDED_CUBIC_CONSUMER_METRIC_V1_AUDIT.json).

## Conditioning after optimization

The fixed secant rewrite started well conditioned, but optimization produced another near-dependency. At the endpoints, normalized source Gram conditions are about11996 and8367 in the secant feature coordinates. Their reconstructed raw-product banks have conditions about12.0million and9997. Reader norms stay roughly .86–1.10, so this is not merely arbitrary scaling.

Inspection finds an ordinary source product nearly parallel to the secant pair's even feature: cosine−.999815 for parent0 and−.999742 for parent1. This is a three-feature collision, not just the original two-product cancellation. A fixed one-pair rewrite cannot be assumed to regularize every subsequent iteration.

The relevant alternatives are longer adequate optimization; diverse independent starts; adaptive charts/restarts when new dependencies appear; and an arithmetic representation allowing a shared parent with multiple quadratic partners. The latter changes the model family and must be priced separately. Orthogonalizing an output feature basis alone changes execution conditioning, not the nonlinear parameter landscape.

[Conditioning audit](FOLDED_CUBIC_TRUST_PILOT_V1_CONDITIONING.json) · [Mathematical solver mapping](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_2017.md).

## Decision and scope

The ten-start small planted comparison demonstrated a concrete benefit from trust-region least squares (3/10 recoveries versus0/10 L-BFGS at the stated budgets). The native comparison now demonstrates continued optimization gain, but neither convergence nor stable circuit identity. We should retain the stronger solver as a usable tool and investigate the new feature collision before launching a large identical continuation sweep. A larger restart campaign remains justified; four short warm fits do not exhaust it.

Next implementation target: an exact coordinate chart for a small cluster of coalescing source products, with independent polynomial/gradient checks, followed by a matched native continuation or cold-start comparison. The chart must preserve the existing rank16 family; adding freely fitted polynomial partners is a distinct hypothesis. Any candidate eventually promoted must predict native removal/interchange, with original normalizers and position dependence retained. These optimizer gains cannot substitute for that circuit test.
