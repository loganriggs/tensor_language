# Small upstream readers preserve most logit effects, but miss a behavioral tolerance

21 September 2026, 04:18 UTC.

**The continuation candidate's two upstream source reads can be approximated economically on native inputs. But the quadratic approximation is not uniformly better than a much simpler linear control.** On fresh FineWeb documents, the quadratic version passes the logit-effect fidelity test while understating continuation prediction damage beyond the registered tolerance. The failure remains part of the result.

## What was extracted?

The original leading continuation feature reads a previous MLP contribution $m_0$ and last-MLP input $h$. With native RMS denominator $s(h)$, its scalar activation is

$$
\phi=\left(\frac{a^\top h-\tfrac12 a^\top m_0}{s(h)}-\alpha\right)
\left(\frac{b^\top m_0}{s(h)}-\beta\right).
$$

Its output is a fixed vector $w$ times $\phi$. We folded the two source reads exactly into the normalized input $z$ of the preceding MLP:

$$
a^\top m_0=z^\top Q_a z,\qquad b^\top m_0=z^\top Q_b z.
$$

The new experiment replaces these two quadratic forms with small programs. It still requires native $z$, $h$ and $s(h)$. In particular, $h$ itself contains earlier computations; this is not a standalone replacement of the preceding MLP or a whole-model compute saving.

## Constant, linear and quadratic parts

For each form $Q$, let $\mu$ and $\Sigma$ be the mean and covariance of the existing calibration inputs, and $\delta=z-\mu$. The exact expansion is

$$
z^\top Qz=c+\ell^\top\delta+
\left(\delta^\top Q\delta-\operatorname{tr}(\Sigma Q)\right),
$$

$$
c=\mu^\top Q\mu+\operatorname{tr}(\Sigma Q),\qquad \ell=2Q\mu.
$$

The constant and linear terms are retained exactly. A rank-$r$ approximation represents the remaining term with $r$ squared linear features. We compared eigen-directions of $Q$ (isotropic) with those of $\Sigma^{1/2}Q\Sigma^{1/2}$ mapped back to the input coordinates (covariance-informed). We also kept **rank zero**, which has only the constant and linear terms.

All coefficients come from original weights and calibration moments; there is no output-target regression. The covariance coefficient metric corresponds to Gaussian quadratic variance, but native inputs are not assumed Gaussian. We measure their actual errors separately.

On calibration inputs, covariance rank16 leaves8.71% relative error in downstream feature variation versus12.83% for isotropic rank16 and20.74% for rank zero. Exact expansion replay is approximately $10^{-15}$.

## Frozen native-effect comparison

We froze all programs before evaluating32 unused FineWeb documents and16 previously evaluated stdlib function snippets. The reference is removal of the **exact native leading feature**, not removal of the entire original projected operator. Each approximation removes its own predicted feature write; final native RMS normalization and softcap remain active.

| Source program | FineWeb: all-site effect error | FineWeb: continuation effect error | Code: all-site effect error |
|---|---:|---:|---:|
| Constant + linear | 13.93% | **8.70%** | 9.76% |
| Isotropic rank16 | 13.38% | 9.76% | 4.65% |
| Covariance rank16 | **9.06%** | 10.06% | 4.76% |
| Covariance rank64 | 6.76% | 7.96% | 3.10% |

Effect error is relative disagreement in vocabulary-centered logit changes. The rank16 covariance program passes the registered15% bar for all, continuation and spaced-word sites in both domains. The rank-zero control also passes that bar.

However, continuation cross-entropy damage on fresh FineWeb is **0.10038 nats/token for the native feature and0.07486 for covariance rank16**. The difference0.02553 exceeds the registered0.02 tolerance. Covariance rank64 still differs by approximately0.02016. These are fidelity failures, even though the smaller removal causes less prediction damage.

A paired10,000-document-draw bootstrap supports the distinction: covariance16 improves overall FineWeb effect error relative to linear (ratio interval0.575–0.745), but worsens continuation-specific error (1.041–1.299). Its continuation CE difference from native has interval−0.03375 to−0.01836. These are descriptive within-panel intervals, not population or simultaneous guarantees.

## Cost and meaning

The two rank16 source programs store39,200 weight coefficients, plus1,152 shared mean entries and34 other constants, and compute32 squares. Rank zero stores2,304 weight coefficients, the shared mean and two constants, with no source quadratic products. Both still need the last-MLP input reader, normalization and output writer.

This result narrows the interpretation: much of the source behavior on model states is captured by the constant and linear expansion. Adding quadratic structure improves overall reconstruction but does not automatically preserve the continuation effect better. We should test which input-role changes the small program predicts, rather than infer a nonlinear mechanism from a lower reconstruction error.

Separately, the frozen original continuation feature passed its point-estimate behavioral checks on the stdlib panel: removal damage0.07847 at continuation sites and−0.01879 at spaced-word sites. The latter is close to the0.02 bound; the same-current-token control had insufficient code coverage and remains inconclusive. That panel is a code shift, not evidence of no pretraining overlap.

## Evidence and execution

- [Calibration fit and declared scope](../../direct_tensor_match/MIDPOINT_SOURCE_COVARIANCE_FIT_V1.json).
- [Frozen panel and program hashes](../../direct_tensor_match/MIDPOINT_SOURCE_NATIVE_PLAN_V1.json).
- [Native results and document records](../../direct_tensor_match/MIDPOINT_SOURCE_NATIVE_V1.json).
- [Paired uncertainty audit](../../direct_tensor_match/MIDPOINT_SOURCE_NATIVE_AUDIT_V1.json).
- [Original feature on stdlib](../../direct_tensor_match/MIDPOINT_STDLIB_CONTINUATION_NATIVE_V1.json).

FineWeb uses skip11000 rows144–175; stdlib uses the existing16 deterministic function snippets. All use256-token contexts, scoring positions16–255. Native source-form replay passed below $10^{-4}$. The48-capture managed job completed successfully. Exported low-rank matrix slices were cloned before freezing so serialization did not retain unused backing matrices; coefficients were unchanged. An earlier stdlib wrapper preflight lacked explicit prediction keys and was corrected before execution; an enqueue command from the wrong directory queued nothing. No thresholds changed after outcomes.

## Compact executable export

A subsequent exact algebraic rewrite absorbs calibration centering into the two source biases and linear readers, and absorbs the inverse QR map into the residual writer. The [standalone executor](../../direct_tensor_match/source_interface.py) needs only native $z,h$ and its compact parameter dictionary. It computes the original recipient RMS explicitly.

The complete linear-source program stores **4,612 scalars**, with no source squares and one final variable product. Rank16 stores **41,508 scalars**, with32 source squares and one final product. These totals include source coefficients, the h-reader, residual writer and centering constants; they exclude the native producers of z and h. This is representation simplification, not a new fitted candidate or a cure for the behavioral failure.

[Export replay and literal prices](../../direct_tensor_match/MIDPOINT_SOURCE_INTERFACE_V1.json): calibration scalar replay below $4\times10^{-15}$ and standalone synthetic residual-write replay below $5\times10^{-14}$. No full QR matrix or calibration activation cache is needed at execution.
