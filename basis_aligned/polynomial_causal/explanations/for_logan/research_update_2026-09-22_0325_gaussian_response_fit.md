# Fitting changes instead of values did not repair the quartic candidates

22 September 2026, 03:25 UTC.

**The response-aware objective is implemented correctly, but changing the output readout is not enough to fix the missing computation in the current quartic feature dictionaries.** It gives less than 1% relative improvement in the primary small-output response metric, against a registered requirement of 15%. Pooled value error increases by about 6%. Both seeds fail the improvement requirement.

This test returns to the **pure quartic MLP16→MLP17 contribution**, projected onto our 16 fixed output directions. It is separate from the preceding single-layer quadratic experiments. Each candidate keeps its 512 learned quartic products fixed and refits only their 16 output coefficients. Its computation remains 1,536 scalar products of variable quantities and 2,385,920 stored floats under our existing accounting. No model-wide circuit claim follows.

## Why try a different objective?

Value reconstruction can reward getting average outputs right without accurately matching their changes. To isolate changes while retaining exact weights-based contractions, define paired artificial inputs

$$
X=\mu+SZ,\qquad
Y=\mu+S\left(\rho Z+\sqrt{1-\rho^2}Z'\right),
$$

where $Z,Z'$ are independent standard Gaussian vectors and $SS^\top$ is the calibration covariance. For reconstruction residual $R=F-\widehat F$, fit

$$
\mathcal E_\rho(R)=
\frac{\mathbb E\|R(X)-R(Y)\|^2}{2(1-\rho)}.
$$

The denominator makes the derivative limit finite. These Gaussian pairs are an explicitly chosen fitting distribution; they are not actual matched token states, causal interventions, or a model of semantic changes.

Write $R_k$ for the degree-$k$ Gaussian Hermite component in whitened coordinates. These are orthogonal polynomial components, not the ordinary monomials of the original residual input. The Gaussian noise operator multiplies component $k$ by $\rho^k$; see [O'Donnell, Gaussian space and the Gaussian noise operator](https://www.cs.cmu.edu/~odonnell/papers/Analysis-of-Boolean-Functions-by-Ryan-ODonnell.pdf). Applying that identity gives

$$
\mathcal E_\rho(R)=\sum_{k=1}^4
\left(1+\rho+\cdots+\rho^{k-1}\right)
\|R_k\|_{L^2}^2.
$$

The constant component gets no weight. At $\rho=0$, this is variance matching. As $\rho\to1$, the weights become $k$, giving the exact squared derivative norm in whitened coordinates. Ordinary Gaussian value matching instead gives every degree, including the constant, weight one.

The implementation computes these Gram and native cross terms by exact contractions. It uses calibration mean/covariance but no actual finite-response labels in fitting. This is weights-first with data-informed geometry, not completely data-free discovery.

## Results on actual text changes

The registered primary setting was $\rho=0.5$, compared with a new value-only Gaussian readout fit on the **same frozen feature bank**. We also recorded $\rho=0,0.9,1$ and the original parent readouts. We did not select the best setting after looking at responses.

The original opened panel has 2,048 states and the existing 30 directed matched pairs. The larger opened panel has 16,384 states from 256 documents and 2,494 fixed matched pairs. Both panels are previously inspected evidence, not fresh confirmation.

| Panel | Seed | Value-only pooled value error | Response-fit pooled value error | Value-only small-output response error | Response-fit small-output response error |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1001 | 6.27% | 6.65% | 63.20% | 62.81% |
| original | 1002 | 6.52% | 6.89% | 60.29% | 60.28% |
| opened256 | 1001 | 7.33% | 7.72% | 57.57% | 57.42% |
| opened256 | 1002 | 7.52% | 7.90% | 59.98% | 59.75% |

“Small-output response error” is the RMS of the 12 per-coordinate relative errors for coordinates 4–15. A response is the difference in the selected native quartic output between the two recorded states. It is not the full transformer's normalized-logit response or a selective semantic intervention.

On the primary panel, response-error ratios are 0.9939 and 0.9998, versus the required maximum 0.85. Both fail. Value-error ratios are 1.061 and 1.056, inside the allowed 1.10 retention limit. The derivative-limit variant also leaves small-output response errors near 60–62%; emphasizing higher Hermite degrees more strongly does not produce the missing improvement in this fixed span.

Ignoring constant offsets has the expected cost: Gaussian mean error rises from roughly 0.014% to roughly 0.97% relative to the native Gaussian mean vector. That is explicitly reported, rather than describing a difference-only fit as preserving values. Adding a constant afterward would not improve differences between inputs, so it would not rescue the failed response criterion.

## What the negative result rules out

It argues against repairing these two frozen CP dictionaries merely by swapping value loss for this particular family of Gaussian response losses and refitting their readouts. It does not rule out learning new factor directions with the response objective, using a better distribution of actual state changes, or finding reusable intermediate computations in a different graph.

The queued output-local quartic residual learner changes the factor directions and adds new products, which is a materially different hypothesis. Its native results remain pending. The separate removal-stage diagnostic concerns output coordinate 1; it cannot be replaced by a small-output aggregate score.

## Checks and code

Twenty independent paired quadrature and gradient comparisons cover five planted teacher/dictionary families and four correlation settings. The $\rho=1$ check uses explicit automatic derivatives. All relative discrepancies are below $6.2\times10^{-15}$, and constant invisibility is checked directly. Native readout normal-equation checks pass below the registered $10^{-8}$ tolerance; degree-summed Grams reproduce the existing value Gram. The full two-seed native screen took about 30 seconds on two CPU threads. These checks support treating the result as a scientific negative, not as a detected instrument failure.

[Plan](../../direct_tensor_match/PAIRED_GAUSSIAN_CP_PLAN_V1.md) · [All native results and per-output errors](../../direct_tensor_match/PAIRED_GAUSSIAN_CP_NATIVE_V1.json) · [Independent controls](../../direct_tensor_match/PAIRED_GAUSSIAN_CP_CONTROLS_V1.json) · [Exact degree-resolved kernels](../../direct_tensor_match/paired_gaussian_cp.py) · [Native executor](../../direct_tensor_match/audit_paired_gaussian_cp.py).
