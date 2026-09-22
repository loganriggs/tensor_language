# Full calibration moments improve prediction, but not enough for component fidelity

22 September 2026, 03:31 UTC.

The Gaussian surrogate is one source of transfer error. Refitting the same frozen quartic feature banks with exact empirical calibration moments reduces pooled value error on the larger opened panel to **5.80% and 5.85%**, versus **7.33% and 7.52%** for Gaussian value-only refits. Small-output response errors remain **49.74% and 50.91%**. The registered two-seed response-improvement criterion fails; no program is promoted.

This is a diagnostic about the chosen fitting distribution. It uses calibration output labels and is algebraically ordinary regression in the fixed feature basis. It is not data-free weight discovery, a semantic circuit result, or proof of out-of-distribution equivalence.

## Matched objects and information difference

The target is the pure quartic MLP16→MLP17 contribution in the same 16 fixed output directions. Both methods use the same two frozen 512-feature CP dictionaries, raw feature scaling, 16-output linear readout, and ridge $10^{-6}$. No factor directions change. The program cost stays at 1,536 variable products and 2,385,920 floats.

For feature evaluations $\Phi_{nj}=\phi_j(x_n)$ and calibration targets $Y_{nv}=F_v(x_n)$, the new readout solves

$$
G_{\rm emp}=\frac{\Phi^\top\Phi}{N},\qquad
X_{\rm emp}=\frac{Y^\top\Phi}{N},\qquad
D=X_{\rm emp}(G_{\rm emp}+10^{-6}I)^{-1}.
$$

Here $N=6,144$ states from 96 calibration prefixes. Quartic feature products involve moments up to degree eight. The Gaussian alternative evaluates the corresponding expectations analytically under $\mathcal N(\mu,\Sigma)$. We verified that its $\mu,\Sigma$ match these calibration states: mean difference is exactly zero and covariance relative discrepancy is $2.73\times10^{-16}$. Thus this is a higher-moment/distribution comparison, not a change of first or second moments. The empirical method includes substantially more data information than a covariance matrix.

The paper's $M$ acts on lifted polynomial tensor coordinates; it is not just the raw input covariance. For a quartic lift $z(x)$, an empirical choice is $M_{\rm emp}=\mathbb E_{\rm cal}[z(x)z(x)^\top]$ (with the appropriate output identity). We never materialize that enormous matrix; $G_{\rm emp}$ is its contraction into the candidate basis. The Gaussian metric supplies a different higher-moment tensor via pairwise contractions. [Paper, sections 2.2–2.3](https://arxiv.org/html/2605.15183v1#S2.SS2).

A finite-sample empirical metric can have a large nullspace. Matching it exactly would not certify equality of polynomials away from the sampled inputs. Likewise, our squared-error fits preserve amplitude, whereas normalized similarity alone identifies proportional functions.

## Prediction and response results

| Panel | Seed | Gaussian value error | Empirical value error | Gaussian small-output response error | Empirical small-output response error |
| --- | ---: | ---: | ---: | ---: | ---: |
| original | 1001 | 6.27% | 4.97% | 63.20% | 54.88% |
| opened256 | 1001 | 7.33% | 5.80% | 57.57% | 49.74% |
| original | 1002 | 6.52% | 5.18% | 60.29% | 57.78% |
| opened256 | 1002 | 7.52% | 5.85% | 59.98% | 50.91% |

The original panel has 2,048 states and 30 directed matched pairs. The larger opened panel has 16,384 states from 256 documents and 2,494 fixed matched pairs. No evaluation target is used in fitting, but both panels have been inspected previously; this is not fresh confirmation. Small-output response error is the RMS of per-coordinate relative errors for coordinates 4–15, not a percentage of failed examples.

The original-panel response ratios are **0.8684 and 0.9584**, versus the required maximum **0.85 for both seeds**. Thus response improvement fails while pooled-value retention passes. The larger panel shows more consistent improvement, but that does not retroactively replace the registered criterion.

Calibration pooled errors are 2.64% and 2.72%. Even there, small-output value RMS errors are 31.43% and 32.10%; they rise to 45.60% and 46.17% on the larger evaluation panel. Both incomplete calibration reconstruction and a transfer gap remain. We have not proved a hard capacity limit: regularization, the fixed dictionaries, and distribution coverage all matter.

## What this changes in the research direction

The previous Gaussian response-loss failure cannot be explained solely by a supposedly insufficient feature span. Using more calibration information improves the readout without adding a single feature. Earlier finite-panel root1 oracle fits already warned against equating Gaussian optimality with absence of a better readout. [Earlier capacity distinction](COEFFICIENT_GUARDED_NATIVE_INTERPRETATION_V1.md).

At the same time, this data-informed repair remains far from component fidelity. It does not justify replacing the weights-first goal with activation regression. The pending learned residual experiment tests new computations; the native-removal geometry experiment tests downstream effects. Both remain necessary independent questions.

Five augmented-design least-squares controls pass. Token hashes verify the ordering of the concatenated 2,048+4,096 calibration states against all 6,144 targets, and the original evaluation ordering. Native normal equations pass the registered $10^{-8}$ bound. Full CPU screen took approximately 2.5 seconds on two threads. No coefficients were exported or installed in the model.

[Plan](MIXED_CP_CALIBRATION_MOMENTS_PLAN_V1.md) · [Results, all output errors and controls](MIXED_CP_CALIBRATION_MOMENTS_V1.json) · [Executor](audit_mixed_cp_calibration_moments.py).
