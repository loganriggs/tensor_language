# A better fitting score is not yet a transferable circuit

21 September 2026, 05:34 UTC.

**Direct functional fitting fixes the training error but does not transfer to other documents.** Allowing the source directions to move gives almost perfect training reconstruction and worse evaluation error. Freezing those directions avoids the severe overfit, but leaves roughly 40% error. Neither candidate passes the registered fidelity screen.

This follows the [joint coefficient-fit failure](research_update_2026-09-21_0524_joint_quartic_metric_failure.md). The useful new distinction is between three questions: are we optimizing the right function, does the structure have enough capacity, and does the fitted computation generalize?

## The controlled comparison

We kept the same weak third native component:

$$
\phi(z,t,s)=
\frac{(t-\tfrac12q_a(z)-\alpha s)(q_b(z)-\beta s)}{s^2}.
$$

Here $q_a,q_b$ are quadratic source reads folded from the preceding MLP's weights; $z$ is its normalized input; $t$ is a scalar read of the last MLP's input; and $s$ is that input's RMS denominator. Native inputs and normalization remain explicit dependencies.

The previous experiment fitted polynomial coefficients but worsened this function. This time we compared:

1. The same kind of coefficient objective.
2. Error in the unnormalized numerator on actual training states.
3. Error in the normalized function on those states.

For each functional objective, we tested no coefficient regularization and a penalty of 0.01 times the relative squared coefficient error. Each family used learning rates 0.01 and 0.05, 400 Adam steps, and the same spectral initialization. Learning rate and saved iterate were selected by the training objective alone.

We used 24 documents for this fit and eight for evaluation, rebuilding source means, covariances, affine terms and initialization from the 24 training documents. **The eight evaluation documents were held out from this adaptation, but had participated in earlier native-mode discovery.** This is not fresh circuit confirmation. A later token-prefix audit also found that evaluation row 28 duplicates training row 19 exactly over the 64 model-input tokens. We preserve the originally registered eight-row results and additionally score the seven distinct evaluation rows, without refitting or reselecting candidates.

Each source retained 16 quadratic directions: 36,864 trainable factor entries across the two forms, with 1,536 training positions. Capacity was fixed across objectives. There were no new model forwards.

## What happened?

The table shows each family's training-selected fit. All errors below concern the same normalized scalar function, even when the fitting objective was different.

| Fit | Training variation error | Evaluation variation error |
|---|---:|---:|
| Original separate-source initialization | 32.80% | **39.98%** |
| Joint coefficient-only fit | 51.96% | 53.18% |
| Numerator functional fit | 0.16% | 48.87% |
| Numerator fit + coefficient penalty | 0.19% | 68.80% |
| Normalized functional fit | 0.10% | 45.44% |
| Normalized fit + coefficient penalty: registered primary | **0.09%** | **78.18%** |

The primary passes the training-improvement gate and fails the evaluation gate, which required error below 15% and improvements over both the initialization and coefficient-only fit. Export replay passes for all fits. The duplicate means the original split was not fully separated; the corrected diagnostic leaves the failure intact.

The functional objective can fit the training function. The failure is now transfer, not inability to lower that training loss. The observed gap is consistent with the very flexible source directions adapting to a small sample. It does not prove that no compact representation exists, nor that additional data or a better structural restriction could not recover one.

## Does freezing the candidate products help?

We then tested a smaller graph edit. Keep the 16 input directions for each source fixed, and fit only the weights of their 32 existing square products. The graph topology and product count stay unchanged.

Each source's weights have an exact linear least-squares update when the other source is fixed. We alternated those updates for 100 sweeps, with three starts and ridge penalties 0, 0.01 and 1. This has **32 trainable coefficients**, rather than 36,864 adjustable feature-direction entries.

| Coefficient-refit penalty | Training error | Opened evaluation error |
|---|---:|---:|
| 0 | 29.57% | 40.00% |
| 0.01: registered primary | 29.88% | 40.04% |
| 1 | 31.68% | 39.75% |

This avoids the severe overfit but does not repair the approximation. The primary again fails the 15% fidelity target. These eight evaluation documents were already opened by the previous comparison; this follow-up is a structural diagnostic.

Taken together, the controls bracket the current problem: free directions fit the sample without transferring, while merely reweighting the existing products is insufficient. Neither result establishes a general limitation on arithmetic circuits.

## How this relates to the paper's moment matrix

Functional fitting on probes can be written as an implicit moment-weighted tensor loss. Let $v_4(x)$ contain all ordered quartic products of the augmented input $x$, and let $\Delta$ be the difference between the two numerator coefficient tensors, flattened into a vector. Then

$$
\mathcal E_{\mathrm{normalized}}
=\Delta^\top M\Delta,
\qquad
M=\frac1N\sum_{n=1}^N
\frac{v_4(x_n)v_4(x_n)^\top}{s_n^4}.
$$

We evaluate the functions directly; we do not materialize this enormous matrix. Thus the functional comparison uses an empirical full moment metric, with the RMS weighting included.

Input covariance alone is not enough to specify this metric for arbitrary distributions. A simple counterexample has two zero-mean distributions with variance one: equally likely $-1,+1$, and values $0,0,-\sqrt2,+\sqrt2$ with equal probabilities. Their eighth moments are **1 and 8**, respectively, so the squared norm of the quartic function $x^4$ differs despite matching covariance.

An independent small-case test explicitly formed the empirical moment matrix and reproduced numerator and normalized losses to below $10^{-15}$ relative error.

## A new weights-first comparison: exact Gaussian moments

We also derived and verified the exact numerator loss under a Gaussian input model with the augmented constant coordinate fixed to one. Gaussian moments can be computed from the mean and covariance. This includes moment contributions that the earlier covariance-whitened coefficient Frobenius norm omitted.

The implementation decomposes the numerator into orthogonal Gaussian Hermite degrees. If $\Delta H_k$ is its degree-$k$ coefficient difference in that representation, then

$$
\mathbb E\big[(N-\widehat N)^2\big]
=\sum_{k=0}^{4} k!\,\|\Delta H_k\|_F^2.
$$

These degrees are mathematical components of an error metric, not newly identified semantic features. Five-node-per-axis Gaussian quadrature independently verified values and symmetric-parameter gradients on six small cases, with discrepancies below $6\times10^{-16}$. An initial gradient test incorrectly allowed nonsymmetric matrix perturbations; correcting the test to the stated symmetric parameter domain resolved that mismatch.

We evaluated existing programs under this metric; **we have not yet optimized it on the trained model**:

| Existing program | Exact Gaussian numerator variation error |
|---|---:|
| Original initialization | **9.03%** |
| Coefficient-only winner | 17.19% |
| Unregularized normalized-function winner | 11.76% |
| Regularized normalized-function winner | 20.18% |
| Fixed-product coefficient refit, penalty 0.01 | 11.03% |

This metric also detects that the fitted programs worsened the numerator relative to the initialization. About 91.4% of the initialization's Gaussian error energy lies in the cubic Hermite component.

These scores are not native errors: the Gaussian input model is an assumption. We evaluate the numerator only. Dividing by a Gaussian-sampled RMS coordinate can produce divergent moments near zero, so that is not silently treated as a valid normalized Gaussian objective.

The next useful comparison is to optimize this exact Gaussian numerator metric, then inspect the native function separately. It offers an analytic weights-first objective without finite-probe interpolation, while retaining explicit data dependence through the fitted input mean and covariance.

## Evidence and limits

- [Duplicate-prefix audit and seven-row rescoring](../../direct_tensor_match/FUNCTIONAL_QUARTIC_SPLIT_AUDIT_V1.json).\n- [Ten functional/coefficient fits](../../direct_tensor_match/FUNCTIONAL_QUARTIC_MODE3_FIT_V1.json).
- [Fixed-product graph refit](../../direct_tensor_match/FIXED_QUARTIC_PRODUCT_FIT_V1.json).
- [Empirical moment identity check](../../direct_tensor_match/EMPIRICAL_QUARTIC_MOMENT_CHECK_V1.json).
- [Gaussian moment implementation](../../direct_tensor_match/gaussian_quartic_moment.py), [independent quadrature checks](../../direct_tensor_match/GAUSSIAN_QUARTIC_MOMENT_CHECK_V1.json), and [existing-program audit](../../direct_tensor_match/GAUSSIAN_QUARTIC_MODE3_AUDIT_V1.json).

The GPU sweep took about 15 seconds; the fixed-product CPU fits took about half a second. Arithmetic used float64. Reported variation errors divide error norm by the centered target's norm. The third native mode remains a computational component, not a newly behaviorally identified circuit. No candidate was promoted, and the broader requirements for fresh/OOD prediction, selective intervention, reuse and complete extraction remain open.
