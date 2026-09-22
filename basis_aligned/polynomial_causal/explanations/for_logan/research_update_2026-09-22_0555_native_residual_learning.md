# New quartic features help modestly, but fail the component-recovery criterion

22 September 2026, 05:55 UTC. All four Adam/Muon fits are complete. The larger-panel evaluation is also complete. L-BFGS is a separate ongoing comparison.

**Learning new features improves the smaller outputs, but the improvement is too small to meet our registered criterion.** Adam reduces their larger-panel value and response errors by about 7%. Muon changes them by less than 0.1% in this setup. This is evidence of partial recovery, not successful circuit extraction.

## What was fitted?

We kept the existing CP512 approximation to the pure quartic MLP16→MLP17 path fixed. Each text state is a 1,152-dimensional normalized input to MLP16; the selected path is measured along 16 fixed output directions. These are output coordinates, not examples or established concepts. Other residual, attention, bias and cross terms are outside this path.

The correction adds eight quartic features to each of the twelve smaller outputs, numbered 4–15:

$$
\widehat F_g(x)=F^{\mathrm{parent}}_g(x)+\sum_{k=1}^{8}c_{gk}\prod_{s=1}^{4}(a_{gks}^{\top}x),\qquad g=4,\ldots,15.
$$

The first four outputs remain unchanged. The directions a are learned from scratch; coefficients c are solved analytically at each optimization step with ridge 1e-6. Fitting uses exact weight contractions under the calibration-derived Gaussian mean/covariance, with fixed output-balancing weights. No text labels are fitted. This is data-informed weight matching rather than a wholly data-free loss.

We tested Adam learning rate 0.1 and Muon 0.01, two identical initial seeds per optimizer, 250 updates each. Forward factor rows are normalized. These are tested configurations, not a universal optimizer ranking. Earlier toys showed substantial scale and duration sensitivity, especially for Muon.

## Original-panel criterion and larger-panel result

The preregistered component criterion required both starts of at least one optimizer to reduce small-output value RMS and response RMS by 15% or more. On the original 2,048-state panel, Adam's reductions were about 8% and 6%, so **the criterion fails**. Both integrity and objective-improvement criteria pass.

The follow-up uses 16,384 states from 256 documents and 2,494 fixed same-token/same-position pairs. This panel was already opened; it is not new OOD confirmation. A response here means the predicted output difference between a matched pair of states, not an intervention in the full transformer.

| Candidate | Small-output value error | Small-output response error | Pooled all-output value error |
| --- | ---: | ---: | ---: |
| Fixed parent |56.58%|58.09%|7.384%|
| Adam25001 |52.59%|54.08%|7.315%|
| Adam25002 |52.54%|54.13%|7.312%|
| Muon25001 |56.58%|58.08%|7.384%|
| Muon25002 |56.57%|58.07%|7.384%|

“Small-output error” is the RMS of twelve per-output relative errors. It gives each of these coordinates equal weight. “Pooled” sums squared errors and target values across all outputs before taking their ratio, so large components dominate. These metrics answer different questions.

The descriptive 15% transfer criterion also fails. All twelve Adam-corrected smaller outputs still have about 39–63% value error and 39–65% response error. Their median document-level small-output error is about 52%; the 90th percentile is about 67%. This is not a failure confined to a few outlier examples. The worst 10% of states account for about 50% of the smaller-output squared residual; the other 90% still account for the remaining half.

## What the result does and does not establish

Adding new computations helps more than these tested Muon fits, and the gain transfers modestly to the larger opened panel. It does not recover accurate individual components. A large reduction of the training objective is not a large relative reconstruction gain: the objective omits a constant teacher-energy term, and its percentage change is not percentage prediction-error reduction.

The instrument checks pass: normal-equation residuals are below 1e-12; exported float32 corrections replay within 2e-6 relative error; original evaluation scores replay within 6e-9 absolute error in the CPU consumer; outputs 0–3 remain bitwise unchanged. The four fits took 446.5 seconds, with measured peak allocated GPU memory about 1.96GB. That excludes queue waiting.

The added program costs 288 products and 442,464 coefficients, taking the parent from 1,536 to 1,824 products. It is a discovery experiment with added capacity, not a smaller final program. No semantic naming, native removal, composition or OOD adoption follows. In particular, keeping outputs 0–3 fixed means this correction cannot repair the earlier coordinate 1 removal failure.

The next comparison is already running: scale-aware L-BFGS on exactly the same correction class, starts and objective. Its first 251-evaluation and final-budget results will distinguish some optimizer limitations from the observed modest native gains. The independent normalization-stage audit has also completed and will be interpreted separately; it concerns a different candidate/path interface.

[Native terminal receipt](../../direct_tensor_match/LOCAL_QUARTIC_RESIDUAL_NATIVE_V1.json) · [All larger-panel measurements](../../direct_tensor_match/LOCAL_QUARTIC_FOLLOWUP_V1.json) · [Original protocol](../../direct_tensor_match/LOCAL_QUARTIC_RESIDUAL_NATIVE_PLAN_V1.md) · [L-BFGS comparison](../../direct_tensor_match/NATIVE_LOCAL_LBFGS_PLAN_V1.md).
