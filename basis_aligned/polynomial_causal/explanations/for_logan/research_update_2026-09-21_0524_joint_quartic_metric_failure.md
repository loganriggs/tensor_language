# Joint quartic fitting improves the tensor score but worsens the function

**Data-provenance correction, 06:12 UTC:** the earlier FineWeb cache stores token chunks and can take multiple chunks from one document. Its row indices do not establish document identities. Earlier “document” counts and cross-document donor claims based only on those indices should be read as chunk-level evidence; document-level independence and uncertainty require a document-identified replication. The numerical results below are unchanged.


21 September 2026, 05:24 UTC.

**Directly fitting the composed quartic numerator worked as an optimization, but failed the functional accuracy screen.** For the weak third native component, covariance-weighted coefficient error fell from 80.86% to 13.97%, while normalized calibration-function error rose from 37.71% to 48.97%. We should not promote this replacement to a circuit on that evidence.

This is the trained-weight follow-up to the [verified quartic objective and five toy structures](research_update_2026-09-21_0504_reuse_limits_and_joint_fit.md). It tests the joint-fitting part of the two-stage research direction, not a general arithmetic-DAG search.

## What was optimized?

The target scalar component is

$$
\phi(z,t,s)=\frac{(t-\tfrac12q_a(z)-\alpha s)(q_b(z)-\beta s)}{s^2}.
$$

Here $z$ is the native normalized MLP16 input, $q_a,q_b$ are two quadratic reads folded directly from its trained weights, $t$ is a linear read of the last MLP's input, and $s$ is that input's RMS denominator. The constants $\alpha,\beta$ center the downstream feature.

With an added constant coordinate, the numerator is a product of quadratic forms in 1,155 augmented coordinates. We jointly fitted the two source forms with 16 directions each, retaining their exact centered constant and linear branches. This has 36,864 trainable factor entries. The fitted program still requires native inputs and explicit RMS division.

Eight Adam fits crossed two coefficient metrics, learning rates 0.01/0.05, and spectral/random initialization. Each received 400 steps with a cosine learning-rate taper. The best coefficient-loss iterate was retained; the winner for each metric was selected without consulting its native-function diagnostic. Source signs and centered affine terms were fixed. Both metric variants used calibration centering and covariance-conditioned parameters, so the isotropic arm is not a completely data-free control.

## Results and registered predictions

| Candidate | Coefficient norm error in its own metric | Normalized calibration variation error |
|---|---:|---:|
| Initial separate-source approximation, isotropic score | 112.92% | 37.71% |
| Joint isotropic winner | **42.11%** | **53.56%** |
| Initial separate-source approximation, covariance score | 80.86% | 37.71% |
| Joint covariance winner | **13.97%** | **48.97%** |

The two coefficient columns use different geometries and are not interchangeable. A relative error above 100% is possible; these scores are norms, not explained-variance percentages.

Both winners used learning rate 0.05 and the spectral start. Random starts were worse at the fixed budget; the isotropic random initial loss was particularly large. This establishes sensitivity to initialization and conditioning, not a universal optimizer or convergence claim.

- Instrument/replay prediction: **PASS**.
- Covariance coefficient error at most 80% of its initial value: **PASS**.
- Selected covariance candidate's native calibration error at most 15% and at most 80% of its initial value: **FAIL**.

No new native intervention or held-out confirmation was run for this failed screen.

## Is this a bug, RMS weighting, or a different objective?

An independent CPU audit reconstructed dense student matrices from the saved executable programs and recomputed the coefficient loss using dense matrix contractions. It agreed with the low-rank training kernel to below $10^{-8}$ absolute error. Saved source-program execution also reproduced the reported native calibration error to below $10^{-15}$. Earlier explicit small-tensor checks verified the symmetrization formula and gradients.

The mismatch already exists before dividing by RMS:

| Candidate | Unnormalized numerator variation error | Normalized function variation error |
|---|---:|---:|
| Initial approximation | **28.27%** | **37.71%** |
| Joint covariance winner | **46.24%** | **48.97%** |

Removing the mean error barely changes these scores: a constant offset accounts for less than 0.6% of error energy in the covariance winner. Therefore neither a simple output offset nor RMS division alone explains the failure.

The evidence instead supports a mismatch between the coefficient objective and the function evaluated on these native states. The coefficient objective measures the symmetrized polynomial on augmented coordinates. Native coordinates obey constraints: the added coordinate equals one, and $t,s,z$ are related through the model. Covariance weighting captures second moments, but quartic output squared error generally depends on moments up to order eight. The present result does not isolate which of these differences dominates.

## Consequence for the research direction

Keep the exact coefficient objective as a verified weight-space baseline. Do not treat its improvement as a certificate of useful functional simplification. The next discriminating comparison is a moment-aware functional objective on the same target, with document-separated evaluation and the same source capacity, compared against this coefficient-fit failure and the original separate-source initialization. That comparison must label its data dependence explicitly.

The full circuit goal remains open: this third component has not been behaviorally identified, broadly reusable inputs are unestablished, and the native input producers remain outside the extracted program.

## Reproduction

- [Eight-fit results and histories](../../direct_tensor_match/JOINT_QUARTIC_MODE3_FIT_V1.json).
- [Independent export and metric audit](../../direct_tensor_match/JOINT_QUARTIC_MODE3_AUDIT_V1.json).
- [Low-rank coefficient kernel](../../direct_tensor_match/quartic_lowrank_metric.py) and [explicit small-case checks](../../direct_tensor_match/QUARTIC_LOWRANK_METRIC_CHECK_V1.json).
- [CPU diagnostic](../../direct_tensor_match/audit_joint_quartic_mode3.py).

The managed GPU job completed in about 14 seconds, with zero native-model forwards. Native diagnostics used the existing 32 calibration documents, each with 64 tokens; they are in-sample evidence. Training used float64 with TF32 disabled. Coefficient scores are relative norms; variation scores divide error norm by the centered target's norm. The independent audit was CPU-only. No threshold was changed after seeing results.
