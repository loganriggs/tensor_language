# Consumer-weighted value component, before native outcome

2026-09-12. The Euclidean component failed behavioral B/C and exact compensated reader rescaling changes it. This candidate changes the metric, not the old verdict.

Use the frozen regional numerator F(q,f)=sum_a beta_a(q) f0 f1 f(2+a). Define H=E[(dF/df)^T(dF/df)], with query/current independent N(0,I), complete-vocabulary token initialization sampled uniformly, and equal average over32 relative positions. Compute H exactly via Gaussian fourth moments and vocabulary sums; no native activation or task-label fit. QK/RMS gates are excluded from this polynomial search metric and retained in native execution. Correlations, non-isotropic contextual inputs, nonlinear finite changes, and final suffix sensitivity are not modeled by H.

Let M be the same frozen native head13.0 four-output value matrix. Solve min_rank1 ||H^(1/2)(M-Mhat)||_F globally with SVD. Mhat=Mvv^T, where v is the leading right singular direction. Apply its equivalent four-output projector P=Mhat pinv(M) to cached head13.0 contributions; P is generally oblique, not orthogonal. No component-size sweep or nonlinear optimization.

CPU control: compensated diagonal gauge covariance <=1e-10 and back-transformed selected-map agreement <=1e-9; independent Monte Carlo fourth moments <=5% relative and query-consumer Gram <=10%. Sampling controls test an analytic metric, not a language-data fit. The model-free distribution is an explicit inductive bias, not evidence that real contexts follow it.

After control passes, repeat EXACTLY the same native A/B/C bars and four cells from STRUCTURED_PRODUCER_NATIVE_V1_PREREGISTRATION.md, now for the consumer-weighted component. A includes the metric control. Keep Euclidean results as fixed comparison; do not rescue missed10%fidelity/10%transfer bars with improvement over old method. Also report signed and unrelated transfers and component/remainder interaction. Cost remains full native QK and states plus the downstream component/suffix.

Opposing predictions: downstream-aware invariant weights reveal behaviorally relevant source reading versus metric invariance alone failing to supply real-context selectivity or sufficient effect. A negative result distinguishes this weight-only local derivative approximation from actual complete path decomposition; it cannot rule out sparse composed structure in other representations.
