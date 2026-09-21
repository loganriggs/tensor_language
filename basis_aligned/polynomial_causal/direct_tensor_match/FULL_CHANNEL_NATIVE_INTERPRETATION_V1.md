**A full-layer compression baseline transfers on opened panels**

All registered endpoint and affine-repair checks pass. Native residual replay is4.70e-8relative;96captures plus fitting take6.39seconds. The covariance-selected3686channel approximation with exact mean/tangent affine correction gives fullMLP logit-effect relative error4.32/3.64%on the two FineWeb panels and2.17/2.08%on the code panels. Mean CEadded is+.00633/+.000828nats on FineWeb and-.00137/+.000976on code. Lower CEadded is better; a negative value in one opened panel does not establish improved language modeling.

The correction reduces error in everypanel/domain for bothinputgeometries. It costs1,328,256extra affine coefficients, leaving11.668%reducedfactorparameter saving relative to all4608native products. Withoutcorrection, the covariance candidate retains20.009%saving witheffecterrors5.20/4.59%FineWeb and3.75/3.68%code. These are parameter/product accounting results, not measured end-to-end speedups.

The effect denominator is the native lastMLP polynomial's logit effect relative to the background with that polynomial removed. We preserveDownbias and actual finalRMS/unembedding/softcap. This evaluates the fullMLP contribution rather than selectedscalarreads, but upstream modelstates remain supplied.

**Successor: cost–error comparison**

The CPU componentwise audit compares allfourpanel/domain effecterrors and CEchanges alongwithliteralcoefficients. The covariance raw program dominates the more expensive Euclidean-affine alternative on alltheseobserved measures. Bothcovariance programs remain useful comparison points: rawis cheaper, affine is more accurate. Preserve both for independent validation, without tuning against these panels. Dominance is descriptive and not uncertainty-adjusted.

These64FineWebdocuments and32codefiles were already opened by previous research. This is not a newfreshconfirmation. No semanticfeatureidentity, selective removal, arbitrarycomponentreuse or standalone token-to-output extraction has been established. The result supplies a stronger baseline for the broader decomposition/circuit search; it is not an adopted circuit or a completion claim.

[Native results](FULL_CHANNEL_NATIVE_V1.json) · [Calibration and affine identity](FULL_CHANNEL_FUNCTION_V1.json) · [Cost–error audit](FULL_CHANNEL_PARETO_AUDIT_V1.json) · [Registered test](FULL_CHANNEL_NATIVE_PLAN_V1.md).
