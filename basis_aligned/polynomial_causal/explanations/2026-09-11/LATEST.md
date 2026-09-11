# Latest research update

**11 September, 06:54 UTC:** [Internal scaling audit and its limits](explanation_2026-09-11_0608.md#stage-balance0654).

Exact internal rescaling reduces squared stage parameter norms 5.55x / 5.66x without changing the fitted function. But the actual objective's stationarity and maximum-gradient measures improve only 1.62x / 1.02x; both 2x targets missed. This is not a convergence fix or a measured speedup.

The unchanged [saved-state continuation](explanation_2026-09-11_0608.md#structured-continuation0648) is live, first start about 2.43% coefficient capture at06:53, still unconverged. Budget30additional minutes/start; checkpoint5minutes. Inspect current per-seed receipts and runner before any resubmission.

[Methods and assumptions](explanation_2026-09-11_0608.md#weights-first-methods) · [Method index](../../WEIGHT_ONLY_METHODS_INDEX.md).

Weights first. FineWeb validation and separately labelled Pile OOD remain later stages.
