**Finite removal fidelity improves substantially, especially after newlines, but the full registered criterion still fails.**

Both frozen mixed-objective CP candidates were tested against the original native root1 projected quartic component, using the same writer, actual MLP17 denominator, original background and final RMSNorm/softcap. Runtime1.83seconds. Instrument PASS; absolute all-cell fidelity FAIL; all-cell20%improvement criterion FAIL. No fitting or seedselection during this test.

Relative error in the full vocabulary logit change caused by removing the component:

|Domain / removal strength|Condition|Old384-product error|CP seed1001|CP seed1002|
|---|---|---:|---:|---:|
|FineWeb / .25|newline|26.08%|7.03%|5.69%|
|FineWeb / 1|newline|31.07%|7.59%|6.09%|
|FineWeb / .25|other|24.43%|11.99%|13.24%|
|FineWeb / 1|other|29.46%|13.00%|13.76%|
|Code / .25|newline|25.94%|5.43%|5.49%|
|Code / 1|newline|26.16%|5.30%|5.35%|
|Code / .25|other|11.69%|8.76%|10.84%|
|Code / 1|other|12.08%|8.08%|9.46%|

All16candidatecells improve over the oldgraph;11/16are below10%. Allnewlinecells pass10%, but the declared absolute criterion included allconditions and bothstarts. The20%relative-improvement gate also fails because seed1002's quarter-strength code/other improvement is only7.26%. Keep these failures; do not narrow the success condition after observing the results.

**Executed prefix redteam:** recomputed every aggregate from document-level energies, then removed each prefix in turn. Every newline leave-one-prefix error stays below10% (largest7.99%). Per-prefix performance is less uniform: seed1001 has15/16newlineprefixes below10%in eachcell; seed1002has16/16exceptFineWebfullstrength15/16. This is descriptive sensitivity on the same16prefixes/domain, not an independent confidence interval or fresh confirmation. FineWeb/other remains the main broad fidelity gap.

This is progress on executing an operational component's removal effect. It is **not a capitalization-circuit identification**: the original native projection's selectivity already failed (case effects also occur elsewhere and differ across domains), and its native reference/energy replays unchanged. The writer was fixed from earlier data-informed work. Neither the reference projection nor the learned scalar is guaranteed to be a single semantic concept. The global quartic coefficient error remains about98%, restart atom identities are unstable, and extracted/full-model composition and untouchedOOD are untested.

Each new16-output candidate costs1536products/2385920coefficients versus384/322048for the oldergraph. These counts describe the full exported program, not a claimed minimal root1-only circuit. Better finite-response fidelity was purchased with higher deployment cost. A smaller faithful shared graph remains the intended outcome.

The enqueuer initially rejected a static `__file__` lookup inside main; the runner was never started on that attempt. Using the alreadydefined root path passed the samepreflight. No numerical criterion or dataset changed. Generic control-window/absolute-replay warnings were inspected: response/error and reference-energy comparisons are relative, and the sole exactzero check is the literal zeroedit.

Next structural work: the exact constant-size support-exchange code now passes exhaustive planted controls. It can replace selected products in the shared hierarchy and refit all output coefficients, making sparsity a learned connection choice rather than a seeded mask. Native execution is pending. Preserve this positive mixed-objective CP result as a fidelity baseline; judge any cheaper graph under the same function and finite-removal measurements rather than coefficient capture alone.

[Native result](MIXED_CP_REMOVAL_V1.json) · [Preregistered plan](MIXED_CP_REMOVAL_PLAN_V1.md) · [Prefix audit](MIXED_CP_REMOVAL_PREFIX_AUDIT_V1.json) · [Feature fitting](MIXED_CP_FEATURES_INTERPRETATION_V1.md).
