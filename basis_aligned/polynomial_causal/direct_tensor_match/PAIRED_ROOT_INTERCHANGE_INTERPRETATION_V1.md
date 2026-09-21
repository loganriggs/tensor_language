**Arithmetic sharing preserves the approximation, but the approximation fails conditional manipulation.**

The frozen32-document screen completed in2.26seconds. Numerical controls PASS (maximum replay4.73e-8, zero-edit exactlyzero). Both response predictions FAIL: no candidate meets10% response error; paired384 does not beat narrow26 by10% in every cell.

| Domain | Donor strength | Narrow26 error | Parent656 error | Paired384 error |
|---|---:|---:|---:|---:|
|FineWeb|.5|42.55%|42.63%|42.46%|
|FineWeb|1|28.41%|28.14%|28.05%|
|Code|.5|36.28%|39.33%|39.19%|
|Code|1|31.59%|36.06%|35.94%|

These compare changed-minus-own-baseline raw-logit responses after final normalization and softcap. Only the purequartic input is interpolated/swapped; recipient denominator and all other terms are fixed. This is not full upstream MLP16 interchange, and it is not evidence about any named semantic behavior. Inputs/pairings were fixed before execution; panels were already opened. Native response RMS .699–1.602 excludes a nearly-zero pooled denominator as the cause.

Follow-up CPU analysis fits the optimal scalar to each pooled candidate response using exact saved inner products. For paired384 the best possible scalar retains97.96–99.06%of squared error; all oracle response floors remain27.86–42.26%. Thus a global output-response scale cannot repair the failure. This is an optimistic post-hoc response-space calculation, not a model-level parameter refit or an untouched test.

The shared compiler has not damaged this parent materially. The native parent itself lacks response fidelity, and the cheaper baseline has comparable FineWeb/better code errors. Further graph compression of this parent cannot fix missing functional structure by exact rewrites. Prioritize changing the learned function/feature dictionary under finite response constraints, with native and cheap baselines retained, over expanding the root-rank sweep. The scalar audit does not prove that arbitrary output refitting cannot help.

[Plan](PAIRED_ROOT_INTERCHANGE_PLAN_V1.md) · [Native rows and checks](PAIRED_ROOT_INTERCHANGE_V1.json) · [Scalar audit](QUARTIC_RESPONSE_SCALE_AUDIT_V1.json) · [Audit implementation](audit_quartic_response_scale.py).
