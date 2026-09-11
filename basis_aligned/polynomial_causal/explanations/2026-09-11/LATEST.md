# Latest research update

**11 September, 10:48 UTC:** [Direct tensor fitting is substantially more effective](explanation_2026-09-11_1021.md#update1048).

Corrected encoding raises both learned overcomplete dictionaries to **53.60% / 53.61%** full folded coefficient capture, versus initial dictionaries **39.17% / 39.38%**. Joint convergence and stable identification remain unmet: complete-function cosine **0.698**, below **0.9**. This is a larger dictionary, not a matched-capacity comparison with earlier methods.

A small frozen FineWeb check finds that the exact radial correction worsens output probabilities. Cached-state analysis shows strong radial/traceless cancellation on natural inputs. No text-based discovery or new million-token sweep.

The coupled L1 optimizer completed without convergence and added less than0.1percentage point. A direct full-tensor step instead reached **55.96% / 55.94%**, with about one second per gradient.

**Running:** sustained full-tensor fitting of shared features and sparse coefficients, with the output matrix solved exactly at each evaluation. First-start optimization is underway; no final convergence or behavioral result yet.

[10:22 hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-11_1022.md) · [Method index](../../WEIGHT_ONLY_METHODS_INDEX.md).
