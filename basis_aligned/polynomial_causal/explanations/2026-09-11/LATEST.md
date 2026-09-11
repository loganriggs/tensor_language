# Latest research update

**11 September, 10:22 UTC:** [Better sparse inference helps; the spherical shortcut fails on FineWeb](explanation_2026-09-11_1021.md).

Corrected encoding raises both learned overcomplete dictionaries to **53.60% / 53.61%** full folded coefficient capture, versus initial dictionaries **39.17% / 39.38%**. Joint convergence and stable identification remain unmet: complete-function cosine **0.698**, below **0.9**. This is a larger dictionary, not a matched-capacity comparison with earlier methods.

A small frozen FineWeb check finds that the exact radial correction worsens output probabilities. Cached-state analysis shows strong radial/traceless cancellation on natural inputs. No text-based discovery or new million-token sweep.

**Running:** coupled dictionary/code optimization of both saved starts, with unchanged L1 objective and corrected encoder. Native convergence is pending.

[10:22 hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-11_1022.md) · [Method index](../../WEIGHT_ONLY_METHODS_INDEX.md).
