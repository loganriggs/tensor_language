# Larger evaluation: the small-feature failures replicate

22 September 2026, 01:12 UTC. Completed evaluation of five frozen candidates on **16,384 token states from 256 new source documents**, each with 16 output coordinates. No candidate was fitted on this panel.

The CP candidates retain reasonably low pooled error, but the error is higher than on the smaller panel, and their small output features remain poorly reconstructed. Increasing the sample size was inexpensive on this instance. This is stronger evidence about prediction on additional FineWeb documents, not evidence of selective causal circuits or distribution shift.

The object is unchanged: the pure quartic contribution through MLP16 and MLP17, projected onto the same 16 fixed output directions. It excludes other residual, attention, bias and cross terms. The numbers below do not measure the full model's logits after final normalization or softcapping.

| Frozen candidate | Old 2,048-state error | New 16,384-state error | 95% document-bootstrap interval | Worst 10% of states: share of squared error |
| --- | ---: | ---: | ---: | ---: |
| Original 384-product graph | 8.13% | 9.61% | 9.19–10.08% | 44.6% |
| CP seed 1001 | 6.32% | 7.38% | 7.07–7.71% | 45.4% |
| CP seed 1002 | 6.59% | 7.58% | 7.26–7.93% | 45.8% |
| Shared support seed 1101 | 20.38% | 21.16% | 20.66–21.68% | 49.0% |
| Shared support seed 1102 | 19.84% | 21.20% | 20.66–21.78% | 50.5% |

The bootstrap resamples entire documents, preserving the dependence between 64 positions in each document. It uses 2,000 resamples with a fixed seed. These intervals quantify variation within this sampling setup; they do not cover all model, corpus, or selection uncertainty. The first two CP candidates both pass the registered aggregate-transfer criterion: pooled error at most 1.25 times the old value, with each of output features 0–2 below 10% error.

That criterion deliberately does not claim uniform component fidelity. The separate prediction that small-feature failures would replicate also held: all twelve features numbered 4–15 exceed 30% relative error in both candidates.

| Fixed output feature | CP seed 1001 error | CP seed 1002 error |
| --- | ---: | ---: |
| 0 | 6.11% | 6.20% |
| 1 | 6.01% | 6.38% |
| 2 | 8.38% | 8.67% |
| 3 | 21.60% | 22.34% |
| 4 | 54.60% | 58.48% |
| 5 | 49.17% | 50.28% |
| 6 | 47.16% | 49.11% |
| 7 | 41.60% | 44.74% |
| 8 | 56.98% | 62.83% |
| 9 | 59.82% | 59.67% |
| 10 | 67.03% | 69.15% |
| 11 | 54.88% | 48.51% |
| 12 | 66.82% | 64.85% |
| 13 | 58.44% | 57.95% |
| 14 | 59.68% | 62.59% |
| 15 | 57.37% | 60.53% |

Thus the larger sample confirms the main limitation: dominant outputs have roughly 6–9% error, while the twelve smaller outputs have roughly **42–69% error**. These are fixed output coordinates, not twelve identified semantic concepts or individual native MLP channels. The JSON contains the energy shares and document-bootstrap interval for every feature.

Error concentration replicates too. The worst 10% of states contribute about **45–46%** of CP squared error, while accounting for only **19–22%** of target energy. The 99th percentile of per-state relative error is about **60–62%**. Individual-state percentages can become large when the reference output is small; denominator-stable quantiles using a common target RMS are also recorded. The result therefore supports uneven error, without reducing it to a few isolated examples.

## Actual cost of testing more

- Selecting and deduplicating the 256 text prefixes: **19.2 seconds**.
- Managed GPU model loading, old-panel replay, new-state/reference capture and cache write, as timed inside the script: **1.94 seconds**. Runner process time was about 3 seconds. This instance benefits from local weights and a fast GPU; it is not a cold-download timing.
- Predicting all 16,384 states with one CP candidate on two CPU threads: **0.62 seconds**.
- Evaluating all five candidates and computing the numerical summaries and bootstrap intervals: **2.66 seconds**.

The earlier concern that new-state capture would be the expensive step was a qualitative expectation. The measured result here is that even this larger capture is cheap. Queue waiting is separate from computation. Model fitting and exact tensor optimization are much more expensive than these evaluation passes.

## What makes the panel new?

One 65-token prefix was selected from each of 256 distinct FineWeb sample-10BT source documents; the first 64 positions were evaluated. Selection excluded known 32-token excerpts from local FineWeb row caches and prior token panels, plus earlier recorded document IDs and text hashes. Matching was checked anywhere in a candidate document. No candidate errors were used to select examples.

This establishes the stated local exclusion checks, not absence from pretraining or every other experiment in the repository. Distinct documents can remain related. Once these results are inspected, this panel is opened evidence: later fitting choices cannot use it and still describe it as untouched validation.

Instrument checks passed. Captured old inputs replayed exactly; old reference values agreed to **2.33e-7 relative error**. Candidate artifact hashes matched the frozen registry. All 16,384 inputs and 16-coordinate labels were finite. Token identities and reference cache are retained for follow-ups.

## A separate compression result

The queued CP refit also completed. It reduces each fitted 512-term CP parent to 256 terms, allowing directions to move after pruning. Cost falls from 1,536 to 768 variable products. Under the data-informed Gaussian metric, parent reconstruction error falls to **0.90% and 0.77%**. Yet native same-token response errors remain about **14.95% and 14.98%**, compared with **9.72% and 11.78%** for the unpruned parents. The registered response-retention criterion fails.

This is a concrete warning for the two-stage approach: a small error when compressing an intermediate fitted model does not automatically preserve the native computation's responses. These refitted candidates were not among the five frozen candidates in the new-panel test above.

[Full fresh-panel measurements](../../direct_tensor_match/RESIDUAL_FRESH_SCORES_V1.json) · [Registered protocol](../../direct_tensor_match/RESIDUAL_FRESH_PLAN_V1.md) · [Capture checks and timings](../../direct_tensor_match/RESIDUAL_FRESH_CAPTURE_V1.json) · [CP parent-refit results](../../direct_tensor_match/CP_PARENT_REFIT_NATIVE_V1.json).
