# Fresh interventions: good combined accuracy, but the stronger shared baseline wins

21 September 2026, 09:10 UTC. Follow-up to [direction fitting and component identity](research_update_2026-09-21_0903_direction_refit_and_component_identity.md).

**The399-product graph reproduces the combined effect reasonably well on fresh inputs, but fails our registered comparison against existing baselines.** Its third component also retains absolute failures. Evaluating the sum therefore does not rescue the candidate or justify dropping its constituent tests.

This moves the evidence beyond the previously examined scalar states: the frozen program was evaluated through the real final MLP, RMSNorm, unembedding and logit softcap on new FineWeb documents and code files. It still receives native intermediate inputs, so this is conditional extraction rather than upstream closure.

## What was frozen and compared

The candidate is the coefficient-loss-selected16_seed816graph:367shared mixed products plus32private squares. Its directions and coefficients were frozen before collecting the new panel. No fitting or target rotation used these outcomes.

| Program | Source products | Floating coefficients |
|---|---:|---:|
| New compact graph |399|896,198|
| Earlier partial-sharing graph |512|897,804|
| Separate pair baselines |768|897,804|

The compact graph saves113products versus the earlier shared graph, with slightly less coefficient storage. Dense projections, additions and native input production still cost work; these counts do not establish a runtime speedup.

We sampled32FineWeb documents and16new standard-library code files. Each document/file contributed a257-token prefix. Previous four panels' document IDs, text hashes and code files were excluded; known FineWeb cached32-token excerpts were checked anywhere in candidate documents. This establishes novelty relative to these extraction panels and known caches, not all project history or model pretraining. All selected code files came from the top-level standard library; the prepared package-directory fallback was not needed.

## The interventions and requirements

At each eligible recipient site we evaluated removal of each original component and removal of all three together. The latter tests their common-writer group. We also replaced the preceding MLP contribution with a same-token donor from another document/file and recomputed the later native state.

- **Natural:** effect of removing the selected contribution in the original state.
- **Hybrid:** removal effect after the donor substitution.
- **Change:** difference between hybrid and natural removal effects.

Errors compare centered vocabulary-logit effects with the original model's corresponding effects. We retain all sites, continuation sites, and spaced-word sites separately on both domains.

The original72comparisons keep their15%natural/hybrid and20%change absolute limits, plus a1.10-times-separate-baseline limit. An additional combined-group test requires its18comparisons to pass the absolute limits and the1.10ratio against **both** the separate and earlier shared baselines. Combined success would not waive individual failures.

## Combined effect: low absolute errors, retained relative regressions

These are all-site results; percentages measure relative error in the native intervention's logit effect.

| Domain / intervention | Compact399 | Earlier shared512 | Separate768 |
|---|---:|---:|---:|
| FineWeb natural |2.72%|2.47%|2.80%|
| FineWeb hybrid |2.65%|2.37%|2.66%|
| FineWeb change |4.73%|4.45%|4.86%|
| Code natural |2.31%|1.90%|2.04%|
| Code hybrid |2.39%|1.93%|2.08%|
| Code change |4.22%|3.88%|4.12%|

Every combined-group absolute limit passes; the largest combined error across all subgroups is6.60%. However, **11of18combined comparisons fail at least one relative-baseline requirement**. The additional group gate therefore fails.

Some misses are small: FineWeb natural all-site error is1.10054times the shared baseline, just beyond1.10. Others are clearer. On code continuation sites, natural-effect error is1.60times the shared baseline and hybrid-effect error is1.88times it.

A4000-resample paired file bootstrap gives a95%interval of **1.47–2.42**for the code-continuation hybrid ratio. Seven combined comparisons have lower interval endpoints above1.10against the shared baseline. These are descriptive pointwise intervals; they do not adjust for multiple comparisons. The donor mapping and trained candidate stay fixed, so the intervals do not include donor-selection or refitting uncertainty.

## Individual components remain part of the result

Across the original72comparisons, four absolute limits and25relative-baseline limits fail. All four absolute failures concern the third component on FineWeb:

| Third-component subgroup | Compact error | Separate baseline |
|---|---:|---:|
| Natural continuation |16.39%|12.45%|
| Natural spaced word |22.01%|17.06%|
| Hybrid continuation |15.04%|12.68%|
| Hybrid spaced word |16.91%|12.78%|

The25relative failures comprise12third-component,5first-component and8combined comparisons. The second component has no relative failure. The earlier partial graph and separate baseline retain the same private third branch, making the candidate's third-component regressions directly interpretable.

Thus the failure is not merely an arbitrary component-coordinate threshold: relative regressions also occur in the combined native output. The exact factor-rotation ambiguity from the previous report remains mathematically valid, but does not remove these errors.

## Checks and research consequence

The native job completed48captures and7,981valid cross-document/file donor pairings in7.57seconds. Final-state replay is exact at the checked precision; folded source-read replay has relative error1.69e-7, passing its1e-4bar. Shapes, same-token donor constraints, document exclusions and frozen hashes passed. An independent CPU audit reconstructed every reported point metric from document-level records and recomputed the combined gate.

The compact graph is an informative cost–accuracy point, not an adopted circuit. Global sharing, a targeted private branch and joint refitting preserve the sum fairly well, but the stronger partial-sharing baseline exposes losses hidden by comparisons only to separate programs. More optimization under the same coefficient metric is not yet justified by these results; the next structural or metric proposal should explain the domain/subgroup tradeoff and retain both constituent and combined interventions.

No new semantic units, standalone upstream computation, or complete circuit-discovery result is established.

## Evidence

- [Frozen candidate](../../direct_tensor_match/COMPACT_GROUP_FREEZE_V1.json), [registered panel and gates](../../direct_tensor_match/COMPACT_GROUP_FRESH_PLAN_V1.json), and [document provenance](../../direct_tensor_match/COMPACT_GROUP_FRESH_ROWS_V1.json).
- [Native results](../../direct_tensor_match/COMPACT_GROUP_FRESH_NATIVE_V1.json).
- [Independent point-metric and bootstrap audit](../../direct_tensor_match/COMPACT_GROUP_FRESH_AUDIT_V1.json).
