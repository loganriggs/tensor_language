# Stage-two update: removing products — 2026-09-21 01:36 UTC

This follows the [overall two-stage review](research_update_2026-09-21_0104_full_coverage_and_shared_baselines.md). We now have a graph edit that cuts the number of nonlinear products, in addition to the earlier sharing of linear projections.

The 512-product candidate passes the registered preservation checks on reused native-model panels. Compared with the original corrected 1,024-product graph, it uses half as many products and about 52% fewer weight coefficients. It remains an approximation with substantial intervention error, not an identified semantic circuit.

## What changed

Start from the shared-input graph. Select a subset of its existing products by measuring how much each can reduce the remaining joint tensor error. This accounts for overlap and cancellation between terms. After selection, refit the output coefficients under the same separable covariance-weighted tensor metric. Keep an exact compact output basis so the refit does not hide its cost in a large dense output matrix.

The selection is greedy, not globally optimal. A small test matches every selected step against exhaustive alternatives and verifies the output refit against a dense least-squares calculation. The weight target is the frozen approximate graph; error against the original native function is evaluated separately.

| Graph | Distinct products | Weight coefficients | FineWeb swap error | Code swap error |
|---|---:|---:|---:|---:|
| Original corrected graph | 1,024 | 2,671,616 | 30.76% | 26.59% |
| Share input projections | 1,024 | 1,426,432 | 31.28% | 27.17% |
| Also select 512 products | 512 | 1,291,264 | 31.66% | 27.57% |
| Also select 256 products | 256 | 1,015,808 | 34.18% | 29.75% |

These swap errors compare changes in native centered logits for the full selected folded contribution. All rows use the same reused panels. Means and upstream source computation are outside the listed weight counts. Product removal can alter output grouping and does not preserve an interpretation of individual original groups.

## What passed, and what did not

The 512-product graph has replacement CE increases of 0.01098 nats/token on FineWeb and 0.02758 on code, both below the registered 0.05 threshold. Full removal and swap errors remain within 5% relative degradation of the previous graph.

A successor audit also compares directly with the original corrected graph, before either simplification. All four intervention errors remain within 5% there too; the worst relative increase is 3.67%. This avoids letting several local tolerances accumulate into a large unreported regression.

The absolute 30% FineWeb swap threshold is still missed on this panel. The new 512-product program has no fresh-panel confirmation yet. The 256-product version saves more but has noticeably larger errors, so it is not the retained preservation candidate.

Joint selection only modestly beats individual-product energy selection in the coefficient metric when both receive the same output refit. We should not attribute all compression to a sophisticated selection rule; the common refit and redundancy in the dictionary matter.

## Relation to the original proposal

This is an implemented stage-two graph edit: remove computations, refit the remaining graph, and price the complete result. It complements the earlier edits that shared input projections and introduced shared output corrections. General graph search, alternative deep hierarchies, stable feature identity and semantic intervention tests remain open.

The scheduled [three-hour math and primary-literature review](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_0135.md) separates exact graph rewrites from lossy edits and checks the assumptions behind the Gram-based objective. The next review is scheduled for 04:35 UTC.
