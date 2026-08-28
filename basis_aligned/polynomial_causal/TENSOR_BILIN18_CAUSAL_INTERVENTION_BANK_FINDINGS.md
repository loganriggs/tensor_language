# Shared-routing causal intervention bank: rank640 passes

Date: 2026-08-28

Status: prospective pass for rank640; prospective failure for rank512.

## Result

The bank contains 16 previously unopened interventions: eight in hash-authorized
natural FineWeb prefixes and eight synthetic affine-token stress tests. Both candidates
are complete standalone programs with exact MLPs and no checkpoint fallback.

| candidate | stored values | saving | mean recovery | recovery 95% LCB | mean cosine | cosine 95% LCB | joint passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| shared-QK-512 | 503,436,726 | 7.7793% | 0.89821 | 0.86343 | 0.94983 | 0.93326 | 8/16 |
| shared-QK-640 | 516,707,766 | 5.3481% | 0.94442 | 0.92726 | 0.97238 | 0.96367 | 14/16 |

Rank512 fails every distributional admission condition despite its earlier single-poke
pass. Rank640 passes the frozen lower-confidence-bound and 75% individual-pass gates.
Relative to rank512 it gains 0.04622 mean recovery and 0.02255 mean cosine, and improves
recovery on all 16 paired fixtures. Exact price, disjoint ownership, total support, zero
native modules/calls/tables, and checkpoint collection also pass.

## Natural contexts versus synthetic stress tests

The bank reveals a meaningful split rather than uniform noise:

| candidate | natural mean recovery / cosine | natural passes | synthetic mean recovery / cosine | synthetic passes |
|---|---:|---:|---:|---:|
| rank512 | 0.96510 / 0.98286 | 8/8 | 0.83132 / 0.91680 | 0/8 |
| rank640 | 0.97943 / 0.99019 | 8/8 | 0.90942 / 0.95458 | 6/8 |

Thus rank512 already transports interventions well on the natural heldout prefixes but
is brittle off the language manifold. Rank640 preserves that natural performance and
materially extends it under synthetic OOD changes. Its two remaining individual misses
are synthetic position-96/160 cases; the distributional certificate allows them because
both lower confidence bounds and 14/16 joint passes remain above the frozen gates.

## Mathematical consequence

The paired monotone improvement strongly identifies shared-routing capacity as the
immediate cause of the earlier instability. It does not prove that the ordinary
activation-covariance objective is globally optimal, but it prunes causally weighted
generalized SVD as an urgent repair: the cheaper next question is whether rank640 also
retains rank512's approximately 0.01-nat cross-task predictive harm.

Rank640 is therefore the first prospectively robust **causal candidate**, not yet a
fully admitted simpler whole model. Predictive validation at the same rank remains the
missing interface. If that passes, the 5.35%-smaller program is the first point jointly
certified for ownership, prediction, and distributional intervention transport.
