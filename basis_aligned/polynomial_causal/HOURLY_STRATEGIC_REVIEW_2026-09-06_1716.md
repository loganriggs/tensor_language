# Hourly circuit strategy review — 2026-09-06 17:16 UTC

## Decision

Keep the program on source-resolved full-component circuits and weight-guided discovery. The query-position ambiguity is now repaired. Static weights are sufficient to nominate attention-head writers, while MLP nomination requires the exact bilinear tensor evaluated on task-conditioned carrier states and still benefits from causal screening. Do not spend the next hour optimizing scalar-only DAS or adding more localization-only circuit counts.

## What changed since 16:17

The cross-task L9H1/H4 subspace question resolved as a null for a literal shared state. The task bases have principal cosines only `0.648/0.406/0.349`, no shared rank at the frozen `0.80` threshold, and asymmetric cross-projection (`has -> is` 0.0846, `is -> has` 0.323). The same L9H1/H4 machinery therefore carries task-typed coordinates rather than one common causal subspace.

The constrained-DAS regularization red team separated memorization from target specialization. Deterministic tangent noise barely changed the bad full-vocabulary solution (heldout `0.519` versus `0.529` unregularized), while full-vocabulary KL reduced the error to `0.0632`, close to difference-in-means `0.0592`. On A2, KL reduced `0.448` to `0.245`, close to difference-in-means `0.239`. KL therefore works by correcting the target; noise does not. At rank one, difference-in-means is already near the Pareto compromise, and the optimized scalar solution is a task-margin quotient rather than the full causal state.

The earlier upstream weight validation was corrected because it patched the final query rather than contextual source positions. A reusable full-sequence intervention library now captures and jointly patches exact pre-`c_proj` head slices and complete MLP outputs at paired per-row carrier banks. On sealed heldout/A2 rows, carrier-position attention effects validate the weight ranking strongly: Spearman `0.916` for has/had and `0.699` for is/was, with top/bottom enrichment `14.7x` and `9.4x`. This is prospective evidence that projector-to-weight contractions efficiently find attention writers.

MLPs behave differently. Carrier-position effects are `6.62x` and `9.08x` larger than their query-position effects, establishing that they are genuine source writers, but the static Down-only weight rank is weak/reversed. Including the complete Left/Right/Down quadratic tensor improves the development correlations only from `0.262/-0.595` to `0.310/-0.476`. Conditioning that exact tensor on live A1 carrier inputs repairs them to `0.595/0.310`. A completely disjoint matched-lexicon A1-to-A2 test then gives prospective correlations `0.500/0.383`, with MLP4 ranked first by the score for both tasks and first/second causally. This misses the frozen `0.40` is/was and `2x` enrichment bars, so activation conditioning is useful but not sufficient; path/suffix sensitivity remains missing.

## Circuit quality and composition

The source-position greedy programs validate the user's distributed-circuit proposal. Six-component programs shared MLP3/4/6/8, beat every singleton on every panel, and recovered `0.736/0.743` has/had and `0.654/0.805` is/was on heldout A1/A2. Extending the still-improving paths to ten components monotonically improved every sealed prefix:

| task | heldout A1 | A2 | direction fraction |
|---|---:|---:|---:|
| has/had | 0.755 | 0.767 | 1.00 |
| is/was | 0.694 | 0.824 | 1.00 |

The v2 terminal is a strict null because is/was heldout A1 missed the frozen `0.70` bar by `0.0056`, not because composition failed. Both ten-component paths hit the new depth ceiling while fit error was still decreasing. They share eight labels: L3H4, L5H1, and MLP2/3/4/6/7/8. A single common program did not improve is/was and one run was invalidated for a preregistered forward-count arithmetic error; shared machinery is real, but task-specific adapters remain necessary.

The portfolio count remains ample: 79 legacy canonical records and 64 unique recent behavior screens at the last full audit. Quality, not count, is the bottleneck. Temporal will/had remains the strongest executable response program at roughly 96% heldout/A2 writer-response recovery. Aspectual/tense now have better source-position and weight evidence, but remain below Tier 5 because the programs are not complete, removal/necessity is incomplete, and no compressed predictor yet replaces captured donor components.

## Trajectory and failure modes

- Attention discovery can now use invariant projector-to-weight contractions to reduce causal search cost substantially.
- MLP discovery should use static tensor incidence as a broad prior, activation-conditioned gate moments as a better ranker, and causal patching for confirmation. A downstream/path Jacobian is the likely next mathematical term.
- Greedy full-component composition is working and generalizing monotonically, but fixed depth ceilings should be replaced by preregistered marginal-improvement stopping rules.
- Shared subspaces must not be inferred from shared module names. The current evidence supports a shared MLP carrier core plus task-typed axes and adapters.
- DAS objectives must be named by scope: scalar quotient objectives for answer/foil manipulation; centered full-logit/KL objectives for state fidelity. Noise regularization alone does not repair a wrong target.

## Throughput audit

Since 16:17 this lane completed the KL/noise DAS red team, task-typed subspace resolution, exact source-position intervention infrastructure, prospective head/MLP writer validation, two depths of greedy full-component composition, a shared-program regularization test, complete static MLP tensor algebra, activation-conditioned tensor algebra, and a disjoint-lexicon prospective causal validation. All GPU work used the managed queue; completed units were committed and pushed.

`CIRCUIT_FOCUS: PASS` — the work increased source accuracy, compositional circuit quality, and discovery efficiency rather than merely accumulating sites.

`CEREMONY_BUDGET: PASS` — reusable position-aware intervention and tensor libraries amortize future experiments; the only invalid run was explicitly preserved and not promoted.

`NOVELTY_LESSON_GATE: PASS` — query-position evidence was corrected, future-timestamp clerical errors were audit-corrected without rewriting bound artifacts, static MLP tensors were not overclaimed, and fresh causal validation was required after development.

## Ranked next moves

1. Add task-conditioned downstream/path sensitivity to the MLP tensor score and validate it on another fresh A1-to-A2 authority; retain causal screening until it clears prospective bars.
2. Continue the strongest aspectual/tense greedy programs with a marginal-improvement stopping rule or expand the candidate pool beyond the weight extremes, then perform selective removal on the final source-resolved set.
3. Move the temporal 96% program from captured-response replay to a compact predictor and test removal, OOD manipulation, and literal compute/state savings.

Next hourly circuit review due around **2026-09-06 18:16 UTC**. Next mathematical review remains due around **2026-09-06 17:30 UTC**.
