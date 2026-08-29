# Foldable global-scale calibration v1

**Status:** prospective mathematical/CPU contract only.  This file does not authorize
row, checkpoint, model, GPU, result, or receipt access.  A source-closed execution
addendum and genuinely fresh row receipt are required before launch.

## Why this is now the highest-information cheap compiler test

The discovery-only deployed-magnitude sweep found that multiplying all 36 compiled
writes by one global scalar changes behavior materially.  Scale 0.5 improved
permutation-normalized model agreement, while scale 0.8 gave the best top-1 accuracy.
The sweep did not actually compute CE or KL: its JSON fields named `allpos_ce` are
`null`, and its third predicate compares enrichment order with top-1 order despite a
comment describing CE.  Therefore no scalar is selected and no faithfulness or
prediction claim is promoted from that sweep.

A global scalar is an unusually cheap program transformation.  It can be folded into
the existing table rows and output factors, adding no deployed floats or runtime
multiplies.  Its description needs one scalar value; this is one float of MDL metadata,
not literally zero information.  Because residual additions retain the unscaled token
stream and each block has RMSNorm, this is not a gauge invariance: changing the scalar
changes the function and must be tested causally.

## Frozen candidate grammar and separate purposes

Use exactly

\[
g\in\{0.35,0.50,0.65,0.80,1.00,1.25\}.
\]

Scale every attention and MLP compiled write by the same \(g\), with \(g=1\) as the
literal deployed baseline.  No per-site live-norm arm is eligible; it was already
catastrophic and it adds a different 36-parameter grammar.

On one fresh calibration role, score target CE, teacher
\(\mathrm{KL}(p_{\rm native}\Vert p_g)\), top-1 accuracy, and teacher top-1 agreement
on identical positions.  Select two scales before sealed rows are loaded:

1. \(g_{\rm CE}\), minimizing target CE;
2. \(g_{\rm KL}\), minimizing teacher KL.

This explicitly separates task prediction from functional imitation.  Ties within
\(10^{-12}\) prefer the scale closest to 1, then the smaller scale.  Top-1 and
agreement cannot select a scale.

## Fresh rows and falsification

A future row freezer must use the pinned local FineWeb parquet, exclude every
registry-recorded document, and create one calibration role plus at least two disjoint
sealed evaluation roles.  All candidate programs and the two selectors must be frozen
before any sealed role is deserialized.  Discovery roles used in the magnitude sweep,
E2, E3, compiler, suffix, or native-Down work are ineligible.

On every sealed role separately:

- predictive pass: \(g_{\rm CE}\) improves target CE by at least 0.005 nat and worsens
  teacher KL by at most 0.010 nat;
- faithful pass: \(g_{\rm KL}\) improves teacher KL by at least 0.005 nat and worsens
  target CE by at most 0.010 nat.

Failure on one role cannot be averaged away.  Report top-1 and agreement changes, but
they do not rescue failed CE/KL gates.  A pass is still discovery-only until replicated
on natural-text and a separately frozen OOD role.

## Required controls and claim boundary

The execution contract must bind the exact settled compiler, scale-sweep discovery
artifact, model/checkpoint, rows, source closure, physical call census, identical scored
positions, native reference, and \(g=1\) known-answer.  It must preserve model state,
publish result create-only and receipt last, and report actual folded storage/compute.

A passing scalar calibrates one existing compiled program.  It does not interpret a
coordinate, explain an attention circuit, repair recursive-state closure, or increase
the strict extraction/removal/OOD ledger by itself.

