# Plain-English wrap of the full night — 2026-09-01 04:27Z (audit-side view)

(Damage = extra prediction error above the original model; LOWER IS BETTER. A "certificate" = one of 62
behaviors demonstrably kept close to the original.)

## What we actually built
One adopted artifact: a 539,595,062-scalar program (1.16% smaller than the original 545.9M) that predicts
within +0.0047 error, keeps 54 of 62 behavior certificates, transports named interventions with ~0.99
fidelity, holds on shifted text, and has an exact parts bill. It earned every gate we know how to ask.

## What we learned trying to make it smaller
Six routes screened in one night (two agents, every claim preregistered, every receipt cross-audited):
- Weight-level structure recovery: dead — trained networks hide their functional structure from their weights.
- Cross-layer shared parts: dead — no atom reuse beyond a coordinate-null.
- Vocabulary factorization: real shared structure, but savings-priced versions damage rare words badly.
- Clever selection (causal bases, Fisher scores, calibration, pair-interactions): every variant lost to
  boring uniform/spaced choices somewhere; nothing targeted survived both corpora.
- Finite-state abstraction: behavioral state exists (one head carries it) but no small machine.
- Error contracts: rank candidates well, bound nothing (3.4x-wide intervals).

## The two laws that explain the night
1. COMPOSITION TAX: parts that are cheap alone cost 1.2-1.8x extra together.
2. ONE-DIMENSIONAL DAMAGE: every compression hurts the 62 behaviors in nearly the same proportions, so
   certificates fall off a cliff together (54 -> 8-32 across the whole tested range), and no re-weighting
   or clever basis escapes it — we tried the escape directly (certificate-gradient hybrids) and the data
   said no: the constraints live inside the same subspace, just with tighter tolerances.

## The honest bottom line
This model's cheaply compressible structure was the attention lookup maps — that's the adopted artifact.
Everything else (MLPs, vocabulary) charges real behavioral prices at measured exchange rates. Further
compression needs genuinely new mathematics, and tonight's map shows precisely which doors are closed.
