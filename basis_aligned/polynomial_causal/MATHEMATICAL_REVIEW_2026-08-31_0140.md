# Mathematical review — 2026-08-31 01:40 UTC

Convention (§2135): damage = CE added above the real model; lower is better. Grounding: the measured retrieval
skeleton (§2158–§2168): backup readers at 14/16 (super-additive +0.058), general suppressor m16, depth-adds-range
law, four defeated addressing theories.

## Top three mathematical moves (new, non-duplicative)

**1. Minimal realization of the backup pair (system identification / Kalman minimality).**
- Object: the value-path outputs of attn14 and attn16 on ind positions.
- Operational definition: if both modules compute approximately the same read map R(context), the pair is a
  NON-MINIMAL realization; a minimal one is ONE shared read + two linear write couplings. Distinguishing
  prediction, derived per §2165's sign rule: *shared computation* implies a document-INDEPENDENT linear map
  between the two outputs (fit on half the documents, transfers to the other half); *independent-but-similar
  functions* implies alignment whose map does NOT transfer.
- Assumptions that may fail: the read is shared only on ind positions; RMSNorm gauge may rotate per-block
  (Procrustes absorbs a fixed rotation, not a per-document one).
- Consequence beyond reconstruction: a compiled shared-read primitive (halves the read's description; predicts
  the §2165 backup quantitatively: zeroing one keeps R available to the other's coupling).
- Cheapest falsifier: one capture run, CCA/Procrustes on FR ind positions, transfer split. → PREREGISTERED as
  rung 77 (backlog): pred_a mean top-8 canonical correlation ≥ 0.6 on ind; pred_b the half-A Procrustes map
  keeps ≥ 0.8 of its R² on half B; pred_c alignment on non-ind positions lower by ≥ 0.2 (retrieval-specific).
  Null: aligned outputs, non-transferring map (independent duplicates).

**2. The suppressor as an opponent process (control-theoretic gain regulation).**
- Object: m16's output vs the summed attention outputs on ind positions, read through the unembedding.
- Definition: R_eff = R − g·B with B aligned to R in logit-effect space. Prediction: cos(logit-effect of m16
  output, logit-effect of attn14+attn16 output) ≤ −0.3 on ind positions and ~0 elsewhere.
- May fail if m16's suppression is routed through later blocks rather than direct logit opposition.
- Consequence: block-16 compiles to read-minus-brake — one affine correction, the "signed correction at m16"
  the skeleton demands, with a measurable gain parameter g.
- Cheapest falsifier: same capture run as move 1 (share the arms). Folded into rung 77 as pred_d? No — one rung,
  one object: registered as rung 78 (backlog), sharing rung 77's captures if built together.

**3. Three-objective Pareto formalization of the two ledgers** (damage, stored values, uncovered components):
  J(λ,μ) framing turns envelope-vs-coverage debates into declared exchange rates. CPU bookkeeping from existing
  receipts; lower priority; parked until the registry next syncs.

## Pruned
Hankel/automata class codes (died empirically, §2148/§2151); IB (closed arc); gauge quotients (settled, §2113);
output-span sharing across tail layers (§2122's coverage-artifact lesson — note move 1 shares the READ, input
side, and is licensed by an interaction measurement, not by span fitting).

## Executed this review
The derivations above (both interaction-sign predictions stated before any run, per §2165's rule) and the
preregistrations of rungs 77–78 into the backlog. The capture script build goes to the next driver wake — the
queue is at depth 2 (rungs 75/76 in flight) and the §2161 pause on a16 constructions remains respected: moves 1
and 2 are attributions/factorization tests, not new stand-ins.
