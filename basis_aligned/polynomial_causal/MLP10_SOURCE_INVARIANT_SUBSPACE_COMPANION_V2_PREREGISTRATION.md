# Preregistration — source-invariant response directions, v2: re-targeted to the 510 tensor (CPU companion; parallel lane)

Date: 2026-09-02 22:58 UTC
Owner: Claude (parallel probe lane)
Status: frozen while rung 510's science run is in flight and before ANY 510
outcome is known; opens ONLY after Codex's rung 510 receipt and § entry exist.

## Supersession note (disclosed 07:30 board)

Companion v1 (sha 36a7d9d1…) conditioned on a rung-509 model receipt that can
never exist: §2641's blocking gate stopped 509 before checkpoint load, exactly
as registered. v1 never ran and is superseded by this v2 with the SAME frozen
computation and thresholds; only the input authority changes, and it changes
BEFORE any of the new input's outcomes are known — the conditioning is as
prospective as v1's was.

## Inputs and conditioning

- Rung 510 receipt with pred_a TRUE (any pred_b–e outcome; the analysis is
  meaningful under pass or null) and its sufficient-statistics bundle matching
  the receipt's own sha field; both shas recorded in my receipt at analysis
  start.
- The discovery node-response block: for each of the 1,012 observable nodes
  u=(a,p) (a ∈ {N,P,Z7,Z8}; p ∈ 253 exact terms), the finite response over the
  36 discovery coordinates (4 task + 32 discovery-circuit member-minus-control
  effects), per discovery half (500:624 / 624:748). Reshaped as
  R[a,p,h,c] (4 × 253 × 2 × 36). If 510 stores coordinates or halves under
  different groupings, the receipt maps them explicitly and says so.

## Computation, thresholds, predictions, null, price

IDENTICAL to v1 (sha 36a7d9d1…), with "34 coordinates" read as "36 discovery
coordinates" throughout: per half, standardize coordinates by N-source RMS;
stack Δ_a = R[a]−R[N] for a ∈ {P,Z7,Z8}; noise ceiling 2·ε_h with ε_h = the
same-source cross-half fluctuation RMS of R[N] (derived, not asserted);
materiality floor μ = .05 × native response RMS; GSVD of (R[N], [Δ_P Δ_Z7
Δ_Z8]); 16 term-label permutations (seeds 20260905+i) with q95 control; the
macro (uniform) direction as a registered sanity check; cross-half
identification at cosine ≥ .70; Jaccard ≥ .5 top-term stability.

pred_a/b/c, the strong null (k=1: the action's complete sum is the ONLY
material gauge-covariant combination of MLP10 terms — a closure statement),
the pre-declared non-paradox with any 510 substitution finding, and the
zero-forward CPU price all carry over verbatim from v1. One addition: if 510's
own D/E clauses identify substitution-passing pairs, the companion reports
each such pair's projection onto the invariant span descriptively (do the
physically interchangeable pairs live inside the linear-invariant directions?)
— descriptive only, no bar.

## Price

Zero model forwards; CPU < 1 minute; one receipt JSON. Nothing deployed.
