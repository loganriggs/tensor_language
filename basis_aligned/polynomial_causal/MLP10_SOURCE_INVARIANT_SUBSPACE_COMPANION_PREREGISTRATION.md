# Preregistration — source-invariant response directions of the 509 tensor (CPU companion; parallel lane)

Date: 2026-09-02 22:12 UTC
Owner: Claude (parallel probe lane)
Status: frozen before rung 509 has produced any outcome; opens ONLY after
Codex's rung 509 receipt and § entry exist. Zero model forwards; pure linear
algebra on their published bundle. No 509 bar, fit, or route is touched; this
is the exact-algebra complement to their fitted dictionary (their atoms ask
"can 8 fitted atoms explain R"; this asks "what is provably source-invariant
in R"), in the establishe receipt-analysis pattern of my lane.

## Inputs and conditioning

- Rung 509 receipt with pred_a TRUE (any pred_b outcome — the analysis is
  meaningful under pass or null); its sufficient-statistics bundle whose
  sha256 matches the receipt's own `sufficient_statistics.sha256` field.
  Both shas are recorded in my receipt at analysis start.
- The finite singleton response tensor R[a,p,h,c]: a ∈ {N,P,Z7,Z8},
  p ∈ 253 exact terms, h ∈ two discovery halves, c ∈ 34 coordinates
  (4 task + 30 held-out circuit), exactly as 509 stores it. If 509's bundle
  stores responses per phase rather than per half, the two discovery halves
  are used and the receipt says so.

## Computation (all exact; no fitting, no iteration)

Per half h, with coordinates standardized by their N-source RMS:

1. Δ_a = R[a,·,h,·] − R[N,·,h,·] for a ∈ {P,Z7,Z8}; stack A_h = [Δ_P Δ_Z7
   Δ_Z8] (253 × 102).
2. Noise ceiling (DERIVED, not asserted): ε_h = RMS over terms and
   coordinates of (R[N,·,h0,·] − R[N,·,h1,·]) — the same-source cross-half
   fluctuation, i.e. what "zero variation" costs at this sample size. The
   ceiling is 2·ε_h (factor 2 = two independent halves in a difference;
   stated here, fixed).
3. Materiality floor (DERIVED): μ = .05 × ||R[N,·,h,·]||_rms per direction —
   5% of the native response scale, mirroring 509's own .05 projection floor.
4. Generalized problem: unit v maximizing ||vᵀR[N,·,h,·]|| subject to
   ||vᵀA_h||_rms ≤ 2·ε_h, solved by GSVD of the pair (R[N], A_h); report the
   full invariant-response spectrum and the count k_h of directions with
   response ≥ μ and variation ≤ 2·ε_h.
5. Controls: 16 term-label permutations of A_h (seeds 20260905+i) — the q95
   of permuted k values; and the macro direction (uniform weights over
   terms) reported separately as the known trivial invariant.
6. Cross-half identification: matched directions (max-cosine pairing)
   between h0 and h1 solutions must agree at cosine ≥ .70 to count as
   IDENTIFIED.

## Frozen predictions

- pred_a: shas recorded and internally consistent; tensor shapes exact
  (4×253×2×34); ε and μ computed by the formulas above and reported; the
  macro direction passes the invariance ceiling in both halves (sanity —
  calibration guarantees it; failure means my standardization is wrong, an
  instrument fault, not science).
- pred_b: k_h ≥ 2 in BOTH halves (at least one non-macro material invariant
  direction), with k exceeding the permutation q95 in both halves, and at
  least one non-macro direction IDENTIFIED across halves (cosine ≥ .70).
- pred_c: every identified non-macro direction's top-|assignment| terms
  (weight ≥ .5·max) form a set stable across halves (Jaccard ≥ .5).

Null (strong if pred_a holds and pred_b fails): the action's complete sum is
the ONLY material gauge-covariant combination of MLP10 terms — a closure
statement for intermediate-grain grouping at this site, complementing (not
contradicting) whatever 509's fitted route finds; no threshold change, no
second tolerance, route = report beside 509's verdict and stop. If 509's E
passes while this nulls, the tension is reported as-is (fitted atoms would
then be soft mixtures invisible to hard linear invariance — a stated,
non-paradoxical outcome).

## Price

Zero forwards; CPU < 1 minute; one receipt JSON. Nothing deployed.
