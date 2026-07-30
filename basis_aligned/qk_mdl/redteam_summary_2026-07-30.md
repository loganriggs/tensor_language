# Defensibility review of RESULTS_summary_2026-07-30.md (2026-07-30) — ALL 8 ACCEPTED
Reviewer cross-checked every quantitative claim against committed JSONs. Clean claims confirmed
(census 23/162, KEY_cap +0.046, L3H8/L2H5 0.94/0.58, steer, dial, L0 3/576, induction 98-111%,
+0.116/+0.077 nulls, joint +0.0039, §35 downgrades all correctly carried). Corrections applied:
- F1 HIGH: "+0.047 (SE .001)" -- SE was welded from the +0.034 CHAIN experiment; +0.047 bottleneck
  has no committed SE. FIXED: SE removed, "~0.003/layer" noted.
- F2 MED: 99.56% headroom belongs to the +0.034 chain; +0.047 bottleneck = 99.4%. FIXED (paired
  correctly to each experiment).
- F3 MED/HIGH: null range "18-100x head-span" fused three experiments; correct = +0.047 bottleneck
  head-span 20-30x + random-576-dim 100x; chain head-span 18x. FIXED.
- F4 MED: "~95-99% causally substitutable" imported a fidelity number into a causal sentence. FIXED:
  "~98-99.8% causal (composed-fold fidelity down to ~94% at aggressive one-hop truncation)".
- F5 MED: "compressed analytic interfaces" contradicts the no-compression limit. FIXED: "PCA-
  bottlenecked analytic interfaces".
- F6 MED: Pythia scale-transfer unsourced in this ledger AND placed under "Open:". FIXED: "out-of-
  ledger, not tested here (Pythia held)".
- F7 LOW: retraction count. FIXED: "lambda bug plus retracted knob-recovery + several framing".
- F8 LOW: "archetype no better" misstated §34. FIXED: "~5x more class-aligned but still fail (2/144)".
ROOT CAUSE of F1-F3: the +0.047 PCA/head-bottleneck and the +0.034 composed-chain are DISTINCT
whole-model tests; their SEs/fractions/nulls must never be merged. Standing check added.
