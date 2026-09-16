# Subject-number L11H3 late-writer specificity audit V2

V1 is retained with terminal `invalid`: over 65 readouts, its float32 Möbius
subtractions reached maximum absolute closure error `1.52587890625e-4`, above
the frozen `1e-4` bar. Selection reproduction, component closure, replay,
relative Möbius closure, and the synthetic fixture otherwise passed.

V2 changes only the numerical dtype of corner values, Möbius dividends, target,
closure, and direct replay comparison to float64. Native model calculations and
port construction remain float32 followed by the previously registered closed
component gauge. The authority, panels, masks, 64 random axes and seed,
selection rule, thresholds, prices, and scientific predictions are unchanged.
V2 binds the failed V1 runner and result in addition to the inherited artifacts.

If V2 passes instrumentation, its three-term pruning and two empirical-rank
specificity tests are licensed. V1 remains invalid regardless of the outcome.
