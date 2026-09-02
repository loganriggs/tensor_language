# Cross-process noise-floor schedule (PROVISIONAL — 3-point fit)

Source: MATHEMATICAL_REVIEW_2026-09-02_1010.md Lyapunov fit on probe run1
(slope 0.1970/layer = 1.218x/layer amplification, residuals <=4e-4 nat).
STATUS: extrapolation from THREE points at depths 5/8/9; treat as
provisional until probe run2 and the b-variant diagnostics test the frozen
predictions (ordering m8>m9>m12, slope 0.20+-0.06). Applies ONLY to
CROSS-PROCESS comparisons of single-token damage quantities on hook
pathways; within-process comparisons are exact (measured <=2.1e-6) and
aggregate statistics are far tighter (canary: bit-stable).

| depth-to-readout (layers) | predicted cross-process noise floor (nat) |
|---|---|
| 0 | 0.0143  <- matches the old 0.015 CUDA-wobble tolerance |
| 1 | 0.0174 |
| 2 | 0.0212 |
| 3 | 0.0258 |
| 4 | 0.0315 |
| 5 | 0.0383  <- fit point |
| 6 | 0.0467 |
| 7 | 0.0568 |
| 8 | 0.0692  <- fit point |
| 9 | 0.0842  <- fit point |
| 10 | 0.1026 |
| 11 | 0.1249 |
| 12 | 0.1521 |
| 13 | 0.1853 |
| 14 | 0.2256 |
| 15 | 0.2747 |
| 16 | 0.3345 |
| 17 | 0.4074 |
| 18 | 0.4961  <- MLP0: cross-process comparison unsound; use in-run baselines |

Bar-setting rule of thumb (advisory): a cross-process bar on a single-token
damage at depth d needs tolerance >= 3x the schedule value, or the
comparison should be redesigned in-run (preferred, see rung 481's design).
