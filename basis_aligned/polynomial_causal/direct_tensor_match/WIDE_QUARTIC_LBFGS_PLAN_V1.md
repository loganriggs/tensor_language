# Wide quartic line-search baseline

22 September 2026, 04:53 UTC. Earlier L-BFGS work concerned different pencil/overlap targets; this tests the exact same five axis-aligned1152-dimensional quartics and two initializations as WIDE_QUARTIC_OPTIMIZER_V1. No subspace oracle, new target or added atoms. Same exact Gaussian value objective, normalized forward factors and ridge1e-6 profiled readout.

Use L-BFGS lr1, strong-Wolfe line search, history50, at most250quasi-Newton iterations and a hard cap500objective/gradient evaluations. Gradient tolerance1e-12, change tolerance1e-14. Retain best fitting-objective checkpoint among valid evaluated points, including line-search trials. Also record best checkpoint within the first251evaluations, matching the baseline's251objective evaluations (250updates); its gradient-evaluation accounting differs by one and will be disclosed. No claim of equal wall time or equal optimizer state/memory.

Prediction: at least8/10 final fits reach5% population value error. Report early-budget and full-budget value/response errors, closure counts, iterations and wall time, regardless. A pass would motivate a native optimizer candidate, not establish native performance or alter the queued Adam/Muon experiment. Contiguous raw parameter buffers avoid the previously documented L-BFGS view-layout failure. No model export or semantic/OOD claim.
