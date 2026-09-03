# Rung 561 preregistration: independent CPU audit of R560 source factors

**Frozen:** 2026-09-03 17:48 UTC, before the R560 result existed

R561 makes no model calls and imports no R560 scoring function. From R560's saved row-level statistics it independently
recomputes:

- all target means, medians, positive fractions, deterministic group-bootstrap lower bounds, and decisions;
- all answer-preserving absolute closer changes, complete-head ratios, full-vocabulary RMS ratios, and decisions;
- all adjacent-wrong-source mean absolute recoveries and decisions;
- the deterministic FIT candidate ordering and the single SELECT verdict;
- every score×payload interaction value and summary;
- checkpoint/input hashes, native replay and factor reconstruction bounds, model-call counts, and opened splits.

Any disagreement marks the R560 result invalid. A matching audit does not by itself turn a scientific null into a held
result; it only establishes that the frozen computation and decision were applied consistently.
