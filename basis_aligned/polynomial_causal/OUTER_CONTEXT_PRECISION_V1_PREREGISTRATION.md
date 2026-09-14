# Fixed-budget outer-context precision audit

Continuation after OUTER_CONTEXT_ANTITHETIC_20260914_0844_RESULT.json.
Registered before execution: 32,768 fresh independent contexts in 512 batches
of 64; seeds 170215000 through 170215511, disjoint from the earlier controls.
Keep the same synthetic raw-state/corner distribution, five exact source sums,
frozen sparse candidate, full nine-entry error Gram and conditional normalization.
Use CPU FP64, two threads, at most 120 seconds of execution. No antithetic samples,
fitting, candidate selection, native-text or body-forward evaluations.

A: complete Gram/direct error replay <=1e-10 relative and all outputs finite.
B: estimated context-level relative standard error <=1% and independent-half
mean energy disagreement <=1%. Both must pass; record panel-based SE separately.
C: nested 8192→16384 and 16384→32768 relative mean-energy changes each <=1%.
Preserve every failure. Report all nine entry estimates and standard errors,
error/reference energy ratio, candidate hash, seeds and actual runtime. Approximate
uncertainty diagnostics are not finite-sample guarantees or native-fidelity bars.

No adaptive sample extension repairs this registration. If precision passes,
the next question is paired ranking precision for distinct matched-cost candidates,
not immediate adoption or fitting to text. If it fails, inspect tail concentration
and distinguish sampling variance from an implementation error before changing
the estimator or its fixed budget. A cheap numerical audit should precede any
new sampling scheme. The earlier 1024-context failure remains recorded.
