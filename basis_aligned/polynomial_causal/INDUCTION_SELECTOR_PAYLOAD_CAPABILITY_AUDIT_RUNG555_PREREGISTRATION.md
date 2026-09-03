# Rung 555 preregistration: independent receipt audit of R554

**Frozen:** 2026-09-03 16:43 UTC, before R554 opens model outputs

R555 is CPU-only and does not import the R554 implementation. After R554 lands it must:

1. bind the exact R554 result hash and verify the checkpoint and every frozen input hash;
2. require exactly 864 unique sequences, 27 model forwards, zero backwards, no weight update, FIT and SELECT only,
   and no FINAL_TEST/OOD opening;
3. require all eight split-by-factorial-cell summaries and independently apply accuracy $\geq0.75$ and bootstrap
   lower mean margin $>0$;
4. require the exact irrelevant-source, filler-change, and lag-extension split-by-variant-by-endpoint cells and apply
   the same two inequalities;
5. in both splits independently apply the selected-match positive fraction $\geq0.70$, bootstrap lower selected-match
   margin drop $>0$, and bootstrap lower selected-versus-irrelevant gap $>0$;
6. reconstruct each prediction flag and the terminal all-gates decision from those leaves.

Any missing cell, extra split, inconsistent count, hash mismatch, or flag mismatch makes the audit invalid. Because
R554 records group-bootstrap summaries but not every group margin, R555 verifies their coverage and recomputes the
decision from the saved summaries; it does not claim to recompute the bootstrap samples. Any later replication must
save raw group margins and independently recompute those bounds.
