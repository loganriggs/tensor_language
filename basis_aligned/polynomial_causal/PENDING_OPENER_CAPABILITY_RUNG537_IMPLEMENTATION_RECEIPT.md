# Rung 537 FIT/SELECT capability implementation receipt

**Frozen:** 2026-09-03 14:31 UTC

The managed entrypoint is
`basis_aligned/bilinear_quotient/ops/pending_opener_capability_rung537.py`, SHA-256
`40241562244cc4e0ee31842731eb4e64f6d1df3dc8210b69587421012fbbe833`.

It binds the preregistration, 288-row multi-family authority, 96-row matched-control
authority, and both receipts by exact hash. It selects only FIT and SELECT before making
model inputs. FINAL_TEST and OOD are never passed to the forward loop.

The evaluator runs 512 prompt sequences in exactly 32 model forwards (batch size 16),
with zero backwards. It measures both directions of both answer-changing families and
both sides of both invariance families. A family cannot pass from a positive mean alone:
the answer-changing gates require both endpoint preferences on at least 75% of pairs,
mean symmetric logit separation above 0.5, and a group-bootstrap 95% lower bound above
zero.

Pre-execution checks:

- Python compilation: pass.
- Static experiment gate: pass, five distinct registered predicates.
- GPU-free outcome-closed dry run: pass; 192 main + 64 control rows, 512 sequences,
  32 expected forwards, allowed roles exactly FIT/SELECT.
- Focused schema/data/contract/math suite: 28 passed.
- Synthetic evaluator tests: a two-sided causal endpoint passes; a one-sided shortcut
  fails; the 75% invariance boundary is exact.

The output is
`basis_aligned/bilinear_quotient/pending_opener_capability_rung537_results.json`.
Only a full capability pass authorizes implementation of the common-site ceiling. No DAS
training and no FINAL_TEST/OOD evaluation occur in this entrypoint.
