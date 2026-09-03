# Rung 543 v2 correction: balance every ordered closer pair

**Frozen:** 2026-09-03 15:24 UTC, before v2 row generation and before any model outcome

The v1 row authority passed exact-prompt uniqueness but sampled delimiter pairs randomly. A pre-outcome audit found
that the 12 ordered type pairs were imbalanced: some SELECT cells had one row and one OOD cell had none. No model was
loaded and no result used v1.

Version 2 changes only delimiter-pair allocation. Each split is divided equally among all 12 ordered pairs from
parenthesis, square bracket, curly bracket, and quote. Therefore FIT has eight semantic groups per ordered pair and
SELECT, FINAL_TEST, and OOD each have four. Prefixes, word tuples, and templates remain deterministically sampled;
the invariant-family identity `(prefix, words, first delimiter, template)` must also be unique so families that do not
use the second delimiter cannot collide.

All v1 fail-closed checks remain binding. Additional checks require exact 8/4/4/4 counts for every ordered delimiter
pair and every counterfactual family. This is a pre-outcome instrument correction, not a scientific result.
