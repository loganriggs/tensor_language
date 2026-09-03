# Rung 545 preregistration: fresh three-value pending-opener rows

**Frozen:** 2026-09-03 15:38 UTC, before row generation and before any model outcome

## Reason for this dataset

R544 tested parenthesis, square bracket, curly brace, and quote. The model failed every native cell in which curly
brace was the correct closer, while all cells among parenthesis, square bracket, and quote were correct. Because that
three-value subset was selected after observing R544, neither R543's unopened FINAL_TEST/OOD rows nor its templates
may be used to confirm it.

R545 is a new row authority for the explicitly post-R544 hypothesis that the model maintains a pending-opener state
over `(`, `[`, and `"`. It uses new prompt templates, prefixes, word pools, content hashes, and split seeds. It loads
no model and measures no outcome.

## Fixed construction

There are six ordered pairs among the three opener types. Each semantic group contains five paired constructions:

1. `direct_three_value_type_substitution`: replace exactly one opener token and change the correct closer;
2. `completed_then_reopened_three_value_order`: complete one delimiter and leave the other pending, then reverse their
   roles;
3. `pending_type_preserved_surface_rewrite`: rewrite and reorder the ordinary words while preserving the pending type;
4. `pending_type_preserved_distance_extension`: insert a neutral clause after the same opener;
5. `pending_type_preserved_nonopener_punctuation`: replace one comma with one colon before the opener.

The first two are answer-changing interchanges. The last three preserve the proposed variable and answer. Every
group contains all five families and belongs to exactly one split.

## Splits and independence

- FIT: 72 groups, exactly 12 per ordered pair;
- SELECT: 36 groups, exactly 6 per ordered pair;
- FINAL_TEST: 36 groups, exactly 6 per ordered pair;
- OOD: 36 groups, exactly 6 per ordered pair.

Prefixes and content-word pools are disjoint across splits and from R543. Every prompt pair and every individual
token sequence must be globally unique. The content-addressed group identifier binds prefix, five words, delimiter
pair, and template. FINAL_TEST and OOD are not for selecting a site, rank, seed, objective, or threshold.

## Fail-closed checks

Generation fails unless all expected counts, round-trip tokenization, single-token direct edits, pair balance,
global uniqueness, group completeness, and split isolation hold. The receipt hashes the row file and this
preregistration. Any future capability or intervention experiment must bind both hashes before loading the model.

