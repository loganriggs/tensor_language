# Task 21 verbatim-copy authority: early design review

**Reviewed:** 2026-09-04 06:02 UTC.  **Scope:** CPU-only review of an unfrozen draft.  This is not a final
reproducibility approval and does not authorize a model, checkpoint, GPU, queue, outcome, or localization run.

## Verdict

**Repair before freeze.**  The linked counterfactuals are useful, but the draft needs two organizational and
statistical repairs before its content hashes become authoritative.

## What the task actually measures

The model sees a short sequence whose final token is repeated and is scored on whether it predicts that token again.
For example,

```text
Repeat exactly: storm soft dawn dawn dawn -> dawn
```

This is a **local last-token repetition** task.  It is not induction or remote retrieval: in every row, the answer is
also the immediately preceding prompt token.  A later localization result must therefore compare attention/MLP paths
against the direct residual-stream path from the final token embedding.  It may not describe a found component as an
attention-mediated copy circuit merely because intervening on that component changes the answer margin.

The four linked edits are nevertheless meaningful:

- `A1` replaces the complete trailing run and changes the answer.
- `A2` preserves an older occurrence of the original target but replaces the latest two-token run and changes the
  answer.  This creates a conflict between older and most-recent repeated identities.
- `P` changes an earlier filler while preserving the trailing run and answer.
- `C` extends the same-target run by one occurrence while preserving the answer.  It tests repetition strength without
  pretending that identical token occurrences have a uniquely identifiable source.

The draft CPU validator regenerated all 384 rows exactly, matched every saved split artifact, verified stable joint
prompt-plus-answer tokenization, and found prompt lengths 8 for FIT/SELECT/TEST and 13 for OOD.  Those are preliminary
checks on mutable files, not frozen evidence.

## Required repair 1: remove the task-number collision

The behavior-bank dossier already uses **task 18** for named-field table retrieval and directs that surface to be
merged with key/value retrieval.  Reusing the same ordinal for verbatim repetition would make later searches and
module dossiers ambiguous even though the behavior IDs differ.

The new strict unit should therefore use **task 21** consistently in filenames, schemas, experiment IDs, rung fields,
documentation, tests, and future result namespaces.  Its stable behavior ID remains `copy.verbatim_repeat`.

## Required repair 2: balance token roles

The draft used independent random samples for each group.  Direct census showed substantial role imbalance:

| Phase and role | Distinct tokens used | Smallest count | Largest count |
|---|---:|---:|---:|
| FIT target | 13 | 1 | 3 |
| FIT alternative | 11 | 1 | 4 |
| FIT control replacement | 14 | 1 | 4 |
| OOD control replacement | 12 | 1 | 6 |

Because token identity directly determines the answer, this can make easy or difficult tokens look like differences
between `A1`, `A2`, `P`, and `C`.  The earlier behavior-bank audit explicitly requires balanced answer, foil, token,
position, and surface frequencies for comparison/retrieval-style tasks.

A simple exact construction is available.  Deterministically retain 84 of the 86 verified single-token words, assign
21 disjoint words to each phase, and create 21 groups per phase.  Within each phase, use one SHA-derived permutation
and distinct cyclic offsets for every semantic role: each filler position, target, alternative, and control
replacement.  Every word then occurs exactly once in each role, while all tokens inside a group remain distinct.

This changes the exact FIT price to:

$$
8\text{ forward calls},\qquad 168\text{ row-side evaluations},\qquad
8\times 2\times 21\times4=1{,}344\text{ raw numeric bytes}.
$$

With 21 examples per cell, the proposed `>=0.85` cell gate means at least 18 correct examples.  The compiler and tests
must derive and assert these integers rather than retain the draft's 24-row constants.

## What a final review must prove

After repair and freeze, an independent review should verify:

1. all task-21 paths and identifiers are collision-free;
2. exact one-per-role token balance in every phase, including OOD's extra filler position;
3. exact regeneration of the four separately stored split authorities;
4. no later-split bytes in the FIT compiler closure;
5. complete, duplicate-free call/row/side coverage and the exact 8/168/1,344 price;
6. complementary capability pass/fail rules and fail-closed null projections; and
7. an explicit statement that a capability pass licenses only a new FIT localization preregistration.

Until those checks pass on committed hashes, there is no execution authority.
