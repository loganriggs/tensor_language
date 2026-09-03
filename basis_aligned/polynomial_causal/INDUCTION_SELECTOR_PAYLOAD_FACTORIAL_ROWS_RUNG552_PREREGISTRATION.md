# Rung 552 preregistration: induction selector × payload factorial rows

**Frozen:** 2026-09-03 16:21 UTC, before constructing rows or loading any model

## Goal

Build a counterfactual dataset that separates two computations inside induction-style copying:

- **selector $S$:** which earlier repeated source token the current query matches;
- **payload assignment $P$:** which follower token is attached to each source.

The dataset is outcome-blind. It does not claim that the model performs the task, choose a component, fit a subspace,
or open any model result.

## The four factorial conditions

Each semantic group contains two earlier adjacent source→payload pairs,

$$
A\rightarrow B,\qquad C\rightarrow D,
$$

and a final query equal to either $A$ or $C$. Payload assignment zero keeps the map above; assignment one swaps the
followers:

$$
A\rightarrow D,\qquad C\rightarrow B.
$$

The four conditions and correct next-token answers are therefore:

| condition | queried source | payload assignment | answer |
|---|---|---|---|
| $S_0P_0$ | $A$ | original | $B$ |
| $S_1P_0$ | $C$ | original | $D$ |
| $S_0P_1$ | $A$ | swapped | $D$ |
| $S_1P_1$ | $C$ | swapped | $B$ |

This creates four answer-changing single-factor edges and two joint-factor diagonals whose answer stays fixed. The
diagonals are important: they can distinguish a compositional selector/payload representation from a direction that
only moves the final $B$-versus-$D$ logit.

## Additional counterfactuals

Every group also contains:

- a **match-breaking necessity edit**: replace the earlier occurrence of the selected source while keeping its
  follower, the final query, and the original answer fixed;
- an **irrelevant-source edit**: replace the unselected source while preserving the selected equality edge and
  answer;
- a **filler change** preserving both source→payload pairs, selector, payload assignment, and answer;
- a **lag extension** adding filler before the query while preserving both relations and the answer.

These are generated for a balanced choice of the four factorial conditions, so controls do not always use one query
or one payload assignment.

## Split and token authority

The group is the statistical unit. Planned groups are FIT 72, SELECT 36, FINAL_TEST 36, and OOD 36. Each group and
all its derived pairs belong to exactly one split. The four splits use disjoint banks of single-token alphabetic GPT-2
pieces and disjoint prefix templates. OOD additionally uses a code/trace-style prefix and longer lag patterns.

Exact prompt sequences must be unique across groups. Reuse of one of the four factorial conditions by multiple edges
inside the same group is intentional and recorded through a condition ID; it is not treated as an independent
example. No variable token ID may occur in more than one split. All texts must round-trip through the pinned GPT-2
tokenizer, source/query equality must be exact at token level, every payload must immediately follow its source, and
all correct answers must be single tokens.

## Required receipt

The receipt records group/row/cell counts, family and split counts, unique sequences, within-group reuse, cross-split
token disjointness, tokenizer identity, hashes, and `model_loaded=false`, `model_forwards=0`,
`model_backwards=0`, `outcomes_opened=[]`. Any failed construction check terminates before publishing rows.

The next model-facing step is a separately preregistered native-capability and complete-state site-ceiling screen on
FIT+SELECT only. FINAL_TEST/OOD remain unopened until selector, payload, and joint-factor rules are fixed.
