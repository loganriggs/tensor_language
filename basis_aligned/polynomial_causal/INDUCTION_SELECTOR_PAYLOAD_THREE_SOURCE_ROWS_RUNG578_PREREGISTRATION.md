# Rung 578 preregistration: induction selector × payload rows with an endpoint-neutral control pair

**Frozen:** 2026-09-03 18:40 UTC, before constructing these rows and before opening any new model output

## Decision this resolves

R554 did not refute the model's selector-by-payload behavior. The model passed all eight FIT/SELECT factorial cells
and both selected-match necessity/selectivity decisions. Its only failed gate was the donor endpoint of the
`SELECT/s1p0` “irrelevant source” control: 6/9 correct and a bootstrap lower mean margin of `-0.0058`.

That control was not causally irrelevant to its endpoint. In every R552 prompt the two target pairs were
`A -> B` and `C -> D`, and the endpoint was the correct payload logit minus the *other target payload* logit. Editing
the unselected source changed the token immediately before that other payload. It therefore changed the contextual
representation of a token used directly in the measured margin, even though the planted hard-equality selector from
R557 was unchanged. With only nine SELECT groups in each `(S,P)` control cell, token and slot variation then made a
single all-or-nothing invariance bar brittle.

R578 asks whether the same selector × payload behavior survives when generic source-edit damage is measured on a
third pair whose source and payload are absent from both endpoint logits. The old edit is retained under the honest
name `contrast_target_source_edit`; it is measured, but is no longer required to be invariant.

## Fresh three-pair construction

Each group has three earlier adjacent pairs:

$$
A\to B,\qquad C\to D,\qquad X\to E,
$$

where only the first two participate in the task. The final query is `A` or `C`. Payload assignment zero uses
`A -> B, C -> D`; assignment one swaps only those target payloads, giving `A -> D, C -> B`. The neutral pair
`X -> E` never changes. All six variable tokens are distinct, and neither `X` nor `E` is an answer or contrast-logit
token.

The four factorial answers remain `B,D,D,B` for `s0p0,s1p0,s0p1,s1p1`. Thus selector-only and payload-only edges
change the answer, while changing both factors preserves it. For each group, the physical order of the `A`, `C`, and
neutral pairs is deterministically rotated. This balances selected, contrast, and neutral source positions rather
than identifying a logical role with one prompt slot.

Each group contains these paired counterfactual families. Every control in items 4--8 is crossed with all four
factorial cells, so SELECT has 36 groups rather than nine in every control cell:

1. two selector swaps, holding all three pairs and payload assignment fixed;
2. two target-payload swaps, holding query and source identities fixed;
3. two joint selector-plus-payload diagonals whose answer remains fixed;
4. a selected-target match break in each factorial cell;
5. a neutral-third-source edit `X -> X'` in each factorial cell;
6. a neutral-third-payload edit `E -> E'` in each factorial cell;
7. a contrast-target-source edit in each factorial cell, reproducing the semantic scope of R552's old “irrelevant” edit without calling
   it invariant;
8. a filler replacement and a lag extension in each factorial cell, preserving all three pairs and the answer.

The selected, neutral-source, neutral-payload, and contrast-source edits each change exactly one token. Decoys are
distinct from every source, payload, filler, prefix, and query token in their group. Every prompt has exactly one
earlier occurrence of the final query and its registered payload follows immediately.

## Frozen split authority

The statistical unit is the complete semantic group. There are 180 groups: 72 FIT, 36 SELECT, 36 FINAL_TEST, and
36 OOD. Every factorial condition and derived control from one group stays in that split. The four splits use
disjoint prefix/layout families and disjoint GPT-2 token-ID banks. Within a split, the builder consumes a disjoint
token block for every group, so sampled variable, filler, extension, and decoy tokens do not cross groups either.
Exact prompt sequences and exact prompt-answer pairs cannot cross groups.

FINAL_TEST and OOD remain unopened. The row artifact records zero model loads, forwards, backwards, and opened
outcomes. OOD uses code/trace prefixes, a different token bank, and longer source-to-query lags.

## Outcome-blind construction checks

The builder and independent tests must establish exactly:

- 180 groups, 5,400 rows, 720 factorial conditions, and 5,040 unique prompt sequences;
- the per-family counts implied above and complete family coverage in every group;
- exact `B,D,D,B` answers and the selector-only, payload-only, and joint-diagonal relations;
- exactly one earlier query match and immediate payload adjacency in every factorial prompt;
- exact one-token selected/neutral/contrast edits at their declared roles;
- unchanged endpoint tokens for every answer-preserving control;
- group-disjoint splits, sequences, prompt-answer pairs, and sampled token blocks;
- tokenizer round trips, artifact hashes, and no model access.

Any failure terminates construction and no capability run is licensed.

## Frozen future native-capability decision

A later separately implemented model run may open FIT and SELECT only after binding this preregistration, the rows,
the receipt, and the independent CPU test hashes. It must evaluate every unique sequence exactly once and use the
semantic group as the bootstrap unit.

For answer tokens `B,D`, define the common signed coordinate

$$
z_{sp}=\operatorname{logit}(B)-\operatorname{logit}(D),
$$

and the selector × payload interaction

$$
I=\tfrac14(z_{00}-z_{10}-z_{01}+z_{11}).
$$

The capability gate is frozen as follows in both FIT and SELECT:

1. every `(S,P)` cell has at least 75% correct answers and a group-bootstrap 95% lower mean correct-vs-other margin
   above zero;
2. the bootstrap 95% lower mean of `I` is above zero;
3. both endpoints of the neutral-source, neutral-payload, filler, and lag controls separately meet the same 75% and
   positive-lower-margin bars;
4. at least 70% of selected-match breaks reduce the base correct-vs-other margin, with positive bootstrap lower mean
   reduction;
5. the paired selected-match reduction exceeds the larger absolute effect of the neutral-source and neutral-payload
   edits, with bootstrap lower mean greater than zero.

The contrast-target-source edit is reported by `(S,P)`, source slot, and direction but is not an invariance gate: it
changes the contextual input to the competing payload and may genuinely affect the endpoint. This reporting tests
the R552 diagnosis rather than hiding it.

Only if all five decisions hold may the existing R557 score/value intervention semantics and the already frozen R558
selector-score × payload-value site lattice be adapted to these rows. R558 must not be run unchanged or on R552.
No attention head, activation rank, subspace dimension, or storage measure is selected by this construction.

## Prior work that remains authoritative

- The four-head equality-fetch tensor already has held extraction and OOD prediction; this rung does not relocalize
  induction or repeat whole-head removal.
- R557's exact score/value transplant algebra is reusable after rebinding it to the new row hash.
- R558's subset scoring, crossed-factor controls, group bootstrap, and Boolean-lattice interaction accounting are
  reusable after the native capability gate holds.
- R459/R464/R500 already establish complete-score sharing and an MLP9 reader on natural text. R531 found no
  identifiable shared individual Q or K branch, so this route stays at operational score/value factors rather than
  treating a head or one Q/K coordinate as the semantic basis.
