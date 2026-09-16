# Research update: a reusable sparse-interaction graph layer

The subject-number result established one circuit with the four desired traits,
but its graph search and evaluation logic were still embedded in a
behavior-specific runner. I extracted that logic into
`sparse_interaction_graph.py` so the next circuit can use the same definitions
instead of reimplementing signs, corner requirements, selection, and metrics.

## What is now shared

Given named intervention ports and measured Boolean-lattice corners, the module
provides:

1. exact vector-valued Möbius decomposition and reconstruction;
2. enumeration of eligible interaction orders;
3. the minimal corner set required by a proposed sparse graph;
4. signed damage atoms whose sum predicts removal damage;
5. coefficient-free greedy selection on discovery rows only;
6. frozen component-relative evaluation on disjoint OOD panels; and
7. explicit rejection of discovery/OOD row overlap, missing corners, invalid
   masks, and shape mismatches.

The exact operations preserve Torch tensors, NumPy arrays, or scalars. NumPy is
only used for discovery selection and reporting. This keeps the algebra usable
for residual vectors and unsupervised DCT responses, not only scalar behavior
margins.

## Independent replay audit

The CPU audit replayed two separately developed exported circuits through the
generic implementation:

- the five-port, nine-edge subject-number graph; and
- the two-port equality MLP4 child/remainder interaction graph.

Maximum parity errors were `4.44e-16` and `2.22e-16`, respectively. A separate
four-port synthetic set function had transform and closure errors below
`2.72e-15`. All five unit tests pass.

For the subject-number graph, the generic corner compiler independently returns
exactly `(0, 1, 2, 4, 8, 9, 12, 16, 20, 24)`. Thus the frozen nine-edge graph
needs 10 corners rather than all 32, a `68.75%` reduction. This is a real
evaluation-price reduction once the graph is frozen; it is not evidence that
discovery can skip the full lattice.

## Positive red-team of the DCT reader null

I also rechecked whether the failed raw weight-only MLP9 unsupervised-reader
experiment was contradicted by the later successful contextual rank-16 DCT
node. It is not. The failed experiment had exact analytic/autodiff contraction,
and a post-hoc seed-matching audit found four individually stable factors. The
stable rank-four output span captured `0.0576` of the CrossFirst reader versus
an equal-rank random median `0.0581` and random maximum `0.0814`. The other four
factors were unstable. That preserves a real weight-tensor structure while
supporting the narrower null: raw weight-only factors are not useful
CrossFirst readers.

The contextual rank-16 node uses native context and a response-weighted output
metric, so its success does not rescue the stronger weight-only reader claim.
This distinction matters for future decomposition: context must remain an open
port when the weight-only tensor is not identifiable or behaviorally aligned.

## Scope and next use

This is infrastructure, not a new behavioral circuit. It removes repeated
implementation choices that can create false negative or false positive
composition results, and it makes each candidate's port count, selected edges,
required corners, and OOD split mechanically inspectable.

The next model-facing experiment should use this unchanged module on a second
behavioral family, freeze the graph on discovery, and require the same four
tests: token/native-input prediction on genuinely unused prompts, executable
extraction with counted ports, equal-norm selective-removal controls, and
component-relative joint composition. A second pass is necessary before the
workflow itself—not just the subject-number circuit—can be called reusable.

## Artifacts

- `sparse_interaction_graph.py`
- `test_sparse_interaction_graph.py`
- `audit_sparse_interaction_graph_v1.py`
- `SPARSE_INTERACTION_GRAPH_V1_RESULT.json`
