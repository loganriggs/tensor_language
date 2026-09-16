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

## A second four-trait family already exists

Auditing the equality chain under the same rubric found that the normalized
MLP4-input L5H5 factor graph already supplies an independent four-trait example.
Its four precision-correction nodes compose exactly; the frozen graph replays
behavior on repository-disjoint code documents; named-node removals and the
rolled control are causally validated; and its standalone executor constructs
the MLP4 product, residual corners, Q/K projections, and scores internally.

I added a hash-bound verifier tying those claims to the source result and made
the boundary explicit: five external native activation ports
(`mlp4_normalized_state,M2,A3,M3,A4`) plus two deterministic rotary-context
inputs. This is a weaker extraction form than the zero-activation-input
subject-number component, and all 4,608 products plus 16 Q/K projections still
run. It nevertheless shows that sparse, explicitly interacting four-trait
graphs occur in two different behavioral families and at different model
boundaries.

This is a retrospective standardization, not a prospective test of the new
selection utility: the equality graph was discovered before the shared module.

## Model-wide extracted-circuit registry

The extracted-circuit directory had grown into a useful but non-comparable set
of packages. I added a generated registry that now records, for every package,
its declared inputs, external activation count, named nodes, explicit package
dependencies, four-trait declarations, verifier outcome, scope, and next
evidence gap. Missing metadata is represented as unknown rather than false.

The current inventory contains 32 package directories, 27 manifests, and 16
declared input boundaries. Exactly two packages currently have all four traits
plus passing hash/evidence verifiers: subject number and equality M4. Thirty
packages remain partial or unassessed. There are four explicit inter-package
dependency edges and no cycles. This is important negative information: the
large number of executors is not evidence that the model has already been
decomposed into 32 four-trait circuits.

The registry is rebuilt from the filesystem and invokes available export
verifiers. Its audit checks complete directory coverage, exact regeneration,
passing four-trait verification, acyclicity, and preservation of unknowns.

## L8H4 equal-norm removal red-team

The registry identified the reversible equality L8H4 edge as the closest
uncertified boundary: OOD installation, exact removal, extraction, and bilinear
composition already passed, but its removal evidence lacked the briefing's
equal-norm same-site directional null. I preregistered and ran 16 isotropic
controls on the frozen 192-document code panel, matching the deployed target
term norm independently at every document and query position.

The instrument passed with maximum per-position norm error `0.000394`. Target
removal produced `808.79` nat summed copy-positive damage versus a random
median `163.86`: a `4.94×` advantage, and the target exceeded all 16 controls.
All four copy subcells and both frozen halves had positive target damage.

The registered overall result remains a near-miss. Target all-noncopy mean
absolute collateral was `0.001864` nat, below the absolute `.01` tolerance but
`1.13×` the random median `0.001648`; therefore the stricter registered
collateral comparison failed. The package is not promoted to four-trait status.

The first result also contained exactly the miscoding failure the positive
red-team policy anticipates: `pred_b` iterated the inherited `CELLS` tuple,
which includes `all_noncopy`, despite the preregistration requiring positivity
only on the four copy subcells. A hash-bound audit corrects `pred_b` to true but
leaves the genuine `pred_d` failure and terminal near-miss intact. A post-hoc
effect-normalized selectivity ratio is reported only as a diagnostic, not as a
replacement gate or new evidence.

## Scope and next use

This is infrastructure, not a new behavioral circuit. It removes repeated
implementation choices that can create false negative or false positive
composition results, and it makes each candidate's port count, selected edges,
required corners, and OOD split mechanically inspectable.

The next model-facing experiment should use this unchanged module
prospectively on another behavioral family, freeze the graph on discovery, and
require the same four tests: token/native-input prediction on genuinely unused
prompts, executable extraction with counted ports, equal-norm
selective-removal controls, and component-relative joint composition. That
prospective pass is still necessary before the *selection workflow*, rather
than only its graph arithmetic and audit rubric, can be called reusable.

## Artifacts

- `sparse_interaction_graph.py`
- `test_sparse_interaction_graph.py`
- `audit_sparse_interaction_graph_v1.py`
- `SPARSE_INTERACTION_GRAPH_V1_RESULT.json`
- `extracted_circuits/equality_l5h5_m4_input_port_factor_graph_v1/verify_export.py`
- `EQUALITY_L5H5_M4_INPUT_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json`
- `CIRCUIT_GRAPH_REGISTRY_V1.json`
- `CIRCUIT_GRAPH_REGISTRY_V1.md`
- `CIRCUIT_GRAPH_REGISTRY_V1_RESULT.json`
- `EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1_RESULT.json`
- `EQUALITY_L8H4_EQUAL_NORM_REMOVAL_NULL_V1_AUDIT.json`
