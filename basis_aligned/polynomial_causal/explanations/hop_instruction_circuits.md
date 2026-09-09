# Where does the hop instruction enter the read computation?

The target is a reusable computation with explicit inputs, operation and
consumers, tested by full-vector prediction, extraction, removal and composition.
All work here uses the existing four-attention checkpoint. The full18-layer
goal and reduced structural description remain outstanding.

Changing only the final hop token leaves every earlier state unchanged by
causality. In particular, all48 binding-source keys and values are identical
across requests. The final read nevertheless changes through both query factors:

\[
R_h=\frac12\sum_{a,s<48}
  (q_{1,a,h}^{\mathsf T}k_{1,a,s})
  (q_{2,a,h}^{\mathsf T}k_{2,a,s})D_a(v_{a,s})/32^2.
\]

Here the target is only the initial-binding contribution to29 logits. Query
suffix sources and the residual readout are excluded explicitly. Both are still
required by the complete model and cannot disappear from its cost ledger.

## Neither native query factor independently selects the hop

[HOP_QUERY_FACTOR_TRANSFER_V1_RESULT.json](../HOP_QUERY_FACTOR_TRANSFER_V1_RESULT.json)
tests all4608 ordered hop transitions on16 fresh IID/OOD worlds. Interchanging
both Q factors reproduces the donor binding read; independent native hooks close
within1.56e-13. Native task accuracy is at least.9635. Both single-factor
hypotheses fail the1% full-vector effect bar: Q1 relativeerrors.594–1.659,
Q2 .826–1.970. No head or transition subset is adopted.

The explicit bilinear mixed term reconstructs joint transfer within1.63e-13,
but its centeredRMS is.911–2.562 times the complete hop-change effect. Thus
correct composition requires the interaction; the factors are not independent
instruction variables under these native-port interventions.

## Hop changes the addressed pattern, not just its strength

A [necessary lower bound](../HOP_COMMON_ADDRESS_BOUND_V1.json) tests whether one
head could reuse one address/message pattern across allfour hops, multiplied
by a scalar. Such a pattern implies rank<=1 for the four rows of all48 single-
source removal vectors, each centered over29 outputs. The best possible rank1
approximation is an optimistic bound; no fitted program is adopted.

All1536 world/query/head cases exceed1% error. Aggregated best-case residuals
are.435–.611 bypopulation/head. Native/source read replay closes within4.55e-13.
This rules out the specified common-address organization, not arbitrary new
circuit groupings or a smaller full model. It is not a rank-allocation sweep.

A second fixed rewrite asks whether hop-h read equals one-hop read at the
advanced query F^(h-1)(q). The [saved-output audit](../READ_AFTER_ADVANCE_AUDIT_V1.json)
uses every768 hop2/3 comparison, with exact hop1 identity controls. Relative
vector errors are.823/.820 forhop2 and1.401/1.404 forhop3 (IID/OOD). A correct
task-level graph-walk algorithm therefore does not explain these native binding
read vectors through this simple shared lookup. This proposal also closes.

## Active test: literal instruction after reusable prefix content

[LATE_HOP_INSTRUCTION_V1_PREREGISTRATION.md](../LATE_HOP_INSTRUCTION_V1_PREREGISTRATION.md)
changes the computational boundary. The residual stream contains a literal
E(hop)/8 term afterthree layers. Swap that term alone at the finalquery, then
run the final layer with its normal RMS, projections and residual. Compare
every output to the actual new-hop input, including full-vocabulary KL and
centered causal-effect vectors. Removing E(hop)/8 must erase hop dependence
if the claimed reusable-content/late-instruction decomposition is correct.

The complementary prefix-computed change is a diagnostic; the complete state
swap is a native donor positive control, not an alternative candidate to adopt.
The primitive passes18 controls, including all physical-state oracles and
unchanged earlier outputs. Fresh33909/33910 integration remains. All387968
native coefficients and any instruction adapter stay charged. No simplification
is established by the preregistration or by exact control execution alone.
