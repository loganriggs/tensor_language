# Hop-instruction transfer between product-attention query factors

The fixed raw/join error rule fails fresh OOD repair coverage, and a shared signed
copy decoder fails full-vector accuracy on all40 opened removal cases. Preserve
both nulls. Change the object to how query instructions choose binding reads.

For a fixed function, binding order and query entity, inputs at hops0..3 differ
only in their final token. Causality therefore makes every initial binding's
postL2 state, final K1/K2/V and normalization identical across requests. The final
query has two independent factors Q1 and Q2 whose product controls its read.

Use fresh random-layout seeds31909/31910, first8 worlds/population, all24 query
entities, all12 ordered distinct pairs of hops0..3:4608 ordered transitions on
16 independent worlds. IID24-cycle/OODthree8-cycle graphs, no renamed duplicate
cohort. No trained outcomes opened. Generate all requests before querying.

For each recipient/donor pair, preserve recipient source K1/K2/V. Replace neither,
only Q1, only Q2, or both finalquery factors with donor factors at the same
absolute RoPE position. The target is the29-dimensional logit contribution from
all48 initial binding tokens, including final O/head folding and .5 residual
coefficient. Exclude all three query suffix sources and the residual readout
from this explicitly partial target. Neither is claimed reproduced by Q transfer.

Native correspondence uses final projection hooks at the last query and a
pattern mask selecting binding-source columns; subtract the known residual
before readout. Independently exported token-derived execution uses the same
declared interface. Both-Q transfer must equal the donor binding read because
source states coincide. The full donor model also changes residual/query-suffix
terms; the primitive must include a live scope-negative control for that claim.

A mechanical: model-free controls, source-state identity, finite results;
all four native/export binding-read vectors and both-Q/donor closure max<=1e-9,
relativeRMS<=1e-10 with1e-6 denominator floor; identity/restored hooks. Native
accuracy on complete requests is reported, separately from mechanical validity.

B Q1-instruction alternative: for each population and all12 ordered hop pairs,
the Q1-only centered binding-read change predicts the donor-minus-recipient
centered vector with relativeRMS<=.01 (1e-6 denominator floor).
C Q2-instruction alternative: the identical bar for Q2-only, scored separately.
These are two prospectively declared hypotheses; report both, never choose a
head or subset of hop pairs after failure. An empty group is untested, not pass.

D composition/accounting: Q1 main effect plus Q2 main effect plus their exact
mixed bilinear interaction reconstructs both-Q transfer within1e-9. Report the
interaction's centeredRMS and relative size; do not interpret additivity of
this expansion as independence if the mixed term is large. If B and C both
fail, the native query factors do not separately supply hop control at these
bars. No rank/decoder/dose fitting or whole-head routing replacement follows.

All387968 original export coefficients and both token-derived contexts are
charged, no native coefficients removed. This is a paired-context partial
read-interface study, not an autonomous hop program or full-model simplification.
FP64,B8 where practical,1800s,256MiB per tensor; GPU only through managed runner.
