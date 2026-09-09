# Which query information drives the two endpoint-read routes?

The exact I/J endpoint partition shows backup behavior on16/63 initially
correct target cases: either node can be removed alone, while removing both
fails. The original I nomination still fails its IID .5 gold-loss threshold.
Do not rescale or reinterpret that test as passed. Distinguish the query-side
computation of I and J before naming them direct versus advanced lookup.

Use all512 opened requests from the same overlapping cohort. With
frozen_payload_lineage_reference.prepare/propagate, transport only the original
query-entity embedding at position49 through the three prefix layers using
the native prefix attention/RMS gates. Let L be that transported payload at
the final query position50; let C=x_query-L. The exact prefix gate/weight
dependencies remain charged. L is a payload lineage, not the effect of a raw
token edit or a proof that it encodes an unadvanced entity. Context-dependent
gates can transform a payload originating from that entity.

Keep final source K1/K2/V, query RMS gain and residual skip native. Evaluate
query Q1/Q2 using raw query states L+C, L, C, and0 at that gain, changing only
position50. For each query state, compute the existing origin-key/endpoint-value
interaction I and total endpoint read T; define J=T-I. Every route is quadratic
in the query state. Its exact decomposition is therefore

    route_full = route_L + route_C + route_mixed,
    route_mixed = route_full-route_L-route_C.

Use existing lineage, interaction, exact-source and source-port primitives.
The source-port executor accepts changed query features in a context copy and
original source features separately. Independent native Q1/Q2 projection hooks
around the existing four-arm key/value oracle verify the terms. Do not change
source/query residual states or normalization while calling it a Q-only edit.

A: zero query gives zero route; native route/I knockout reproduces saved
ORIGIN_ENDPOINT_INTERACTION_V1 rows SHA
4e25c055bc7822349261b88db9ade9bb0cad7c9caaf6d525bc56344285f82f8b;
native all-root lineage closes; Q-port oracles agree abs<=1e-9/relative<=1e-10
using live sums. Include a planted mixed-query negative so cross terms cannot
be silently omitted. All rows and query/hop groups remain.

B fixed query-lineage nomination on forward hop3, separately IID/OOD:
knocking out I_L from native logits must recover [.8,1.2] of the signed gold
effect of knocking out full I, while removing I-I_L gives absolute ratio<=.2.
Conversely J_C must recover [.8,1.2] of full J's effect and J-J_C absolute
ratio<=.2. Denominator below .001 invalidates the panel, never filters it.
Report complete29-vector relative errors with floor1e-6 and every other
query/hop effect. A task-level nomination is not full-output sufficiency.

If B fails, reject this clean payload-lineage assignment; do not select a
different head, root or pair of polynomial terms. If it passes, further
identity/interchange tests are needed before calling I a direct lookup or J
an advance-then-read route. The earlier pure read-after-advance null is not
rescued. No fresh/OOD identification claim from these opened diagnostics.

CPU threads2,batch8 FP64,alarm180s,tensors below256MiB,no GPU/training.
All387968 export coefficients, prefix gates, opaque remainder and adapter
computation remain priced. This protocol begins the next concrete action:
the exact query-side factorization and its fixed falsifiers are specified;
implement the small Q-port oracle and execute the bounded screen next.
