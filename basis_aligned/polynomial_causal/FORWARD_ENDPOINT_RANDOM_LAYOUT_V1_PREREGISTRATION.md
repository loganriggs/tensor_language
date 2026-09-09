# Automatic forward endpoint-message family on random layouts

Endpoint swaps pass on two registered fixed-location joins. The shared compiler
exactly expresses each selected E-value message as rho(t,s)*phi(endpoint), with
rho=P_L2H1(t,s)*RMSgain_L2(s), phi=O_L2H1 V_L2H1 Emb(endpoint)/8. A token parser
finds earlier v→w records supplying later u→v records. It preserves the complete
native complement rather than assuming an entire head is a pure equality kernel.

Test whether this circuit family generalizes to arbitrary serialization and many
simultaneous consumers. Fresh seeds24909/24910;16 IID24-cycle and16 OODthree8-cycle
worlds. Shuffle all24 binding records uniformly, with no fixed slots. Each world
has all24 query entities×hops0..3 (96 variants);32 independent worlds,3072 queries.
Perworld endpoint map sigma is one random24-cycle, generated without model access.
It has no fixed entities and is applied only to the compiled endpoint fields.

Parser includes exactly source-record-index<target-record-index and
source.key==target.value. It sends source.value's embedding term to both key/value
positions of that target through L2H1. Split by even/odd target record index into
two groups with disjoint destination supports. All inferred joins are retained.
The original L2 routing/gains and exact remaining native computation stay live
as token-derived context, fixed across compiled field interventions.

For queryq and h in{2,3}, let u=F^(h-2)(q), v=F(u), w=F(v). If the v→w record
precedes u→v, the relevant join is forward: all-field mapping predicts sigma(w).
A single parity-group mapping predicts sigma(w) iff the u→v target has that
parity. Otherwise predict the original answer. h0/1 and backward-oriented h2/3
queries are controls. This eligibility rule is structural and fixed before
native outputs; do not filter native errors or select favorable positions.
It is a falsifiable semantic extrapolation, not an algebraic guarantee.

Arms native, identity, mapEven, mapOdd, mapAll, removeAll. Removal deletes the
parsed endpoint E terms, not whole native heads or complete route writes.
Candidate uses independently exported weights and computes its own phi/cache;
original native model supplies independent pattern/value-input hook reference.

A instrument: CPU parser/identity/hook controls, finite; full-output candidate
vs native intervention max<=1e-9,relativeRMS<=1e-10(floor1e-6); native baseline
same bars; all native capable group accuracies>=.8 for eligible hop2/3 separately
in each population, each eligibility/parity group nonempty. Exact all-field
removal goldP loss>=.25 on eligible hop2/3 perpopulation. Retain all failures.
B family retarget: mapAll desired-answer accuracy>=.8 and mean desired-answer
probability gain>=.5 overnative, separately for eligible h2/h3 ineachpopulation.
C reuse: matching mapEven/mapOdd meets those same retarget bars for each
eligible hop/parity/population. Nonmatching single group abs mean original-gold
probability change<=.10 ineach eligible hop/parity/population.
D specificity/composition: mapAll abs mean original-gold probability change<=.10
for each hop0/1 and backward-oriented hop2/3 perpopulation; at finalqueries,
raw-logit mapAll effect minus sum of mapEven/mapOdd effects max<=1e-9.

One valid failure rejects this arbitrary-layout semantic family as specified;
no head, layout, endpoint map, source, rank or routing-fit rescue. If it passes,
promote the automated circuit-family interface with exact scope and remaining
opaque dependencies, then attack the remaining structural-description gap.
All387968 independent native constants remain charged, plus24576byte derived
dictionary cache. No new independent coefficients or native parameters removed.
This test cannot by itself establish the user's structural simplification goal.
B4 orB8 FP64,1800s,<256MiB/tensor, managed GPU only. No calibration or fitting.
