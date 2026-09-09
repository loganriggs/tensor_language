# Lower bounds for separating the three query-root inputs

Both entity/task and entity/document dominance hypotheses failed. Stop trying
to assign a whole route to one input class. Use their measured mixed contrasts
to test the entire class of additively separated response programs.

For each frozen native context and route R=I or J, let f(a,b,c) be its centered
query-logit contribution with raw Q input aL+bH+cD, native query gain and native
source features. Consider the binary root-gating cube a,b,c in {0,1}. An
approximator additive across any partition has zero mixed finite difference
between variables in different blocks. For a separated L/H pair,

    C_LH = f(1,1,0)-f(1,0,0)-f(0,1,0)+f(0,0,0).

If e is its approximation error, C_LH is the signed sum of those four errors.
Triangle inequality gives max_vertex ||e||_RMS >= ||C_LH||_RMS/4. This holds
for arbitrary nonlinear functions within blocks, not just linear functions or
our rejected candidates. The same applies to L/D. Norm can stack the complete
fixed population/query/hop group; rows retain the same corner index throughout.

Use saved mixed terms LH/LD from ENDPOINT_MIXED_QUERY_INPUTS_V1 rows SHA
c4866f1202dcab9fed8236b983da8577c985b6f15dd7b787d5472537bfe42a6d
and full route vectors from ENDPOINT_ROUTE_QUERY_LINEAGE_V1 rows SHA
f790aa9c396b12e57d207f821be9217413255d1ac170b553999426163687b048.
Require identical metadata and replay LH+LD=LC using live native sums.

For the three nontrivial bipartitions, report bounds:
L | HD: max(||LH||,||LD||)/4;
H | LD: ||LH||/4;
D | LH: ||LD||/4.
The unmeasured HD edge can only strengthen these bounds; do not claim it is
zero. Normalize by the full native route RMS (floor1e-6) and compare with .01.
This is a uniform four-corner error bound relative to one fixed native-route
reference, not the previous per-intervention relative-error bar. Report every
population/query/hop/route group, including small effects and floors.

Instrument: exact stored-table partition check, finite values, and elementary
scalar controls. For f(a,b)=ab, the lower bound .25 is attained by
g(a,b)=.5a+.5b-.25 on the four corners; a purely additive function has zero
contrast. The inequality is derived here from triangle inequality. Its numeric
right side is evaluated in FP64, not interval-certified. Report a separate
4e-9 contrast-slack sensitivity check as a robustness diagnostic, not a proof
of real-native arithmetic error. All rows are opened; no model/GPU/fit.

A positive bound constrains additive grouping over these precise Q-port root
interventions. It does not rule out a joint bilinear node, a different variable
map, arbitrary context-dependent circuits, or any whole-model approximation
without this interface. All native weights remain priced. This test changes
the permissible circuit boundary rather than proposing another rank or root
subset. CPU threads2,alarm180s,tensors below256MiB.
