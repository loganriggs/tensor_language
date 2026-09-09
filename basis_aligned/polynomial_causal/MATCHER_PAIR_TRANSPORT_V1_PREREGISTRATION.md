# Transport the contextual matcher pair as one unit

Individual QK factors fail both raw and consumer-conditioned matching gates.
Matched middle-label changes can give large single-factor hybrid score norms,
while the joint product has a similar aggregate norm. That does not yet show
pointwise or behavioral invariance. Test the complete pair with live readers.

Use all opened32 worlds and six orders in suffix_join_middle_match_reference.py.
Each recipient is the base case; its donor is the corresponding both-label-change
case, which preserves middle matching, query identity and answer. Retain all
four source-pair cells and the original H1-forward/H2-backward assignment.
Fork each recipient into query hops0–3, reusing the same binding-only donor
factors. This gives768 recipient requests, not768 independent worlds.

Keep recipient tokens, values and every other score unchanged. At the selected
cells set the product to recipient/recipient (00), donor1/recipient2 (10),
recipient1/donor2 (01), or donor/donor (11). Include zero-score route removal.
Compute every factor from tokens. The compiled intervention forms the literal
L2 residual write delta from changed scores and recipient V/O, then uses the
existing exact final-layer source-edit executor. The independent native oracle
replaces attention cells directly. All native source/query normalization stays live.

A mechanical: all five full-output cells and native cut-effect vectors agree
at1e-9 absolute/1e-10 relative RMS; original/saved score endpoints replay and
binding factor values are identical across query hops; controls finite and live.
B pair portability: donor/donor gate transfer reproduces recipient native full
output at meanKL<=.001 and p99<=.01, for both all-token and query distributions,
in each population/orientation/hop group. Its centered query-output change must
also be <=.01 of the native route-removal effect RMS, denominator floor1e-6.
Report near-zero denominators and every group. Aggregate gate RMS is not this test.
C composition instrument: compiled mixed full-output effect11-10-01+00 agrees
with the native factorial at the same1e-9/1e-10 bars. Report individual-factor
effects and mixed strengths; do not assume logit or probability additivity.

This is an opened native transport screen. A B pass requires fresh confirmation
before identification; a B failure closes this matched-label gate transport,
with no scalar/entity/role fit or selected cell/head/hop rescue. Individual arms
are diagnostics, not fallback candidates. All387968 native export constants,
prefix computation and adapters remain charged, with zero coefficients removed.
No structural model reduction is inferred from exact intervention machinery.
B4FP64,1800s,256MiB per tensor; GPU only through the managed queue.
