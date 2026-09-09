# Equality routing after the reconstruction pilot

The fixed hypothesis that the existing small model's first attention layer can be
replaced by an entity-equality/type rule failed. No new circuit meets the four
requested properties. This is a counterexample to that specific simplification,
not evidence that a simple joint decomposition is impossible.

The operation was explicit: map a query/source token pair to one of37 cases under
renaming of24 entities, then use a head-and-position coefficient to route native
values. Four heads reused the predicate. Embeddings, live normalization and values,
the output projection, the MLP, later attention and the full29-way output head
remained in the executable and in its cost. The native first-layer Q/K parameters
were physically removed in the candidate; no cached native activations were used.

This fixed grammar is motivated by the characterization of permutation-equivariant
matrices in [Deep Sets](https://papers.neurips.cc/paper_files/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf).
The theorem specifies the form of an invariant kernel; it does not assert that
this learned model implements one. That distinction was the experiment.

Eight CPU controls passed: exact planted rule recovery, perturbation rejection,
generic-kernel rejection, exhaustive orbit coverage, permutation invariance,
projection idempotence, and two checks that independent consumers remain distinct
even when values are shared.

The first GPU result is preserved as **instrument-invalid**. A lag-only table failed
its frozen full-logit tolerance because the model's cached rotary tables contain
FP32 rounding. The repair recomputed each absolute query/source kernel with those
actual cached factors. Its unreduced native patterns and complete native forward
both matched exactly in the observed FP64 runs. The repaired diagnostic then failed:

| Population | Mean KL, all positions | Mean KL, answer positions | Joint-removal relative error |
|---|---:|---:|---:|
| Fresh24-cycles |13.606|6.570|1.033|
| Renamed entities |13.617|6.852|1.028|
| OOD short-cycle functions |13.657|6.635|1.029|

The frozen KL mean bar was0.001 nats; the joint-removal error bar was0.01. Removal
means deleting first-layer equal-entity attention edges in head0, head1, or both,
with every later state recomputed. Interaction predictions also failed. The same
full token sequences are used across methods without filtering on model correctness.
Renaming is a within-distribution metamorphic control; short-cycle function topology
is the designated distribution shift. The positional repair reuses v1's populations,
so these are diagnostic counterexamples rather than a fresh confirmation.

The repair's separate absolute-versus-lag candidate logit test also failed its1e-4
tolerance (max0.00052–0.00078), even though the scientific failures are much larger.
That false predicate is retained. The symmetry rejection relies on the passing
native absolute-position control, not on treating these two approximations as exact.

The proposed lag program used370624 parameters/table constants versus400640 native,
including35520 unexplained orbit/position coefficients and335104 remaining native
parameters. This would have saved30016 constants, but failed fidelity. The exact
positional repair uses8453908 diagnostic table constants and is explicitly not a
compact candidate. Neither table earns interpretation credit.

An immediate CPU follow-up checked a specific missing dependency: token-dependent
query/key norm products. Dividing them out did not reveal an equality rule. Residual
fractions were0.961,0.946,0.919,0.813 across four heads, against a prospective0.10
bar in at least two heads; zero heads passed. Restoring the gains still leaves
residual fractions0.988,0.936,0.883,0.645. The gain grammar would add232 constants
(including special tokens; the initial board note's192 counted entities only).

The two GPU executions took1.11 and1.61 seconds after model loading, respectively;
the CPU gain diagnostic took0.081 seconds. These are execution measurements, not
total research or authoring time. Both jobs ran through Supervisor's managed runner.

The result closes naive first-layer name-invariant routing and its norm-gain variant.
It does not test a joint read-route-write rule with a different latent representation,
nor a circuit in the full bilin18 model. A better next boundary is first-layer local
token transport: test a shared self/previous-token source rule while retaining its
complete coupled query/key/value computation. This changes a causal dependency in
the executable instead of averaging away arbitrary token identity. Any successful
local transport would still need its payload computation explained and tested in
the larger model; it would not complete the durable goal.

Primary receipts: `../EQUALITY_ROUTER_V1_RESULT.json`,
`../EQUALITY_ROUTER_V2_ABSOLUTE_POSITION_RESULT.json`, and
`../EQUALITY_ROUTER_GAIN_V1_RESULT.json`. Preregistrations and executors are committed
alongside them. The original reconstruction compiler and its null remain unchanged.

## Local-transport successor: rejected

`LOCAL_TRANSPORT_V1_RESULT.json` has since landed. Its compiled two-shift program
passes dense-restricted replay and both branches have nonzero native removal effects,
but query KL is 1.579, 1.468, and 1.351 on fresh cycles, renaming, and short-cycle OOD.
Joint-removal relative errors are 0.665, 0.652, and 0.639. Thus local source transport
also fails the fixed hypothesis; increasing the radius is not the registered successor.

The next contextual hypothesis is answer-history reuse in the final attention layer.
A token-only generator audit shows about 21% repeated `(entity, hop)` queries, which
could explain the weak checkpoint's higher-hop floor without chaining several lookups.
`CONTEXTUAL_HISTORY_V1_PREREGISTRATION.md` tests this with separate initial-binding,
matching-history and control-history removals, fresh unique-query and short-cycle OOD,
and an exact full-vocabulary readout decomposition. This is prospective; generator
coverage alone does not identify a learned circuit.
