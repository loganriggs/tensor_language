# Rung518 preregistration: circuit-defined quotient of MLP0 head-by-source pieces

Registered 2026-09-03 02:24 UTC before any rung518 model outcome.

## Question and reason for this object

Rung517 found that five cross-head attention-source relations have stable MLP0 effects, but they are diffuse and
strongly substitutable. Rung417 had already shown that nine whole-head MLP0 interaction paths are not duplicate
services at attention1/MLP1. Neither result used the existing62 circuit families to define the units.

Rung518 crosses the two decompositions. It splits each of attention0's nine heads into the same five exact source
relations, producing45 fixed head-by-relation pieces. It asks whether pieces with different native head identities
have the same finite downstream task-and-circuit effects, and whether source pieces inside one head have demonstrably
different effects. Native heads are candidates, not presumed semantic units.

This directly targets cross-module grouping and within-module splitting, held-out circuit prediction, and physical
interchange. It does not optimize rank, reconstruction error, sparse-code width, quantization, or parameter count.

## Exact pieces and interventions

For head `h`, query `q`, source `s`, and source relation `g`, define

`a[h,g,q] = O_h sum_{s in g(q)} pattern[h,q,s] value[h,s]`,

using the deployed attention0 scores, value mixing, rotary positions, causal mask, and output projection. The45
pieces plus one explicitly retained arithmetic remainder must sum to the deployed attention0 write.

For each piece `i`, compute two MLP0-only interventions while attention0's direct residual-stream write remains
native:

- `SINGLE_i`: arithmetic remainder plus piece `i` is the context supplied to MLP0;
- `DROP_i`: the full native context minus piece `i` is supplied to MLP0.

`EMPTY` and `FULL` are shared baselines. A piece's signed singleton benefit is `CE_EMPTY - CE_SINGLE_i`; its signed
removal effect is `CE_DROP_i - CE_FULL`. Both operating points are retained separately. They are never averaged
before equivalence testing, because rung517 showed strong context redundancy.

## Existing task and circuit coordinates

Reuse the hash-checked rows, copy-task masks, and circuit masks exposed by rung510's input validator. Discovery uses
documents500:748, split at624, with the32 discovery circuit families. Confirmation uses documents752:1000, split at
876, with the other30 circuit families. Documents748:752 remain unused.

For each circuit tag, the response coordinate is the intervention's mean cross-entropy change on member positions
minus its mean change on matched control positions. Singleton and removal backgrounds are separate. The four frozen
copy-task coordinates are also separate. Thus every atom has, in each document half:

- two32-coordinate circuit vectors and two four-coordinate task vectors during discovery;
- two30-coordinate circuit vectors and two four-coordinate task vectors during confirmation.

This uses the circuits as measurements that can separate or merge computations. It does not assume their labels are
complete, and it does not compare coordinates with different circuit identities across splits. Each proposed atom
pair is compared within the32 discovery coordinates and then independently within the30 held-out coordinates.

## Pairwise operational equivalence

Test all `45 choose 2 = 990` unordered pairs without ranking. For pair `(i,j)`, fit one signed scale

`beta = (response_i dot response_j) / (response_j dot response_j)`

on the concatenated singleton/removal circuit vector in discovery half0. The same `beta` must predict both operating
points, all four task coordinates, discovery half1, and later the untouched30 circuit families. This is a same-use
test, not similarity of stored activation vectors.

A discovery pair is material only if both atoms have circuit RMS at least`.0005` nat and task-vector norm at least
`.00025` nat in both operating points and document halves. It must have `.25 <= |beta| <= 4`. In both discovery
halves, signed circuit cosine must be at least`.85`, both directional relative residuals at most`.50`, signed task
cosine at least`.70`, and both task residuals at most`.65`.

Sixteen fixed circuit-coordinate permutations independently scramble one atom's circuit identities while preserving
every marginal response. A real discovery relation must contain1--16 pairs and its count must strictly exceed the
95th percentile of the16 permuted counts. No top-k pair list or threshold relaxation is allowed.

## Frozen predictions

### A — exact and live instrument

All causal query/source edges belong to exactly one relation; the45 pieces plus arithmetic remainder reconstruct the
deployed attention0 write with relative squared error at most`1e-8`; FULL reproduces native MLP0 and logits exactly;
EMPTY replays deterministically; all90 SINGLE/DROP edits are live in both discovery halves; call counts, row hashes,
task supports, circuit supports, and the32/30 circuit partition match; and eight planted45-atom response problems
recover exactly their four planted proportional pairs with no false positives.

### B — small circuit-defined relation

Between1 and16 of the990 atom pairs pass every discovery materiality, scale, circuit, task, and two-half rule, and
the real count strictly exceeds the fixed permutation-control 95th percentile.

### C — held-out circuit prediction

At least one frozen discovery pair passes both confirmation document halves on the other30 circuit families using the
unchanged discovery `beta`: circuit cosine at least`.75`, both circuit residuals at most`.55`, task cosine at least
`.70`, and both task residuals at most`.65`, separately retaining singleton and removal backgrounds.

### D — bidirectional physical interchange

For at least one confirmed pair, replace atom `i` by `beta * atom_j` in the otherwise native MLP0 context and perform
the reverse replacement with `atom_i / beta`. On confirmation rows, each direction must recover at least70% of the
target atom's removal-effect vector over the frozen task plus circuit coordinates, preserve its direction at cosine
at least`.80`, and add at most`.002` nat absolute damage on the frozen off-target task mask.

### E — a native-boundary-changing circuit unit

At least one physically validated equivalence crosses native attention-head boundaries. In addition, for one of the
two participating heads, another material source-relation piece from that same head must be non-equivalent to the
validated component in both discovery halves: absolute circuit cosine at most`.50` in at least one operating point,
with a different largest-magnitude circuit coordinate. This is the minimum evidence that the operational component
both merges across heads and splits a native head.

## Interpretation and stopping rules

- A failure repairs only the instrument; no model outcome is interpreted.
- If A holds and B fails, the fixed head-by-relation atoms do not expose a small circuit-defined quotient. Leave this
  basis rather than tuning thresholds, adding rank, or grouping by activation similarity.
- If B holds but C fails, record an in-sample relation only and do not inspect physical substitutions.
- If C holds but D fails, the relation is predictive but not interchangeable and cannot be called one circuit.
- D licenses an operational equivalence component. E is required before claiming that it improves on native head
  boundaries.
- Even a full pass is identification, not adoption or compression: it adds and saves zero deployed parameters. A
  later executable implementation must still price and jointly compose the accepted components.

Maximum discovery work is92 model forwards per four-document batch (`EMPTY`, `FULL`, and45 SINGLE/DROP pairs), or
5,704 forwards. Confirmation and physical substitution open only for at most16 frozen candidates and are bounded by
4,216 additional forwards. No backward pass or training is used.
