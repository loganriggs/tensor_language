# Temporal/is–was rank46 task-rank4 upstream weight-edge atlas

## Decision

The physical graph and task response programs are now separately identified.  This
experiment tests their missing connection: which native components write each task's
rank-four program, and whether the exact tensors explain that routing.  It advances
cross-boundary grouping, within-module splitting, computational specification, and
stable identification; it is not a rank or reconstruction sweep.

## Frozen objects

- The 46-source support is copied byte-for-byte from the corrected rank46 authority.
- Response sites are `MLP1`, `MLP3`, `MLP4`, `MLP6`, `L8H1`, `L9H1`, `L9H4`,
  and `L11H3`.
- At every response site, temporal and is–was rank-four bases are frozen from the
  complete rank ladder.  No basis, source set, or threshold changes after execution.
- Use the capability-qualified OOD rows and the same full native head/module patch
  semantics as the rank46 source graph.  The reverse direction remains unopened for
  confirmation after this discovery atlas.

## Measurements

For every source component `s` and task `t`, patch the complete cached output of `s`
from donor into base while all other components remain live.  At each later response
site `r`, measure the induced response delta `d[s,t,r]` and its frozen coordinate
energy `||Q[t,r]^T d||^2 / ||d||^2`, signed projection onto the task's native
base-to-donor response, and cross-task coordinate energy.

For a causally live attention edge, translate `Q` through the exact local tensors as
value-input and residual-output factors `W_V^T Q` and `W_O Q`.  For an MLP response
covector `a = W_down^T Q`, use the gauge-invariant quadratic reader
`S(a)=1/2[L^T diag(a)R + R^T diag(a)L]`.  Report the exact contraction/closure against
the patched source delta.  Weight alignment alone is never called an edge.

The discovery output is a sparse task-by-source-by-response incidence tensor plus a
ranked list of edges.  The live-edge cutoff is frozen at absolute signed target
response `0.05`.  A subsequent intervention will patch the top task-specific
edge set and its complement; the atlas itself is a screen.

## Registered predictions

1. **Authority/instrument.** All hashes, row identities, semantic positions, full
   head/module endpoints, and frozen bases reproduce; self patches are identity;
   values are finite; dry-run price matches execution.
2. **Sparse causal incidence.** For each task, at least one source-to-response edge
   carries at least `0.20` signed target response and the top quartile of edges carries
   at least `0.70` of total positive incidence.  Failure means the response is more
   distributive than this atlas resolution.
3. **Task typing.** At least two response sites have an own-task/cross-task coordinate
   incidence ratio of at least `2`, and the task edge tensors are not identical
   (Jaccard below `0.8` at the frozen live-edge cutoff).
4. **Weight closure.** At least `80%` of promoted edges at MLP/attention response sites
   have normalized exact-weight closure error at most `0.15`.  Failure distinguishes
   contextual/nonlinear routing from a directly readable tensor edge.
5. **Known graph replay.** Aggregating all 46 patched sources preserves the established
   ordering of the eight response sites and explains at least `0.80` of the full
   rank46 task-mode coordinate effect.  This is a tripwire against an atlas that ranks
   locally impressive but globally irrelevant edges.

## Price and terminal

Cache four native base/donor task batches, then execute one full-component patch per
source and task: at most 96 model forwards, zero fitting updates, zero model updates,
and zero transformer backwards.  A 5/5 result is `task_typed_weight_edge_atlas`.
Otherwise preserve the exact partial/null and choose the next experiment from the
failed clause.  No DAS optimization or threshold relaxation is allowed in this rung.
