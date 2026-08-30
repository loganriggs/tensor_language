# Hourly strategic review — 2026-08-30 08:28 UTC

## Outcome first

The strict whole-model ledger remains:

- **5.348245316%** certified removable storage
  (`29,196,288 / 545,904,054` values);
- **10.923302467%** named deletion-CE (`0.57968 / 5.30682`);
- **4.72714 nat = 89.076697533%** unexplained deletion-CE;
- **0/68** terminal circuits jointly passing extraction, selective removal, and OOD.

This review produced executable infrastructure, not a numerical model result. Commits
through `9d4d835a` add three missing FIT-executor primitives:

1. exact semantic reload of authority, source, audit, parents, protocol, and namespace;
2. a serialization-independent hash of every named model parameter and buffer;
3. a common create-only terminal record for mutually exclusive receipt/failure.

The exact detached-worktree suite is **67/67**. Independent outcome-blind audit gives
this executor sub-layer GO, with 21-file closure
`b3c210fc18df76ee956a655a51da5c7d3442c36ef7958f7004049576e8c71ad7`.
No causal-response model forward or protected outcome was opened. The verdict is not
authorization for a public executor or scientific launch.

## What changed strategically

The previous highest-priority blocker was whether canonical FIT inputs and the bundle
could be trusted. That boundary is independently GO. The blocker has moved one
interface downstream: a sole owner still has to join authority → inputs → checkpoint →
one-use collector → bundle → manifest → terminal receipt.

The new terminal representation improves that chain. A fully staged JSON record
contains the authority hashes, exact partial-artifact aggregate, and success/failure
payload. The common `TERMINAL` path and selected `RECEIPT` or `FAILURE` path are hard
links to the same inode. Thus success and failure serialize on one create-only link;
if the second link fails, the terminal itself still contains the complete record.

The model hash is logical rather than a `torch.save` file hash. For each sorted state
name it hashes

$$
(\text{name},\text{dtype},\text{shape},\text{raw tensor digest}),
$$

then hashes the ordered list. Equal model states therefore compare equal even if a
serialization library changes irrelevant container bytes. A one-element mutation in
a tiny model changes the digest; restoring the same state reproduces it.

## Running work and new evidence

The GPU is occupied by the independent ten-circuit a8/a16 DAS seed-stability job. One
seed completed in 694 seconds and the second is in progress. Partial rows are not
banked as conclusions. This is useful because the prior M16 test showed that
single-seed selectivity margins can vary by 0.046–0.319, much larger than several
reported threshold clearances.

The expired eight-hour entrypoint deadline was 2026-08-29 12:00 UTC; it no longer
controls this review. No plan or unrun runner is counted as evidence.

## Largest remaining gaps

1. **89.08% residual causal CE is unnamed.**
2. The signed $2\times49\times49\times\text{document}$ response object has not yet
   been measured on bilin18.
3. MLP0/1/2 simplifications have not composed across live residual/RMSNorm interfaces.
4. No circuit has all four properties: extraction, selective removal, low collateral,
   and OOD transport.
5. Geometry and HOSVD do not reliably select causal shared/private structure.
6. The complete FIT executor, manifest validator, failure observation, and receipt
   schema are still absent.

## Ranked actions after pruning

### 1. Complete the canonical no-argument FIT owner

Highest information gain and whole-model relevance. It must internally fix CUDA
float32 bilin18, 496 FIT rows, 49 sources, batch size four, and 12,400 forwards. It
must own the private input guard and bundle callback and never return directions or an
EVAL capability. Remaining work: bundle-to-manifest derivation, protected pre-forward
guard, model/checkpoint before/after checks, hash-bound failure observation, and final
receipt.

### 2. Fit shared/private block terms on signed FIT responses

Compare independent factors, shared parents plus sparse children, and a
parameter-matched dense control. Select on FIT only; score signed EVAL prediction per
stored value and multiply-add. This directly tests whether a DAG is useful simplicity.

### 3. Apply quotient-Jacobian gauge accounting

For every candidate factorization, compare Jacobian nullity with its known gauge
group. Reject decompositions with unexplained non-gauge null directions. CPU cost is
low and the result guards editability and canonicalization.

### 4. Finish bracket or successor as a terminal circuit

These have mature extraction/removal/OOD code and are closer to 1/68 than new circuit
discovery. They also provide qualitatively different downstream observables for early
layer decomposition.

### 5. Estimate the suffix observability/Fisher quotient

Compress an early write only modulo directions the downstream suffix cannot
distinguish. Validate with finite interventions and CE, not only infinitesimal or local
MSE. This may turn rank-512-looking maps into smaller task-relevant programs.

## Pruned this hour

- **More M16 rank-one fitting:** seed variance and weak selectivity make return low.
- **Geometry-only HOSVD/SAE selection:** useful initializer, falsified as a causal
  hierarchy selector.
- **Unsigned concentration ratios:** discard sign, scale, counts, and additivity.
- **Full circuit powersets:** premature with 0/68 terminal circuits.
- **Launching FIT now:** the complete source-closed owner has not passed audit.

## Executed action and falsification coverage

The CPU-side implementation and attacks cover:

- authority role, logical identity, source, audit, parent, protocol, and output-path
  drift;
- exact model-state replay and a one-parameter mutation;
- success/failure exclusivity through one terminal path;
- late protected-state failure before any terminal link;
- a forced second-link failure, proving the terminal still stores a complete record;
- lock replacement at both hard-link boundaries and in-place same-inode mutation
  before the second link;
- create-only inode identity between terminal and receipt/failure.

This advances the actual executor interface and is not counted as explained model
behavior. The next code action is the manifest and canonical owner; the next scientific
action remains the independently audited FIT collection.
