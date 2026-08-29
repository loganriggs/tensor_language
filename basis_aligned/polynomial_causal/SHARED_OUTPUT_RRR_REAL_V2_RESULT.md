# Shared-output RRR v2 — result and interpretation

**Completed:** 2026-08-29 06:09 UTC  
**Scope:** discovery only; no validation, generalization, semantic-coordinate, or
serialized-deployment claim.

## Short result

The experiment found real but limited sharing across the 36 context-free output maps.
A single common output basis is more storage-efficient than the best independently
allocated site ranks at small ranks 64 and 128.  It is nevertheless too restrictive to
match 36 independent rank-matched maps, and at rank 512 it loses even to the independent
maps given exactly the same storage.  The registered E2.1 and E2.2 decisions therefore
both fail.

This does **not** increase the fraction of the network strictly explained.  It does
identify the next mathematical form worth testing: a hierarchical shared trunk plus
site-specific residual bases, rather than either one global basis or 36 unrelated ones.

## What was computed

At site $j$, the fitted context-free map has input token embedding $x\in\mathbb{R}^{1152}$
and output write $y_j\in\mathbb{R}^{1152}$.  The global shared program uses

$$
\widehat y_j=xA_jV^\top,
$$

where every site has its own input-to-coordinate map $A_j$, but all 36 sites store one
orthonormal output dictionary $V$.  The typed program stores two dictionaries, one for
attention and one for MLP sites.  The independent program stores a different dictionary
$V_j$ at every site.

All bases and maps were chosen from the 5,419 covered fit token types before evaluation
rows were deserialized.  At evaluation time all 36 native attention/MLP writes were
replaced.  Covered token types used exact frozen one-token table rows; uncovered types
used the factorized embedding map.  Cross entropy (CE) is the model's next-token loss;
lower is better, and differences are measured in nats per scored token.

## Artifact integrity

- V1 failed before any evaluation metric because a CPU mask was indexed by CUDA token
  IDs.  Its authority and failure remain preserved.
- V2 changed only that device placement and bound both v1 artifacts.
- V2 result SHA256:
  `19d65e2c6d4a0cff19ddfb76ddbe62dcd26c462a695e006c457da85a89adc053`.
- V2 receipt SHA256:
  `57f699d680a7ea010f6ec8b12c3c33d61f1b3f540ad2891517fe751074dbdd56`.
- Semantic replay passes, the model-state hashes are identical, every registered call
  count and literal price replays, all legacy anchors pass, and no optimizer/backward
  call occurred.
- Runtime was 319.67 seconds; peak allocated CUDA memory was 4,214,539,264 bytes.

## Registered decisions

### E2.1: one global dictionary

For a rank to pass, the global arm had to satisfy both conditions on all three roles:

1. be no more than 0.01 nat worse than an independent map at the same rank;
2. beat the strongest independent rank allocation at exactly the same storage by at
   least 0.01 nat.

No rank passed both.

| Rank | Global minus same-rank independent CE | Global minus equal-storage independent CE | Decision |
|---:|---:|---:|---|
| 64 | +0.0616 / +0.0703 / +0.0657 | -0.0230 / -0.0242 / -0.0220 | equal-storage win, rank-matched loss |
| 128 | +0.0378 / +0.0461 / +0.0396 | -0.0363 / -0.0353 / -0.0283 | equal-storage win, rank-matched loss |
| 256 | +0.0367 / +0.0476 / +0.0364 | -0.0082 / -0.0023 / -0.0008 | neither registered margin |
| 512 | +0.0239 / +0.0273 / +0.0228 | +0.0121 / +0.0142 / +0.0136 | loses both comparisons |

The three numbers in a cell are `skip7000 / skip11000 / skip1200`.  Negative means the
shared program has lower CE and is better.

At rank 512, sharing cuts map storage from 42,467,328 to 21,823,488 float32 values,
a 48.61% map reduction.  Because the exact covered-token tables are common and large,
full program storage falls from 267,204,096 to 246,560,256 floats, a 7.73% reduction.
That saving costs about 0.023--0.027 nat against the same-rank independent program and
0.012--0.014 nat against the best independent program at the same storage.

### E2.2: attention versus MLP dictionaries

At exactly equal map storage, typed rank 481 was compared with global rank 494.  Typed
minus global CE was

$$
-0.00250,\quad -0.00237,\quad -0.00004\ \text{nat},
$$

well below the registered 0.01-nat improvement.  E2.2 fails.

Typing is more useful at very small equal rank: typed rank 64 improves over global rank
64 by 0.0135--0.0155 nat while storing 2.70% more map floats.  The advantage mostly
vanishes by ranks 128--512.  Thus attention/MLP type is a weak coarse partition, not a
sufficient canonical decomposition.

## Covered-control bookkeeping correction

The immutable result reports `covered_identity_control=false` with a spread of 0.07042.
That calculation accidentally took the maximum over both arms **and document roles**;
the three roles naturally have different CE.  The intended control is the arm spread
within each fixed role.  Recomputed directly from the receipt-bound result:

| Role | maximum covered CE minus minimum covered CE across all 24 arms |
|---|---:|
| skip7000 | 0.0 |
| skip11000 | 0.0 |
| skip1200 | 0.0 |

The covered-table identity control therefore passes exactly.  This post-outcome
bookkeeping correction cannot rescue either scientific decision: no rank satisfies the
two E2.1 CE conditions, and the E2.2 CE margin independently fails.  The original result
and receipt are not modified.

## What the result says about structure

### There is a shared low-rank language, but not one universal language

At ranks 64 and 128, a common dictionary beats the strongest independent allocation at
the same number of stored floats by roughly 0.022--0.036 nat on every role.  That is
positive evidence that output directions recur across sites.  It is more than local
MSE: the advantage appears in the fully composed model's CE.

But a global basis loses to independent maps when every site receives the same rank.
At rank 512 it also loses at equal storage.  Sites therefore need important private
directions in addition to shared ones.  A single projector is an overcompression.

### Complexity is extremely uneven across sites

The exact-storage independent allocator spends fit-only rank slots very unevenly.  For
the global-rank-64 budget, site ranks range from 0 to 525 with median 1; for the
global-rank-512 budget they range from 0 to 1,076 with median 120.5.  Most capacity goes
to the earliest MLP sites, while several attention sites receive almost none.

This is a useful warning about “rank” as simplicity: a common rank per site is easy to
describe but ignores heterogeneous causal demand.  Literal stored floats and
whole-program CE are more meaningful currencies here.

### Local spectral error understates downstream sensitivity

At rank 512, the global and independent penalized residual fractions are approximately
0.08022 and 0.07767, a small local difference, yet their downstream CE differs by
0.023--0.027 nat.  As in Family F, a modest Euclidean change can have a material causal
effect after composition.  Local energy is diagnostic; it is not the success metric.

## Consequence for the project plan

E2.3, sparse rotation of one winning shared projector, is pruned because no shared arm
passed E2.1.  Rotating a losing subspace cannot restore missing private directions.

The higher-return successor is a hierarchical or DAG-like factorization:

$$
\widehat y_j=xA^{\mathrm{shared}}_jV_{\mathrm{shared}}^\top
             +xA^{\mathrm{private}}_jV_j^\top.
$$

The shared trunk captures directions reused across many sites; a fit-only allocator
gives site-specific residual ranks where downstream demand is largest.  Compare this
hybrid with the global, typed, and exact-price independent arms at identical literal
storage and whole-program CE.  This directly tests the user's proposed hierarchical/
DAG structure and is now better motivated than sparse labeling of an all-global basis.

The other high-priority branch remains the prospective MLP3 native-Down behavioral-port
test, because it asks a different question: whether a small causally selected product
program transfers and supports finite edits on fresh documents.
