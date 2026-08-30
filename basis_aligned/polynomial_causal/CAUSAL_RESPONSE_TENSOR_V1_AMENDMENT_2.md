# Causal-response tensor v1 — prospective amendment 2

Frozen after the outcome-blind audits of commits `e65c915e` and `fa909d35`, and
before any v1 bilin18 model load, forward, fitted direction, response cell, or EVAL
outcome. Amendment 1 remains controlling except where this document makes the
input, numerical, ledger, and publication interfaces more exact.

## Why a second amendment is necessary

The first two-stage amendment correctly separated FIT response discovery from EVAL,
but its first backend draft left four ambiguities that an independent audit rejected:

1. it cast each full direction to float32 before the component SVD and residual;
2. it accepted caller-supplied rows/specs instead of reconstructing them from parents;
3. it aggregated projection calls only by component rather than preserving every
   phase/source/batch event;
4. it could mint an EVAL-capable direction object when the FIT bundle was published,
   before a receipt existed.

No scientific v1 outcome was opened. The following prospective repair controls.

## Exact canonical FIT input

The model-input tensor is the first 257 columns of each frozen 513-token curated row.
It therefore contains 256 next-token prediction positions, exactly matching every
census mask. Columns 257 through 512 are outside the registered census and must not be
passed to this collector. Canonical reconstruction from the four pinned parents gives:

| object | exact SHA-256 |
|---|---|
| model rows, CPU int64 `[1000,257]` | `1786a30bc0d27d26324486e582a539cc292428c2f3f4f1ed7594014390a437ce` |
| sorted FIT row indices, CPU int64 `[496]` | `6873c2a279bf73fe17c38d72ac25003f4741825efc271ff91b6b783615cdd815` |
| sorted FIT source-document IDs, CPU int64 `[343]` | `0f514805a7615e5ef3fe862eb8bf37bebfe8c57b8b7e781fbb25907c729b808d` |
| ordered `(component,tag)` serialization | `86d0bd7250102fc8dcdee517562fcadda74f2f6bf6d026582bcab71a33f24ca0` |
| logical map of all 49 member/slice mask hashes | `a8e033d981e82b5e39404ed5ee705119897e1d5d5a1cceaf80ea12c0b711a5aa` |

The FIT role is exactly 496 sorted row indices from 343 source documents. The 49
specifications are ordered by component `a8,a16,m16,a3,m14,m13`, then lexicographic
tag. A production execution receives no caller-supplied path, role, specification, or
mask. It reconstructs all of them after authority publication and exact-equals the
five hashes above.

## Exact numerical repair

Member/off sums, means, contrast norms, normalized full directions, component SVDs,
shared directions, residual projections, and residual norms are computed on CPU in
float64. The full and residual deploy directions are each cast exactly once to
float32, after all relevant normalization and SVD arithmetic.

The full contrast failure boundary remains exactly zero or nonfinite; there is no
positive epsilon cutoff. The residual failure boundary remains norm at most
`1e-6`. The relative leading-singular-value gap failure boundary remains at most
`1e-6`. Signs are fixed before the deployment cast.

## Exact compact event ledger

The FIT artifact stores an int64 projection event-count tensor with axes
`[2 phases,49 ordered sources,124 batches]` and an int64 capture event-count tensor
with axes `[6 ordered components,124 batches]`. Every entry must equal one. It also
stores the explicit axis labels and aggregate per-component counts. This compact
representation is equivalent to 12,152 individually keyed projection events and 744
capture events, without thousands of repeated string keys.

The native physical census is unchanged: 12,400 outer calls, with all 18 attention
and all 18 MLP sites called exactly 12,400 times.

## FIT publication exposes no EVAL program

The FIT bundle publisher may return only the SHA-256 of the exact bytes it privately
replayed. It must not return directions, document IDs, a reusable mapping, or any
EVAL-capable object. Semantic replay hashes stable bytes and deserializes those exact
bytes from memory; hashing a path before and after a separate path-based load is not
sufficient.

A fresh-process EVAL loader may expose a one-use program only after it joins and
replays all of the following exact artifacts:

1. FIT authority;
2. FIT bundle and every internal tensor digest;
3. FIT manifest;
4. completed FIT receipt, published last;
5. unchanged source, parent, checkpoint, role, order, and support bindings.

Bundle publication must accept an adjacent lifecycle guard that rechecks lock
ownership, authority, parents, protected state, and terminal absence immediately
before its create-only link. Publication failure is not a scientific result and cannot
authorize EVAL.

## Status

This amendment authorizes implementation and outcome-blind audit only. It is not FIT
execution authority. FIT remains NO-GO until the complete source-closed lifecycle and
its independent audit are committed and pushed. EVAL remains separately NO-GO until a
completed FIT receipt and a separately audited receipt-bound loader exist.
