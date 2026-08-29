# Terminal copy fit-role head means v1 preregistration

Status: **outcome-blind CPU contract only; collection remains launch NO-GO**.

## Purpose

The E4 copy screen replaces a selected physical head write by its fit-role mean at the
same query position. This artifact freezes how those means are computed. It does not
authorize a checkpoint load, model forward, selection role, or behavioral claim.

## Frozen population and weighting

- Input role: exactly the ordered 192 `fit_natural` documents published by
  `terminal_copy_induction_v2_rows_receipt.json`.
- Each document contributes one 256-token model input and has equal weight at every
  position. No token label, copy cell, loss, logit, selection statistic, final row, or
  OOD row is read while fitting means.
- The ordered document IDs and exact row-receipt hash must be bound before checkpoint
  access. A duplicate, omission, reorder, or different batch traversal aborts.

## Physical object

On the native fit execution, collect the source-owned physical head writes for only

`L5H5, L7H3, L8H3, L8H4, L13H0, L14H7`.

For document $d$, position $p$, and head $h$, define

$$
\mu_h(p)=\frac{1}{192}\sum_{d=1}^{192}w_{d,h}(p).
$$

The state passed to every adapter must be the native normalized attention input for
that layer and the value bus must be the native layer-0 bus. All MLPs remain native.
The adapter must rebind the checkpoint check and report both the exact unpartitioned
native write and the known separately accumulated bfloat16 head-sum discrepancy.

## Deterministic reduction

The accumulator consumes documents in receipt order independently at each of layers
5, 7, 8, 13, and 14. It transfers each selected source-bfloat16 write to CPU float64
and performs one addition per document in that order. Consequently the serialized
float64 master and its published float32 runtime cast are invariant to GPU batch
boundaries. This is a float64 accumulation of bfloat16 physical writes, not an exact
real-valued native mean. The bank is sparse: it contains exactly one head at layers 5,
7, 13, and 14 and exactly heads 3 and 4 at layer 8. There are no unnamed-head slots.

For the L8 pair, physical removal uses one atomic bfloat16 `select((3,4))` call. The
replacement first sums the two separately accumulated published float32 means in head
order `(3,4)`, casts the sum to the native dtype, and evaluates exactly
`(native_full - selected_pair) + mean_sum`; reassociation is not permitted.

The receipt must bind:

- source and checkpoint hashes;
- row-receipt and ordered-document hashes;
- exact named layer/head bank;
- per-layer document and adapter-decomposition counts;
- accumulator and published dtypes;
- sparse master/runtime mean tensor shapes and separate raw hashes;
- native full-write/value-bus replay and head-sum discrepancy bounds;
- zero model losses/logits/labels read; and
- create-only authority, result, manifest, and receipt-last publication.

## Claim boundary

A valid mean receipt closes only the fit-ablation parameter prerequisite. It is not an
E4.1--E4.3 outcome, does not select a candidate, and authorizes no final/OOD access.
Selection may begin only after a separate production scorer/source-closure audit.
