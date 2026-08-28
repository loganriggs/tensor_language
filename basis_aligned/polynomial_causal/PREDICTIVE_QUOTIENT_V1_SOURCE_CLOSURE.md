# Predictive quotient v1: source-closure design

Date: 2026-08-28

Status: reviewed design boundary, not execution authority. No model or role was loaded.

## Compatibility conclusion

Do not add a second raw-model forward, expose a `StudentStep`, reinterpret the existing
teacher ledger, or let a numerical launcher receive logits/codes/VJPs. The least
invasive compatible path is to extend the existing observed suffix transaction at its
already sealed teacher-result consumption boundary:

1. configure the canonical selected P/P/N program under a quotient-only run context;
2. run `ObservedBilin18Adapter.run_student` with autograd enabled;
3. run the route's existing coordinate or O/O/N teacher exactly as licensed, preserving
   the current `TraceIdentity`, native-call ledger, and broker completion path;
4. add `consume_predictive_quotient(...)` to the sealed teacher result, internally
   consuming the validation pair `(codes, logits)` once;
5. construct all 16 categorical targets, run all 16 VJPs on that one graph, split the
   primary and replication banks, and call `summarize_fisher_batch` twice;
6. clear teacher tensors, codes, logits, targets, VJPs, and graph aliases before
   returning the two `FisherBatchSummary` objects plus the ordinary closed ledgers.

The teacher is not part of $O$; it is still produced because this preserves the proven
student/teacher lifecycle and makes the quotient incapable of opening a cheaper side
door around validation authority. The extra forward is an explicit compute cost.

## Required interface leaf

The quotient derivative is with respect to the physical MLP0 code *after its producer*
and before every downstream consumer. It must not depend on whether restored program
parameters happen to have `requires_grad=True`.

Add a quotient-only hook mode which, at MLP0, replaces the numerically identical code
with

```python
interface_code = predicted.detach().requires_grad_(True)
```

before both the physical projected write and any T-route parent read. Capture this exact
leaf as student code 0. MLP1 and all later blocks must consume the leaf-bearing physical
write, so `autograd.grad(logits, interface_code)` measures every direct and indirect
suffix path. The mode must be bound into a separate quotient run-context hash and is
illegal for fit, ordinary validation selection, mapped controls, or final scoring.

Without this explicit leaf, graph availability would accidentally depend on trainable
producer parameters. Detaching after the forward is too late and produces a disconnected
interface.

## Exact internal and returned objects

Internal only:

- MLP0 interface leaf `[4,256,64]`, float32;
- student logits `[4,256,50304]`, float32;
- target IDs `[16,4,192]`, int64;
- VJPs `[16,4,192,64]`, reduced in float64;
- the route's ordinary teacher tensor until closure.

Caller-visible:

- two `FisherBatchSummary` objects, each containing only 64-by-64 outer sums,
  row/assigned-position outer sums, counts, hashes, assignments, and scalar identities;
- the existing student, teacher, and observed tensor-free closure ledgers.

No API returns the interface leaf, program states, logits, targets, VJPs, dispatcher,
gateway, teacher tensor, or autograd handle.

## Fail-closed requirements

The future implementation must test all of the following before any role opens:

1. quotient mode without exact validation phase/selected program/support fails;
2. full code/logit graph shapes are exactly `[4,256,64]` and `[4,256,50304]`;
3. a post-hoc `[4,192,64]` code slice is rejected as the interface;
4. the leaf is numerically identical to the selected program code and is consumed by
   both the physical MLP0 write and T parent when applicable;
5. every scored suffix logit is connected to the leaf where graph causality permits;
6. program/model/native parameters receive no gradients or `.grad` mutation;
7. primary and replication summaries share one source identity and assignments but
   have the exact distinct frozen probe seeds and target hashes;
8. missing, duplicate, reordered, replayed, or cross-support summaries fail collector
   closure;
9. original-call, outer-forward, dispatcher, hook-restoration, and broker ledgers close
   exactly as the corresponding ordinary validation route;
10. success and every injected failure revoke all aliases and leave model, ship,
    program, hook, coordinator, and broker inert.

## Present blockers and next implementation unit

The canonical suffix validation execution/publisher and final source pair are still not
complete, so the selected program and legal quotient run context do not yet exist. This
is a lifecycle dependency, not a GPU, `rspd`, FineWeb, or mathematical blocker.

The fake-model interface unit is now implemented in
`predictive_quotient_v1_interface_proof.py`. It proves exact post-producer numerical
identity, physical-write and parent-read connectivity, suffix connectivity, producer
disconnection, protected-parameter immutability, one-use consumption, and alias
revocation on success and injected failure. Its five tests pass as part of the 21-test
combined quotient suite. This is a graph-boundary proof, not a production consumer.

The next implementation unit is the actual adapter hook mode and sealed teacher-result
consumer, but it remains illegal to construct until the selected canonical program and
quotient-only run context exist. Only after that source and its tests are committed may
a production launcher be written.
