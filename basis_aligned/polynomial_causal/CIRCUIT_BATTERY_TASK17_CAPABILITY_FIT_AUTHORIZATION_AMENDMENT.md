# Task 17 FIT native-capability execution authorization amendment

**Frozen prospectively:** 2026-09-04 05:25 UTC. **Status:** authorization successor awaiting final independent review;
not yet enqueued. This is a new immutable amendment. It changes no authority row, prediction, threshold, metric,
model computation, call request, evidence value, or pass/fail continuation.

## Reviewed chain being authorized

This amendment binds the complete prospective chain below:

| Object | SHA-256 or Git identity |
|---|---|
| original FIT capability preregistration | `0fea3731f59c8b9f9b1d1e898f2b4dbca65f706406b69f1b3e429e85bc621a63` |
| independent compiler approval | `0494f037748a5e781d038c9960875fbb1e1ee219711c78649246d402e8e6b5c4` |
| model-execution amendment | `f90b0b91ee5256ed6d5962300cf8a82666efc304edbc5d273d043b623388e7e4` |
| original producer VETO | `e722e50717962c3da0b63cf875a0ceda1872ed844bfdfaac23426c719fe77348` |
| publication-repair amendment | `0c4a20b751cc05c5373b3a1d0eab95164ffc70e5dbe685cc12a9dbb341ff8301` |
| publication-repair provenance correction | `14a982abbc79de99e970dea2d352952e22e70717e7e9f677ace23370f3e7685b` |
| repaired producer | `3dcf04c0f776c056f3701967a666025ed8b63cab4d7e60a868fd766b00ac98ea` |
| approved blocked adapter | `15d60e1760581228b69d214ffcebebf5231a15cd5a09d018bda4bd98bae69ca5` |
| independent publication-repair approval | `6b4c526ec69342f33d731eadc34d50b78014dedc39cac9d1a2b89df02b8077b4` |

The repair approval is committed in exact Git commit `19957cd331e0c6fdc919cd661700789280d7791f` and targets exact corrected
successor `af7393a38f724a6ce7ce10119f8b9852744c099b`. The VETO remains part of the record: it rejects the unsafe original
publisher, while the later approval closes only that defect for the repaired bytes.

## Exact and single-run authority

Subject to a final different-agent approval of the authorization-enabled adapter and the trusted runner's hash-pinned
queue protocol, this amendment authorizes exactly one managed invocation of the task-17 FIT native-capability screen. That invocation is
limited to:

- the frozen 96 FIT rows and no generated or read SELECT, TEST, or OOD rows;
- exactly 8 native forward calls and 192 explicit row-side evaluations;
- exactly 1,536 bytes of retained numeric evidence: one `float32[24]` answer-logit array and one `float32[24]`
  maximum-foil-logit array for each call;
- zero backward calls, weight updates, gradients, activations, readers, writers, heads, MLPs, subspaces, or
  localization outputs; and
- the already frozen create-only result/evidence/receipt namespaces, with evidence then result then receipt-last
  publication by `renameat2(RENAME_NOREPLACE)`.

No second execution, direct producer invocation, alternate adapter, changed artifact, changed call prefix, additional
array, retry after a complete receipt, or unreviewed queue record is authorized. An instrument/runtime/checkpoint/canary/
namespace/call/array/price failure produces no scientific terminal and grants no retry automatically; any proposed
retry requires an audit of the failure and fresh explicit authority.

## Scientific terminal and continuation remain unchanged

If the frozen capability bars pass, the only licensed continuation is to write and freeze a separate FIT-only
localization preregistration. If any capability bar fails, the valid terminal remains `hard_abort`, all scientific
projection fields remain null, and no localization namespace may be created. Neither terminal opens SELECT, TEST, or
OOD. Capability success does not identify a component or circuit, and capability failure is not permission to change
the frozen bars or data.

## Final review and hash-pinned execution dependency

This amendment licenses construction of an authorization-enabled adapter successor; it is not an enqueue receipt. The
successor must bind this amendment and the repair review, retain the exact reviewed producer and scientific closure,
and receive a final independent review at its exact SHA-256. Before enqueue, the runner and enqueue helper must have a
separately reviewed backward-compatible hash-pinned protocol. The queue record for this run must be exactly the final
reviewed adapter SHA-256, a tab, and its absolute path. At dequeue, the trusted runner must safely capture that path,
require the captured bytes to have the queued SHA-256, and compile and execute those captured bytes without reopening
the mutable path.

Only after the final review binds the adapter hash, current bytes are rechecked, and that runner protocol is itself
reviewed and live may the exact adapter path and digest be passed once through `ops/enqueue.sh`. Until then, no queue edit, enqueue, model import,
checkpoint read, CUDA use, model forward, result, evidence, receipt, localization namespace, or outcome access is
authorized.
