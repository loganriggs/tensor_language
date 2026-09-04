# Task 21 FIT native-capability execution authorization amendment

**Frozen prospectively:** 2026-09-04 06:55 UTC. **Status:** authorization successor awaiting final independent
review; not enqueued. This new immutable amendment changes no authority row, prediction, threshold, metric, model
computation, call request, evidence value, output namespace, or pass/fail continuation.

## Exact reviewed chain

This amendment is based only on the exact approved task-21 authority/compiler and blocked producer build. The Git
objects and file digests below are normative:

| Object | Exact Git identity or SHA-256 |
|---|---|
| authority/compiler build commit | `9ebab94615eade27b1eb63e4f2c6239337b71dc9` |
| authority/compiler independent approval commit | `ca088ce0906160958a2586cff50b707699b7eb88` |
| task-21 authority/compiler review | `3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50` |
| task authority source | `bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2` |
| complete authority logical digest | `191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b` |
| FIT authority file | `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94` |
| FIT authority-record digest | `c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8` |
| capability preregistration | `da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3` |
| capability compiler source | `43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390` |
| compiled capability contract | `5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2` |
| call manifest | `ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e` |
| metric manifest | `e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf` |
| blocked producer build commit | `000a113eed35c7e8fac0d2ceed126925963cd0d7` |
| producer implementation preregistration | `3009aff99543e34e8a7d33a486035e5168c136f168c18ebb8e3fd8a3ad290882` |
| model-facing producer | `395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf` |
| approved blocked adapter | `f7721d1b484ec7a9891dc72fc22618d403330c65092ebbb5d6d8fac68b31eced` |
| producer-review commit | `6b8fe576594bb82a5a2093f2338603040739c9af` |
| producer-review document | `8763602a753345a19312613160d32b3ffe537a7ebfcb4bcf4c83905a25b7ed29` |

The independent producer review approved only exact build `000a113eed35c7e8fac0d2ceed126925963cd0d7`
as input to this separate prospective authorization. The authorization-enabled adapter must capture and hash-check
this amendment and the exact review document in addition to retaining the reviewed producer and complete scientific
closure. The successor has a new digest and is not covered by the earlier review.

## Exact one-run authority

Subject to final different-agent approval of the exact authorization-enabled adapter and use of the separately
reviewed hash-bound managed queue, this amendment authorizes exactly one managed invocation of the task-21
`verbatim_repeat.copy` FIT native-capability screen. The invocation is limited to:

- the frozen FIT authority only, with no generated or read SELECT, TEST, or OOD authority;
- exactly eight native forward calls, four base-side then four donor-side;
- exactly 168 explicit row-side evaluations: 21 rows in every call, including repeated prompt evaluations required
  by the compiled contract;
- exactly 1,344 raw numeric evidence bytes: one C-contiguous `float32[21]` answer-logit array and one C-contiguous
  `float32[21]` maximum-foil-logit array for each call;
- no full logits, hidden states, activations, gradients, backward calls, parameter updates, readers, writers, heads,
  MLPs, subspaces, localization labels, or later-phase artifacts; and
- only the already frozen task-21 result, evidence-directory, and receipt namespaces, installed create-only as
  evidence then result then receipt last with `renameat2(RENAME_NOREPLACE)`.

No direct producer invocation, alternate adapter, changed artifact, changed call order or prefix, extra array,
second complete execution, or unreviewed queue record is authorized. Runtime, checkpoint, canary, namespace, call,
array, or price failure produces no scientific terminal and grants no automatic retry. Any retry proposal requires
a failure audit and fresh explicit authorization.

## Scientific terminal and continuation

The frozen opposing outcomes are unchanged. If every frozen capability bar passes, the sole permitted scientific
continuation is a new, separately frozen FIT-only localization preregistration. If a capability bar fails, the valid
terminal is `hard_abort`, every scientific projection field is null, and no localization namespace may be created.
Neither outcome opens SELECT, TEST, or OOD. Passing capability does not identify a model component or circuit, and a
failure does not permit changing the authority, model, bars, or metrics.

## Final review and execution dependency

This amendment licenses construction of one authorization-enabled adapter successor; it is not an enqueue receipt.
That adapter may set `EXECUTION_AUTHORIZED=True` only while it exactly captures this amendment, the producer review,
and every reviewed source and authority role. It must receive final independent approval at its exact SHA-256 before
execution. The managed queue record must bind that final reviewed digest, and the trusted runner must safely capture,
verify, compile, and execute those exact bytes without reopening a mutable path.

Until that final review and separately approved runner condition are satisfied, no queue edit, enqueue, model import,
checkpoint read, CUDA use, model forward, result, evidence, receipt, localization namespace, or task-21 outcome access
is authorized.
