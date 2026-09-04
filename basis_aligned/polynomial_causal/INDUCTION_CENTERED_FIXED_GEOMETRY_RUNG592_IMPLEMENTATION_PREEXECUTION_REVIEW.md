# Independent pre-execution review: R592 centered fixed-geometry implementation

**Reviewed:** 2026-09-04 UTC, before any R592 model execution or outcome

**Candidate commit:** `0bd259b7d5a499a863741338f8b55dc11368f344`

**Review mode:** immutable Git blobs, CPU/model-free only

**Verdict:** **BLOCKED — do not enqueue or execute these bytes**

## Exact candidate

| artifact | SHA-256 |
|---|---|
| producer `induction_centered_fixed_geometry_rung592.py` | `c52e1225c128de98b01d33649eb4227ff99e63177a8cbd85b9fd0556b4bf5aee` |
| model runtime `induction_centered_fixed_geometry_rung592_runtime.py` | `df2d59245dc5bd407c96af0a8a6d1c98a70ae25f1925c4540dbd47bb956254a1` |
| managed adapter `execute_induction_centered_fixed_geometry_rung592.py` | `a104a53411a68527f2702ff9999a9045925ad47cffe56ee6f9966a4eb1e65531` |
| owner test | `85f73a6b35f4e9960320bf23996ebc595d02dcd5a76f34ceaef51a6d502c7d54` |
| fake-runtime test | `f5ea5e005991d57f6d23b5df44d1eccb500ec59469c20f70745ce37f1f6980c0` |
| adapter test | `203225f98635680b723f56527a8325d8d2d56e84d6b552008cd3fa3d18cf4dfd` |
| dry run | `a2c6e760b9b87d70b5a444a11d5bd9f76b0090330fc31e67bdee710aa31e517d` |

The adapter correctly pins the preregistration, executable amendment, diagnostic-prefix amendment, nonfinite-mask
amendment, and their independent reviews. The producer also reaches the R585/R591 transitive executable closure only
through hash-verified snapshots. The centered-factor derivation, v7 claim boundary, and earlier independent reviews
match the hashes named by the frozen preregistration lineage.

## What is correct in the intended computation

The model-free manifest reconstructs 1,728 FIT endpoints, 3,744 FIT directions, 54 endpoint calls, and 117 directed
chunks. Each directed chunk has exactly `native`, `replay`, `score`, `payload`, and `joint`, for 639 FIT calls. SELECT
has 864 endpoints, 1,872 directions, 27 endpoint calls, and 59 directed chunks, for 322 calls; its last five calls use
the same 16-row, width-30 token tensor. Maximum registered price is therefore 961 forwards, zero backwards, and zero
updates.

The intervention formula itself is the frozen centered formula:

$$
\Delta_{\mathrm{replay}}=0,\quad
\Delta_{\mathrm{score}}=B(E_y,U_x)-B(E_x,U_x),\quad
\Delta_{\mathrm{payload}}=B(E_x,U_y)-B(E_x,U_x),\quad
\Delta_{\mathrm{joint}}=B(E_y,U_y)-B(E_x,U_x).
$$

All factors come from endpoint captures made before directed arms. The runtime adds the selected frozen change to the
untouched live attention write; it never subtracts a live equality term. L8H3 and L8H4 are summed and added in one
layer-8 transaction. The directed native observer checks recipient `e`, `u`, and the registered recipient-containing
hybrids against the cache. Token bytes are rehashed before each call. FIT scores are completed before the condition
that may open SELECT; FINAL and OOD remain closed. Normal and invalid publication rename evidence first, JSON second,
and receipt last.

These properties are useful but do not make the exact candidate executable or auditable.

## Blocking findings

### 1. The real output width is incompatible with every frozen R592 logit array

R592 freezes `VOCAB = 50_257` and allocates every returned logit array as `[batch, 50257]`. Its pinned observed-model
facade freezes `LOGIT_VOCAB = 50_304`; because the checkpoint config also has vocabulary size 50,304,
`forward_with_dispatch(..., require_production=False)` returns `[batch, width, 50304]`. R585 likewise uses 50,304.

Both runtime paths assign that full returned vector into a 50,257-element row. The first endpoint forward can finish,
but the host assignment then raises a shape error. The producer treats that call as incomplete and hard-aborts. Thus,
even if the managed gate were bypassed, the exact candidate could make one model call but could not finish the first
manifest item or publish evidence.

This is also a frozen specification error, not merely a local allocation typo: the amendment explicitly registers
`[N_d,4,50257]` logit differences and the 3,010,595,328 / 1,505,297,664-byte prices. Full model vocabulary at 50,304
instead requires 3,013,410,816 / 1,506,705,408 data bytes. Repair requires a prospective amendment choosing the actual
50,304 full-logit vocabulary (recommended), then updating shapes, byte prices, producer/runtime constants, dry run,
tests, and all inherited vocabulary-size calculations. Slicing to tokenizer vocabulary would be a different scientific
object and also requires an explicit amendment.

### 2. Neither candidate entry point passes the managed repository gate

The exact adapter and producer both fail `ops/gate.py` because they expose zero literal `pred_*` keys. The producer is
also rejected because `evaluate_completed_call` uses `error` as an exception-handler alias and later reuses `error` as
an ordinary local name. The managed adapter therefore cannot be accepted by `enqueue.sh`; the executable call count is
zero under the authorized workflow.

Repair both static failures without weakening the gate: expose the registered predicates in the required static form
and use unique exception/local names.

### 3. Missing mandatory observations are incorrectly published as a completed call

The diagnostic-prefix amendment says a call is completed only after every required observation is materialized. A
missing required observation must be an unpublishable hard abort. Instead, `run_manifest_calls` accepts any returned
array mapping, and `evaluate_completed_call` catches every missing/extra/wrong-shaped array under one `ValueError` and
adds `fixed_width_token_manifest_failed`. It then writes the incomplete mapping as a completed call and publishes an
invalid diagnostic.

A planted score response missing `hook_deltas.npy` reproduces this misclassification. Repair by validating completion
and the full mandatory evidence contract outside the publishable predicate path. Only a genuinely complete response
may be evaluated for the ordered runtime predicates.

### 4. The invalid receipt does not content-bind all recognized evidence

The normal receipt hashes every evidence file. The invalid receipt does not: it hashes `call_prefix.jsonl`, but only
lists the other evidence-tree paths. Per-call raw arrays happen to have hashes inside the prefix, while
`nonfinite_mask_index.json` and the actual files under `nonfinite_masks/` have no hash anchored by the receipt. Changing
the index after publication does not change any receipt field; the index is what supplies the claimed mask hashes.

Repair the invalid receipt so it includes path, byte length, and SHA-256 for every evidence file, including the mask
index and every mask. Validate exact set equality before receipt publication.

### 5. Complete evidence cannot reconstruct the retained native-attention hard gate

The retained gate compares an independently reconstructed full attention write with the model's native attention write
and checks a transient `native_full_write_reconstruction_max_abs`. That scalar exists only in the executor response
metadata used immediately after a call. It is absent from complete phase evidence and the normal result.

Moreover, the runtime fills both `native_head_write.npy` and the purported
`independent_full_native_write.npy` from the exact same `term["head_output"]` expression. These duplicate bytes neither
provide an independent computation nor preserve the original native full-attention write. Consequently an auditor
cannot reconstruct the gate that authorized a complete scientific terminal.

Repair by saving the actual native attention write and independently reconstructed full write at the exact registered
coordinates (or another sufficient raw representation), giving them unambiguous names, and deriving the gate from
those receipt-bound arrays. Do not substitute duplicate selected-head vectors or trust a transient scalar.

### 6. The two large memmap evidence files are not explicitly fsynced

Ordinary `.npy` and JSON files call `os.fsync`. `hook_deltas.npy` and `logit_differences.npy` use `open_memmap`, call
`.flush()`, delete the mappings, and are then hashed, but their file descriptors are never explicitly fsynced before
rename. Fsyncing only the containing directory does not satisfy the frozen requirement that evidence files be hashed
and fsynced before receipt-last publication.

Repair by explicitly fsyncing both closed memmap files after their final flush and before hashing/publication. Also
check that the completed offset equals the frozen direction count and that the derived arrays are finite before they
can enter normal evidence.

## Tests and execution decision

The candidate's three focused suites pass `27 passed`. The independent test packet passes four positive invariants and
records seven strict expected failures covering both gate entry points and the five contract failures above:

```text
4 passed, 7 xfailed
```

The packet additionally exercises hook mismatch, nonfinite output, and raised/incomplete calls at each of replay,
score, payload, and joint; checks no publication after a raised call; statically verifies exactly two runtime facade
call sites, frozen-only centered additions, one L8 transaction, FIT-first opening, and receipt-last rename order.

`ops/preflight.py` reports no advisory findings, but the authoritative static gate fails as described. The live
model-free adapter dry run matches the committed artifact and reports zero model calls. No model, Torch, checkpoint,
CUDA, GPU, queue, SELECT, FINAL, OOD, or R592 outcome was opened during this review.

**Decision:** the centered scientific design may remain the basis of a repaired successor, but commit
`0bd259b7d5a499a863741338f8b55dc11368f344` is not execution authority. A prospective spec amendment for the output
width plus code repairs for all six findings and a fresh different-agent exact-byte review are required before enqueue.
