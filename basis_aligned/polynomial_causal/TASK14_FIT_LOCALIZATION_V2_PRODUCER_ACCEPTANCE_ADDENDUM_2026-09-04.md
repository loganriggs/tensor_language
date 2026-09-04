# Addendum to the task-14 FIT-localization-v2 producer acceptance contract

**Frozen:** 2026-09-04 11:14 UTC, before producer construction and before the compiler candidate has passed independent review.

**Parent contract:** `TASK14_FIT_LOCALIZATION_V2_PRODUCER_ACCEPTANCE_2026-09-04.md`, commit `ecb37c0ab`, SHA-256 `1724fa6de7ece875cd633976841159302e04033ca008af6e6437ee159a935b46`.

An independent source-only red-team found two contradictions and nine missing executable requirements in the parent contract. This addendum corrects them prospectively. Where the two documents differ, this addendum controls. It is not compiler approval, implementation authority, model/checkpoint/GPU access, memory- or throughput-canary authority, result publication, queue/enqueue, or execution authorization.

## 1. Conditional call-index replay, not global contiguous consumption

The binary call index contains every mutually exclusive conditional template. One physical execution intentionally does **not** consume every entry contiguously.

The producer must:

1. capture and hash-check the complete compiler, manifest, and binary index;
2. deterministically regenerate and validate the complete global index before any model access;
3. evaluate only the frozen DAG predicates available at a stage;
4. for every activated chunk, seek to its frozen `call_index_offset`, validate the complete frozen slice and slice hash, and consume that chunk's call IDs in order;
5. record inactive conditional chunks explicitly as inactive, without treating their skipped global entries as missing calls; and
6. require the activated-chunk sequence and root to match the frozen terminal path.

Within an active chunk, deletion, reordering, duplication, or an incomplete slice is an instrument hard abort. Across chunks, a branch inconsistent with the frozen DAG is an instrument hard abort. The parent contract's phrases “next global entry,” “calls may not be skipped,” and “partial call-index completion” refer only to the next entry in each **activated chunk slice** and to completion of the selected conditional path, not execution of inactive templates.

## 2. Reconstruct physical batch contents before every call

The compiler's public call descriptor retains item IDs, roles, uses, positions, boundaries, and `batch_binding_sha256`; it does not retain literal token arrays or expanded target/donor coordinate tables. Immediately before device transfer or model execution, the producer must re-resolve from the captured frozen authority:

- the ordered CPU token rows;
- target and donor endpoint IDs and sides;
- target and donor H/Q token coordinates independently;
- prediction positions and answer/foil token IDs;
- optimizer slot and normalizer roles, including `A_C` endpoints; and
- the equal-sequence-length batch partition.

It must recompute `batch_binding_sha256`, require equality with the descriptor, then recompute the call ID and require equality with the active index entry. A hash match on item IDs alone is insufficient. Cross-syntax tests must change a literal donor coordinate and prove rejection.

## 3. Exact carried `x0` and block-0 `v1` state

The physical model carries two auxiliary states in addition to the residual: initial normalized embedding state `x0`, and the block-0 value-side state `v1` reused by later attention blocks.

The previously underdefined physical choice is now explicit and must also be accepted by the repaired compiler review:

- At boundary `-1`, edit the normalized embedding residual **before** `x0` is established; the edited trajectory establishes its own live `x0` and block 0 derives live `v1` from that trajectory.
- At boundaries `0..17`, preserve the target prefix's live `x0` and `v1`; edit only the declared residual coordinate.
- In a two-site or ordered-reader arm, the later site receives the complete trajectory produced by the earlier edit, including any resulting change to live `x0` or `v1`. The producer never substitutes donor `x0` or donor `v1` as an undeclared extra intervention.

Native-boundary and self-donor identity tests are insufficient. Tiny models must make logits separately sensitive to `x0` and `v1`, exercise a non-self donor at boundary `-1`, and verify the rules above through at least one later attention site.

## 4. Graph retention until the single backward

All graph-carrying forward batches in one logical optimizer step contribute to one final backward. Every differentiable per-batch scalar—including slot terms and current-projector full-DISCOVERY normalizers—must remain device-resident and graph-connected, be combined without `.item()`, NumPy conversion, CPU transfer, or detach, and receive exactly one backward after the final graph batch.

The parent contract's “offload retained evidence after each call” excludes live graph scalars and intermediates. Those may be released or offloaded only after that logical step's backward and ledger finalization. A targeted test must make only the first graph batch contribute a nonzero gradient and prove that contribution reaches the optimizer update.

## 5. A task-specific result and evidence validator is mandatory

The generic `result_contract.py` does not define task-14-v2's nine scientific terminals or exact evidence package. Producer construction must therefore include a frozen task-specific schema and independent validator specifying:

- every result field, dtype, finite/infinite representation, ordering, and hash;
- every terminal's required and null fields under the exact frozen precedence;
- every active and inactive chunk receipt and the activated-path root;
- every evidence filename, file order, array shape/dtype/byte count, and package hash;
- the distinction between publishable completed scientific terminals and malformed-runtime/optimizer/hash/deadline/OOM/publication failures that leave no completed package; and
- an explicit finite JSON representation for the preregistered positive-infinity leakage sentinel.

The final task-specific validator must reconstruct the terminal from primitive evidence. Neither a producer-supplied terminal string nor the generic envelope validator is sufficient. A valid finite scientific miss may publish its complete all-null later-stage package; deadline, OOM, incomplete call-index, malformed instrument, or publication failure may not masquerade as that scientific terminal. Publication failure necessarily leaves no completed receipt.

## 6. Complete runtime gates and asynchronous CUDA timing

The implementation must enforce all compiler runtime settings: TF32 disabled, deterministic algorithms enabled, cuDNN benchmarking disabled, and exact `CUBLAS_WORKSPACE_CONFIG=:4096:8`. It must verify canary 1 and canary 2 before and after **each exact DAG major stage**, rehash the checkpoint before load, after device copy, after each major stage, and before publication, and reset/read CUDA peak counters around the same named stages.

The external watchdog starts before real runtime bootstrap and can terminate a hung model call even when the Python thread does not return. The monotonic eight-hour deadline and p99 remaining-time test use CUDA synchronization immediately before starting and immediately after completing each timed model call; otherwise queued asynchronous kernels would escape the measurement. Failed-work accounting occurs in `finally`: if a forward completes and later validation raises, forward/graph/example/token counts still increase, while a backward or optimizer update is charged only if physically completed.

Tests must inject a completed forward followed by validation failure and check the exact ledger. Stage labels may only be the compiler DAG's named major stages; the producer may not invent coarser groupings that reduce canary, hash, memory, or deadline checks.

## 7. Preauthorization peak and throughput receipts

The worst-shape memory canary and p99 throughput measurement are two separate prospective managed artifacts. Neither may inspect task authority or outcomes. Each requires its own preregistration, exact producer/model/checkpoint/runtime/device binding, independent review, hash-bound managed enqueue, create-only receipt, and no automatic retry.

Both exact receipts and their independent review digests must predate, be captured by, and be verified by the later authorization successor and its final reviewer. The successor may authorize exactly one task invocation on the same device/runtime identity within the reviewed free-memory and remaining-time rules. A canary result cannot be pasted into an already authorized adapter.

## 8. Exact reserved namespace and pre-runtime freshness

“Run-unique” in the parent contract means the exact reserved one-invocation namespace
`task14_fit_localization_v2_fit_v1`; it does not permit choosing a random or outcome-dependent path. Before runtime bootstrap, model import, or checkpoint access, `lstat` must prove the exact result, evidence, and receipt destinations absent, treating every file, directory, valid symlink, and dangling symlink as occupied.

Publication retains the parent contract's create-only evidence→result→receipt-last order. A publication failure is an incomplete instrument run with no completed package, not a publishable `instrument_invalid` scientific record.

## 9. Closure and deterministic serialization

The producer and both adapters must capture this addendum and the parent acceptance object by exact commit/path/SHA in addition to the compiler/review chain. The later authorization successor must also capture the exact producer review, peak-canary preregistration/review/receipt, throughput preregistration/review/receipt, and prospective authorization amendment.

Under at least `PYTHONHASHSEED=0,1,999`, the call schedule, dry-run serialization, all tie-broken selections, result/evidence serialization, and receipts must be byte-identical. This includes branches built from sets or dictionaries. The final approval binds one exact adapter digest and one exact reviewed managed enqueue-helper digest for one invocation without retry.

## 10. Effect on the active compiler review

Two items require compiler repair or explicit approval before producer construction:

1. replace the ambiguous global “next call-index entry” wording with whole-index preflight plus activated per-chunk slice replay; and
2. bind the boundary-`-1`/`x0`/`v1` physical semantics in the compiler's model contract.

The compiler review has independently found additional coherent-mutation validation holes. All compiler blockers must be repaired and receive a fresh exact-commit review. No producer construction begins from the currently vetoed `ea16e22d` candidate.
