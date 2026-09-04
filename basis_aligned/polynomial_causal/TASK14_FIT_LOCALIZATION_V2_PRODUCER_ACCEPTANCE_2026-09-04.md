# Prospective acceptance contract for the task-14 FIT-localization-v2 producer

**Frozen:** 2026-09-04 11:02 UTC, before any producer or execution adapter exists and while independent review of the physical compiler is still open.

**Compiler candidate:** exact commit `ea16e22d28d125274ca4353f46e434c2826e0b02`.

This document is a review rubric, not an approval of that compiler and not authority to build until its independent review approves it. It never authorizes model/checkpoint/CUDA access, a memory canary, task execution, publication, queueing, SELECT/TEST/OOD access, or reuse of an existing result namespace. A later producer candidate must satisfy every item below from immutable Git objects and receive a fresh different-agent review.

## 1. Immutable ancestry and phase isolation

The producer and blocked execution adapter must bind, by exact path and SHA-256:

- task-14 v2 authority commit `8f41f51cdf7e073063201cc48760622607ce91b9`;
- authority review commit `2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923` and review SHA-256 `2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384`;
- the exact approved compiler commit and every compiler artifact, including the binary ordered call index;
- the eventual independent compiler-review commit and review digest;
- the frozen model source, facade, checkpoint identity, runtime versions, canaries, result contract, and publication helper used in real mode.

Only FIT authority may be captured or opened. SELECT, TEST, OOD, earlier task-14 outcome packages, and any localization result are forbidden. Dry-run/import/test paths must be model-free, CUDA-free, outcome-free, queue-free, and publication-free. The first real adapter must remain unconditionally execution-blocked; authorization is a later separately reviewed successor.

## 2. Exact call-index replay and accounting

Before every model call, the runtime must regenerate the next complete canonical descriptor from the captured compiler and compare its SHA-256 call ID to the next 32-byte entry in the captured binary call index. A descriptor includes stage, branch, item IDs and roles, logical step, position, boundary, target and donor token coordinates, fit configuration, normalizer uses, and all `A_C` endpoint slots. Calls may not be skipped, reordered, combined across incompatible sequence lengths, silently retried, or appended after a terminal branch.

The runtime ledger must charge, for successful and failed calls alike where work occurred:

- physical forward batches;
- graph-carrying forward batches;
- autograd backward calls;
- logical optimizer updates;
- evaluated sequences;
- evaluated tokens;
- active chunk IDs and their ordered roots.

One logical optimizer step may use several graph batches, but must accumulate the registered objective and invoke `backward()` exactly once. The activated-chunk root and every count must equal the exact conditional path selected by the frozen predicates. Partial call-index completion, an exceeded bound, or a deadline stop is an instrument hard abort without a scientific terminal.

## 3. Model computation and boundary semantics

The real computation must reproduce:

$$
\text{tokens}\to W_E\to\operatorname{RMSNorm}_{in}\to
\text{18 complete native blocks}\to\operatorname{RMSNorm}_{final}\to W_U
\to30\tanh(\text{logits}/30).
$$

Boundary `-1` is the normalized embedding input before block 0. Boundary $b\in\{0,\ldots,17\}$ is the residual after complete block $b$; execution resumes at block $b+1$. Boundary 17 therefore continues only through final normalization, unembedding, and softcap. A tiny-model identity test must reproduce native logits at all 19 boundaries for both registered sequence lengths.

Every intervention forward starts from target tokens and recomputes the complete native prefix. The runtime may cache detached float32 CPU copies of the declared target/donor H and Q residual vectors, but not a full-sequence boundary state or a projected suffix. The edited target graph must remain live through the remainder of the model. Full-state, projector, necessity, two-site, reset, and rescue arms must use the compiler's formulas exactly.

Target and donor H/Q positions are independent semantic coordinates. Cross-syntax donors must not assume aligned positions. Because this model has no attention mask, a batch may contain only equal sequence lengths; padding different lengths is forbidden. Each call gathers its declared prediction coordinate rather than using an unconditional last-token index.

## 4. Differentiable projector fitting

The learned frame uses float32 parameters and differentiable QR with deterministic column signs. Initialization is the compiler's SHA-256 Rademacher construction, not Python's hash, a mutable random stream, or a post-hoc spectral initializer. The spectral calculation remains a DISCOVERY-only diagnostic with no success predicate, validation selector, or change to registered DAS initialization.

For every optimizer step:

1. zero only projector optimizer gradients;
2. construct the current orthonormal frame without `.data` mutation;
3. evaluate all compiler-declared slots and normalizers, including `A_C` endpoint slots;
4. differentiate through the current-projector full-DISCOVERY median normalizer;
5. call backward exactly once;
6. require a finite, nonzero frame gradient and no model-parameter gradient;
7. apply the exact cosine-scheduled Adam update.

Cached native tensors must remain unchanged. Projected activations may not be detached or reused between optimizer steps. All five seeds must remain healthy; no failed seed may be discarded or replaced.

## 5. Ordered causal semantics

- **Sufficiency:** replace only the selected coefficient by the donor's coefficient.
- **Single-site necessity:** set the selected coordinate to its frozen DISCOVERY mean.
- **Two-site necessity:** apply both neutralizations in ascending boundary order; the later edit acts on the state produced after the earlier edit.
- **Reader reset:** apply the upstream H donor edit, propagate it to Q, then replace the selected Q coordinate with the native target Q coordinate.
- **Reader rescue:** neutralize H, propagate that edit to Q, then insert the natural donor Q coordinate.

Only the explicit reset arm may discard the live propagated Q value. Tests must distinguish live propagation, native reset, and donor rescue. A self-donor projector edit must reproduce native logits; non-target residual positions must remain bitwise unchanged. Every intended site must fire exactly once, and injected failures must leave no installed hook or dispatcher state.

## 6. Decisions, arrays, and scientific terminals

The producer must implement the compiler's discovery retention, site selection, locked validation, rank-two/rank-four falsifier, seed, necessity, redundancy, reader, and terminal-precedence rules literally. Finite low causal ceilings and finite failed transfer are scientific outcomes, not runtime errors. Nonfinite values, malformed arrays, missing calls, changed hashes, optimizer failures, deadline failure, or publication failure are instrument invalidity.

All retained numeric arrays must have the exact declared shape/order/dtype, be finite and C-contiguous on publication, and satisfy the compiler's raw-byte formula. Scalars used for thresholds and selection are recomputed in float64 on CPU with the declared quantile and tie rules. Result validation must reconstruct every terminal predicate from retained evidence; a caller-supplied status string is never trusted.

No partial evidence package may survive an incomplete run. A valid scientific hard abort publishes the exact all-null later-stage structure required by the result contract; it does not open another phase or create retry authority.

## 7. Runtime, deadline, memory, and publication

The implementation must use a monotonic clock before and after every model call, enforce the eight-hour ceiling, and refuse to begin a call without the separately reviewed measured p99 call-time budget remaining. It has no automatic retry.

Before task execution, a separate hash-bound managed canary must measure the exact producer/model surface on the registered worst shape: batch 192, length 8, rank 4, two simultaneous forward graphs, one backward, one QR, and one Adam update. That canary itself requires preregistration, independent review, and managed enqueue. The task runtime may load the model only if free device memory is at least

$$
\max\!\left(\left\lceil1.25M_{reserved}\right\rceil,
M_{reserved}+2^{31}\right),
$$

and must hard-abort without a scientific terminal if the reviewed peak cap is exceeded or an out-of-memory error occurs. Retained evidence moves to CPU after each call.

The namespace is run-unique and create-only. Freshness checks must treat files, directories, valid symlinks, and dangling symlinks as occupied. Publication must use Linux `renameat2(RENAME_NOREPLACE)` with no weaker fallback, install evidence then result then receipt last, never overwrite a racer, and roll back only inode identities created by the current invocation.

## 8. Required adversarial tests before review

At minimum, the candidate tests must cover:

1. identity at every boundary and both sequence lengths;
2. self-donor identity and single-position isolation;
3. different target/donor H and Q coordinates across syntax;
4. equal-length batching, tail batches, and rejection of padding;
5. live finite projector gradients, no model gradients, and unchanged caches;
6. deterministic QR/Rademacher replay and repeated-eigenvalue stability;
7. all three ordered-reader states;
8. hook cleanup and exactly-once site dispatch under injected failures;
9. full manifest/call-index replay, including interior mutation, deletion, reorder, role changes, and `A_C` deletion;
10. one backward per logical step with several graph batches;
11. exact decision reconstruction and coherent threshold/terminal mutations;
12. monotonic deadline failures immediately before and after a call;
13. exact array order/shape/dtype/contiguity/finiteness/raw bytes;
14. temporary-memory canary binding and free-memory rejection;
15. dangling symlink, late destination race, staged crash, and hostile rollback-inode replacement;
16. no forbidden phase, prior result, model, checkpoint, CUDA, queue, or publication access in import, dry run, or tests.

A focused suite is not enough. The reviewer must also run the relevant task authority/compiler/framework/publication/managed-queue suite and add coherent attacks not copied from the builder's tests.

## 9. Review and execution boundary

A fresh different agent must inspect the producer from an exact commit, reconstruct the call schedule and prices independently, run the tests above without model/GPU access, and publish APPROVE or BLOCK with exact hashes. Approval licenses only construction of a separate prospective authorization successor. That successor must bind the producer review, preserve every gate, prove captured-module object identity, receive another exact-commit review, and be submitted only through the reviewed hash-bound managed enqueue helper. Direct execution is never authorized.
