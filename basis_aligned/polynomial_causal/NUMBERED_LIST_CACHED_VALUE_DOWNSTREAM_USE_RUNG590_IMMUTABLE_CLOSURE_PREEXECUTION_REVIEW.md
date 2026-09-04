# R590 immutable-closure pre-execution review

Date: 2026-09-04 UTC

Reviewed commit: `5fc3144ebe4c080473a3e780fc7519cd8cd08f8c`

Verdict: **BLOCKED**

This is an independent CPU-only review of the exact committed R590 repair. I
did not load the model, open CUDA, enqueue work, or open an R576, R579, R584, or
R590 scientific outcome. The outcome-blindness attack replaces the file-hash
function and raises before either named prior-outcome file is read.

## Exact reviewed packet

- producer: `5cc4544158312d7fa6224bf46c635acbb0d4a11fc2d620cedc2516d169f5966e`
- owner test: `49f6f7a998bfb69331c36391f5d3c16d9b702c1fd60b4da5b09c920f3832e5b0`
- dry run: `817f457ba1cc9737735182f495c54a3956be8c5dd6267bb5d8222f40e750d603`
- managed adapter: `275f1c4d72f538283daba1b417be7e33e0c1749f0c1e21a2be1d0a6143f23f57`
- adapter test: `4c5bd25cdf06e21f823c9e09fdd57a7ca54d8700aa23a379a7913e2fc8c6b174`
- prospective note: `a6641a20a456d30895a9ba807c22ec74e7695fe5c84ce4300b909787c603afa7`
- v6 handoff: `d1fdedd90ffff29e6790042b9c9a6ad84278849c3f66707cb586317832fdad1c`

The working-tree copies had these exact blob hashes during review. The producer
diff from the previously reviewed `cf00f555d` science consists only of binding
the v6 handoff and the updated prospective-note hash. The causal formulas,
rows, gates, retained evidence, terminal rules, and price did not change. The
model-free receipt still declares 510 possible calls, legal realized totals of
379/419/510, zero backwards, zero weight updates, and closed FINAL_TEST/OOD.

## What the repair gets right

1. The adapter itself imports only standard-library modules before capturing
   its frozen files.
2. Capture uses no-follow file descriptors, verifies regular files, detects a
   file changing during the read, and retains the verified bytes in memory.
3. The principal executable modules are compiled from that snapshot in an
   explicit order. Merely loading them created no R590 result, receipt, or
   evidence namespace and did not load a model.
4. The exact v6 handoff is present in both producer and adapter authorities.
5. The earlier review's positive findings about evidence reconstruction,
   finite publication, atomic package handling, balanced support, split
   closure, and 510-call shape accounting remain applicable because those
   scientific parts are byte-for-byte unchanged apart from authority metadata.

## Blocking defects

### 1. A verified dependency reopens executable code from a mutable path

The adapter compiles the verified R582 helper into the module name
`numbered_list_cached_value_downstream_use_rung582`. It then compiles R588 from
verified bytes. But R588's `load_r582_helper()` ignores the verified module:
it calls `spec_from_file_location` on `R582_HELPER` and executes that pathname
again. A planted path swap after snapshot capture causes the swapped helper to
execute. Thus the recursive executable closure is not immutable, violating v6
lesson 26 and its `managed_exec_uses_the_verified_immutable_source_bytes` test.

### 2. The advertised model-free dry run reaches prior scientific outcomes

`producer.run_dryrun()` calls `execution_plan()`, which calls
`source_hashes() -> validate_authorities() -> R588.verify_preoutcome_authority()`.
R588's authority mapping includes the R576 result and R579 audit, and hashes
both files. Hashing reads their bytes. This is exactly the transitive
prior-outcome dependency forbidden by v6 lesson 25, even though R590 does not
parse their numerical contents. It also means the managed preflight for both
dry and real modes depends on mutable outcome paths that the adapter did not
capture.

### 3. Saved provenance can name bytes other than the code that ran

The adapter executes the captured producer, but `source_hashes()` and every
result/receipt implementation field later hash `SCRIPT` and `TEST` by pathname.
The call-site censuses likewise parse R584 and R590 path contents rather than
the captured sources. A planted producer-path swap changes the implementation
digest reported by the in-memory verified producer. The computation therefore
can run captured byte set A while its prospective result claims mutable path
byte set B. This breaks the evidence-to-executable join even if the scientific
calculation itself is unchanged.

## Required prospective repair

1. Make R588 consume the already verified R582 module (or inject a captured
   helper explicitly); no executable dependency may call a path loader after
   snapshot capture.
2. Split R588's authority checks into an outcome-free row/code authority used
   by R590 dry run and a scientific-outcome authority used only where outcomes
   are explicitly in scope. Prove the complete dry-run call graph never opens
   the R576 result or R579 audit.
3. Pass the immutable source snapshot and digest map into R590 provenance and
   call-site census code. Result and receipt hashes must describe the actual
   compiled bytes, not a later read of their old paths.
4. Retain the existing scientific contract unchanged, then obtain a fresh
   independent exact-byte review. This review does not authorize a model run.

## Planted checks

The focused review test reports `2 passed, 3 strict xfailed`. The passing checks
bind the exact packet/v6 hashes and verify that immutable module loading creates
no R590 outcome namespace. The three strict expected failures are the mutable
recursive R582 import, transitive dry-run outcome read, and executed-byte versus
reported-provenance path swap described above.

Test file:
`test_numbered_list_cached_value_downstream_use_rung590_immutable_closure_review.py`
