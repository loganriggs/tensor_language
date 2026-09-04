# R593 immutable-dispatch amendment — Linux per-argument limit

Registered: 2026-09-04 04:00 UTC, after the first managed R593 dispatch failed before model import.

## Observed implementation failure

The different-agent-approved R593 adapter verified all frozen bytes, unused outcome namespaces, model-free dry run,
and free-space gate. Its `scientific_command()` then base64-encoded the 103,879-byte producer inside one Python `-c`
argument. The encoded source is 138,508 bytes. On this machine:

$$
\operatorname{ARG\_MAX}=2{,}097{,}152,
\qquad
\operatorname{MAX\_ARG\_STRLEN}=32\times4096=131{,}072.
$$

The complete command was below `ARG_MAX`, but its single launcher argument exceeded Linux's per-string limit.
`os.execv` failed with `OSError: [Errno 7] Argument list too long`. The managed log is
`bilinear_quotient/runlogs/execute_induction_centered_fixed_geometry_rung593.log`.

This failure occurred before the child interpreter, Torch, checkpoint, CUDA, model, or any R593 outcome namespace was
opened. It is not a scientific terminal and does not expose FIT/SELECT outcomes.

## Frozen repair

Only the immutable-byte transport in `ops/execute_induction_centered_fixed_geometry_rung593.py` may change.

1. Read and SHA256-check the producer exactly as before.
2. Create an anonymous Linux memory file with `os.memfd_create(..., os.MFD_ALLOW_SEALING)`.
3. Write the exact producer bytes, seek to the beginning, and apply all four seals:

   $$
   F_{\rm SEAL\_WRITE}\;|\;F_{\rm SEAL\_GROW}\;|\;F_{\rm SEAL\_SHRINK}\;|\;F_{\rm SEAL\_SEAL}.
   $$

   Failure to apply or read back these seals aborts before model import.
4. Mark only that descriptor inheritable across `exec`.
5. Execute `python -I -c <small launcher> <fd>`. The launcher reads exactly the registered byte length, rejects a short
   or overlong stream, independently checks the producer SHA256, closes the descriptor, and executes the compiled bytes
   with the same logical `__file__`, `__name__`, immutable producer hash, and adapter hash as before.
6. No base64 source or other large string may appear in `argv`. Every individual argument must be below 4,096 bytes.
7. If the injected `exec_function` returns during a test or fails, the adapter closes the descriptor in `finally`.

The memory file has no pathname and cannot be written, grown, or truncated after sealing. This preserves the original
reason for embedding the verified bytes—dispatch cannot reopen a swapped producer path—without relying on a command-line
payload.

## Unchanged experiment contract

The producer, runtime, preregistration, authority rows, score/content/joint interventions, registered predictions,
nulls, exact $10^{-5}$ instrument threshold, FIT-first decision, call manifests, storage schemas, capacity thresholds,
and all six create-only R593 namespaces are unchanged. Maximum model work remains 639 FIT plus 322 SELECT forwards,
zero backwards, and zero updates. No rerun is allowed after any scientific or invalid-instrument terminal.

## Required model-free tests and review

Before another managed dispatch:

- the adapter's existing hash, namespace, dry-run, and one-byte-below-capacity tests must pass;
- a fake memfd test must reconstruct byte-identical producer contents, verify the complete seal mask, confirm the
  descriptor is inheritable, and prove every argument is shorter than 4,096 bytes;
- a child-process test using a harmless fixture must show that truncation, appended bytes, or a wrong digest aborts
  before executing the fixture;
- an injected returning/failing `exec_function` must not leak the descriptor;
- producer, runtime, science tests, and frozen dry-run bytes must remain unchanged;
- producer and adapter must pass the repository gate and advisory preflight; and
- a fresh different agent must exact-review the amendment, implementation, new hashes, and the preserved empty R593
  namespaces before re-enqueue.

The failed runlog is preserved. No threshold or scientific claim changes because of this amendment.
