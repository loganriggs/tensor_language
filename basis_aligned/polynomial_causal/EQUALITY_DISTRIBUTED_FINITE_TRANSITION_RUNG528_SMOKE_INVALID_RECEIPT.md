# Rung 528 first smoke invalid receipt

**Detected:** 2026-09-03 11:00 UTC

The first smoke artifact is operationally invalid and is not used as evidence for GPU eligibility.

The enqueue helper correctly ran the submitted wrapper once with `BQLIB_DRYRUN=1` as a GPU-free preflight. The
wrapper ignored that environment variable and called the CUDA smoke unconditionally. The result file was created at
`10:59:28 UTC`, before the managed runner recorded its start at `10:59:48 UTC`. The queued process then exited 1
because overwrite protection found the pre-existing result.

The preserved invalid artifact has SHA-256
`436d98a2c5f66fc8fdedf1143d2cd4d145e73134bd076708085614262cd83374`. Its 22 forwards happened to pass every
numerical check and retained no task or circuit effects, but those numbers cannot certify the managed execution
requirement.

The correction is a new `gpu_smoke_v2` wrapper and output namespace. Both the original and v2 wrappers now branch to
the CPU-only dry run whenever `BQLIB_DRYRUN=1`; only their normal managed invocation can call CUDA. The v2 result must
be born after the runner's logged start and the completion ledger must report exit 0 before it is interpreted.

No scientific threshold, candidate, data split, or full-run code changed. The invalid artifact is preserved so the
failure is auditable rather than silently replaced.
