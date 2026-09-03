# Rung 532 managed v3 smoke receipt

**Completed:** 2026-09-03 12:35 UTC

V3 completed with exit code 0 and its fail-closed instrument predicate true. In addition to all v2 numerical and edit
checks, it executed the corrected real 62-circuit support accumulator. No per-arm loss or scientific circuit outcome
was accumulated or reported.

- core SHA-256: `142f4a0f05d582413fb6eac1820654dc6d4491690af9742e0a2d81eac719fdb8`
- v3 wrapper SHA-256: `82c298105d4e5f57256c3dbc03431f13ae87220e61bc8275b485096c32112913`
- v3 log SHA-256: `53004ae5b1e3cad2e3e2ed25b6b6091a1f93599e7c5a3593825f4ecfc84520a2`

The log is byte-identical to the v2 log because the new structural check is enforced inside the predicate but does
not emit circuit support counts. It records 21 calls, exact native replay, exact factor product, reconstruction error
`4.56e-14`, donor/target edit minima `7.890/2.147`, zero dead edits, and checkpoint identity.

This is the terminal smoke authority for the full 2,625-forward run.
