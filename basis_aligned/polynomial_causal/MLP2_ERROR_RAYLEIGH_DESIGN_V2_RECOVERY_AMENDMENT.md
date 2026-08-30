# MLP2 error-Rayleigh DESIGN v2 recovery amendment

**Frozen after the v1 terminal failure and before any v2 response access:**
2026-08-30 00:20 UTC.

The audited v1 DESIGN collector opened model responses but failed before publishing a
ledger or scientific receipt.  Its control-binding hash called NumPy directly on a
BF16 tensor, which raises `TypeError('Got unsupported ScalarType BFloat16')`.

The spent artifacts are immutable:

- v1 authority SHA-256:
  `d5d6f785a61568ed1aa6979af1eeea76183d1ffb6f080415cc294a68252ae8db`;
- v1 failure SHA-256:
  `a8b6a88d342db2f2b2e3720cf87bb40caac4333d240dc27e04498d078585bbba`;
- no v1 ledger, receipt, predictor, HELDOUT authority, or HELDOUT outcome exists.

V2 makes exactly one implementation change: all control-binding tensor hashes encode
the original dtype string and shape, then hash exact contiguous raw bytes through a
CPU `uint8` view.  No tensor is numerically cast.  The programs, rows, amplitudes,
controls, response computations, finite endpoint, call census, predictor, and
scientific gates are unchanged.

V2 uses fresh `mlp2_error_rayleigh_v2_*` transaction paths, binds and revalidates both
spent v1 hashes in every protected snapshot, and requires a fresh exact source-bound
independent audit.  V1 outcomes may not be reused as scientific sufficient statistics.
