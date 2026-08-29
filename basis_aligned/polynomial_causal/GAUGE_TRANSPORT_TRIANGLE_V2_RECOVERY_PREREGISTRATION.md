# Gauge transport triangle v2 one-change recovery

**Frozen before recovery execution:** 2026-08-29 08:34 UTC

V1 failed before loading the checkpoint or opening any scientific row because
`bilin18_observed_model_facade.py` imports `jacclust.tt_model`, while the runner had
not placed the repository root on Python's import path. The receipt-exclusive v1
failure has file SHA256
`d4f278eb5640ef7ea321fde864b0619feb80aedabd2341a6cf12214ff90d5c61` and records
`ModuleNotFoundError: No module named 'jacclust'`. It has no partial result or state.

V2 may make exactly one implementation change: add the immutable repository root
containing `jacclust/` to `sys.path` before importing the observed-model facade. It
must otherwise preserve the v1 checkpoint, 384 unique-document rows, bases, support
ranks, finite interventions, random seeds, amplitude grid, ridge, maps, metrics,
decision thresholds, and preliminary-screen scope specified in
`GAUGE_TRANSPORT_TRIANGLE_V1_EXECUTION_PREREGISTRATION.md`.

V2 uses a fresh source-closed authority and fresh create-only receipt/failure files.
The v1 failure remains immutable. The scientific result/state filenames were never
created by v1 and remain the fresh terminal targets. Any additional failure requires
another explicit recovery rather than an in-place retry.
