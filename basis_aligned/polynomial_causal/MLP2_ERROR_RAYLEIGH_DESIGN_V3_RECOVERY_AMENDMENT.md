# MLP2 error-Rayleigh DESIGN v3 recovery amendment

**Frozen after the v2 scorer terminal and before any v3 response access:**
2026-08-30 01:10 UTC.

V2 DESIGN collection completed and published an authority, tensor ledger, and receipt.
The separately audited scorer then failed before scorer authority and before opening the
DESIGN ledger.  The v2 collector source closure had unnecessarily included the future
scorer and its tests.  Repairing the scorer therefore made current-working-tree replay
of the immutable collector authority fail even though the response computation and all
v2 collection artifacts were unchanged.

The exact spent v2 artifacts are immutable:

- DESIGN authority: `346fc2a85b195b907e1ca60c3976acb824d46f7ad6b88261df525442fd2e3cd8`;
- DESIGN ledger: `171798a41c450b17302b9907853633daef865fb4bbfd9b9ec0cdaedb6137c8bd`;
- DESIGN receipt: `16bb0597ed78830a2c166ab68ca2faadc2b077faedf02900df31de822a51b880`;
- collector audit: `1a4ac5a36fd140fc3c2b62fa2dc1f450048c46a7613fa6a496cd2162cf737790`;
- scorer audit: `80fc2c78a1047fc96e822e43340100cc98d6734e7d3eda42ce35c2766b64cf72`;
- scorer failure: `5a7021bc64ea9d6cfff3d9f6814c7cd626e6a528bbb62f8ba9735939ba188a38`.

No v2 scorer authority, bundle, receipt, or lock exists.  No v2 HELDOUT authority,
ledger, receipt, failure, or lock exists.

V3 deliberately recollects DESIGN rather than treating the v2 ledger as a scientific
sufficient statistic.  It uses the same frozen rows, programs, backgrounds, controls,
amplitudes, seeds, feature functions, finite endpoint, predictor families, ridge grid,
nulls, and scientific thresholds.  The only implementation changes are:

1. fresh `mlp2_error_rayleigh_v3_*` collector, scorer, and lock namespaces;
2. exact protected binding of the v2 completed-collection/scorer-failure lineage; and
3. removal of the downstream scorer and scorer tests from the *collector* source
   closure.  They remain in the scorer source closure.  Thus future scorer lifecycle
   repair cannot retroactively invalidate a receipt-complete collection authority.

V3 requires a fresh exact source-bound independent collector audit before either role
opens, and a separate fresh scorer audit before DESIGN is deserialized for fitting.
V1 and v2 artifacts must never be deleted, overwritten, or silently promoted.
