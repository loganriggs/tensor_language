# MLP2 error-Rayleigh scorer v4 recovery amendment

**Frozen after the v3 scorer terminal and before any v4 DESIGN access:**
2026-08-30 01:24 UTC.

V3 DESIGN collection is receipt-complete.  Its independently audited scorer stopped
before publishing scorer authority and before deserializing DESIGN because an unrelated
concurrent commit was local HEAD but had not yet reached the remote-tracking ref during
the scorer's ancestry check.

The exact v3 scorer failure is immutable:

- failure SHA-256: `d715167e26aec84378d6a48bbcabe8dfd3953cc8d108b959b8b300e88a16c3a6`;
- `design_ledger_may_have_opened=false`;
- no v3 scorer authority, bundle, receipt, or lock exists.

V4 is scorer-only.  It reuses the exact receipt-complete v3 DESIGN authority, ledger,
and receipt and uses fresh `mlp2_error_rayleigh_v4_design_predictor_*` transaction paths.
It changes no predictor feature, family, ridge value, null, control, selection rule, or
HELDOUT threshold.  The scorer source closure adds this amendment and binds the exact v3
failure/absence state.

The implementation change is only source-authority selection: the v4 scorer takes its
source commit from the exact independent audit artifact and verifies the current scorer
bytes against that audited commit.  It does not use moving repository HEAD.  A fresh
exact-source independent v4 scorer audit is required before DESIGN deserialization.
HELDOUT remains sealed until a v4 predictor receipt and separately audited unlock path
exist.
