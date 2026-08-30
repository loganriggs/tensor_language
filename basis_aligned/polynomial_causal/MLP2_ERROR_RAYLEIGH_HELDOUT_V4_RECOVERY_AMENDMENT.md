# MLP2 error-Rayleigh HELDOUT v4 recovery amendment

**Frozen after the v4 predictor receipt and before HELDOUT access:**
2026-08-30 01:31 UTC.

The v3 DESIGN collection and v4 scorer are receipt-complete.  HELDOUT has never opened.
The v4 predictor artifacts are immutable:

- authority: `83f80fd8815318b2424141ef025deca6f8187d343d1617790f812dc0a0bc27e5`;
- bundle: `d9c42a542822675997a8711457dbe9e806e6bf515c25a8e7ef4d5f700cbef5bf`;
- receipt: `7747c25591a6c3779c571e6b7f563c6a06ad46f531911840bc612058c1fd9e7b`;
- scorer audit: `4996d574affe06defb11851518a0f101cd69bda17c6451f24fbb78c90bae6568`.

V4 HELDOUT uses a fresh `mlp2_error_rayleigh_v4_heldout_*` namespace and a fresh exact
collector audit.  It changes no row, program, feature, response, control, predictor,
or scientific threshold.

The implementation separates two valid source checks:

1. **Current executable check:** the HELDOUT collector must byte-match its fresh v4
   collector audit before opening HELDOUT.
2. **Historical artifact check:** the receipt-complete v3 DESIGN collection and v4
   scorer are verified against the committed source maps and audits that existed when
   their authorities were published.  Later downstream wiring may differ in the current
   checkout without retroactively invalidating immutable completed artifacts.

Historical replay still checks every committed blob hash, audit/source-map join, parent
snapshot, row receipt, authority/ledger/receipt hash, predictor reproduction, and spent
lineage.  It does not permit an uncommitted or unaudited current HELDOUT executor.

HELDOUT remains locked until this amendment, runner, tests, v4 predictor receipt, and
fresh v4 collector audit all replay exactly.
