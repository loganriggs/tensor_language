# Rung 529 managed v2 smoke: exact and live

**Managed execution:** 2026-09-03 11:37:31--11:37:39 UTC, runner exit `0`

**Result:** `equality_shared_private_transition_consensus_rung529_gpu_smoke_v2_results.json`

**Result SHA256:** `03a039a0ea4735f196d9f84457803f88ac95eea46b19ae89de8b8eef5223d213`

**Core runner SHA256:** `8d3c00b73efaca5427d32d3fec0cd8629ffe3cddb70389a9d4c878cb38730f85`

The v2 instrument passes. It ran exactly 37 model forwards; constructed all 26 consensus, private, single-donor, and
wrong-sign states; reproduced all four target boundaries exactly after the one model-boundary cast; and exercised
live A14/M17 continuation patches. Minimum edit-state RMS was `3.6854`, minimum continuation-patch RMS was `1.0772`,
and peak allocated GPU memory was `3.64 GB`.

No task masks, CE differences, circuit effects, candidate identities, or scientific outcomes were retained. This
receipt licenses only the separately hashed full launcher. It does not itself support the shared-computation claim.
