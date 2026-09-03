# Rung 523 terminal receipt — attention8 projector optimizer diagnosis

**Scored:** 2026-09-03 09:23 UTC  
**Status:** `raw_adam_through_qr_closed`  
**Claim level:** optimizer calibration only; no circuit evidence

## Frozen question and answer

Rung 523 asked whether either of two minimal changes made all fifteen attention8 leave-one-circuit-out projector fits
healthy: use one fixed response scale for each target and attention map, or lower Adam's learning rate from `0.03` to
`0.003`. A third arm combined both changes. Every arm retained rank 4, the same seeds, 200 updates per fit, the same
training and held-out validation examples, and the same control penalty.

No arm passed the registered all-fits rule:

| Optimization rule | Fits passing both training and held-out checks | Losses above 100 | Losses above 1,000 | Decision |
|---|---:|---:|---:|---|
| Fixed target/map scale, learning rate 0.003 | 7/15 | 0 | 0 | Fail |
| Fixed target/map scale, learning rate 0.03 | 1/15 | 0 | 0 | Fail |
| Row-specific scale, learning rate 0.003 | 5/15 | 49 | 13 | Fail |

The fixed scale completely removed the catastrophic numerical spikes. It did not solve inconsistent optimization:
the best fixed-scale arm improved held-out validation for 10/15 fits and the late training window for 8/15, but only
7/15 improved both. The registered decision therefore adopts no arm, licenses no repeat of rung 522, and closes raw
Adam through differentiable QR for this route. This says nothing about whether a selective attention8 subspace
exists because the optimizer never became a valid measuring instrument.

## Independent audit

The CPU auditor independently reloaded all 45 saved `1152 x 4` frames and checked each tensor hash, candidate record,
diagnostic record, health score, three-arm decision, dependency hash, split seal, and exact model-call ledger. It
passed. TEST was inaccessible and unopened; omitted targets were not evaluated; no removal call ran. The exact price
was 9,000 optimization forwards, 9,000 backwards, 515 inference-only forwards, zero removal forwards, and 865.083
seconds.

Immutable artifacts:

- result SHA-256: `8126ba1ca090f334f57e873f3bb7afe3d02603e9f410935b0dfc48239bb12fe3`
- frame archive SHA-256: `c50f530d3e98e98538e1fc95133feb9b712b3e973775cad4b801e974d2a5a569`
- terminal audit SHA-256: `ef72eb7acbd23fc48b4e5467859c98af71ee9e656672a60bd1b221b805f66741`

## Consequence

Do not continue with more Adam learning-rate, response-normalization, or rank sweeps. Rung 524 is a CPU-only planted
test of a direct optimizer on the space of rank-4 subspaces. It changes the optimization geometry instead of tuning
the failed coordinate system. Its purpose is still circuit-level: establish whether we have a valid instrument for
finding an attention8 piece whose held-out downstream effect is selective. Rank 4 remains only a matched probe size,
not an interpretation claim. If the direct optimizer cannot recover a known planted subspace, close this attention8
route and pivot to the exact MLP0 token-only, token-by-context, and context-only decomposition.
