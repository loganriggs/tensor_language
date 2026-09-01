# Bilin18 MLP module dossier index

*Started 2026-09-01 after the rung-388 audit. This file is the mandatory duplicate-work check before a new MLP
experiment. “Rank” is meaningless unless the row also says which object was ranked and how it was evaluated.*

## Shared native computation

Every MLP receives `x in R^1152` and computes

`y = Down[(Left x) elementwise-times (Right x)] + bias`,

with `Left, Right: R^1152 -> R^4608` and `Down: R^4608 -> R^1152`. The native price is 15,926,400 stored
numbers per MLP. The 4,608-dimensional middle vector is the product-feature vector.

Four objects must not be conflated:

1. **Whole-layer replacement:** an executable substitute for `x -> y`; this is the strongest compression claim.
2. **Input-map factorization:** simplifies `Left` and `Right` while retaining all 4,608 products and native `Down`.
3. **Output-map factorization:** simplifies `Down` on observed product activations; it need not simplify how those
   activations are computed.
4. **Output-direction projection:** keeps a few directions of the layer output; its rank is not product width or
   input rank.

## Coverage table

The historical `r80` column below is the activation-conditioned rank needed for a factorization of **Down** to
recover 80% of that module's loss benefit on the old held-out evaluation (§713). It is not a whole-MLP rank.

| MLP | old Down-map r80 | current dossier | duplicate-work note |
|---:|---:|---|---|
| 0 | 8 | [MLP0_CURRENT_UNDERSTANDING.md](MLP0_CURRENT_UNDERSTANDING.md) | exact position-zero fold, token/context decomposition, Down and shared-input replacements already extensive |
| 1 | 128 | not yet consolidated | high-benefit, high-rank early module; check §§13–19 and §§713–714 before work |
| 2 | 256 | not yet consolidated | early-middle map; check §713 and MLP0→MLP2 composition records |
| 3 | 256 | not yet consolidated | check §713 and depth/class records |
| 4 | 64 | not yet consolidated | adopted p768 shared-input replacement; check §§2416–2422 and §§2464–2475 |
| 5 | 256 | not yet consolidated | check §713 and bias/linearization records |
| 6 | 512 | not yet consolidated | individually small old Down-map benefit; do not infer joint irrelevance |
| 7 | 512 | not yet consolidated | p768 input replacement/composition already screened |
| 8 | 512 | not yet consolidated | p768 input replacement/composition already screened |
| 9 | 512 | not yet consolidated | mid-stack quadratic-form ranks and input replacement already screened |
| 10 | 512 | not yet consolidated | lowest output-mode top-512 retention in rung 385; check §2482 |
| 11 | 512 | not yet consolidated | input replacement already screened |
| 12 | 512 | not yet consolidated | input replacement already screened |
| 13 | 512 | not yet consolidated | input replacement already screened |
| 14 | 512 | not yet consolidated | p768 input replacement and compositions already screened |
| 15 | 4 | not yet consolidated | late input replacement, sequential refit, and mode spectra already screened |
| 16 | 1 | [MLP16_CURRENT_UNDERSTANDING.md](MLP16_CURRENT_UNDERSTANDING.md) | old reported-13,832/corrected-14,984-number quadratic replacement must be rerun before another late-core design |
| 17 | 4 | [MLP17_CURRENT_UNDERSTANDING.md](MLP17_CURRENT_UNDERSTANDING.md) | rank-2 forms, output projections, functional Down rank, names, and causal checks already exist |

The missing dedicated files are documentation debt, not permission to ignore the linked ledger results. A new
experiment on any unconsolidated module must first promote its relevant ledger sections and receipts into a
standalone dossier.

## Required preflight for a new MLP experiment

Before registering a run, its module dossier must answer:

- What exact object is being simplified: whole layer, input maps, product features, Down, or output directions?
- What are every intermediate shape and the literal stored-number/byte bill?
- Which earlier scripts and ledger sections tested the same object or a strict superset?
- Were earlier numbers local R², output variance, CE recovery relative to deletion, absolute CE damage, behavior
  checks, or causal effects?
- Are fit/evaluation rows and checkpoint version comparable?
- What matched-price control makes the new question different?
- What result would stop this family rather than trigger rank/site tuning?
