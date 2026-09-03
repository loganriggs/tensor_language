# Preregistration — write-PCA truncation CE map over all 36 write sites — Claude CPU lane

Registered 2026-09-03 15:44 UTC (system clock), BEFORE running. Script: `ops/site_write_pca_truncation_ce_map_probe.py`.
Results: `site_write_pca_truncation_ce_map_probe_results.json`. Price: CPU only, 0 GPU forwards, ~96 + 64 x 43 = ~2,850
document forwards (256 tokens), est. 12-18 min on an idle runner. Companion to §2692 (usage ranks) and the queued MLP16/17
surrogate probe.

SIGN CONVENTION (§2135): all CE figures are CE ADDED ABOVE THE REAL MODEL on held-out documents — LOWER IS BETTER. This is a
DESCRIPTIVE map (activation-PCA truncation of each write, one site at a time); nothing is installed into the §312 frontier and
the closed §2118 metric-constructed-basis items are not reopened (bases here are data covariances of writes, scored by CE).

## Instrument
Natural rows (fit_natural): docs 0-95 FIT (centred covariance of each of the 36 writes — attention write and MLP write of each
block — over all 256 positions), docs 96-159 EVAL (CE over all 256 targets). For site s and rank k: write' = mu_s + U_k U_k^T
(write - mu_s), applied to that site only, everything else exact (tt_model-semantics manual forward, CE match <= 1e-4 checked).
k = 32 for every site; k = 8 additionally for attention 1/6/17 and MLP 16/17.

## Predictions (scored exactly as written)
- pred_a_instrument: manual CE == module CE within 1e-4 on 4 EVAL docs.
- pred_b_low_usage_attention_sites_cheap: the three lowest-usage attention writes of §2692 (blocks 1, 6, 17; natural eff rank
  21 / 31 / 20) each add <= .02 nat at k = 32. Null: any adds >= .10.
- pred_c_usage_rank_orders_truncation_cost: Spearman(FIT-half eff rank of the write, CE added at k = 32) across the 36 sites
  >= .6. Null: <= .2.
- pred_d_high_usage_sites_expensive: at least one site with FIT-half eff rank >= 500 adds >= .30 nat at k = 32. Null: every such
  site adds <= .10.

## Reading rules
b tests whether low in-situ usage rank of an ATTENTION write is a real compressibility handle (frontier-relevant: the frontier
compresses attention by weight rank). c tests whether the §2692 usage map is a usable price predictor. d tests that high usage
rank is a genuine obstruction rather than a covariance artefact. The full 36-row table is disclosed. Failures preserved; no
circuit claim; no explained-fraction change. Frozen inputs: §2692 results 63483cec..., checkpoint 680d6c26..., fit_natural
666a3201....
