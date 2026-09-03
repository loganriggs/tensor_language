# How is the mlp11–15 pool's value routed — through the 16 core directions mlp16/17 read, or directly to the readout? — preregistration

Registered 2026-09-03 21:16Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; means/fillers from docs 96–191; baseline 3.0322401) — LOWER IS
BETTER. Descriptive; nothing installs into §312.

## Question
§2721: mean-ablating all of mlp11–15 (the "pool") with mlp16/17 intact costs .724 nat. §2719: the pool's writes are mostly outside
the 16-dim core (oracle core on late7 recovers 42%). §2722: the core input coordinates of mlp16/17 are supplied diffusely. So
where does the pool's .724 go — into the 16 core coordinates that mlp16/17 compute on (and from there into their core write), or
straight to the readout / to mlp16/17's non-core input? And does the extracted 16-dim program of §2720 (own weights on core input
+ mean filler) still depend on the pool?

## Arms (pool = mlp11..15 jointly; P_M = §2710 16-dim CORE_TN; μ_l = fit-set means)
- POOL_MEAN: pool → μ (reference .724).
- POOL_CORE_ONLY: pool writes μ + P_M(w − μ) (keeps only the pool's core variation).
- POOL_NONCORE_ONLY: pool writes w − P_M(w − μ) (removes only the pool's core variation).
- POOL_MEAN + W16_17_MEANFILL: pool → μ AND mlp16/17 replaced by their §2720 own-weights-on-core-input program (mean filler). Compare to
  W16_17_MEANFILL alone (§2720 .309): the pool's value TO the 16-dim program is Δ_prog = CE(POOL_MEAN + prog) − CE(prog).
- POOL_MEAN + MEAN(16,17): the pool's value with 16/17 gone (§2721 1.885 − .848 = 1.037).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; POOL_MEAN within .02 of .724; MEAN(16,17) within .02 of .848; W16_17_MEANFILL within .02 of .309.
- **pred_b_pool_value_is_mostly_non_core**: CE(POOL_NONCORE_ONLY) ≤ .30 (removing only the pool's core variation costs ≤ .30 of .724). Null: ≥ .55.
- **pred_c_pool_core_alone_is_not_enough**: CE(POOL_CORE_ONLY) ≥ .40. Null: ≤ .15.
- **pred_d_the_16_dim_program_still_needs_the_pool**: Δ_prog ≥ .50. Null: ≤ .20.
- **pred_e_pool_matters_more_when_16_17_are_gone**: CE(POOL_MEAN + MEAN(16,17)) − CE(MEAN(16,17)) ≥ 1.2 × CE(POOL_MEAN). Null: ≤ 1.0 ×.

## Price
96 fit docs × 2 passes + 64 × (1 + 7) ≈ 700 GPU document-forwards ≈ 15 s. Output late_pool_routing_probe_results.json.
Frozen: this file, §2721 results (884a0bba…), §2720 results (57772458…), checkpoint, fit_natural.pt.
