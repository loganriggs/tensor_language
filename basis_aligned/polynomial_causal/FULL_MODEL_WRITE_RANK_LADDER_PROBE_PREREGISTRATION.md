# Full-model write-rank ladder — preregistration

Registered 2026-09-03 20:34Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split: bases from docs 96–191, baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into the §312 frontier (the frontier prices weights, not in-situ write subspaces).

## Question
§2709 (late 14) and §2711 (early 8) give the joint price of rank-k in-situ write truncation for two thirds of the sites. What
does the WHOLE model cost when all 36 writes are truncated to rank k at once — the honest price curve of a "rank-k write program"
— and is the three-way (early / mid / late) decomposition additive at the stack level?

## Arms
Plain in-situ write PCA per site from docs 96–191. ALL36(k) for k ∈ {64, 128, 256, 512, 768}; EARLY8(k), MID14(k) (attn/mlp
4…10), LATE14(k) at the same k (EARLY8/LATE14 at 64…512 are instrument reproductions of §2711/§2709). Stack cross term
X_stack(k) = ALL36(k) − EARLY8(k) − MID14(k) − LATE14(k).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.0322401; EARLY8(128) within .01 of .692 and LATE14(128) within .01 of .486;
  all four ladders monotone in k.
- **pred_b_all36_256**: ALL36(256) ≤ .80 nat. Null: ≥ 1.5.
- **pred_c_all36_512**: ALL36(512) ≤ .25 nat. Null: ≥ .50.
- **pred_d_stack_cross_term**: X_stack(256) ≤ .5 × (EARLY8 + MID14 + LATE14)(256) (the three stacks interact less than the
  sites within a stack). Null: ≥ 1.0 × the sum.
- **pred_e_768_near_free**: ALL36(768) ≤ .05. Null: ≥ .15.

## Price
96 fit + 64 · (1 + 4·5) + 8 ≈ 1,450 GPU document-forwards ≈ 25 s (the 36-basis fit is the same one pass). Output
full_model_write_rank_ladder_probe_results.json. Frozen: this file, §2709 (6b2708a3…) and §2711 (1e0c96c5…) results, checkpoint, fit_natural.pt.
