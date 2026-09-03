# Who supplies the 16 core input coordinates that mlp16/mlp17 compute on? — preregistration

Registered 2026-09-03 21:11Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; means from docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into §312.

## Question
§2720: mlp16/17's own weights reproduce 64–75% of their value from the 16 core coordinates c = U_Mᵀ x̂ of their input alone. §2715
(provenance of the core in the STREAM, by energy) named attn6/attn7/attn1/attn5 and a block-5 offset as the pre-16 contributors.
Energy is not value: this rung asks which upstream WRITES the 16 input coordinates causally depend on, by replacing each upstream
write's CORE COMPONENT by its mean (leaving its other 1136 directions intact) and measuring the CE cost — a 32-arm map over
attn0..attn15 and mlp0..mlp15, plus the ALL-upstream arm and a per-kind split.

## Arms
CORE_MEAN(s): w_s → w_s − P_M(w_s − μ_s) for one site s ∈ {attn l, mlp l : l ≤ 15} (32 arms).
CORE_MEAN(ALL32), CORE_MEAN(all attn ≤ 15), CORE_MEAN(all mlp ≤ 15). Reference: MEAN(mlp16+17) (.848, §2716).
Also reported: the stream's core-coordinate variance explained by each site's core write (energy attribution, fit set), for
comparison with the causal costs.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN(mlp16+17) within .02 of .848.
- **pred_b_core_input_is_load_bearing**: CORE_MEAN(ALL32) ≥ .50. Null: ≤ .15.
- **pred_c_few_suppliers**: the top-3 single-site costs sum to ≥ .60 × the sum of all 32 single-site costs. Null: ≤ .30 ×.
- **pred_d_energy_names_the_suppliers**: Spearman(single-site cost, energy attribution) ≥ .60 over the 32 sites. Null: ≤ .20.
- **pred_e_attention_supplies_more_than_mlp**: CORE_MEAN(all attn ≤ 15) ≥ 1.5 × CORE_MEAN(all mlp ≤ 15). Null: ≤ 0.8 ×.

## Price
96 fit docs (covariances of the 36 writes + stream) + 64 × (1 + 1 + 35) ≈ 2500 GPU document-forwards ≈ 30 s.
Output core_input_provenance_probe_results.json. Frozen: this file, §2720 results (57772458…), checkpoint, fit_natural.pt.
