# Where does the late stack's superadditive ~1 nat sit? The mean-ablation lattice over mlp11–17 — preregistration

Registered 2026-09-03 21:10Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; means from docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive; nothing installs into §312.

## Question
§2719: mean-ablating mlp11–15 one at a time costs .03–.05 each (sum .20); mlp16+17 together .848; all seven 1.885. The .84-nat
excess is either (A) redundancy AMONG mlp11–15 (they back each other up; removing all five costs far more than .20 even with
16/17 intact), or (B) an interaction BETWEEN the groups (mlp16/17 compensate for missing earlier blocks; 11–15 only matter once
16/17 are gone), or both. This rung measures the full 2^5 lattice of mlp11–15 mean-ablations, once with mlp16/17 intact and once
with mlp16/17 mean-ablated (64 arms).

## Arms
For every subset S ⊆ {11,…,15}: MEAN(S) and MEAN(S ∪ {16,17}). Define
  within5 = MEAN({11..15}) − Σ_s MEAN({s})                          (redundancy among the five, 16/17 intact)
  between = MEAN(all 7) − MEAN({11..15}) − MEAN({16,17})              (group interaction)
  pair_ij = MEAN({i,j}) − MEAN({i}) − MEAN({j}) for the 10 pairs (16/17 intact)
  marg_s|drop = MEAN({s} ∪ {16,17}) − MEAN({16,17}) (each block's marginal value once 16/17 are gone).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN({16,17}) within .02 of .848; MEAN(all 7) within .03 of 1.885; singles within .02 of §2719.
- **pred_b_between_group_interaction_dominates**: between ≥ .50 and between ≥ 2 × within5. Null: between ≤ .15.
- **pred_c_five_jointly_cheap_with_16_17_intact**: MEAN({11..15}) ≤ .40. Null: ≥ .80.
- **pred_d_marginals_grow_without_16_17**: median_s marg_s|drop ≥ 3 × median_s MEAN({s}). Null: ≤ 1.5 ×.
- **pred_e_pairwise_interactions_small**: median pair_ij ≤ .02 (16/17 intact). Null: ≥ .05.

## Price
96 fit docs (one covariance pass) + 64 × (1 + 64) ≈ 4300 GPU document-forwards ≈ 45 s. Output late_mlp_subset_lattice_probe_results.json.
Frozen: this file, §2719 results (c0aaa202…), checkpoint, fit_natural.pt.
