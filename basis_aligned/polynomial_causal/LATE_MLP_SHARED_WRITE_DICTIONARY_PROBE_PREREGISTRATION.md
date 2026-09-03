# Late-MLP shared write dictionary — preregistration

Registered 2026-09-03 20:29Z (box clock; a first draft line said 20:31Z, corrected before the script hash was frozen), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): every number is CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split as §2703/§2709: bases from docs 96–191, baseline 3.0322401) —
LOWER IS BETTER. Descriptive; nothing installs into the §312 frontier.

## Question (re-usability of components across MLPs)
§2709: the 7 late MLPs (mlp11–17) jointly truncated to their OWN top-128 write directions cost .385 nat, and the cross terms
shrink more slowly than the singles with k — the residual tails of different late MLPs look aligned. If the late MLP writes
share a subspace of the residual stream, then ONE dictionary (a single K-dimensional subspace fitted on the pooled writes of all
seven) applied at all seven sites should cost little more than seven separate ones — that is a reusable component. If they do
not share it, the shared dictionary is far worse. This is the direct CE test of "reuse the same write subspace across MLPs".

## Arms (MLP7 = mlp11…17 only; all truncations simultaneous)
- SEP_k: each site truncated in its own in-situ PCA frame, k ∈ {32, 128, 512} (SEP_128 = §2709's JOINT_MLP7(128) = .385).
- SHARED_K: one basis U_K from the PCA of the pooled per-site-centred writes of all seven sites (equal weight per site), each
  site truncated to μ_s + U_K U_Kᵀ (w − μ_s), K ∈ {32, 128, 512}.
- PAIR_SEP_128(s,t) and PAIR_SHARED_128(s,t) for the six adjacent pairs (11,12) … (16,17): the two sites truncated together with
  own bases (k=128 each) vs one pooled basis (K=128) for both.
Weights-only: captured-energy fraction of each site's write under its own top-128 vs under the shared top-128.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4 of 3.0322401; SEP_128 within .01 of .385; SEP and SHARED monotone in K.
- **pred_b_shared_dictionary_cheap**: SHARED_128 ≤ 1.5 × SEP_128. Null: ≥ 2.5 × SEP_128.
- **pred_c_shared_512_converges**: SHARED_512 ≤ SEP_512 + .03. Null: ≥ SEP_512 + .10.
- **pred_d_energy_overlap**: the shared top-128 captures ≥ .7 of the energy that the site's own top-128 captures, at ≥ 5 of 7
  sites. Null: ≤ 3 of 7.
- **pred_e_adjacent_pairs_share**: PAIR_SHARED_128 ≤ 1.3 × PAIR_SEP_128 for ≥ 4 of the 6 adjacent pairs. Null: ≤ 1 of 6.

## Price
96 fit + 64 · (1 + 3 + 3 + 12) + 8 ≈ 1,320 GPU document-forwards ≈ 20 s. Output late_mlp_shared_write_dictionary_probe_results.json.
Frozen: this file, §2709 results (6b2708a3…), checkpoint, fit_natural.pt.
