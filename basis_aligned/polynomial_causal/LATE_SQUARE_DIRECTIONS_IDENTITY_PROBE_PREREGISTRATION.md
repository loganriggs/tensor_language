# What are the five shared square directions of mlp16/17? — identity, producers, pool-error concentration, and a random-subspace CE control — preregistration

Registered 2026-09-03 21:54Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; bases/fillers on docs 96–191; baseline 3.0322401) — LOWER IS
BETTER. Descriptive; nothing installs into §312.

## Question
§2729: mlp16 and mlp17 square the same five stream directions (principal cos² .9999/.997/.988/.966/.905 between their top-8 square
spaces), and a shared 8-dim square space serves both. What ARE these directions? Four identity measures on the shared square
basis u_1…u_8 (eigvectors of S₁₆ + S₁₇ in the 16-dim core coordinates; stream directions q_j = P u_j):
(1) position in the core: energy of u_j on the top-6 core eigvectors of P_M (chance 6/16 = .375);
(2) readout: fraction of q_j inside LM_128 (top-128 right singular vectors of lm_head.weight; chance .111; §2713 CORE_16 overall .18);
(3) producers: covariance attribution of Var(x_pre-mlp16 · q_j) to each of the 32 upstream writes (attn0…attn16, mlp0…mlp15)
    (share_s = cov(w_s·q, x·q)/var(x·q); Σ shares reported as a check on the λ-mixing);
(4) pool-error concentration: with mlp11–15 under OWN_32_TOK (§2730), energy of the pool's write error (w_own − w_real) along q_j
    vs the mean along the other core directions (after projecting the error to the core);
and one CE control: (5) the program of §2732 (PROG, square space + rank-8 read, no offset) with the square space restricted to the
TOP-5 shared directions vs to 5 RANDOM directions inside the 16-dim core (median of 3 seeds).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; the five principal cos² between the top-8 square spaces of mlp16 and mlp17 reproduce
  §2729 within .01 each; PROG with the shared top-8 within .02 of §2732's .246.
- **pred_b_square_dirs_are_top_core_pcs**: mean energy of u_1…u_5 on the top-6 core eigvectors ≥ .70 (chance .375). Null: ≤ .45.
- **pred_c_not_readout_facing**: mean fraction of q_1…q_5 in LM_128 ≤ .30 (as the core overall, §2713). Null: ≥ .50.
- **pred_d_pool_error_concentrates_on_square_dirs**: (mean error energy along q_1…q_5) / (mean along the other 11 core directions) ≥ 1.5. Null: ≤ .8.
- **pred_e_five_shared_beat_five_random**: CE(PROG, top-5 shared) ≤ .45 AND median CE(PROG, 5 random core directions) ≥ .90. Null: shared ≥ .60 OR random median ≤ .55.
Descriptive (no bar): the producer attribution table (32 sites × 5 directions); pool share Σ_{mlp11–15} share_s per direction.

## Price
96 fit docs × 3 passes + 64 × (1 + 1 + 1 + 3 + 1) ≈ 740 GPU document-forwards ≈ 25 s.
Output late_square_directions_identity_probe_results.json. Frozen: this file, §2732 results
(late_stack_extracted_program_probe_results.json), §2729 results (late_core_square_features_probe_results.json), checkpoint, fit_natural.pt.
