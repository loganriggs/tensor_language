# The whole late MLP stack as one program: [one linear context map at block 11] + [mlp16/17 own weights on 16 core coordinates] — preregistration

Registered 2026-09-03 21:21Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191; baseline 3.0322401) — LOWER IS BETTER.
Descriptive price of a candidate program; nothing installs into §312.

## Question
§2724: one 1152 × 1152 linear map of the block-11 stream replaces mlp11–15 at 52% of their value (five in sequence: 62%). §2720:
mlp16/17's own weights on their 16 core input coordinates (+ token filler) replace them at 75%. §2723: the two channels interact
(the 16-dim program needs the pool). What does the WHOLE late MLP stack cost when BOTH replacements are applied together, against
MEAN(all 7) = 1.885 — and how large is the composition penalty?

## Arms
- MEAN7 (ref 1.885) · ONESHOT_LIN alone (ref .345) · W16_17_TOKFILL alone (ref .214) · SEQ_LIN alone (ref .278).
- COMBINED_ONESHOT: ONESHOT_LIN for the pool + W16_17_MEANFILL (no fitted map inside 16/17; the cheapest program).
- COMBINED_ONESHOT_TOK: ONESHOT_LIN + W16_17_TOKFILL.
- COMBINED_SEQ_TOK: SEQ_LIN + W16_17_TOKFILL (the richest fitted program).
Composition penalty π = CE(COMBINED_ONESHOT_TOK) − CE(ONESHOT_LIN) − CE(W16_17_TOKFILL).

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN7 within .03 of 1.885; ONESHOT_LIN within .02 of .345; W16_17_TOKFILL within .02 of .214; SEQ_LIN within .02 of .278.
- **pred_b_combined_program_recovers_half_the_stack**: CE(COMBINED_SEQ_TOK) ≤ 0.95 (≥ 50% of 1.885 recovered). Null: ≥ 1.40.
- **pred_c_composition_penalty_bounded**: π ≤ .40. Null: ≥ .80.
- **pred_d_sequence_helps_in_combination**: CE(COMBINED_SEQ_TOK) ≤ CE(COMBINED_ONESHOT_TOK) − .05. Null: ≥ CE(COMBINED_ONESHOT_TOK).
- **pred_e_cheapest_program_still_beats_mean_by_a_third**: CE(COMBINED_ONESHOT) ≤ 1.25. Null: ≥ 1.60.

## Price
96 × 3 fit passes + 64 × (1 + 7) ≈ 800 GPU document-forwards + ridge solves ≈ 25 s. Output late_stack_combined_program_probe_results.json.
Frozen: this file, §2724 results (c3e1b9f3…), §2720 results (57772458…), checkpoint, fit_natural.pt.
