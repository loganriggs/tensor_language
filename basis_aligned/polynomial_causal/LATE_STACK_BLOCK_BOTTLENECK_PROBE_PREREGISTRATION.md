# Preregistration — late_stack_block_bottleneck_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:27Z (box clock). Follows §2739 (ALL7_TOK_256 = .297). Independent of late_stack_width_and_token_rank_probe (queued ahead of it).

## Question
Where does the seven-block late program's remaining .297 live? Three decompositions at fixed k = 256 with the token filler:
(1) SINGLE_l — only block l replaced (others real), l = 11..17: the block's cost alone; (2) STACK_MINUS_l — all seven replaced except
block l, which is real: the marginal gain of restoring one block inside the stack; (3) the composition penalty
STACK − Σ_l SINGLE_l. Plus one arm for the ranked item "input-centred shared core": the last two blocks on a SHARED 16-dim core taken
from the pooled centred INPUT covariance of mlp16/17 (each block keeps its own input mean), constant filler — to compare with §2738's
write-derived shared core (KEEP_CORE16 .309) and the per-block own-16 (.2425).

## Arms (own weights; own centred input PCA + means on docs 96–191; eval docs 0–63; output unrestricted; TOK = §2730 ridge read)
STACK = ALL7_TOK_256 (reproduces §2739). SINGLE_l_TOK_256 (7). STACK_MINUS_l (7). SHARED_IN16_LAST2_CONST (1).

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; STACK within .02 of .297.
- pred_b_composition_penalty_positive: STACK − Σ_l SINGLE_l ≥ .04 (the blocks' errors compound, as in §2735's κ). Null: ≤ .00.
- pred_c_one_block_dominates: max_l (STACK − STACK_MINUS_l) ≥ .08. Null: ≤ .04 (≈ .297/7 — evenly spread).
- pred_d_bottleneck_is_mlp16: argmax_l (STACK − STACK_MINUS_l) = 16 (its input has the highest effective rank, 544, §2738). Null:
  argmax ∈ {11, 12, 13}.
- pred_e_input_shared_core_beats_write_core: SHARED_IN16_LAST2_CONST ≤ .28. Null: ≥ .30 (no better than the write-derived core).
Descriptive: per-block SINGLE_l and STACK_MINUS_l tables; "alone" vs "marginal in stack" per block.

## Price
96 fit docs + 64 × (1 + 16 arms) = 1184 GPU document-forwards, ~20 s. Frozen: this file, §2739 results, checkpoint, fit_natural.pt.
Reproduction tolerance .02.
