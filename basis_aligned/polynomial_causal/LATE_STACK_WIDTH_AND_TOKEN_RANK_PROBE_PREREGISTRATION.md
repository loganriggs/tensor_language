# Preregistration — late_stack_width_and_token_rank_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:25Z (box clock). Follows §2739 (ALL7_TOK_256 = .297, ALL7_CONST_256 = .397 — best extracted late stack; token filler worth .100 at k=256).

## Question
The seven-block late program "own weights on the block's top-k input PCs + filler" has two width knobs left unpriced: (1) k — how does
the stack improve beyond 256 (is .297 near a floor or halfway down a curve), and (2) the token map's rank — the TOK filler is a
rank-full D×D ridge read per block; if a rank-r read keeps its value the program's per-block price is k directions + r token dims + one
constant, not a full matrix. A third, descriptive knob: allocation (pool wide, last two narrower) vs uniform width at matched budget.

## Arms (all: own weights; each block its own centred input PCA + means on docs 96–191; eval docs 0–63; output unrestricted)
ALL7_TOK_k for k ∈ {128, 256, 384, 512, 768} (256 reproduces §2739). ALL7_CONST_k for k ∈ {512, 768}.
ALL7_TOK256_r for r ∈ {16, 64, 256}: k = 256, the token map A_l replaced by its rank-r SVD truncation (same means, same PCs).
POOL512+LAST2_256_TOK: pool blocks k = 512, mlp16/17 k = 256 (3072 directions total) — compare with ALL7_TOK_384 (2688) and _512 (3584).

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401 and ALL7_TOK_256 within .02 of §2739's .297.
- pred_b_all7_tok_512: ALL7_TOK_512 ≤ .20. Null: ≥ .28 (doubling the width from 256 buys under a third of the remaining .297).
- pred_c_all7_tok_768: ALL7_TOK_768 ≤ .12. Null: ≥ .20.
- pred_d_token_read_rank_256_enough: ALL7_TOK256_r256 − ALL7_TOK_256 ≤ .03. Null: ≥ .07.
- pred_e_token_read_is_high_rank: ALL7_TOK256_r64 − ALL7_TOK_256 ≥ .05 (a 64-dim token read loses at least half the .10 filler
  value). Null: ≤ .02 (the token read is essentially low-rank).
Descriptive (no bar): the CONST curve at 512/768; the allocation arm vs the uniform arms; the ALL7_TOK curve's doubling ratios.

## Price
96 fit docs + 64 × (1 + 11 arms) = 864 GPU document-forwards, ~15 s on the 5090. Frozen inputs: this file, §2739 results
(late_stack_constant_filler_probe_results.json), checkpoint, fit_natural.pt. Reproduction tolerance .02.
