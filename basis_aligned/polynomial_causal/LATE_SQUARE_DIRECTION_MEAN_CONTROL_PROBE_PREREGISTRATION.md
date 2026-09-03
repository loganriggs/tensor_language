# Preregistration — late_square_direction_mean_control_probe

Registered 2026-09-03 22:09Z (box clock). Claude lane. The control §2736 named: its ablation x̂ ← x̂ − (x̂·q_j) q_j removed the MEAN
of the coordinate x̂·q_j along with its per-token variation. Nothing here installs into the §312 frontier.

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split; fits docs 96-191)
— LOWER IS BETTER.

## Question

§2736: zeroing q₁ from the input of the real mlp16/17 costs 2.00 nats. Is that (H_mean) the loss of a CONSTANT the bilinear form
uses as its effective linear term — x̂·q₁ has a large mean m₁ and A[q₁,·]·m₁ is a linear read of the rest of the input — or
(H_info) the loss of per-token INFORMATION carried on q₁? The two make opposite predictions for a mean-preserving ablation.

## Instrument (all arms patch mlp16 AND mlp17 together)

- Stats on the fit set, per block l ∈ {16,17}: m_{l,j} = mean(x̂_l·q_j), σ_{l,j} = std(x̂_l·q_j) for j = 1..8; the "mean dominance"
  m²/σ² per direction; also for comparison the same for the whole 16-dim core (‖mean core coeff‖² vs total core variance).
- REAL_MEANABL_q_j (j = 1..5): x̂ ← x̂ + (m_{l,j} − x̂·q_j) q_j — the coordinate is pinned to its fit-set mean; variation removed, mean kept.
  REAL_MEANABL_top5, REAL_MEANABL_core16, REAL_MEANABL_rand5_s (s = 0,1,2; the seeded 5-dim core subspaces of §2736, mean-pinned).
- REAL_ZERO_q1 (the §2736 arm, re-run as the anchor, expected 2.003 ± .05).
- REAL_MEANONLY_q1: x̂ ← x̂ − m_{l,1} q₁ — the variation is kept, only the constant removed (the complementary half of the anchor).
- PROG_SHARED8 (reference .246) and PROG_SHARED8_meanpin_u_j (j = 1..5): the compiled program with c_j pinned to its fit-set mean.
- Derived: mean fraction f₁ = CE(REAL_MEANONLY_q1) / CE(REAL_ZERO_q1); info fraction g₁ = CE(REAL_MEANABL_q1) / CE(REAL_ZERO_q1);
  rand_med = median_s CE(REAL_MEANABL_rand5_s).

## Predictions (bars fixed before the run)

- pred_a_instrument: baseline within 1e-4 of 3.0322401; PROG_SHARED8 within .02 of .246; REAL_ZERO_q1 within .05 of 2.003.
- pred_b_constant_carries_most: CE(REAL_MEANABL_q1) ≤ 1.0 (mean-preserving removal costs at most half of the zeroing cost). Null: ≥ 1.6.
- pred_c_q1_is_mean_dominated: m²/σ² for q₁ ≥ 2.0 in both blocks. Null: ≤ 1.0 in either block.
- pred_d_q1_still_informative: CE(REAL_MEANABL_q1) ≥ .10 (its per-token variation still matters more than any other single direction
  did in §2736, max .093). Null: ≤ .03.
- pred_e_five_beat_random_mean_preserved: CE(REAL_MEANABL_top5) ≥ 2.0 × rand_med. Null: ≤ 1.2 ×.

## Price

Fits (192 fit docs) + 17 arms × 64 eval docs ≈ 1,300 GPU doc-forwards ≈ 30 s on the 5090. Output:
late_square_direction_mean_control_probe_results.json. Frozen: this file, late_square_directions_ablation_probe_results.json (§2736),
late_stack_extracted_program_probe_results.json (§2732), checkpoint, fit_natural.pt.

## What each outcome means

b,c TRUE: q₁'s 2.0 nats is mostly the constant — the last two blocks' bilinear form uses a fixed stream offset along q₁ as a linear
term; the "critical direction" is a bias carrier, and the program's dependence on u₁ is the same fact. d TRUE on top: q₁ ALSO carries
per-token information beyond every other direction. b FALSE/null: the variation itself is critical — q₁ is a genuine information
channel into the last two blocks, and the mean is incidental.
