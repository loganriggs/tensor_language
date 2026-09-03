# Preregistration — late_square_directions_ablation_probe

Registered 2026-09-03 22:04Z (box clock). Claude lane. Follows §2734 (identity of the shared square directions) and §2729/§2732
(exact 16-dim compile of mlp16/17 on a shared 8-dim square space). Nothing here installs into the §312 frontier.

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split; fits on docs
96-191) — LOWER IS BETTER.

## Question

§2734 showed the five shared square directions q_1..q_5 (= P u_j, u_j the top eigvectors of S16+S17) are what the compiled
mlp16/17 polynomial needs (PROG_SHARED5 .273 vs five random core directions median 1.76). That is a fact about the PROGRAM. Is it
a fact about the REAL blocks? I.e. does the real mlp16/17 pair read its CE-relevant input through the same five directions, and
in the same ORDER of importance? If yes, the square directions are a reusable component of the real model (compositional reuse
across two blocks, identified from the weights), not an artefact of the compile.

## Instrument (all arms patch mlp16 AND mlp17 together)

- REAL_ABL_q_j (j = 1..5): real block, real token path, input x̂ replaced by x̂ − (x̂·q_j) q_j (one core direction removed from the
  MLP input of both blocks; the residual stream itself is untouched). Also REAL_ABL_top5 (all five removed), REAL_ABL_core16 (the
  whole 16-dim core removed from the input), REAL_ABL_rand5_s (s = 0,1,2: a random 5-dim subspace of the core removed; seeded QR).
- PROG_SHARED8 (reference .246), PROG_SHARED8_minus_u_j (j = 1..5: Π built from the 8 shared directions with u_j dropped → rank 7),
  PROG_SHARED5 (reference .273), PROG_SHARED5_minus_u_j (rank 4).
- Derived: real per-direction cost r_j = CE(REAL_ABL_q_j); program per-direction cost p_j = CE(PROG_SHARED8_minus_u_j) − CE(PROG_SHARED8);
  rand_med = median_s CE(REAL_ABL_rand5_s); Spearman ρ(r, p) over the five directions.

## Predictions (bars fixed before the run)

- pred_a_instrument: baseline within 1e-4 of 3.0322401; PROG_SHARED8 within .02 of .246 (§2732); PROG_SHARED5 within .02 of .273 (§2734).
- pred_b_real_blocks_need_the_five: CE(REAL_ABL_top5) ≥ .30. Null: ≤ .10 (the real blocks route around five input directions).
- pred_c_five_beat_random_in_the_real_blocks: CE(REAL_ABL_top5) ≥ 2.0 × rand_med. Null: ≤ 1.2 ×.
- pred_d_same_order: ρ(r, p) ≥ .7 over the five directions (arms named above; five points, so this is coarse and stated as such).
  Null: ρ ≤ .3.
- pred_e_five_carry_half_the_core: CE(REAL_ABL_top5) / CE(REAL_ABL_core16) ≥ .5. Null: ≤ .25.

## Price

Fits (192 fit docs incl. filler pass) + 25 arms × 64 eval docs ≈ 1,800 GPU doc-forwards ≈ 40 s on the 5090. Output:
late_square_directions_ablation_probe_results.json. Frozen: this file, late_square_directions_identity_probe_results.json (§2734),
late_stack_extracted_program_probe_results.json (§2732), checkpoint, fit_natural.pt.

## What each outcome means

b,c,e TRUE + d TRUE: the shared square space is a real, ordered, reusable input component of the last two blocks — compositional
reuse identified from weights alone. b TRUE but d FALSE: the directions matter to the real blocks but the compile mis-orders them
(the program's importance profile is its own). b FALSE/null: the real blocks are robust to losing the five directions — then the
program's dependence on them is a property of the 16-dim restriction, not of the model, and "reusable component" is overstated.
