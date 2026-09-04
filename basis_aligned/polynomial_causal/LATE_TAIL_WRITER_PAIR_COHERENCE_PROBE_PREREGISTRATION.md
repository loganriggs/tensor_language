# late_tail_writer_pair_coherence_probe — preregistration

Registered 2026-09-04 01:51Z (box clock). Claude, LANE 1 CUDA. Parent: late_tail_writer_identity_probe (§2793). Frozen inputs: this file,
late_tail_writer_identity_probe_results.json (§2793, sha b2908e2a…), checkpoint blob 680d6c26…, fit_natural.pt 666a3201….

SIGN CONVENTION (§2135): every CE number below is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs
96–191) — LOWER IS BETTER. Descriptive. Nothing installs into the §312 frontier; §2118 stays closed.

## Question

§2793: no single late writer's tail is worth more than 8% of the joint late-origin tail cost (.0711), and the ten single-writer drops sum
to .52 of it. Under §2788's quadratic law (CE added ≈ δᵀHδ for a small input perturbation δ), the joint cost of dropping writers j and k
together is cost_j + cost_k + 2·κ_jk·√(cost_j·cost_k), where κ_jk is the cosine of the two writers' (centred, tail-projected) contributions
in the H-metric. Orthogonal contributions give Σ singles / joint = 1; perfectly aligned ones give 1/N. This rung measures κ_jk for every
pair of writers 8..16 (36 pairs; writer 17 costs .0001 and is excluded from the grid) and asks four things:

* is the coherence generic (most pairs super-additive)?
* does it CONCENTRATE ON ADJACENT WRITERS — the re-write chain (§2791: each late MLP re-writes ~30% of its read into the tail, so writer
  j+1's tail is partly a function of writer j's) — or is it FLAT across distance (a shared direction all writers push)?
* do pairwise terms reconstruct the nine-writer joint (is the quadratic law pairwise-complete), or do higher-order terms dominate?
* does loss-metric coherence track the writers' plain centred input-space cosine (pooled over the readers that see both)?

## Program (identical readers to §2793)

Blocks 8–17 MLP reads: core (top-768 of U_8) exact; from the tail, the λ-propagated writes (attention + MLP + Down bias) of the writers in
the window are removed from every downstream reader and replaced by their fit-set means. Windows: DROP_ALL (writers 8–17), D_8_16
(writers 8–16), W8…W16 (singles), W8_REPEAT (W8 again in the same process — the within-run noise floor), P{j}_{k} for 8 ≤ j < k ≤ 16.
Fit pass additionally records, per reader, the writer × writer Gram of the tail-projected components; the centred input-space cosine
cos_jk pools readers l ≥ k. cross_jk = cost_jk − cost_j − cost_k; κ_jk = cross_jk / (2√(cost_j cost_k)); d = k − j.

## Predictions (bars fixed before running)

* pred_a_instrument: baseline within 1e-4 of 3.0322401; SPLIT8_1024, LATE_MLP_768, DROP_ALL within .015 of §2793 (.0374 / .1249 / .0711);
  every single-writer cost W8…W16 within .003 of §2793's (.00592 .00425 .00566 .00508 .00413 .00577 .00333 .00163 .00110).
* pred_b_pairs_generically_super_additive: fraction of the 36 pairs with cross_jk > 0 is ≥ .75. NULL (independent wires): ≤ .50.
* pred_c_coherence_falls_with_distance: Spearman(d, κ_jk) over the 36 pairs ≤ −.4 (arm: the re-write chain). NULL (flat shared
  direction): ≥ 0.
* pred_d_adjacent_writers_coherent: median κ over the 8 adjacent pairs (d = 1) ≥ .20. NULL: ≤ .05.
* pred_e_pairwise_quadratic_law_reconstructs_the_joint: [Σ_j cost_j + Σ_{j<k} cross_jk] / cost(D_8_16) ∈ [.75, 1.33]. NULL (higher-order
  terms dominate): ≤ .5 or ≥ 2.
* pred_f_loss_coherence_tracks_input_cosine: Spearman(κ_jk, cos_jk) over the 36 pairs ≥ .5. NULL: ≤ .1.

Expected under my reading of §2791/§2793: b TRUE, c TRUE (κ ≈ .3–.5 adjacent, ≈ 0 at d ≥ 5), d TRUE, e TRUE, f TRUE. A c-null result
(flat κ) would mean the writers share a common loss-relevant direction rather than a chain, and would redirect the next rung to that
direction (a single tail vector all late writers push).

## Noise

Within-run CUDA wobble is reported as |W8_REPEAT − W8|; individual cross terms of order .001 are near this floor, which is why b, c, d,
f are aggregates over 36 pairs and e is a sum. No MC sampling; eval set is fixed (docs 0–63).

## Price

1 run, ~50 arms × 64 docs + 2 × 96 fit docs ≈ 3,400 GPU document-forwards, ≈ 70 s on the RTX 5090.
