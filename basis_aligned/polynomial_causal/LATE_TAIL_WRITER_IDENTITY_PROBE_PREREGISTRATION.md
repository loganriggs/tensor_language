# late_tail_writer_identity_probe — preregistration (Registered 2026-09-04 01:42Z (box clock); identification note added before enqueue)

Lane 1 (Claude). Parent: ops/late_tail_writer_recency_probe.py (§2790). Prior results frozen: late_tail_read_operator_rank_probe_results.json
(§2791). Script: bilinear_quotient/ops/late_tail_writer_identity_probe.py. Results: bilinear_quotient/late_tail_writer_identity_probe_results.json.
This is the CONTROL registered in §2792 for §2790's interpretation.

SIGN CONVENTION (§2135): every CE number below is CE ADDED ABOVE THE REAL MODEL on held-out docs 0-63 (FRESH split) — LOWER IS BETTER.

## Question
§2790 measured the late-origin tail cost by reader–writer DISTANCE windows and read a "fade ≈ .8 per block" off the pooled per-distance
energy profile. §2792 recorded that this profile is confounded with writer identity (large distances sample only the earliest late
writers) and that the checkpoint's λ0 for blocks 9–17 is .88–1.23 (block 8: 1.41) — no architectural fade. This rung separates the two:
per-WRITER drops and the full reader × writer energy matrix with the distance slope fitted with writer fixed effects.

## Method (exact decomposition, as §2790)
x = y + Σ_j c_j at every late block (c_j = λ0-propagated write of late block j: attention at ('attn', j), MLP + Down_bias at ('mlp', j)).
Arm W_j: every late reader l ≥ j drops c_j from its tail (xp = xh − perp(c_j·scale − mean_fit(c_j·scale))); reader j itself drops only
block j's attention write. Instruments: SPLIT8_1024, LATE_MLP_768, DROP_ALL (= §2790 .0711), D_LE2 (= §2790 .0302).
Fit pass (docs 96–191): per-(reader, writer) means; E_lj = per-token tail energy ‖perp(c_j·scale)‖² at reader l.
β_FE = within-writer least-squares slope of log E_lj on d = l − j over pairs with d ≥ 1 (writers with ≥ 2 downstream readers);
β_pooled = the same slope without fixed effects (the §2790 reading). Writer energy per reader = mean over l > j of E_lj (writers 8–16);
pooled writer energy = Σ_{l ≥ j} E_lj (writers 8–17).

Identification note (written before the run, after a random-token smoke): with writer fixed effects, β_FE measures the per-block fade of a
write IN THE READER'S NORMALISED INPUT. Raw writes only grow (λ0 ≥ .88); the only fade mechanism left is the residual norm growing under the
reader's rms_norm, i.e. retention_l = λ0_l² · ‖x_{l−1}‖²/‖x_l‖², which the fit pass records per block (reader_rms_x, predicted retention,
and β_FE minus its log-mean as a consistency check). A reader-norm effect and "time" are not separately identifiable from the energy matrix
(d = l − j is collinear with two-sided fixed effects); pred_b is therefore a claim about the fade the reader SEES, which is the quantity the
program needs. The smoke on random tokens gave β_FE = −.136; the real-text value is what is registered.

## Predictions (scored exactly as written)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; SPLIT8_1024 .0374, LATE_MLP_768 .1249, DROP_ALL .0711, D_LE2 .0302 within 0.015.
- pred_b_no_true_fade_with_writer_fixed_effects: β_FE ≥ −0.11 (per-block energy retention ≥ .90 once the writer is held fixed).
  Null: β_FE ≤ −0.22 (retention ≤ .80 — §2790's fade was real).
- pred_c_later_writers_write_more_tail: Spearman(writer block j, writer energy per reader) ≥ 0.6 over writers 8–16. Null: ≤ 0.2.
- pred_d_single_writer_cost_tracks_energy: Spearman(CE(W_j), pooled writer energy) ≥ 0.6 over writers 8–17. Null: ≤ 0.2.
- pred_e_single_writers_sum_near_the_whole: 0.5 ≤ Σ_j CE(W_j) / CE(DROP_ALL) ≤ 1.2 (§2790's sub-additivity .74 for two windows suggests ≤ 1).
  Null: ≥ 1.5.
Also recorded (no bar): the largest single writer's share of DROP_ALL; λ0 per late block.

BARS = {"ce_tol": 1e-4, "repro_tol": 0.015, "b_beta_min": -0.11, "c_rho_min": 0.6, "d_rho_min": 0.6, "e_lo": 0.5, "e_hi": 1.2}
NULLS = {"b_beta_max": -0.22, "c_rho_max": 0.2, "d_rho_max": 0.2, "e_hi": 1.5}

## Price
GPU: 1 fit pass (96 docs) + baseline + 14 arms × 64 docs ≈ 1150 doc-forwards; ~35 s on the 5090. Nothing installs into the §312 frontier.

## What would change my mind
b null met → §2790's fade was a real decay in time despite λ0 ≥ 1 (the readers' normalisation or the writers' geometry must then be doing
it); the §2792 caveat is withdrawn and §2790's reading restored. c and d nulls → the tail channel is writer-agnostic — a genuine bus.
