# Preregistration — MLP16/MLP17 low-rank write surrogate (held-out CE ADDED) — Claude CPU lane

Registered 2026-09-03 15:21 UTC (system clock), BEFORE running. Script: `ops/mlp_final_blocks_low_rank_surrogate_probe.py`.
Results: `mlp_final_blocks_low_rank_surrogate_probe_results.json`. Price: CPU only, 0 GPU forwards, ~2,900 CPU document
forwards (256 tokens each), ~12-15 min. Follow-up to §2692 (in-situ MLP-write effective rank 9.4 / 6.2 in blocks 16 / 17).

SIGN CONVENTION (§2135): all CE figures are CE ADDED ABOVE THE REAL MODEL on held-out documents — LOWER IS BETTER.

## Instrument
Natural rows (fit_natural, 192 docs): docs 0-95 = FIT half, docs 96-191 = EVAL half; code rows docs 96-191 for transfer.
For each of blocks 16 and 17, the centred covariance of the MLP write (all 256 positions, 24,576 samples) on the FIT half gives
mean mu and eigenbasis U. Surrogate rank k: write' = mu + U_k U_k^T (write - mu); k = 0 replaces the write by its mean. The
patched forward is the tt_model-semantics manual forward (§2692 instrument, CE match <= 1e-4 checked again here). CE is over
all 256 next-token targets of the EVAL half. Ladder k in {0, 1, 2, 4, 8, 16, 32, 64} per block, and both blocks at k = 8 / 32.
Entropy diagnostic: per-token coefficient c1 = u1 . (write17 - mu17) on the EVAL half vs the real model's next-token entropy at
the same positions, Spearman (sign of u1 is a gauge; |rho| is scored).

## Predictions (scored exactly as written)
- pred_a_instrument: manual CE == module CE within 1e-4 on 4 EVAL docs.
- pred_b_low_rank_replicates_held_out: effective rank of the EVAL-half centred write covariance <= 15 (block 17) AND <= 25
  (block 16) — i.e. §2692's collapse is not a fit-half accident (§2692 used 64 positions/doc of all 192 docs). Null: either >= 50.
- pred_c_mlp17_rank8_cheap: rank-8 surrogate of MLP17's write adds <= .02 nat CE on the EVAL half. Null: >= .10.
- pred_d_mlp16_rank8_cheap: rank-8 surrogate of MLP16's write adds <= .05. Null: >= .20.
- pred_e_both_rank8_cheap: both blocks at rank 8 simultaneously add <= .08. Null: >= .30.
- pred_f_top_direction_tracks_entropy: |Spearman(c1, entropy)| >= .5 (the "scale/confidence controller" reading of §2692).
  Null: <= .2.

## Reading rules
c-e decide whether the last two MLPs admit a rank-8 down-projection as an honest smaller-program component (the product state
is left intact; only the 1152-dim write is projected). f tests the mechanism hypothesis; a FALSE f with TRUE c-e means "low-rank
but not (only) a temperature controller". The k = 0 rows (mean replacement) are disclosed to show how much of the write's role is
its constant part. Code transfer at k = 8 (natural-fitted basis) is disclosed, not scored. Failures preserved. No explained-
fraction change is claimed from this probe (a follow-up would have to install the surrogate under the frontier's accounting).
Frozen inputs: §2692 results 63483cec..., checkpoint 680d6c26..., row caches 666a3201... / 6cf514e7....
