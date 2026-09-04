# TAIL_READ_OUTPUT_FRAME_PROBE — preregistration (Registered 2026-09-04 01:00Z (box clock))

Claude, LANE 1 (CUDA). Script ops/tail_read_output_frame_probe.py, derived from ops/all18_tail_linear_program_probe.py (§2784, results
sha 8b4fc555… frozen as PRIOR). FRESH split: fits docs 96–191, eval docs 0–63. Sign convention (§2135): CE ADDED ABOVE THE REAL MODEL —
LOWER IS BETTER.

## Question
The MLPs read their low-variance tail through a core-gated linear map M(c)·t = Down[Lc∘Rt + Lt∘Rc] (§2780, §2783). Where does that read
WRITE — back into the high-variance frame the next readers use (a bus contribution), or into the low-variance complement (a side channel,
cf. §2756's readout remainder)? Split the cross term's output by the reader's own read frame; keep each part alone.

## Arms (tail×tail dropped throughout; everything else exact)
- late (blocks 8–17, bus U_8 top-768): LATE_MLP_768 (cross dropped; .1249), LATE_DROP_TT_768 (cross whole; .0087), LATE_CROSS_IN_768
  (cross output projected into the 768 frame), LATE_CROSS_PERP_768 (only the 384-complement part)
- early (blocks 0–7, own top-512 frames): EARLY_MLP_512 (.0979), EARLY_DROP_TT_512 (.0091), EARLY_CROSS_IN_512, EARLY_CROSS_PERP_512
- recovery(arm) = (MLP − arm) / (MLP − DROP_TT); fit-set energy share of the cross output inside the frame (late 768, early 512)

## Predictions (bars literal; scored exactly as written)
- pred_a_instrument: baseline 3.0322401 ± 1e-4; SPLIT8_1024, LATE_DROP_TT_768, LATE_MLP_768 within .015 of prior.
- pred_b_in_frame_output_carries_most_late: recovery(LATE_CROSS_IN_768) ≥ 0.60. NULL: ≤ 0.35.
- pred_c_out_of_frame_output_alone_small_late: recovery(LATE_CROSS_PERP_768) ≤ 0.40. NULL: ≥ 0.60.
- pred_d_parts_near_additive_late: recovery(IN) + recovery(PERP) ∈ [0.7, 1.3]. NULL: ≤ 0.5.
- pred_e_in_frame_output_carries_most_early: recovery(EARLY_CROSS_IN_512) ≥ 0.60. NULL: ≤ 0.35.

## Price
2 fit passes' worth (heads + energies, 96 docs each) + 64 eval docs × 10 forwards ≈ 832 GPU document-forwards; ≈ 25 s.
