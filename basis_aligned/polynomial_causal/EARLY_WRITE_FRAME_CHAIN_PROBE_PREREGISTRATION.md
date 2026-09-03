# early_write_frame_chain_probe — preregistration (Registered Registered 2026-09-03 23:39Z (box clock))

Lane 1 CUDA (Claude). Follows §2759 (the early read-frame rotation is carried entirely by the MLP write step; the blend is inert)
and §2756 (blocks 8–17 read and write through one bus frame with the ≈7% remainder routed to the readout). Question: is the early
program a CHAIN of frames — each early MLP write lands in the NEXT site's read frame (the part of the write outside its own input
frame being exactly the part the next frame adopts), with the remainder outside the next frame either readout-bound or dead — or do
later blocks read what the next frame does not hold? And are attention writes, which rotate the frame less (§2759), more fully
inside the next frame than MLP writes?

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER; gaps "arm − EARLY22_OWN_768" = extra damage.

Construction: §2753's own top-768 input cores U_s (rms-normed inputs) and §2756's write statistics (mean μ_s, centred write
covariance Cw_s) for all sites, fit docs 96–191. Write routing as §2756: Route(s, U) splits the centred write into its part inside
span(U) and the remainder; 'delete' drops the remainder, 'readout' adds it to the residual just before the final norm (Buf/FinalAdd).
oe(s, U) = 1 − tr(UᵀCw_sU)/tr Cw_s (out-of-frame write energy). For l = 0..7: oe_own(l) = oe(mlp_l, U_mlp_l), oe_next(l) =
oe(mlp_l, U_attn_{l+1}); oe_attn_next(l) = oe(attn_l, U_mlp_l). All arms keep the 22 early sites reading through their own
768-frames (EARLY22_OWN_768, §2753) and touch only the writes of blocks 0–7:
MLPW_OWNFRAME_DEL_768 (mlp_0..7 writes projected onto their own input frame, remainder deleted);
MLPW_NEXTFRAME_DEL_768 (mlp_0..7 writes projected onto attn_{l+1}'s frame, remainder deleted);
MLPW_NEXTFRAME_RO_768 (same, remainder routed to the readout);
ATTNW_NEXTFRAME_DEL_768 (attn_0..7 writes projected onto mlp_l's frame, remainder deleted).

Arms: EARLY22_OWN_768, MLPW_OWNFRAME_DEL_768, MLPW_NEXTFRAME_DEL_768, MLPW_NEXTFRAME_RO_768, ATTNW_NEXTFRAME_DEL_768.

Frozen: this file, §2759 results (block_boundary_blend_rotation_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; EARLY22_OWN_768 within .02 of .057.
- pred_b_next_frame_adopts_the_write: oe_own(l) > oe_next(l) for all 8 early MLP writes AND median_l oe_own(l)/oe_next(l) ≥ 1.5.
  Null: median ratio ≤ 1.1.
- pred_c_next_frame_is_the_writes_home: (MLPW_NEXTFRAME_DEL − OWN) ≤ 0.5 × (MLPW_OWNFRAME_DEL − OWN). Null: ≥ 0.9 ×.
- pred_d_early_chain_with_readout_is_cheap: MLPW_NEXTFRAME_RO_768 − EARLY22_OWN_768 ≤ .040. Null: ≥ .120.
- pred_e_attention_writes_sit_inside_the_next_frame: (ATTNW_NEXTFRAME_DEL − OWN) ≤ 0.5 × (MLPW_NEXTFRAME_DEL − OWN)
  AND median_l oe_attn_next(l) ≤ median_l oe_next(l). Null: CE ratio ≥ 1.0.

Price: 1 fit pass (96 docs) + baseline + 5 arms × 64 docs = 480 GPU document-forwards (≈ 25 s). Descriptive; nothing installs into
the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
