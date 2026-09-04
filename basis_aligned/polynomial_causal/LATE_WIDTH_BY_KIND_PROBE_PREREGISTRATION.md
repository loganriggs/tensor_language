# late_width_by_kind_probe — preregistration (Registered 2026-09-04 00:15Z (box clock))

Lane 1 CUDA (Claude). Follows §2771 (blocks 8–17 lose .137–.164 at k = 768 whether through own or shared frames; the loss
compounds through the settled region). Which reads need the width — the late ATTENTION reads (Q/K/V, per-head 128-dim
bottlenecked, §2679) or the late MLP reads (bilinear, full-width, §2673–§2676)? Arms constrain only one kind of late read
through the bus frame U_8 at k ∈ {768, 896}, the other kind and all early sites untouched.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Priors: BUS_768 (both kinds) = .1636 (§2770/§2771).

Arms: SPLIT8_1024 (instrument), BUS_768 (repro), LATE_ATTN_768, LATE_MLP_768, LATE_ATTN_896, LATE_MLP_896, BUS_896 (both kinds at 896).

Frozen: this file, §2771 results (late_width_control_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; BUS_768 within .015 of .1636.
- pred_b_late_mlp_reads_carry_the_width: LATE_MLP_768 ≥ 2 × LATE_ATTN_768. Null: LATE_ATTN_768 ≥ LATE_MLP_768.
- pred_c_late_attention_reads_cheap_at_768: LATE_ATTN_768 ≤ .040. Null: ≥ .100.
- pred_d_kinds_are_subadditive: LATE_ATTN_768 + LATE_MLP_768 ≤ 1.2 × BUS_768 (the two kinds read the same truncated stream; §2771's
  compounding is across blocks, not kinds). Null: ≥ 2 × BUS_768.
- pred_e_bus_896_under_p06: BUS_896 ≤ .060 (the late cliff lies between 768 and 896 as the whole-model curve .197/.096/.034
  suggests; then blocks 8–17 need ≈ 896 dimensions). Null: ≥ .120.

Price: 1 fit pass (96 docs) + baseline + 7 arms × 64 docs = 608 GPU document-forwards (≈ 25 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
