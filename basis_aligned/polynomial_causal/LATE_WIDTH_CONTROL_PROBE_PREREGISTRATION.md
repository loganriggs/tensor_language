# late_width_control_probe — preregistration (Registered 2026-09-04 00:10Z (box clock))

Lane 1 CUDA (Claude). INDEPENDENT PHYSICAL CONTROL for a correction. early_block_read_cost_map_probe (landed 00:08Z, to be
§2770) found that the k = 768 read cost of program v2 is NOT in the early blocks (each block .002–.006, sum .030) but in the BUS
(blocks 8–17 reading through one 768-frame: .164 of the joint .225). That contradicts the interpretive sentences of §2764(3) and
§2769(1)/(3) ("the early frames are the price cliff / the residual cost is the early READ cost"). Before that correction is
published, this rung asks whether the late cost is a property of the SHARED bus frame or of the late blocks' width use itself:
each late site reading through its OWN 768-frame (no sharing), the early sites through their own 768-frames (no sharing), and
the bus cost mapped per late block.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER.

Arms: SPLIT8_1024 (instrument), BUS_768 (repro of .1636), LATE_OWN_768 (blocks 8–17 each through its own site frame at 768,
early untouched), EARLY_OWN_768 (blocks 0–7 each through its own site frame at 768, late untouched), ONE_L<l>_768 for l = 8..17
(block l's attention and MLP through U_8 at 768, all else untouched).

Frozen: this file, early_block_read_cost_map_probe_results.json, checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; BUS_768 within .015 of .1636.
- pred_b_late_width_use_is_real_not_a_sharing_artifact: LATE_OWN_768 ≥ .100 (own frames remove the sharing; §2752's all-own
  program at 768 was .197). Null: ≤ .050 (then the bus cost is the sharing, and the correction is NOT warranted as stated).
- pred_c_early_own_frames_cheap_at_768: EARLY_OWN_768 ≤ .050. Null: ≥ .100.
- pred_d_late_per_block_costs_are_additive: Σ_{l=8..17} c_l / BUS_768 in [0.5, 1.5]. Null: ≥ 3 or ≤ 0.25.
- pred_e_late_cost_is_spread: the three largest late c_l carry ≤ 60% of Σ c_l (ten settled blocks share the width use). Null: ≥ 80%.

Price: 1 fit pass (96 docs) + baseline + 14 arms × 64 docs = 1056 GPU document-forwards (≈ 40 s). Descriptive; nothing installs
into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
