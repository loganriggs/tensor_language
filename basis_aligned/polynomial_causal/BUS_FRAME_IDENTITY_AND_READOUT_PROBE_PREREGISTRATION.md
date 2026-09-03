# bus_frame_identity_and_readout_probe — preregistration (Registered 2026-09-03 23:22Z (box clock))

Lane 1 CUDA (Claude). Follows §2754/§2756 (blocks 8–17 read and write through ONE 1024-frame U_8, the top-1024 of their 20 sites'
averaged input covariance) and §2757 (block 17 rotates away from U_8). Two questions: (i) is the bus frame a property of the late
half or of the averaging — does the late-7 frame (blocks 11–17, §2745's core widened) or the last-4 frame (blocks 14–17) serve
blocks 8–17 equally well? (ii) does the UNEMBED read through the bus — what does it cost to project the final rms-normed
residual onto U_8 before the lm_head, versus onto the final input's own top-1024 frame?

Sign convention (§2135): every number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits docs 96–191) —
LOWER IS BETTER. Gaps are "arm − reference" = extra damage.

Construction: §2754's SPLIT8 read program at k = 1024 (blocks 0–7 own cores; blocks 8–17 one shared core). Shared-core variants:
U_8 (blocks 8–17, 20 sites), U_L7 (blocks 11–17, 14 sites), U_L4 (blocks 14–17, 8 sites), each the top-1024 eigenvectors of the
plain average of the named sites' centred rms-normed input covariances. FINAL read patch (new site, applied to the residual just
before the final rms-norm): x̂ = rms_norm(x); x̂ ← x̄_f + U Uᵀ(x̂ − x̄_f) with x̄_f the fit-set mean of the final rms-normed residual;
the model's final rms-norm then re-normalises it before lm_head. U ∈ {U_8[:, :1024], U_f[:, :1024]} where U_f is the top-1024 of the
final input's own covariance C_f. Statistics: capture(C, U) = tr(UᵀCU)/tr C for C ∈ {C_f, C_U = W_Uᵀ W_U (the unembed's Gram)} and
U ∈ {U_8, U_L7, U_f}; principal angles between U_8 and U_L7 at 1024 (≤ 128 can be non-zero).

Arms: SPLIT8_1024 (inst, prior .0374), SPLIT8_L7FRAME_1024, SPLIT8_L4FRAME_1024, FINAL_ON_BUS_1024, FINAL_OWN_1024.

Frozen: this file, §2756 results (settled_frame_bus_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374.
- pred_b_bus_is_the_late_frame: SPLIT8_L7FRAME_1024 − SPLIT8_1024 ≤ .005. Null: ≥ .020.
- pred_c_last4_frame_still_serves: SPLIT8_L4FRAME_1024 − SPLIT8_1024 ≤ .015. Null: ≥ .040.
- pred_d_unembed_reads_through_the_bus: FINAL_ON_BUS_1024 ≤ .030. Null: ≥ .080.
- pred_e_bus_is_nearly_the_readout_frame: FINAL_ON_BUS_1024 − FINAL_OWN_1024 ≤ .015. Null: ≥ .050.

Price: 1 fit pass (96 docs) + baseline + 5 arms × 64 docs = 480 GPU document-forwards (≈ 20 s). Descriptive; nothing installs
into the §312 frontier.
