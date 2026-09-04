# late_tail_token_fill_probe — preregistration (Registered 2026-09-04 00:27Z (box clock))

Lane 1 CUDA (Claude). Follows §2775 (the late-MLP width cost is concentrated on a decile of ordinary, FREQUENT targets — rare
targets carry 0.68× their share; it is not a rare-token dictionary). First vocabulary item for the tail: replace the truncated
768-complement of each late MLP's input by its ridge prediction from the CURRENT token's embedding (§2730 "tok" fill: fb = mx +
(e − me)·A_l, A_l fitted per late block on docs 96–191, LAM 1e-2), instead of the fit-set constant. If the tail the late MLPs
read is token-determined, a linear per-token fill recovers a large part of the .125.

Sign convention (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 — LOWER IS BETTER. Priors: LATE_MLP_768
= .1249, LATE_MLP_896 = .0662 (§2773).

Arms: SPLIT8_1024 (instrument), LATE_MLP_768 / LATE_MLP_896 (const fill, repro), LATE_MLP_768_TOK / LATE_MLP_896_TOK (token fill).
Recovered fraction f_k = (CONST_k − TOK_k) / CONST_k. Also measured (no CE): R²_tail(k) per late block on the eval docs = 1 −
E‖P_⊥(xhat − mx − (e − me)A)‖² / E‖P_⊥(xhat − mx)‖² with P_⊥ the projector off the top-k bus directions.

Frozen: this file, §2775 results (late_width_per_token_probe_results.json), checkpoint, fit_natural.pt.

- pred_a_instrument: baseline 3.0322401 within 1e-4; SPLIT8_1024 within .015 of .0374; LATE_MLP_768 within .015 of .1249.
- pred_b_token_fill_recovers_quarter_at_768: f_768 ≥ 0.25. Null: f_768 ≤ 0.05.
- pred_c_token_fill_recovers_quarter_at_896: f_896 ≥ 0.25. Null: f_896 ≤ 0.05.
- pred_d_token_fill_never_hurts: TOK_k ≤ CONST_k + .003 at both k. Null: TOK_768 ≥ CONST_768 + .010.
- pred_e_tail_is_partly_token_determined: mean over late blocks of R²_tail(768) ≥ 0.10. Null: ≤ 0.02.

Price: 2 fit passes (96 docs each) + 1 eval collection pass + baseline + 5 arms × 64 docs = 640 GPU document-forwards (≈ 25 s).
Descriptive; nothing installs into the §312 frontier; bases are data covariances scored by CE only (§2118 stays closed).
