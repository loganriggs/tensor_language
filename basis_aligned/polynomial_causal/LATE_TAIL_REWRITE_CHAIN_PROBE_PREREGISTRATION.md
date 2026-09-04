# late_tail_rewrite_chain_probe — preregistration

Registered 2026-09-04 01:57Z (box clock). Claude, LANE 1 CUDA. Parent: late_tail_writer_pair_coherence_probe (§2794). Frozen inputs: this
file, late_tail_writer_pair_coherence_probe_results.json (§2794, sha ccc0b632…), checkpoint blob 680d6c26…, fit_natural.pt 666a3201….

SIGN CONVENTION (§2135): CE numbers are CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split) — LOWER IS BETTER. The R²
values below are out-of-sample explained variance (fits on docs 96–191, scored on docs 0–63); they are NOT CE numbers. Descriptive; nothing
installs into the §312 frontier; §2118 stays closed.

## Question

§2794: the late writers' tail contributions are coherent in the loss metric with a cosine κ that decays with distance (.20 adjacent → .02
at distance 8) — a chain — while their plain input-space cosines are ≈ .01. I attributed the chain to §2791's re-write (each late MLP
writes ~30% of its read output into the tail, so writer j+1's tail content is partly a transform of writer j's). That is a testable
linear-transfer claim: how much of MLP l's tail write is an (out-of-sample) linear function of writer j's tail component as l reads it?

## Program (no drops)

Exact forward with the §2793 Tracker (λ-propagated per-writer components c_j). At each late MLP l: Y = its write in tail coordinates
(out @ Ut, 384 dims); X_j = (c_j·scale) @ Ut for every j < l (45 reader–writer pairs); X_full = xh @ Ut; X_prev = X_{l−1}. Centred
second moments on the fit set (docs 96–191) give a closed-form ridge fit (λ = 1e-2 × mean eigenvalue of XᵀX); the same moments on the
eval set (docs 0–63) give the out-of-sample R² (mean model = fit-set mean). PRIOR κ from §2794 is frozen in the script:
PRIOR_KAPPA = {"8_9": 0.1413, "8_10": 0.1321, "8_11": 0.0886, "8_12": 0.0742, "8_13": 0.1033, "8_14": 0.116, "8_15": 0.0612, "8_16": 0.0215, "9_10": 0.1906, "9_11": 0.1913, "9_12": 0.1083, "9_13": 0.0883, "9_14": 0.105, "9_15": 0.0732, "9_16": 0.0391, "10_11": 0.1787, "10_12": 0.169, "10_13": 0.1655, "10_14": 0.1609, "10_15": 0.0902, "10_16": 0.0528, "11_12": 0.2033, "11_13": 0.1688, "11_14": 0.1675, "11_15": 0.1335, "11_16": 0.048, "12_13": 0.2133, "12_14": 0.2045, "12_15": 0.1557, "12_16": 0.0685, "13_14": 0.3061, "13_15": 0.203, "13_16": 0.1355, "14_15": 0.2973, "14_16": 0.1666, "15_16": 0.1451}

## Predictions (bars fixed before running)

* pred_a_instrument: baseline within 1e-4 of 3.0322401; SPLIT8_1024 and LATE_MLP_768 within .015 of .0374 / .1249.
* pred_b_transfer_falls_with_distance: Spearman(distance l−j, R²_{l←j}) over the 45 pairs ≤ −.4. NULL: ≥ 0.
* pred_c_adjacent_transfer_dominates_far: median R² at distance 1 ≥ 2.0 × median R² at distance ≥ 5. NULL: ≤ 1.2.
* pred_d_transfer_tracks_loss_coherence: Spearman(R²_{k←j}, κ_{j,k}) over the 36 pairs with j < k ≤ 16 ≥ .5. NULL: ≤ .1.
* pred_e_tail_write_is_mostly_linear_in_tail_input: median over readers of the out-of-sample R² of the tail write on the FULL tail input
  ≥ .5 (§2791: the read is J(c)·t, linear in t at fixed gate; the linear-in-t-only fit measures how much the gate varies). NULL: ≤ .2.

Expected: b, c, d TRUE; e uncertain (my guess .4–.6). If b/c/d are TRUE but e is low, the chain is real but carried by a gate-varying
map; if d is FALSE while b/c are TRUE, the transfer exists but is not what the loss metric aligns — the coherence would then have to come
from the readers' shared output frame (§2791's 70%-core output) rather than from content propagation.

## Price

1 run: 96 + 64 recording forwards + 64 × 3 instrument forwards ≈ 350 GPU document-forwards, ≈ 25 s.
