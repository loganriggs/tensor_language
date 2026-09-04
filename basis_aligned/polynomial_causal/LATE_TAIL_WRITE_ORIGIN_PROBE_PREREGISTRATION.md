# late_tail_write_origin_probe — preregistration

Registered 2026-09-04 02:01Z (box clock). Claude, LANE 1 CUDA. Parent: late_tail_rewrite_chain_probe (§2795). Frozen inputs: this file,
late_tail_rewrite_chain_probe_results.json (§2795, sha 88f00e3c…), checkpoint blob 680d6c26…, fit_natural.pt 666a3201….

SIGN CONVENTION (§2135): every CE number is CE ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; fits on docs 96–191) —
LOWER IS BETTER. Descriptive; nothing installs into the §312 frontier; §2118 stays closed.

## Question

§2795: no late MLP's tail write is linear in its tail input (OOS R² ≈ .001). I read that as "the tail is re-generated from the core":
in the exact product split MLP(c+t) = MLP(c) + J(c)t + MLP(t) (§2791; c = bus-768 core part of the rms-normalised input, t = the 384-dim
tail part), the tail part of the write should be dominated by the core-only term MLP(c). This rung measures it directly and prices it:
which term writes the tail, and which term's tail write matters downstream? Block 17's .33 linear pass-through of block 16's tail (§2795)
predicts an elevated cross-term share there.

## Program

Blocks 8–17 MLP: compute the three terms exactly (verified max|Σ terms − MLP(c+t)| / max|MLP(c+t)| ≤ 1e-3 on every fit chunk — RELATIVE; amended from an absolute 1e-3 after the random-token smoke showed float32 accumulation error of 8e-3 absolute on writes of magnitude ~1e3, before enqueue, 02:03Z). Fit pass (docs
96–191): each term's tail-part sum (→ mean), centred tail-part energy, core-part energy. Arms (docs 0–63): DROP_CORE_TAILOUT,
DROP_CROSS_TAILOUT, DROP_TT_TAILOUT replace the tail part of that term's output by its fit-set mean at the write site in all ten late
MLPs (core parts kept, everything downstream including the unembedding sees the change); DROP_ALL_TAILOUT does it for all three at once
(= the whole late-MLP tail write mean-replaced). Instruments SPLIT8_1024, LATE_MLP_768. share_l(term) = centred tail energy of the term /
Σ_terms (per block); medians over blocks 8–17.

## Predictions (bars fixed before running)

* pred_a_instrument: baseline within 1e-4 of 3.0322401; SPLIT8_1024 / LATE_MLP_768 within .015 of .0374 / .1249; relative split error ≤ 1e-3.
* pred_b_core_term_writes_most_of_the_tail: median over blocks of share(core) ≥ .60. NULL: ≤ .35.
* pred_c_cross_term_writes_a_minority_of_the_tail: median share(cross) ≤ .35. NULL: ≥ .50.
* pred_d_cross_tail_write_cheaper_than_core_tail_write: CE(DROP_CROSS_TAILOUT) / CE(DROP_CORE_TAILOUT) ≤ .5. NULL: ≥ 1.0.
* pred_e_block17_cross_share_elevated: share_17(cross) / median_{8..16} share(cross) ≥ 2.0. NULL: ≤ 1.2.

Expected: b, c, d, e TRUE. If b is FALSE with c's null met (the cross term writes most of the tail), then the tail IS a transform of the
tail after all — but a gate-varying one that no fixed linear map captures — and §2795's "re-generated" reading is withdrawn in favour of
"re-written through a token-varying gate".

## Price

1 run: 96 fit forwards + 64 × 7 ≈ 550 GPU document-forwards, ≈ 25 s.
