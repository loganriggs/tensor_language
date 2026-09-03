# Anatomy of the late-stack composition penalty (π = .180, §2732): truncation vs interface, core-borne vs perpendicular, per pool block, and co-adaptation — preregistration

Registered 2026-09-03 21:50Z (box clock), before the script exists. Lane 1 (CUDA). SIGN CONVENTION (§2135): CE numbers are CE
ADDED ABOVE THE REAL MODEL on held-out docs 0–63 (FRESH split; heads/fillers on docs 96–191; baseline 3.0322401) — LOWER IS
BETTER. π(X, Y) = CE(X + Y) − CE(X) − CE(Y). Descriptive; nothing installs into §312.

## Question
§2732: pool OWN_32_TOK (.319) + the mlp16/17 shared-square program PROG (.246) compose to .745, π = .180, three times the fitted
stack's .066 (§2725). Where does the .18 live? Four separable hypotheses, each with its own arm: (1) it is the program's
TRUNCATION (square space 8 / read rank 8 / no offset) being brittle → replace PROG by the FULL exact 16-dim compile (own weights
on the core + token filler, output restricted to the core; = §2727's COMPILED_TOK) and re-measure π; (2) it is CORE-BORNE — the
pool's write error lands in the 16 core directions the program squares → GUARD arm (pool OWN_32 write with its core component
replaced by the real write's: w′ = w_own + PPᵀ(w_real − w_own)) and the converse COREERR arm (w′ = w_real + PPᵀ(w_own − w_real));
(3) it is concentrated in particular pool blocks → each block truncated alone, with and without PROG, π_l; (4) it is a FITTING
artefact of the filler → CO-ADAPT: refit the mlp16/17 fillers (x̄_⊥, A_fill) on the pool-perturbed stream and re-measure COMBINED.

## Arms (everything else real)
MEAN7 · POOL (= OWN_32_TOK, ref .319) · PROG (ref .246) · FULL (16-dim exact compile of mlp16/17, ref §2727 .233 ± .02) · COMBINED
(ref .745) · COMBINED_FULL = POOL + FULL · POOL_GUARD · POOL_GUARD + PROG · POOL_COREERR · POOL_COREERR + PROG · POOL_l (l = 11…15, one
block truncated) · POOL_l + PROG · COMBINED_COADAPT. Derived: π = π(POOL, PROG); π_full = π(POOL, FULL); π_guard = π(POOL_GUARD, PROG);
π_coreerr = π(POOL_COREERR, PROG); π_l = π(POOL_l, PROG); Σπ_l.

## Predictions (bars fixed now)
- **pred_a_instrument**: baseline within 1e-4; MEAN7 within .03 of 1.885; POOL within .02 of .319; PROG within .02 of .246; COMBINED
  within .02 of .745; FULL within .03 of .233.
- **pred_b_penalty_is_interface_not_truncation**: π_full ≥ .12 (two thirds of .180 survives when the program is exact). Null: π_full ≤ .06.
- **pred_c_guarding_the_core_removes_it**: π_guard ≤ .06. Null: π_guard ≥ .14.
- **pred_d_core_error_alone_carries_it**: π_coreerr ≥ .10. Null: π_coreerr ≤ .04.
- **pred_e_coadaptation_does_not_repair_it**: CE(COMBINED) − CE(COMBINED_COADAPT) ≤ .03. Null: ≥ .08.
Descriptive (no bar): the five π_l and Σπ_l vs π (super-/sub-additivity of the penalty across pool blocks).

## Price
96 fit docs × 4 passes (covariances, fillers, pool heads, co-adapt pass) + 64 × 21 arms ≈ 1,730 GPU document-forwards ≈ 35 s.
Output late_stack_composition_penalty_anatomy_probe_results.json. Frozen: this file, §2732 results
(late_stack_extracted_program_probe_results.json), checkpoint, fit_natural.pt.
