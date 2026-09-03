# Preregistration — late_stack_write_routing_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:49Z (box clock). Follows §2747 (the fourteen late sublayers READ one 768-core but write broadly: deleting the out-of-core write costs .147).

## Question
WHO reads the out-of-core part of the late writes? Split each late write (14 sites, 11–17) into its part inside the shared read
core, P(w − μ), and its part outside, (I − P)(w − μ). Route the outside part either (i) straight to the readout (added back to the
residual just before the final norm, invisible to every later sublayer) or (ii) to the later sublayers only (kept in the stream,
subtracted just before the final norm). If (i) is cheap and (ii) is dear, the late stack's inter-sublayer traffic is confined to
the read core and its broad writes are for the logits — a communication graph: a 768-dim bus between sublayers, full width to the
readout.

## Arms (eval docs 0–63; fits 96–191; P = U_read U_readᵀ, U_read = §2745 joint late input core, first k columns; μ_s = site write mean)
DELETE_k (k ∈ {512, 768}): out-of-core part deleted (= §2747 WRITE14_ON_READ_CORE at 768).
TO_READOUT_k (k ∈ {512, 768}): out-of-core part removed from the stream and added to the residual before the final norm.
HIDDEN_FROM_READOUT_768: out-of-core part kept in the stream and subtracted before the final norm.
LATE14_JOINT_768: §2745 read program (reproduction).
BUS_768: the read program (reads on the 768 core) + TO_READOUT_768 at the same fourteen sites.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; DELETE_768 within .02 of .147; LATE14_JOINT_768 within .02 of .109.
- pred_b_later_sublayers_do_not_need_it: TO_READOUT_768 ≤ .05. Null: ≥ .12.
- pred_c_readout_needs_it: HIDDEN_FROM_READOUT_768 ≥ .08. Null: ≤ .03.
- pred_d_bus_program_equals_read_program: |BUS_768 − LATE14_JOINT_768| ≤ .02. Null: ≥ .06.
- pred_e_narrower_bus: TO_READOUT_512 ≤ .10. Null: ≥ .25.
Descriptive: DELETE_512; additivity DELETE_768 − (TO_READOUT_768 + HIDDEN_768); per-site out-of-core energy fractions.

## Price
96 fit docs + 64 × (1 + 7 arms) = 608 GPU document-forwards, ~13 s. Frozen: this file, §2747 results, checkpoint, fit_natural.pt.
