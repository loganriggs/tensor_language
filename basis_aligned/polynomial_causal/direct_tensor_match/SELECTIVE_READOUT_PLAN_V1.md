# Selective readout native test — 2026-09-20 22:46 UTC

Candidate: SELECTIVE_SCALAR_READOUT_V1.pt, selected after diagnostic decoder comparison, unchanged six primitive readers/four roots and three scalar outputs; only feature1 readout comes from fixed_0.0001 calibration fit. Same scalar cost13916coeff/10products plus4608writercoefficients. No fitting in this test.

Reuse existing context256 FineWeb32/code16 panels and token-only donor maps. Native removals and aligned/same-token swaps,96forwards. Tests use the same native background, original output directions, normalization and softcap. Only original artifact and output paths become parameters in the reused swap evaluator. Vector-output adapter for the removal evaluator is scaffolding, not the candidate's deployed representation.

pred_a_instrument: evaluator checks pass; native target energies match archived runs; untouched modes0/2/3 predicted effect/error energies match archived originals at relative1e-5. pred_b_swap: feature1 same-token cosine>.9 and relativeeffecterror<.4 on BOTH domains. pred_c_joint: joint relativeeffecterror no worse than original on BOTH domains for BOTH removal and same-token swap. Also retain every individual result and aligned swaps. Native target identity and operation controls do not demonstrate semantic/task selectivity. No new independent confirmation claim.
