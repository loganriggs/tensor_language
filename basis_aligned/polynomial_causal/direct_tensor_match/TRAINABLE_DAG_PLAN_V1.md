# Fixed-topology arithmetic DAG controls — 2026-09-20 19:27 UTC

Purpose: verify a differentiable graph with tied computation reuse, constants, mixed polynomial degrees, and auditable parameter pricing before fitting native graphs. CPU only.

Five teacher structures: affine quadratic with residual; shared linear sum across outputs; square of a quadratic; shared quadratic across quartic branches; mixed-degree skip connections. Learned leaf/readout coefficients; fixed unit aggregation and shared constant. Same topology is a capacity witness, not discovered structure. Two independent random parameter initializations per family; Adam .03,1500 steps;1024 fixed N(0,I) fitting probes,4096 independent validation probes. Checkpoint by fitting loss only; validate once after fitting. Keep best and final training losses. Muon comparison follows instrument validation, not a claim in this screen.

- pred_a: oracle compiler and fitted export replay <1e-10 relative; directional finite-difference gradient discrepancy <1e-6.
- pred_b: every graph has degree bound <=4 and attempted degree>4 node rejected without mutating the graph; unit constants shared.
- pred_c: at least4/5 families have one random start with validation relative error<.01. Failure with passing oracle is optimizer failure, not capacity failure.

Price all trainable coefficients even if a fitted value happens to be unit or zero; fixed unit aggregation is structural. Training on artificial probes is a Gaussian function metric, not coefficient Frobenius. Fresh Gaussian probes do not establish text OOD or semantic interpretation.
