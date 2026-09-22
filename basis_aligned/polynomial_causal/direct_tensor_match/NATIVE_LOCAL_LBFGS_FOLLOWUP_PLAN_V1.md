# L-BFGS larger-panel residual evaluation

22 September2026,05:13 UTC. Prepared before native L-BFGS results exist.

Reuse audit_local_quartic_followup.py with --lbfgs. Keep its default Adam/Muon mode and primary predictions unchanged. Require exactly both seeds25001/25002 in the L-BFGS receipt, artifact and parent hash matches, old-panel metric replay within1e-4, and outputs0–3 bitwise protected. Evaluate all16384 already-opened states and2494 fixed matched pairs; no fitting or new sample capture.

Report pooled and all16 per-output value/response errors, equal-small-output RMS, all256 document errors, and state residual distributions for both all outputs and outputs4–15. Record median/p90/p95/p99/max state percentages, percentages using one common target RMS, worst1/5/10% share of squared error and target energy on those same states, mean-bias share and centered residual spectrum. Small denominators can inflate state-relative percentages, so both normalizations remain visible. A zero residual has zero concentration/bias/spectrum shares by convention. These summaries do not attribute input-variable causality.

The descriptive transfer bar remains both starts with at least15% improvement in both equal-small-output value and response RMS versus CP1001. Report failures and absolute errors, not just the best start or percentage gain. This opened panel is not fresh OOD validation. Only final exported checkpoints are scored here; first251 metrics remain in the native receipt on the original panel. No semantic or causal adoption follows.

Command: /venv/main/bin/python basis_aligned/polynomial_causal/direct_tensor_match/audit_local_quartic_followup.py --lbfgs
