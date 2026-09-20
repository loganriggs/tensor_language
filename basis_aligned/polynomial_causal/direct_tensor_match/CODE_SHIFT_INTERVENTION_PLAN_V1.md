# Frozen programs under a local Python-code domain shift

2026-09-20 21:38 UTC. No fitting or candidate selection. Panel generation is
implemented and executed by build_code_shift_panel.py, with source/token hashes.
Sixteen tracked Python files, first129 tokens each, selected lexicographically
without model evaluation. This is a deliberate domain shift relative to FineWeb
calibration, not a representative independent code corpus. Pretraining overlap
is unknown; related files limit effective sample size.

Repeat NATIVE_QUARTIC_BRANCH_PLAN_V1 and NATIVE_MODE_INTERVENTION_PLAN_V1 on
CODE_SHIFT_PANEL_V1.pt at context128 with all original arms and unchanged bars.
No relaxed thresholds: branch replay<1e-5, ten-product CE<.02 and KL<.02,
logit disturbance<half ablation; individual-mode replay<1e-5, modes0/1
cos>.9/error<.4, all4 cos>.8/error<.65. Report failures and file-wise outcomes.
The previous FineWeb aggregate passed mode bars but mode3's bootstrap interval
crossed the threshold and token-loss effect error was72%. This check tests
whether apparent reusable features survive a changed input domain.

Candidate and canonical basis remain frozen. The normalization denominators,
attention, residual/cross terms and final softcap remain native. All GPU work
uses managed queue. Price and intervention definitions identical to prior plans.
