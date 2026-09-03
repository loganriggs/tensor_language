# Preregistration — late_stack_write_core_probe (Claude, lane 1 CUDA)

Registered 2026-09-03 22:42Z (box clock). Follows §2742/§2745 (the fourteen late sublayers READ one shared input core). Independent of the attention-width rung queued ahead.

## Question
Do the fourteen late sublayers also WRITE into one shared subspace — and is it the same subspace they read? If so the late stack
lives in a k-dim subspace of the residual stream: it reads from it and writes into it, and the program is U_read = U_write = U.

## Arms (eval docs 0–63; fits docs 96–191; write w of site s replaced by μ_s + P(w − μ_s), μ_s = the site's fit-set write mean)
WRITE14_OWN_k for k ∈ {256, 512, 768}: each late sublayer's write projected onto its own top-k centred write PCs.
WRITE14_SHARED_k for k ∈ {256, 512, 768}: one write core = top-k of the average of the fourteen centred write covariances.
WRITE14_ON_READ_CORE_768: the writes projected onto the READ core of §2745 (joint late input core, first 768 columns).
READ_WRITE_768: reads (all fourteen on the joint input core, §2745 construction: .109) AND writes (shared write core) together.

## Predictions (CE added above the real model, docs 0–63, LOWER IS BETTER — §2135)
- pred_a_instrument: baseline within 1e-4 of 3.0322401; LATE14_JOINT_768 read arm within .02 of .109 (run as the read half of the
  READ_WRITE construction, reported separately).
- pred_b_shared_write_core_is_free: WRITE14_SHARED_768 − WRITE14_OWN_768 ≤ .03. Null: ≥ .10.
- pred_c_write_768: WRITE14_SHARED_768 ≤ .15. Null: ≥ .40.
- pred_d_write_subspace_is_the_read_subspace: WRITE14_ON_READ_CORE_768 − WRITE14_SHARED_768 ≤ .05. Null: ≥ .20.
- pred_e_read_and_write_together: READ_WRITE_768 ≤ .30. Null: ≥ .50.
Descriptive: own/shared write curves at 256/512; write effective ranks per site; capture ratios of the read core on the write covariances.

## Price
96 fit docs (one collect pass for 14 write covariances + 14 input covariances) + 64 × (1 + 9 arms) = 736 GPU document-forwards, ~15 s.
Frozen: this file, §2745 results, checkpoint, fit_natural.pt.
