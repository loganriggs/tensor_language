# Scale-session draft (width 1152, rented box) — for the original session to fold in

Scale session, started 2026-08-04 ~18:00 UTC. This file is owned by the scale
session per the SCALE_RUN.md protocol; the original session folds it into
RESULTS_l0_mdl.md.

## Box and protocol facts (differ from the handoff spec in three ways)

- The rented cards are 2× **RTX 4080 SUPER 32 GB** (Ada, compute capability
  8.9), not the advertised RTX 5090s. bf16 fine; torch 2.11.0+cu128.
- **Effective batch 32 does not fit in 32 GB** at width 1152: micro-batch 32
  OOMs, so every arm runs micro 16 × 2-step gradient accumulation (preflight
  peak ~25.9 GiB; the grad-accumulation positive control passes at rel
  3.1e-5). The gradient math is identical to real batch 32; step time ~0.95
  s/step, full epoch ≈ 2.5 h/arm plus ~25 min of lr sweep.
- The original cooc corpus is not on this box (never committed).
  `data_fineweb_cooc_tokens.npy` here is a SUBSTITUTE built from fresh34k
  rows [0:6000] (pure-eval docs) so the harness imports; **no old-held cooc
  numbers are produced by the scale session** and none of the qk_s_ JSONs
  contain any.

Data: train prefix = corpus_fresh shards 00..06 concat rows [0:298496]
(9,328 steps × eff batch 32, single pass, epoch_order(0), identical across
every arm). Scale held = shard06 last 1500 rows (never trained; the 4-row
gap [298496:298500] is simply unused). Second held set = fresh34k rows
[33000:34500], the small-scale E-run held set, evaluated per-token on every
arm for direct comparability with qk_e0/e1..e5.

Positive controls pass in every arm process before training: slots model
with identity projections + zeroed writes == vanilla-1152 at init (max
|logit diff| 0.0 fp32); vectorized group penalty == naive loop at slot dim
48 (rel 1.3e-7); accumulated micro-grads == one-shot grads (rel 3.1e-5).

## Arms (qk_s_ files; one per GPU, all on the identical data order)

| arm | architecture | optimizer | status |
|---|---|---|---|
| vanilla | zero-init-write MiniBilin A, d12 w1152 | AdamW, swept | RUNNING |
| slots | partitioned write slots, no lasso | AdamW, swept | RUNNING |
| gc1e4 | slots + group-lasso 1e-4 (recipe base) | AdamW, swept | queued (round 2) |
| muonbase | slots + gc 1e-4, loss-lasso | Muon 2D + AdamW emb, swept | queued (round 2) |
| gc3e5 | slots + group-lasso 3e-5 | AdamW, swept | queued (round 3) |
| E1 slotnorm | slots + per-slot RMSNorm | AdamW, swept | queued (round 3) |

## §S1 The width-1152 gate (partition cost at slot dim 48) — PENDING

Key question (E5 re-pricing): the slot PARTITION alone cost +0.234 of the
+0.342 at width 264 / slot dim 11. Does it shrink at slot dim 48?

- vanilla final held CE (scale held / f34k): PENDING
- slots-only minus vanilla, paired seq-clustered SE: PENDING
- gc3e5 minus vanilla and minus slots-only: PENDING (round 3)
- gc1e4 minus vanilla (the old unconditional recipe at scale): PENDING (round 2)

## §S2 The width-1152 optimizer gate — PENDING

muonbase (Muon on 2D hidden, loss-lasso, matching qk_e0m conventions) vs
gc1e4 (AdamW), same architecture, same data order. Small-scale reference:
Muon won vanilla by −0.094 but LOST the lasso base by +0.076. E7a's proximal
verdict had not been pushed when the scale Muon arm launched, so the scale
arm keeps the loss-lasso convention deliberately for comparability.

- muonbase minus gc1e4, paired seq-clustered SE: PENDING

## §S3 E-series at width 1152 — PENDING (round 3+)
