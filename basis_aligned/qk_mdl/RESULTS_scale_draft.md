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

## §S1 The width-1152 gate (partition cost at slot dim 48) — HEADLINE IN

**The slot-partition cost roughly HALVES at width 1152 / slot dim 48:
slots-only minus vanilla = +0.1243 nats (seq-clustered SE 0.0011) on the
scale held set, +0.1259 (SE 0.0012) on fresh34k — vs +0.234 (SE 0.002) at
width 264 / slot dim 11 (qk_e5).** Direction confirmed on both held sets;
the fresh34k number is directly comparable to the small-scale E-runs.

- vanilla: held scale CE 4.11304, f34k CE 4.16121 (lr 0.001 interior winner,
  0 spikes, 9,328 steps).
- slots-only: held scale CE 4.23733, f34k CE 4.28708 (lr 0.002 interior
  winner, 0 spikes). Slots swept HIGHER than vanilla (0.002 vs 0.001), same
  direction as small scale.
- Train-CE curves + held-100 checkpoints every 2000 steps are in
  qk_s_w1152_{vanilla,slots}.json; per-token losses in the _heldloss /
  _f34kloss npys; paired stats in qk_s_w1152_gate.json.
- gc3e5 minus vanilla / slots-only: PENDING (round 3)
- gc1e4 minus vanilla (the old unconditional recipe at scale): RUNNING
  (round 2, GPU 0)

## §S2 The width-1152 optimizer gate — THE SMALL-SCALE RESULT FLIPS

**Muon WINS on the lasso base at width 1152, even with the loss-lasso
convention: muonbase minus gc1e4 = −0.0850 nats (seq-clustered SE 0.0011)
on scale held, −0.0863 (SE 0.0012) on fresh34k.** At width 264 Muon LOST
this comparison by +0.076 (qk_e0m). Both arms: same architecture
(slots + gc 1e-4), same data order, each with its own interior lr winner
(Muon 0.02 / AdamW 0.002; Muon's embedding-AdamW split at 0.002).

- muonbase: held scale CE 4.31098, f34k CE 4.36100, 0 spikes, ~1.13 s/step
  (Newton-Schulz overhead ~13% over AdamW).
- Implication: Muon is the retrain-default candidate at real width even
  before E7a's proximal fix; if proximal beats loss-lasso on top of this,
  the margin only grows. (Caveat: "recipe cost under Muon" vs a MUON vanilla
  is not measured here — muonbase minus ADAMW vanilla is +0.198, an upper
  bound on that quantity since Muon won vanilla at small scale too.)

## §S3 E-series at width 1152 — PENDING (round 3+)
