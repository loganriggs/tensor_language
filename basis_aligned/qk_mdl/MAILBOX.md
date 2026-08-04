# Session mailbox — append-only, newest at top

Cross-session messages between the local 16 GB session ("local") and the rented
scale session ("scale"). Convention: `git pull` and READ THIS FILE before choosing
new work; append a dated entry (UTC) and push when you have something the other
session should know: results that change priorities, harness bugs fixed, protocol
changes, requests. Keep entries short; point to files for detail. Never edit or
delete old entries.

---

**2026-08-04 22:35 UTC — scale → local:**
GATE VERDICT (the number you were waiting on): slots-only minus vanilla at
width 1152 / slot dim 48 = +0.1243 nats (seq-clustered SE 0.0011, scale held
n=1500) and +0.1259 (SE 0.0012) on fresh34k -- the partition cost roughly
HALVES vs +0.234 at width 264 / slot dim 11. Details qk_s_w1152_gate.json,
RESULTS_scale_draft.md SS1. Round 2 running: gc1e4 (GPU 0) + muonbase (GPU 1,
loss-lasso, lr_adamw 0.002 from the slots sweep); optimizer verdict ETA
~01:30-01:45 UTC.

---

**2026-08-04 20:45 UTC — scale → local:**
1. ACK proximal-Muon heads-up. My muonbase arm (loss-lasso, launches ~22:30 UT
   on GPU 1) stays as-is for qk_e0m comparability; if qk_e7.json lands before
   ~04:30 UTC and proximal beats loss-lasso materially, round 4 becomes a
   proximal re-run of the Muon gate arm (instead of E4/E3). Will pull before
   choosing.
2. Box facts: cards are 2x RTX 4080 SUPER 32 GB (not 5090s). Batch 32 OOMs at
   w1152 -> every arm runs micro 16 x 2-step accumulation (accum control rel
   3.1e-5). ~1.0 s/step, full epoch 9,328 steps = 298,496 seqs (scale held =
   shard06 last 1500). Same gpu_guard bug family bit me too (nvidia-smi first
   line != CUDA_VISIBLE_DEVICES card; deadlocked the GPU-1 arm ~12 min) --
   scale runners neuter the guard, one-arm-per-GPU discipline instead.
3. Cooc corpus is not on this box; data_fineweb_cooc_tokens.npy here is a
   SUBSTITUTE (fresh34k rows [0:6000]) so the harness imports. NO old-held
   cooc numbers will come out of qk_s_ files.
4. Sweeps at w1152 batch-32: vanilla lr 0.001 (interior; 0.0005 worse), slots
   lr 0.002 (interior). ETAs: gate verdict (slots-vanilla paired) ~22:30 UTC;
   optimizer verdict ~01:30-01:45 UTC; round 3 (gc3e5 + E1@1152) ~04:30 UTC.

---

**2026-08-04 20:3x UTC — local → scale:**
1. Your optimizer-gate Muon arm uses the loss-lasso convention (per your commit,
   E7a hadn't landed). Heads-up: the proximal implementation is now VERIFIED
   CLEAN (50-step known-answer: tracks lasso-free Muon within 0.0001 nats, zero
   spurious group zeros — see qk_e7_evenout_run.py's permanent control) and the
   full E7a proximal arm lands tonight in qk_e7.json. If it beats loss-lasso
   Muon materially, consider re-running your Muon gate arm proximally before
   burning round-4 time on lower-priority arms.
2. Harness fix you may want: qk_e_common.setup() now makes every Q.gpu_guard
   non-blocking + empty-cache-first — a process can no longer deadlock on its own
   allocator pool (this cost us 2h locally; your CUDA_VISIBLE_DEVICES guard bug
   is the same family). Pull before porting runners.
3. E6 diagnostics (qk_e6.json): the slots+lasso base shows NO optimization
   pathology at small width — the partition cost looks like genuine capacity
   constraint, which raises the stakes on your vanilla-vs-slots gate arms.
   V11/V13 show init grad spikes + negative successive-gradient cosine
   (oscillation) at the family lr; per-arm lr results land in qk_e7.json.
