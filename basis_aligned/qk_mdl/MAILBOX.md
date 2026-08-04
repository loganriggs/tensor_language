# Session mailbox — append-only, newest at top

Cross-session messages between the local 16 GB session ("local") and the rented
scale session ("scale"). Convention: `git pull` and READ THIS FILE before choosing
new work; append a dated entry (UTC) and push when you have something the other
session should know: results that change priorities, harness bugs fixed, protocol
changes, requests. Keep entries short; point to files for detail. Never edit or
delete old entries.

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
