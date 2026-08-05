# Builder-agent brief (read this first; it replaces transcript history)

You are building/queueing experiments in /workspace/tensor_language/basis_aligned/qk_mdl
for the interpretable-architecture program. Everything you need is on disk; do not
ask for context.

## Protocol (non-negotiable)
- Fresh single-epoch batch-16 protocol: 8250 steps, corpus_fresh/ shards prefix
  (132,000 rows, epoch_order(0), identical across arms), held = fresh34k.npy rows
  [33000:34500]. NEVER train a second pass over any sequence.
- Harness: qk_e_common.py (setup(), train_arm(), paired_fresh(), probe_arm(), Muon +
  prox, non-blocking guards, SMOKE mode via QK_SMOKE=1 — CPU-only but needs a
  GPU-free moment at import on some paths). Reference model factories and
  conventions live there; runners qk_e0..e15*.py are worked examples.
- The readable recipe (reference arm E9a / qk_e9_a.pt): partitioned write slots +
  per-slot RMSNorm + Muon 0.02 (embedding on AdamW 0.004) + in-loss group-lasso
  3e-5. Controls: E0a vanilla (qk_e0a_vanilla264), E0b slots+lasso AdamW 1e-4.
- POSITIVE CONTROLS BEFORE TRAINING, always: identity reductions must pass at
  exactly zero; penalties vs naive loop; every wrong headline in this program was
  caught by a known-answer control, never by inspection.
- Save into every result JSON: train-CE curve every 200 steps, held-100 curve,
  lr/gc/width/batch/steps/corpus fields, paired deltas with seq-clustered SEs.
- Nonzero write init std 0.02 (x 1/sqrt(width/384) at other widths). Zero-init
  writes + routing = dead-gradient trap. lr winner at a sweep-grid edge -> widen.
- Chains: detached bash chain scripts gated by EXACT-NAME pgrep (substring pgrep
  self-matches killed us twice); runners idempotent on checkpoints; verify pid
  then EXIT — never babysit; you get no credit for waiting.
- Git: push results as they land; MAILBOX.md (append-only, newest first) is the
  cross-session channel with the scale box; their files are qk_s_*, ours qk_e*.
- Reporting: spelled-out prose, descriptive names (readable recipe, shared-values,
  budgeted-routing, funnel, split-reads), 2-3 concrete examples with numbers.

## Current state pointers (read the JSONs, not history)
- Frontier + all verdicts: qk_e0..e15 JSONs, RESULTS_scale_draft.md, MAILBOX.md.
- Key facts: proximal lasso never binds (free but zero readability — in-loss only);
  saturation verdict in qk_e14.json census; effective-params accounting in
  qk_e15.json; naming table in qk_e13.json.
