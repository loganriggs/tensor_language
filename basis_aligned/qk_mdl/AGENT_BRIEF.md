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

## Two hard rules added 2026-08-08
- NEVER `git stash` (especially `-u`) while a run is in flight: untracked
  checkpoints vanish and the running job dies. Fix a rejected push with
  fetch+rebase, staging only the intended paths.
- Quote CE taxes against the MATCHED-OPTIMIZER baseline: at w264 Muon
  vanilla is 4.7570 (qk_e0m.json), AdamW vanilla 4.8513 (E0a). Every
  structured arm is Muon-trained, so 4.757 is the fair reference; using
  4.851 understates every tax by ~0.094 nats.

## Sign is a gauge freedom (standing rule, Logan 2026-08-08)
No positivity constraints exist anywhere in these models, so the sign of any
intermediate factor is gauge, not mechanism: flip it in one factor and back
in another and the function is identical. Only signs of COMPLETE paths to an
observable are invariant. A negative attention weight is NOT suppression —
contribution = pattern x (value->output->unembed), so negative x negative is
a positive push on the attended content and the negative effect lands on
non-attended positions. The pattern is itself a product of two branches, so
even its sign is a product of two meaningless signs. Never report
suppression/inhibition/anti-correlation from a factor in isolation; compose
to logits and confirm causally. Measured precedent: raw coefficient sign
anti-correlated with behaviour (-0.45) while the composed score tracked it
(+0.85, causal direction 5/5).
