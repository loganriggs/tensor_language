# Deeper-circuits program: finding the next induction heads

**Goal.** Train bilinear (tensor) attention models of increasing depth/width on **natural
text**, find the validation *datapoints* (tokens) that differentially improve when depth is
added, cluster those datapoints by their causal circuits, then — for each discovered circuit
— build a minimal toy task and study its **training dynamics** the way Singh et al. (2024,
`induction_head_training_dynamics_paper.txt`) did for induction heads. Induction (depth 2)
is the mold; chained k-hop retrieval (depth 3, already reverse-engineered in
`results_hop.md`) is the first rung above it.

**Decision (2026-07-08, Logan):** discovery corpus is natural text from the start (not a
synthetic mixture). Synthetic toy tasks are used *downstream* to isolate circuits found in
text, mirroring how the FSL toy task isolated induction.

## Pipeline (the loop we iterate)

1. **Ladder training** (`lm_train.py`): bilinear attn-only models, RMSNorm, depths 1–5
   (later: head counts 1/2/4/8, interleaved bilinear MLPs), ≥3 seeds each, on TinyStories
   with a small BPE vocab. Checkpoints saved through training for dynamics analysis.
2. **Shared validation** (`lm_eval.py`): the same frozen ~10M-token validation set run on
   every model → per-token CE matrix (model × token), saved once, appended per model.
3. **Differential datapoints** (`differential.py`): a token is **learned** by depth d when
   median-over-seeds CE < τ_learned; it is **depth-d-gated** when learned at depth d,
   NOT learned at depth d−1 (median CE > τ_unlearned, a strictly higher bar — the
   hysteresis gap makes the claim robust), and the gap replicates across seeds.
   Sanity gate: mean val CE must decrease monotonically with depth, else stop and debug.
4. **Circuit clustering**: for depth-gated tokens, cheap attribution fingerprints (per-head
   zero-ablation ΔCE vectors) → cluster → same-algorithm groups; verify a sample per
   cluster with probes/patching. Named circuits get a toy isolation task.
5. **Toy isolation + training dynamics** (the paper's method): minimal task, minimal
   architecture, progress measures per head, phase-change decomposition into subcircuits
   (clamping where feasible), seed-reliability stats.
6. **Atlas** (`atlas/`): one HTML page per circuit — examples of its datapoints, per-head
   roles, probe heatmaps, training-dynamics curves.

## Current status

- Done previously: k-hop depth ladder (attn2 fails hop≥2; attn3 unlocks hops 2–3 as a
  class; mechanism = per-layer pointer advance in a rotated basis; attn3 vs attn4
  reliability identical at 1/3 seeds; curriculum backfires 0/4).
- Repo reorganized (legacy → `archive/`, logs → `logs/`).
- Next: text pipeline + ladder training (tasks #2, #3), then eval matrix (#4).

## Architecture & training recipe (carried over from what worked)

- Bilinear attention (product of two dot products, causal mask, no softmax), rotary.
- `lerp` residual for attention (`add` stalls induction), `add` for bilinear MLPs.
- RMSNorm ON (affine-free, pre-norm) — required for depth ≥3 stability; note it breaks
  strict polynomiality (recorded gotcha, accepted trade-off per Logan).
- Adam 1e-3, warmup 200 + cosine; answer-position loss for toy tasks, full next-token CE
  for text; grad-clip only when norm is off.

## Gotchas / failure modes to check *before* believing a result

1. **CE monotonicity**: more layers must lower average val CE. If a deeper model is worse
   on average, it's an optimization failure (recipe, norm, lr), not a discovery.
2. **Seed lottery**: higher circuits form in only a fraction of seeds (hop-3: ~1/3 at any
   depth). Every claim needs ≥3 seeds; "depth unlocks X" must hold for *best-of-seeds*,
   and reliability is reported separately. Never conclude from seed 0 alone (this bit us
   twice: attn4 reliability, single-doc circuit trace).
3. **Cherry-picked mechanism stories**: single-example attention traces overfit (the hop-3
   "clean composition" story died on 2447-query aggregate). Mechanism claims need
   aggregates + probes, not one doc.
4. **Ablation redundancy trap** (paper §3.1): knock-out ablations understate a head's role
   when heads are additive/redundant; use knock-all-but-one too.
5. **Threshold hysteresis**: "learned" (τ_learned) and "not learned" (τ_unlearned) must be
   different thresholds with a gap, or noise near a single threshold produces fake
   depth-gated datapoints. Natural text tokens are stochastic — many tokens are never
   "learned" by any depth; that's expected, they're just not datapoints of interest.
6. **Tokenizer/vocab confound**: embedding params scale with vocab; keep vocab fixed
   across the whole ladder so depth is the only variable. Same for d_model, n_ctx, data
   order (fix the data seed across depths so models see identical batches).
7. **Bilinear attention on text is unproven here**: scores are unnormalized (no softmax),
   so long contexts change score scale; watch for divergence/plateaus at n_ctx 256 and
   compare a softmax control if text CE looks broken.
8. **RMSNorm breaks the tensor property** (input-dependent division). For any circuit we
   want to read off in closed form, retrain the isolated toy version norm-free +
   grad-clip (known-stable at depth ≤3 per session 5).
9. **Induction plateau as attractor**: failed deep seeds stall at copy/induction
   (f^1-at-chance signature). Track per-cluster accuracy during training to catch
   plateau-stuck runs early; curriculum does NOT fix this (0/4) — escape levers TBD.
10. **`${WORKSPACE}` is not a volume on this instance** unless verified — check
    `vast-capabilities | jq '.instance.workspace_is_volume'`; push git regularly; don't
    keep irreplaceable state only in `runs*/`.

## Questions for Logan

- (2026-07-08) Corpus resolved: natural text. I'm starting with **TinyStories + BPE-1024,
  n_ctx 256, d_model 128, 4 heads**; veto/adjust anytime.
- Depth ladder starts at 1–4 layers (5 later if the CE curve says there's headroom).
- OK to spend GPU on a softmax-attention control ladder too? (One control per depth, for
  the "is bilinear the bottleneck" question.) Default: yes, after the bilinear ladder.
