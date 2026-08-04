# Scale-run handoff — instructions for the Claude session on the rented GPU

You are a Claude session on a rented Vast instance (single ~40 GB GPU, or several
GPUs), cloned fresh from this repo. Your job for the next ~12 hours: run the
interpretable-architecture experiments AT SCALE (width 1152, bilin18's width) on
fresh data, and push results back. Read RESULTS_l0_mdl.md §108–§111 for scientific
context; this file is the operational spec. Logan is the user; report in spelled-out
prose with concrete numbers, no invented shorthand.

> **RE-PRICING (2026-08-04, read before choosing arms):** on fresh single-epoch
> data the slots+lasso base costs **+0.342 nats vs vanilla** at width 264
> (qk_e0.json; SE 0.002) — the old ~+0.08 was memorization-subsidized. Per-slot
> RMSNorm (E1) BEATS the base by 0.026 (only arm that wins; add it to your gate's
> slots arm as a third arm if budget allows). Certified-zero annealing recovers
> to +0.052 after fine-tune with 50% of read groups exactly zero (qk_e3.json).
> Typed token slot cost +0.222 at 11 dims — skip at scale unless widened. Your
> width-1152 gate MUST include a lasso-coefficient arm (gc 3e-5 or per qk_e5.json
> once it lands) — deciding whether the recipe keeps the lasso at full strength
> is now the main open question, alongside whether the +0.34 shrinks with width.

## Two sessions in parallel — coordination protocol

The ORIGINAL session stays live on its own 16 GB machine, doing new ideas and
interpretation at small scale. You do the big runs. Git is the only sync channel:

- `git pull` BEFORE choosing arms: the original session's overnight chain writes
  qk_e0/e0m/er/e1..e4.json (small-scale fresh-data results for controls, Muon,
  readout replications, and E1–E4) and pushes them as they land. If they're there,
  use them to prioritize (scale up what showed signal; drop what clearly failed).
- Name YOUR files with an `qk_s_` prefix (qk_s_w1152_controls.json,
  qk_s_e1_run.py, …) so nothing collides with the original session's files.
- Commit + push results as each arm finishes (not just at the end), message style:
  one-paragraph finding headline. Pull before every push; merge conflicts should be
  impossible if the prefix discipline holds.
- Don't edit RESULTS_l0_mdl.md directly (the original session owns it) — write your
  section drafts to RESULTS_scale_draft.md and the original session will fold them in.

## What to run

All arms are INDEPENDENT — with k GPUs run k arms concurrently, one per GPU
(`CUDA_VISIBLE_DEVICES=i python …`). No DDP needed or supported; do NOT shard one
model across GPUs, it's never worth it at these sizes.

**Single 40 GB GPU (~12 h serial budget, ~8–10 arms):** run in this order —
1. Width-1152 fresh-data controls: vanilla AND slots+lasso base ("the gate") —
   THE decision experiment for the retrain recipe: does the slots+lasso cost
   (~+0.08 at width 264) vanish at real width, as it did at 768? It crashed on the
   16 GB card at preflight and has NEVER run — you own it. Use real batch 32 bf16
   (no accumulation needed at 40 GB; preflight it), lr per sweep below.
2. Optimizer gate at width 1152: the slots+lasso base under Muon (lr sweep
   {0.01, 0.02, 0.04}) vs AdamW (WIDEN the grid: {0.002, 0.004, 0.008} — 0.004 won
   at the sweep edge at width 264, still improving). If Muon confirms its
   small-scale lead, it becomes the retrain default and your remaining arms can
   use it — but keep every within-comparison on ONE optimizer.
3. E1 per-slot RMSNorm and E4 typed token slot at width 1152 — the two
   highest-value interpretability bets (slot dim is 48 at width 1152 = 1152/24;
   E4's token slot is one 48-dim slot).
4. E2 CP-rank caps (r ∈ {8, 32, 128}) and E3 anneal-to-certified-zeros (anneal
   from your trained width-1152 slots base).
5. If time remains: N=6 window at 1152; V11 rank-1 logit decoders at 1152.

**Multi-GPU (k cards):** same list, one arm per card, controls first (everything
pairs against them). With 4×24 GB instead of 1×40 GB, use effective batch 32 via
2-step accumulation per card and you'll finish the whole list plus the step-5
extras. Spend any leftover GPU-hours on seeds of the gate (arms 1–2), not new
variants — the gate verdict wants tight error bars.

**Step budget per arm:** with 300k fresh sequences and batch 32, a full single
epoch is 9,375 steps (~1–2 h/arm at width 1152 on an A100-class card — measure
your real step time in the first 100 steps and rescale the plan; if arms run long,
cut the per-arm budget to a fixed prefix, e.g. 6,000 steps = 192k sequences, THE
SAME prefix for every arm, rather than dropping arms 1–2).

## Convention checklist (non-negotiable, from this program's history)

Identical data order across compared arms; paired sequence-clustered SEs; positive
controls BEFORE training (identity checks — every wrong headline in this program
was caught by a known-answer control, never by inspection); nonzero write init
std 0.02 × 1/sqrt(width/384); save train-CE curve every 200 steps AND held-CE
checkpoints (2000/4000/6000/8000-style) into the result JSON; 30·tanh(logits/30);
bf16 autocast training, bf16 eval. Build on `qk_e_common.py` (fresh-stream sharded
loading, curve saving, from-scratch Muon, paired evals) — NOT the older 6-epoch
harness; `qk_w1152_train.py` shows the width patch + micro-batch preflight pattern.

Known pitfalls that already burned us once: zero-init writes + multiplicative
routing = dead-gradient fixed point (init std 0.02); removing/linearizing RMSNorm
blows up (it's TN-expressible via a gauge/dummy bond — never delete it for interp
reasons); lr winner at a sweep-grid edge means widen the grid; never train a
second pass over any sequence (memorization protocol).

## Data (single-epoch, no-memorization protocol — Logan's requirement)

Corpora in `corpus_fresh/` (committed shards, uint16, rows of 513 GPT-2 tokens;
see MANIFEST.txt):

- `corpus_fresh/shard00..06.npy` — 300,000 sequences, FineWeb docs 45,367–267,574.
  Concatenate in shard order; every arm trains on the SAME prefix, one pass.
  Scale held set: the LAST 1,500 rows of shard06 (never train on them).
- `corpus_fresh/fresh34k.npy` — pure eval corpus (docs 20,001–45,366): rows
  [33000:34500] are the held set the small-scale E-runs report on. Evaluate your
  scale models on it too, for direct comparability with the small-scale numbers.
- Need more (e.g. multi-GPU eats 300k fast)? `python qk_corpus_build.py N START_DOC`
  with START_DOC > the last doc in corpus_fresh/MANIFEST.txt (currently 267,574).
  ~2 min per 300k sequences on 16 CPU cores, network-bound, no HF token needed.
  Commit the new MANIFEST so doc-range bookkeeping stays global.

## GPU requirements / compute profile

- Width-1152 depth-12 needs ~13 GB at effective batch 8 (measured); 40 GB runs
  real batch 32 comfortably. Ampere or newer (bf16). On Blackwell (RTX 50xx, B200)
  the torch wheel must be CUDA ≥ 12.8 — check
  `vast-capabilities | jq '.hardware.gpu.cuda'` → `min_cuda_for_wheels`; a cu124
  wheel installs cleanly then dies at the first kernel.
- At T=512, attention's O(T²) pattern is ~5% of FLOPs at width 1152 (share ≈
  T/(8D+T)); dense matmuls dominate. Batch size is the utilization lever.
- Disk ≥ 20 GB free (checkpoints 100–300 MB each, corpus 0.4 GB).

## torch.compile (optional, measure it)

Untested on these architectures. All custom pieces (slots = indexed writes,
bilinear s1*s2 pattern, per-module scalars, group-lasso) are plain dense ops —
expected to compile. Protocol: compile, verify compiled-vs-eager CE on one batch
agrees to ~1e-3 bf16 (positive-control habit), fall back to eager on graph-break
spam or NaN. Record compiled-vs-eager step time in the JSON.

## Setup on the fresh instance

```
git clone https://github.com/loganriggs/tensor_language.git
cd tensor_language/basis_aligned/qk_mdl
source /venv/main/bin/activate            # or the image's torch env
uv pip install datasets transformers      # torch ships with the image; check its CUDA vs the GPU
python - <<'EOF'                          # smoke: corpus integrity
import numpy as np, glob
for p in sorted(glob.glob('corpus_fresh/shard*.npy')):
    a = np.load(p, mmap_mode='r'); assert a.dtype == np.uint16 and a.shape[1] == 513 and a.max() <= 50256
    print(p, a.shape)
EOF
```

bilin18 itself is NOT needed (tier2_model/load_elriggs only matters for probing the
original model — skip it on the rented box). If a probe helper import-fails on a
bilin18 dependency, stub that probe and note it rather than downloading the model.

## Reporting back

Push, per arm: the result JSON (lr sweep table; final + curve CE on BOTH held
sets; seq-clustered SE vs the paired control; probe outputs — wiring Spearman,
token-determined variance; exact data prefix as shard list + row range), the
runner, and a RESULTS_scale_draft.md paragraph. The gate verdict (arm 1) and the
optimizer verdict (arm 2) are the two numbers Logan is waiting on — push those the
moment they exist, don't batch them with the rest.
