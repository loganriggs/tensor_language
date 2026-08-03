# Scale-run handoff (rented GPU session)

Prepared 2026-08-03 for a ~12-hour run of the interpretable-architecture experiments at
larger width / on more GPUs, from a fresh session that clones this repo. Read
RESULTS_l0_mdl.md §108–§111 for scientific context; this file is the operational spec.

## What to run (priority order)

All arms are INDEPENDENT — with k GPUs run k arms concurrently, one per GPU
(`CUDA_VISIBLE_DEVICES=i`). No DDP needed or supported.

1. **Fresh-data controls at width 1152**: vanilla and slots+lasso base (the small-scale
   fresh-data versions are E0a/E0b in qk_e1234; reuse those runners with the width patch
   from qk_w1152_train.py `patch_width`). These anchor every delta.
2. **E1 per-slot RMSNorm**, **E4 typed token slot** at width 1152 — the two highest-value
   interpretability bets.
3. **E2 CP-rank caps** (r ∈ {8, 32, 128} at width 1152), **E3 anneal-to-certified-zeros**
   (anneal from the width-1152 slots base once trained).
4. If time remains: N=6 window at width 1152; V11 rank-1 logit decoders at width 1152.

Convention checklist per arm (non-negotiable, from this program's history):
identical data order across compared arms; paired sequence-clustered SEs; positive
controls before training (identity checks — every wrong headline here was caught by a
known-answer control, never by inspection); nonzero write init std 0.02 (× 1/sqrt(width
ratio) vs 384); save train-CE curve every 200 steps AND held-CE checkpoints into the
result JSON; 30·tanh(logits/30); bf16 autocast training, bf16 eval convention.

## Data (single-epoch, no-memorization protocol — Logan's requirement)

Never train more than ONE pass over any sequence. Corpora in `corpus_fresh/`
(committed shards, uint16, rows of 513 GPT-2 tokens; see MANIFEST.txt):

- `corpus_fresh/shard00..06.npy` — 300,000 sequences, FineWeb docs 45,367+.
  Concatenate in shard order; take a prefix sized to your step budget
  (steps × effective_batch ≤ 300,000). Held set: LAST 1,500 rows of the last shard
  (never train on them).
- `corpus_fresh/fresh34k.npy` — the 34,500-sequence corpus the small E-runs used
  (docs 20,001–45,366); rows [33000:34500] are its held set. Evaluate scale models on
  this held set too, for direct comparability with the small-scale numbers.
- Old multi-epoch corpus `data_fineweb_cooc_tokens.npy` is gitignored/not shipped;
  its held set is superseded by the fresh ones for new runs.
- Need more data? `qk_corpus_build.py N START_DOC` rebuilds/extends (start_doc must be
  > the last doc in corpus_fresh/MANIFEST.txt; ~1–2 minutes per 100k sequences on 16
  CPU cores, network-bound; no HF token needed).

## GPU requirements

- **VRAM**: width-1152 depth-12 trains under ~13 GB at effective batch 8 with grad
  accumulation (measured here). 24 GB (3090/4090/L40S) is comfortable; 40–80 GB
  (A100/H100) lets you raise the real batch to 32–64 — do that and rescale lr
  (sqrt-scaling from the swept lr, then re-sweep {0.5×, 1×, 2×} for 400 steps).
- **Architecture**: Ampere or newer (bf16 required). On Blackwell (RTX 50xx, B200) the
  torch wheel must be CUDA ≥ 12.8 (`vast-capabilities | jq '.hardware.gpu.cuda'` →
  `min_cuda_for_wheels`); a cu124 wheel installs cleanly then dies at first kernel.
  This machine runs torch 2.12.1+cu130.
- **Compute profile**: at T=512 attention's O(T²) pattern is only ~5% of FLOPs at width
  1152 (share ≈ T/(8D+T)); matmuls dominate, and at small widths the 50,257-vocab
  unembedding is up to half the forward cost. So prefer one fast big-matmul card over
  exotic long-context hardware; batch size is the main utilization lever.
- **Disk**: ≥ 20 GB free (checkpoints ~100–300 MB each at width 1152, corpus 0.4 GB,
  HF cache small).

## torch.compile

Untested on these architectures (that is itself worth one arm of measurement). The
custom pieces (slots = indexed writes, bilinear s1*s2 pattern, per-module scalars,
group-lasso penalty) are all plain dense ops — expected to compile, and the small-kernel
fusion should help most below width 768. Protocol if you try it: compile the block
forward only (`torch.compile(model, mode='default')`), verify the positive control
(compiled vs eager CE on one batch agrees to ~1e-3 bf16), and on ANY graph-break spam or
NaN divergence fall back to eager — throughput is not worth a silent numeric change.
Record compiled-vs-eager step time in the JSON if you run it.

## Setup on a fresh Vast instance

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

The mini-model harness is self-contained (qk_tokenline_train.py is the shared Q module;
qk_v8_train.py layers the V8 recipe on it; qk_w1152_train.py shows the width patch +
micro-batch preflight; qk_e*_run.py are the experiment runners). bilin18 itself is NOT
needed for these runs (tier2_model/load_elriggs only matters for probing the original
model — skip it on the rented box).

## Reporting back

Commit result JSONs + runner edits + a RESULTS_l0_mdl.md section draft; push to main.
Include in every JSON: lr sweep table, final + curve CE on BOTH held sets, seq-clustered
SE vs the paired control, probe outputs (wiring Spearman, token-determined variance),
and the exact data prefix used (shard list + row range) so runs are reproducible.
