> **2026-08-08: the rented box is OFFLINE and may not return. LOCAL owns the
> entire grid.** Every cell here is a tiny model — width 256 depth 4 is still
> minutes on the 16 GB card — so nothing in this program is blocked by losing
> the big machine. The one thing that IS blocked is the parent program's
> width-1152 predicate-basis confirmation, which needs 31 GB; that is noted
> in ../qk_mdl/BRAINSTORM_STATE.md, not here.

# Training grid — claim a cell by editing this file and pushing BEFORE starting

Status: `unclaimed` / `local:running` / `scale:running` / `done` (+ CE)
Head dim fixed at 16, so heads = width/16.
**Primary corpus = trained byte-level BPE at V=8192** (`tf_corpus_b8192/`,
stems `..._b8192_...`, zero UNK). The truncated-GPT-2 corpus is now a labelled
comparison arm, stems `..._v8192_...`. Fresh single-epoch protocol; 3 seeds/cell.
**Cross-tokenizer and cross-vocabulary numbers are quoted in BITS PER BYTE
only** (README.md protocol); nats/token are within-tokenizer.

## Primary grid (depths 1-2) — tok=bpe, V=8192

| depth | width | heads | owner | status |
|---|---|---|---|---|
| 1 | 32 | 2 | local | **done** seeds 0,1,2 — CE 5.4130 ± 0.0041, interpreted (RESULTS FINDINGS 1-6) |
| 1 | 64 | 4 | local | **done** seeds 0,1,2 — CE 5.0442, interpreted |
| 1 | 128 | 8 | local | **done** seeds 0,1,2 — CE 4.7234, interpreted |
| 1 | 256 | 16 | local | **done** seed 0 — CE 4.5583, interpreted (retrained locally on bpe; scale's .pt never arrived) |
| 2 | 32 | 2 | local | s0 **done** (CE 5.3628); **local:claimed 2026-08-08 05:30 — s1,s2 training** |
| 2 | 64 | 4 | local | s0,s1 **done** (CE 5.0211 s1); **local:claimed 2026-08-08 05:30 — s2 training** |
| 2 | 128 | 8 | local | **local:claimed 2026-08-08 05:30 — s0,s1,s2 training** |
| 2 | 256 | 16 | local | **local:claimed 2026-08-08 05:30 — s0 training** (scale box offline; bpe rerun) |

**Depth-2 interpretation is claimed by local (2026-08-08 05:30)**: `tf_interp2.py`
extends the depth-1 ladder to two layers (per-LAYER attention split, the
layer-1-reads-layer-0 composition test, per-layer head drops and MLP truncation,
and the ladder-ORDER reversal control demanded by the adversarial-review stage).

## Tokenizer-distortion arm (added 2026-08-08) — KEPT, not discarded

Same architecture, same V, same underlying text, DIFFERENT tokenizer: trained
byte-level BPE (0% UNK) vs the original top-K GPT-2 truncation (13.2% UNK at
V=8192). Local's cells were re-pointed from the originally queued V=4096 to
**V=8192 so the only difference from the primary is the tokenizer, not the
vocabulary size**; the scale box's already-running width-256 cells are truncated
V=8192 and therefore land in this arm exactly as-is, which is why they are kept.

The question this answers is not "which trains better" but **how much of the
interpretability picture is an artifact of the tokenizer**: do the UNK-dominated
tables have different top token pairs, a different behavioural inventory, a
different rung-5 remainder? A 13.2%-UNK corpus puts one symbol in ~1 of every 8
positions; if that does not change the fold, that is a real robustness result,
and if it does, every truncated-vocabulary interpretability paper is on notice.

| depth | width | tok | owner | status |
|---|---|---|---|---|
| 1 | 32 | trunc V=8192 | local | local:running (tf_chain2 stage 6) |
| 1 | 64 | trunc V=8192 | local | local:running (tf_chain2 stage 6) |
| 1 | 128 | trunc V=8192 | local | local:running (tf_chain2 stage 6) |
| 1 | 256 | trunc V=8192 | scale | scale:running (KEEP — relabelled into this arm) |
| 2 | 256 | trunc V=8192 | scale | scale:running (KEEP — relabelled into this arm) |

## Architecture-variant slice (Logan 2026-08-08) — the comparison that motivates the program

Same tiny setting, different explainable architectures, each interpreted in
full. At depth 1 there are 2 modules and at depth 2 there are 4, so the whole
wiring diagram is a 2x2 or 4x4 table that can be written out by hand — the
variants' claims (who reads whom, what is named, what is discrete) become
directly checkable rather than statistical.

Variants: A `vanilla` | B `slots` (partition + per-slot norm + in-loss lasso)
| C `bandwidth` (B + true-small decoders, savings into wider slots)
| D `predicate` (C + named attention terms; parent program's leader)
| E `codebook` (C + variable-k discrete slot content)
| F `shrink` (B + shrinking embedding channel with floor)

**Phase V1 — the comparison slice** (depth 2, width 128, seed 0; 6 models):

| variant | owner | status |
|---|---|---|
| A vanilla | local | unclaimed |
| B slots | local | unclaimed |
| C bandwidth | local | unclaimed |
| D predicate | local | unclaimed |
| E codebook | local | unclaimed |
| F shrink | local | unclaimed |

**Phase V2 — width sweep of the informative variants** (depth 2, seed 0;
which variants carry forward is decided by V1, not pre-committed):

| variant | widths 32/64 | width 256 |
|---|---|---|
| (decided by V1) | local | local |

**Phase V3 — seeds** (3 seeds on whatever V1/V2 leaves standing; the parent
program's rule is that no structure claim survives a single seed).

The question V1 answers is NOT which variant wins on loss — at this size that
is nearly meaningless. It is: **do the architectures that claim to be more
interpretable actually compute the same thing by different means, or do they
compute something different?** Same-solution-different-encoding and
different-solution are distinguishable here because both models are fully
folded: compare materialized tables, behavioural inventories, and the rung-5
reconstruction remainders directly.

## Depth ladder (after the primary grid has first results)

| depth | width | owner | status |
|---|---|---|---|
| 3 | 64 | LOCAL | unclaimed |
| 3 | 128 | LOCAL | unclaimed |
| 3 | 256 | LOCAL | unclaimed |
| 4 | 64 | LOCAL | unclaimed |
| 4 | 128 | LOCAL | unclaimed |
| 4 | 256 | LOCAL | unclaimed |

## Baselines (matched-optimizer, per width — REQUIRED before quoting any cost)

| kind | widths | owner | status |
|---|---|---|---|
| bigram table (closed form, no training) | n/a | local | **done** b8192 2.030 bits/byte, b4096 2.137, v8192 2.050 honest, v4096 2.115 honest |
| unigram floor | n/a | local | **done** b8192 2.806 bits/byte, b4096 3.020, v8192/v4096 2.493 honest |
| same-size softmax+GELU transformer | 32-256 | local | unclaimed |

The softmax/GELU baseline answers the first question a reviewer will ask:
what does the no-softmax bilinear architecture cost in prediction quality,
at each size? If the gap grows with size, the fold's tractability is being
bought with capability and that must be reported alongside every result.

## Vocab check

| vocab | owner | status |
|---|---|---|
| bpe 16384 at width 256 depth 2 | scale | unclaimed (corpus not built; one `tf_tokenizer.py` call) |
| bpe 4096 vs bpe 8192 same cell (vocab-size check) | local | unclaimed |
| bpe 2048 low-end probe | local | measured as a tokenizer only (2.899 bytes/token — a 512-token sequence would see 1484 bytes, which changes the task); no corpus built |

Answers whether conclusions are vocabulary-size artifacts, SEPARATELY from the
tokenizer-distortion arm above (which holds V fixed).  Both BPE corpora
(`tf_corpus_b8192/`, `tf_corpus_b4096/`) are ALREADY BUILT, so these cells cost
only their training runs.  Every comparison here crosses vocabularies, so it is
quoted in bits/byte.

## Corpus (rebuilt 2026-08-08 with a trained BPE) — read before quoting any CE

Both boxes regenerate byte-identical data with
`python tf_tokenizer.py corpus 8192 4096`; no shards are committed. The identity
gates are the per-shard sha256 in `tf_corpus_b{V}/MANIFEST.json` plus
`tokenizer_sha256` for the committed `tf_bpe_{V}.json`. The text is recovered by
decoding the parent program's GPT-2 shards
(`../qk_mdl/corpus_fresh/shard00..06.npy`), which is exact — control C1 shows
ids → text → ids is bit-identical.

| split | rows | purpose |
|---|---|---|
| train | 240,000 | single epoch = 15,000 steps x batch 16, each row visited once |
| held | 6,000 | pure eval, never trained (first 1,500 used for the paired eval) |
| est | 30,000 | table fitting / smoothing-parameter tuning |
| spare | 24,000 | lr sweeps only, so no model touches train before its epoch |

Same row counts and the same disjoint source-row (hence text) regions as the
truncated build. The BPE stream is ~18% longer than GPT-2's on the same text, so
every region yields MORE than its target rows (train: 283,648 available at
V=8192, 319,279 at V=4096) and we take the target-row prefix — the single-epoch
arithmetic is untouched and every split now carries slack.

**UNK is 0.000 by construction** (256-byte initial alphabet). What must be
quoted with every CE instead is the tokenizer and its bytes/token: V=8192 BPE is
3.755 bytes/token, so a 512-token sequence sees 1,922 bytes of text, against
2,277 bytes for the truncated arm — whose extra reach is illusory because 13.2%
of its tokens are unreadable. **No CE is compared across tokenizers except in
bits/byte**; `tf_baselines_*.json` carries both units, and the truncated arm
additionally carries `unk_repair_bits_per_byte` (0.442 at V=8192) without which
its bits/byte is not a code length for the text at all.

## Architecture variants (code ready, runs not queued by local)

`tf_model.py` takes `variant` as a single config field: `vanilla`, `slots`,
`bandwidth`, `predicate`, `codebook`, `shrink`, each transcribed from the parent
program. All six pass the fold identity gate and all five reductions to their
parent variant are **bit-exact (0.0)**. See MAILBOX.md for the config convention.
