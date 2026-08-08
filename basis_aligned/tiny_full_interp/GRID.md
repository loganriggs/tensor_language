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
| 1 | 256 | 16 | local | **done** seeds 0,1,2 — CE 4.4592 ± 0.0026, interpreted |
| 2 | 32 | 2 | local | **done** seeds 0,1,2 — CE 5.3166 ± 0.0090, interpreted (tf_interp2) |
| 2 | 64 | 4 | local | **done** seeds 0,1,2 — CE 4.9124 ± 0.0054, interpreted |
| 2 | 128 | 8 | local | **done** seeds 0,1,2 — CE 4.5503 ± 0.0065, interpreted |
| 2 | 256 | 16 | local | **done** seeds 0,1,2 — CE 4.2453 ± 0.0032, interpreted. **INDUCTION APPEARS HERE** (+0.0938 ± 0.0086) |

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
| A vanilla | local | **done** seed 0 (reused checkpoint, re-measured through the same code path as B-F) — held CE 4.65117 / ladder CE 4.5545, induction −0.0138 |
| B slots | local | **done** seed 0 — CE 4.74182, **induction +0.1129**, attention-to-attention KL [0.574, 0.123] |
| C bandwidth | local | **done** seed 0 — CE 4.62626, **induction +0.0965**, [0.600, 0.150] |
| D predicate | local | **done** seed 0 — CE **4.38428** (best of the six), **induction +2.5934** from 16 named scalars, [0.352, 0.071] |
| E codebook | local | **done** seed 0 — CE 4.74797, **induction +0.0540**, [0.113, 0.108] |
| F shrink | local | **done** seed 0 — CE 4.73574, **induction +0.0510**, [0.301, 0.148] |

**Verdict (RESULTS.md FINDING 11): DIFFERENT, not a relabelling.** All five
non-vanilla variants open the attention-to-attention path the plain model shuts
(vanilla 2.4e−5 nats, variants 0.113–0.600) and all five induct at width 128
where vanilla needs 256. Every knockout is quoted as [zero, resample].

**Mechanism-decomposition and control arms** (added after the seed-0 result, all
at depth 2 width 128 seed 0 unless stated):

| arm | purpose | status |
|---|---|---|
| `slots_writeinit_only` (n_slots 1, lasso 0) | isolates the nonzero decoder init, the one confound shared by all five variants | **done** — CE 4.65758, induction −0.0095, path shut at 3.5e−6: the init explains nothing |
| `slots_nolasso` (n_slots 4, lasso 0) | isolates the in-loss group lasso from the partition | **done** — CE 4.76072, induction +0.0836: the partition does it, the lasso adds +0.029 |
| `bandwidth_slot32`, `predicate_slot32` | matched embedding (stream 128, not 160) | local:running |
| `vanilla_lr0.01/0.04`, `slots_lr0.01/0.04` | kills "it is the learning rate" | local:running |
| depth-1 cells of all five variants | matched null for the natural-text swap probe | local:running |
| seeds 1 and 2 of all five variants | the 3-seed rule | local:running |

Analysis code for this slice: `tf_interp3.py` (`VariantFold` + a variant-agnostic
ladder).  Every stage is run through ONE code path for all six variants,
including vanilla, so a variant difference cannot be a difference of analysis
code; that path is separately gated against `tf_interp2.ladder2` on the vanilla
checkpoint (positive control `tf_interp3_control.json`).

**CE bookkeeping note (found while claiming this slice).** The depth-2 CEs in the
primary-grid table above (4.5503 at w128 etc.) are the RUNG-5 LADDER's
`_model_ce` — 96 held sequences at T=256. The training-protocol held CE (1500
sequences at T=512, the number in `{stem}.json['run']['final_held_ce']`) is
5.4131 / 5.0181 / 4.6463 / 4.3254 at widths 32/64/128/256, about 0.09 nats
higher because the context is half as long. Both are correct; they are different
measurements and must not be mixed. This slice quotes both.

**Phase V2 — width sweep of the informative variants** (depth 2, seed 0;
which variants carry forward is decided by V1, not pre-committed):

V1 says every variant is informative on the routing question, so the V2 cell that
matters most is the one that dates the transition: **the plain model's induction
appears between width 128 and 256, and the slot variants already have it at 128
— so the question V2 should answer is how far DOWN the slot variants carry it.**
Priority order, unclaimed:

| cell | question |
|---|---|
| `slots` d2 w64 and w32 | how far below 128 does the partition carry induction? |
| `predicate` d2 w32 | the named term is one scalar per head; does it work at any width? |
| `slots` d2 w256 | does the advantage survive where the plain model also inducts? |
| `vanilla` d2 w192 | still the right cell for locating the plain model's own transition |

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
