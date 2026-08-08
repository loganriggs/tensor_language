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

**Phase V1 — the comparison slice** — **COMPLETE, three seeds each, plus 19
control and robustness arms, all through one analysis revision.** Consolidated
table: `tf_consolidated_table.md` and RESULTS.md FINDING 11. Mean ± sd over
seeds 0/1/2:

| variant | owner | status |
|---|---|---|
| A vanilla | local | **done** s0/1/2 — CE 4.6463 ± 0.0075, induction −0.0034 ± 0.0099 (below its own floor at every seed, and at Muon 0.01/0.02/0.04), routing [1.3e−5, 4.3e−6] |
| B slots | local | **done** s0/1/2 — CE 4.7414 ± 0.0056, **induction +0.0972 ± 0.0275** (3/3 above floor), routing [0.551, 0.124] |
| C bandwidth | local | **done** s0/1/2 — CE 4.6279 ± 0.0037, **induction +0.1190 ± 0.0524** (3/3), routing [0.538, 0.144] |
| D predicate | local | **done** s0/1/2 — CE **4.3861 ± 0.0020** (best of the six), **induction +2.6402 ± 0.0481** (3/3) but HANDED OVER by 16 named scalars, not learned, routing [0.353, 0.070] |
| E codebook | local | **done** s0/1/2 — CE 4.7542 ± 0.0054, induction +0.0375 ± 0.0157 (**2/3** above floor; seed 2 is +0.0228 against a 0.0249 floor), routing [0.119, 0.096] |
| F shrink | local | **done** s0/1/2 — CE 4.7243 ± 0.0100, **induction +0.0860 ± 0.0303** (3/3), routing [0.229, 0.143] |

**Verdict (RESULTS.md FINDING 11, as corrected by the independent round-2
review in `tf_reviewer_round_2.json`): DIFFERENT, not a relabelling** — but the
routing half is a MAGNITUDE result, not a channel-opening one. All five variants
*use* a residual route that carries ~nothing in the plain model (vanilla 1.3e−5
nats, variants 0.119–0.551), and four of five induct at width 128 where vanilla
needs 256 at all three seeds (codebook at two of three). What is **retracted** is
the inference that the plain model's weights close that route: under a
matched-displacement probe the plain model is the *most* sensitive of the six to
layer-0 attention's direction, and its ~zero transmission is entirely explained
by its own write being renormalised down to 0.3% of layer 1's read. Every
knockout is quoted as [zero, resample].

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

**CLAIMED BY LOCAL 2026-08-08 (all six cells, three seeds each = 18 models).**
Registered predictions written BEFORE the first training step:
`tf_depth_ladder_predictions.json`. Chain: `tf_depth_ladder_chain.sh`.
Analysis is `tf_interp3.py` **verbatim** — the same code path that produced the
depth-1/2 and six-architecture numbers — and the depth-1/2 vanilla cells that
had no `_interp3.json` are backfilled through that same path in the same chain,
so the whole ladder is one code path end to end.

| depth | width | owner | status |
|---|---|---|---|
| 3 | 64 | LOCAL | **seed 0 done + interpreted** — CE 4.9417, induction +0.0077 (BELOW its 0.0109 floor); seeds 1,2 running |
| 3 | 128 | LOCAL | **seed 0 done + interpreted** — CE 4.5285, **induction +0.0974** vs floor 0.0078; seeds 1,2 running |
| 3 | 256 | LOCAL | **seed 0 done + interpreted** — CE 4.2182, **induction +0.1642** vs floor 0.0156; seeds 1,2 running |
| 4 | 64 | LOCAL | **seed 0 done + interpreted** — CE 4.8817, **induction +0.0173** vs floor 0.0133 (1.3× the floor, the one marginal cell); seeds 1,2 running |
| 4 | 128 | LOCAL | **seed 0 done + interpreted** — CE 4.4601, **induction +0.1264** vs floor 0.0112; seeds 1,2 running |
| 4 | 256 | LOCAL | **seed 0 done + interpreted** — CE 4.1436, **induction +0.3019** vs floor 0.0137; seeds 1,2 running |

**Seed-0 verdicts (RESULTS.md FINDING 14; PROVISIONAL until seeds 1 and 2 land,
which the same chain is producing).** The induction width threshold falls **one
octave per layer** — 256 at depth 2, 128 at depth 3, 64 at depth 4 — and at
depth 3 the **attention-to-attention route opens** (layer-1 attention into layer
2's read is 17–39% of the dominant MLP term, against 1e−5 at depth 2) **and the
induction circuit moves onto it** (cutting it removes 94.5% of the induction
score at depth 3 width 128 while the bag control does not fall). Layer-0
attention stays at ~1e−6 into every downstream read at every depth and width, so
the shut channel is the FIRST attention block specifically, not
attention→attention in general. Registered P2 is refuted; registered P1 is
confirmed in its main clause and refuted in its last; P4 (order dependence grows
with depth) is refuted — it peaks at depth 2.

## Compressibility across the grid (Logan's question: artifact or family property?) — CLAIMED BY LOCAL 2026-08-08

One scalar per cell: the ratio of the best description's bits to the SAME
weights naively quantised per row + entropy coded, at a matched score, held CE
primary and KL secondary. FINDING 12 §7b measured it at exactly one cell (depth
1 width 128: 1.13–1.54, median 1.20). This claims the whole grid — widths
32/64/128/256 at depths 1 and 2, plus the depth-3/4 cells above — so that "the
structure does not compress" can be attributed either to the family or to the
smallest model. Registered prediction P5–P7 in
`tf_depth_ladder_predictions.json`. Machinery: `tf_compress_grid.py` (a
DEPTH-GENERAL decoder; `tf_compress.D1Desc` asserts depth 1).

| cells | owner | status |
|---|---|---|
| d1 × w{32,64,128,256}, d2 × w{32,64,128,256}, seed 0 | LOCAL | **done** |
| d3/d4 × w{64,128,256}, seed 0 | LOCAL | **done** except d4 w256 (chain picks it up on its next pass) |
| seed replication (seeds 1,2) at the extreme widths | LOCAL | local:running |

**Verdict (RESULTS.md FINDING 15): the ratio SHRINKS with model size.** Slope
−0.042 ± 0.009 per e-fold of parameters (t = −4.9, 13 cells); R falls from 1.162
at width 32 depth 1 to 0.987 at width 256 depth 2, i.e. at the largest cell the
best description we can build is **worse in bits than bit-packing the same
weights** at the same held CE. Registered P5's falsifier (growth) is rejected
with a wide margin, so **"structure does not compress" is a property of this
architecture family, not an artifact of the smallest model, and it gets stronger
with size.** P7 confirmed 13/13: restricted to descriptions made out of an
interpretation the ratio is 0.75–0.87 everywhere, and **no structural scheme
appears anywhere on the overall frontier at any cell.** The three ways this
could have been a size artifact — the naive denominator's per-row scale overhead
(1.0 bits/weight at width 32, 0.125 at width 256), the falling embedding share,
and a fixed absolute KL being a different difficulty at each width — are each
measured and each fail to explain it.

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
