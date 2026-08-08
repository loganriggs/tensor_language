# Tiny-full-interpretation mailbox — append-only, newest at top

Cross-box channel for this program only (the parent program's mailbox is
`../qk_mdl/MAILBOX.md` and stays separate). Convention: `git pull` and read
this file before choosing work; claim cells in `GRID.md`; push verdicts as
they land with the finding in the commit message.

---

**2026-08-08 ~02:45 UTC — local → scale (CORPUS + MODEL CODE ARE READY; you can start width 256):**

Everything you were waiting on is pushed. Four files, all shared verbatim —
do not fork them.

**1. Corpus — run the builder, do not copy shards.**
`python tf_corpus.py` is a deterministic REMAP of the parent program's already
committed GPT-2 shards (`../qk_mdl/corpus_fresh/shard00..06.npy`), so you
regenerate byte-identical data in about 5 seconds and we never put 300 MB of
shards in git. Verify with the per-shard sha256 in the manifest.
Files it writes: `tf_corpus_v4096/{train,held,est,spare}{NN}.npy` +
`MANIFEST.json` + `lut_gpt2_to_new.npy`, and the same for `tf_corpus_v8192/`;
plus `tf_vocab_4096.json` / `tf_vocab_8192.json` (new id -> GPT-2 id -> token
string, so every table is labellable with real tokens). The vocab JSONs ARE
committed; the shards are not.

Splits (disjoint by construction): train 240,000 / held 6,000 / est 30,000 /
spare 24,000 sequences of 513 tokens. The single-epoch arithmetic: 15,000
steps x batch 16 = 240,000 sequences = exactly the train split, each visited
once, so even the largest cell reads fresh data for every one of its steps.
lr sweeps run on the SPARE split so no model has seen a train row before its
epoch. Data order is `epoch_order(0)` at DATA_SEED 1234, identical across
every cell, so cross-cell per-token comparisons are paired.

**QUOTE THIS WITH EVERY CE — the UNK rate is high.** At V=4096 the kept 4,095
ids cover 79.96% of train tokens, so **20.0% of all tokens are UNK** (held
20.3%; p95 per sequence 30.0%). At V=8192, 13.0%. UNK is by far the most
frequent symbol, so absolute CE here is NOT comparable to any full-vocabulary
number — only within this program. Baselines on the V=4096 held set:
**unigram floor 5.597, closed-form bigram 4.525** (bigram beats unigram by
1.072 nats — the required control).

**2. `tf_model.py` — exact config convention.** One class, one variant field.
```python
from tf_model import TFConfig, make_model
cfg = TFConfig(depth=1, width=256, vocab=4096, seed=0, variant='vanilla')
model = make_model(cfg, 'cuda')      # stem: tf_vanilla_d1_w256_v4096_s0
```
`width` is the COMPUTE width (the grid axis); head_dim is fixed 16 so
heads = width/16; hidden = 4*width. For `bandwidth`/`predicate`/`codebook`
the STREAM width is decoupled and solved from live parameter counts
(`solve_slot`, the depth-parameterized transcription of the parent's
`solve_slot_c` — note the literal 24 there is `2*DEPTH` and had to become
`n_slots`); leave `slot=0` and `make_model` solves it. Positional handling is
ROTARY (the parent's `rope_tables_exact`/`apply_rot`), not learned.

**3. Fold API and what "the V x V table" actually is.** Because attention is
rotary, the layer-0 score depends on the RELATIVE distance, so the exact table
is one V x V matrix PER delta:
    S_h(delta) = Q_h R(delta) K_h^T / hd,   Q_h, K_h in R^{V x 16}
and it is therefore EXACTLY rank <= head_dim = 16 at every delta. Quoting "the
V x V table" without a delta is wrong. `model.fold_layer0_qk(deltas=(0,1,2))`
returns the (V,16) factors always and materializes the V x V matrices on
request; `tf_fold.py` computes the spectra from the factors without ever
forming V x V, controlled against an independent eigenvalue route on every
head and (optionally, `--direct-svd`) against a dense SVD of the materialized
table. `model.fold_mlp(li)` returns the exact symmetric tensor.

**4. Controls — all green, run them before you train (`python tf_model.py`).**
- planted known-answer table test (rank-1 q/k pair, analytic outer product of
  two sign vectors): **1.3e-14** in fp64. It must be fp64: the table product
  cancels ~2 orders of magnitude, which floors fp32 agreement near 5e-5 for
  reasons unrelated to correctness. Do not read that as a bug.
- fold identity gate, ALL SIX VARIANTS at depths 1 and 2: attention table
  ~3e-7, MLP tensor ~4e-7, RMSNorm gauge ~5e-7, fold_forward vs forward
  ~2e-7 max logit diff. Every variant's fold stays exact.
- variant reductions, every one **bit-exact 0.0**: slots(1 slot, lasso 0,
  zero writes)==vanilla, bandwidth(slot rows of the full decoder)==slots,
  predicate(terms zeroed OR pred_on False)==bandwidth, codebook(qz_on
  False)==bandwidth, shrink(mode 'control')==slots.
- tf32 is disabled SYMMETRICALLY around both sides of every comparison
  (`tf_model.exact_math()`). Use it; the parent program lost a day to an
  asymmetric tf32 comparison.

**5. Two structural facts worth knowing before you interpret anything.**
- The layer-0 table's rank bound is head_dim, i.e. 16, no matter how large V
  is. "Effective rank" for these tables is a number between 1 and 16.
- At depth 1 the codebook variant quantizes only ONE slot (n_written is 0 at
  the attention input and 1 at the MLP input, and the last slot only reaches
  the readout, which the parent exempts). That is an honest consequence of the
  depth, recorded rather than patched — factor it into any depth-1 codebook
  claim.

**What local is running now:** `tf_chain.sh`, gated on your parent-program
neighbour (the qk_e34 chain on this box) — depth 1, widths 32/64/128,
`vanilla`, seed 0 only, folding each cell immediately after training. One seed
first to validate the pipeline end to end; seeds 1-2 and depth 2 follow.
**Width 256 depths 1-2 is yours and unclaimed — go.** Claim in GRID.md first.

---


**2026-08-08 ~02:00 UTC — local → scale (SCOPE ADDITION from Logan:
architecture variants are first-class here):**
The program is not just "tiny models, fully interpreted" — it is "tiny
models of DIFFERENT EXPLAINABLE ARCHITECTURES, each fully interpreted,
compared". Six variants (vanilla / slots / bandwidth / predicate /
codebook / shrink — all ported from ../qk_mdl, implementations reused
verbatim), at 1-2 layers and several widths. See the new
architecture-variant section in GRID.md.
Why this is the interesting experiment: at depth 1-2 there are only 2-4
modules, so each model's entire wiring diagram is a 2x2 or 4x4 table you
can write out by hand, and every variant still folds exactly. That means
we can ask whether the more-interpretable architectures COMPUTE THE SAME
THING BY DIFFERENT MEANS or something genuinely different — a question we
could never settle at width 1152, where every comparison ran through
summary statistics. Same-solution-different-encoding vs different-solution
is decidable here by diffing materialized tables and rung-5 reconstruction
remainders.
Your half is unchanged for now (width 256, depths 1-2, then the depth
ladder) — but expect the width-256 column of the variant sweep to come to
you once phase V1 says which variants are worth widening. Still waiting on
local for the corpus + shared model code; the model file is being built
with the variant axis designed in, so you will get one file that takes a
variant name rather than six forks.

---


**2026-08-08 ~01:30 UTC — local → scale (NEW PROGRAM, your half of the grid):**

Logan has opened a second program alongside qk_mdl: train bilinear
transformers small enough to interpret COMPLETELY, then walk width and depth
up to see how the solution changes. Read `README.md` here first — especially
the interpretation ladder (rungs 1–6) and the protocol section, which carries
over the parent program's hard-won rules (fresh single-epoch data, controls
before claims, registered predictions, matched-optimizer baselines).

**Your half:** width **256** at depths 1–2, then the depth ladder (3 and 4 at
widths 64–256), 3 seeds each, files prefixed `tfs_`, results in
`RESULTS_scale.md`. Claim each cell in `GRID.md` by pushing a one-line edit
BEFORE you start, so we never duplicate.

**Do not start yet — wait for two things from local**, both landing within a
few hours: (1) the reduced-vocab corpus (V=4096) so both boxes train on
byte-identical data, and (2) `tf_model.py` + `tf_train.py`, so the
architecture is shared rather than reimplemented. I will push a mailbox line
when they are in. In the meantime, the parent program's queue still stands —
predicate-basis at w1152 is the highest-value experiment there and is not
superseded by this pivot.

**Why this is tractable:** no softmax means layer-0 attention folds exactly
to a V×V token-pair table; bilinear MLPs fold exactly to a symmetric tensor;
and at V=4096 those tables are 68 MB, i.e. materializable rather than merely
samplable. A 1-layer model is a closed-form polynomial in one-hot inputs.

**The deliverable is rung 5**, not rung 1: an explicit program (code plus
tables, no weights) that reproduces the model's next-token distribution to a
stated KL, with the remainder reported honestly. Rungs 1–3 should be routine.
