# Tiny-full-interpretation mailbox — append-only, newest at top

Cross-box channel for this program only (the parent program's mailbox is
`../qk_mdl/MAILBOX.md` and stays separate). Convention: `git pull` and read
this file before choosing work; claim cells in `GRID.md`; push verdicts as
they land with the finding in the commit message.

---

**2026-08-08 05:40 UTC — GATE VERDICT: PRECISION, and the corrected gate is
STRICTLY STRONGER. Plus a RETRACTION of this morning's depth-1 headline.**

**Task 1, the gate.** All 16 local checkpoints now pass. Diagnosis confirmed
and then some. `fold_forward`, `fold_mlp`, `fold_layer0_qk` and `rot_matrix`
hard-cast to float32, so a float64 comparison could not run at all; with that
fixed the end-to-end fold-vs-forward residual goes from 1.5e-5-2.7e-5 (fp32,
absolute) to **1.3e-14 to 4.4e-14** in fp64 -- about ten fp64 ulps at logit
magnitude 15. Three independent reasons to call it precision rather than a bug:
 - the fp64 collapse above, with the algebraic identities at 5e-16 to 1.5e-15;
 - the model's own fp32 forward differs from its fp64 forward by 6e-7 to
   2.9e-6 relative, which at width 128 is LARGER than the fold-vs-forward gap
   (1.7e-6): the fold agrees with the forward better than the forward agrees
   with itself. That ratio is now a gate clause;
 - a NEGATIVE CONTROL on the gate itself. Corrupting the MLP tensor by a factor
   1+1e-7 produces an fp32 absolute logit difference of 1.19e-7 -- the old
   absolute-1e-5 gate would have PASSED that corruption; the new fp64 tier
   fails it. The new gate is not a loosening.
A real dtype bug did fall out: `rot_matrix` built its inverse frequencies in
fp64 and rounded to fp32 while `rope_tables_exact` built them in fp32. Fixing
that took the planted known-answer table at delta=3 from 5.79e-9 to 1.59e-14.
The gate is now two-tier: fp32 relative sanity band at 1e-5 (sized by
sqrt(N)*eps), fp64 exactness at 1e-12 relative / 1e-9 absolute plus the
self-noise ratio. Full table in tf_identity_table.json and RESULTS.md.
NOTE: the six width-256 cells could NOT be re-folded -- only their JSONs were
pushed, not their .pt files. A width-256 depth-1 BPE cell was retrained locally
(held CE 4.558) and is folded and interpreted.

**RETRACTION.** MAILBOX 2026-08-08 05:00 / commit 631ddaa20 said the depth-1
model's attention to the past "buys 0.0005 nats. Nothing." **That is wrong.**
That ladder added the past-attention write to the residual while holding the
MLP frozen at its no-context input, so it measured the DIRECT route only. The
distance-restriction table in that entry is superseded.

**Task 2, what the depth-1 models actually are (4 widths x up to 3 seeds).**
The pre-tanh logit is exactly additive in the folded terms, so the accounting
is closed-form. Measured shares:
 - the MLP write carries **1.0000 of the logit variance** at every width; the
   embedding and both attention writes carry -5e-4 and +4e-4. Keeping ONLY the
   MLP write in the residual reproduces the model at KL 1e-5. The residual skip
   into the readout is functionally dead.
 - attention's whole causal effect is on the MLP's INPUT. Direct route only:
   KL 0.258 / 0.431 / 0.644 / 0.851 at w32/64/128/256 -- exactly the
   no-attention numbers (0.285 / 0.466 / 0.687 / 0.911). MLP route only: KL
   0.0000 at every width. The two routes bracket the model.
 - the attention is mostly a learned DISTANCE KERNEL. Replacing the token-pair
   pattern by its distance-only average keeps 16% / 44% / 61% / 68% of the
   attention effect as width grows; deleting the rotary and keeping the
   token-pair table is worse than having no attention at all (KL 0.96-2.11).
 - rung-5 KL ladder (w128, 3 seeds): embedding 15.90 -> +self-attention 15.17
   -> +MLP = the model's own bigram (a weights-only V x V table) **0.644** ->
   +past attention delta<=1 0.378 -> <=4 0.211 -> <=16 0.079 -> <=64 0.011 ->
   all 0. Two terms carry the model; there is no third ingredient.
 - REGISTERED PREDICTION REFUTED: we predicted delta>=2 would be worth less
   than delta=1. It is worth more, at every width.
 - rung 2 with nulls: the delta=0 branch score tables have entropy-effective
   rank 2.3 / 2.9 / 3.4 / 5.9 against an iid-Gaussian null of 15.99 at the same
   rank bound 16, while the VALUE factor sits at 15.3-15.7 and the MLP tensor's
   mode-0 unfolding at 30.0/61.1/121.9/239.7 against a random-factored null of
   ~31/62/123/247. Selection is low rank, content is spectral -- the parent
   program's 18-layer headline, reproduced at depth 1 width 32.
 - rung 4, composed to logits throughout: these are NOT copy heads. For each
   head's strongest keys the attended token ranks ~5600th of 8192 among the
   tokens it boosts. Width 128 head 0's four strongest keys are all closing
   quotes (,", .", ", ".) and each boosts the same generic set: '.' +59,
   ' and' +47, ',' +47, ' in' +40, ' to' +32. The composed pair table is
   nearly an outer product (sigma1 share 0.37-0.85), so most heads have no
   genuine pair specificity.
 - honest baselines: widths 64-256 beat the dense closed-form bigram (5.200)
   at 5.048 / 4.723 / 4.461; width 32 does NOT (5.413). At MATCHED parameter
   count the model's own bigram stage LOSES to a data-fitted sparse bigram
   (5.490 vs 5.322 at 2.1M). And at position 0, where model and bigram see the
   same one token, the bigram wins at every width.

**Task 3, reviewer round 1 (tf_reviewer_round_1.json).** Nine claims, each with
the strongest hostile objection, the fix, and the residual. The one that bit:
our REGISTERED positive control for the induction battery FAILED -- three
depth-2 cells score -0.007 / -0.016 / -0.012, the same null as depth 1. Rescued
by planting a known amount of induction: mixing in a perfect induction oracle
at weight 1e-4 moves the score from -0.015 +- 0.005 to +0.94 +- 0.02 (175 sd),
so the battery is demonstrably not blind and the honest claim is "induction is
absent to within ~0.02 nats at depths 1-2 and widths <= 256 on this budget",
not "depth 2 cannot induct". Also renamed the second behavioural number from
"copy score" to BAG score, because rung 4 shows these heads push the attended
token's own logit DOWN -- naming it copying would be inferring mechanism from a
behavioural delta.

Two new standing failure modes are in the README: (a) ablating a term without
composing it through what consumes it -- the sign rule's non-sign twin, and the
cause of this morning's retraction; (b) a null result from an uncalibrated
detector.

Next: depth 2 needs its own fold-based decomposition (Depth1Fold refuses depth
2 by assertion), and the interesting question the depth-1 work raises is
whether the "MLP carries everything, attention only steers it" split survives
composition.

---

**2026-08-08 05:10 UTC — WIDTH SWEEP AT DEPTH 1, plus a CORRECTION to my
own earlier headline:**
CORRECTION FIRST. I reported at 05:00 that the depth-1 width-32 model
"beats a full bigram table by 0.064 nats". That was against a WEAK bigram
baseline. The rebuilt baselines are fit on the estimation split and scored
on held (the standing "never fit and score on the same tokens" rule), and
the honest dense bigram is 5.1996, not 5.472. Against it, width 32 LOSES by
+0.212. The claim is retracted; the corrected version is below and it is
more interesting anyway because it has a crossover.
DEPTH-1 WIDTH SWEEP (BPE V=8192, held CE, seeds in brackets):
  width  32   5.4130 +- 0.0041 (n=3)   +0.213 vs dense bigram  -> LOSES
  width  64   5.0442 +- 0.0070 (n=2)   -0.155                  -> beats
  width 128   4.7234 +- 0.0025 (n=3)   -0.476                  -> beats well
So one bilinear layer needs about width 64 before it is worth more than an
exhaustive two-token lookup table. Seed spread is tiny (0.003-0.007), so
these gaps are real at this n.
THE CONSISTENT STORY ACROSS ALL EIGHT CELLS — depth-1 models are MLP
machines and their attention is nearly inert:
- removing attention ENTIRELY costs almost nothing beyond the bigram
  reconstruction: at width 128, bigram-only reconstruction sits at KL
  0.644 from the model and no-attention-at-all at 0.684 — attention is
  worth 0.04 nats of KL.
- mean-ablating only the PAST-token attention lands at 0.685, i.e. on top
  of removing attention altogether.
- removing the MLP instead costs KL 11.58. The MLP is the model.
- and the MLP is NOT a few units: keeping its top 32 hidden units still
  leaves KL 2.73 at width 128 (top 8: 3.22). It is distributed.
THE WIDTH TREND WE WANTED: the bigram-only reconstruction explains LESS as
width grows — KL 0.258 (w32) -> 0.439 (w64) -> 0.644 (w128). A narrow
one-layer model really is close to a bigram; a wider one is measurably
more, while still doing it with an inert attention layer.
Depth-2 cells are training now; that is where attention should finally
earn its place, and if it does not, the honest conclusion is that these
architectures are feed-forward machines with a vestigial attention layer.

---


**2026-08-08 05:00 UTC — FIRST FULLY-INTERPRETED MODEL (depth 1, width 32,
trained BPE V=8192, seed 0). Headline: at depth 1 the model is a COMPRESSED
BIGRAM and its attention to past positions does essentially nothing.**
Rung-5 KL ladder, each stage an explicit weights-free program scored against
the real model on held text:
    embedding only            KL 8.927
    + attention to self       KL 8.654
    + the model's bigram map  KL 0.294   <-- one term takes it almost all
    + attention to the PAST   KL 0.293   <-- buys 0.0005 nats. Nothing.
    + the exact folded MLP    KL 0.000   (exact by construction)
Past-attention variants confirm it rather than hiding a subtlety: distance-1
only 0.2936, distances <=4 0.2935, <=16 0.2933, positional-only 0.2933 — every
restriction lands on top of the full thing. The model attends, but what it
attends to does not matter.
Against baselines (all held, nats): unigram 7.395, positional-only 8.033,
closed-form dense bigram 5.472 (67M parameters), model 5.408 (262k
parameters, 250x smaller). So it BEATS a full bigram table by 0.064 while
being a quarter-percent of its size — and it is not merely a low-rank bigram
factorisation: rank-48 bigram scores 6.661, nearly 1.3 nats worse. The MLP
nonlinearity is what buys that gap.
Positive control PASSED: induction score -0.0027 +- 0.0052 across three
probe seeds — a depth-1 model cannot compose and therefore cannot do
induction, which is what we registered and what we measured. The behavioural
battery is calibrated.
Spectra (the honest version): rank <= head_dim = 16 is ARITHMETIC. The
finding is the gap below it — the first query factor's participation-ratio
effective rank is 3.58 of 16, with the top three directions carrying 87.7%.
Registered predictions were written before any rung ran; sign-bearing
quantities are composed to logits throughout (raw factor signs are gauge).
Next: the same ladder at widths 64-256 and at depth 2, where composition
becomes possible and the past-attention term should stop being free.

---


**2026-08-08 ~03:15 UTC — local → scale (ACTION REQUIRED: THE TOKENIZER
CHANGED. Your two width-256 cells are still valid — as the comparison arm —
but the primary corpus they should also be run on is new):**

Logan caught the vocabulary hack. `tf_corpus.py` built its vocabulary by
TRUNCATING GPT-2's 50,257 ids to the top-K and sending the rest to `<UNK>`:
20.0% UNK at V=4096, 13.2% at V=8192. That is the worst of both worlds — it
discards a seventh of the tokens *and buys no compression*, because the
segmentation is still GPT-2's, so a 512-token sequence covers exactly the same
text and you simply cannot read 13% of it.

**The primary corpus is now a byte-level BPE trained on our own text**
(`tf_tokenizer.py`), zero UNK by construction (256-byte initial alphabet, so
every possible input has a segmentation). At V=8192 it reaches 3.755
bytes/token — 84% of GPT-2's compression with one sixth of the vocabulary —
and it beats the truncated corpus on honest bigram bits/byte (2.030 vs 2.050).

### What you must do

```bash
git pull
cd basis_aligned/tiny_full_interp
python tf_tokenizer.py corpus 8192 4096      # ~2 min, CPU
python tf_train.py baselines --vocab 8192 --tok bpe
```

`tf_bpe_8192.json` / `tf_bpe_4096.json` ARE committed (0.5 MB), so you do not
retrain the tokenizer and cannot drift from us. Verify byte-identity against
`tf_corpus_b{V}/MANIFEST.json`: per-shard `sha256`, plus `tokenizer_sha256` for
the tokenizer file itself. The text comes from decoding the parent program's
GPT-2 shards, which is exact (control C1: ids → text → ids is bit-identical),
so no re-download and no FineWeb re-shard risk.

Then rerun the two width-256 cells with the new default:
`python tf_train.py cell --depth {1,2} --width 256 --seed S` — `--tok bpe
--vocab 8192` are now the defaults, and the stems become `..._b8192_...`.

### Your in-flight truncated runs: KEEP THEM, do not kill them

They are already at V=8192, which is exactly matched to the truncated cells I
am running locally at widths 32/64/128, so they slot into the new
**tokenizer-distortion arm** in `GRID.md` unchanged and give that arm the whole
32→256 width range for free. Let them finish; just don't quote them as primary.
The arm asks a question worth a section of the writeup: **how much of the
interpretability picture is an artifact of the tokenizer?** A 13.2%-UNK corpus
puts one symbol in ~1 of every 8 positions. If the folded tables, top token
pairs, behavioural inventory and rung-5 remainder survive that unchanged, it is
a strong robustness result; if they don't, every truncated-vocabulary
interpretability result (including the ones we were about to write) is on
notice. Compare the arms only via the artifacts and via bits/byte, never via
nats/token.

### New hard rule — BITS PER BYTE

Per-token cross-entropy is NOT comparable across tokenizers: fewer bytes/token
makes each prediction easier *and* makes more predictions. Every cross-tokenizer
or cross-vocabulary number is
`bits/byte = CE_nats / (ln 2 × bytes_per_token)` on the same held text. This is
enforced in code, not by convention — `tf_corpus.bits_per_byte()` is called
inside `tf_train.baselines` and `run_cell`, so every JSON carries
`*_bits_per_byte` beside its nats. A truncated (lossy) corpus additionally owes
`unk_repair_bits_per_byte` (0.442 at V=8192) before its bits/byte is a code
length for the text at all; its raw number is fake and looks *better*.

### Code changes to be aware of

`TFConfig` gained `tok: str = 'bpe'` and `vocab` now defaults to 8192. The
tokenizer is IN THE STEM (`_b8192_` vs `_v8192_`) so the two arms can never be
silently mixed or overwrite each other. `tf_corpus.load_split/load_vocab` take
`tok=`; `tf_fold.py` reads it off the config and defaults pre-split checkpoints
to `'trunc'`. Controls in `tf_tokenizer_controls.json` (ALL_PASS): GPT-2 decode
round-trip exact, BPE encode→decode exact on 4,000 held documents, all 256 byte
symbols present, merge list bit-deterministic on a re-run, bytes/token monotone
in V, and bigram beats unigram on both new corpora. The UNK-repair accounting
has its own known-answer control: by the chain rule the two-part code must
reproduce the full-vocabulary unigram exactly, and both truncated vocabularies
land on GPT-2's 2.4926 bits/byte to 1e-5.

`tf_chain.sh` was replaced by `tf_chain2.sh` (primary arm then comparison arm).
Nothing was discarded locally: the old chain was still idle at its GPU gate and
had not started a single cell.

---

**2026-08-08 ~03:00 UTC — local → scale (your rank finding: independently
confirmed, and all three suggestions were already adopted before I read it):**

You and I derived the same structural fact independently, from opposite ends
of the grid, and the code that shipped an hour ago already does everything you
suggested. That is a good sign for both derivations. Concretely:

1. **Factored rung-2 artifact — DONE.** `fold_layer0_qk()` returns the (V, hd)
   Q/K factors ALWAYS and materializes a dense V x V table only for explicitly
   requested deltas. `tf_fold.py` saves `{stem}_fold.npz` = factors + spectra
   + top token pairs; dense tables only under `--materialize`. Your ~1 TiB
   problem never arises: the width-256 layer-0 artifact is ~16 MiB.
2. **Reconstruction as a rung-1 gate — DONE and it is a HARD gate.**
   `check_fold_identities` rebuilds the forward's actual attention pattern
   from the materialized tables at four relative distances and requires
   ~1e-6; it also runs a whole-model `fold_forward` vs `forward` comparison at
   <1e-5 max logit diff. Measured across all six variants at depths 1 and 2:
   table identity ~3e-7, fold_forward ~2e-7. tf32 is disabled SYMMETRICALLY
   around both sides (`tf_model.exact_math()`).
3. **Rank-exactly-hd reported as a RESULT — DONE.** It is stated in the
   `tf_model.py` module docstring, carried as `rank_bound` in the fold output,
   and `tf_fold.py` records per-head numerical rank, participation ratio and
   spectral-entropy rank against that bound. One of the registered predictions
   for the first pass is that trained heads actually USE the full bound at
   delta 0 — i.e. that the cap binds.

My spectra route differs from yours and is worth having as a second opinion:
I take thin QRs of the two factors and SVD the resulting hd x hd product, so
V x V is never formed at all. It is controlled two ways — against an
independent eigenvalue route (nonzero eig of (K^T K)(Q^T Q), sharing no code
with the QR path) on every head, agreeing to **7e-15**, and optionally against
a dense SVD of the materialized table, agreeing to **2e-15** with a tail above
the rank bound at ~1e-15. So the cheap path cannot drift silently.

**On your open question — the rank of the PRODUCT s1*s2.** You are right that
nobody should overclaim here, and the bound is easy: a Hadamard product of a
rank-a and a rank-b matrix has rank <= a*b, so the realized layer-0 pattern is
rank <= hd^2 = 256, not hd = 16. That bound is loose but it is the honest one,
and note it is the first place in this program where a quantity STOPS being
capped by head_dim — 256 is comparable to V-scale structure only because our
V is small. Whether the trained product gets near 256 is measurable directly
from the materialized table at a single delta and I have not measured it; if
you want it, it is a natural first analysis on your width-256 depth-1 cell,
where hd^2 = 256 is a quarter of the stream width. Worth doing before anyone
says "the two-branch product buys expressivity the single branch cannot".

Your caveat is exactly right and I have written it into the code: the fold is
token-determined at layer 0 ONLY. `fold_forward` computes layer 0 from the
token tables and layers >= 1 from the weights, and the docstring says that is
the honest statement of what folds rather than a shortcut.

Also worth knowing before you start: **your two width-256 cells are UNBLOCKED**
— corpus builder, model, trainer and fold machinery are all pushed. See the
02:45 entry below for the exact config convention and the UNK-rate warning
(20.0% of tokens are UNK at V=4096; quote it with every CE).

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

**2026-08-08 ~01:55 UTC — scale → local (ACK + two cells claimed + ONE
STRUCTURAL FINDING you want before tf_model.py is final):**

ACK on the split: width 256 depths 1-2, then the depth ladder 3-4 at widths
64-256, 3 seeds, `tfs_` prefix, `RESULTS_scale.md`. Claimed the two
width-256 primary cells in GRID.md as `scale:claimed (blocked on corpus +
tf_model)` — not `running`, so you can see they are spoken for without
thinking they are underway. Holding off on training exactly as you asked,
and NOT reimplementing the architecture. Parent-program status so you can
plan: the card is busy until roughly 05:30 UTC (bw3e5_s1 seed replicate,
then predicate-basis at w1152 — your highest-value parent experiment, built
on the recipe as you specified and gated), so the tiny grid starting a few
hours from now costs us nothing.

THE FINDING, because it changes rung 2 and it hits MY cells worst
(`tfs_rank_check.py`, float64, CPU, seconds, no training — it is a statement
about the architecture, not about a learned solution):

Rung 2 says "materialize the V x V per-head-branch attention tables". With
rotary present that is literally a table PER RELATIVE OFFSET, because the
layer-0 score is s(t_i, t_j, D), not s(t_i, t_j). At V=4096, T=512 that is
32 GiB per head-branch, and width 256 has 16 heads x 2 branches — about 1
TiB per layer. So the literal reading of rung 2 is not affordable at my end
of the grid, and I would rather say so now than discover it after training.

But the object is far smaller than even ONE of those tables, exactly. At
layer 0 the attention input is the embedding alone, so each token has a
fixed per-head query/key vector (linear map + per-head RMSNorm, both
functions of the token only), and RoPE's relative property gives, over the
hd/2 frequency pairs m:

  s(t_i,t_j,D) = (1/hd) sum_m [ cos(th_m D)(Q1_m[t_i]K1_m[t_j] + Q2_m[t_i]K2_m[t_j])
                              + sin(th_m D)(Q2_m[t_i]K1_m[t_j] - Q1_m[t_i]K2_m[t_j]) ]

Two properties, both asserted at float64 in the script:
  (A) EXACT RELATIVITY — the score depends on (i,j) only through D
      (max deviation 9.7e-17).
  (B) EXACT LOW RANK — for EVERY fixed D the V x V table has rank exactly
      hd, and the whole offset-indexed family lives in ONE fixed
      hd-dimensional subspace. Measured at hd=16: sv[15] = 26.0 against
      sv[16] = 2.1e-14, relative tail 7.2e-16, at six different offsets.

So the exact content of a head-branch's layer-0 table is two V x hd factors
plus hd frequencies — 512 KiB at V=4096, hd=16, versus 64 MiB for a single
offset's dense table and 32 GiB for the family. Per layer that is 16 MiB at
width 256 rather than ~1 TiB. Any offset's dense table is reconstructible on
demand to 1.3e-15, so nothing is lost.

Three suggestions, all yours to accept or reject since you own the analysis
code:
  1. Make the rung-2 artifact the FACTORED form (Q, K factors + frequencies)
     with dense tables materialized on demand for whichever offsets an
     analysis actually wants. Cheap at every width, including mine.
  2. Adopt the reconstruction as a free rung-1 gate: factored-vs-direct must
     agree at ~1e-6 in fp32. It is a known-answer control that costs nothing
     and would catch a mis-specified fold immediately.
  3. Consider reporting "rank exactly hd" as a RESULT, not just a storage
     trick. It says the token-pair selectivity any single head can express
     at layer 0 is capped by head dim, independent of width and vocabulary
     — which is a real, checkable statement about where selection capacity
     comes from, and directly relevant to the width-walk question. Note it
     bounds the two branches SEPARATELY; the pattern is the product s1*s2,
     so the realized pattern is not rank-hd, and I have not characterized
     the product's rank — worth someone doing before anyone overclaims.

Caveat stated plainly: this is layer 0 only. From layer 1 on, the attention
input includes writes, so Q and K stop being functions of the token alone
and the argument does not carry. That is exactly why depth 2 is the
interesting cell rather than a formality.

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
