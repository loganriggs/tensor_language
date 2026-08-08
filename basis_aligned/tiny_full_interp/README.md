# Tiny full-interpretation program

**Goal: train bilinear transformers small enough to interpret COMPLETELY —
not "we found some circuits", but a full accounting of what every parameter
does — and then walk width and depth up to watch how the solution changes.**

Started 2026-08-08 (Logan's pivot). Sibling program to `../qk_mdl/`, which
studies interpretability-enhancing architecture at scale; this one studies
total interpretability at small scale. They share the fold machinery and the
fresh-data protocol; they do NOT share checkpoints or claims.

## Why "fully interpret" is achievable here and nowhere else

Three properties compose:

1. **No softmax.** Attention is `(q1·k1)(q2·k2)/d²` with a causal mask — a
   polynomial. Layer-0 attention therefore folds EXACTLY into a token-pair
   score table: for every ordered pair of vocabulary items, a number,
   computed from weights alone with no data assumptions.
2. **Bilinear MLPs.** `MLP(x) = D·(Lx ⊙ Rx)` is exactly a symmetric
   third-order tensor `T[o,i,j]`, and RMSNorm folds out as a scalar gauge
   (`MLP(rms(x)) = D·T(x,x)/‖x‖²`, verified to ~1e-7 in the parent program).
   No gating, no ReLU, no approximation.
3. **Small vocabulary.** With V ≈ 4096–8192 the exact V×V tables are
   68–268 MB in fp32 — materializable, printable, diffable. This is the
   enabler that makes "exhaustive" affordable; at V = 50257 the same object
   is 10 GB per head-branch and only samplable. The vocabulary is a
   **byte-level BPE trained on this corpus** (zero UNK by construction), not
   a truncation of GPT-2's — see the vocabulary section below.

Consequence: a 1-layer model is a closed-form polynomial in one-hot inputs,
and a 2-layer model is that polynomial composed once. There is no part of
these models we are forced to describe statistically.

## The scientific question (not just "can we")

The parent program measured, on an 18-layer model, that **selection is
nameable but content is a spectral, non-class-nameable dictionary at every
layer**. That was measured at one size. The question here:

> At what width/depth does full interpretability break, and *what* breaks
> first — does content become spectral at width 32, or only at 256? Does the
> model discover induction at depth 2 regardless of width, or only above a
> width threshold? Do independently-seeded models at the same size converge
> to the same solution, or to different ones?

That makes this a study of **how the solution changes as capacity grows**,
with a fully-known object at every point on the curve.

## The grid

| axis | values | why |
|---|---|---|
| depth | **1, 2** (focus), then 3, 4 | 1 = closed form; 2 = composition appears; 3–4 = where the parent program's "grounding" begins |
| width | **32, 64, 128, 256** (heads scale with width, head dim 16 fixed) | log spacing; 8× range brackets the transition |
| seeds | 3 per cell | the parent program learned the hard way that single-seed structure claims do not survive (see `../qk_mdl/BRAINSTORM_STATE.md`, FOUNDATIONS CORRECTION) |
| vocab | **8192 (primary), 4096 (companion)**, both trained byte-level BPE | exact-table tractability; zero UNK by construction |
| tokenizer | **bpe (primary)**, trunc (labelled comparison arm) | measures how much the tokenizer distorts the interpretability picture |

Primary cells = {1,2} × {32,64,128,256} × 3 seeds = 24 models. All are minutes
to ~1 hour each; the whole primary grid is a day of one GPU, not a week.

## Protocol (inherited, non-negotiable)

- **Fresh single-epoch data**, never a second pass over any sequence — the
  parent program found multi-epoch training silently subsidizes structure
  claims by memorization. Small-vocabulary corpus built from the same FineWeb
  text with a byte-level BPE trained on that text (UNK impossible).
- **BITS PER BYTE for every cross-tokenizer number.** Per-token
  cross-entropy is NOT comparable across tokenizers: a tokenizer with fewer
  bytes/token makes each prediction easier *and* makes more predictions, so
  nats/token can move either way for reasons that have nothing to do with
  modelling quality. Any claim that spans two tokenizers (or two vocabulary
  sizes, which is the same thing) must be stated as
  `bits/byte = CE_nats_per_token / (ln 2 × bytes_per_token)`
  on the same held text. This is applied mechanically:
  `tf_corpus.bits_per_byte()` is called inside `tf_train.baselines` and
  `run_cell`, so every baselines JSON and every cell JSON carries
  `*_bits_per_byte` next to its nats. Nats/token remain the right unit
  *within* one tokenizer (paired per-token deltas between cells).
  A lossy vocabulary (the truncated arm) additionally owes
  `unk_repair_bits_per_byte` before its bits/byte is a code length at all
  — see below.
- **Positive controls before every claim.** Identity reductions must pass at
  exactly zero; every headline in the parent program that turned out wrong
  was caught by a known-answer control and never by inspection.
- **Registered predictions** written into the results JSON *before* training
  or analysis runs.
- **Matched-optimizer baselines.** Quote costs against a baseline trained
  with the same optimizer (the parent program understated every tax by 0.094
  nats for a week by comparing Muon arms to an AdamW baseline).

## The interpretation ladder (what "fully interpreted" means, concretely)

A model is **complete** on this ladder when every rung has a machine-checked
artifact, not a narrative:

1. **Exact fold** — every layer written as its tensor; gate ~1e-6.
2. **Materialized tables** — the V×V per-head-branch attention score tables
   and the MLP tensor, on disk, with their spectra.
3. **Behavioral inventory** — what the model actually computes, measured:
   bigram statistics, skip-grams, induction, positional priors, each with a
   causal ablation and a floor.
4. **Content accounting** — for every write direction, either a name that
   passes a substitution gate, or an explicit statement that it is spectral,
   with the measurement that shows it.
5. **Reconstruction** — an explicit program (code + tables, no weights)
   that reproduces the model's next-token distribution to a stated KL, and
   the honest remainder.
6. **Convergence** — the same artifacts across seeds, with a similarity
   measure, so "the model learns X" is a claim about the size, not the run.

Rungs 1–3 are expected to be routine; rung 5's remainder is the real
deliverable, and rung 6 is what makes the width/depth curve meaningful.

## Division of labour

Two machines, coordinated exactly as in the parent program: git + an
append-only mailbox, verdicts pushed as they land.

**Local box (RTX 5070 Ti, 16 GB)** — the small end and the analysis:
- trains widths **32, 64, 128** at depths 1–2, all 3 seeds (18 models, each
  minutes; the whole set is a few hours)
- owns the **tokenizer and corpus build** (byte-level BPE over the same
  FineWeb text) and pushes the tokenizer JSON + manifest so both boxes
  regenerate byte-identical data
- owns **all rung 1–6 analysis machinery** (folds, tables, behavioral
  battery, reconstruction), because the analysis is small and the code must
  be shared
- files: `tf_*` prefix

**Scale box (RTX 5090, 31 GB)** — the large end and the depth ladder:
- trains width **256** at depths 1–2, and the depth ladder **3, 4** at
  widths 64–256, all 3 seeds
- runs the same analysis code on its own checkpoints (code comes from local;
  do not fork it)
- files: `tfs_*` prefix
- its own results doc `RESULTS_scale.md`; local owns `RESULTS.md`

Neither box duplicates a cell. If a box is idle, it takes the next unclaimed
cell from `GRID.md` (claim by pushing a one-line edit before starting) rather
than inventing work.

## Files

- `README.md` — this file
- `GRID.md` — the cell table with claim/status per cell
- `MAILBOX.md` — cross-box channel (newest first, append-only)
- `tf_tokenizer.py` — trains the byte-level BPE, builds the PRIMARY corpus,
  owns the cross-tokenizer bits/byte comparison and its controls
- `tf_corpus.py` — the truncated-GPT-2 comparison corpus, plus the shared
  loaders for both corpora (`load_split(..., tok=)`) and `bits_per_byte()`
- `tf_model.py` — the tiny bilinear transformer (shared by both boxes)
- `tf_train.py` — training with the fresh protocol
- `tf_fold.py` — rung 1–2: exact folds and materialized tables
- `tf_behavior.py` — rung 3: behavioral inventory with causal ablations
- `RESULTS.md` — local results

## Vocabulary decision (2026-08-08, REVISED same day) — trained BPE, not truncation

**Superseded approach.** The first build reduced the vocabulary by *truncating*
GPT-2's 50,257 ids to the top-K and mapping the rest to `<UNK>`: 20.0% UNK at
V=4096, 13.2% at V=8192. That is a crude hack and Logan caught it. Truncation
is strictly the worst of both worlds — it throws away a fifth of the tokens
*and buys no compression at all*, because the segmentation is still GPT-2's,
so a 512-token sequence covers exactly the same text; you simply cannot read
13–20% of it. UNK becomes the most frequent symbol, dominating every table's
rows and columns, and the cross-entropy stops being a code length for the text.

**What we do instead.** `tf_tokenizer.py` trains a **byte-level BPE on our own
FineWeb text** (recovered by decoding the parent program's GPT-2 shards, which
is exact — the round-trip control C1 shows ids → text → ids is bit-identical).
Two properties follow:

- **Zero UNK by construction.** The initial alphabet is all 256 byte symbols
  (`initial_alphabet=ByteLevel.alphabet()`), so every possible input has a
  segmentation. This is a structural guarantee, not a measured rate.
- **Better compression at the same V**, because every merge is chosen for this
  data. At V=8192 the trained BPE reaches 3.755 bytes/token — 84% of GPT-2's
  4.447 with 1/6 the vocabulary.

Small vocabularies are also not a compromise at our scale. *Scaling Laws with
Vocabulary: Larger Models Deserve Larger Vocabularies* (arXiv 2407.13623,
NeurIPS 2024) finds the compute-optimal vocabulary grows with the
**non-vocabulary** parameter count, and our bodies are 25k–1.6M parameters —
so a few thousand types is near-optimal here rather than a concession.

**The measurement** (`tf_tokenizer_compare.json`; all bits/byte on the same
13.69 MB of held text; n-grams fitted on the same 40,000-row train text sample):

| tokenizer | bytes/token | UNK | tokens for the held text | bytes a 512-token sequence sees | unigram bits/byte | bigram bits/byte |
|---|---|---|---|---|---|---|
| GPT-2 50257 | 4.447 | 0% | 3,078,001 | 2277 | 2.493 | **2.001** |
| truncGPT2-4096 | 4.447 | 20.3% | 3,078,001 | 2277 | 1.817 *(2.493)* | 1.462 *(2.138)* |
| truncGPT2-8192 | 4.447 | 13.2% | 3,078,001 | 2277 | 2.051 *(2.493)* | 1.645 *(2.086)* |
| newBPE-2048 | 2.899 | **0%** | 4,721,412 | 1484 | 3.275 | 2.309 |
| newBPE-4096 | 3.334 | **0%** | 4,105,378 | 1707 | 3.019 | 2.172 |
| **newBPE-8192** | **3.755** | **0%** | 3,645,823 | 1922 | 2.805 | **2.082** |

Numbers in *(parentheses)* are the **honest** cost. A truncated vocabulary is a
**lossy code**: UNK is unrecoverable, so its raw bits/byte is not a code length
for the text and its apparent advantage is entirely fake. The honest figure adds
`unk_repair_bits_per_byte` — the cost of naming which discarded GPT-2 id each
UNK was, under the train-split unigram restricted to the discarded set (0.676
bits/byte at V=4096, 0.442 at V=8192). **Known-answer control:** by the chain
rule the two-part unigram code must reproduce the full-vocabulary unigram code
exactly, and it does — both truncated rows land on 2.4926 bits/byte, GPT-2's
value, to 1e-5.

Fitted on the *full* train split instead of the 40k sample (the numbers the
program actually quotes, `tf_baselines_*.json`), the bigram ordering is:

| corpus | bigram bits/byte |
|---|---|
| **BPE-8192** | **2.030** |
| truncGPT2-8192 (honest) | 2.050 |
| truncGPT2-4096 (honest) | 2.115 |
| BPE-4096 | 2.137 |

So at V=8192 the trained BPE wins outright *and* is lossless. At V=4096 the
truncated code's honest number is marginally lower, but only because the repair
term is priced with an oracle unigram over the discarded set that no trained
model has — the truncated model still cannot emit those tokens at all.

**Decision: V=8192 trained byte-level BPE is the primary corpus**
(`tf_corpus_b8192/`), with **V=4096 BPE** as the companion vocabulary-size point
(`tf_corpus_b4096/`), and the **truncated GPT-2 V=8192** corpus retained as a
labelled *tokenizer-distortion comparison arm* (`tf_corpus_v8192/`), never as a
default. The tokenizer is in the checkpoint stem (`_b8192_` vs `_v8192_`) so the
two arms can never be silently mixed. V=2048 was measured as a low-end probe and
is not built as a corpus: at 2.899 bytes/token a 512-token sequence sees only
1484 bytes, which starts to change what the task *is*.

### Why the constraint is NOT table size

A structural finding from the
same build says table size does not stop us. The layer-0 attention table is
**exactly rank <= head_dim (16) per branch**: the score for (query token i,
key token j) is `(q1(e_i)·k1(e_j))·(q2(e_i)·k2(e_j))`, so each branch factor
is a V x 16 matrix product, and the realized pattern is their Hadamard
product (rank <= 16^2 = 256). **The exact artifact is therefore four V x 16
factor matrices per head, not a V x V grid** — 3 MB per head even at the
full 50k vocabulary. V x V materialization is a convenience for printing and
diffing, not a requirement of exactness, and it can be done in chunks for
whatever token subset an analysis cares about.

So the binding constraint is not memory, it is **parameter balance**. At
these widths the embedding dwarfs the body:

| | width 32 | width 128 | width 256 |
|---|---|---|---|
| V=4096 | 84% embedding | 57% | 40% |
| V=8192 | 91% | 73% | 57% |
| V=50257 | 98.5% | 94% | 89% |

A model that is 94% embedding is not a model whose *computation* we are
interpreting — it is a lookup table with a small transformer attached, and
the interesting structure would be swamped. The parameter-balance argument is **unchanged by the tokenizer
switch** — it depends only on V, not on how the V types were chosen — and it is
what fixes V=8192 as primary (embedding share 57–73% at the widths that matter)
with V=4096 as the companion and V=16384 as a check at width 256, where the
balance is best. What the switch changes is that those V types are now *earned*
merges over this text rather than the head of GPT-2's frequency list, so the
V=8192 model can represent every input instead of 87% of one.

This also reframes rung 2 of the ladder: the deliverable is the **factor
matrices with their spectra**, and V x V grids are rendered on demand for
the token subsets an analysis names.

## Precision: foldABLE architecture, folded ANALYSIS (2026-08-08, Logan's catch)

We train the ordinary parameterization (query/key projections, the three
bilinear MLP matrices, embeddings) and fold a trained checkpoint afterwards
for analysis. The fold is exact — the folded object reproduces the model's
forward pass to ~3e-7, gated — but it is an analysis artifact, not the
training object.

We deliberately do NOT train the folded form, and the reason is substantive:
the fold maps into a LARGER model class, not onto the same one. A bilinear
MLP at width 128 with hidden 512 is ~196k parameters; its folded tensor is
128^3 ~ 2.1M and rank-unconstrained, so optimizing the tensor directly would
leave the low-rank subset the factored form occupies and produce a strictly
more expressive model whose folds correspond to nothing we trained. Same on
the attention side: the factored form is what GUARANTEES the score table is
rank <= head_dim per branch, which is exactly the property that makes the
exact artifact small (four V x 16 factor matrices per head). Training the
table directly would discard that guarantee.

So: foldable by construction, folded for analysis, exactness verified by a
gate rather than assumed.

## Adversarial review is a required stage, not an optional pass (Logan 2026-08-08)

Every interpretation of every cell goes through explicit reviewer rounds
before it is believed, and the rounds critique BOTH the claim and the
technique that produced it. The parent program's record is the argument:
every wrong headline there was caught by a control or a reviewer, never by
inspection, and several survived weeks because nobody attacked them.

Procedure per cell:
1. **Interpret** — work the ladder, register predictions first.
2. **Self-red-team** — the analyst lists, for each claim, the strongest
   objection to the claim AND to the method, then fixes what it can and
   marks what it cannot. Written into the results JSON as
   `reviewer_round_1`.
3. **Independent review** — a reviewer who did NOT produce the
   interpretation attacks it with the checkpoint and code in hand, and is
   explicitly instructed to look for the standing failure modes below.
   Written as `reviewer_round_2`.
4. **Fix round** — the analyst answers each surviving objection with a
   measurement, a retraction, or a documented limitation. Nothing is
   allowed to stay in the "we will check later" state.

Standing failure modes to attack every time (from the parent program's
actual mistakes):
- **Arithmetic dressed as a finding.** The layer-0 table is rank <=
  head_dim BY CONSTRUCTION. Only rank far below that bound is a result.
- **Fitting and evaluating on the same tokens.** Any table, dictionary or
  reconstruction must be fit on an estimation split and scored on a
  held-out one.
- **Beating nothing.** A reconstruction must beat a same-parameter-count
  alternative, not just beat chance.
- **Precision mistaken for correctness** (and its converse). Compare in
  float64 before calling a gate failure a bug; compare symmetrically
  (the parent program lost a day to a one-sided tf32 setting).
- **Sign without composition (the error most often made here).** In a model
  with no positivity constraint anywhere, the sign of an intermediate
  quantity is a GAUGE FREEDOM of the factorization, not a property of the
  computation: flip a sign in one factor and flip it back in another and
  the function is unchanged. Only the sign of a COMPLETE PATH to an
  observable is invariant.
  Concretely for this architecture: a negative attention weight is not
  suppression. The contribution of attending position j is
  `pattern[i,j] x (value->output->unembedding of x_j)`, so
  negative x negative = a POSITIVE push on the attended content, and the
  negative effect lands on the NON-attended positions instead. Worse, the
  pattern is itself a product of two branches `(q1.k1)(q2.k2)`, so even
  "the attention sign" is a product of two individually meaningless signs.
  RULE: never report a sign, a "suppression", an "inhibition" or an
  "anti-correlation" from any factor in isolation. Compose to the logits
  and confirm causally. In the parent program the raw coefficient sign was
  ANTI-correlated with the behaviour it appeared to describe (Pearson
  -0.45) while the composed score tracked it at Spearman +0.85 and matched
  the causal direction 5 of 5.
- **Ablating a term without composing it through what consumes it (added
  2026-08-08, after it cost us a published claim in this program).** The
  first depth-1 ladder added the past-attention write to the residual while
  holding the MLP frozen at its no-context input, measured ~0.0005 nats, and
  reported "attention to the past does nothing". Attention is in fact worth
  0.29-0.91 nats of KL depending on width, and *all* of it arrives by moving
  the MLP's argument. This is the sign rule's non-sign twin: a term's value
  is the value of its COMPLETE PATH, so an ablation has to let the
  downstream nonlinearity see the change. Every knockout must be stated as
  "route X removed, everything downstream recomputed", and mutually
  exclusive routes must be shown to BRACKET the full model.
- **A null result with an uncalibrated detector.** "We measured no
  induction" is worth nothing until the battery is shown to detect a planted
  induction of known size. Plant it (mix in an oracle at weight eps, sweep
  eps) and quote the null as an upper bound in nats. Our registered
  positive control -- "a depth-2 cell must show induction" -- FAILED, and the
  planted-oracle calibration is what kept the null result meaningful.
- **Single seed.** Structure claims need three; the parent program had a
  readability ordering reverse between two seeds.
- **Uncalibrated nulls.** Shuffle nulls with near-zero spread produce
  z-scores in the thousands that are not effect sizes.
