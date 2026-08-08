# Tiny fully-interpretable transformers: what we established

Standalone; no prior knowledge of the surrounding programme assumed. Every number
traces to a named file in this directory. Written 2026-08-08 with two training jobs in
flight; claims depending on them are marked **provisional**.

---

## 1. What this is, and why it is possible

Train language models small enough to account for *completely* — not "we found some
circuits" but what every parameter does — then walk width and depth up to watch the
solution change.

Three properties make that possible (`README.md`, `tf_model.py`). Attention has **no
softmax**: the score `(q1·k1)(q2·k2)/d²` is a polynomial, so the first layer folds into
a token-pair table. The feed-forward block is **bilinear**, `Down(Left(x) ⊙ Right(x))`,
exactly a symmetric third-order tensor, with RMSNorm folding out as a scalar gauge. The
**vocabulary is small**: 8192, a byte-level BPE trained on this corpus, zero unknown
tokens by construction, 3.755 bytes/token (`tf_tokenizer_compare.json`). Rotary
position embedding makes the first-layer score depend only on relative distance, and at
each distance the table is **exactly rank ≤ head dimension (16)** — singular value
15 = 26.0 against 16 = 2.1e-14 (`tfs_rank_check.py`) — so the exact artifact is two
`V × 16` factor matrices per head-branch, megabytes at any width. The fold is gated,
not assumed: fp64 fold-versus-forward residual **1.3e-14 to 4.4e-14**, with a negative
control confirming the gate rejects a corruption the older absolute gate passed
(`tf_identity_table.json`).

**The interpretation ladder**, six rungs, each demanding a machine-checked artifact:
(1) exact fold; (2) materialised tables with spectra; (3) behavioural inventory with
causal ablations and calibrated floors; (4) content accounting — every write direction
gets a name that survives a substitution gate, or an explicit statement that it is
spectral; (5) an explicit reconstruction of the model's predictions to a stated
divergence, with an honest remainder; (6) convergence across seeds. Rungs 1–4 and 6 are
complete; rung 5 became the hardest negative (§4).

**Scale, counted from disk** (checkpoints plus training JSONs carrying a final held
loss): **52 distinct trained configurations, 128 trained models** — 101 across 38
primary cells, 27 across 14 control arms. Two are a discarded buggy arm; six are a
tokenizer-comparison arm whose weights were never pushed. By depth 28/57/34/9 at depths
1/2/3/4; by width 7/13/80/3/7/18 at 32/64/128/144/192/256; by architecture 61 plain, 27
private-channel, 12 shrinking-channel, 10 bandwidth-limited, 10 named-term, 8
discrete-message.

---

## 2. What these models actually do

**A depth-1 model is a bigram machine that outgrows the bigram.** Held cross-entropy
against a dense closed-form bigram (5.200 nats, 67M entries, fitted on estimation,
scored on held) is 5.413 / 5.048 / 4.723 / 4.461 at widths 32 / 64 / 128 / 256
(`RESULTS.md` FINDING 3). **The crossover is about width 64.** Counter-readings are
recorded: at matched parameters the model's own bigram stage loses to a data-fitted
sparse bigram (5.490 vs 5.322 at 2.1M), and at position 0 the bigram wins at every
width.

**They are feed-forward machines whose attention acts through the feed-forward block's
input.** That write carries 1.0000 of the logit variance at every width; keeping only
it reproduces the model at divergence 1e-5, so the residual skip into the readout is
dead. The *direct* route to the readout alone lands exactly on the no-attention numbers
(0.258/0.431/0.644/0.851 against 0.285/0.466/0.687/0.911) while the route *through* the
feed-forward block alone lands on 0.0000 — the two bracket the model (FINDING 2).
Attention is largely a learned distance kernel: its distance-only average retains
16/44/61/68% of the effect as width grows.

**Induction: the threshold moves once, the magnitude grows monotonically**
(`tf_depth_ladder.json`, FINDING 16; three model seeds per cell).

| width | depth 2 | depth 3 | depth 4 |
|---|---|---|---|
| 64 | null | +0.004 | +0.010 |
| 128 | null | +0.109 | +0.158 |
| 256 | +0.094 | +0.221 | +0.294 |

Defined over **model** seeds the threshold moves **once** — 256 at depth 2, 128 at
depths 3 *and* 4 — after which depth buys magnitude, not octaves. It is
criterion-dependent (three defensible criteria give 256/128/64, 256/64/64, 256/128/128),
so the adopted headline is the surface, not a threshold. The probe is calibrated: a
planted induction oracle at weight 1e-4 moves the score 175 standard deviations, and a
depth-matched content-free control leaves 0 of 48 arms above floor (maximum +0.0010).

**Head-level localisation.** Depth 2, width 256, baseline 0.0870: dropping layer-0 head
1 gives 0.0115 (87% of the effect), layer-1 head 15 gives 0.0384, layer-0 head 0 gives
0.0436; unimportant heads move it −0.003 to −0.006. The wiring is not the textbook
circuit: deleting that head's write from layer 1's query/key/value read changes the
score by 0.0000, while deleting it from the first feed-forward block's input reproduces
everything (0.0841 → 0.0083). At depth 3 that changes — cutting the *second* attention
block out of the third layer's read removes **0.857 ± 0.103** of the induction score at
width 128; cutting the first removes 0.000 everywhere. The first attention block writes
almost nothing at any depth or width; every later one writes ~250× more.

---

## 3. The six architectures compared

**plain**, an ordinary shared residual stream; **private write channels**, the stream
partitioned into one separately-RMSNormed slot per module plus an in-loss group lasso;
**bandwidth-limited writes**, the same with genuinely small decoders and the savings
reinvested into wider slots; **named attention terms**, the same plus hand-written
predicate terms in the attention score, one learned scalar per head per term;
**discrete messages**, the same plus variable-k vector quantisation of slot content;
**shrinking embedding channel**, private channels plus a token remnant that shrinks
with depth to a floor. All six fold exactly, all five reductions to their parent
variant are bit-exact, and all were analysed through one code path (`tf_interp3.py`,
gated at 1.9e-6).

**Depth 2, width 128, three seeds** (`tf_consolidated_table.md`; held CE at context
512; the plain model's seed spread here is 0.0074 on CE and 0.0086 on induction). The
route column is the divergence when the first attention block's write is deleted from
layer 1's read only, [zeroed, resampled]:

| architecture | held CE | induction | route |
|---|---|---|---|
| plain | 4.6463 ± 0.0075 | −0.0034 ± 0.0099 (below floor 3/3) | [1.3e−5, 4.3e−6] |
| private channels | 4.7414 ± 0.0056 | +0.0972 ± 0.0275 (3/3) | [0.551, 0.124] |
| bandwidth-limited | 4.6279 ± 0.0037 | +0.1190 ± 0.0524 (3/3) | [0.538, 0.144] |
| **named terms** | **4.3861 ± 0.0020** | +2.6402 ± 0.0481 (3/3) | [0.353, 0.070] |
| discrete messages | 4.7542 ± 0.0054 | +0.0375 ± 0.0157 (2/3) | [0.119, 0.096] |
| shrinking channel | 4.7243 ± 0.0100 | +0.0860 ± 0.0303 (3/3) | [0.229, 0.143] |

Four of five acquire induction where the plain model has none at three seeds *and*
three learning rates (0.01/0.02/0.04), and all five transmit through a route the plain
model uses at ~1e−5. The shared nonzero decoder initialisation explains none of it
(induction −0.0095, route 3.5e−6); the partition plus per-slot norm is the mechanism,
and the lasso adds ~0.029.

**Named attention terms are the exception, and the capability is INSTALLED, not
LEARNED.** The term `1[token before key == query token]` *is* an induction head written
down. At depth 2, zeroing the 16 named scalars returns the model to −0.0028, exactly
the plain model's null. At depth 3 the test is stronger (`tf_reviewer_round_5.json`):
one scalar removes 98.0% at all three seeds; zeroing every named term leaves
−0.0028 ± 0.0115, below zero at two of three seeds and 0.111 *below* the plain model at
the same cell; held CE moves 4.3147 → 4.553, past the plain model's 4.5276. The loss
win and the induction win are one object.

**Depth 3, width 128, three seeds** (`tf_d3_variant_table.md`). The rule was registered
before the first training step (`tf_d3_variant_predictions.json`): >2× the plain
model's induction for ≥3 of 5 variants = the advantage persists; <0.5× for ≥3 =
inverts; otherwise **accelerant**.

| architecture | held CE | induction | ratio |
|---|---|---|---|
| plain | 4.5276 ± 0.0007 | +0.1085 ± 0.0133 | 1.00 |
| named terms | 4.3146 ± 0.0008 | +2.7578 ± 0.0954 | 25.4× |
| bandwidth-limited | 4.5307 ± 0.0134 | +0.2617 ± 0.0287 | 2.41× |
| discrete messages | 4.6496 ± 0.0188 | +0.1491 ± 0.0138 | 1.37× |
| shrinking channel *(8×16 forced)* | 4.7024 ± 0.0057 | +0.1150 ± 0.0215 | 1.06× |
| private channels *(8×16 forced)* | 4.7433 ± 0.0084 | +0.0822 ± 0.0181 | 0.76× |

**Verdict: ACCELERANT** — what the architectures bought at depth 2, depth supplies by
itself at depth 3. The independent fifth review finds the verdict word stable for any
bar in [1.5×, 5.0×], under every leave-one-seed-out subset and in 100% of 729
single-seed combinations at 2.0×, but it changes the count. Three of five are **not
separated** from the plain model over model seeds (Welch t = −2.03, 3.66, 0.44 against
4.30 needed). A **parameter-matched plain control** trained for the review (depth 3,
width 144, 2,299,824 parameters — more than any variant; CE 4.4703 ± 0.0056, induction
0.1448 ± 0.0462) drops bandwidth-limited writes to **1.81×** (t = 3.73, not separation,
0.060 nats *worse* on loss) and discrete messages to 1.03×, leaving only the
installed-term arm (19.05×, t = 42.7, 0.156 nats better). The two masked-decoder arms
ran a forced 8×16 slot geometry because 128 is not divisible by 2 × depth = 6; the same
change at the published depth-2 cell costs private channels 0.097 → 0.020 induction and
0.149 nats of CE, so their depth-3 numbers do not measure those architectures at their
intended geometry.

**Provisional:** the exact-geometry control (depth 3, width 192, where 6 × 32 divides)
is still training; its declared-final read-out (`tf_geom_controls.json`) has one seed
for private channels and none for the shrinking channel, so the 1.84× quoted there is a
single-seed number.

---

## 4. Compression — the hardest negative

Rung 5 became a frontier: description length in bits against divergence from the model,
tables fitted on estimation and scored on held text, every codebook, index, scale and
histogram charged (`tf_compress*.py`, FINDING 12).

**The honest denominator.** The headline point is 7.594 Mbit at divergence 0.0042:
5.66× below fp32, 4.74× below the best *lossless* recompression of the same weights,
2.83× below fp16, 2.17× below 12-bit uniform — and **1.15× below those same weights
naively quantised to the same divergence**. Only the last measures a discovery; over the
whole frontier it is **1.13–1.54×, median 1.20×**, and its best is bought by
distillation, not structure. The bill was recounted independently to the bit, and the
embedding half of the winning description was serialised with an arithmetic coder and
decoded from its own blob (5,169,672 real bits against 5,169,617 charged).

**No structural description beats bit-packing at any budget worth having.** Against the
lower convex hull of the recoding schemes, pure structure is 1.05–1.76× behind and
tensor-decomposition structure in the body 3.2–3.6× behind. Structure wins only below
~1.5 Mbit, where the description has already discarded a third to a half of what the
model knows.

**Why — a conversion law.** For an entropy-coded quantiser at fixed step, replacing a
source by a residual of variance ratio *v* saves exactly `−½ log₂ v` bits per weight at
the same distortion; with R² = 1 − v, R² 0.26 (spelling) buys 0.21 bits/weight, R² 0.41
(co-occurrence) buys **0.37**, R² 0.9375 buys 2.00 — halving a 4-bit code. **To matter
you need R² above about 0.94**, and nothing measured comes close. The coder is provably
competent (gross saving 0.52–0.63 bits/weight, between the variance-law and row-range
bounds) and the regression matrix itself costs 0.259 bits/weight, eating 44% of it.

**Compressibility versus size, and the sign is the wrong one.** One scalar per cell —
bits of the best description over the same weights naively quantised at matched held
cross-entropy — with an identical scheme family at 13 cells spanning depths 1–4 and
widths 32–256 (`tf_cgrid_summary.json`, FINDING 15). Slope **−0.042 ± 0.009 per e-fold
of parameters (t = −4.9)**, from 1.162 at depth 1 width 32 to **0.987 at depth 2 width
256**: at the largest cell the best description we can build is *worse in bits* than
bit-packing the same weights at the same held loss. Restricted to descriptions made out
of an interpretation it is **0.750–0.869 at 13 of 13 cells**, and — stronger, and
unregistered — **no structural scheme appears anywhere on the overall frontier at any
cell**. Five ways this could have been a size artifact were each measured; the trend
survives all, weakest at −0.029 ± 0.006. Scaling up will not rescue structural
compression here.

---

## 5. What is interpretable and what is not

**Selection is low-rank; content is spectral.** At depth 1 the distance-0 branch score
tables have entropy-effective rank 2.3/2.9/3.4/5.9 against an independent-Gaussian null
of 15.99 at the same rank bound, while the feed-forward tensor's unfolding sits at
30.0/61.1/121.9/239.7 against a random-factored null of ~31/62/123/247 (FINDING 4).
**The split holds in all six architectures** at depth 2 (`tf_consolidated_table.md`):
content sits at 0.978 of its same-shape random null in the plain model and 0.996–1.000
in all five architectures built to make content legible; selection at 0.199–0.347.

**That null has a measured blindness threshold.** Planting content confined to an
*r*-dimensional input subspace, the statistic reads 0.02–0.09 of the null at *r* = 2,
0.27–0.83 at *r* = 8, 0.80–0.96 at *r* = 16 and 0.95–0.99 at *r* = 32 —
indistinguishable from the models' own values. The supported claim is therefore
**"content is not confined to fewer than roughly 8–16 of the stream's 128–160 input
directions"**, not "content has no structure".

**Nameable structure does exist.** A sparse-dictionary description of the folded
first-layer rows gives atoms whose top-32 users have **surface-class purity 0.79 against
a random-token null of 0.49**, no dead atoms (`tf_dict_atoms.md`, FINDING 13): single
capital letters; capitalised name prefixes (` De ` Br ` Mc`); derivational suffixes
(`ably ally ful ive able`); contraction tails (`'ll 't 'd 've`); digits; spatial
prepositions (` toward ` along ` onto`).

**So: interpretable structure exists in these models and does not compress.** The
dictionary is genuinely structural and genuinely nameable, and simply not short. Zero of
214 descriptions in one sweep and zero of 208 in a disjoint sweep beat the model's held
cross-entropy while imitating it; quantisation beats the dictionary by ~5× in bits on
both seeds.

---

## 6. The methodological results

Standing failure modes, each with the measurement that established it, each having cost
a published claim (`README.md`).

| failure mode | the measurement |
|---|---|
| Composing along the wrong route | direct write→readout composition correlates 0.002–0.02 with a head's causal effect; through-feed-forward, 0.63–0.98 |
| Ablating without composing through the consumer | feed-forward frozen at its no-context input makes past attention look worth 0.0005 nats; unfrozen, 0.29–0.91 |
| Sign is a gauge freedom | flip a sign in one factor and back in another: same function. The pattern is a product of two branches, so its sign is a product of two meaningless signs; only a complete path to an observable is invariant |
| A component's value is ladder-position dependent | attention is worth 8.17 nats added first and 0.61 added last at depth 2, with a **3.8–7.6 nat interaction** larger than either last-position marginal; the interaction grows with width (3.9/7.1/10.7/14.3 at widths 32–256) |
| Zeroing is not the harshest ablation, and the order is not universal | resampling beat zeroing at 13 of 14 layer-cells in plain models; at depth 3, zeroing is harsher at 11 of 12 pairs for partitioned streams, by up to 9.2× |
| A probe-seed floor is not a detection threshold | it shrinks as 1/√n, so any nonzero score clears it eventually; three criteria gave three width thresholds on the same data |
| Retractions travel with the METHOD, not the cell | "the route opens" was retracted at depth 2, reasserted at depth 3 without rerunning the control, retracted again: over 243 write/read pairs, ablation divergence on write norm share gives r = 0.9944, slope 1.992, residual 0.264 dex — what *no* gating looks like |
| Marginal cells: seed spread equals the effect | at width 64 the spread (0.004–0.007) is the effect size; seven route-use fractions were struck for sd ≥ mean |
| Reporting from in-flight artifacts | two wrong reports in one day: a bound read as a detection, a compressibility level off by 0.5 |
| A law verified on one model class quoted over another | the quadratic magnitude law reproduces on plain checkpoints (slope 2.004, r 0.997) and fails on every variant (−0.36, −0.16, −0.18, −0.4, +0.61) |

---

## 7. The retraction ledger

Every one was caught by a control or an outside reviewer; none by inspection.

| withdrawn claim | what killed it | stood |
|---|---|---|
| Width 32 beats a full bigram by 0.064 nats | baseline refitted on estimation; it loses by 0.213 | ~10 min |
| Attention to the past buys 0.0005 nats; attention is inert | route decomposition through the feed-forward block: 0.29–0.91 nats | ~1 h |
| These are copy heads | composed to logits, the attended token ranks ~5600th of 8192 | same day |
| The composed pair table describes what the head does | correlation 0.002–0.02 with its causal effect | ~1.5 h |
| The induction circuit is the textbook two-layer one | deleting from layer 1's read: nothing; from the feed-forward input: everything | ~10 min |
| A layer-ordering flip at width 128 | resample ablation reverses it | pre-publication |
| Norm shares show the variants open a route | slot-norm imposed on the plain model moves it 0.00424 → 0.00434, not 1.27 | ~15 min |
| The variants open a route the plain model leaves shut | matched-displacement probe: the plain model is the *most* sensitive of the six | ~2.5 h |
| All five variants induct at width 128 | discrete messages fail their own floor at seed 2 (+0.0228 vs 0.0249) | ~2.5 h |
| Merging tokens is the worst code we measured | fp32 centroids, no residual; at 4 bits, 16.876 → 2.123 Mbit | ~1.3 h |
| 5.7× smaller than the model | matched-divergence naive quantisation: 1.15×, median 1.20× | ~1.3 h |
| The parent programme's exact-anchor-row winner does not port | it does; but stratified precision matches it at equal bits, so it is bit allocation | ~1.5 h |
| Nothing beats the model on the data | refitting to data gives 4.70937 at 7.455 Mbit against 4.71140 — then explained by a confound control (full precision gains 0.00727 nats) | ~20 min |
| Compressibility level 1.666, extrapolating to 1.40 at 550M | read from in-flight JSONs; finished, it runs 1.162 → 0.987 | ~15 min |
| Depth-4 width-64 induction: bound → detection → **null** | three revisions, each from adding model seeds; mean +0.0099, below floor | ~2 h |
| The threshold falls one octave per layer | three seeds: it moves once | ~2 h |
| The attention-to-attention route opens at depth 3 | the magnitude regression; the depth-2 error repeated one depth up | ~2.5 h |
| Route-use 94.5%, route share 0.386 | three seeds: 0.857 ± 0.103 and 0.299 ± 0.075 | ~2 h |
| Route-table entry 0.220 | transcription; layer 2's fraction is 0.086 | ~30 min |
| Two of five clear the 2× bar at depth 3 | one clears; the second's interval is [1.96, 2.86], and 1.47× on natural text | ~1 h |
| The named-terms loss win is −0.1435 | two cross-entropy measurements mixed unlabelled; it is **−0.2130** | ~1 h |
| Bandwidth-limited writes at 2.41× | a parameter-matched plain control puts it at 1.81×, unseparated | ~1 h |

---

## 8. Open questions and next steps

1. **The architecture's own cost has never been measured.** A same-size
   softmax-plus-GELU baseline is listed in `GRID.md` and remains unclaimed; everything
   here is conditional on the no-softmax bilinear family.
2. **Finish the exact-geometry control** (depth 3, width 192, three seeds per
   masked-decoder arm); until then the depth-3 verdict for those two architectures is
   geometry-confounded.
3. **Retrain the named-term architecture without its named terms.** The knockout bounds
   what *this* model's terms carry, not what the architecture would learn.
4. **Find a representation with R² above 0.94, or stop.** The conversion law rules out
   incremental structural coders by arithmetic; the untested direction with a real prior
   is putting description length into the *loss*.
5. **Push under the content-spectrality floor** at roughly 8–16 planted input
   directions — that is what separates "structureless" from "not low-rank".
6. **Make two-probe agreement standard.** The synthetic battery and the natural-text
   order-only swap agree at Pearson 0.9996 across the six architectures, yet their
   *ratios* differ by up to 3×.
