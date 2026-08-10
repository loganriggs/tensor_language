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

## Phase V4 — THE SIX ARCHITECTURES AT DEPTH 3 (CLAIMED BY LOCAL 2026-08-08 17:30, RUNNING)

The interaction the slice and the ladder jointly set up and neither ran. At
depth 2 the five interpretable architectures used a residual route the plain
model left empty AND inducted at width 128 where the plain model needed 256.
At depth 3 the plain model does both by itself. **So are the architectures an
ACCELERANT for what depth supplies (a), do they still ADD something (b), or do
they INTERFERE (c)?** Predictions PD1–PD7 registered before the first training
step in `tf_d3_variant_predictions.json`; chain `tf_d3_variant_chain.sh` (log
`tf_d3_variant_chain.log`); verdict produced mechanically by
`tf_d3_variant_report.py` → `tf_d3_variant_slice.json` / `tf_d3_variant_table.md`.

| cell | owner | status |
|---|---|---|
| vanilla d3 w128 s0/1/2 | LOCAL | **done** (from the depth ladder, not retrained) — CE 4.5276 ± 0.0007, induction +0.1085 ± 0.0133 |
| slots / bandwidth / predicate / codebook / shrink, d3 w128, s0/1/2 | LOCAL | **done** — verdict (a) ACCELERANT, `tf_d3_variant_table.md` |
| `slots` d2 w128 `_g8`, s0/1/2 — slot-geometry control A | LOCAL | **done** — CE 4.8904 ± 0.0075, induction +0.0200 ± 0.0191 (0.21× the 4×32 answer) |
| `shrink` d2 w128 `_g8`, s0/1/2 — control A, second masked arm | LOCAL | **done** — CE 4.8354 ± 0.0060, induction +0.0291 ± 0.0025 (0.34×) |
| vanilla / slots / shrink d3 **w192** s0/1/2 — the exact 6×32 geometry (control B) | LOCAL | **done** — plain +0.1911 ± 0.0175, slots +0.3206 ± 0.0424 (1.68×), shrink +0.3557 ± 0.0966 (1.86×) |
| `slots` d3 w192 `_g8` s0/1/2 — 8×24 vs 6×32 at FIXED width (control B2) | LOCAL | **done** — +0.2050 ± 0.0149 vs +0.3206, Welch t −4.45, CE +0.062 nats: **slot geometry is load-bearing at matched size** |
| vanilla d3 **w144** s0/1/2 — parameter-matched plain control (2,299,824 params, more than any variant) | LOCAL | **done** — CE 4.4703 ± 0.0056, induction +0.1448 ± 0.0462. At matched parameters only `predicate` still beats the plain model |

**PHASE V4 IS CLOSED (2026-08-08 23:20).** Verdict (a) ACCELERANT, and it
survives the round-5 independent review (`tf_reviewer_round_5.json`, eight
objections) — stable against the bar (1.5×–5.0×), the seeds, a second probe,
the slot geometry and parameter matching. Three published sentences were
corrected and one sub-claim withdrawn; see RESULTS FINDING 17.

**SLOT-GEOMETRY DEVIATION, forced by arithmetic and documented rather than
hidden.** The masked-decoder variants (`slots`, `shrink`) need one slot per
module, i.e. n_slots = 2·depth = 6 at depth 3, and the stream must partition
evenly — but 128 is not divisible by 6. The only n_slots that both divides 128
and leaves every module a nonempty write mask is **8**, so `slots` and
`shrink` run 8 slots of 16 instead of depth 2's 4 of 32 (two slots written by
nothing). The small-decoder variants are unaffected: they scatter into 6
solved slots, stream 168. The two controls in the table price the deviation —
the same geometry change at the depth-2 cell whose answer is already
published, and depth 3 at width 192 where 6 × 32 is exact.

**Round-4 review compliance.** Per `tf_reviewer_round_4.json`, this slice
quotes every route KL beside the write's **norm share of the read it enters**
(a read-ablation KL is quadratic in that share, r = 0.994), and decides
induction over **model seeds**, not against a probe-noise floor.

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
| 3 | 64 | LOCAL | **done, 3 seeds, interpreted + route-use** — induction +0.0035 ± 0.0041, model-seed t 1.47, **NULL** |
| 3 | 128 | LOCAL | **done, 3 seeds** — CE 4.5276 ± 0.0007, **induction +0.1085 ± 0.0133** (t 14.1) |
| 3 | 256 | LOCAL | **done, 3 seeds** — **induction +0.2207 ± 0.0605** (t 6.3) |
| 4 | 64 | LOCAL | **done, 3 seeds** — induction +0.0099 ± 0.0068, model-seed t 2.52, **NULL** (seed 0's +0.0173 was the highest of the three) |
| 4 | 128 | LOCAL | **done, 3 seeds** — **induction +0.1583 ± 0.0277** (t 9.9) |
| 4 | 256 | LOCAL | **done, 3 seeds** — **induction +0.2945 ± 0.0284** (t 18.0) |

> **THREE SEEDS ARE IN AND THE LADDER HAS BEEN INDEPENDENTLY REVIEWED — read
> RESULTS.md FINDING 16 before the paragraph below, which is the superseded
> seed-0 verdict and is kept only as the record.** Corrections: the threshold
> moves **ONCE** (256 at depth 2, 128 at depths 3 AND 4), and even that is
> criterion-dependent; the routing language is **retracted** and restated as a
> magnitude result (read-ablation KL is quadratic in the write's norm share,
> r = 0.994 over 243 pairs); the route-USE result survives at 86% ± 10%, not
> 94.5%. Machinery: `tf_route_seeds.py`, `tf_reviewer_round_4.py`,
> `tf_reviewer_round_4.json`.

**Seed-0 verdicts (RESULTS.md FINDING 14; SUPERSEDED — see the box above).**
The induction width threshold falls **one
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
| same-size softmax+GELU transformer | 64-256, depths 1-3 | local | **CLAIMED BY LOCAL 2026-08-08 23:05, RUNNING** (`tf_baseline_std.py`, chain `tf_baseline_chain.sh`, gated behind `tf_geom_control_chain.sh`) |

The softmax/GELU baseline answers the first question a reviewer will ask:
what does the no-softmax bilinear architecture cost in prediction quality,
at each size? If the gap grows with size, the fold's tractability is being
bought with capability and that must be reported alongside every result.

## THE FOLDABILITY TAX — conventional-transformer baseline (CLAIMED BY LOCAL 2026-08-08 23:05)

Everything this programme has measured is relative to another FOLDABLE model.
This cell closes that. Predictions — Logan's AND the analyst's, which disagree
on two of three — are registered in `tf_baseline_predictions.json` BEFORE the
first training step.

Conventional = softmax attention (one query/key branch, `q.k/sqrt(16)`) +
`Down(GELU(Up(x)))`. Identical in every other respect: same corpus, same
tokenizer, same 15,000-step single-epoch data order, same Muon+AdamW at the
SAME learning rate 0.02 every foldable cell used, same head dimension 16, same
rotary, same per-head query/key RMSNorm, same tied embedding, same readout, same
held evaluation, same induction probe (called verbatim through a shim with a
positive control that it reproduces a published foldable number).

**Two parameter arms, because the family is the BIGGER model at nominal
expansion** (18·W²+W of body per block against 12·W²+W): the `x4` arm is the
conventional 4× expansion, which gives the conventional model ~12% FEWER total
parameters; the `x7` arm sets the expansion to 7×, which makes the body exactly
18·W²+W and the total parameter count bit-identical at every cell.

| depth | width | arm | owner | status |
|---|---|---|---|---|
| 1 | 64 / 128 / 256 | x4 nominal, 3 seeds | LOCAL | local:running |
| 2 | 64 / 128 / 256 | x4 nominal, 3 seeds | LOCAL | local:running |
| 3 | 64 / 128 / 256 | x4 nominal, 3 seeds | LOCAL | local:running |
| 1-3 | 64 / 128 / 256 | x7 matched-parameter, 3 seeds | LOCAL | local:running |
| 2 | 128 | no-QK-norm control, 3 seeds | LOCAL | local:running |
| 1-3 | 128 | lr 0.01 / 0.04 full-length bound, seed 0 | LOCAL | local:running |

Machinery: `tf_baseline_std.py` (model + duplicated-and-gated training loop +
controls), `tf_baseline_probe.py` (induction through the verbatim probe),
`tf_baseline_report.py` → `tf_baseline_std.json` / `tf_baseline_table.md`.

### Seed-0 result, all nine cells, both parameter arms (2026-08-09)

The conventional model crosses the induction floor **one octave of width
earlier at every depth ≥ 2**, and scores 2–6× higher wherever both induct. The
held cross-entropy tax at exactly matched parameters is 0.054–0.127 nats,
growing with width. Full table in RESULTS.md FINDING 18. Seeds 1 and 2, the
query/key-norm control and the learning-rate bound are still running.

## THE ATTENTION × FEED-FORWARD FACTORIAL — what the tax is MADE OF (CLAIMED BY LOCAL 2026-08-09, gates passed, runs gated behind the baseline chain)

The baseline above changes **two** things at once — softmax versus an
unnormalised two-branch product, and a GELU gate versus an ungated bilinear
product — so it cannot say which one buys the induction. This 3×2 factorial
changes them one at a time.

| | feed-forward `bilin` (ours, foldable) | feed-forward `gelu` (conventional) |
|---|---|---|
| attention `bilin` (ours, foldable) | = the published family vanilla model | **new arm** |
| attention `bilinnorm` (diagnostic only) | **new arm** | **new arm** |
| attention `softmax` (conventional) | **new arm** | = the published conventional baseline |

`bilinnorm` is the same two-branch product divided by its row L1 norm. It is
**NOT a proposed architecture** — its denominator depends on every visible key,
so it does not fold to a fixed token-pair table and can never be reported as a
foldable win. Its only job is to separate *competition between keys* from *the
exponential* as the active ingredient in softmax. L1 rather than row-sum
because the product is signed and a row sum can cross zero.

Every arm's hidden size is set to hold the body at the family's 18·W²+W; the
one arm that cannot match exactly (softmax frees 2·W², which does not divide by
3) lands 256 parameters low out of 590,080, recorded per arm.

| gate | result |
|---|---|
| G1 `(bilin, bilin)` reproduces `tf_model.TinyBilin` vanilla under state-dict transplant | **PASS, max abs diff 0.0** |
| G2 `(softmax, gelu)` reproduces `tf_baseline_std.StdTransformer` at the matched expansion | **PASS, max abs diff 0.0** |
| G3 closed-form body counts equal live counts, all six arms | **PASS** |
| G4 `bilinnorm` block 0 == `bilin` block 0 ÷ its row L1, and the two arms genuinely differ | **PASS, max abs diff 0.0** |
| probe corner: the factorial's foldable path returns a published foldable induction score | **PASS, 1.7e-7** |

G1 and G2 are what earn the right to read the hybrids: the code reproduces both
known corners bit-for-bit, so an off-diagonal arm is that same code with one
factor flipped.

Predictions registered in `tf_factorial_predictions.json` before the file was
written. Machinery: `tf_factorial.py`, `tf_factorial_probe.py` (calls the
baseline probe unchanged), chain `tf_factorial_chain.sh`.

| depth | width | arms | owner | status |
|---|---|---|---|---|
**STATUS 2026-08-09 13:25 — the two-factor plan above was SUPERSEDED and the
design is now THREE factors over FOUR cells.** The query/key cap turned out to
interact with the attention factor (FINDING 19), so holding it fixed would have
confounded the very thing the factorial measures; it is varied as a third
factor. And the whole decomposition turned out to depend on the cell
(FINDING 21), so it is run over a 2×2 of cells rather than one.

| cell | design | status |
|---|---|---|
| depth 2, width 128 | full 3×2×2, three seeds every corner | **DONE** — three-way 88.8%, softmax×cap pair 12.3% |
| depth 3, width 64 | full 3×2×2, three seeds every corner | **DONE** — three-way 48.4%, pair 43.5% |
| depth 3, width 128 | full 3×2×2, three seeds every corner | **DONE** for the 2×2×2; row-L1 diagnostic finishing — three-way −1.3%, pair 75.9% |
| depth 2, width 64 | full 3×2×2, three seeds every corner | queued (`tf_factorial5_chain.sh`, gated); closes the cell factorial |

Machinery: `tf_factorial.py` (+ gates), `tf_factorial_probe.py`,
`tf_factorial2_chain.sh` / `tf_factorial2_seeds.sh` (depth 2 width 128),
`tf_factorial3_reordered.sh` (depth 3 width 64), `tf_factorial4b_chain.sh`
(depth 3 width 128), `tf_factorial5_chain.sh` (depth 2 width 64). Predictions
registered per cell in `tf_factorial{2,3,4,5}_predictions.json`; independent
review in `tf_factorial_independent_review.json`.

**Headline as it now stands (RESULTS FINDING 21 and its 12:30 subsection):**
softmax and an uncapped logit range are **jointly necessary and individually
worthless at every cell** — softmax alone never exceeds 3.1% of the move, the
uncapped range alone never 0.8%, the pair is worth 12% / 44% / 76%. Whether a
GATE must also join them depends on model size: required at the smallest cell,
optional at the largest.

**Two claims from earlier today were corrected, and a memoryless reader should
know which:** the "+0.2522 nats foldability tax" is RETRACTED (it held our side
to vanilla while letting the conventional model pick its best configuration; a
foldable arm at 7% fewer parameters pays +0.088 and our unconstrained best
wins by 0.008), and "the row-L1 diagnostic is dead" is qualified to "under 10%
of softmax at the same corner" — it reaches +0.1836 at the largest cell.

**STATUS 2026-08-09 04:50 — the baseline chain is DONE (exited clean at 04:37,
all stages). The factorial chain was STOPPED two minutes in, before writing any
checkpoint, and must be REDESIGNED before relaunch: `FacConfig` defaults to
`qk_norm=True`, so every arm including the softmax arms would have run under the
per-head query/key cap that the control shows costs the conventional model 0.163
nats and 8.3x of its induction by throttling softmax concentration. A factorial
that measures "how much does softmax buy" while capping softmax would understate
its own headline factor invisibly. The foldable query/key-norm control
(`tf_qknorm_chain.sh`, 3 cells, ~15 min) has the card and decides the right
common setting for all six arms. DO NOT relaunch `tf_factorial_chain.sh` until
that lands and the arm configuration is chosen from data.**

**(SUPERSEDED) operational note — interpose the factorial before the
baseline chain's third seed.** `tf_factorial_chain.sh` is launched and sitting
in its gate loop (it re-checks every 120 s and starts within two minutes of the
card going quiet). `tf_baseline_chain.sh` will reach **stage 5, the third
seed** — 18 cells, roughly 90 minutes — before it exits, and a third
conventional seed is worth less than the factorial's attribution. Both chains
skip any cell whose `.pt` already exists, so the safe interposition is:

1. Watch `tf_baseline_chain.log` for `stage 5: ` (or for stage 4's
   learning-rate report having been pushed).
2. `pkill -f -x "/bin/bash ./tf_baseline_chain.sh"` and any live
   `python tf_baseline_std.py` — the in-flight cell is lost and simply retrains
   later, which is why this is only worth doing at a stage boundary.
3. The factorial chain's gate opens on its own within two minutes.
4. When the factorial chain exits, relaunch `./tf_baseline_chain.sh`; it
   re-runs its gate, skips all 40-odd completed cells on checkpoint existence,
   and picks up at the third seed.

Do **not** edit either chain script while it is running — bash reads a script
incrementally, so an in-place edit can corrupt the execution of a running
chain.

## Vocab check

| vocab | owner | status |
|---|---|---|
| bpe 16384 at width 256 depth 2 | scale | unclaimed (corpus not built; one `tf_tokenizer.py` call) |
| bpe 4096 vs bpe 8192 same cell (vocab-size check) | local | DONE 2026-08-10 (RESULTS.md 16:00): interaction share 91.4% at V=4096 vs [81.4, 92.2] at V=8192 — central cell not a vocabulary artifact; all three registered predictions held |
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
