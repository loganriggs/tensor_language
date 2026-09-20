# 20 September — Folding from the embedding forward: what the first three blocks are made of

**Path now.** From the token embedding forward, every quantity in blocks 0–2 that a single token determines is an exact table, and every attention head's pattern is an exact function of (current token, previous token, offset). Read that way, the early attention is a **bank of positional filters** — 19 of 27 heads are separable "gain(cur) × kernel(offset) × content-weight(prev)" products, two of them are literally "subtract the running mean", eight are content matchers — and the early MLPs are near-pure bilinear maps of what they read, cheap to compress on every mode by context statistics but not by single-token statistics.

**What changed today.** The lane went from single-token folds (exact structure) to bigram folds (the first genuine cross-token tensor), to installed programs priced in CE, to two lessons about method: single-component edits nominate but do not certify (the heads back each other up), and Frobenius energy in any metric predicts nothing about CE until an edit is run.

Rows throughout: statistics and fits on the 480×513 skip80 FineWeb cache (+192 skip11000 for the last two fits); **every CE number is on the 192×512 skip7000 rows, fresh** (never used for any selection or fit), as CE added above the native forward (3.13241 on all 512 positions), lower is better. Tags: **fold** = exact algebra on tables/grids, **edit** = installed in the model, **response** = statistics of real activations or gradients, **fit** = optimised against the loss.

## The most instructive result: the token metric loses under CE (v615/v616)

Single-token folds said the "right" metric to compress MLP-0 was the token second moment — a 64³ Tucker core retains 23–33% of the tensor there against 0.33% in coefficient space (fold, v611; the 0.33% reproduces the 28 August HOSVD exactly). Installed as a projection, the token-metric frames cost the **same or more CE** than plain coefficient frames at every rank (edit, v615), and a frame built from the single-token *input* statistics cost **+5.6 nats** at rank 128 (edit, v616). Frames from the model's actual contextual activations cost 0.056 at the same rank. Single-token tables are exact for structure and wrong as statistics: in context 24% of MLP-0's input is the previous token (fold, v612) and the tables never see it.

## Claim table

| # | claim | tag | rows | numbers | status |
|---|---|---|---|---|---|
| 1 | MLP-0 is near-pure bilinear in the token; MLP-1's single-token write is still 39–62% quadratic despite an input that is 81% one constant direction | fold | whole vocabulary | quadratic share 0.89–0.91 (MLP-0), 0.39 uniform / 0.62 freq (MLP-1) | held (v610) |
| 2 | the previous-token channel into MLP-0 is separable: gain(cur) × content(prev) | fold | 1024² unigram grid | mode-t rank 1, mode-s 129, out 116; head 0.3 carries 0.79 | held (v612) |
| 3 | MLP-0's cur×prev interaction write is dense | fold | same grid | mode ranks 381 / 104 / 764; 32×128×128 core keeps 0.20 | held (v613) |
| 4 | previous token reaches MLP-1 mainly one-hop (attention-1), 30% via the MLP-0 relay; MLP-1 amplifies prev-dependence ×5.5 | fold | same grid | Var_s shares 0.55 / 0.30 / 0.06; 0.037 → 0.204 | held (v614) |
| 5 | context-PCA projections compress MLP-0 cheaply; single-token frames do not transfer | edit | fresh | in-256 0.019, out-256 0.011, joint 0.031 at 4.5× fewer values; single-token in-128 5.56 | held (v616/v617) |
| 6 | all three early MLPs cheap on every mode; ~512 product directions suffice | edit | fresh | hidden k=512: 0.018 / 0.024 / 0.016 | held (v618) |
| 7 | compression does not compose across layers; cascade-fitting makes it worse | edit | fresh | stack 0.406 vs parts 0.197; cascade 0.576 | falsified route (v619/v620) |
| 8 | reader sensitivity (Fisher) improves the output frame only ~22%; Fisher is diffuse | response+edit | fresh | 0.028 vs 0.036 at r=128; Fisher 90%-rank 851 | held as a small effect (v621) |
| 9 | layer-0 heads have positional kernels: 0.3 previous-token, 0.6/0.8 short, 0.4 medium, 0.7 long ~1/d; content heads taper only through rotary | fold | 4096² grid, d=1…512 | see appendix table | held (v622) |
| 10 | five layer-0 heads are gated positional filters κ(d)A(t)B(s) | edit | fresh | all five installed: 0.0094 (0.3 alone 0.0037 vs 0.112 ablation) | held (v623) |
| 11 | six layer-1 heads likewise; head 1.8 is "subtract the running mean of its values" | edit | fresh | six: 0.0126; 1.8 := −1/i: 0.0027 (token program 0.558) | held (v624–v627) |
| 12 | twelve heads of blocks 0–1 as a program compose across layers | edit | fresh | 0.036 vs parts 0.032 | held (v625/v627) |
| 13 | at layer 2 the tables get the kernel shape right and the scale wrong; 2.7 is a second running-mean subtractor | fold+edit | fresh | 2.6 token program 0.163 vs kernel 0.0015; 2.7 0.948 vs 0.0020 | held (v628) |
| 14 | singly, every early head is a fixed kernel to the loss; jointly they are not | edit | fresh | singles ≤ 0.009 (2.5: 0.025); banks 0.03–0.05; 27-head union 0.355 | held (v629) — method lesson |
| 15 | the joint cost is the positional heads' imprecision compounding, not the content heads' content | edit | fresh | 19 positional kernel-only 0.231; 8 content kernel-only 0.072 | held (v630) |
| 16 | the program form is right; the closed-form parameters were wrong | fit | fit skip80(+11000), fresh test | 0.136 → 0.050 with 9.8k fitted numbers (kernels + one gain per head); 1.6M refitted table entries no better | held (v631/v632) |
| 17 | price ladder of blocks 0–2 attention patterns | fit | fresh | fitted kernels only: 19 heads 0.087 (9.7k numbers), 27 heads 0.156 (13.9k) | measured (v633) |

## What this means

- **Exact structure comes from the tables; compression statistics come from contexts.** Single-token folds found the separable previous-token channel, the head kernels, the running-mean heads, and the degree structure of the MLPs — none of which context statistics would have named. But every frame or scale read off those tables that was then installed in context was wrong by a scalar or worse (claims 5, 13, 16).
- **Early attention is mostly positional.** The 19 separable heads are windows of three widths, previous-token taps, "previous minus mean" (2.4), and two running means (1.8, 2.7). Ten thousand fitted numbers reproduce their joint behaviour to 0.087 nats; the closed-form token gains buy a further 0.037; the eight content heads' joint content dependence is worth 0.069.
- **Programs compose when they replace patterns, not when they project activations.** Pattern programs add nearly linearly across layers (claim 12); MLP projections compound (claim 7). A pattern error is bounded by the attended values; a projection error is re-multiplied by every bilinear layer downstream.
- **Redundancy is the trap for single edits** (claim 14). Only joint replacement certifies that a structure is unneeded.

## Limitations that carry weight

- CE is on 192 held-out rows of one distribution (FineWeb); the kernels and tables were fitted on 480–672 rows. No OOD test.
- Programs replace **patterns only**; values, output projections, the diagonal (self) term, and the six-to-eight content heads stay native. The MLP compression points are per layer and do not stack.
- All grid statements (ranks, separability) are on a 1024² or 4096² unigram-sampled token grid; they are exact for those pairs and sample-based as statements about the vocabulary.
- The single-token bigram fold was verified against the real model at T=2 (replay 5.7e-7); deeper single-token tables (layers 1–2) were verified only through the layer-1 input (3.7e-7).


## Addendum (05:30 UTC) — the atlas past layer 2, and the induction circuit from the front

Fixed positional kernels stay cheap one head at a time through layer 4 (banks 0.043 / 0.035; edit, v634). Layer 5 is where content first becomes load-bearing: the bank costs 0.163 and head **5.7** alone 0.083 — it is the model's costliest head (+0.84 when its off-diagonal pattern is zeroed), the position-0 sink already in the ledger (§432/§451: top read at position 0 for 99.5% of queries, re-verified in v700) whose remaining three quarters of off-diagonal mass is spread over the context with ~1/i weights; **not** the induction head (response, v635/v636). An earlier line here called it "not a sink"; that was wrong (see the correction section below).

The induction head is **5.5** (response, 9.6× enrichment on keys whose predecessor is the current token; 1% of keys carry 4.3× its cost, edit, v636). Its first half is not a head: the "my predecessor was X" key feature is written redundantly by the nineteen positional heads of blocks 0–2 — the ten previous-token taps and the nine window heads are *each* sufficient (cut one family: 0.86× / 0.90× of the enrichment remains; cut both: 0.23×; response, v637/v639) — and amplified in layers 3–4 (0.37× when those are cut, v638). Its write raises the logit of the successor of the earlier copy at 87% of eligible positions, 2.7× the effect on random tokens (direct path, approximate), and cutting the head costs 4.2× more per eligible position than elsewhere (edit, v640).

| claim | tag | rows | numbers | status |
|---|---|---|---|---|
| 18 | fixed kernels cheap singly through layer 4; layer 5 breaks it at 5.7 | edit | fresh | banks 0.043 / 0.035 / 0.163; 5.7 alone 0.083 | held (v634) |
| 19 | 5.7 is not induction; it IS the position-0 sink (§432) with 75% of its mass spread over the context | response | opened (64 rows) | induction 1.1× base; argmax at position 0 for 99.5% of queries (v700); 26% of |mass| there; zeroing it +0.84 | corrected (v635/v636/v700) |
| 20 | 5.5 is the induction head | response+edit | opened / fresh | 9.6× enrichment; eligible keys carry 4.3× its cost | held (v636) |
| 21 | its key feature is written redundantly by taps and windows, amplified in 3–4 | response | opened | families alone 0.86× / 0.90×, both 0.23×, layers 3–4 0.37×, layers 0–4 anchor 0.12× | held (v637–v639) |
| 22 | its write copies the successor token to the logits | response+edit | opened / fresh | positive at 0.87; cost 0.020 vs 0.005 per position | held (v640) |

Instrument notes preserved: v635's base-rate broadcast bug (corrected in v636; shares unaffected), v637's doubled capture counter (fixed, re-run).


## Addendum (05:50 UTC) — the whole-model kernel atlas

The single-head kernel census was run for all 18 layers (edit, v629 / v634 / v641–v644): **159 of 162 heads** are singly replaceable by a 512-number positional kernel at ≤ 0.011 nats (exceptions 5.7 at 0.083, 2.5 at 0.025, 3.8 at 0.011); per-layer banks cost 0.004–0.05 except layer 5 (0.163). The control holds — real patterns vary far more than their kernels (median variance ratio 12 in layers 12–17, response, v645) — so the cheapness is robustness, not constancy. The joint numbers (edit, v645): all 162 kernels at once **+1.45 nats**, of which layers 0–4 alone are +1.36 and layers 6–17 together +0.70; freezing layer 5 on top of everything else *helps* (1.92 → 1.45), because its two big content heads amplify a corrupted residual. Attention-pattern content in bilin18 is therefore a distributed, deeply redundant resource whose joint effect sits in the first five layers; no single head or layer carries more than a tenth of it.

| layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| bank (CE added) | .032 | .034 | .049 | .043 | .035 | .163 | .027 | .021 | .039 | .016 | .013 | .018 | .007 | .015 | .026 | .004 | .012 | .009 |


## Addendum (07:05 UTC) — the MLP program, fitted and priced

The per-layer projection programs of the MLPs compound when stacked (claim 7) — but, as with the attention kernels, that was parameters, not form. Refitting the six rank-limited maps per layer jointly against the loss (initialised at context-PCA, validation-stopped on 96 fit rows never trained on; fit, v650–v653) gives a program that composes and scales linearly in depth:

| MLP layers | values (vs native) | unfitted (context-PCA) | fitted, held-out at the validation-chosen step |
|---|---|---|---|
| 0–2, (256, 512, 256) | 17.7M (2.7×) | 0.406 | **0.179** |
| 0–2, (128, 256, 128) | 8.8M (5.4×) | 1.446 | **0.251** |
| 0–5, (256, 512, 256) | 35.4M (2.7×) | 0.530 | **0.349** |
| 0–5, (128, 256, 128) | 17.7M (5.4×) | 1.319 | **0.442** |
| 0–17, (256, 512, 256) | 106.2M (2.7×) | 1.443 | **1.011** |

About 0.056 nats per layer at (256, 512, 256): honest, priced, and — against the registered whole-model Pareto set — not competitive, which says the value in these MLPs is not in per-mode rank. The rule that carries forward: **projection-only stacks compound; fitted programs compose** (attention kernels 1.45 → 0.96, early MLPs 0.41 → 0.18).

The attention content budget is flat (edit, v649): keeping the 9 / 18 / 54 most content-critical heads native (ranked on fit rows) recovers 0.25 / 0.34 / 0.56 of the 0.96 — pattern content is ~0.02 nats per head across the whole model, not a property of a few heads.


## Addendum (14:50 UTC, answering Logan) — novelty against the dossier, what a "kernel" replaces, and why the lane drifted

### 1. What was already in the dossier, and what is new

I did not re-read the ledger before opening this lane; checking it now (§280, §343, §398, §428, §432, §451, §475 and the board's closed-items list), here is the honest split.

**Already established before today**
- Head 0.3 is a previous-token head whose top read folds *exactly* from weights, tokens and rotary (§475) — my v612/v622 rediscovered this with more detail (the offset kernel, separability, the A/B/κ tables, and the +0.0037 edit price), but the headline was known.
- Head 5.5 is the induction head (§280, §398: "a5 contains the induction head (5,5)"), and induction heads are the least compressible heads in the model, high-rank in QK by construction (§343). My v635–v640 re-derive 5.5 from the front and add the key-side and output-side edits; the identification itself is not new.
- Head 5.7 is the position-0 attention sink and the costliest head (+0.916 to delete; §432, §451). **My board text called it "not a sink" (v636, v647); v700 re-measured with both instruments: top read at position 0 for 99.5% of queries on both row sets, value norm 771 vs 209 — the ledger was right.** What my numbers add is only that its remaining off-diagonal mass (75%) is spread ~1/i over the context.
- The stack's costliest heads are previous-token readers rather than identity matchers (§428), and deep-layer content behaves like a recency-weighted bag of words (§894/§930 lineage) — my kernel atlas is consistent with both but was not needed to state them.
- Tucker/HOSVD compression of the MLPs is on the board's **closed** list. My v611 re-derived the coefficient-space numbers exactly and then tried token-metric frames (v615), context frames (v616–v621) and fitted rank-limited programs (v650–v653). The new construction is not on the closed list, but its conclusion (not competitive with the Pareto set) agrees with the closure; treat that chapter as a confirmation with a new mechanism, not a discovery.

**New, as far as the ledger shows**
- The per-head *positional kernels* and their separability (gain(cur) × kernel(d) × content-weight(prev)) at layers 0–2, with named kernel types (previous-token, short/medium/long windows, "previous minus mean"), and the observation that heads 1.8 and 2.7 are running-mean subtractors (1.8 ≡ −1/i exactly; edit +0.0027). The ledger's bag-of-words results are about features, not about heads implementing a subtraction.
- The three-table program for twelve heads of blocks 0–1 priced at +0.036, and the price ladder for the 19 positional heads (fitted kernels only 0.087 at 9.7k numbers; with the closed-form gates 0.050).
- The whole-model kernel atlas: singles, per-layer banks, the joint numbers (+1.45 closed-form / +0.96 fitted), the flat content budget, and the pattern-variance control. The ledger has per-head deletion costs and motif classes; it did not have "replace the pattern by its positional mean" as an instrument or the joint/redundancy picture that comes out of it.
- The redundancy of the induction key feature — no tap or family of taps is necessary; taps and windows are each sufficient (v637–v639). §428 lists the taps as costly; the necessity structure is new.
- On the folding side proper: the degree census of the token-folded MLPs (v610), the bigram Tucker tables at MLP-0/1/2 (v613, v656, v658), the hop census (v614, v657), and the negative that single-token statistics do not transfer to contexts as compression frames (v616).

### 2. What "a 512-number positional kernel" means, exactly

For one head *h*, the model computes an unnormalised pattern `P_h(i, j) = (q̂_i·k̂_j / 128) · (q̂2_i·k̂2_j / 128)` for every key `j ≤ i` and forms the head output `Σ_j P_h(i, j) v_j`. The kernel edit keeps the diagonal term `P_h(i, i)` native and replaces every off-diagonal entry by a number that depends only on the offset:

`P_h(i, j) := κ_h(i − j)` for `j < i`, with `κ_h ∈ ℝ^512` (one number per offset 1…512).

So yes — it replaces the whole **QK side** of the head (`c_q, c_k, c_q2, c_k2` and the rotary; 4 × 1152 × 128 = 590k weights) by a lookup on relative position; it is an ALiBi-style fixed bias, except that this model has no softmax, so the kernel *is* the attention weight. The **values** (`c_v`, the token branch `v1` mixed with λ), the output projection `c_proj`, and the self term are untouched. "Closed-form" κ is the mean of the real pattern per offset on 64 rows; "fitted" κ is that vector refit jointly against CE with the model frozen. The three-table program of v623 is the same thing with a token gate on each side: `P_h(i, j) := κ_h(i − j) · A_h(tok_i) · B_h(tok_j)`.

So "159 of 162 heads are singly replaceable at ≤ 0.011" means: for any one head, the model barely cares which positions its QK circuit picks, as long as the values it mixes are still weighted by the usual offset profile — because the other heads still carry the content-dependent routing. The joint numbers (+0.96 fitted) are the honest measure of how much content-dependent routing the model needs in total.

### 3. Why most of the day is not embedding-forward folding

You are right, and here is what happened, in order.

- v609–v614 and v656–v658 are embedding-forward folds in your sense: exact single-token and bigram tables contracted into the early tensors (degree census, Tucker tables, hop census). That is the part of the day that answers your question.
- v615 was the first edit — does the token-metric fold *price* under CE? It did not (the single-token frames transfer badly). The follow-ups (v616–v621) chased the reason with **context** statistics and gradients; that is compression work, not folding from the embedding, and it partly retreads the closed Tucker item.
- v622–v628 came back to the tables: kernels, gates and programs of the layer-0/1/2 heads derived from single-token tables (embedding-forward objects). At layer 2 the tables got the kernel shapes right and the scales wrong (v628), and I replaced table-derived kernels with **context-mean kernels** (v629). From that point the instrument was "replace a head's pattern by its real mean" — an edit/response census on the model's own activations. The atlas (v634–v649), the induction thread (v635–v640), the fitted programs (v646, v650–v653) and the gate-reading checks (v660–v662) all use that instrument. None of them folds anything from the embedding.
- Three reasons for the drift, none of them good enough: each rung's failure suggested a cheap next rung with the instrument already in hand; the loop I set up says "never leave the queue empty", which biases toward cheap edit censuses over the harder folding rungs; and I never re-anchored to the premise. I also ran the whole night without reading the ledger for the heads I was naming, which is how the 5.7 error happened.

What is still embedding-forward and worth building on: the single-token tables through block 2 (exact), the bigram Tucker tables and the (cur, prev) interaction structure, the hop census through block 2, and the A/B/κ tables of blocks 0–1 (weights-only objects that install at +0.036). The natural embedding-forward continuations I did **not** do: the three-token fold (skip-trigrams; the layer-0 heads with d=2 peaks), the two-bilinear-layer order-five tensor with your sparse-core fit, and contracting the front-side tables against Codex's readout-side object to close the loop through the whole network. If you want the lane to stay on the folding premise, those are the next rungs, and the atlas/compression machinery should be treated as instruments for pricing them, not as the lane.

## Appendix A — layer-0 positional kernels (fold, 4096² grid; pattern rms / separable fraction)

| head | d=1 | d=2 | d=4 | d=8 | d=16 | d=32 | kernel |
|---|---|---|---|---|---|---|---|
| 0.3 | .100/.94 | .052/.87 | .011/.20 | .006/.24 | .002/.19 | .0015 | previous token |
| 0.6 | .037/.88 | .026/.80 | .010/.18 | .007 | .006 | .005 | short window |
| 0.8 | .063/.89 | .053/.87 | .025/.70 | .010/.48 | .005 | .004 | short window |
| 0.4 | .042/.91 | .043/.91 | .028/.87 | .020/.82 | .013/.75 | .009/.67 | medium window |
| 0.7 | .061/.92 | .071/.93 | .060/.92 | .037/.89 | .025/.85 | .015/.78 | long ~1/d window |
| 0.0/0.1/0.2/0.5 | .010–.014 / .30–.48 | ≈ | ≈ | .009–.013 | .007–.011 | .006–.009 | content, recency taper |

## Appendix B — receipts

`basis_aligned/bilinear_quotient/circuits/followups/embedding_forward_*_v609…v633_result.json`, runners `ops/run_embedding_forward_*.py`, shared lib `ops/embedding_forward_lib.py`, tables `*_v623_tables.pt`, `*_v624_tables.pt`, `*_v628_tables.pt`, kernels `*_v629_kernels.pt`, fitted kernels `*_v633_kernels.pt`. Board entries 03:21–05:1x UTC, 20 September. Instrument-bar failures preserved as written: v615 pred_a (the 3.29205 record was the old arc harness's measure), v628 pred_e (held vacuously on a broken set), v634 first run (index bug, re-run).
