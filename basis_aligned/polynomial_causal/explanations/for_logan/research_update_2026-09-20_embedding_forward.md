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
