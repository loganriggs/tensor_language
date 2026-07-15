# OV circuit + bilinear-MLP interaction blocks (546M, layer/block 0)

Logan's steer: the O projection "needs to hook up somewhere" — its output basis means
nothing alone; fold it against its readers. For a bilinear MLP reading x = n(e + a)
(embedding + attention-out), the hidden splits EXACTLY (frozen empirical rms) into

    (Lx)⊙(Rx) = [Le⊙Re] + [Le⊙Ra + La⊙Re] + [La⊙Ra]
                  self       cross             source-pair

## Block importance (drop one block, ΔCE; split gate exact to 2.4e-7)

| dropped block | ΔCE |
|---|---|
| self (embedding × itself) | +1.291 |
| **cross (current token × attention-out)** | **+0.840** |
| source-pair (attention-out × itself) | +0.187 |

The cross block — Logan's "attention-out conditioned on that specific token" — is a
first-class citizen (~2/3 of self). The near-one-hot intuition is MOSTLY right: the
source×source block is 5–7× smaller, but previous tokens do interact inside the bilinear
layer (+0.19 nats), via the a⊙a term.

## OV sparse-on-its-own: content is NOT coarsely classable

Layer-0 value tables v_h(t) (V×128 per head, the folded OV input; patch replaces layer-0
values everywhere they're used, incl. the v1 share to later layers):

| compression | DL ratio (v-tables) | ΔCE (L2-fit) |
|---|---|---|
| vq64 | 0.003 | +2.019 |
| vq1024 | 0.023 | +0.883 |
| svd16 | 0.125 | +1.295 |
| svd64 (half rank) | 0.501 | +0.114 |
| zero | 0 | +4.362 |

**The selection/content dichotomy:** QK (selection) is a coarse ~256-token-class
computation (vq64 ≈ +0.015 raw, negative CE-trained); OV (content) needs fine token
identity (vq64 +2.02) — the transported content behaves like the raw embedding did in
basis_aligned e6 ("the tokens are the objects"), while selection is classable.

## CE-trained OV tables

| OV codebook | DL ratio | L2-fit | CE-trained |
|---|---|---|---|
| vq1024 | 0.023 | +0.917 | +0.568 |
| vq4096 | 0.084 | +0.782 | +0.475 |

CE-training recovers only ~38% — hard token-classing fails for content even with the
behavioral objective (contrast QK, which went NEGATIVE under the same treatment).

## Sparse coding rescues content (the e7 move, confirmed)

Signed top-k dictionaries per head (each token = k-sparse combination of shared atoms):

| OV codebook | DL ratio | ΔCE (L2-fit) | ΔCE (CE-trained) |
|---|---|---|---|
| topk n=512, k=4 | 0.050 | +0.315 | — |
| topk n=2048, k=4 | 0.083 | +0.169 | — |
| **topk n=512, k=16** | 0.170 | **+0.034** | **−0.019** |
| topk n=2048, k=16 | 0.209 | +0.043 | — |
| (hard vq256, for contrast) | 0.033 | +1.383 | ~+0.57 at k=1024 |

**The refined dichotomy:** selection (QK) tolerates HARD CLASSES; content (OV) needs
SPARSE COMBINATIONS — but under the right prior + behavioral training, both circuits of
layer 0 compress to better-than-original: QK at −0.039 (vq256 CE-trained), OV at −0.019
(512 atoms × 16 coefficients, CE-trained). Exactly the basis_aligned e7 pattern (hard vq
+0.87 vs sparse +0.26 on the embedding), now on both attention circuits.

Next: the V×V cross-block codebook as its own object (token t × transported token s →
hidden), per Logan's suggestion — now justified by the +0.84 block importance.

## The complete MLP-0 decomposition (cross-block + self-block codebooks)

Both input sides of every interaction block classed independently (split-gate 1.2e-7;
`cross_block_codebook.json`, `self_block_codebook.json`):

| block (importance) | class tolerance |
|---|---|
| self, e⊙e (+1.29) | k=256: +0.097 · k=1024: +0.056 · k=4096: +0.030 |
| cross, e⊙a (+0.84) | current-side k_t=256: **+0.043** · source-side k_s=256: **+0.055** · both: +0.206 (superadditive) · 4096²: +0.085 |
| pair, a⊙a (+0.19) | untested (small block) |

**Synthesis for layer 0 of the 546M model:** every *interaction* is class-tolerant —
QK selection at ~256 hard classes (−0.039 CE-trained), the bilinear MLP's self and cross
blocks at ~256–1024 classes per input side — while the single class-INTOLERANT object is
the direct value/residual transport (+1.38 at vq256), which instead sparse-codes
(−0.019 CE-trained, 512 atoms × 16 coefficients). And the sharp contrast: classing source
content inside the cross term costs +0.055 where classing it globally costs +1.38 —
content precision matters through the residual transport path, not through the MLP
interaction. Computations that COMPARE or COMBINE tokens need only their classes;
the computation that CARRIES a token forward needs its fine identity.

**CE-trained MLP-0 codebooks** (assignments frozen, three class tables trained through
the frozen model): combined self@256 + cross@256² goes **+0.166 → +0.022**. Layer-0
component scoreboard, all CE-trained: QK selection **−0.039**, OV transport **−0.019**,
MLP interactions **+0.022** — the grand-combined single compressed layer-0 is the next
flagship arm.
