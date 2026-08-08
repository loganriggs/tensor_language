# RESULTS — local box (tiny full interpretation)

Newest first. Every number here is reproducible from the JSONs named beside it;
nothing is quoted from a transcript. Registered predictions are written into
each results JSON *before* the rung that tests them runs, and the ones that
were **refuted** are marked as such rather than quietly dropped.

---

## 2026-08-08 — FINDING 1: the fold gate failures were PRECISION, and fixing the dtype made three independent controls sharper

**Verdict: precision, not a bug — and the corrected gate is strictly stronger
than the one it replaces.**

The old criterion mixed units: three relative algebraic checks at 1e-6 and one
**absolute** end-to-end logit check at 1e-5. Logits here live on
`30·tanh(·/30)` and reach 15–20, where one fp32 ulp is already ~1e-6, so the
absolute budget was about eight ulps for a forward pass that accumulates
thousands of roundings. **Four of the six trained cells failed on that clause
alone**, while every algebraic identity passed at 2–5e-7. (Naively making the
logit clause relative at 1e-6 does not help: it fails five of six, because the
fp32 algebraic tolerance of 1e-6 is itself at the rounding floor — the width-128
truncated-tokenizer cell sits at 1.05e-6 in fp32 and 7.7e-16 in fp64.)

Three pieces of evidence, not one:

1. **fp64 collapse.** Making `fold_forward`, `fold_mlp`, `fold_layer0_qk` and
   `rot_matrix` dtype-clean (they hard-cast to float32 and crashed on a
   `.double()`d model) lets the same comparison run in fp64. The end-to-end
   residual drops from 1.5e-5–2.7e-5 to **1.3e-14–4.4e-14 absolute**, i.e. about
   ten fp64 ulps at logit magnitude 15. The algebraic identities go to
   5e-16–1.5e-15 relative.
2. **The forward disagrees with itself by more.** The gate now measures
   `max|forward_fp32 − forward_fp64| / max|logit|`, the reference's own fp32
   noise: 6e-7 to 2.9e-6. At width 128 that is **larger** than the
   fold-vs-forward gap (1.7e-6). The fold agrees with the forward better than
   the forward agrees with itself.
3. **A negative control proves the new gate is not a loosening.**
   `tf_model.gate_negative_control` corrupts the MLP tensor by a factor
   `1+1e-7` and rolls the value factors by one head. The 1e-7 corruption
   produces an fp32 absolute logit difference of **1.19e-7** — the superseded
   absolute-1e-5 gate would have **passed** it; the new fp64 tier fails it
   (9.9e-9 > 1e-9). Both corruptions are caught; the clean model passes.

A dtype bug *was* found and fixed, just not in the fold algebra: `rot_matrix`
built its inverse-frequency vector in fp64 and rounded to fp32 while
`rope_tables_exact` built it in fp32, putting a one-ulp wedge between the
folded and the forward rotation. With both at the same dtype:

| control | before | after |
|---|---|---|
| planted known-answer table, δ=3 | 5.79e-9 | **1.59e-14** |
| fp64 attention-table identity | (could not run) | **7e-16** |

### The corrected gate (two tiers)

* **fp32 sanity band** — every identity, *relative*, < 1e-5 (sized by
  `sqrt(N)·eps_fp32`; the two paths do the same ~1e3–1e4 multiply-accumulates in
  a different order).
* **fp64 exactness** — algebraic identities < 1e-12 relative, end-to-end
  < 1e-9 absolute, **and** the fold-vs-forward gap ≤ 10× the forward's own
  fp32-vs-fp64 self-noise. The last clause is what would catch a small genuine
  bug hiding under a fixed threshold.

### Identity table (all local checkpoints; `tf_identity_table.json`)

fp32 columns are relative except the logit-abs column; fp64 columns are the
real exactness gate.

| stem | pass | fp32 mlp T | fp32 gauge | fp32 attn | fp32 logit abs | fp32 logit rel | fp64 mlp T | fp64 attn | fp64 logit abs | fwd self-noise | gap/noise | planted δ=3 | neg ctl |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| d1_w128·bpe_s0 | ✓ | 4.5e-07 | 5.2e-07 | 9.8e-07 | 2.7e-05 | 1.73e-06 | 1.4e-15 | 6.8e-16 | 4.4e-14 | 2.9e-06 | 0.59 | 1.6e-14 | ✓ |
| d1_w128·bpe_s1 | ✓ | 5.5e-07 | 4.5e-07 | 6.4e-07 | 3.8e-05 | 2.47e-06 | 1.7e-15 | 7.1e-16 | 3.0e-14 | 3.4e-06 | 0.72 | 1.6e-14 | ✓ |
| d1_w128·bpe_s2 | ✓ | 4.2e-07 | 4.8e-07 | 1.6e-06 | 3.8e-05 | 2.47e-06 | 1.3e-15 | 1.2e-15 | 4.4e-14 | 3.7e-06 | 0.66 | 1.6e-14 | ✓ |
| d1_w128·trunc_s0 | ✓ | 2.9e-07 | 4.6e-07 | 1.1e-06 | 2.4e-05 | 1.73e-06 | 1.5e-15 | 6.6e-16 | 2.9e-14 | 1.7e-06 | 0.99 | 1.6e-14 | ✓ |
| d1_w256·bpe_s0 | ✓ | 5.3e-07 | 5.2e-07 | 1.5e-06 | 4.7e-05 | 2.90e-06 | 1.8e-15 | 8.5e-16 | 5.9e-14 | 4.3e-06 | 0.68 | 1.6e-14 | ✓ |
| d1_w32·bpe_s0 | ✓ | 2.8e-07 | 3.4e-07 | 4.6e-07 | 2.1e-05 | 1.40e-06 | 8.5e-16 | 7.2e-16 | 3.7e-14 | 6.2e-07 | 2.28 | 1.6e-14 | ✓ |
| d1_w32·bpe_s1 | ✓ | 1.9e-07 | 3.7e-07 | 3.8e-07 | 1.7e-05 | 1.22e-06 | 6.9e-16 | 6.6e-16 | 3.6e-14 | 1.5e-06 | 0.79 | 1.6e-14 | ✓ |
| d1_w32·bpe_s2 | ✓ | 2.6e-07 | 3.3e-07 | 2.9e-07 | 1.3e-05 | 9.29e-07 | 8.6e-16 | 7.8e-16 | 4.7e-14 | 9.3e-07 | 1.00 | 1.6e-14 | ✓ |
| d1_w32·trunc_s0 | ✓ | 3.1e-07 | 3.9e-07 | 9.2e-07 | 7.2e-06 | 5.63e-07 | 4.7e-16 | 1.5e-15 | 2.1e-14 | 1.2e-06 | 0.49 | 1.6e-14 | ✓ |
| d1_w64·bpe_s0 | ✓ | 2.5e-07 | 3.5e-07 | 4.5e-07 | 1.5e-05 | 9.21e-07 | 1.0e-15 | 6.5e-16 | 2.6e-14 | 1.9e-06 | 0.49 | 1.6e-14 | ✓ |
| d1_w64·bpe_s1 | ✓ | 4.2e-07 | 4.0e-07 | 2.0e-07 | 1.1e-05 | 8.05e-07 | 1.1e-15 | 4.6e-16 | 2.4e-14 | 1.6e-06 | 0.52 | 1.6e-14 | ✓ |
| d1_w64·bpe_s2 | ✓ | 3.4e-07 | 3.6e-07 | 4.5e-07 | 1.2e-05 | 8.13e-07 | 9.5e-16 | 9.4e-16 | 2.9e-14 | 2.2e-06 | 0.37 | 1.6e-14 | ✓ |
| d1_w64·trunc_s0 | ✓ | 2.3e-07 | 2.5e-07 | 3.3e-07 | 6.7e-06 | 4.86e-07 | 8.1e-16 | 8.5e-16 | 1.3e-14 | 5.8e-07 | 0.83 | 1.6e-14 | ✓ |
| d2_w32·bpe_s0 | ✓ | 2.2e-07 | 3.1e-07 | 9.4e-07 | 1.5e-05 | 1.11e-06 | 7.3e-16 | 2.1e-15 | 4.0e-14 | 1.9e-06 | 0.59 | 1.6e-14 | ✓ |
| d2_w64·bpe_s0 | ✓ | 2.5e-07 | 3.6e-07 | 4.9e-07 | 1.0e-05 | 7.18e-07 | 1.2e-15 | 1.1e-15 | 2.3e-14 | 3.6e-06 | 0.20 | 1.6e-14 | ✓ |
| d2_w64·bpe_s1 | ✓ | 3.7e-07 | 3.8e-07 | 3.2e-07 | 1.2e-05 | 9.43e-07 | 1.1e-15 | 6.0e-16 | 2.1e-14 | 2.8e-06 | 0.34 | 1.6e-14 | ✓ |

All **16** local checkpoints pass, depth-2 cells included.

Also green in `tf_identity_table.json` but omitted above for width: the
all-heads factor-indexed attention identity (a V×V-free recomputation of the
same quantity, added because materializing 16 heads × 4 distances in fp64 at
V=8192 is 68 GB and OOM-killed the width-256 fold), the QR-vs-eigenvalue
spectrum control (2e-13 to 2e-10 relative) and the factor-vs-dense-SVD control
(2e-16 to 1e-14).

**The six width-256 cells from the scale box could NOT be re-folded: only their
JSONs were pushed, not their `.pt` files** (`*.pt` is untracked here). A
width-256 depth-1 cell was retrained locally on the primary BPE corpus instead
(held CE 4.5583) and is folded and interpreted; it is the `w256` column
throughout.

---

## 2026-08-08 — FINDING 2: at depth 1 the model is a QUADRATIC FORM with an attention-driven input; the residual stream is invisible at the readout

Depth-1 vanilla with `n_slots = 1` folds exactly (verified to 1e-6 relative in
fp32, `decomposition_control` in every `*_interp.json`) into

```
e_i      = Ehn[t_i]                                    (current token only)
p_h[i,j] = s1_h(t_i,t_j,i−j) · s2_h(t_i,t_j,i−j)        (token-pair × distance)
A_i      = Σ_h Σ_{j≤i} p_h[i,j] · OV_h[t_j]
M_i      = T(rms(e_i+A_i), rms(e_i+A_i)) + b
logits_i = 30·tanh( rms(e_i+A_i+M_i) · W_Uᵀ / 30 )
```

Because RMSNorm is a scalar gauge, the pre-tanh logit is **exactly additive** in
the three folded terms, so their shares can be read off with no approximation.
Measured on held text (mean over 3 seeds):

| width | ‖e‖ | ‖A₀‖ | ‖A_past‖ | ‖M‖ | logit share of M |
|---|---|---|---|---|---|
| 32 | 5.66 | 0.85 | 4.0 | 3268 | **0.99988** |
| 64 | 8.00 | 1.7 | 7.3 | 6069 | **1.00017** |
| 128 | 11.31 | 3.1 | 11.2 | 10349 | **1.00006** |
| 256 | 16.00 | 4.9 | 15.9 | 18431 | **1.00018** |

(the four shares sum to 1 to 5e-8 by construction; the small excess over 1 is
the embedding term's *negative* share, −5e-4)

Causal confirmation, not just geometry: discarding the embedding **and** both
attention writes from the residual and keeping only the MLP write reproduces
the model at **KL 1e-5 to 3e-5**. The skip connection into the readout is
functionally dead.

### The attention's whole effect is on the MLP's INPUT

This is the correction of an earlier claim (see the retraction below). The two
routes, as mutually exclusive ablations that bracket the model (KL from the
real model, mean ± sd over 3 seeds):

| stage | w32 | w64 | w128 | w256 (1 seed) |
|---|---|---|---|---|
| no attention at all | 0.285 ± 0.025 | 0.466 ± 0.015 | 0.687 ± 0.010 | 0.911 |
| past attention **direct route only** (MLP frozen at its no-context input) | 0.258 ± 0.020 | 0.431 ± 0.013 | 0.644 ± 0.007 | 0.851 |
| past attention **MLP route only** (A_past removed from the residual) | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| full model | 0 | 0 | 0 | 0 |

The direct route lands on the no-attention number; the MLP route lands on zero.
100% of what attention buys is delivered by moving the quadratic form's
argument.

---

## 2026-08-08 — FINDING 3 (RUNG 5): the KL ladder, and what each component buys

All stages are weights-free table programs (look up rows of `Ehn`, `A0`, `M0`,
`OV`, index the branch factors by token id, apply the rotary, read out with
`W_U`); no stage calls the network's forward. Scored on **held** text; the only
fitted objects in the ladder (the token-independent distance profile, the
mean-ablation value) are fitted on the **estimation** split. KL from the real
model, nats/token, mean ± sd over 3 seeds.

| stage | w32 | w64 | w128 | w256 (1 seed) |
|---|---|---|---|---|
| embedding only | 8.880 ± 0.201 | 12.294 ± 0.410 | 15.900 ± 0.208 | 18.873 |
| + attention to self (δ=0) | 8.676 ± 0.250 | 11.790 ± 0.470 | 15.171 ± 0.302 | 18.02 |
| **+ MLP ⇒ the model's own bigram** (weights-only V×V table) | **0.258 ± 0.020** | **0.431 ± 0.013** | **0.644 ± 0.007** | **0.852** |
| + past attention, distance ≤ 1 | 0.167 ± 0.013 | 0.266 ± 0.012 | 0.378 ± 0.004 | 0.485 |
| + past attention, distance ≤ 4 | 0.092 ± 0.005 | 0.148 ± 0.006 | 0.211 ± 0.005 | 0.276 |
| + past attention, distance ≤ 16 | 0.029 ± 0.001 | 0.051 ± 0.004 | 0.079 ± 0.001 | 0.106 |
| + past attention, distance ≤ 64 | 0.004 ± 0.001 | 0.007 ± 0.001 | 0.011 ± 0.000 | 0.015 |
| + past attention, all distances (= exact) | 0 | 0 | 0 | 0 |

Ablation variants at the same stage:

| variant | w32 | w64 | w128 | w256 |
|---|---|---|---|---|
| pattern replaced by its token-independent distance profile | 0.240 | 0.260 | 0.271 | 0.292 |
| — i.e. fraction of the attention effect that is PURELY POSITIONAL | 16% | 44% | 61% | 68% |
| pattern with the ROTARY REMOVED (δ=0 table at every distance) | 0.960 | 1.294 | 1.695 | 2.113 |
| top 4 of 8 rotary frequency pairs kept | 0.101 | 0.195 | 0.247 | 0.361 |
| top 2 of 8 rotary frequency pairs kept | 1.252 | 1.560 | 3.151 | — |
| MLP restricted to its 64 most-used hidden units (of 128/256/512/1024) | 0.671 | 1.396 | 2.138 | — |

Readings:

* **Two terms carry the model.** The weights-only bigram table takes KL from 8.9
  to 0.26 at width 32; the folded past attention takes the rest to 0. There is
  no third ingredient.
* **The attention is mostly a learned DISTANCE KERNEL, not a content lookup.**
  Replacing the whole token-pair pattern with its distance-only average keeps
  16% of the attention's value at width 32 but **61% at width 128**; removing
  the distance information and keeping only the token-pair table is
  catastrophic (1.7 nats, worse than having no attention at all). The
  query/key token dependence is the *minority* contribution at the widest cell.
* **Registered prediction REFUTED (`rung3_skipgram`).** We registered that
  distance ≥ 2 would be worth less than distance 1. At width 128 the δ=1 term
  buys 0.649−0.378 = 0.271 and everything beyond it buys 0.378 — the longer-range
  skip-grams are worth **more**, and the same ordering holds at every width.
* **Registered prediction PARTLY REFUTED (`rung3_positional`).** We registered
  that the distance-only pattern would destroy most of the attention gain. It
  destroys most of it at width 32 and a minority of it at width 128.
* **The MLP is not compressible in its own basis.** Half the hidden units (a
  genuine CP-term truncation, since the bilinear MLP *is* a rank-`hidden`
  symmetric CP decomposition) leaves KL 0.67–2.14, i.e. worse than deleting the
  attention entirely.

### Against data baselines (held CE, nats/token; baselines fitted on train/est)

| predictor | CE | parameters |
|---|---|---|
| unigram | 7.260 | 8 192 |
| positional-only (p(next\|position), fitted on est) | 7.718 | 512·8 192 |
| low-rank bigram, rank 32 | 6.649 | 524 288 |
| low-rank bigram, rank 64 | 6.469 | 1 048 576 |
| sparse bigram, top 262 144 counts + unigram backoff | 5.675 | 524 288 |
| sparse bigram, top 1 048 576 counts + unigram backoff | 5.322 | 2 097 152 |
| dense closed-form bigram (α = 1000) | 5.200 | 67 108 864 |
| **model, width 32** | 5.413 | 280 608 |
| **model, width 64** | 5.048 | 598 080 |
| **model, width 128** | 4.723 | 1 343 616 |
| **model, width 256** | 4.461 | 3 400 704 |
| model's own bigram stage, w32 / w64 / w128 | 5.720 / 5.566 / 5.490 | 524k / 1.05M / 2.10M tables |

Honest readings, including the ones that do not flatter the model:

* Widths 64 and 128 beat the dense bigram table with 50–100× fewer parameters.
  Width 32 does **not** (5.413 vs 5.200).
* At **matched parameter count** the weights-only *model-bigram stage* **loses**
  to a data-fitted sparse bigram (5.490 vs 5.322 at 2.1M). The model only wins
  once its attention term is included. So "the model is a better bigram than a
  bigram" is false; "the model is a better *context* model than a bigram" is
  true from width 64.
* The comparison is not made fair by parameter count alone, because the model
  sees the whole prefix and the position. The position profile settles it: at
  **position 0**, where the model and the bigram see exactly the same one token
  of context, the bigram wins at every width (5.489 vs 5.855–6.056). The model
  overtakes it from about position 8 at widths 64–128 and never at width 32.

---

## 2026-08-08 — FINDING 4 (RUNG 2): selection is low rank, content is not — with nulls

`rank ≤ head_dim` and `rank ≤ hidden` are **arithmetic**, not findings. What is
reported is the distance below the bound, measured by spectral-entropy
effective rank `exp(H(σ/Σσ))`, against an iid-Gaussian null of the same shape.

| object | bound | trained (mean ± sd over 3 seeds) | null |
|---|---|---|---|
| branch score table s1, δ=0, w32 | 16 | **2.28 ± 0.61** | 15.991 ± 0.001 |
| branch score table s1, δ=0, w64 | 16 | **2.91 ± 0.32** | 15.991 ± 0.001 |
| branch score table s1, δ=0, w128 | 16 | **3.40 ± 0.32** | 15.991 ± 0.001 |
| branch score table s1, δ=0, w256 (1 seed) | 16 | **5.93** | 15.991 ± 0.001 |
| query factor Q1, w128 | 16 | 5.08 | 15.996 ± 0.001 |
| **value factor Vv, w128** | 16 | **15.56 ± 0.02** | 15.996 ± 0.001 |
| MLP tensor, mode-0 unfolding, w128 | 128 | **121.9 ± 0.1** | 123.2 (random *factored* tensor, same shapes) |
| MLP tensor, mode-0 unfolding, w32 | 32 | 30.0 ± 0.1 | ~31 |
| MLP tensor, mode-0 unfolding, w256 | 256 | 239.7 | 246.8 |
| value factor Vv, w256 | 16 | 15.66 | 15.996 ± 0.001 |

This is the parent program's headline reproduced at the smallest possible
scale: **selection (query/key) is strongly low rank — three effective
directions out of sixteen, against a null of sixteen — while content (the value
factor and the MLP tensor) is spectrally indistinguishable from a random object
of the same shape.** Registered prediction `rung2_low_rank` is **half right**:
the score-table part was predicted and confirmed; the MLP part predicted "well
below its bound" and is refuted.

The low-rank selection claim also has a causal version: keeping only the top 4
of 8 rotary frequency pairs (the only δ-equivariant way to cut a head's
subspace) keeps most of the attention's value.

---

## 2026-08-08 — FINDING 5 (RUNG 3): no induction at depth 1 **or depth 2**, and the metric is calibrated

Three matched synthetic conditions with identical token multisets, scored on
the second copy only: `repeat = [R][R]`, `shuffled = [shuffle(R)][R]`,
`control = [R'][R]`; induction score = CE(shuffled) − CE(repeat) (needs
**order**, hence composition, hence ≥ 2 layers); bag score = CE(control) −
CE(shuffled) (needs only a bag).

| cell | induction score | bag score |
|---|---|---|
| depth 1, w32 (3 seeds) | −0.006 ± 0.002 | +0.015 ± 0.002 |
| depth 1, w64 (3 seeds) | −0.012 ± 0.002 | +0.031 ± 0.006 |
| depth 1, w128 (3 seeds) | −0.026 ± 0.002 | +0.060 ± 0.002 |
| depth 1, w256 (1 seed) | −0.034 ± 0.009 | +0.081 ± 0.009 |
| depth 2, w32 (1 seed) | −0.007 ± 0.008 | +0.021 ± 0.008 |
| **depth 2, w64 (2 seeds)** | **−0.014 ± 0.003** | +0.044 ± 0.014 |

**The registered depth-2 positive control FAILED**: we registered that a depth-2
cell "must show a nonzero induction score, otherwise the metric is broken". It
did not. So the null was re-established a different way — by planting a known
amount of induction and finding the detection floor. Mixing the model with a
perfect induction oracle at weight ε:

| ε | induction score (depth-2 w64) |
|---|---|
| 0 | −0.0154 ± 0.0054 |
| 1e-4 | **+0.940 ± 0.023** |
| 3e-4 | +1.412 ± 0.037 |
| 1e-3 | +2.030 ± 0.053 |
| 1e-2 | +3.399 ± 0.081 |

A mixture weight of **0.01%** already moves the score by 175 standard
deviations. The battery is not blind; these models simply have no induction.
The honest statement is therefore: *at depths 1–2 and widths ≤ 128 on this
corpus and this 15 000-step single-epoch budget, induction is absent to within
~0.02 nats* — which is a statement about this regime, not a proof that depth 2
cannot induct.

The second number is deliberately called a **bag** score, not a copy score:
rung 4 shows the attended token ranks near the *bottom* of what attending to it
boosts, so naming the bag effect "copying" would be inferring a mechanism from
a behavioural delta.

---

## 2026-08-08 — FINDING 6 (RUNG 4): the heads are not copy heads, and the composed pair table barely factorises

Everything here is composed to logits before it is named (the standing sign
rule): the object measured is
`C_h(t,u,δ)[v] = p_h(t,u,δ)·(OV_h[u]·W_Uᵀ)[v]`, never a raw factor.

* **Not copy heads.** For each head's eight strongest keys, the median rank of
  the attended token among the tokens it boosts is **≈ 5 600 of 8 192**.
  Attending to a token pushes its own logit *down* relative to a random pair
  (identity-pair z of −3.4 to +1.8 across heads). This is reported as "attending
  to a token does not push its own logit up", **not** as suppression.
* **The pair table is close to an outer product.** The σ₁ share of the composed
  (query, key) matrix is 0.37–0.85 per head (median ≈ 0.74), with entropy rank
  2.5–14 out of a bound of 256. Most heads therefore have almost no genuine
  *pair* specificity: what they do is approximately (a score for the query) ×
  (a fixed write for the key).
* **What attending does is generic.** Width 128, head 0: the four strongest keys
  are all closing-quote tokens (`,”`, `.”`, `”`, `”.`) and every one of them
  boosts the same continuation set — `.` (+59), ` and` (+47), `,` (+47), ` in`
  (+40), ` to` (+32). That is a punctuation-context head that writes a generic
  "sentence continues" direction, not a content lookup.
* **Token-class claims, with a frequency-matched null** (400 draws, same size,
  drawn with train unigram probability). Only classes at |z| > 3 are named. At
  widths 32 and 64 head 0's strongest value directions are enriched for
  whitespace-initial lowercase word pieces (z = +3.8, +5.6) and depleted of
  capitalised pieces (z = −3.0, −4.9). At width 128 **nothing** clears |z| = 3
  and no class is named.
* **Registered prediction REFUTED (`rung4_tokens`).** We registered that the
  composed copy score would be dominated by a few token pairs and enriched for
  identical tokens. It is diffuse (effective pair fraction 0.08–0.44 of all
  sampled pairs) and identity pairs are *de*-enriched.

---

## RETRACTION (2026-08-08, same day)

`MAILBOX.md` 2026-08-08 05:00 and commit `631ddaa20` reported that at depth 1
"attention to the past buys 0.0005 nats — nothing" and that "every distance
restriction lands on top of the full thing". **That is wrong.** The ladder that
produced it added `A_past` to the residual while holding the MLP frozen at its
no-context input, so it measured the *direct* route only. Attention is worth
0.29 / 0.47 / 0.69 nats of KL at widths 32 / 64 / 128, and every bit of it goes
through the MLP. The distance-restriction table in that entry is superseded by
the one in FINDING 3.

The failure mode is exactly the one the standing sign/gauge rule describes, in
a non-sign form: **a term was scored without composing it through the
downstream nonlinearity.** It is now in the README failure-mode list.

The self-red-team of every claim above, with what was fixed and what could not
be, is `tf_reviewer_round_1.json`.
