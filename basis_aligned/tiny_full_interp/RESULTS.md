# RESULTS — local box (tiny full interpretation)

Newest first. Every number here is reproducible from the JSONs named beside it;
nothing is quoted from a transcript. Registered predictions are written into
each results JSON *before* the rung that tests them runs, and the ones that
were **refuted** are marked as such rather than quietly dropped.

---

## 2026-08-08 — FINDING 7 (DEPTH 2): "attention is inert" was a property of the LADDER, not of the model

**Verdict: the depth-1 headline does not survive its own adversarial test, at
either depth.** The claim came from one increment — the gap between the
bigram-only reconstruction and the no-attention-at-all knockout — and that
increment is not attention's marginal value. It is the gap between two
*different* reduced models, one of which (the bigram) still contains the
self-attention term and has already frozen the context away.

The same two components, added in both orders, on held text, KL from the true
model (`*_order.json`, `tf_interp2.ladder_order`; mean ± sd over 3 seeds):

| cell | attention added FIRST | attention added LAST | ratio | MLP first | MLP last |
|---|---|---|---|---|---|
| depth 1, w32 | 2.030 ± 0.028 | 0.290 ± 0.027 | **7.0** | 8.549 | 6.810 |
| depth 1, w64 | 3.460 ± 0.231 | 0.475 ± 0.017 | **7.3** | 11.779 | 8.794 |
| depth 1, w128 | 4.659 ± 0.151 | 0.707 ± 0.009 | **6.6** | 15.158 | 11.206 |
| depth 1, w256 | 4.074 | 0.939 | 4.3 | 17.923 | 14.787 |
| depth 2, w32 | 4.224 ± 0.050 | 0.371 ± 0.012 | **11.4** | 8.033 | 4.180 |
| depth 2, w64 | 7.670 ± 0.522 | 0.617 ± 0.006 | **12.4** | 11.564 | 4.510 |
| depth 2, w128 | 11.633 ± 0.473 | 0.941 ± 0.007 | **12.4** | 15.351 | 4.659 |
| depth 2, w256 | 15.561 | 1.229 | 12.7 | 18.541 | 4.208 |

Readings, including the ones that cost us a headline:

* **No single number is "what attention is worth".** It ranges over a factor of
  4–13 depending only on where in the ladder it is added. The order-free
  Shapley average is the honest scalar; the depth-1 mailbox number (0.04 nats)
  is neither marginal — it is smaller than *both*.
* **What actually changes with depth is attention's STANDALONE capability, not
  its necessity.** Attention-with-no-MLPs goes from KL 8.88 (depth 1, w64) to
  4.55 (depth 2, w64): two attention layers compose into something twice as
  good on their own. Its marginal on top of the MLPs barely moves
  (0.47 → 0.61). The MLPs still do the same job, so the second attention layer
  is mostly *redundant capability*, not new function.
* **Under an on-distribution (resample) ablation attention is worth 2-3x more
  than the zeroing says** — 1.12/1.44 nats at depth 1 widths 128/256 and
  1.51/2.01 at depth 2 — so every "attention is cheap" number in this program,
  including the ones above, is a LOWER bound. See FINDING 8.
* **The depth-1-style increment reproduces at depth 2 and is still small**
  (no-attention-at-all minus bigram: 0.032 / 0.049 / 0.107 / 0.131 at widths
  32/64/128/256). Registered prediction `d2_attention_not_inert` is **half
  right**: the two framings do continue to disagree, as predicted, but the
  absolute knockout cost at width 64 is 0.61, not the ">1.0 nats" registered.
  **Refuted on the number, confirmed on the mechanism.**

The full depth-2 ladder (KL from the model, held text, mean ± sd over 3 seeds;
width 256 is one seed):

| stage | d2 w32 | d2 w64 | d2 w128 | d2 w256 |
|---|---|---|---|---|
| embed only | 8.44 | 12.21 | 16.31 | 19.77 |
| model's own bigram (weights-only table) | 0.333 ± 0.008 | 0.559 ± 0.001 | 0.815 ± 0.007 | 1.058 |
| no attention at all | 0.366 ± 0.012 | 0.608 ± 0.007 | 0.922 ± 0.007 | 1.189 |
| past attention mean-ablated | 0.357 ± 0.015 | 0.588 ± 0.003 | 0.889 ± 0.003 | 1.202 |
| no MLP (both) | 4.19 ± 0.05 | 4.55 ± 0.50 | 4.70 ± 0.45 | 4.18 |
| pattern replaced by its distance profile | 0.248 | 0.268 | 0.318 | 0.418 |
| rotary removed | 1.81 | 3.00 | 3.50 | 3.71 |

CE and bits/byte (BPE V=8192, 3.755 bytes/token): depth 2 reaches 5.3166 /
4.9124 / 4.5503 / 4.2446 nats at widths 32–256 (2.043 / 1.888 / 1.748 / 1.631
bits per byte), against depth 1's 5.4130 / 5.0477 / 4.7234 / 4.4613. **A second
layer buys 0.10–0.22 nats — less than one width doubling buys** (0.37).

---

## 2026-08-08 — FINDING 8 (DEPTH 2): layer 1 reads the MLP, not the attention — the composition channel is 0.1–0.4% wide

The two attention layers, deleted separately (KL from the model; the deletion
is a full re-run of the folded pipeline, so everything downstream responds):

| cell | delete layer-0 attention | delete layer-1 attention | delete both | sum of the two |
|---|---|---|---|---|
| w32 | 0.123 ± 0.016 | **0.232 ± 0.017** | 0.366 | 0.355 |
| w64 | 0.253 ± 0.020 | **0.334 ± 0.014** | 0.608 | 0.587 |
| w128 | **0.559 ± 0.013** | 0.510 ± 0.032 | 0.922 | 1.069 |
| w256 | **0.889** | 0.621 | 1.189 | 1.510 |

**Registered prediction `d2_layer_split` REFUTED**: we registered that layer 0
dominates at every width. Layer *1* dominates at widths 32 and 64, and under
the zero-ablation the ordering appears to flip at 128. And the two deletions
are *super*-additive at 32–64 (joint > sum: the layers back each other up) and
*sub*-additive at 128–256.

**But the flip is an artifact of the ablation, and the reviewer round caught
it.** A zeroed write is off distribution, so a **resample ablation** was added
(`resample_ablation`): replace the layer's attention write with the write that
same layer produced on a *different* sequence — a real output of that module,
on distribution by construction.

| cell | layer 0: zero → resample | layer 1: zero → resample | both: zero → resample |
|---|---|---|---|
| d1 w128 | 0.703 → **1.118** | — | 0.703 → **1.118** |
| d1 w256 | 0.939 → **1.435** | — | 0.939 → **1.435** |
| d2 w32 | 0.129 → 0.215 | 0.232 → **0.473** | 0.371 → **0.667** |
| d2 w64 | 0.260 → 0.376 | 0.336 → **0.594** | 0.617 → **1.007** |
| d2 w128 | 0.566 → 0.535 | 0.520 → **0.861** | 0.941 → **1.510** |
| d2 w256 | 0.905 → 0.782 | 0.644 → **1.075** | 1.229 → **2.013** |

Two consequences, both against our own earlier statements:

* **Zeroing was the GENTLER intervention almost everywhere.** The resample cost
  exceeds the zero cost at 13 of 14 layer-cells, so the knockout numbers quoted
  above (and at depth 1) *understate* attention's value rather than inflating
  it with distribution shift. The only exceptions are layer 0 at widths 128–256,
  where 12–14% of the zeroing cost is distribution shift.
* **The layer ordering does NOT flip.** Under the on-distribution ablation,
  layer-1 attention costs more than layer-0 attention at **every** width. The
  flip at 128 was a property of the zeroing, and the honest statement is
  "layer 1 carries more, and the zero-ablation understates that at large
  widths."

**What layer 1 reads** (`composition_budget`, held text). Layer 1's module
input is `rms(e + A0 + M0)`; the shares of that vector's norm, and the relative
change in layer 1's own attention pattern when each write is deleted **from the
read only** (the residual is untouched, so nothing else moves):

| cell | share of read: e | share: layer-0 attention | share: MLP-0 | pattern change without layer-0 attention | without MLP-0 |
|---|---|---|---|---|---|
| w32 | 0.37% | **0.075%** | 99.98% | **0.14%** | 145% |
| w64 | 0.31% | **0.114%** | 99.98% | **0.19%** | 124% |
| w128 | 0.31% | **0.227%** | 99.96% | **0.33%** | 126% |
| w256 | 0.31% | **0.416%** | 99.91% | **0.60%** | 121% |

And the causal version, in the ladder: substituting `rms(e + M0)` for layer 1's
read — i.e. deleting layer-0's attention write from what layer 1 sees —
reproduces the model at **KL 0.0000 at every width and seed**. Substituting
`rms(e)` costs 0.80–1.46, which is *worse* than deleting layer-1 attention
outright, and substituting `rms(e + A0)` costs 0.86–1.68.

So: **the attention→attention path — the one the textbook induction circuit
runs on — is numerically closed in these models.** Layer 1's selection is a
function of the layer-0 MLP's write and essentially nothing else. The channel
does widen monotonically with width (0.075% → 0.416%), which is the only
structural quantity we have found that moves in the direction of composition.

---

## 2026-08-08 — FINDING 9 (DEPTH 2): induction APPEARS, at width 256, and it does not use the residual-stream composition path

**Registered prediction `d2_induction` REFUTED at width 256, held at 128.** We
registered, before measuring the unmeasured cells, that the induction score
would stay within ±0.05 nats and under 3 standard errors at depths 2, widths
128 and 256.

| cell | induction score | bag score | detectable-effect floor (3 SE) |
|---|---|---|---|
| depth 1, w32 / w64 / w128 / w256 (3 seeds each) | −0.006 / −0.012 / −0.026 / −0.035 | +0.015 / +0.031 / +0.060 / +0.081 | — |
| depth 2, w32 (3 seeds) | −0.008 ± 0.002 | +0.020 | 0.008 |
| depth 2, w64 (3 seeds) | −0.014 ± 0.002 | +0.045 | 0.011 |
| depth 2, w128 (3 seeds) | −0.003 ± 0.010 | +0.086 | 0.010 |
| **depth 2, w256 (3 seeds)** | **+0.0938 ± 0.0086** | +0.133 | 0.006–0.017 |

(depth-2 width-256 per seed: +0.0841, +0.0965, +0.1007, each 5–17× its own
floor; the depth-1 width-256 matched cells are −0.0354 ± 0.0015 over 3 seeds,
so the flip is between depths at fixed width, not a width effect on its own.)

The width-256 score is **five times the battery's own detectable-effect floor**
and the first positive value anywhere in the program. It is corroborated by an
independent probe on **real held text**: destroying the induction evidence with
a **bag-preserving swap** (exchange the token that followed the earlier
occurrence with another prefix token — a permutation, so the prefix multiset is
identical and only the adjacency changes) costs the model 0.244 nats on the
induction target. Because a *depth-1* model — which structurally cannot compose
— also scores positive on that probe (its distance kernel notices the swap),
the depth-1 cell at the same width is used as the **matched null**:

| width | depth-1 null | depth 2 | excess | t |
|---|---|---|---|---|
| 32 | +0.023 | +0.026 | +0.003 | 0.2 |
| 64 | +0.041 | +0.055 | +0.015 | 0.9 |
| 128 | +0.067 | +0.103 | +0.036 | 1.7 |
| **256** | +0.085 | **+0.241** | **+0.155** | **5.5** |

(width 256 is 3 depth-1 seeds against 3 depth-2 seeds; the other widths are
3 against 3 as well.)

### The circuit, and why it is not the textbook one

Located by ablation (`tf_induction_circuit.py`,
`tf_vanilla_d2_w256_b8192_s0_induction_circuit.json`):

| intervention | induction score | KL cost |
|---|---|---|
| none | 0.0841 ± 0.0065 | 0 |
| drop **layer-0 head 1** | **0.0083 ± 0.0051** | 0.186 |
| drop layer-1 head 15 | 0.0353 ± 0.0064 | 0.016 |
| drop both | −0.0025 ± 0.0035 | 0.189 |
| delete layer-0 head 1 **from layer 1's Q/K/V read** | **0.0841** | — |
| delete layer-0 head 1 **from MLP-1's input** | 0.0841 | — |
| delete layer-0 head 1 **from MLP-0's input** | **0.0083** | — |
| control: delete a *different* layer-0 head from layer 1's read | 0.0841 | — |

Layer-0 head 1 is one of the two heads with the most distance-1 attention mass
(11.0% and 11.9%, against 0.8–8% for the other fourteen), and layer-1 head 15
has the most in its layer (10.8%). So the *participants* are the ones the
standard story names. **The wiring is not.** Deleting head 1's write from what
layer 1's queries and keys read changes the induction score by 0.0000; deleting
it from what the layer-0 **MLP** squares reproduces the entire effect. The
previous-token signal reaches layer-1 attention **through the MLP**, which is
exactly what FINDING 8's composition budget predicts, since layer 1's read is
99.9% MLP-0's write and 0.4% layer-0 attention.

### Replicated on three seeds, including the route decomposition

`tf_w256_seeds_chain.sh` trained depth-2 width-256 seeds 1 and 2 (and depth-1
width-256 seeds 1 and 2 for the matched null). Everything holds:

| seed | induction | natural-text swap | the head that carries it | its distance-1 share (rank in layer 0) |
|---|---|---|---|---|
| 0 | +0.0841 | +0.244 | layer-0 head 1 | 0.110 (2nd of 16) |
| 1 | +0.0965 | +0.236 | layer-0 head 6 | 0.114 (1st of 16) |
| 2 | +0.1007 | +0.242 | layer-0 head 5 | 0.119 (2nd of 16) |

and the route decomposition is the same in all three — deleting the head's
write from layer 1's read leaves the score at 0.0841 / 0.0965 / 0.1007
(unchanged to 4 decimals), deleting it from MLP-0's input gives 0.0083 /
0.0131 / −0.0318 (the whole effect, and at seed 2 an overshoot past zero).
The head index is arbitrary across seeds; what replicates is that it is one of
the two heads with the most distance-1 attention mass, and that its route is
the MLP.

**Selection-effect control:** the heads were chosen on probe seeds 0–4, so the
entire decomposition was re-scored on **disjoint probe seeds 100–104** and
reproduces to within 0.001 at every cell.

---

## 2026-08-08 — FINDING 10 (ADVERSARIAL REVIEW): the rung-4 composed table does not predict what its head causally does — FINDING 6 is corrected

The standing rule is "compose to the logits **and confirm causally**". FINDING 6
did the first half. Doing the second half breaks it.

For every head, the agreement between the rung-4 object
`p_h · (OV_h W_Uᵀ)` — the head's **direct** route to the logit — and the head's
actual causal effect `logits(full) − logits(drop h)` on held text:

| cell | direct-route Pearson (per head) | through-MLP Pearson |
|---|---|---|
| depth 1, w32 | 0.17–0.39 | 0.63–0.83 |
| depth 1, w64 | 0.03–0.42 | 0.69–0.91 |
| depth 1, w128 | 0.00–0.43 | 0.77–0.95 |
| depth 1, w256 | −0.01–0.19 | 0.87–0.98 |
| depth 2, layer 0 | **0.002–0.02** | 0.93–0.96 |
| depth 2, layer 1 | 0.51–0.70 | 0.94–0.98 |

This is FINDING 2 biting back: the direct route is dead, so an object built out
of the direct route describes nothing. The correct composition — propagating
the head's write through the MLPs, which is *exact* here because the MLP is
bilinear — tracks the causal effect at 0.63–0.98 with 92–95% sign agreement.

**What that costs FINDING 6.** Its headline was "the heads are not copy heads:
the median rank of the attended token among the tokens it boosts is ≈5600 of
8192, i.e. attending to a token pushes its own logit *down*". Re-derived
causally — build the two-token context `[u, t]`, drop the head, and rank the
attended token `u` by how much the head's presence pushes it:

| cell | causal median rank of the attended token (of 8192), per head |
|---|---|
| depth 1, w32 | 1003, 1902 |
| depth 1, w64 | 286, 2834, 2867, 3310 |
| depth 1, w128 | 296, 694, 2190, 2316, 2563, 3144, 3689, 3752 |
| depth 1, w256 | 425, 508, 795, 2234 … 4880 |
| depth 2, w64 | layer 0: 3526, 3582, 3670, 5231; layer 1: **572**, 1084, 2255, 5222 |

**Retraction:** "≈5600 of 8192, pushed down" is a statement about the direct
composed table, not about the heads. Causally the median is 286–4880, several
heads put the attended token in the top 4–6% of the vocabulary, and no head is
anywhere near the "pushes its own token down" description. The *weaker* claim
survives: no head is a copy head in the strict sense (rank 0), the effect is
diffuse, and identity pairs are not specially favoured.

Everything else in the reviewer round is in `tf_reviewer_round_1_depth2.json`.

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
