# Task-circuit decomposition: closing-bracket / closing-quote behavior in bilin18

Independent task-scoped decomposition, run as a sanity check against the full-model
decomposition (qk_circuit_atlas.json). Model: `bilin18`
(gpt2-bilinear-sqrd-attn-18l-9h-1152embd; 18 layers x 9 heads, no softmax,
pattern = (q1.k1)(q2.k2)/d^2 causal unnormalized, bilinear MLPs).

Scripts: `s1_stimuli.py`, `s2_patching.py`, `s3_das.py`, `s4_weight_reduction.py`.
Results: `stimuli.json`, `patching.json`, `das.json`, `weight_reduction.json`.

## 1. Behavior verification (stimuli.json)

40 clean/corrupted pairs per task; clean uses a space-separated opener
(`"The dogs ( which was near..."`, `'She said " the dogs...'`) so the corrupted token
sequence is EXACTLY the clean sequence minus the single opener token (357 `G(` / 366 `G"`)
- verified by assertion on every pair. Content words, prefix (opener position), and clause
length (opener-to-prediction distance, roughly 3-12 tokens) all varied.
Split: 30 analysis / 10 held-out per task.

Boost = logprob(closer) at final position, clean minus corrupted. Closer tokens chosen
empirically: `)` (id 8) and `"` (id 1) beat the space-prefixed variants by ~2.2 and ~3.6 nats.

| task | mean boost | min | frac > 0 | analysis-30 | held-out-10 |
|---|---|---|---|---|---|
| paren | **7.03** | 4.56 | 1.00 | 6.95 | 7.27 |
| quote | **5.01** | 3.33 | 1.00 | 4.80 | 5.65 |

The original attached-opener style (`"(which"`, matching qk_algo_probe.py) gives 6.92 /
6.51 on fresh samples - consistent with the probe's 5.44 / 6.40. Behavior confirmed and
robust to the space-separated reformulation.

## 2. Single-component patching importance (patching.json)

Corrupted run; ONE component's activation (162 head outputs `yh4[:,:,h,:]`, 18 MLP
outputs) replaced at all positions by its clean-run value, deletion-mapped
(corr t <- clean t for t < opener_pos, else clean t+1). Metric: recovered fraction of the
clean-vs-corrupted closer-token logit difference at the final position, mean over the 30
analysis pairs per task.

**Top-10 (mean of paren and quote):**

| rank | component | mean | paren | quote |
|---|---|---|---|---|
| 1 | head L13 H8 | **0.561** | 0.600 | 0.521 |
| 2 | MLP L6 | 0.123 | 0.152 | 0.094 |
| 3 | head L7 H2 | 0.114 | 0.155 | 0.073 |
| 4 | head L13 H3 | 0.100 | -0.000 | 0.200 |
| 5 | head L4 H0 | 0.094 | 0.124 | 0.064 |
| 6 | head L5 H7 | 0.092 | 0.098 | 0.086 |
| 7 | head L11 H5 | 0.091 | 0.083 | 0.098 |
| 8 | head L8 H8 | 0.074 | 0.047 | 0.102 |
| 9 | head L9 H7 | 0.070 | 0.076 | 0.064 |
| 10 | head L10 H5 | 0.063 | 0.079 | 0.047 |

A single late head, **layer 13 head 8, carries the majority of the effect for BOTH
openers** (0.60 paren / 0.52 quote). Head L13 H3 is quote-specific (0.20 vs 0.00).
The rest is a long tail of mid-stack heads shared across both tasks.

**Cumulative recovery, top-k patched jointly:**

| k | 1 | 2 | 3 | 5 | 10 | 20 |
|---|---|---|---|---|---|---|
| paren | 0.60 | 0.70 | 0.77 | 0.82 | 0.90 | 0.92 |
| quote | 0.52 | 0.58 | 0.61 | 0.77 | 0.90 | 0.97 |

Ten components recover 90% of the behavior; the circuit is sparse but not a single edge.

**Comparison with the full-model atlas (punct task, mean-ablation knockout on FineWeb):**
Spearman rank correlation over all 180 components = **0.27** (p = 2e-4); Pearson = 0.04;
**top-10 overlap = 0/10**.

- Disagreement (headline): the atlas's punct top-6 is dominated by early MLPs
  m1, m0, m2, m3, m4 (m1 alone: +3.30 CE knockout) plus head L0 H3 - all of which are
  near-irrelevant here (my ranks 25-80, recovery 0.003-0.03). Those components serve
  *general punctuation prediction* (base-rate/tokenization machinery), not the
  differential open-then-close computation this task isolates.
- Agreement (real but buried): my top head L13 H8 IS in the atlas's top decile (punct
  rank 11 of 180), and 8 of my top-10 sit in the atlas's top ~25 (m6 rank 15, L7 H2
  rank 20, L4 H0 rank 14, L11 H5 rank 13, L10 H5 rank 10...). The atlas sees the circuit
  - it just cannot separate it from generic punctuation infrastructure.
- Genuine disagreements: head L13 H3 (my rank 4, quote-specific) is atlas rank 145 with
  *negative* knockout importance, and head L8 H8 (my rank 8) is atlas rank 98 -
  redundancy makes them invisible to single-knockout but visible to patching.

Conclusion for the sanity check: the full-model punct importances are NOT a valid proxy
for the bracket-matching circuit (rank corr 0.27, zero top-10 overlap), but the task
circuit is contained within the atlas's broader punct set; the direction of error is
"atlas over-credits universal early MLPs, under-credits redundant late heads".

## 3. DAS-lite at layer 13 (das.json)

Learned r-dimensional orthonormal subspace Q (QR of a trainable 1152 x r matrix) of the
residual stream entering layer 13; interchange x <- x + QQ^T(x_clean - x_corr) at the
opener position and/or the final position; differentiable through layers 13-17; trained
on the 60 analysis pairs (both tasks jointly, loss = (1 - recovery)^2), evaluated on the
20 held-out pairs. Ceiling = full 1152-dim vector replacement. Flip = fraction of held-out
pairs with recovery > 0.5. Random control = random orthonormal subspace, same r (5 seeds).

| intervention | r | train recov | held recov | held paren / quote | flip | random ctrl |
|---|---|---|---|---|---|---|
| opener pos (ceiling) | 1152 | | -0.05 | | | |
| opener pos | 1 | 0.13 | 0.12 | 0.10 / 0.13 | 0.00 | 0.000 |
| opener pos | 4 | 0.30 | 0.25 | 0.24 / 0.25 | 0.00 | 0.000 |
| opener pos | 16 | 0.38 | 0.28 | 0.25 / 0.32 | 0.00 | -0.001 |
| final pos (ceiling) | 1152 | | 0.52 | | | |
| final pos | 1 | 0.61 | **0.56** | 0.61 / 0.52 | 0.75 | 0.001 |
| final pos | 4 | 0.95 | **0.87** | 0.93 / 0.81 | 1.00 | 0.001 |
| final pos | 16 | 0.99 | 0.89 | 0.96 / 0.82 | 1.00 | 0.008 |
| both (ceiling) | 1152 | | 0.46 | | | |
| both | 1 | 0.62 | 0.57 | 0.61 / 0.53 | 0.80 | 0.001 |
| both | 4 | 0.98 | **0.91** | 0.95 / 0.87 | 1.00 | 0.001 |
| both | 16 | 1.00 | 0.93 | 0.98 / 0.88 | 1.00 | 0.007 |

- **The task information at layer 13 is low-dimensional: 1 direction recovers 56-57%
  (matching the single-head patch), 4 directions recover 87-91% held-out**, with 100%
  flip rate and a shared subspace across parens and quotes. Random subspaces do nothing.
- Position: the effect lives at the **final position** by layer 13. Injecting the clean
  opener-position residual at layer 13 recovers essentially nothing even with full
  replacement (ceiling -0.05) - the "an opener is pending" signal has already been moved
  to/accumulated at the query-side positions by layers < 13; re-inserting the opener
  token's residual that late is useless. (Learned opener-position subspaces reach 0.25-0.28
  by exploiting off-distribution directions - see caveat.)
- Caveat, stated honestly: trained subspaces at r >= 4 EXCEED the full-vector replacement
  ceiling (0.87-0.93 vs 0.52). DAS optimization finds directions whose amplification
  downstream overshoots what a natural clean-state substitution produces; the r = 1
  number (0.56, = the head-patch effect) is the conservative dimensionality claim, and
  the r = 4 numbers should be read as "a 4-dim subspace suffices to steer the behavior",
  not "the model stores exactly 4 dims of it".

## 4. Ethan's weight-reduction method (weight_reduction.json)

Matrix selection within head L13 H8: zeroing any of c_q/c_k/c_q2/c_k2/c_proj (head slice)
gives the identical floor (boost 6.46 -> 5.56 held-out; the multiplicative pattern dies if
any factor dies). Zeroing **c_v does nothing** (boost 6.49): the head's value path is the
*block-0 value stream* (v is mixed with layer-0's v via lamb) - the head only computes
WHERE to attend; WHAT it copies comes from layer 0. Rank-4 data-free truncation hurt c_q2
most (6.07), so W = **c_q2 head slice (64 x 1152)** was picked.

Important dynamic-range fact: killing the whole head costs only **0.90 nats of the 6.46-nat
boost** (held-out). Knockout is much weaker than patching (0.56 recovery) - the behavior is
heavily redundant in the intact model, consistent with the long tail in step 2. Task metric
below is therefore floor-normalized: frac = (boost(r) - 5.56) / (6.46 - 5.56).

X = actual inputs to W (rms-normed layer-13 residual) on 300 fresh task prompts, n = 3668
positions; Y = W X^T; W'_r = Y_r @ pinv(X^T, rcond = 1e-4). Control: plain SVD of W at the
same r. FineWeb CE on rows 500-519, len 128: base 3.4281 (W = 0 floor: 3.4321).

| r | data-cond floor-frac | data-cond CE damage | data-free floor-frac | data-free CE damage |
|---|---|---|---|---|
| 1 | 0.74 | +0.0011 | 0.69 | +0.0025 |
| 2 | **1.04** | +0.0008 | 0.49 | +0.0022 |
| 4 | 1.04 | +0.0007 | 0.57 | +0.0025 |
| 8 | 1.06 | +0.0005 | 0.83 | +0.0021 |
| 16 | 1.07 | +0.0007 | **0.98** | +0.0013 |
| 32 | 1.03 | +0.0003 | 0.99 | +0.0006 |
| 64 | 1.01 | +0.0002 | 1.01 | +0.0002 |

- **Minimal rank for >= 90% of the head's contribution: r = 2 data-conditioned vs r = 16
  data-free - an 8x rank saving.** The data-conditioned rank-2 W even slightly overshoots
  (1.04, +0.03 nats above the intact model). Data-free truncation is non-monotonic
  (r = 2 WORSE than r = 1: 0.49 vs 0.69 - the second principal component of W actively
  interferes on-distribution).
- General damage is negligible throughout (max +0.0025 CE ~ 0.07%); even W = 0 costs only
  +0.004. On the raw (un-normalized) boost metric every r >= 1 keeps >= 93%, which is why
  floor normalization is the honest metric here.

## 5. Summary and surprises

1. Behavior real and strong: +7.0 nats (paren) / +5.0 nats (quote), 100% of 80 pairs.
2. Circuit: one late head (L13 H8) carries ~56% for both opener types; top-10 components
   (mostly mid-stack heads + MLP L6, shared across tasks) recover 90%. One quote-specific
   head (L13 H3).
3. Sanity check vs full-model atlas: punct importances rank-correlate only 0.27 with the
   task circuit, zero top-10 overlap - task-scoped patching and whole-distribution
   knockout answer different questions. But the task circuit is a subset of the atlas's
   top-25, so the decompositions are consistent, not contradictory.
4. The steering subspace at layer 13 is tiny (1 dim ~ 56%, 4 dims ~ 90%) and lives at the
   final position, not the opener position, by that depth.
5. Ethan's data-conditioned reduction works as advertised: rank 2 vs rank 16 data-free for
   the same 90% task retention, with no measurable FineWeb damage.
6. Surprises: (a) the head's own c_v is irrelevant - it routes the block-0 value stream
   (lamb mixing), i.e. QK decides where, layer 0 decides what; (b) full-vector clean
   substitution at the opener position at layer 13 recovers NOTHING (the pending-opener
   signal has already migrated to the query side); (c) trained DAS subspaces overshoot the
   natural replacement ceiling - a caveat on interpreting learned-subspace recovery as
   stored dimensionality; (d) knockout vs patching asymmetry is large (0.9 nats vs 56% of
   a ~7-nat gap) - the intact model has heavy backup for this behavior.

Failures / limitations: none of the runs failed; limitations are the overshoot caveat in
step 3, the small dynamic range of the knockout-based task metric in step 4 (0.9 nats,
addressed by floor normalization), and stimulus diversity (templated English, one opener
per sequence, opener depth 1).
