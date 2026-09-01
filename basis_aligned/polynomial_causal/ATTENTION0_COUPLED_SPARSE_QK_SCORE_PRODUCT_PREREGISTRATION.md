# Preregistration — coupled sparse query/key score-product generator (rung 430)

Date: 2026-09-01 20:31 UTC  
Claim level: held-out composition, physical-generator, and semantic-candidate screen; not adoption

## Why this is a different experiment

Rung 426 learned one sparse code by reconstructing the concatenated query and key factor vectors. It established
that cross-head same-token structure is real and efficient, but factor reconstruction was its training objective.
Its atom-derangement predicate missed one random-pair score clause even though routed output and cross-entropy were
strongly damaged. This leaves open the second experiment requested in explanation section 14.3: learn query roles,
key roles, and their composition through the score that attention actually computes.

This rung is not a post-result atom-count sweep. It changes the object and objective. It separates query and key
codes at the same total stored price, and it trains them on exact bilinear scores and the product of the two score
branches. It links a sparse vocabulary to real computation by requiring a no-native-QK execution path, full
attention output, downstream consumers, and cross-entropy. It links candidate semantics to stable query-atom ×
key-atom contributions rather than visual inspection of decoder rows.

## Exact object and split

For token `t`, head `h`, and score branch `b`, the exact folded factors are

`q_b(t,h), k_b(t,h) in R^128`.

Concatenate all 18 query entries into `Q(t) in R^2304` and all 18 key entries into `K(t) in R^2304`. Learn separate
signed top-k codes and decoders

`Q_hat(t) = b_q + z_q(t) D_q`,  `K_hat(t) = b_k + z_k(t) D_k`,

with 512 query atoms and 512 key atoms. Each reconstructed 128-vector is root-mean-square normalized before use.
For relative offset `delta`,

`s_hat_(h,b)(x,y,delta) = q_hat_b(x,h)^T R_delta k_hat_b(y,h) / 128`,

and the complete attention weight is

`P_hat_h(x,y,delta) = s_hat_(h,1)(x,y,delta) s_hat_(h,2)(x,y,delta)`.

Token IDs with `t mod 5 != 4` are FIT and `t mod 5 = 4` are SELECT, exactly as rung 426. Encoders, decoders, and all
fine-tuning use only FIT token IDs and FIT random token pairs. SELECT token types, SELECT random pairs, and the 96
frozen SELECT documents remain unseen until scoring. FINAL is unopened.

## Frozen architecture and literal price

All arms have 512 query atoms and 512 key atoms. At the saving point, query and key each store 27 signed nonzeros
per token; at the equal-code diagnostic they each store 36.

- two FP16 decoders: `2 × 512 × 2304 × 2 = 4,718,592 bytes`;
- two FP16 biases: `2 × 2304 × 2 = 9,216 bytes`;
- k27+k27 FP16 coefficients and uint16 indices: `50,257 × 54 × 4 = 10,855,512 bytes`;
- total QK54: **15,583,320 bytes**;
- k36+k36 total QK72: **19,201,824 bytes**.

These exactly equal rung 426 G54 and G72, respectively. Encoders are training machinery and are absent from the
deployed bundle. The four native layer0 Q/K/Q2/K2 maps are absent from the candidate execution path. Native V/O and
all later modules remain.

## Frozen arms and objectives

All learned arms begin from the same factor-only warm start, optimizer family, batches, and step counts.

1. **SQ54 — separate factor-only baseline.** Query k27 and key k27 codes minimize balanced factor squared error.
2. **SC54 — score-coupled arm.** Warm-start from SQ54 and optimize balanced first- and second-branch score error on
   FIT token pairs and rotary offsets, plus a frozen-weight factor anchor.
3. **CP54 — complete-product arm.** Warm-start independently from the same SQ54 checkpoint and optimize both branch
   scores and their multiplicative product, plus the same factor anchor. This is the primary candidate.
4. **CP72 — equal-code diagnostic.** Decode the learned CP family at query k36 and key k36. It is fixed before
   SELECT and cannot be substituted for CP54 after the result.
5. **PP54 — pair-label-permuted control.** Repeat CP54 fine-tuning from the same warm start and with identical steps,
   but independently permute the key token used to define each FIT score/product target while retaining the real
   query/key factor anchor. It preserves target marginals and optimization capacity while destroying the real
   query-role × key-role relation.
6. **WH54 — wrong-head/branch contraction control.** At evaluation only, retain CP54's reconstructed branch scores
   but pair branch-1 head `h` with branch-2 head `h+4 mod 9` before value routing. It changes no factors, modes, or
   stored price.
7. **CP54-R — restart diagnostic.** Repeat the SQ54 warm start and CP54 fine-tuning with the registered alternate
   seed. It is used only for atom and atom-pair stability, not candidate selection.

Warm start: 1,200 factor steps. Fine-tuning: 1,000 steps. Each fine-tuning batch contains 128 independently sampled
FIT query/key token pairs and a uniformly sampled offset from `{1,2,4,8,16,32,64,128}`. The score-coupled loss is
the mean of the two branch relative squared errors. The product-coupled loss is their mean plus the complete-product
relative squared error. Every fine-tuned arm adds `0.25` times the balanced factor loss. These weights, steps, and
offsets are frozen before execution.

## Binding measurements

1. Balanced SELECT query-factor and key-factor fractions of variance unexplained.
2. SELECT random-pair branch-score and complete-product relative squared errors at every registered offset.
3. Frozen SELECT-document full attention0-write error, immediate MLP0 and all attention1 Q/K/Q2/K2/value consumer
   errors, and document-mean cross-entropy damage. Damage is cross-entropy added above native; lower is better.
4. No-native-QK replay after zeroing all four layer0 Q/K maps.
5. Active atom-pair concentration. Treat each bias as one constant atom. For each SELECT pair and entry, decompose
   the reconstructed score into at most `28 × 28` query-atom × key-atom contributions. Report the relative score
   error after retaining the 32, 64, and 128 largest-magnitude contributions and the fraction of absolute
   contribution mass they contain.
6. Restart stability. Hungarian-match query atoms and key atoms separately using absolute decoder cosine across
   CP54 and CP54-R. After matching, compare the 256 atom pairs with greatest mean absolute contribution on SELECT;
   report decoder median cosine and atom-pair Jaccard overlap.

Rung 426 G54/G72 and rung 424's continuous block are immutable report anchors. The sparse candidate is physically
simpler only relative to matched stored baselines; it is closer to the real computation only to the extent that
full write, consumer, CE, and no-native execution pass. It is semantically candidate-bearing only to the extent that
atom-pair concentration and restart stability pass.

## Frozen predictions

**A — valid physical instrument.** Exact fold is at most `1e-10`; token splits are exact and disjoint; every learned
loss decreases by at least 20%; stored dtypes, shapes, and byte bills are exact; no-native-QK replay relative squared
error is at most `1e-12`; and the deterministic SQ54 rerun reproduces its own pre-SELECT checkpoint hash within both
same-seed evaluation paths.

**B — product coupling learns the real relation.** CP54 SELECT complete-product error is at most `0.80 ×` SQ54 and
at most `0.90 ×` SC54; its mean branch error is at most `1.10 ×` SC54; and its complete-product error is at most
`0.85 ×` PP54. CP54 full-write error is at most `0.90 ×` SQ54 and its CE damage is no more than SQ54 `+0.002 nat`.

**C — coupled sparsity improves the rung-426 generator at matched price.** CP54 complete-product error is at most
`0.90 ×` rung-426 G54's `0.6437466217802437`, and CP72 error is at most `0.90 ×` rung-426 G72's
`0.483746191580884`. CP54 CE damage is at most rung-426 G54 `+0.002 nat`; CP72 CE damage is at most rung-426 G72
`+0.002 nat`; and WH54 is at least 25% worse than CP54 on full-write error and at least `0.01 nat` worse on CE.

**D — stable sparse compositions exist.** The top-64 active atom-pair contributions reconstruct SELECT branch
scores with relative squared error at most `0.30`; CP54 versus CP54-R has median matched decoder cosine at least
`0.50` on both query and key sides; and the matched top-256 atom-pair Jaccard overlap is at least `0.20`.

## Strong null and routing

The strong null fires if A fails; if CP54 improves complete-product error by less than 5% over SQ54; if CP54 is
within 2% of PP54; if CP72 product error is at least `1.25 ×` rung-426 G72 or its CE damage exceeds rung-426 G72 by
more than `0.01 nat`; or if WH54 is within 2% of CP54 on full-write error or CE damage.

- A/B/C/D pass without the null: sparse query roles, key roles, and stable compositions become the primary
  interpretation and physical-generator candidate. Next test fresh/OOD documents and the 62 downstream behaviors,
  then compare directly with a physically generated rung-424 continuous block.
- B/C pass but D fails: the sparse generator is useful, but its atoms are a non-unique coordinate system. Keep the
  executable savings while assigning semantics only to stable subspaces or downstream response classes.
- D passes but B/C fail: readable atom pairs do not explain the real attention computation efficiently; retain them
  only as descriptive probes.
- Strong null: close this sparse composition family at the frozen budget and prioritize the direct continuous
  composite generator. Do not tune atom count, k, loss weights, thresholds, or controls after SELECT.

Even a full pass is not whole-model adoption. Fresh and shifted text, the 62-behavior census, frontier composition,
and signed causal interventions remain mandatory.
