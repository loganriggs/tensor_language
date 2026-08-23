# Readout (L15-17) — near-linear read, the merge, and the frequency calibrator

**One line:** attention inert; MLPs are near-linear reads that rotate the stream into the token
basis; block 17 holds the rank-1 frequency calibration; the grammar/content merge into logits is
additive-linear.

## Established facts
- **Attention 15-17: inert** (per-head costs ≤0.006, §1083; §1047).
- **MLPs near-linear reads:** mlp17 0.85, mlp16 0.78/0.94-bilinear (§1046); readout does the bulk
  of output formation (logit-lens CE 5.8@L15→3.26; §944) via ~95% linear rotation (§945);
  readout reads class 13× + position 6× (§851). mlp16 bilinear loss-rank ~64 (§1040).
- **mlp16 structure (§1090, held-out):** output is 82% token-variance (structural fact) but the
  4%-variance dev part carries ~87% of its CE value; dev-only substitution (context without the
  token baseline) costs 0.795 vs mean-abl 0.138 — the big token component is REDUNDANT for CE
  (stream already carries identity, §690) yet load-bearing as a baseline if you keep the dev part.
  §1084's "readout = token calibration lookup" was WITHDRAWN then reinstated only as this refined
  structural claim (§1088→§1090).
- **The merge (§1082/§1086):** final logits ≈ additive-linear in the content component (cosine
  0.77 between logit-delta of content removal and −W_lm·c). Head/tail division is about magnitude:
  content removal hits rare-target log-prob 16× an unscaled random control — but only 1.7× a
  NORM-MATCHED control; argmax fragility is magnitude-generic (norm-matched random flips 41% vs
  content 38%; §1086 retracted §1082's "content picks the winner"). Final-residual deviation is
  much lower-rank than mid-stack content (top-64 = 76% var; §1082).
- **Block 17 = dominant frequency calibrator (§624-662):** rank-1 w_freq (removal kills it 103%,
  random 0-2%), ~40% aligned with unembedding log-freq axis; calibration components in 5 layers,
  block 17 dominating 5-10×; w_freq lives 88% in the massive dims (§676).
- **Errors/calibration (§972-980):** top-1 class-correct 2/3; hedges to function words when
  unsure (calibrated deferral, ECE 0.009); content misses land in-topic 40% vs 15% random.

## Benchmark status
Readout band ~0.56-0.9 (§906/§940; mlp17 0.85, mlp16 0.78 §1046). The merge is understood
(additive-linear); block-17 calibration fully isolated (the model's ONE clean rank-1 knob).

## Gotchas
- Variance-rank ≠ functional rank (mlp17 §615/§660); low-variance tail carries loss.
- At the final residual, most content-specificity is already merged — deletions there act like
  generic damage (§1086); test content claims mid-stack, not at the readout.

**CEILING CERTIFIED (§1131-1133):** held-out — mlp17 0.842 linear / 0.821 top-256 own-neurons;
mlp16 0.813 / 0.807. Two instrument families converge at both MLPs: ~0.81-0.84 capturable, the
final ~0.2 neither sparse nor linear (§660 law certified band-wide). Token augmentation redundant
(−0.009). mlp15 0.40 at near-zero stakes. Do not chase 0.9 here with sparsity/linearity instruments. THIRD FAMILY (§1139): fitted rank-64 quadratic gains +0.01 (0.822/0.857; train R² rises to 0.975 but held-out doesn't move) — the ceiling is a THREE-FAMILY LAW; module closed.

## Open
- Nothing pressing beyond the standing §1069 middle-attention remainder upstream.
