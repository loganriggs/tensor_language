# Results — memorization-in-bilinear-layers post experiments

(Each part appends its own section; see README.md for coordination rules.)

## Part 1 (3-class Dog/Cat/Catfish toy; figures F1-F8, claim check C1)

Reproduce everything with `python part1.py --stage all` (CPU, minutes). Conventions exactly as the
handoff: y = D((Lx)\*(Rx)), folded tensor T[i,j,c] = sum_h D[c,h] sym(L[h,i] R[h,j]), boolean
diagonal-as-linear, diagonals outlined in every heatmap, 5 seeds per trained result.

### 1a. Hand-coded clean model (F1) — all rank claims verified

- Each per-class matrix B_c is a single symmetric off-diagonal pair (value 0.5 at the class's
  feature pair, everything else exactly 0). Acceptance PASS for all three classes.
- Symmetric eigendecomposition: every B_c has eigenvalues exactly (+0.5, -0.5, 0), with
  eigenvectors (e_a + e_b)/sqrt(2) and (e_a - e_b)/sqrt(2) on the class's feature pair — the
  claimed +-1/2 structure, PASS.
- Asymmetric slices M_c = sum_h D[c,h] l_h r_h^T (computed without symmetrizing, per the
  Pitfalls): singular values (1, 0, 0) for every class — asymmetric rank exactly 1, PASS.
- Fold consistency: max |D((Lx)\*(Rx)) - x^T B x| over all 8 boolean inputs = 0 (exact).

### 1b. Trained overcomplete model H=8 (F2, F3) — pre-registered prediction CONFIRMED, both channels

Registration: predictions/part1_1b_prediction.md, committed in 07021c31e (2026-08-10T20:30:44Z),
before any training ran; part1.py git-gates the training stage on that commit.

Dataset note (degenerate-dataset warning): trained on the 3 single-class keys plus the all-zeros
"none" input with a uniform soft target, softmax cross-entropy, full-batch AdamW, 6000 steps.
Because the pure bilinear layer has no bias, the all-zeros input has exactly zero logits (uniform
softmax) for every parameter setting, so that row contributes a constant loss and zero gradient —
training v1 is effectively the 3 positive examples. This was NOT degenerate in practice: all
15 runs (5 seeds x 3 weight decays) reach 100% key accuracy, converge to consistent structure
(cross-seed folded-tensor cosine: mean 0.91 / min 0.86 at weight decay 0 and 1e-3; mean 0.89 /
min 0.78 at 1e-2), and the loss is bounded away from zero by the all-zeros row. The loss was not
changed. Generalization on the other inputs (seed 0, weight decay 1e-3), inputs ordered
000,100,010,110,001,101,011,111: Dog, Cat, Dog, Dog, Catfish, Cat, Catfish, Cat — i.e. the
three keys are right; single-feature inputs and the all-ones input fall to whichever class is
least penalized (the all-ones input is Cat at weight decay 0/1e-3 but Dog at 1e-2 seed 0).

Which channel did SGD use? BOTH predicted channels, on every seed and every weight decay:

- Positive off-diagonal on the class's feature pair (weight decay 1e-3, mean over 5 seeds,
  range in parentheses): Dog B[furry,happy] = +3.36 (+1.99..+4.39); Cat B[furry,whiskers] =
  +3.50 (+2.96..+3.79); Catfish B[whiskers,happy] = +3.23 (+2.16..+3.98). No sign flips.
- Negative DIAGONAL on the class's absent feature — the single largest negative term (linear
  "not-X" logic via the boolean diagonal): Dog B[whiskers,whiskers] = -3.14 (-3.73..-2.08);
  Cat B[happy,happy] = -4.08 (-5.31..-2.76); Catfish B[furry,furry] = -3.06 (-3.96..-2.12).
  No sign flips.
- Negative off-diagonals coupling the absent feature to the pair, smaller but consistent:
  between -1.29 and -2.41 in the mean, no sign flips.

Weight decay {0, 1e-3, 1e-2} does NOT change the channel — same signature at all three values,
magnitudes shrink slightly at 1e-2. Not predicted (see surprises.md): positive diagonals on the
class's OWN two features also appear (about +0.7 to +2.2 in the mean) but these are the
least stable entries — sign flips across seeds in 7 of 9 class/feature cells at weight decay 1e-3.
F3 shows the diagonal bar chart: the anti-diagonal (absent feature) dwarfs the own-feature
diagonals in every class.

### 1c. Undercomplete H=2 (F4, F5)

- H=2 with 3 classes trains to 100% key accuracy on all 5 seeds (no fallback to the
  6-feature/6-class variant needed), but the solutions genuinely differ across seeds:
  cross-seed folded-tensor cosine mean 0.69, min 0.50 (vs 0.87 minimum for H=8). F4 therefore
  shows seed 0, and the per-seed tensors are in figures/F4_B_H2.npy.
- The interaction matrices are visibly "more mixed up": large negative cross-class entries
  (e.g. seed 0 Cat B[happy,happy] = -7.04, Catfish B[furry,furry] = -6.27, off-diagonal
  magnitudes 4.9-5.9 in cells that are 0 in the clean model).
- F5 stored-key Gram: keys defined as the model's hidden representation of each class input,
  h_c = (L z_c)\*(R z_c), cosine-normalized. (The raw input Gram Z^T Z is regime-independent —
  diagonal 2, off-diagonal 1 — so the regime contrast lives in the stored representation.)
  Overcomplete H=8: |off-diagonal| mean 0.18, max 0.49 (near diagonal). Undercomplete H=2:
  |off-diagonal| mean 0.48, max 0.96 (heavy off-diagonal mass). Acceptance met.

### C1 (F6) — the centerpiece: predicted vs measured collateral, r = 0.94

Ordering obeyed: predictions/part1_c1_predictions.{json,md} (all 60 predicted values) committed
in 710d50f81 at 2026-08-10T20:33:10Z; the measurement stage is git-gated on that commit and ran
afterwards. Predicted collateral for edit-class c and victim v = |cos(h_c, h_v)|, the stored-key
Gram off-diagonal. Edit = KKT-edit class c to uniform: the minimal-norm rank-1 update to D along
h_c that makes the logits at z_c exactly equal (weights + the key only, no corpus). Measured
collateral = victim's margin drop on its own key.

- All 60 points (3 edits x 2 victims x 5 seeds x 2 regimes): Pearson r = 0.943 (p = 2.4e-29),
  Spearman rho = 0.931 (p = 5.6e-27). Acceptance (strong rank correlation) PASSES.
- Within-regime: overcomplete r = 0.871 (Spearman 0.817); undercomplete r = 0.931 (Spearman 0.917).
- Regime contrast: undercomplete collateral is ~3.7x larger (mean margin drop 6.02 vs 1.63;
  max 13.25 vs 6.89) and contains the only victim accuracy flip (1/30 vs 0/30). Margins are
  large enough (typically 8-15 nats) that most collateral does not reach an argmax flip in this
  tiny toy; margin drop is the sensitive measure.

### 1d. Edits on the toy (F7, F8)

Pull-out (exact-match verification):
- Hand-coded model: slice T[:,:,Dog], rebuild a standalone detector from the symmetric
  eigendecomposition (H=2); max |mismatch| vs the full model's Dog logit over all 8 inputs =
  2.2e-16 — exact. CP of the asymmetric slice: singular values (1,0,0); the rank-1 L/R/D
  reconstruction also matches with error 0. Rank-1 asymmetric recovery PASS.
- Trained H=8 (weight decay 1e-3, seed 0): standalone detector H=3, max mismatch 6.4e-15 —
  exact. The trained asymmetric Dog slice has singular values (6.13, 2.45, 1.83): asymmetric
  rank 3, NOT rank 1 (the trained model spreads the fact across a rank-3 slice; see surprises.md).

Two-class sub-model (Dog + Cat):
- Minimal hidden dimension for an exact re-decomposition of the two symmetric slices: H = 2 for
  the clean model (relative squared Frobenius error 1e-273 at H=2 vs 0.5 at H=1) and H = 2 for
  the trained model (1.8e-13 at H=2 vs 0.26 at H=1). Two facts fit in two hidden units even
  though each slice alone is symmetric rank 2-3.
- Discriminator T[:,:,Dog] - T[:,:,Cat], clean: eigenvalues (+0.707, -0.707, 0) with
  eigenvectors (0.707, +-0.5, -+0.5) — i.e. furry combined with (happy - whiskers)/2: the shared
  feature drops toward zero weight in the eigenvalues' balance and the discrimination is carried
  by happy-vs-whiskers, paired with furry. Trained (mean of seeds): eigenvalues
  (+9.25, -7.86, -0.09) — effectively rank 2, same qualitative structure: top eigenvector
  (0.54, 0.80, -0.28) = "furry and happy, not whiskers", second (0.63, -0.16, 0.76) with
  negative eigenvalue = "furry and whiskers" penalized.

Path removal (extended features [furry, happy, whiskers, hands, dog-ears], classes Dog, Cat,
Catfish, Human; Dog has paths (furry,happy) and (furry,dog-ears); Human = (hands,dog-ears);
overcomplete H=12 and undercomplete H=3 both train 5/5 seeds to 100% on the 5 keys):
- PRIMARY edit (the surgical instrument): minimal-norm update to Dog's D-row with two linear
  constraints — the folded (furry,dog-ears)->Dog entry goes exactly to 0 and the
  (furry,happy)->Dog entry is exactly unchanged. Because only Dog's D-row changes, every other
  class's slice, including (hands,dog-ears)->Human, is untouched by construction.
- Overcomplete H=12: fully surgical — zero accuracy changes on all 5 keys on all 5 seeds; the
  targeted Dog(furry,dog-ears) key takes by far the largest margin hit (-10.8 mean), other keys
  move little (means -1.7 to -3.6, some seeds positive).
- Undercomplete H=3: same edit, larger collateral — margin swings on non-Dog keys are much
  bigger and wilder (Cat ranges -14.0 to +9.8 across seeds; Catfish mean -6.4), and 1/5 seeds
  flips the Dog(furry,dog-ears) key to Human. Surgical-overcomplete / collateral-undercomplete
  contrast confirmed at the margin level.
- Secondary edit (naive baseline): greedy whole-unit ablation (zero D[:,h]) until the path entry
  is gone. Catastrophic in BOTH regimes: it needs 5-8 of 12 units even when overcomplete,
  destroys the preserved entries ((furry,happy)->Dog +3.41 -> +0.35; Human's entry +3.64 -> +1.40),
  and flips 40-80% of Dog keys plus (undercomplete) 60-100% of other-class keys. Trained H=12
  networks do NOT store the (furry,dog-ears) path in dedicated units — unit-level surgery is the
  wrong knife even with room to spare.
- Honest caveat (also in surprises.md): zeroing the (furry,dog-ears) interaction entry does NOT
  make the Dog(furry,dog-ears) key stop classifying as Dog in most runs (9/10 across regimes with
  the primary edit) — the margin drops by ~11-12 but stays positive, because diagonal linear
  terms (furry, dog-ears diagonals and competitors' negative diagonals) carry a large share of
  path 2. The off-diagonal interaction is one ingredient of the path, not the whole path.

## Part 2: 100 random facts (n = 20-bit keys, 10 classes) — F9-F12

Everything from `part2.py` (seeded, re-runnable end to end; stage list in the docstring).
Numbers in `figures/part2_metrics.json`. All edits below use model weights + the fact-key
list only (C = sum_k z_k z_k^T over the 100 stored keys); no training corpus is accessed
anywhere in Part 2.

Setup and sizing. 100 distinct Bernoulli(1/2) 20-bit keys, classes uniform (counts
11,5,10,9,16,10,10,14,11,4). Key overlap is meaningful: mean inter-key |cos| 0.51; per-fact
max off-diagonal Gram overlap ranges 0.65-0.89. H sweep {10,20,30,40,50,60,80,100}:
SGD memorizes 100/100 already at H=10; the ALS construction reaches 100% argmax accuracy at
H=20 (but with target MSE 0.107) and exact interpolation from H=40 (MSE 5.7e-11).
Working point H* = 40, chosen as the smallest H with 100% memorization for BOTH methods AND
exact ALS interpolation (deviation from a pure accuracy criterion, stated here: with an
inexact construction, F9's "construction" would not be a well-defined reference).

Construction convention. D is FIXED at the tiled negative identity (exactly D = -I when
H = C; D = -[I|I|...|I] for H = mC) and never trained; L and R found by ALS with exact
convex per-class least-squares block solves, 60 iterations, best of 3 restarts. The KKT
machinery: the single-fact edit is rank-1 in the C^{-1}z* direction with coefficient
(y* - f(z*)) / (z*^T C^{-1} z*)^2 per class. The joint minimum-C-norm interpolant from the
zero tensor (S lambda = Y with S = Hat^2 elementwise, Hat = Z C^{-1} Z^T) is well conditioned
(cond S = 52, fit error 9.6e-15) and the cyclic single-fact edit provably reaches it
(51 passes to 1e-6; relative distance 1.7e-7). Add-a-101st-fact demo on the ALS
construction: new fact stored, all 100 old facts retained, mean |margin change| on old
facts 0.23 (max 2.58).

### F9 — SGD vs construction (acceptance PASSES; D = -I REFUTED)

Folded-tensor similarity of SGD (5 seeds) to the ALS construction, with BOTH required
baselines (mean [min, max] over 5):

| metric | SGD vs constr | random init | permuted-fact constr |
|---|---|---|---|
| Frobenius cos, class-centered (= Gaussian-input corr) | 0.119 [0.092, 0.154] | 0.001 [-0.020, 0.012] | -0.004 [-0.021, 0.028] |
| Frobenius cos, off-diagonal only | 0.133 [0.101, 0.172] | 0.001 [-0.023, 0.027] | -0.003 [-0.018, 0.036] |
| boolean-input logit corr (2^16 sample) | 0.134 [0.100, 0.169] | 0.008 [-0.021, 0.045] | -0.017 [-0.046, 0.006] |
| fact-key logit corr | 0.582 [0.504, 0.614] | -0.008 [-0.051, 0.045] | 0.000 [-0.067, 0.033] |

Clear separation from both baselines on every metric (no overlap of ranges) — the
acceptance criterion passes. Interpretive caution that the post should carry: the absolute
similarity is small, and re-running the ALS construction from different inits gives
construction-vs-construction similarity 0.07-0.13 — the same range as SGD-vs-construction.
SGD seeds agree with each other at 0.32 [0.25, 0.41]. So SGD lands in the same wide
solution family as the construction (far from both nulls), not on the construction itself;
SGD's own basin is tighter than the construction's.

D approx -I: REFUTED. With D trained freely under L1 (1e-3), per-hidden-unit dominance
(largest |D| entry / column L1 mass) is only 0.39-0.49 across seeds (a one-hot column would
be 1.0), and the dominant entries are negative only 40-55% of the time — sign flips across
seeds flagged (flag = True). SGD does not converge to per-unit single-class routing of
either sign. Weight statistics (F9 panel i): the construction's L is heavy-tailed (entries
to +-10) with compact R; SGD spreads magnitude evenly across L, R, D.

### F10 — extraction recovery vs Gram overlap

Blind extraction (top-|K_c| eigenvectors of each symmetrized class slice, matched to keys
or C^{-1}-keys at |cos| >= 0.8) FAILS: recovery 0-1% per seed, at every overlap bin, for the
ALS construction (0%), for the KKT interpolant (3%), and for SGD with H = 100 and H = 400
(0-1%; side-check seed 0). Mean best match score 0.52 = the typical inter-key overlap 0.51,
i.e. eigenvectors match keys no better than keys match each other. Facts are not stored as
eigen-separable rank-1 components; the slice is a compressed joint code (per-class key-frame
residual operator norms 11-30, comparable to the slice eigenvalues themselves).

Informed key-frame attribution (least-squares decomposition of the folded tensor onto the
dictionary {(C^{-1}z_k)(C^{-1}z_k)^T}, fact recovered if argmax_c lambda_ck = stored class):
SGD 44-51% per seed vs 10% chance; ALS construction 52%; pure KKT interpolant 96%. This is
where the recovery-vs-overlap relationship lives: binned by max off-diagonal Gram overlap,
recovery falls 0.61 -> 0.43 from the lowest-overlap bin (0.70) to the highest (0.86);
corr(recovered, overlap) = -0.14, corr(attribution margin, overlap) = -0.15 (pooled, 500
fact-seed points). Direction as predicted, but overlap explains little variance.

### F11 — unlearning 10 facts via the KKT edit (prediction registered pre-measurement)

Edit set [10, 27, 28, 35, 44, 55, 58, 67, 73, 88], target uniform, applied to all 5 SGD
folded tensors by cyclic single-fact KKT edits (converged, 14-15 passes). Predictions
(per-victim margin changes from the joint-KKT lambda through the squared hat-matrix
off-diagonals, plus a pure-Gram proxy sum_m Hat[j,m]^2) were committed in
`predictions/part2_f11_predictions.json` at commit 43bf4289c (2026-08-10T20:38:42+00:00),
BEFORE the measurement commit ebe594ab6; commit time = registration time.

- Forgetting: max deviation from uniform < 1e-9 on all 10 facts, all seeds (an argmax-based
  count reads 9-10/10 only because argmax over exactly-equal logits is tie-breaking noise).
- Collateral on the 90: 2 argmax flips total over 5 seeds x 90 victims; mean |victim margin
  change| 1.79-3.05 per seed.
- Predicted vs measured: Pearson r = 1.000000, Spearman = 1.000000; max |predicted -
  measured| margin change = 1.8e-10. The closed-form Gram prediction of the KKT edit is
  EXACT (the edit is linear algebra; the registered prediction is the theorem, the
  measurement its verification). The scale-free Gram proxy (no lambda solve) still ranks
  collateral at Spearman r = 0.47.
- Naive-removal baseline (subtract the rank-1 component in the RAW key direction, one pass,
  no re-tensioning): 9.2x worse mean |margin change| (15.8-32.3), 249/450 victims flipped
  (retained-fact accuracy 0.30-0.60 vs 0.989-1.000 for KKT), and it does not even forget —
  residual deviation from uniform 23-68 logits from cross-talk among the 10 edits.
  KKT single-pass (no re-tensioning): collateral same as converged KKT, forgetting
  incomplete (deviation 5-12) — re-tensioning buys exact forgetting, C^{-1}-weighting buys
  the 9x collateral reduction.

### F12 — behavior on ALL 2^20 inputs (exhaustive enumeration, batched; 6 s CPU)

Method: exhaustive enumeration in batches of 65536 (not a derived bound), for the KKT
interpolant, the ALS construction, and SGD seed 0. Margin = top1 - top2 logit.

| model | max off-fact margin | min on-fact margin | frac off-fact > min on-fact | analytic bound | tightness |
|---|---|---|---|---|---|
| KKT interpolant | 12.17 | 5.00 | 1.4e-2 | 689.9 | 0.018 |
| ALS construction | 33.52 | 5.00 | 2.1e-1 | 1238.0 | 0.027 |
| SGD seed 0 | 63.26 | 10.90 | 2.0e-1 | 2151.4 | 0.029 |

The analytic Gram-overlap bound margin(x) <= 2 max_k|lambda_ck| x^T C^{-1} x (exact form
for the interpolant; + 2||E_c||_2 ||x||^2 residual term for ALS/SGD, lambda from key-frame
projection) HOLDS on every one of the 2^20 inputs but is loose: 35-55x above the realized
maximum. Off-fact margins are NOT uniformly small: for ALS/SGD, ~20% of the 2^20 - 100
inputs carry a larger margin than the least-confident stored fact (mostly key neighbors and
unions); the minimum-C-norm interpolant is an order of magnitude better behaved (1.4%).
The claim the post can make is exactly the bound and no more — pointwise "no confident
class off the fact set" is false.
