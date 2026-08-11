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

Path removal — REDESIGNED 2026-08-10 per Logan (deviation from the handoff's 4th-class
Human/hands/dog-ears design, documented here). One extra feature instead of one extra class:
features [furry, happy, whiskers, tail], classes still Dog/Cat/Catfish. Each class keeps a
core feature ANDed with either furry or tail: Dog = (furry,happy) | (happy,tail),
Cat = (furry,whiskers) | (whiskers,tail), Catfish = (whiskers,happy). Five keys; the kept
Cat(whiskers,tail) path SHARES tail with the removed Dog(happy,tail) path. Overcomplete H=12
and undercomplete H=3 both train 5/5 seeds to 100%. Registered predictions in
predictions/part1_f8_tail_prediction.md (three rounds, each committed before its measurement).

- Edit 1, minimal T-entry edit: least-norm Dog D-row update, (happy,tail)->Dog exactly 0,
  (furry,happy)->Dog exactly preserved; Cat/Catfish slices untouched by construction.
  Overcomplete: surgical in the weight sense (0 accuracy flips anywhere) but the Dog(h,t) key
  is STILL classified Dog on 5/5 seeds — margin drops -8.5 mean but stays positive. The
  diagonals co-carry the path (registered prediction expected a flip: FAILED, replicating the
  old design's surprise). Undercomplete: same edit, real collateral — 2/5 seeds flip
  non-target keys (seed 0: both Cat keys -> Dog; seed 3: Dog(f,h) -> Cat, target -> Catfish).
- Edit 2, FULL tail removal: least-norm zeroing of ALL four tail-involving entries of Dog's
  slice, preserving (furry,happy) — 5 exact constraints, so expressible only when H >= 5
  (in undercomplete H=3 the edit does not exist: fewer free parameters than constraints).
  Overcomplete result: STILL no flip on any seed (registered 0.65-confidence prediction
  FAILED). With every tail term gone, f_Dog({happy,tail}) = the (happy,happy) diagonal alone
  — Dog's core-feature linear term, shared with the kept (furry,happy) path — and it still
  outscores Cat and Catfish. Weight-basis path surgery cannot remove this fact without
  touching shared structure.
- Edit 3, FUNCTIONAL key-frame edit (F8b; the Part-2 KKT family): least-norm Dog D-row update
  with constraints in the stored-key frame — f_Dog(z_{h,t}) set to (max competitor logit - 1),
  f_Dog(z_{f,h}) exactly preserved. Registered prediction CONFIRMED both halves:
  overcomplete flips Dog(h,t) on 5/5 seeds (to Catfish 4x, Cat 1x) with ZERO other changes
  (Dog(f,h) margin delta exactly 0.0000) and the (happy,tail) tensor entry is NOT zeroed
  (+3.82 -> -0.42 on seed 0; the entry is not the fact). Undercomplete: target flips 5/5 but
  3/5 seeds also flip Cat keys to Dog — the mean accuracy deltas on Cat(f,w)/Cat(w,t)/Catfish
  are -0.60/-0.60/-0.40. Same edit, same code path: collateral is a property of the REGIME
  (shared stored keys), not of the edit.
- Edit 4, greedy whole-unit ablation baseline: catastrophic in both regimes (overcomplete:
  needs 4-9 of 12 units, flips 40-60% of Dog keys and 20% of Cat(f,w); undercomplete: zeroes
  all 3 units, everything becomes Dog). Facts are not stored in dedicated units.

Figure-display convention (2026-08-10, per Logan): all Part-1 figures now display ONE seed
(seed 0), not seed-means; F2 shows a single trained row (wd = 0) with no weight-decay sweep
in the figure (the sweep is still trained and reported in the stage log); cross-seed stats
stay in this file. Downstream stages (F5, F6/C1, F7, pull-out) still use the wd=1e-3 model
family their registered predictions were committed against.

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

### Part 2 addendum: the D-to-minus-I sweep (Logan's feedback on F9 panel iii)

Registered in predictions/part2_Dsweep_prediction.md before running; numbers in part2_dsweep.json; figure F9b.
Logan reports D -> -I locally under a sparsity constraint, as one of many solutions. The sweep (L1 in
{1e-3..1e-1}, H=40, 5 seeds, 100% memorization everywhere) resolves the apparent conflict in three parts:
1. F9's "dominance 0.39-0.49" averaged over ALL 40 columns including near-dead ones. Restricted to LIVE
   units (column norm > 1e-3 of max), dominance is already 0.78-0.88 at our original L1=1e-3 and rises to
   0.87-0.95 at L1=1e-1, while live units collapse to 10-12 (~= C=10, one unit per class). The signed
   one-hot column structure Logan saw IS there and strengthens with sparsity.
2. The SIGN is gauge in our parameterization: flipping L_h's sign flips the product's sign, and the L1
   penalty is invariant under (L_h, D[:,h]) -> (-L_h, -D[:,h]); measured negative-dominant fractions sit at
   0.25-0.75 (mean ~0.5) at every L1, exactly as a sign gauge predicts. "-I vs +I" is unidentified here;
   the invariant statement is |D| -> scaled class-permutation. A convention that fixes signs (e.g., D
   initialized at -I, or nonnegativity on L/R) would make -I the selected representative.
3. Prediction scoring: dominance-rises-with-L1 held; negative-fraction -> 1 FAILED (explained by the gauge);
   memorization never broke, even at L1=1e-1 (predicted it might).

## Part 3: facts in 2 bilinear blocks with a residual stream — F13

From `part3.py` (stages sweep/verify/control/measure/figure; seeded, re-runnable).
Registered predictions P1-P4 in `predictions/part3_predictions.md`, committed before the
sizing sweep and all measurements. Metrics in `figures/part3_metrics_N1200.json`.

Sizing (documented deviation from the handoff's "200 facts": 200 turned out to be far
below single-layer capacity, so the fact count is the sizing knob).
- Architecture: x -> x + B1(x) -> ... + B2(...), linear readout, no biases; d = 20 stream,
  n = 20-bit keys, 10 classes; AdamW, 25k steps, 5 seeds.
- A single bilinear block memorizes 200, 400, and 800 facts at every H tried — 400+ facts
  with only ~210 quadratic monomials available, so margin classification capacity is far
  beyond the interpolation count. The ceiling sits between 800 and 1200: at 1200 facts a
  single block gets 54.0-54.2% and is EXPRESSIVITY-limited, not parameter-limited — H = 40
  and H = 210 (the full quadratic span) give identical accuracy to three decimal places.
- Two blocks at H = 40 per block (4,800 block parameters vs the failing single block's
  12,600) memorize all 1200 with loss ~0; H = 120 two-block holds 2000 facts at 100%.
- Working point: N* = 1200, H* = 40 per block.

Metric positive controls (gate): train the 2-block model with one block frozen at zero
(200 facts, H = 40). The attribution metric assigns 0 of 200 memorized facts to the dead
layer in both directions (177 to the live layer, 23 to the linear bin). PASS both ways.

Attribution by single-layer evaluation (5 seeds, bins over the 1200 facts):
- linear (correct with both blocks off) 104-156; layer-1-only 124-157; layer-2-only
  127-157; both 14-33; NEITHER 739-790 (62-66% of all facts).
- The chance floor is 10% (10 classes, 120 facts): the linear and single-layer bins all
  sit at or barely above chance. Single-layer evaluations are close to random with respect
  to the stored labels — essentially NO fact is attributable to one layer.
- P1 (attribution not disjoint, < 30% in clean single-layer bins) CONFIRMED: 21-25%.

Cross terms (full logits minus the additive degree-2 surrogate W(z + B1(z) + B2(z))):
- The additive surrogate keeps only 17.2-18.9% accuracy — it loses 973-994 of 1200 facts.
  P2 (>= 25% lost) CONFIRMED at ~82% lost: the composed degree-3/4 terms are not a
  correction, they ARE the memory.
- Median |cross| / |full logits| per fact: 0.97-0.99 — ~98% of logit magnitude on stored
  keys comes from the composed terms.
- Per-fact margin contribution of the cross terms (F13 panel ii): almost entirely
  positive, median ~+40 (seed 0), i.e. the composed terms build the margin.

Cancellation / the draft's "negation for every fact" hunch (F13 panel iii):
- P3 predicted median cos(cross, layer-1 contribution) < 0 (cancellation). REFUTED:
  medians are +0.07 to +0.13 (layer 1) and -0.07 to +0.004 (layer 2) — broad,
  near-zero-centered distributions. The composed term neither systematically cancels nor
  amplifies a layer's direct write; it is a nearly orthogonal third channel.

Ablations (survivors out of 1200):
- zero block 2: 185-237 survive; zero block 1: 175-198 survive. Both catastrophic
  (~16-20%, vs 10% chance). P4 predicted removing block 2 hurts MORE; REFUTED in 4/5
  seeds — block 1 is slightly more load-bearing (block 2's contribution is computed on
  the block-1-enriched stream, so removing block 1 also corrupts block 2's input).

Verdict for the post (handoff allows "it is/isn't cross-layer" + one figure): it IS
cross-layer — at capacity-stressed sizing, SGD stores facts almost entirely in the
composed degree-3/4 structure spanning both blocks, with no per-layer fact partition and
no evidence for the negation hunch at the logit level.

### Part 3 follow-ups per Logan 2026-08-11 (F13b capacity curve, F13c closed-form edits)

Bin relabel (F13, terminology question): the attribution bins classify each fact by
whether a SINGLE layer suffices. "both" = either layer alone classifies the fact
(redundant storage); "neither" = no single layer suffices — which is exactly where
cross-term (composed) storage lands. Figure labels renamed to "either alone (redundant)"
and "needs both (composed)".

F13b capacity-vs-width (registered P9). Trained on a fixed pool of 4000 facts, seed 0:
- 1 block: 955 facts at H=20, then 1022-1026 for every H in {40, 80, 120, 210, 300} —
  flat to three digits across a 7.5x width range, including past H=210 (the full
  quadratic span). The plateau is the expressivity wall of the degree-2 function class.
- 2 blocks: 1244 (H=10), 1994 (H=20), 3522 (H=40), 4000 = the whole pool (H >= 80).
  True 2-block capacity at H >= 80 exceeds the pool; the curve is width-limited below
  that, never expressivity-limited in the range tested.
- P9 CONFIRMED both halves (plateau even flatter than the predicted <10% variation;
  plateau LEVEL ~1024 was higher than the guessed ~650 — with a 4000 pool the model
  selects an easier 1024-fact subset than a fixed 1200-fact set forces).

F13c closed-form edits in the 2-layer model (registered P5-P8). The model is LINEAR in
the last block's output map: logits = W x1 + G h2 with G = W D2, x1 = z + B1(z),
h2 = (L2 x1)*(R2 x1) in R^40. The entire Part-2 KKT machinery transfers with h2 as the
stored-key frame and C = sum_k h2(z_k) h2(z_k)^T. Delta-D2 realized as pinv(W) Delta-G;
exactness verified against the actual edited network (max deviation < 1e-6).
- P5 removal (10 random facts -> uniform, retain-aware C^-1-weighted joint KKT): exact
  forgetting by construction, but 516-536 of 1190 retained facts flip (~45%) with median
  retained margin drop 22.7-25.3 — essentially the whole median margin (24.4). CONFIRMED
  (predicted >= 5%) but the magnitude is the story: the same machinery that cost 2/450
  flips in Part 2 (100 facts, same 40-dim frame size) costs ~45% at 1200 facts. The
  frame is 30x overloaded; even the collateral-OPTIMAL edit destroys half the store.
- P6 frame comparison: the readout-frame edit (Delta W, keys x2 in R^20) flips 679-773
  (1.3-1.5x the h2 frame). Direction confirmed, the registered 2x threshold NOT met —
  scored REFUTED on its stated bar.
- P7 injection of 10 NEW facts (raise the new label's logit to best-competitor + 1,
  same least-norm machinery): 10/10 land exactly on every seed; 218-390 of 1200 stored
  facts flip (~18-32%) — same order as removal, slightly cheaper. CONFIRMED.
- P8 pull-out: the naive F10-style metric (component classifies its own key) returns
  1199-1200/1200 — DEGENERATE, not recovery: with 1200 dictionary atoms in a 40-dim
  frame, a_k <d_k, h2_k> is a whitened copy of the model's own logits (metric bug caught
  before claiming; see surprises). The meaningful component-DELETION test: removing fact
  k's least-squares component breaks fact k 0/1200 times, flips a median 0.0% of other
  facts, clean pull-outs 0/1200. No per-fact component is selective OR load-bearing —
  facts have no addressable weight-space location in the composed regime. P8's letter
  (<10% on the naive metric) is moot (metric saturates upward); its substance (no
  per-fact pull-out) is maximally confirmed. Class-level pull-out (one row of W plus
  both blocks) remains exact by construction. Generalizable-behavior pull-out is not
  defined for random facts — Part 4's structured variant is where that lives.

### Removal-methods ladder (Logan's question: is the KKT collateral fundamental?) — F13d

Registered P10-P14 (predictions/part3_predictions.md, addenda 2-3, committed before each
measurement). All methods are closed-form/convex, corpus-free, fact-list-only. Numbers in
figures/part3_edits2.json and part3_edits3.json. A note on the "forget n/10" counters for
LP methods: the removal equalities pin the removed keys' logits to EXACTLY uniform
(deviation ~1e-11); argmax at exact uniformity is a tie-break artifact, so forgetting is
exact for every LP method regardless of that counter.

- A. KKT L2-minimal (from F13c): exact forgetting, 516-536/1190 retained flips.
- B. Weighted LS refit of the joint readout [W,G] (P10): 236-300 flips but FORGETTING
  FAILS — only 1-6 of 10 removed facts actually flip (deviation from uniform 35-57
  logits). It trades forgetting for retention; not a valid removal method. P10's flip
  clauses technically held, but the registered prediction omitted a forgetting criterion
  — scored as moot, lesson noted.
- C. Margin LP over Delta-G (P11, the decisive test): INFEASIBLE on all 5 seeds —
  minimum total hinge violation 5067-5873 with 552-620 facts violated at the optimum.
  This is a CERTIFICATE: no edit of the last block's output map (any Delta-D2, not just
  the KKT one) can remove the 10 facts while keeping all 1190 at margin >= 0.5. The
  ~45% KKT collateral is fundamental to the last-layer frame, not an objective artifact
  (the LP optimum's own flips, 522-578, match the KKT edit's). P11 CONFIRMED.
- D. One exact R2-frame repair round after C (P12): 233-274 flips — a 52-56% cut vs A,
  beating the certified single-frame floor by editing a SECOND frame. P12 CONFIRMED.
- E. Alternating single-frame margin LPs, G -> R2 -> L2 -> ... (P14, seed 0), removal
  equalities re-imposed exactly every round: retained flips 522 -> 257 -> 121 -> 118 ->
  23 -> 0 -> 0. Round 5 reaches TOTAL SUCCESS: all 1190 retained facts hold with margin
  >= 0.5, the 10 removed keys sit at uniform to 3e-12, verified by exact forward
  evaluation of the edited network. P14 CONFIRMED far beyond its bar (predicted <10%
  within 6 rounds at confidence 0.4; got 0%).
- Oracle retrain on the 1190 (P13): 100% retention, 0/10 removed still correct.
  CONFIRMED. The alternating-LP editor matches the oracle's behavior on the fact set
  with six convex solves and no training.

Verdict: composed (cross-layer) storage is UNEDITABLE layer-locally — provably — but
FULLY editable cross-layer with pure closed-form steps. The storage knowledge that
matters is which parameter blocks the function is linear in (D2, W, R2, L2 each are,
with the others fixed); alternation through those frames is the editing algorithm.

## Part 2 addendum (2026-08-11): the margin-LP / multi-frame editor in one layer — F11d

Registered P15/P16/P16b/P16c in predictions/part2_f11d_prediction.md (three commits, each
before its measurement). One deviation, stated: the overload arms train without the L1
penalty (l1=0, 20k steps) so that memorization is exact at large N.

One layer has the same three exact linear frames as the 2-layer last block (D, R, L) plus
one the 2-layer model cannot afford: the FOLDED TENSOR itself (Delta-B, C x 210 = 2100
dof — every quadratic logit function). Results, same 10-fact removals as F11:

- P15 CONFIRMED (working point, 100 facts, all 5 seeds): the D-frame margin LP alone is
  feasible — exact-uniform forgetting (dev ~1e-12; seed 4 3e-4, LP tolerance), 0/90
  retained flips, one round. Upgrades F11's KKT result (2/450 flips, 0.44%) to exactly
  zero; naive rank-1 subtraction remains the disaster baseline (55% flips + fails to
  forget).
- P16 half-failed as registered: at 350 facts (load 8.8x) the D-frame LP is STILL
  feasible (0 flips, one round) — no ladder needed yet.
- Escalation (P16b REFUTED): at 600 facts (15x) the D-frame goes infeasible (255 flips at
  its optimum) but ONE more weight-frame round (D -> R) reaches 0 — the ladder appears in
  one layer. At 900 facts (22.5x, still 100% memorized; capacity ~1024) the weight-frame
  alternation STALLS: 572 -> 403 flips over 7 rounds, violation plateauing ~733. Near
  capacity, a single layer is NOT weight-frame editable — unlike the 2-layer model at
  30x load, whose last-block frames sit on a composed representation rich enough for
  alternation to converge (F13d).
- P16c CONFIRMED: the tensor-frame margin LP is feasible at 350, 600, AND 900 — one
  round, zero collateral, exact-uniform forgetting (dev <= 3e-9), verified by evaluating
  the edited folded tensor. The alternation stall is a parameterization artifact of
  coordinate descent in L/R/D, not a function-class limit. (Note: realizing the edited
  tensor as an explicit bilinear layer may need H up to d per class — folding is the
  free maximal frame for ONE layer only.)

Story for the post: editability = which LINEAR frame you can afford. A single bilinear
layer gets its maximal frame for free by folding (210 x C coefficients), and in it
removal is one convex solve at any memorizable load. A composed 2-layer model's maximal
frame is the degree-4 polynomial space (~d^4 — unaffordable); it must edit in weight
frames, where alternation converges BECAUSE composition makes those frames rich. The
weight-frame ladder is the fallback that happens to work; folding is the primitive that
makes one layer trivially editable.
