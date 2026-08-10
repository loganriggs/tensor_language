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
