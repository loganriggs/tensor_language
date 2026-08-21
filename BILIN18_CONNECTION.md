
## 626. Block 17's frequency suppression is a NET-BENEFICIAL
## CALIBRATION with the exact predicted trade-off: it costs a little CE
## on frequent targets to gain more on rare ones (+0.43 nats net).

The trade-off test for 625. Per-position CE with vs without a block,
split by whether the target next-token is in the top-20 most frequent
tokens. Baseline CE: all 3.354, frequent-target 1.607, rare-target
4.083 (rare tokens are far harder, as expected).
  block 17 removed: dCE all +0.434, frequent-target -0.168, rare-target
    +0.686. ALL THREE registered predictions HELD:
    (a) removing block 17 RAISES CE at rare targets (+0.686) -- its
        suppression of frequent COMPETITORS was helping rare
        predictions.
    (b) removing block 17 LOWERS CE at frequent targets (-0.168) -- its
        suppression of the correct frequent token was hurting there.
        Opposite sign to (a): the calibration trade-off is real.
    (c) NET +0.434: block 17 is net-beneficial. The trade nets positive
        because rare targets outnumber frequent ones (8666 vs 3622), so
        the rare-target gain (+0.686) outweighs the frequent-target
        loss (-0.168).
  CONTROL block 1 (a writer) removed: dCE all +1.505, frequent +0.770,
    rare +1.812 -- RAISES CE at BOTH (same sign). A writer hurts
    everything; only block 17 makes the calibration trade-off.
CLOSES the block-17 calibration thread (624-626). Block 17's causal
function is a net-beneficial frequency calibration: it sacrifices ~0.17
nats on common tokens to save ~0.69 nats on rare ones, and because rare
targets dominate the corpus it nets +0.43 nats. This is the precise,
quantified mechanism behind 624 (suppresses newline/article) and 625
(suppression scales with frequency, corr +0.64). The readout layer is a
CALIBRATOR, definitively: it reads the distribution (615/618) and
applies a frequency correction that trades frequent-token accuracy for
rare-token accuracy, net positive. This also grounds the whole read/
write arc (619-622): the read and write axes are orthogonal because the
last layer's job is not to write any class but to recalibrate the
distribution the writer layers produced. Queued block_calibration_
profile to finish the picture: is calibration localized to block 17, or
do the last few blocks all calibrate? (per-block corr(log-freq, delta)
across all 18 blocks, + how much of block 17's action is pure frequency).

## 627. Calibration is LOCALIZED to two blocks: block 17 (strong,
## +0.64) and a weaker SECOND calibrator at block 5 (+0.36); all other
## blocks write or are neutral. Block 17 is ~41% pure frequency bias.

Full 18-block profile of corr(log token frequency, per-token removal-
delta): positive = calibrator (suppresses frequent tokens), negative =
writer (builds them).
  CALIBRATORS (corr > 0.2): block 17 (+0.643, dominant) and block 5
    (+0.356, weaker). Only two.
  WRITERS (corr < -0.2): blocks 0 (-0.24), 1 (-0.28), 3 (-0.23),
    4 (-0.28), 7 (-0.28), 8 (-0.30), 9 (-0.28). The strong writers of
    frequent tokens cluster in the front and early-middle.
  NEUTRAL (|corr| < 0.2): blocks 2, 6, 10-16 -- weakly negative or
    ~zero; the late-middle blocks 10-16 barely touch the frequency
    axis.
  (0) block 17 corr +0.643 reproduces 625. HELD.
  (a) LOCALIZED HELD: block 17 is the strongest positive, and only 2
      blocks are calibrators. Calibration is a localized function, not
      spread across the back half.
  (b) PURITY: block 17's per-token deltas are explained by log-frequency
      alone at R^2 0.413 -- ~41% of the readout layer's action is a pure
      unigram-frequency bias, the other ~59% is context-dependent
      calibration. So block 17 is substantially but NOT purely a
      unigram-bias subtractor; most of its work is context-sensitive.
  NULL "False" (minor, informative): early blocks 0-4 are NOT all
      strong writers -- block 2 is frequency-neutral (-0.03). The front
      writers are blocks 0,1,3,4; block 2 does something frequency-
      orthogonal.
NEW loose end: block 5 is a genuine-looking SECOND calibrator (+0.36),
consistent with 624 where ablating block 5 slightly RAISED P(newline)/
P(article) (it suppresses those high-frequency classes too). So the
model has a weak mid-network calibrator at block 5 and a strong final
one at block 17. This CLOSES the localization question for the
calibration thread; block 5 warrants a causal confirmation. Queued
calibrator_ce_profile: run the 626 CE trade-off (help rare targets,
hurt frequent targets) for blocks 5, 6, 17 and a writer control -- does
block 5 show the genuine calibrator signature, cross-validating the
frequency-correlation metric against causal CE?

## 628. CORRECTION to 627: block 5 is NOT a second calibrator. The
## causal CE test shows block 17 is the UNIQUE net calibrator; block 5
## is a large WRITER with a relative frequency tilt. Frequency-
## correlation flags tilt, not net calibration.

Cross-validating 627's frequency-correlation calibrators against the
626 causal CE trade-off. Baseline CE: frequent-target 1.607, rare-
target 4.083.
  block 17: dCE freq -0.168, rare +0.686, all +0.434. The ONLY block
    whose removal HELPS frequent targets (dCE_freq < 0) -- the true
    calibrator signature (it net-suppresses correct frequent tokens).
  block 5: dCE freq +0.227, rare +2.091, all +1.542. Removing block 5
    HURTS frequent targets too (positive) -- block 5 net-WRITES frequent
    tokens, it does not suppress them. It is a large writer (dCE all
    +1.54, one of the biggest) that helps rare targets even more.
  block 8 (writer control): dCE freq +0.157, rare +0.297 -- writer,
    both positive. HELD.
  block 6 (neutral): dCE freq +0.024, rare +0.119 -- tiny, near-neutral.
THE CORRECTION: 627 called block 5 a "second calibrator" from its
positive frequency correlation (+0.36). The causal CE test REFUTES that
as a net-calibration claim. The defining calibrator signature is
dCE_freq < 0 (removal helps correct frequent tokens = the block was
net-suppressing them), and ONLY block 17 meets it. Block 5's positive
frequency correlation reflects a RELATIVE tilt -- block 5 down-weights
frequent tokens relative to rare ones -- but on net block 5 still
WRITES frequent tokens (removing it hurts them). A relative frequency
tilt is not net calibration.
METHOD LESSON (two flawed proxies caught by the CE ground truth):
(1) the frequency-correlation metric (625/627, mean-logit delta vs
frequency) detects relative frequency tilt, NOT net calibration -- a
big writer with a tilt scores positive. (2) The asymmetry metric
(rare dCE - freq dCE) that this script's prediction (a) used is ALSO
insufficient: block 5's asymmetry (+1.86) EXCEEDS block 17's (+0.85)
purely because block 5 is a huge writer that helps rare targets more --
the coded prediction (a) returned True but is misleading. The ground
truth for "net calibrator" is the SIGN of dCE at frequent targets, and
by that test block 17 stands alone. So the corrected picture: block 17
is the model's single net frequency calibrator; every other block,
block 5 included, is a writer (some, like block 5, with a frequency-
relative tilt). This SIMPLIFIES the calibration story back to a single
dedicated calibrator at the readout layer, and adds a caution about
proxy metrics for calibration. Propagated to RESULTS; the report only
ever named block 17 (stands, no change). Closes the calibration thread.
