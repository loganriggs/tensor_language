
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

## 630. Depth-of-computation synthesis: FRONT (0-2) decides the class
## and carries prediction, MIDDLE (6-16) refines rare/content tokens
## within-class (light, rare-weighted, no class writing), BACK (17)
## calibrates frequency. CE cost by depth band.

CE with each contiguous depth band mean-ablated (baseline CE: all 3.354,
freq-target 1.607, rare-target 4.083):
  front[0-2]     dCE all +7.02  (freq +3.88, rare +8.32)
  early-mid[3-5] dCE all +1.92  (freq +0.42, rare +2.55)
  mid[6-8]       dCE all +0.61  (freq +0.32, rare +0.74)
  late-mid[9-11] dCE all +0.72  (freq +0.18, rare +0.94)
  [12-14]        dCE all +0.41  (freq +0.05, rare +0.56)
  [15-16]        dCE all +0.60  (freq +0.08, rare +0.82)
  back[17]       dCE all +0.43  (freq -0.17, rare +0.69)
  (a) FRONT DOMINATES HELD: [0-2] costs +7.02, far the largest.
  (b) MIDDLE IS LIGHT HELD: each middle band costs only +0.4-0.7 nats,
      ~10x less than the front.
  (c) BACK CALIBRATES HELD: [17] has dCE_freq < 0 (helps frequent
      targets) while raising CE overall -- the calibrator sign (628).
  Sub-additive: sum of band costs 11.71 > all-18-ablated 8.25 --
  redundancy across bands (destroying the front makes later damage
  partly moot).
THE SYNTHESIS (assembling 624-630, a depth-of-computation account):
  FRONT (blocks 0-2): decides next-token CLASS (629, top writer for all
    9 classes) and carries the bulk of prediction (+7 nats when removed,
    both frequent and rare). This is where the LM computation lives.
  MIDDLE (blocks 6-16): light for prediction (+0.4-0.7 nats/band) and
    writes NO class (629), but its cost is RARE-WEIGHTED (rare-target
    dCE 0.6-0.9 vs freq 0.05-0.3). So the middle is not idle -- it
    refines WITHIN-class specific-token identity, mostly for rare/
    content tokens, not which class comes next. This reconciles 629
    ("middle writes no class") with the middle costing something:
    class is set up front, the middle picks the specific (usually rare)
    token.
  BACK (block 17): frequency calibration, shifting mass from function
    classes to content classes (629), net-beneficial (626).
HONEST CAVEAT: the front's +7.0 magnitude is inflated by error-
compounding (front ablation propagates through 15+ downstream layers),
so it should not be read as a clean additive share; the robust claims
are (i) the middle's ABSOLUTE cost is small and rare-weighted (an upper
bound independent of compounding) and (ii) the back's calibrator sign.
Queued middle_within_class to test the synthesis directly: does middle
ablation degrade P(correct TOKEN) far more than P(correct CLASS)
(within-class refinement), while front ablation kills both?

## 631. Synthesis CONFIRMED: the middle refines WITHIN-class token
## identity. Ablating the middle destroys the specific token (93% of
## P) but spares the class (only 55%); the front drops both. Closes the
## depth-of-computation phase (624-631).

Direct test of 630's synthesis. Per position, P(correct next token) and
P(correct class) (mass on all tokens of the target's class), with the
front [0-2] vs the middle [6-16] mean-ablated. Baseline: P(token)
0.252, P(class) 0.635.
  FRONT [0-2] ablated: P(token) drops 88.2%, P(class) drops 73.2%
    (sparing +0.151). The front takes down BOTH -- it decides the class.
  MIDDLE [6-16] ablated: P(token) drops 93.1%, P(class) drops only
    55.3% (sparing +0.378). The middle takes down the specific TOKEN
    far more than the CLASS -- it refines within-class identity.
  (a) HELD: middle spares class over token (token-drop 0.931 > class-
      drop 0.553). (b) HELD: front drops class about as much as token.
  NULL HELD: middle sparing (+0.378) is 2.5x the front's (+0.151) --
  within-class refinement is specifically a middle property, not a
  generic ablation effect. (Rare targets sharpen it: middle spares
  class at 0.530 while token collapses 0.991.)
THE DEPTH-OF-COMPUTATION ACCOUNT, confirmed and closed (624-631):
  FRONT (blocks 0-2): decides the next-token CLASS (629 top writer for
    all 9 classes; 631 front ablation drops class as hard as token) and
    carries the bulk of the loss (630, +7 nats). The LM decision is
    made here.
  MIDDLE (blocks 6-16): refines WITHIN-class specific-token identity
    (631, spares class 2.5x more than the front), light for total loss
    and rare-weighted (630), writes no class (629). Class is set up
    front; the middle picks the specific (usually rare/content) token.
  BACK (block 17): net-beneficial FREQUENCY CALIBRATION (626), a class-
    level shift of mass from function classes to content classes (629),
    the model's single calibrator (628).
This is a clean three-stage division of labor -- decide the class,
refine the token, calibrate the frequency -- each stage established by
its own causal test with controls and nulls. Phase boundary: updating
the report artifact with this depth-of-computation account. Queued
middle_refines_which_class to localize WHICH classes the middle refines
(content classes should show large within-class sparing; function
classes little, having few members to choose among).

## 633. Per-block localization of within-class refinement FAILS to a
## redundancy confound: single-block sparing is highest at the FRONT
## (block 1 +0.63), not the middle. The band-level synthesis (631)
## stands; refinement cannot be localized below the band with ablation.

Attempt to locate WHERE the space_word content-word refinement (632)
happens, per block. Per-block within-class sparing (token-drop minus
class-drop) for space_word:
  block 0 +0.39, block 1 +0.63 (top), block 2 +0.51, block 3 +0.28,
  block 4 +0.20, block 5 +0.26, block 6 +0.05, block 7 +0.13,
  block 8 +0.29, block 9 +0.11, blocks 10-16 +0.02..0.19, block 17
  -0.18 (calibrator: drops class more than token).
  (a) FAILED: middle per-block sparing (avg 0.10) is LOWER than front
      (avg 0.51) -- the opposite of the registered prediction.
THE CONFOUND (why (a) fails and the per-block metric is invalid here):
single-block ablation leaves the model's REDUNDANT class-writers intact.
The class "a space-word is coming" is written across the front blocks
(0-2) redundantly (629 + the program's pervasive redundancy findings).
So removing ANY one block spares the class -- the others still write it
-- inflating the sparing metric, and MOST at the front, where the
remaining redundant class-writers are. The per-block sparing therefore
measures "how redundantly is the class written around this block", not
"does this block refine within-class". It cannot localize refinement.
  The VALID measure is the cumulative-band comparison (631/632):
  removing the WHOLE middle band [6-16] spares the class (class-drop
  0.30) while destroying the token (0.97) -> sparing +0.67; removing the
  WHOLE front band [0-2] drops class and token together (0.70 / 0.98) ->
  sparing +0.27. Removing the whole band removes its redundant class-
  writing at once, so the band comparison is not confounded, and it
  correctly shows the middle spares class while the front decides it.
  block 17's NEGATIVE per-block sparing (-0.18) is real and un-
  confounded (it is the sole calibrator, no redundant partner): it drops
  class MORE than token, the frequency-calibration signature.
HONEST OUTCOME: I cannot localize within-class refinement below the
band level with single-block ablation -- redundancy defeats it, the same
wall this program hit for article/newline unit clusters (610-616). The
band-level three-stage synthesis (631: front decides class, middle
refines within-class, back calibrates) is the resolution-limited but
valid statement. This closes the "where exactly" question as
un-answerable by ablation and the depth-of-computation phase overall.
Next: pivot to the FRONT/input -- how blocks 0-2 decide the class from
the current token vs context (tracing toward the embedding, the
program's core goal). Queued front_token_vs_context.
