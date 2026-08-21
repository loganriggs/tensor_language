
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

## 634. The front decides class mostly via the token-local MLP, with
## context a substantial secondary input; NEWLINE is the lone context-
## driven class (front attention writes it, the front MLP suppresses
## it). Context for class identity enters entirely at the front.

Mean-ablating the front [0-2] ATTENTION (context) vs the front MLP
(token-local transform), per class, plus a late [10-12] attention
control. Relative P(class) drop at class-target positions:
  class        front-attn  front-mlp  late-attn   driver
  newline        +0.558     -0.825      +0.218    context(attn)
  determiner     +0.517     +0.932      +0.055    token(mlp)
  preposition    +0.574     +0.890      -0.001    token(mlp)
  pronoun        +0.602     +0.926      +0.071    token(mlp)
  digit          +0.575     +0.971      +0.053    token(mlp)
  punct          +0.338     +0.765      +0.047    token(mlp)
  capitalized    +0.300     +0.870      +0.042    token(mlp)
  space_word     +0.267     +0.785      +0.005    token(mlp)
  subword        +0.310     +0.899      +0.038    token(mlp)
FINDINGS:
1. THE FRONT MLP IS THE PRIMARY CLASS-WRITER. For 8 of 9 classes,
   ablating the front MLP hurts the class more (0.77-0.97) than ablating
   front attention (0.27-0.60). The next-token class is largely a
   token-local MLP computation in the first three blocks. But attention/
   context is a SUBSTANTIAL secondary input -- front-attention ablation
   still costs 27-60% for every class -- so class identity is
   token-transform-dominated with real contextual contribution, not
   purely token-driven. (This refines 614's article story: the attn0
   bigram is a genuine contributor -- determiner front-attn +0.517 --
   but the front MLP carries the larger share.)
2. NEWLINE IS THE LONE CONTEXT-DRIVEN CLASS, with a sign flip. Front-
   attention ablation drops P(newline) (+0.558) while front-MLP ablation
   RAISES it (-0.825: removing the MLP increases newline probability).
   So whether a line breaks next is decided by CONTEXT via front
   attention (positional/where-in-the-line information), and the front
   MLP actively SUPPRESSES newline (favoring content continuation). The
   two front paths pull opposite ways for newline; for every other class
   they agree (both positive).
3. CONTEXT ENTERS AT THE FRONT. Late [10-12] attention ablation barely
   touches any class (drops 0.00-0.22, aggregate 0.53) vs front-attention
   aggregate 4.04. Whatever context sets the next-token class is read in
   by the first three blocks; the late-middle attention does not
   contribute to class identity. (a) HELD, NULL ok.
CAVEAT: front-MLP mean-ablation is a large perturbation (the MLP is the
main per-position compute), so absolute MLP-drop magnitudes are inflated;
the robust signals are the cross-class CONTRASTS (newline's unique sign
flip; attn contributing 27-60% everywhere) and the front-vs-late
attention gap. Queued newline_trigger to trace the newline context
signal to concrete input features (previous-token class, line length)
and confirm front attention carries them.

## 637. The newline and article TRIGGERS are a 0-layer embedding->
## unembedding BIGRAM -- present, and STRONGER, when all 18 blocks are
## skipped. The network ATTENUATES the raw bigram, it does not compute
## the trigger. Recontextualizes 634-636.

The direct path skips every block: logits = unembed(rms_norm(embedding))
-- what the current token predicts with zero context and zero block
computation, purely the learned embedding-unembedding bigram.
  DIRECT (no blocks): P(newline) after . ! ? = 0.418, after a word =
    0.0003 (elevation +0.418). be_pref +0.032 (be -> a/an),
    prep_pref -0.345 (prep -> the, very strong).
  FULL (all 18 blocks): newline elevation +0.290; be_pref +0.033,
    prep_pref -0.111.
FINDINGS:
1. THE TRIGGERS ARE EMBEDDING-LEVEL BIGRAMS. The current token's
   embedding, read straight by the unembedding, ALREADY contains the
   punctuation->newline trigger (0.418 after punct vs 0.0003 after a
   word) and the be->a/an / prep->the article split. The network does
   not COMPUTE these triggers; they are memorized in the embedding o
   unembedding table. (b) HELD (article split present in direct), NULL
   ok (no elevation after a word).
2. THE BLOCKS ATTENUATE, NOT AMPLIFY. Prediction (a) FAILED
   informatively: the direct newline elevation (0.418) is LARGER than
   the full model's (0.290); direct/full = 1.44. Likewise prep->the is
   -0.345 direct vs -0.111 full. The 18 blocks NET REDUCE the raw
   bigram's over-confident trigger toward a calibrated level --
   consistent with the calibration/suppression findings (block 17
   frequency calibration, front-MLP newline suppression). The network's
   job on these tokens is temperance, not computation.
3. RECONTEXTUALIZES 634-636. Those used mean-fill ablation, which is
   confounded by error-compounding (mean-filling an early block corrupts
   the input to all downstream blocks). The clean 0-block direct path
   shows:
   - 634's "front MLP is the primary class-writer (0.93 drop)" over-
     reads: mean-filling the front MLP corrupts 15 downstream blocks;
     the trigger itself is embedding-level, not MLP-computed. What the
     front MLP (and the network) mainly does is set the article-
     prediction MAGNITUDE and temper the bigram, not create the class.
   - 635's "front attention carries the newline trigger" is a partial
     reinforcement (front-attn ablation cut elevation 0.291->0.176, a
     +0.115 contribution) layered on top of the dominant embedding
     bigram -- not the source.
   - 636's a/an-vs-the split is embedding-level (prep->the is even
     stronger in the direct path); attention/MLP modulate it.
DEEPEST INPUT TRACE (the program's goal, reached): the article and
newline circuits bottom out in the EMBEDDING o UNEMBEDDING bigram table
-- the trigger word's embedding directly encodes the next-token bias --
and the 18 blocks act as a tempering/calibration stage on top. Caveat:
the direct path (clean 0-layer) and the mean-fill ablations (compounding-
confounded) measure different things; where they disagree, the direct
path is the clean attribution. Queued class_bigram_vs_computed to
generalize: which classes are embedding-bigram-driven (network
attenuates) vs genuinely computed by the blocks (network amplifies)?

## 638. Refines 637: the blocks don't just ATTENUATE the embedding
## bigram -- they DISCRIMINATE it with context. At trigger positions the
## bigram over-fires and the network suppresses it (637); at true-target
## positions the network AMPLIFIES it. Every class is amplified at its
## own target positions; digit/punct/capitalized most.

Per class, direct-path (embedding->unembedding, no blocks) vs full-model
P(class) at CLASS-TARGET positions (where that class actually comes
next), ratio full/direct:
  subword 1.10, determiner 1.24, space_word 1.30, newline 1.61,
  pronoun 1.55, preposition 2.12, capitalized 4.02, punct 5.33,
  digit 8.32. ALL > 1: every class is AMPLIFIED at its target positions.
RECONCILES WITH 637 (they measure different position sets):
  - 637 measured P(newline) at END-PUNCT positions (the trigger): direct
    0.418 > full 0.290 -- the bigram OVER-fires at all '.', and the
    network SUPPRESSES it (most '.' are mid-paragraph, not line-ends).
  - 638 measures P(newline) at NEWLINE-TARGET positions (newline
    actually follows): direct 0.271 < full 0.435 -- the network
    AMPLIFIES the correct cases.
  Both are true: the embedding bigram is a blunt, CONTEXT-BLIND trigger
  (fires the same at every '.'); the 18 blocks add CONTEXT to
  DISCRIMINATE which instances actually fire -- lowering the bigram where
  it is wrong, raising it where it is right. So 637's "the network
  attenuates the bigram" is the false-positive half; the full statement
  is "the network context-conditions the bigram", which is sharpening,
  not mere tempering.
THE COMPUTED VS BIGRAM SPECTRUM (full/direct at target positions):
  - MOSTLY BIGRAM (network adds little): subword 1.10, determiner 1.24,
    space_word 1.30 -- the embedding already predicts these well at their
    target positions (direct 0.36-0.75).
  - HEAVILY COMPUTED (weak bigram, network builds with context): digit
    8.32, punct 5.33, capitalized 4.02 -- direct only 0.07-0.14, the
    network raises them 4-8x using context.
  - MODERATE: newline 1.61, pronoun 1.55, preposition 2.12.
  (a) FAILED as registered (function classes not "attenuated" at target
  positions -- they're amplified, because target != trigger positions);
  (b) HELD (content computed); NULL ok (bigram is class-specific).
THE CORRECTED INPUT-TRACE SYNTHESIS (634-638): the circuits' triggers
live in the embedding->unembedding bigram (637), but that bigram is
context-BLIND; the 18 blocks' job on these predictions is to
DISCRIMINATE the bigram with context -- amplifying true firings,
suppressing false ones -- and to compute from scratch the classes the
bigram barely encodes (digit, punct, capitalized). Queued
newline_discrimination to demonstrate the discrimination directly: split
end-punct positions by whether a newline actually follows, and show the
full model separates them while the bigram cannot.

## 640. CORRECTION/quantification of 637: the embedding->unembedding
## direct path is NOT a working bigram LM -- its CE (12.65) is WORSE
## than uniform (10.83). The embedding encodes the trigger PREFERENCE
## GEOMETRY for high-signal function-word bigrams, not a calibrated
## whole-distribution bigram. The blocks do essentially all the work.

Quantifying the input-tracing phase. Cross-entropy:
  uniform (log V) 10.83; direct bigram 12.65 (all), 8.65 (freq), 14.32
  (rare); full 3.35 (all), 1.61 (freq), 4.08 (rare).
  (0) FAILED: the direct path does NOT beat uniform -- it is 1.82 nats
      WORSE. The embedding->unembedding shortcut is a BAD language model
      overall.
  (a) blocks help massively: full 3.35 vs direct 12.65 (-9.3 nats).
  (b) blocks help rare (-10.24) more than frequent (-7.05). HELD.
  NULL ok (direct freq CE 8.65 < rare 14.32).
  bigram's share of info gain over uniform: -0.24 (NEGATIVE).
THE CORRECTION to 637's headline ("triggers are a 0-layer embedding
bigram"): that overstated it as if the embedding were a functional
bigram LM. It is not -- read straight through the unembedding it scores
worse than uniform. WHY: the embedding and unembedding are trained to
operate THROUGH the 18 blocks, so the direct map is OFF-DISTRIBUTION --
over-confident on a handful of bigram-favored tokens and miscalibrated
garbage for the general distribution (rare/content tokens), which log
loss punishes.
WHAT STANDS (637-639, correctly scoped): for SPECIFIC high-signal
function-word triggers the direct path's RELATIVE preference is real and
even roughly calibrated -- P(newline)=0.42 after '.' matches the ~36%
actual base rate of newline-after-end-punct (181/506); prep->the,
be->a/an are present. So the embedding encodes the trigger PREFERENCE
GEOMETRY (the '.' embedding leans toward newline more than a word's
does), visible on targeted contrasts. But this is a property of a few
directions, NOT a working bigram model, and it does not survive as a
whole-distribution predictor.
THE HONEST SYNTHESIS of the input-tracing phase (634-640): the embedding
carries the RELATIVE trigger geometry for the function-word circuits;
the 18 blocks do essentially ALL the actual predictive work (+7.48 nats
over uniform; the direct path is net-negative), most of it context
discrimination (638-639) and computing the classes the embedding barely
encodes (638), concentrated on rare targets (b). Propagated to the
report capstone (which had called the triggers a "memorized bigram
lookup" -- tempered to "relative preference geometry the blocks turn
into a real predictor"). Phase closed.

## 641. The digit "class" is TWO circuits: number CONTINUATION (prev
## digit -> digit) is an embedding BIGRAM (full = direct); number
## INITIATION (first digit, no preceding digit) is what 638's 8.3x
## "computed" captured. Front attention discriminates continuation.

Tracing the most-computed class (638: digit full/direct 8.3x). P(digit)
grouped by whether the current token is a digit (224 prev-digit
positions, 12064 not):
  direct (bigram):  prev-digit 0.0627  prev-not 0.0022  (28x elevation)
  full:             prev-digit 0.0652  prev-not 0.0168
  front-attn-abl:   prev-digit 0.0980  prev-not 0.0194
  front-mlp-abl:    prev-digit 0.0082  prev-not 0.0047
FINDINGS:
1. NUMBER CONTINUATION IS A BIGRAM. At prev-digit positions the
   embedding bigram already predicts a digit next (0.063 vs 0.002
   elsewhere, 28x), and the full model barely changes it (0.065 ~ 0.063).
   (b) FAILED as registered (full is NOT >2x direct) -- informatively:
   digit-continuation is NOT computed, it is a memorized bigram, unlike
   638's headline.
2. RESOLVES THE APPARENT CONFLICT WITH 638. 638's digit full/direct =
   8.3x was measured at ALL digit-TARGET positions; this shows
   continuation (prev-digit) is bigram-carried, so 638's "computed" must
   be dominated by the OTHER digit-targets -- number INITIATION, the
   FIRST digit of a number preceded by a word/space ("$|5", "page |3",
   "in 2|023"), where the bigram is weak and the model computes "a
   number is likely here" from numeric context. So the digit class is
   two mechanisms: continuation (bigram) + initiation (computed).
   Registered as a hypothesis; the queued split experiment confirms it.
3. FRONT ATTENTION DISCRIMINATES CONTINUATION. Ablating front attention
   RAISES digit-continuation (0.065 -> 0.098): attention normally trims
   over-continuation (not every digit is followed by another -- "3." can
   end), the same bigram-over-fires / attention-discriminates pattern as
   newline (639). Front-MLP ablation collapses it (0.065 -> 0.008), the
   error-compounding artifact (637 caveat), not a clean attribution.
This shows a token "class" can hide two distinct circuits with different
mechanisms -- a caution for the class-level analyses (629-638): "digit
is computed" (638) is true only for initiation; continuation is a
bigram. Queued digit_init_vs_cont to confirm the split directly:
digit-target positions preceded-by-digit (continuation, expect bigram:
full~direct) vs not (initiation, expect computed: full>>direct).

## 643. THE SENTENCE-BOUNDARY FANOUT (crown of the input-tracing phase):
## one embedding-bigram trigger ('.'), three grammatical outcomes
## (newline / capitalized / continuation), and the 18 blocks ROUTE among
## them by context while the bigram fires identically.

After a sentence-ending '.', the next token is a NEWLINE (181 cases), a
CAPITALIZED word (162), or a lowercase/other continuation (163) --
well-balanced. P(newline) and P(capitalized) at these positions:
  DIRECT (embedding bigram), by what ACTUALLY follows:
    ->newline:     P(nl) 0.416  P(cap) 0.103
    ->capitalized: P(nl) 0.419  P(cap) 0.103
    ->other:       P(nl) 0.419  P(cap) 0.103
    routing: nl -0.003, cap -0.001  (ZERO -- identical regardless of
    outcome; the bigram is context-blind)
  FULL (all 18 blocks):
    ->newline:     P(nl) 0.472  P(cap) 0.242
    ->capitalized: P(nl) 0.224  P(cap) 0.454
    ->other:       P(nl) 0.192  P(cap) 0.251
    routing: nl +0.248, cap +0.212
THE DEMONSTRATION: the '.' embedding bigram proposes a fixed blunt
distribution (~42% newline, ~10% capitalized) at EVERY period, blind to
what follows. The 18 blocks READ CONTEXT and ROUTE: they raise P(newline)
to 0.472 where a paragraph break really comes and drop it to 0.224 where
a new sentence (capitalized word) comes; and they raise P(capitalized)
to 0.454 where a sentence continues and drop it to 0.242 where a
paragraph breaks. The two outcomes ANTI-correlate -- high-newline goes
with low-capitalized and vice versa -- the blocks trade off "end the
paragraph" against "start a new sentence" using context the bigram
cannot see. All predictions HELD (0, b, NULL).
This is the cleanest, most complete statement of the model's job on
structural predictions, crowning the input-tracing phase (634-643): the
EMBEDDING supplies a blunt context-blind trigger from the current token;
the 18 BLOCKS supply the entire context-conditional routing among the
grammatical continuations that trigger allows; and block 17 calibrates
the frequencies. Trigger -> route -> calibrate. Reported to the artifact.
Queued routing_attention_localization to trace WHERE the routing is
computed: does front-attention ablation collapse the newline-vs-
capitalized routing (context enters at the front, 634/635) while late
attention does not?

## 644. The sentence-boundary routing is computed by FRONT ATTENTION:
## ablating front [0-2] attention collapses the routing 80% (0.459 ->
## 0.091), ablating late [10-12] attention leaves 82% intact. Closes the
## input-tracing phase (634-644) with a mechanistic locus.

Localizing the newline-vs-capitalized routing (643). Routing totals
(nl_route + cap_route):
  baseline    0.459 (nl +0.248, cap +0.212)
  front_attn  0.091 (nl +0.047, cap +0.044) -- 80% COLLAPSE
  late_attn   0.377 (nl +0.197, cap +0.180) -- 82% RETAINED
  front_mlp   0.100 (nl +0.083, cap +0.017) -- also collapses (compounding
              artifact, not a clean locus)
FINDINGS: (a) HELD -- ablating front attention removes 0.368 of the
0.459 routing (80%); the sentence-boundary routing lives in the first
three blocks' attention. (b) HELD -- ablating late-middle attention
removes only 0.083 (18%); routing is NOT a late-attention phenomenon.
The clean contrast (front-attn vs late-attn, both targeted context-path
ablations) localizes the routing to FRONT attention, matching 634/635
(the context that sets class identity, and here routes among the
outcomes a trigger allows, is read in by the first three blocks).
THE COMPLETE SENTENCE-BOUNDARY CIRCUIT, end to end:
  1. TRIGGER: the '.' embedding, read by the unembedding, fires a blunt
     context-blind distribution (~42% newline, ~10% capitalized) at
     every period (637/643).
  2. ROUTE (front attention, blocks 0-2): reads the surrounding context
     and routes the probability -- to newline where a paragraph really
     breaks, to a capitalized word where a new sentence starts, to
     continuation otherwise (643/644). This is 80% of the routing.
  3. CALIBRATE (block 17): trim the over-predicted frequent tokens
     (626-629).
This closes the input-tracing phase (634-644): the model's structural
predictions are a blunt embedding trigger, routed by front attention,
calibrated by the last layer -- trigger -> route -> calibrate, each stage
causally established with controls and nulls. Phase boundary; report
carries the full account. Next thread: a fresh circuit -- induction/
copying (does the model copy a token's earlier continuation, and where
is that computed -- likely NOT front attention, a contrast to routing?).
Queued induction_natural.

## 647. COMPONENT-LEVEL: the induction circuit is carried by identifiable
## HEADS, led by L5.H5 (z +3.99 to the copy-source, argmax 33%), then
## L8.H4/H6, L10.H8. Unlike the diffuse routing/MLP circuits, induction
## reaches the head level. (Pattern evidence; causal test queued.)

Scoring every head by how strongly its raw double-QK pattern at
induction positions targets the copy-source s (the token after the
current token's earlier occurrence), over 4672 induction positions.
z-score of pat[i,s] within the head's valid keys; argmax-frac = how
often s is the head's top key; control = a random valid key.
  L5.H5    z +3.99   argmax 0.334   control-z -0.03
  L8.H4    z +2.39   argmax 0.235   control-z -0.03
  L8.H6    z +2.05   argmax 0.154   control-z -0.04
  L10.H8   z +1.35   argmax 0.124
  L13.H2   z +1.06 ; L2.H1 +0.98 ; L14.H0 +0.98 ; L14.H7 +0.90
FINDINGS:
  (a) INDUCTION HEADS EXIST AND ARE LOCALIZABLE. L5.H5 is the dominant
      induction head -- it points its attention at the copy-source with
      z +3.99 and is the single top key at 33% of induction positions
      (chance ~ 1/i, negligible). L8.H4, L8.H6, L10.H8 are secondary.
      This is the FIRST component-level circuit in the program: named
      heads, not a diffuse band. Induction heads are localizable where
      the MLP feature circuits (610-616, 633) and the sentence-boundary
      routing (644, "front attention" but no single head) were not.
  (b) FRONT+MID LAYERS. The induction heads sit in L2-L14 with the
      strongest in L5/L8/L10 -- consistent with 645's band-level result
      (induction distributed across front+mid attention), now resolved
      to the specific heads within those bands.
  NULL: control-key z ~ 0 (-0.04..+0.006) for every top head -- the
      targeting is specific to the TRUE copy-source, not a generic
      attention shape.
CAVEAT: this is PATTERN evidence (the heads point at the copy-source),
which is correlational. The causal confirmation -- ablate these heads
and show copying drops -- is queued as induction_head_ablate. If it
holds, the induction circuit becomes the program's first fully
component-level, causally-verified circuit (named heads, specific
pattern, causal necessity), and the deep-dive deliverable's "we can
localise to front attention but not to a single head" limitation will
be updated for induction specifically.

## 648. Causal confirmation with the program's universal twist: the
## induction heads are causally REAL (ablation drop 5.4x random) but
## REDUNDANT (top-4 removes only 10% of copying). Identifiable and
## causally-directional, but not necessary -- redundancy reaches the
## head level too. Tempers 647.

Mean-ablating the induction heads and measuring P(B) (the copied
continuation) over 4672 induction positions:
  baseline               0.1402
  ablate top-4 induction 0.1268  (drop 0.0134, 9.6% of the signal)
  ablate L5.H5 only      0.1340  (drop 0.0062, 4.4%)
  ablate 4 random heads  0.1377  (drop 0.0025, 1.8%)
FINDINGS:
  (a) CAUSALLY REAL. Ablating the top-4 induction heads {L5.H5, L8.H4,
      L8.H6, L10.H8} drops copying 5.4x more than 4 random heads
      (0.0134 vs 0.0025); L5.H5 alone drops it 2.5x more than random.
      Predictions (a,b) and the NULL all HELD -- the heads 647 found by
      pattern are confirmed to causally contribute to copying, and the
      contribution is specific (random heads barely matter).
  (b) BUT REDUNDANT. The effect is SMALL: ablating even the top-4
      induction heads removes only 9.6% of the copying signal (P(B)
      0.140 -> 0.127), and L5.H5 alone only 4.4%. Copying largely
      SURVIVES ablation of the strongest heads -- it is distributed
      across many heads, no single head (or top handful) is necessary.
TEMPERS 647's "first fully component-level circuit". The precise,
honest statement: for induction we reach component-level IDENTIFICATION
(L5.H5 is the dominant induction head, by a specific attention pattern
z+3.99, causally confirmed drop > random) but NOT component-level
NECESSITY (the heads are redundant; the circuit does not collapse when
they are removed). This is the program's universal redundancy signature
-- diffuse over MLP units (610-616, 633), over depth bands (630), over
attention for routing (644, "front attention" not one head), and now
over induction HEADS -- reaching every level we probe. The strongest
localization the model permits is "these are the heads that do it, and
they matter more than chance," never "this head is the circuit."
SYNTHESIS of the induction thread (645-648): the model does genuine
in-context copying (645), strongest for rare tokens and distance-robust
(646, true induction not skip-bigram), implemented by identifiable
front+mid heads led by L5.H5 (647), which are causally real but
redundant (648). Queued induction_redundancy to quantify the
redundancy curve: cumulative ablation of the top-K induction heads --
how many must go before copying collapses?

## 650. Q5 METHOD WORKS: the block-17 frequency calibration isolates to
## a RANK-1 direction w_freq -- removing it kills the calibration
## (necessary), a random direction doesn't (specific). The
## behavior-conditioned low-rank method reaches finer grain than
## clustering (578-581) or head-ablation (649) could.

Behavior-conditioned low-rank isolation of the calibrator (624-629).
w_freq = cov(mlp17 output, target log-frequency), the rank-1 output
direction tracking frequency. Freq/rare-target CE per variant:
  full           freq 1.607  rare 4.083
  mean_ablate    freq 1.475  rare 4.676   (calibration removed: 626 sig)
  keep_wfreq     freq 1.765  rare 4.483   (rank-1 only)
  remove_wfreq   freq 1.365  rare 4.694   (rank D-1: w_freq removed)
  keep_random1   freq 1.442  rare 5.202   (control)
Calibration rare-benefit (mean - full) = +0.593 nats.
FINDINGS:
  (0) HELD: mlp17 shows the calibration signature -- mean-ablating it
      helps freq CE (1.607->1.475) and hurts rare CE (4.083->4.676).
  (b) w_freq IS NECESSARY -- the clean isolation. Removing just the
      rank-1 w_freq direction collapses the rare-benefit to the fully-
      ablated level (rare 4.694 ~ mean 4.676, 103% lost) AND drops freq
      CE to 1.365 (below mean-ablate -- the frequency suppression is
      gone). So removing one direction removes the calibration MORE
      cleanly than ablating the whole layer (which also removes content).
      w_freq is the calibration direction.
  NULL: keeping a random rank-1 direction gives rare 5.202 (worse than
      removing everything, -89%) -- the frequency direction is special,
      not any direction.
  (a) FAILED, and it is informative not a problem: keeping ONLY w_freq
      recovers just 33% of the rare-benefit. mlp17 is not only a
      calibrator -- it also writes content classes (629, subword/
      capitalized, its rank-8 output). Keeping rank-1 throws that away,
      so the FULL layer benefit needs w_freq INTERACTING with the
      content dims (through rms_norm+softmax); the calibration itself is
      rank-1, the layer is not.
THE POINT FOR Q5: the behavior-conditioned low-rank method SUCCEEDS at
finer-grained isolation where the earlier methods failed. Unit-
clustering (578-581) found stable-but-inert groups; head-ablation (649)
found copying distributed across ~all heads; here, conditioning on the
behavior (frequency) and testing a rank-1 direction by NECESSITY +
SPECIFICITY isolates the calibration to a single output direction. The
lesson: isolate by (behavior-conditioned direction, removal test),
not by (component, clustering). Caveat: the specificity control so far
is keep-random; the matching remove-random control (does removing a
random rank-1 leave calibration intact?) is queued as
lowrank_calibrator_confirm, plus the keep-rank-r sufficiency curve.

## 652. BOUNDARY of the low-rank method: it isolates the CALIBRATOR (an
## additive/subtractive bias, rank-1) but FAILS on the newline ROUTING
## (a conditional computation). Removing the behavior-conditioned rank-1
## direction does nothing to routing (-2%). The correlational direction
## is a READOUT, not the causal carrier (read!=write again).

Applying 651's recipe to the newline routing (643/644). w_route =
cov(residual-after-block-2, newline-follows) at end-punct positions.
Routing R = P(nl | end-punct & follows) - P(nl | end-punct & not):
  baseline            R +0.264
  remove w_route      R +0.268  (lost -2%, i.e. NO effect)
  remove random-1 x3  R +0.264  (lost 0%)
FINDINGS:
  (a) FAILS -- removing the rank-1 behavior-conditioned direction does
      not collapse the routing (-2%, within noise, same as random). The
      routing does NOT live in a removable low-rank residual direction,
      unlike the calibration.
  WHY, two compatible reasons:
   1. The routing is a CONDITIONAL computation ("raise P(newline) IF
      this period ends a line"), not a fixed additive shift. A bias is a
      direction you can subtract; a context-dependent modulation is a
      function, with no single direction to remove. 644 already showed
      the routing needs the WHOLE front attention (80% collapse when
      ablated), i.e. it is a distributed/high-rank computation, not a
      residual feature.
   2. w_route is a READOUT CORRELATE, not the causal carrier. It DECODES
      "a line-end follows" from the post-front residual (it correlates
      with the outcome) but removing it does not stop the routing --
      either the information is high-rank, or later blocks reconstruct it
      from other components. This is read!=write (619-622, 620) at the
      residual-stream level: the direction that decodes the outcome is
      not the direction that causes it.
THE METHOD'S SCOPE (completes the Q5 answer honestly): behavior-
conditioned low-rank + removal isolates ADDITIVE/SUBTRACTIVE low-rank
components -- biases and calibrations, like block 17's frequency
correction (rank-1, clean). It does NOT isolate CONDITIONAL/ROUTING
computations -- these are distributed, high-rank, and/or downstream-
reconstructed, and the correlational direction is a decode-not-cause
readout. So "can we isolate components to finer grain" has a
component-dependent answer: YES for corrective biases (rank-1), NO for
context-conditional routing (needs the whole front attention). Queued
routing_rank_curve to confirm: remove the top-r behavior-conditioned
directions (r=1..32) -- does routing ever collapse (high-rank residual
feature) or never (front-attention-computed, not residual-stored)? --
and report w_route's probe AUC (is it a good readout despite being
causally inert?).
