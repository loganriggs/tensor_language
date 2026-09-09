# Source-support interaction law: trained v1 audit

The final native attention layer has pointwise RMS and fixed causal/RoPE operations,
a sum over source terms depending on only query and source states, and no final
normalization/softmax on logits. Independently removing disjoint L2 source writes
therefore creates a Boolean intervention function with at most pair interactions.
Pair interactions can appear only at changed query positions reading another
changed source. Unchanged finalquery positions have no pair interaction. This is
a mathematical architecture restriction, not evidence of smaller learned weights.

Fresh16 IID24-cycle worlds20909 and16 OODthree8-cycle worlds20910. Three disjoint
three-edge query chains start at cycle-permutation offsets0/8/16. Place their facts
at slots(1,8,17),(3,10,19),(5,12,21), with orientations forward/backward/forward,
orders(0,2,1),(0,1,2),(0,2,1). Shuffle remaining15 facts uniformly. Twelve queryforks
(three starts×hops0..3) perworld; record native capability without filtering.
Selected exact L2 writes use the same four source→destination cells and native
H1/H2, at distinct destinationpairs. Base post-L2 states recomputed from tokens.

Evaluate all8 removal subsets. Construct the degree2 anchored response from the
zero,three singles andthree pairs only, then predict the withheld triple. No fitting
or use of triple output in its prediction. Compute Boolean Mobius coefficients.
Compare all29 logits/allpositions. The paired source-group support has its only
allowed pair outputs at the later changed destinationpair; finalqueries are outside.

Overlap control: the three exact E/Y0/Y1 V contributions of the backward route
all edit the same destinationpair. Evaluate the same8 subsets; thirdinteraction
need not vanish. The live synthetic overlap controls already pass (thirdmax.00112
and exact integer416), so the instrument detects a violated support hypothesis.

A instrument: source masks disjoint/causal, native model vs cached-prefix suffix
replay<=1e-9 (relativeRMS<=1e-10 withfloor1e-6), all finite, synthetic controls pass.
B disjoint law: withheld triple all-logit maxerror<=1e-9 andrelativeRMS<=1e-10;
pair effectsoutsideallowedquerysupport<=1e-9; every population has some allowed
pair effect>=1e-6 so this is not merely a zero-response fixture.
C overlap control: pooled third-interaction centered-logit RMS>=1e-6 perpopulation.
D probability caveat: report pair interactions after log_softmax at unchanged
finalqueries; predict pooledmax>=1e-6 perpopulation even where raw-logit pair
interactions vanish. This guards against inferring noncompositionality from KL/CE.

Prediction/support claims are exact algebra with FP64 validation, not trained
OODextraction or compactness.32native worlds384queryvariants,16arms (8disjoint+
8overlap),B4FP64,1800s,<256MiB/tensor,managedGPU. Runtime implementation retains
all400640 native parameters and per-input response evaluations. No rank/basis/
threshold sweep. A numerical failure repairs a concrete implementation issue;
a valid violation requires checking the stated locality/suffix assumptions.
