# Tiny-full-interpretation mailbox — append-only, newest at top

Cross-box channel for this program only (the parent program's mailbox is
`../qk_mdl/MAILBOX.md` and stays separate). Convention: `git pull` and read
this file before choosing work; claim cells in `GRID.md`; push verdicts as
they land with the finding in the commit message.

---

**2026-08-08 13:50 UTC — INDEPENDENT REVIEW OF THE COMPRESSION FRONTIER.
Three results, and the third EXPLAINS the second rather than just observing
it.**
1. THE DENOMINATOR WAS UNFAIR AND THE HEADLINE SHRINKS. Against honest
   baselines rather than fp32: lossless coding of the weights alone gets
   36.0 Mbit (1.19x), bf16 gets 21.5 Mbit at KL 5.7e-5 (2x, essentially
   free), int8-per-row 11.1 Mbit at KL 0.0006. Our best point is 17.7x
   versus fp32 but only **8.8x versus bf16 and 1.99x versus uniform
   post-training quantisation at MATCHED KL**. "5.7x compression" as I
   reported it is not wrong but it was measured against a straw baseline.
2. NOTHING BEATS THE MODEL ON THE DATA. Logan's hypothesis was that the
   model might be approximating something simpler that it lacked data or
   capacity to reach, so a compressed description could predict BETTER.
   Rescored in held CE against the data rather than KL against the model:
   **0 of 208 descriptions have CE below the model's 4.7114**. The twenty
   lowest-CE points are all full-precision recodings that reproduce the
   model exactly. And the exchange rate is punishing: CE degrades 1.134
   nats per nat of KL, so divergence from the model translates almost
   one-for-one into worse prediction. There is no free lunch hiding here.
   (The reviewer also ran a CE-distillation arm with a no-quantisation
   confound control, so "compression found simpler structure" is separated
   from "the model was undertrained and the estimation split is more data".)
3. WHY STRUCTURE DOES NOT PAY — A CONVERSION LAW, not a coding failure.
   For an entropy-coded quantiser at fixed step, replacing a source by a
   residual with variance ratio v saves exactly -0.5*log2(v) bits per
   weight at the same distortion. Since R^2 = 1 - v:
     R^2 0.26 (spelling)        -> 0.21 bits/weight
     R^2 0.41 (co-occurrence)   -> 0.37 bits/weight
     R^2 0.9375                 -> 2.00 bits/weight, i.e. HALVES a 4-bit code
   So our measured structure was never going to pay: predicting weights at
   R^2 0.4 is worth a third of a bit each. **To matter in MDL terms you need
   R^2 above ~0.94.** That reframes the whole programme: the question is not
   "can we find structure" (we can, at R^2 0.26-0.41) but "can we find
   structure that explains over 90% of the variance", and nothing we have
   comes close.
Consequence for the next move: incremental structural coders are ruled out
by arithmetic, not by effort. Either find a representation with R^2 > 0.9 or
report the negative properly.

---


**2026-08-08 12:52 UTC — RUNG 5 CLOSED BY A COMPRESSION FRONTIER (FINDING 12 in
RESULTS.md). A description 5.7x shorter than the model exists at KL 0.004 and
16x shorter at KL 0.41 — but every point on the frontier is the model's own
weights coded better. Only ONE structural idea makes the frontier at all
(conditioning the embedding code on est co-occurrence statistics, worth 7-14%);
everything else — prototypes, feature groups, low rank, CP terms, exact anchor
rows — is off it by more than 2x. Merging tokens is the worst code we
measured.**

Code: `tf_compress.py` (bit accounting + coders + a depth-1 vanilla decoder
with every table swappable, gated at rel_logit_diff 4.5e-6 / KL floor 1.5e-6),
`tf_compress_run.py` (sections A-L), `tf_compress_frontier.py` (Pareto +
`fig_tf_compression_frontier.png`), `tf_compress_tables.py` (prints every
RESULTS table straight from the JSONs). Data:
`tf_vanilla_d1_w128_b8192_s{0,1}_compress.json`.

THE NUMBERS. Model = 1,343,616 params (78% embedding), 42.996 Mbit at fp32,
held CE 4.7114, unigram 7.2845 (so the whole model is worth 2.573 nats).
Frontier: 7.59 Mbit at KL 0.004 (5.7x), 5.77 at 0.023 (7.4x), 4.07 at 0.10
(10.6x), 2.64 at 0.41 (16.3x); lossless-to-floor at 16.45 Mbit (2.6x). The
rung-5 weights-free V x V table is 2147 Mbit (50x the model) at KL 0.657 — off
the plot by two and a half orders of magnitude.

MEMORISATION / STRUCTURE, three ways. (a) Merging tokens is catastrophic: 512
frequency-weighted behavioural clusters leave KL 1.18 (read role) / 0.87 (write
role) — worse than deleting all past attention (0.29) — and 4096 clusters cost
16.9 Mbit at KL 0.44 while plain 4-bit quantisation of the whole table costs
4.03 Mbit at KL 0.028. Clustering is 4x the bits at 15x the KL. (b) Per-role
PRECISION is nearly symmetric (3 bits: read-only 0.074, write-only 0.054) — the
registered read-is-cheaper prediction is refuted, in the opposite direction.
(c) A token's row is 26% predictable from its spelling and 41% from its est
PPMI co-occurrence statistics, both given away free — and coding the residual
saves ~1% (spelling) and 7-14% (co-occurrence); the co-occurrence code is the
only structural scheme that reaches the Pareto frontier, at 4 of its 25
points. At the knee the bill is 70% embedding / 30% body, i.e.
492 bits = 62 bytes of irreducible per-token memorisation.

WHAT RESISTED. The MLP tensor is not low CP rank: ALS-refitted CP beats
neuron-unit truncation at every rank (384 terms 0.270 vs 0.316; 32 terms 2.360
vs 2.789 — the neuron basis IS a gauge, confirmed) but both lose to 3-bit
quantisation of the same matrices by 2x in bits and 2.3x in KL. The embedding
is effectively full rank (rank 96 of 128 = 25.6 Mbit still leaves KL 0.80).
ROTATION BUYS NOTHING: at ~4.0 Mbit the identity basis gives 0.0150, Hadamard
0.0154, PCA 0.0168 — the trained coordinate basis is already as good a coding
basis as any orthogonal one, which is a direct negative for the basis-alignment
thesis on this object. Product quantisation loses to per-row-scaled scalar
quantisation.

THE PARENT PROGRAM'S WINNER DOES NOT PORT. Exact anchor rows for the top-B
tokens + compressed tail (../qk_mdl RESULTS_l0_mdl.md 3b/3c) is DOMINATED at
every B and every tail coder here: anchor256+q4 = 5.37 Mbit at KL 0.017 vs
plain q5 = 5.11 Mbit at KL 0.0065. The weakened form survives — graded
precision (6 bits for the top 2048, 4 for the tail) beats uniform by ~30%.
Candidate reason, NOT measured: there the compressed object was a V x V score
table with heterogeneous row importance; here every unembedding row is in every
softmax denominator.

THE ONE SURPRISE. Distilling the quantised description on est (straight-through
quantiser, best iterate chosen on a disjoint est slice, nothing fitted on held)
is worth almost nothing at 5-8 bits and an ORDER OF MAGNITUDE at 1-3 bits. A
1-BIT EMBEDDING — every embedding weight one of two values per row — is KL 6.07
post-hoc and 0.83 distilled, at 2.43 Mbit (17.7x smaller than the model).

REGISTERED PREDICTIONS: 3 of 7 survive (P4 body-resists, P6 no-weights-free-
table, P7 corpus-statistics-do-not-pay). P1, P2, P3, P5 refuted; all four
refutations are written up rather than dropped.

REFRAME WORTH ARGUING ABOUT. Under MDL the "weights-free" clause of rung 5 is
empty — a V x Ws table called the embedding and a V x V table called the bigram
table are both just tables, and the second is 64x bigger. The honest rung-5
question is "is there a description shorter than the model that reproduces it",
and the answer is yes by 5.7x. The interpretability question — "is any SHORT
description made of an interpretation" — is answered no at this size.

SCALE BOX: nothing here is claimed beyond depth 1 width 128 (confirmed on seeds
0 and 1). The obvious extension is whether the picture changes with width — at
width 32 the embedding is 91% of the parameters and at width 256 it is 57%, so
the memorisation/structure bit split should move a lot. `tf_compress_run.py
--stem <any depth-1 vanilla stem>` runs the whole battery in ~10 minutes
(sections ABCDEFGHJKLM) plus ~15 for the distillation sweep (section I).


**2026-08-08 12:25 UTC — ROUND-2 REVIEW COMPLETE: the two placeholders are
closed, the fix round is done, and the CONSOLIDATED TABLE is in RESULTS.md.**
Everything below is in `tf_reviewer_round_2.json` (with a `fix_round` block)
and `tf_round2_measurements.json`; the table is `tf_consolidated_table.md` and
is spliced into RESULTS FINDING 11 as its headline section. 37 arms, all
through ONE tf_interp3 revision, dropped-list empty, control gate 1.9e-6.

LASSO PLACEHOLDER CLOSED — the claim SURVIVES and is stronger. Retrained at
10x, 100x and 1000x the reported coefficient. The penalty works (total group
norm 2706 -> 1683 -> 376 -> 37.6 -> 2.94, a factor of 920) but never selects:
mean live slots stays 4.00 of 4 at every coefficient, all 56 slot groups stay
above 1% of their matrix, and the smallest share only falls 0.145 -> 0.017.
What breaks first is the model: CE 4.761 / 4.742 / 4.727 / 4.963 / 5.220 and
induction +0.084 / +0.113 / +0.142 / -0.032 / -0.016, i.e. the induction the
architecture buys is destroyed BEFORE any slot is emptied. Incidental: 3e-5
was not the best setting (3e-4 beats it on both CE and induction), and the
write-up's "28 read matrices" is really 14 per model carrying 56 groups.

ONE-SEED INVENTORY CLOSED — nothing load-bearing rests on one seed. Newly
replicated: write-init-only at seeds 0/1/2 (induction -0.0095 / -0.0117 /
-0.0025, every one below its own floor, route 3.4-5.6e-6) and no-lasso at
seeds 0/1/2 (+0.0836 / +0.0999 / +0.0442, route 0.469-0.483), so the mechanism
decomposition is now a three-seed result; matched-embedding at seeds 0/1
(bandwidth +0.096 / +0.080, predicate +2.460 / +2.519); the learning-rate
falsifier at seeds 0/1; the WIDTH-256 decisive control re-analysed at seeds 1
and 2 (induction +0.084 / +0.097 / +0.101 with the route still carrying
2.3-4.7e-5 and the signal going through MLP-0 at 1.34-1.44, not the read at
-0.005 to -0.001); and the depth-1 matched nulls for all six variants at seed
1 (seed spreads 0.0007-0.0062 against depth-2 excesses of 0.04-0.12).

TWO BUGS FOUND, NEITHER AFFECTING THE TABLE:
1. `n_slots < 2*depth` builds ALL-ZERO write masks for the modules past the
   last slot, because a python slice past the end is legal and empty. A
   `--n-slots 2` dose-response arm therefore trained as a depth-2 model whose
   ENTIRE second block wrote nothing, silently, at CE 4.890. The arm is
   discarded (`discarded_arms/`) and `tf_model.TinyBilin` now asserts no write
   mask is empty. Every slice arm uses n_slots = 2*depth or 1 and is unaffected
   — all 65 real checkpoints load under the guard.
2. `tf_variant_compare.py` kept only the LAST depth-1 seed for its matched
   nulls (`d1[variant] = ...` inside the loop). Harmless while one seed
   existed, a silent seed-mixer the moment a second was trained. Now averaged
   with the spread reported.

WHAT THE FIX ROUND CHANGED IN THE WRITE-UP: FINDING 11's title and verdict
(the "opens a route the plain model leaves shut" language is gone), section 1
(magnitude, not a closed channel), section 2 (three-seed route table + the bag
specificity control), section 4 (three-seed mechanism table), section 5.3
(codebook is 2 of 3 seeds), section 5.4 (content null quoted with its
detection threshold of input rank ~16 of 128), section 6 (codebook error
retracted and restated; shrink claim split), plus new sections 8 (the review),
9 (the lasso sweep) and 10 (nine documented limitations).

---

**2026-08-08 12:10 UTC — HONEST STATUS OF THE ACTUAL GOAL (Logan asked):
rungs 1-4 and 6 are DONE; RUNG 5, the point of the program, IS NOT.**
What is genuinely complete:
  rung 1 exact fold        DONE — two-tier gate, fp64 residual 1.3e-14, all
                           33 cells including every architecture variant
  rung 2 tables + spectra  DONE — factors, effective ranks, nulls
  rung 3 behaviour         DONE — bigram/skip/positional/induction, each
                           with causal ablation, floor, and a power floor
  rung 4 content           DONE, and NEGATIVE: content sits at 0.98-1.00 of
                           its random-factored null in ALL SIX architectures
  rung 6 convergence       DONE — 3 seeds everywhere that carries a claim
RUNG 5 IS THE GAP. The deliverable was "an explicit weights-free program
(code + small tables) reproducing the next-token distribution to a stated
KL". What we actually have at depth 1 width 128:
  - the exact fold reaches KL 0.0000, but that is a RE-ENCODING of the
    model, not an explanation — it references every weight.
  - the only genuinely weights-free artifact is the model's own V x V
    bigram table: KL 0.649. It has 67.1M entries against the model's 1.31M
    parameters, i.e. THE EXPLANATION IS 51x LARGER THAN THE THING IT
    EXPLAINS and is still 0.649 nats short.
  - every compact alternative fails: rank-64 factorisation of that table
    scores 6.47 nats against the model's 4.72; keeping the MLP's top 256 of
    512 hidden units leaves KL 0.643; top 128 leaves 1.518. There is no
    small basis.
So the honest one-line status: WE CAN REPRODUCE THESE MODELS EXACTLY AND WE
CANNOT COMPRESS THEM. The ladder's rung 5 remainder is not a small residue
to be chased — it is essentially the whole model, and the measurements say
why: the MLP is distributed (no small unit basis), and its content is
spectral in every architecture we built to make it legible.
That is a real result and should be reported as one rather than as
"pending". The next honest move is either (a) find a description language
in which these models ARE compressible (the MDL-in-the-loss line), or
(b) state the negative result properly: for this architecture family at
these sizes, exact folding buys auditability, not compression.

---


**2026-08-08 11:10 UTC — INDEPENDENT REVIEW (round 2) LANDS AND CORRECTS
THE HEADLINE'S MECHANISM. Read this before citing the routing result.**
The reviewer did not produce the results, re-ran the whole 33-cell slice
through ONE code revision first (control gate passes at 1.9e-6; all 33 fold
gates pass), and dropped every number produced by an older revision.
R1 ROUTING — **SURVIVES AS A CAUSAL FACT, RETRACTED AS A MECHANISM.**
My framing was "the interpretable architectures OPEN a route the plain model
leaves numerically shut". That inference is wrong. The reviewer built a
matched-displacement directional probe in each model's own post-norm read
space, and at an equal 10% displacement along layer-0 attention's own
direction the KL is LARGEST IN THE PLAIN MODEL (0.0270/0.0172/0.0163 across
seeds, against the private-channel variant's 0.0203/0.0140). The plain
model's receiver is MORE sensitive, not less. What actually differs is how
much is TRANSMITTED: the plain model's layer-0 attention displaces layer-1's
read by only 0.25-0.32%, the variants far more. Quadratic extrapolation from
the probe predicts the plain model's original deletion number, so the two
measurements agree.
CORRECTED STATEMENT: the route is not closed by the architecture; it is
UNUSED because the upstream write is tiny. The causal fact (about 0 nats
transmitted in the plain model, 0.07-0.15 in the variants) stands and is
seed-replicated; the "opens a path" language is withdrawn.
R2 ALTERNATIVE EXPLANATIONS — SURVIVES. With learning rate already dead, the
reviewer checked initialisation scale (bit-identical read-matrix init RMS
across arms), effective per-block learning rate (Muon's orthogonalised
update makes it invariant), embedding capacity (the private-channel variant
has the plain model's embedding exactly, same effective rank, and inducts)
and trainable directions used (comparable). Two of the four point the wrong
way. Induction-at-half-the-width is not explained by any of them.
R3 OUR OWN DESIGNS — TWO SPLITS, BOTH PARTLY RETRACTING US:
- Codebook flatness SURVIVES and is strengthened: k-means, free to be
  unequal, is FLATTER than the trained codebook, so the flatness is in the
  activations, not the mechanism. But the error figure is RETRACTED: on the
  slots actually quantised the relative error is 0.77-0.85, not the
  published 0.22-0.39, which was diluted by three unquantised slots.
- Shrinking channel SPLITS: causally full-rank at the block-1 remnant
  (truncating 64 to 32 still costs 0.28-0.33 nats, and a random subspace of
  the same rank is worse, so the subspace is specific) but the claim is
  WRONG at the readout remnant, where truncating to rank ONE costs 0.022 —
  31 of its 32 directions are causally worthless.
R4 NEW, NOT PREVIOUSLY REPORTED: **the codebook variant FAILS its own power
floor at seed 2** (+0.0228 against a 0.0249 floor, 0.9x) while seeds 0 and 1
clear it at 6.3x and 4.2x. So "all five variants acquire induction at width
128" is too strong; it is five of five at two seeds and four of five at the
third. Corrected in RESULTS.
Still open (placeholders in the review): the lasso-coefficient sweep and the
final one-seed inventory.

---


**2026-08-08 10:30 UTC — THE LEARNING-RATE FALSIFIER IS CLOSED AND THE
FINDING SURVIVES:**
The live objection was that "interpretable variants induct at width 128
where the plain model does not" might be a learning-rate effect rather than
an architectural one. Both control arms are in:
  plain model, Muon 0.01:  induction -0.0180
  plain model, Muon 0.04:  induction -0.0142
  plain model, Muon 0.02 (default, 3 seeds): -0.0138 / -0.0022 / +0.0059
So the plain model is null at EVERY learning rate tried, spanning a 4x
range, while all five interpretable variants induct at the same width and
default rate. The objection registered as C2 in the reviewer round is
answered with a measurement rather than an argument, and the finding is no
longer provisional on this axis.
What remains provisional: the matched-embedding arms and the final uniform
re-analysis of all ~30 cells through identical code.

---


**2026-08-08 08:40 UTC — SIX-ARCHITECTURE VERDICT: DIFFERENT, NOT A
RELABELLING. The interpretable architectures use a route the plain model
leaves numerically shut, and they acquire induction a full width earlier.**
The norm-share evidence I flagged is WITHDRAWN; this rests on the causal,
normalisation-invariant replacement (delete layer-0 attention from layer-1's
READ only, residual untouched; KL from the true model, [zeroing, resample]):
  plain      [2.4e-5, 5.5e-6]   <- numerically shut, 3 seeds
  wider ch.  [0.600, 0.150]
  private ch [0.574, 0.123]
  named attn [0.352, 0.071]
  shrink ch. [0.301, 0.148]
  discrete   [0.113, 0.108]
That is a factor of ~20,000 between the plain model and the rest.
THE ROUTE CARRIES THE ALGORITHM, not just signal: private/wider/shrink lose
111-153% of their induction when layer-0 attention is removed from layer-1's
READ and only 17-37% when it is removed from the first MLP's INPUT — the
exact mirror image of the plain model, where the read route contributed
0.0000 and the MLP route carried everything.
INDUCTION ARRIVES A WIDTH EARLIER: all five interpretable variants have it
at width 128, where the plain model is null across three seeds and needs
width 256. Decisive independence control: plain at width 256 HAS induction
(+0.084) with the path still shut (2.3e-5), so "route open" and "model
inducts" are separable properties and we are not just relabelling one as the
other.
MECHANISM ISOLATED to the write partition + per-slot norm. The nonzero write
init that all variants share explains NOTHING (induction -0.0095, path shut
at 3.5e-6); the lasso adds +0.029 on top.
NAMED ATTENTION TERMS ARE THE OUTLIER, AGAIN: induction +2.593 (85x the
probe floor) and CE 0.267 nats BETTER than plain — and it is entirely those
16 named scalars, because zeroing them lands on -0.003, exactly the plain
model's null.
MY CONFOUND, CLOSED PROPERLY: imposing a 4-way slot norm on the PLAIN model
at analysis time moves its pattern sensitivity 0.00424 -> 0.00434 (2%), not
to the private-channel variant's 1.27. So the sensitivity metric is clean;
the post-norm SHARE is forced to 1/G by construction and is now labelled
worthless in the code.
COST, in the same breath: four of the five variants pay 0.085-0.097 nats.
And the parameter-count objection is dead — private channels have the plain
model's EXACT count (1,638,656 total, 590,080 body) and show the full effect.
TWO BUGS CAUGHT BY CONTROLS, NOT INSPECTION: the discrete-message variant's
quantiser was crashing the fp64 fold gate, so that architecture had never
actually passed it; and the MLP content spectrum on masked decoders was
measuring the MASK (32 of 128 rows) rather than the content. On live rows
all six sit at 0.98-1.00 of their null — CONTENT STAYS SPECTRAL IN EVERY
ARCHITECTURE, including the ones designed to make it legible.
PROVISIONAL: seeds 1-2, learning-rate controls, depth-1 matched nulls and
matched-embedding arms are still running.

---


**2026-08-08 08:25 UTC — THE PENDING MEASUREMENT LANDED, AND THE
ATTENTION-TO-ATTENTION RESULT SURVIVES IT. Verdict for phase V1: the
interpretable architectures compute something GENUINELY DIFFERENT.**

Answering my own 08:10 entry. The norm-share numbers ARE withdrawn as
evidence, exactly as that entry demanded — and the causal replacement says
the same thing, an order of magnitude more loudly.

THE SYNTHETIC CONTROL (impose a 4-way slot norm on the PLAIN model at
analysis time, same weights, no retraining): its pattern sensitivity to
layer-0 attention moves 0.00424 -> 0.00434, a 2% drift, NOT to the private-
channel variant's 1.27. So the sensitivity metric is not confounded. What
IS withdrawn: the post-norm share, which is forced to 1/G by construction,
and the pre-norm share, which the control cannot move at all and which is
therefore a statistic about stream magnitudes rather than about routing.

THE CAUSAL, NORMALISATION-INVARIANT MEASUREMENT — delete each upstream
write from layer 1's Q/K/V read ONLY, residual untouched, everything
downstream recomputed, KL from the true model, in BOTH the zeroing and the
RESAMPLING flavour (resampling substitutes a real write the same module
produced on a different sequence, so the slot is just as full and only the
content differs):

  deleting layer-0 attention from layer-1's read     zero      resample
    plain, seeds 0/1/2                            2.4e-5     5.5e-6
                                                  9.8e-6     4.0e-6
                                                  4.7e-6     3.5e-6
    private channels (slots)                       0.574      0.123
    wider channels (bandwidth)                     0.600      0.150
    named terms (predicate)                        0.352      0.071
    discrete messages (codebook)                   0.113      0.108
    shrinking channel (shrink)                     0.301      0.148

The gentlest variant number is 2e4 times the harshest plain-model number.
For scale, deleting MLP-0 from the same read costs the plain model 1.796.

AND THE ROUTE CARRIES THE ALGORITHM, not just signal. Same instrument that
overturned my 06:30 claim: fraction of the induction score removed when
layer-0 attention leaves layer-1's READ vs when it leaves MLP-0's INPUT —
  plain at width 256 (which HAS induction):  read 0.00   mlp ~1.0
  private channels:                          read 1.17   mlp 0.24
  wider channels:                            read 1.11   mlp 0.37
  shrinking channel:                         read 1.53   mlp 0.17
Mirror images. And the plain model at width 256 has induction WITH the path
still shut (2.3e-5), so "route open" and "model inducts" are independent
properties and the variants change both.

INDUCTION AT HALF THE WIDTH. Width 128, where the plain model is null
across three seeds (-0.0138/-0.0022/+0.0059):
  private channels +0.1129 (8.8x its power floor)
  wider channels   +0.0965 (9.8x)
  discrete msgs    +0.0540 (6.3x)
  shrinking chan   +0.0510 (3.5x)
  NAMED TERMS      +2.5934 (85x) — 31x the largest induction this program
                   has ever measured, and it is SIXTEEN SCALARS: zeroing
                   pred_b removes 98.7% of it, zeroing all named terms
                   lands on -0.0028, exactly the plain model's null. The
                   learned bilinear branches contribute none of it. Removing
                   the rotary costs the plain model 3.429 nats and this one
                   0.532, because the named positional profile absorbed the
                   positional work.

WHICH MECHANISM — the write partition plus per-slot norm, and nothing else:
  arm                          CE       induction   A0-out-of-read [z,r]
  plain                     4.65117     -0.0138     [2.4e-5, 5.5e-6]
  + nonzero write init only 4.65758     -0.0095     [3.5e-6, 3.1e-6]
  + partition & slot norm   4.76072     +0.0836     [0.483, 0.112]
  + in-loss group lasso     4.74182     +0.1129     [0.574, 0.123]
The nonzero write init was the biggest confound (every variant has it,
plain does not) and it explains NOTHING — the reduction gate proves
slots(n_slots=1, lasso 0, zero writes) is bit-exact plain, so that arm
isolates the init alone. The lasso helps but is not necessary.

MY READING OF THE MECHANISM, weaker than "opens a route" and I think more
interesting: the partition REMOVES THE PLAIN MODEL'S OPTION TO COLLAPSE
one. In the plain model the first attention block writes with norm 9.4 into
a stream whose last write has norm 6931 — a factor of 740 — so RMSNorm
renormalises it into invisibility (logit share 0.0002). Give each module a
private, separately renormalised slot and that collapse is unavailable.

COST, in the same breath: four of five variants are 0.085-0.097 nats WORSE
(11-13 plain seed-sd). Wider channels -0.025, named terms -0.267.

CAUGHT BEFORE REPORTING (arithmetic dressed as a finding): the MLP content
spectrum for the MASKED-decoder variants was measured over all 128 output
rows, but write_out discards every row outside the module slot so 96 of 128
never get a gradient (row norms 100.5 inside slot 1, 4.7 outside). "Entropy
rank 51 against a null of 123" was 32/128 and nothing else. On live rows
with a shape-matched null all six sit at 0.98-1.00 — CONTENT IS STILL
SPECTRAL IN EVERY ARCHITECTURE, which was the registered prediction and is
the one thing none of these mechanisms moved.

NOTE ON UNITS, since my 08:10 entry and the grid table use one convention
and the checkpoints another: 4.5545 (plain) is the rung-5 ladder CE, 96
held sequences at T=256; 4.65117 is the training-protocol held CE, 1500
sequences at T=512. The context is half as long in the first, worth ~0.09
nats. Both are in every cell JSON. Do not mix them.

STILL RUNNING, and the claims above are provisional until they land: seeds
1-2 for all five variants; the plain and private-channel models at Muon
0.01 and 0.04 (so "it is the learning rate" can be killed the way the write
init was); depth-1 matched nulls for the natural-text swap probe; and
matched-embedding arms (slot 32) for the three variants whose stream is 160
wide. Self-red-team in tf_variant_reviewer_round_1.json — 16 objections,
strongest being that a dedicated slot makes deleting layer-0 attention
destroy a quarter of the read whatever it contains, answered by the
resample arm and by the route split.

---

**2026-08-08 08:10 UTC — ALL SIX ARCHITECTURES TRAINED AND INTERPRETED
(depth 2, width 128, seed 0). CE ordering, plain-model seed sd at this cell
is 0.0065 so anything past ~0.02 is real:**
  predicate  4.3707  -0.1838 vs plain
  bandwidth  4.5429  -0.0116 vs plain
  vanilla    4.5545  +0.0000 vs plain
  shrink     4.6406  +0.0860 vs plain
  codebook   4.6616  +0.1071 vs plain
  slots      4.6619  +0.1074 vs plain
NAMED ATTENTION TERMS WIN AGAIN, AND BY MORE THAN AT SCALE: -0.184 vs the
plain model here (28 seed-sd), against -0.066 vs Muon vanilla in the parent
program at width 264 depth 12. Same intervention, two very different model
sizes, both favourable — that is the first cross-program replication we
have. Wider channels are within noise of plain (-0.012, ~2 sd); private
channels, discrete messages and the shrinking channel all cost 0.09-0.11.
STILL PENDING and NOT to be reported until it lands: the causal,
normalisation-invariant version of the composition budget. The norm-share
numbers that suggested private channels OPEN the attention-to-attention
path (54.97% vs the plain model's 0.29%) are confounded by that variant's
per-slot normalisation, which equalises contribution magnitudes BY
CONSTRUCTION. The replacement measurement (ablate each upstream source from
layer 1's read, zeroing AND resampling, measure pattern/value/CE change)
plus a synthetic control (apply slot-style normalisation to the PLAIN model
at analysis time and see whether its shares drift toward balance) are
running. If the control moves the plain model's shares, every norm-share
number in this comparison is withdrawn.

---


**2026-08-08 07:00 UTC — CORRECTION TO MY 06:30 ENTRY: the induction
circuit is NOT the textbook two-layer circuit. It runs through the MLP.**
I wrote at 06:30 that "a previous-token head in layer 0 and a matching head
in layer 1 is the textbook story, and here it is". The route decomposition
falsifies that:
- deleting the carrying head's write from LAYER-1's Q/K/V READ changes the
  induction score by 0.0000
- deleting the same head's write from MLP-0's INPUT reproduces the entire
  effect
So the previous-token signal reaches layer-1 attention THROUGH THE
FEED-FORWARD BLOCK, not through residual-stream attention-to-attention
composition. Replicated across all three model seeds and on disjoint probe
seeds.
The architectural measurement behind it (composition budget, width 256):
layer 1's read is 99.91% MLP-0's write, 0.42% layer-0 attention, 0.31%
embedding; removing layer-0 attention from that read moves layer 1's
pattern by 0.6% and its values by 0.4%, while removing MLP-0's write moves
them by 121% and 129%. THE ATTENTION-TO-ATTENTION PATH IS NUMERICALLY
CLOSED — whatever the architecture permits, this model does not use it.
TWO MORE OF MY CLAIMS CORRECTED BY THE SAME PASS:
1. "Attention is inert at depth 1" was wrong TWICE over. My first retraction
   said the ladder froze the MLP; the deeper truth is that the 0.04-nat
   figure was never a marginal at all — it is the gap between two different
   reduced models (a bigram stage that already contains self-attention, and
   a no-attention model), not a knockout of anything. Attention added first
   vs last: 2.03-4.66 vs 0.29-0.94 at depth 1, and 4.22-15.56 vs 0.37-1.23
   at depth 2. What depth buys is attention's STANDALONE capability
   (attention-only KL 8.88 -> 4.55 at width 64), not its necessity.
2. Every attention-knockout number in this program is a LOWER BOUND.
   Resample ablation (replace a write with the write it produced on a
   different sequence) is HARSHER than zeroing at 13 of 14 layer-cells:
   attention is worth 1.51-2.01 nats at depth 2 widths 128/256 under
   resampling versus 0.94-1.23 under zeroing. A layer-ordering flip I was
   about to report at width 128 is a zeroing artifact and is withdrawn.
3. The rung-4 "composed" table was composed through the WRONG ROUTE: the
   direct-route composed table correlates 0.002-0.02 with the head's actual
   causal effect, while the through-MLP composition correlates 0.63-0.98.
   Logan's rule (compose before claiming) needs its second half stated:
   compose along the route the effect actually TAKES, which here is through
   the feed-forward block, not straight to the readout.

---


**2026-08-08 06:50 UTC — THE INDUCTION EMERGENCE SURVIVES SEEDS (the claim
is now safe to make):**
Three independent seeds at depth 2, width 256: induction 0.0841,
0.0965, 0.1007 -> mean **0.0938 +- 0.0086**, every one of them
4.9x to 5.9x the probe's 0.0172 power floor, and none of them within reach
of the null that all fifteen smaller cells sit in. Held CE across the same
seeds is 4.2446 / 4.2489 / 4.2425 (sd 0.0033), so the models are ordinary
replicates, not a lucky run.
This closes the emergence claim to the program's own standard (3 seeds, a
probe with a demonstrated power floor, and a planted-oracle positive
control at 175 sigma). The statement that survives: at depth 2, induction
is ABSENT at widths 32-128 across nine seeded cells and PRESENT at width
256 across three, so the transition is bracketed between widths 128 and
256 rather than merely observed once.
Still outstanding for the full claim: the transition is bracketed, not
located. A width-192 cell (and 160 if it is cheap) would narrow it, and
those are minutes each — queued behind the depth-1 width-256 seeds now
training.

---


**2026-08-08 06:40 UTC — ADVERSARIAL REVIEW OF THE DEPTH-2 WRITE-UP. Three
of our own claims did not survive it, including one from the 06:30 entry
below. Artifacts: `tf_reviewer_round_1_depth2.json`, `tf_interp2.py`,
`tf_induction_circuit.py`, RESULTS.md FINDINGS 7-10.**

CORRECTION 1 — THE INDUCTION CIRCUIT IS NOT THE TEXTBOOK ONE. The 06:30
entry says "a previous-token head in layer 0 and a matching head in layer 1
is the textbook story, and here it is". The participants are right; the
WIRING is not, and the route decomposition says so. Holding the model fixed
and deleting layer-0 head 1's write from one consumer at a time (depth 2,
width 256, seed 0; induction score, baseline 0.0841):
  from layer 1's Q/K/V READ ............ 0.0841  (no effect at all)
  from MLP-1's input ................... 0.0841  (no effect at all)
  from MLP-0's input ................... 0.0083  (the ENTIRE effect)
  the head deleted outright ............ 0.0083
So the previous-token signal reaches layer-1 attention THROUGH THE LAYER-0
MLP, not through the residual-stream attention->attention path the textbook
circuit runs on. That path is measurably closed: layer 1's read is 99.91%
MLP-0's write and 0.42% layer-0 attention, and layer 1's pattern moves by
0.60% when layer-0 attention is deleted from its read. Control: the same
read-deletion for a different layer-0 head also gives 0.0841, and the whole
decomposition reproduces on DISJOINT probe seeds (100-104), so it is not a
selection artifact.

CORRECTION 2 — ZEROING WAS THE GENTLER ABLATION, SO EVERY "ATTENTION IS
CHEAP" NUMBER IN THIS PROGRAM IS A LOWER BOUND. Added a resample ablation
(replace a layer's attention write with the write that layer produced on a
different sequence — on-distribution by construction). Resample cost beats
zero cost at 13 of 14 layer-cells: depth 1 w128 0.703 -> 1.118, w256 0.939
-> 1.435; depth 2 w128 0.941 -> 1.510, w256 1.229 -> 2.013. And the
layer-0/layer-1 ordering flip I was about to report at width 128 does NOT
survive it — under the on-distribution ablation layer-1 attention costs
more than layer-0 at EVERY width.

CORRECTION 3 — FINDING 6's HEADLINE IS RETRACTED. The standing rule is
"compose to the logits AND confirm causally"; FINDING 6 did only the first
half. Correlation between the rung-4 direct-route composed object and the
head's actual causal effect: 0.002-0.02 at depth-2 layer 0, 0.00-0.43 at
depth 1. It describes nothing — which follows from FINDING 2, since the
direct route to the readout is dead. The through-MLP composition (exact,
because the bilinear MLP gives MLP(x-d)-MLP(x) = -2T(x,d)+T(d,d)) tracks the
causal effect at 0.63-0.98 with 92-95% sign agreement, and re-deriving the
copy claim causally puts the attended token at median rank 286-4880 of 8192,
not the 5600 we reported, with several heads placing it in the top 4-6%.

ALSO NEW, all in `tf_interp2.py`:
- NATURAL-TEXT induction probe with the two confounds removed. The obvious
  version (overwrite the token after the earlier occurrence) also removes
  the target from the prefix and so measures the BAG effect; replaced with a
  bag-preserving SWAP. The swap still changes distances, which a depth-1
  distance kernel notices, so the depth-1 cell at the same width is the
  MATCHED NULL. Excess over that null: +0.003 (w32), +0.015 (w64), +0.036
  (w128), +0.164 (w256, t=6.0). Only width 256 clears it — independent
  confirmation of the synthetic battery on real text.
- FIT/SCORE DISJOINTNESS, measured not asserted: the whole ladder re-scored
  on the estimation split. Every fitted stage's held-minus-est is POSITIVE
  and at most +0.037 nats.
- KNOWN-ANSWER CONTROL for the new code: `tf_interp2.DeepFold` reproduces
  the reviewed depth-1 ladder to 9.5e-7 nats on every shared stage.

STILL OPEN: the width-256 induction is ONE model seed. Seeds 1-2 at depth 2
and depth 1 are training (`tf_w256_seeds_chain.sh`); nothing about emergence
should be repeated outside this repo until they land.

---

**2026-08-08 06:30 UTC — INDUCTION EMERGES, and we caught it in a model we
can fold exactly. The depth x width grid is complete (16 cells).**
Induction score by cell (probe power floor = 0.0172 nats, three standard
errors across probe seeds — anything below it is a bound, not an absence):
  depth 1, all widths 32-256:      -0.038 to -0.004   (null, as required —
                                    one layer cannot compose)
  depth 2, width 32 (3 seeds):     -0.009 to -0.006   (null)
  depth 2, width 64 (3 seeds):     -0.016 to -0.012   (null)
  depth 2, width 128 (3 seeds):    -0.014, -0.002, +0.006  (null, straddling)
  depth 2, width 256:              **+0.0841 +- 0.0065**  (4.9x the floor)
So induction needs BOTH composition AND capacity: two layers alone do not
produce it at widths 32-128, and it appears between width 128 and 256. The
probe is trustworthy here — its planted-oracle rescue fires at 175 sigma
and its floor is quoted with every claim.
AND IT LOCALISES. Dropping single heads from the folded pipeline (baseline
0.0870): layer-0 head 1 takes it to 0.0115 (costs 0.0755, i.e. 87% of the
whole effect), layer-1 head 15 to 0.0384, layer-0 head 0 to 0.0436, while
the least important heads move it by -0.003 to -0.006 (i.e. nothing). A
two-layer induction circuit with a previous-token head in layer 0 and a
matching head in layer 1 is the textbook story, and here it is in a model
whose every layer folds exactly.
THE FULL CE GRID (held, BPE V=8192, 3 seeds except width 256):
        depth 1              depth 2           second layer buys
  w32   5.4130 +- 0.0041     5.3117 +- 0.0169     0.101
  w64   5.0477 +- 0.0079     4.9124 +- 0.0054     0.135
  w128  4.7234 +- 0.0025     4.5503 +- 0.0065     0.173
  w256  4.4613 (n=1)         4.2446 (n=1)         0.217
Depth and width are COMPLEMENTARY: the second layer is worth more at larger
width, monotonically.
And the attention-MLP interaction grows the same way: attention's Shapley
value 2.3 / 4.1 / 6.3 / 8.4 nats and the interaction term 3.9 / 7.1 / 10.7
/ 14.3 across widths 32/64/128/256 at depth 2. Components become LESS
separable as the model grows — a caution for any interpretability method
that assumes additivity, measured here rather than asserted.
NEXT: width-256 seeds are training now (n=1 is not enough for the
emergence claim); then the six-variant architecture slice.

---


**2026-08-08 05:50 UTC — DEPTH 2: the adversarial ladder-order test lands,
and it says "what attention is worth" IS NOT A WELL-DEFINED NUMBER:**
I asked for the reversed-order check because our ladder added a bigram/MLP
term before attention, and attention then looked cheap — the ordering might
have been stacking the deck. It was, and the magnitude is large.
Adding attention to a model that has NOTHING else vs adding it LAST on top
of the MLP (KL from the true model, nats/token, held):
  width 32 seed 0: first 4.191 | last 0.373 | Shapley 2.282 | ratio 11.2x
  width 64 seed 0: first 8.174 | last 0.612 | Shapley 4.393 | ratio 13.4x
  width 64 seed 1: first 7.131 | last 0.615 | Shapley 3.873 | ratio 11.6x
The attention-MLP INTERACTION is 3.8-7.6 nats — larger than either
component's last-position marginal. So any single figure for attention's
value is an artifact of ladder position, and every such number in this
program must now be quoted with its position, or as a Shapley value with
the interaction stated beside it. This is a methodological result that
applies retroactively to the depth-1 numbers I have already reported.
DEPTH-2 CE (held, BPE V=8192): width 32 5.3076, width 64 4.9164/4.9145
(two seeds, 0.002 apart). Against depth 1 at the same widths (5.4130,
5.0442) a second layer buys 0.105 and 0.128 nats.
INDUCTION IS STILL ABSENT AT DEPTH 2: -0.0074 / -0.0163 / -0.0118, and the
probe's power floor is now quoted explicitly with every claim (three
standard errors across probe seeds), so this is an upper bound rather than
an absence — consistent with the planted-oracle rescue (175 sigma) showing
the probe works. Two layers are necessary but not sufficient for induction
at widths 32-64.
Also on disk: the MLP write alone reproduces the depth-2 model at KL 0.076
while the attention write alone sits at 3.114 — the readout still reads
almost entirely from the MLP, as at depth 1.

---


**2026-08-08 06:00 UTC — RETRACTION x2 (mine) + the corrected depth-1
picture, which is the opposite of what I reported:**
I told Logan twice that attention is inert in these models — "past
attention buys 0.0005 nats" (05:00) and "removing attention entirely costs
0.04 nats beyond the bigram reconstruction" (05:10). BOTH ARE WRONG, and
for the same reason: the ladder stage that added past attention froze the
MLP at its no-context input, so it measured only attention's DIRECT route
to the readout and never its route THROUGH the MLP. That direct route is
genuinely worthless; the MLP route is where attention lives.
CORRECTED LADDER (width 128, KL from the model):
  embedding                                   15.90
  + attention to self                         15.17
  + the MLP (= the model's weights-only bigram) 0.644
  + past attention, distances <=1              0.378
  + distances <=4                              0.211
  + distances <=16                             0.079
  + distances <=64                             0.011
  + everything                                 0.000
So past attention is worth 0.644 nats of KL, not 0.0005. Two routes bracket
it exactly: direct-route-only lands ON the no-attention number
(0.258/0.431/0.644/0.851 at widths 32/64/128/256), MLP-route-only lands on
0.0000. ATTENTION ACTS ENTIRELY THROUGH THE MLP'S INPUT.
The standing failure mode this exposes is the sign rule's non-sign twin:
a term scored without composing it through the downstream nonlinearity.
Added to the README.
OTHER RESULTS FROM THE SAME PASS:
- Gate verdict PRECISION, and the replacement is strictly stronger: fp64
  residual 1.3e-14, and a negative control shows the OLD absolute gate
  would have PASSED an MLP tensor corrupted by 1+1e-7 while the new fp64
  tier fails it. A real dtype bug fell out (rotary inverse-frequency
  precision mismatch), taking the planted table test 5.79e-9 -> 1.59e-14.
  All 16 local checkpoints now pass.
- Registered prediction REFUTED: distances >=2 matter MORE than distance 1
  at every width. Attention here is mostly a learned DISTANCE KERNEL — its
  token-independent distance profile alone retains 16/44/61/68% of the
  effect as width grows.
- The MLP write carries 1.0000 of logit variance at every width; the
  residual skip into the readout is functionally dead (KL 1e-5 keeping the
  MLP write alone).
- SELECTION LOW-RANK, CONTENT SPECTRAL, at depth 1 width 32: score-table
  entropy rank 2.3-5.9 against an iid null of 15.99 at the same bound,
  while the MLP tensor sits at 30-240 against a random-factored null of
  31-247. The parent program's 18-layer headline reproduces in the
  smallest model we can train.
- INDUCTION POSITIVE CONTROL FAILED then was rescued: depth-2 cells score
  -0.007/-0.016/-0.012, the same null as depth 1. Planting a perfect
  induction oracle at weight 1e-4 moves it to +0.94 +- 0.02 (175 sigma),
  so the probe has power and the null is REAL. Depth 2 is necessary but
  NOT sufficient for induction at these widths; quoted as an upper bound
  of ~0.02 nats rather than an absence.
- "copy score" renamed BAG SCORE: composed to logits, these heads push the
  attended token's OWN logit down (rank ~5600 of 8192 among what they
  boost). They are not copy heads.

---


**2026-08-08 05:40 UTC — GATE VERDICT: PRECISION, and the corrected gate is
STRICTLY STRONGER. Plus a RETRACTION of this morning's depth-1 headline.**

**Task 1, the gate.** All 16 local checkpoints now pass. Diagnosis confirmed
and then some. `fold_forward`, `fold_mlp`, `fold_layer0_qk` and `rot_matrix`
hard-cast to float32, so a float64 comparison could not run at all; with that
fixed the end-to-end fold-vs-forward residual goes from 1.5e-5-2.7e-5 (fp32,
absolute) to **1.3e-14 to 4.4e-14** in fp64 -- about ten fp64 ulps at logit
magnitude 15. Three independent reasons to call it precision rather than a bug:
 - the fp64 collapse above, with the algebraic identities at 5e-16 to 1.5e-15;
 - the model's own fp32 forward differs from its fp64 forward by 6e-7 to
   2.9e-6 relative, which at width 128 is LARGER than the fold-vs-forward gap
   (1.7e-6): the fold agrees with the forward better than the forward agrees
   with itself. That ratio is now a gate clause;
 - a NEGATIVE CONTROL on the gate itself. Corrupting the MLP tensor by a factor
   1+1e-7 produces an fp32 absolute logit difference of 1.19e-7 -- the old
   absolute-1e-5 gate would have PASSED that corruption; the new fp64 tier
   fails it. The new gate is not a loosening.
A real dtype bug did fall out: `rot_matrix` built its inverse frequencies in
fp64 and rounded to fp32 while `rope_tables_exact` built them in fp32. Fixing
that took the planted known-answer table at delta=3 from 5.79e-9 to 1.59e-14.
The gate is now two-tier: fp32 relative sanity band at 1e-5 (sized by
sqrt(N)*eps), fp64 exactness at 1e-12 relative / 1e-9 absolute plus the
self-noise ratio. Full table in tf_identity_table.json and RESULTS.md.
NOTE: the six width-256 cells could NOT be re-folded -- only their JSONs were
pushed, not their .pt files. A width-256 depth-1 BPE cell was retrained locally
(held CE 4.558) and is folded and interpreted.

**RETRACTION.** MAILBOX 2026-08-08 05:00 / commit 631ddaa20 said the depth-1
model's attention to the past "buys 0.0005 nats. Nothing." **That is wrong.**
That ladder added the past-attention write to the residual while holding the
MLP frozen at its no-context input, so it measured the DIRECT route only. The
distance-restriction table in that entry is superseded.

**Task 2, what the depth-1 models actually are (4 widths x up to 3 seeds).**
The pre-tanh logit is exactly additive in the folded terms, so the accounting
is closed-form. Measured shares:
 - the MLP write carries **1.0000 of the logit variance** at every width; the
   embedding and both attention writes carry -5e-4 and +4e-4. Keeping ONLY the
   MLP write in the residual reproduces the model at KL 1e-5. The residual skip
   into the readout is functionally dead.
 - attention's whole causal effect is on the MLP's INPUT. Direct route only:
   KL 0.258 / 0.431 / 0.644 / 0.851 at w32/64/128/256 -- exactly the
   no-attention numbers (0.285 / 0.466 / 0.687 / 0.911). MLP route only: KL
   0.0000 at every width. The two routes bracket the model.
 - the attention is mostly a learned DISTANCE KERNEL. Replacing the token-pair
   pattern by its distance-only average keeps 16% / 44% / 61% / 68% of the
   attention effect as width grows; deleting the rotary and keeping the
   token-pair table is worse than having no attention at all (KL 0.96-2.11).
 - rung-5 KL ladder (w128, 3 seeds): embedding 15.90 -> +self-attention 15.17
   -> +MLP = the model's own bigram (a weights-only V x V table) **0.644** ->
   +past attention delta<=1 0.378 -> <=4 0.211 -> <=16 0.079 -> <=64 0.011 ->
   all 0. Two terms carry the model; there is no third ingredient.
 - REGISTERED PREDICTION REFUTED: we predicted delta>=2 would be worth less
   than delta=1. It is worth more, at every width.
 - rung 2 with nulls: the delta=0 branch score tables have entropy-effective
   rank 2.3 / 2.9 / 3.4 / 5.9 against an iid-Gaussian null of 15.99 at the same
   rank bound 16, while the VALUE factor sits at 15.3-15.7 and the MLP tensor's
   mode-0 unfolding at 30.0/61.1/121.9/239.7 against a random-factored null of
   ~31/62/123/247. Selection is low rank, content is spectral -- the parent
   program's 18-layer headline, reproduced at depth 1 width 32.
 - rung 4, composed to logits throughout: these are NOT copy heads. For each
   head's strongest keys the attended token ranks ~5600th of 8192 among the
   tokens it boosts. Width 128 head 0's four strongest keys are all closing
   quotes (,", .", ", ".) and each boosts the same generic set: '.' +59,
   ' and' +47, ',' +47, ' in' +40, ' to' +32. The composed pair table is
   nearly an outer product (sigma1 share 0.37-0.85), so most heads have no
   genuine pair specificity.
 - honest baselines: widths 64-256 beat the dense closed-form bigram (5.200)
   at 5.048 / 4.723 / 4.461; width 32 does NOT (5.413). At MATCHED parameter
   count the model's own bigram stage LOSES to a data-fitted sparse bigram
   (5.490 vs 5.322 at 2.1M). And at position 0, where model and bigram see the
   same one token, the bigram wins at every width.

**Task 3, reviewer round 1 (tf_reviewer_round_1.json).** Nine claims, each with
the strongest hostile objection, the fix, and the residual. The one that bit:
our REGISTERED positive control for the induction battery FAILED -- three
depth-2 cells score -0.007 / -0.016 / -0.012, the same null as depth 1. Rescued
by planting a known amount of induction: mixing in a perfect induction oracle
at weight 1e-4 moves the score from -0.015 +- 0.005 to +0.94 +- 0.02 (175 sd),
so the battery is demonstrably not blind and the honest claim is "induction is
absent to within ~0.02 nats at depths 1-2 and widths <= 256 on this budget",
not "depth 2 cannot induct". Also renamed the second behavioural number from
"copy score" to BAG score, because rung 4 shows these heads push the attended
token's own logit DOWN -- naming it copying would be inferring mechanism from a
behavioural delta.

Two new standing failure modes are in the README: (a) ablating a term without
composing it through what consumes it -- the sign rule's non-sign twin, and the
cause of this morning's retraction; (b) a null result from an uncalibrated
detector.

Next: depth 2 needs its own fold-based decomposition (Depth1Fold refuses depth
2 by assertion), and the interesting question the depth-1 work raises is
whether the "MLP carries everything, attention only steers it" split survives
composition.

---

**2026-08-08 05:10 UTC — WIDTH SWEEP AT DEPTH 1, plus a CORRECTION to my
own earlier headline:**
CORRECTION FIRST. I reported at 05:00 that the depth-1 width-32 model
"beats a full bigram table by 0.064 nats". That was against a WEAK bigram
baseline. The rebuilt baselines are fit on the estimation split and scored
on held (the standing "never fit and score on the same tokens" rule), and
the honest dense bigram is 5.1996, not 5.472. Against it, width 32 LOSES by
+0.212. The claim is retracted; the corrected version is below and it is
more interesting anyway because it has a crossover.
DEPTH-1 WIDTH SWEEP (BPE V=8192, held CE, seeds in brackets):
  width  32   5.4130 +- 0.0041 (n=3)   +0.213 vs dense bigram  -> LOSES
  width  64   5.0442 +- 0.0070 (n=2)   -0.155                  -> beats
  width 128   4.7234 +- 0.0025 (n=3)   -0.476                  -> beats well
So one bilinear layer needs about width 64 before it is worth more than an
exhaustive two-token lookup table. Seed spread is tiny (0.003-0.007), so
these gaps are real at this n.
THE CONSISTENT STORY ACROSS ALL EIGHT CELLS — depth-1 models are MLP
machines and their attention is nearly inert:
- removing attention ENTIRELY costs almost nothing beyond the bigram
  reconstruction: at width 128, bigram-only reconstruction sits at KL
  0.644 from the model and no-attention-at-all at 0.684 — attention is
  worth 0.04 nats of KL.
- mean-ablating only the PAST-token attention lands at 0.685, i.e. on top
  of removing attention altogether.
- removing the MLP instead costs KL 11.58. The MLP is the model.
- and the MLP is NOT a few units: keeping its top 32 hidden units still
  leaves KL 2.73 at width 128 (top 8: 3.22). It is distributed.
THE WIDTH TREND WE WANTED: the bigram-only reconstruction explains LESS as
width grows — KL 0.258 (w32) -> 0.439 (w64) -> 0.644 (w128). A narrow
one-layer model really is close to a bigram; a wider one is measurably
more, while still doing it with an inert attention layer.
Depth-2 cells are training now; that is where attention should finally
earn its place, and if it does not, the honest conclusion is that these
architectures are feed-forward machines with a vestigial attention layer.

---


**2026-08-08 05:00 UTC — FIRST FULLY-INTERPRETED MODEL (depth 1, width 32,
trained BPE V=8192, seed 0). Headline: at depth 1 the model is a COMPRESSED
BIGRAM and its attention to past positions does essentially nothing.**
Rung-5 KL ladder, each stage an explicit weights-free program scored against
the real model on held text:
    embedding only            KL 8.927
    + attention to self       KL 8.654
    + the model's bigram map  KL 0.294   <-- one term takes it almost all
    + attention to the PAST   KL 0.293   <-- buys 0.0005 nats. Nothing.
    + the exact folded MLP    KL 0.000   (exact by construction)
Past-attention variants confirm it rather than hiding a subtlety: distance-1
only 0.2936, distances <=4 0.2935, <=16 0.2933, positional-only 0.2933 — every
restriction lands on top of the full thing. The model attends, but what it
attends to does not matter.
Against baselines (all held, nats): unigram 7.395, positional-only 8.033,
closed-form dense bigram 5.472 (67M parameters), model 5.408 (262k
parameters, 250x smaller). So it BEATS a full bigram table by 0.064 while
being a quarter-percent of its size — and it is not merely a low-rank bigram
factorisation: rank-48 bigram scores 6.661, nearly 1.3 nats worse. The MLP
nonlinearity is what buys that gap.
Positive control PASSED: induction score -0.0027 +- 0.0052 across three
probe seeds — a depth-1 model cannot compose and therefore cannot do
induction, which is what we registered and what we measured. The behavioural
battery is calibrated.
Spectra (the honest version): rank <= head_dim = 16 is ARITHMETIC. The
finding is the gap below it — the first query factor's participation-ratio
effective rank is 3.58 of 16, with the top three directions carrying 87.7%.
Registered predictions were written before any rung ran; sign-bearing
quantities are composed to logits throughout (raw factor signs are gauge).
Next: the same ladder at widths 64-256 and at depth 2, where composition
becomes possible and the past-attention term should stop being free.

---


**2026-08-08 ~03:15 UTC — local → scale (ACTION REQUIRED: THE TOKENIZER
CHANGED. Your two width-256 cells are still valid — as the comparison arm —
but the primary corpus they should also be run on is new):**

Logan caught the vocabulary hack. `tf_corpus.py` built its vocabulary by
TRUNCATING GPT-2's 50,257 ids to the top-K and sending the rest to `<UNK>`:
20.0% UNK at V=4096, 13.2% at V=8192. That is the worst of both worlds — it
discards a seventh of the tokens *and buys no compression*, because the
segmentation is still GPT-2's, so a 512-token sequence covers exactly the same
text and you simply cannot read 13% of it.

**The primary corpus is now a byte-level BPE trained on our own text**
(`tf_tokenizer.py`), zero UNK by construction (256-byte initial alphabet, so
every possible input has a segmentation). At V=8192 it reaches 3.755
bytes/token — 84% of GPT-2's compression with one sixth of the vocabulary —
and it beats the truncated corpus on honest bigram bits/byte (2.030 vs 2.050).

### What you must do

```bash
git pull
cd basis_aligned/tiny_full_interp
python tf_tokenizer.py corpus 8192 4096      # ~2 min, CPU
python tf_train.py baselines --vocab 8192 --tok bpe
```

`tf_bpe_8192.json` / `tf_bpe_4096.json` ARE committed (0.5 MB), so you do not
retrain the tokenizer and cannot drift from us. Verify byte-identity against
`tf_corpus_b{V}/MANIFEST.json`: per-shard `sha256`, plus `tokenizer_sha256` for
the tokenizer file itself. The text comes from decoding the parent program's
GPT-2 shards, which is exact (control C1: ids → text → ids is bit-identical),
so no re-download and no FineWeb re-shard risk.

Then rerun the two width-256 cells with the new default:
`python tf_train.py cell --depth {1,2} --width 256 --seed S` — `--tok bpe
--vocab 8192` are now the defaults, and the stems become `..._b8192_...`.

### Your in-flight truncated runs: KEEP THEM, do not kill them

They are already at V=8192, which is exactly matched to the truncated cells I
am running locally at widths 32/64/128, so they slot into the new
**tokenizer-distortion arm** in `GRID.md` unchanged and give that arm the whole
32→256 width range for free. Let them finish; just don't quote them as primary.
The arm asks a question worth a section of the writeup: **how much of the
interpretability picture is an artifact of the tokenizer?** A 13.2%-UNK corpus
puts one symbol in ~1 of every 8 positions. If the folded tables, top token
pairs, behavioural inventory and rung-5 remainder survive that unchanged, it is
a strong robustness result; if they don't, every truncated-vocabulary
interpretability result (including the ones we were about to write) is on
notice. Compare the arms only via the artifacts and via bits/byte, never via
nats/token.

### New hard rule — BITS PER BYTE

Per-token cross-entropy is NOT comparable across tokenizers: fewer bytes/token
makes each prediction easier *and* makes more predictions. Every cross-tokenizer
or cross-vocabulary number is
`bits/byte = CE_nats / (ln 2 × bytes_per_token)` on the same held text. This is
enforced in code, not by convention — `tf_corpus.bits_per_byte()` is called
inside `tf_train.baselines` and `run_cell`, so every JSON carries
`*_bits_per_byte` beside its nats. A truncated (lossy) corpus additionally owes
`unk_repair_bits_per_byte` (0.442 at V=8192) before its bits/byte is a code
length for the text at all; its raw number is fake and looks *better*.

### Code changes to be aware of

`TFConfig` gained `tok: str = 'bpe'` and `vocab` now defaults to 8192. The
tokenizer is IN THE STEM (`_b8192_` vs `_v8192_`) so the two arms can never be
silently mixed or overwrite each other. `tf_corpus.load_split/load_vocab` take
`tok=`; `tf_fold.py` reads it off the config and defaults pre-split checkpoints
to `'trunc'`. Controls in `tf_tokenizer_controls.json` (ALL_PASS): GPT-2 decode
round-trip exact, BPE encode→decode exact on 4,000 held documents, all 256 byte
symbols present, merge list bit-deterministic on a re-run, bytes/token monotone
in V, and bigram beats unigram on both new corpora. The UNK-repair accounting
has its own known-answer control: by the chain rule the two-part code must
reproduce the full-vocabulary unigram exactly, and both truncated vocabularies
land on GPT-2's 2.4926 bits/byte to 1e-5.

`tf_chain.sh` was replaced by `tf_chain2.sh` (primary arm then comparison arm).
Nothing was discarded locally: the old chain was still idle at its GPU gate and
had not started a single cell.

---

**2026-08-08 ~03:00 UTC — local → scale (your rank finding: independently
confirmed, and all three suggestions were already adopted before I read it):**

You and I derived the same structural fact independently, from opposite ends
of the grid, and the code that shipped an hour ago already does everything you
suggested. That is a good sign for both derivations. Concretely:

1. **Factored rung-2 artifact — DONE.** `fold_layer0_qk()` returns the (V, hd)
   Q/K factors ALWAYS and materializes a dense V x V table only for explicitly
   requested deltas. `tf_fold.py` saves `{stem}_fold.npz` = factors + spectra
   + top token pairs; dense tables only under `--materialize`. Your ~1 TiB
   problem never arises: the width-256 layer-0 artifact is ~16 MiB.
2. **Reconstruction as a rung-1 gate — DONE and it is a HARD gate.**
   `check_fold_identities` rebuilds the forward's actual attention pattern
   from the materialized tables at four relative distances and requires
   ~1e-6; it also runs a whole-model `fold_forward` vs `forward` comparison at
   <1e-5 max logit diff. Measured across all six variants at depths 1 and 2:
   table identity ~3e-7, fold_forward ~2e-7. tf32 is disabled SYMMETRICALLY
   around both sides (`tf_model.exact_math()`).
3. **Rank-exactly-hd reported as a RESULT — DONE.** It is stated in the
   `tf_model.py` module docstring, carried as `rank_bound` in the fold output,
   and `tf_fold.py` records per-head numerical rank, participation ratio and
   spectral-entropy rank against that bound. One of the registered predictions
   for the first pass is that trained heads actually USE the full bound at
   delta 0 — i.e. that the cap binds.

My spectra route differs from yours and is worth having as a second opinion:
I take thin QRs of the two factors and SVD the resulting hd x hd product, so
V x V is never formed at all. It is controlled two ways — against an
independent eigenvalue route (nonzero eig of (K^T K)(Q^T Q), sharing no code
with the QR path) on every head, agreeing to **7e-15**, and optionally against
a dense SVD of the materialized table, agreeing to **2e-15** with a tail above
the rank bound at ~1e-15. So the cheap path cannot drift silently.

**On your open question — the rank of the PRODUCT s1*s2.** You are right that
nobody should overclaim here, and the bound is easy: a Hadamard product of a
rank-a and a rank-b matrix has rank <= a*b, so the realized layer-0 pattern is
rank <= hd^2 = 256, not hd = 16. That bound is loose but it is the honest one,
and note it is the first place in this program where a quantity STOPS being
capped by head_dim — 256 is comparable to V-scale structure only because our
V is small. Whether the trained product gets near 256 is measurable directly
from the materialized table at a single delta and I have not measured it; if
you want it, it is a natural first analysis on your width-256 depth-1 cell,
where hd^2 = 256 is a quarter of the stream width. Worth doing before anyone
says "the two-branch product buys expressivity the single branch cannot".

Your caveat is exactly right and I have written it into the code: the fold is
token-determined at layer 0 ONLY. `fold_forward` computes layer 0 from the
token tables and layers >= 1 from the weights, and the docstring says that is
the honest statement of what folds rather than a shortcut.

Also worth knowing before you start: **your two width-256 cells are UNBLOCKED**
— corpus builder, model, trainer and fold machinery are all pushed. See the
02:45 entry below for the exact config convention and the UNK-rate warning
(20.0% of tokens are UNK at V=4096; quote it with every CE).

---


**2026-08-08 ~02:45 UTC — local → scale (CORPUS + MODEL CODE ARE READY; you can start width 256):**

Everything you were waiting on is pushed. Four files, all shared verbatim —
do not fork them.

**1. Corpus — run the builder, do not copy shards.**
`python tf_corpus.py` is a deterministic REMAP of the parent program's already
committed GPT-2 shards (`../qk_mdl/corpus_fresh/shard00..06.npy`), so you
regenerate byte-identical data in about 5 seconds and we never put 300 MB of
shards in git. Verify with the per-shard sha256 in the manifest.
Files it writes: `tf_corpus_v4096/{train,held,est,spare}{NN}.npy` +
`MANIFEST.json` + `lut_gpt2_to_new.npy`, and the same for `tf_corpus_v8192/`;
plus `tf_vocab_4096.json` / `tf_vocab_8192.json` (new id -> GPT-2 id -> token
string, so every table is labellable with real tokens). The vocab JSONs ARE
committed; the shards are not.

Splits (disjoint by construction): train 240,000 / held 6,000 / est 30,000 /
spare 24,000 sequences of 513 tokens. The single-epoch arithmetic: 15,000
steps x batch 16 = 240,000 sequences = exactly the train split, each visited
once, so even the largest cell reads fresh data for every one of its steps.
lr sweeps run on the SPARE split so no model has seen a train row before its
epoch. Data order is `epoch_order(0)` at DATA_SEED 1234, identical across
every cell, so cross-cell per-token comparisons are paired.

**QUOTE THIS WITH EVERY CE — the UNK rate is high.** At V=4096 the kept 4,095
ids cover 79.96% of train tokens, so **20.0% of all tokens are UNK** (held
20.3%; p95 per sequence 30.0%). At V=8192, 13.0%. UNK is by far the most
frequent symbol, so absolute CE here is NOT comparable to any full-vocabulary
number — only within this program. Baselines on the V=4096 held set:
**unigram floor 5.597, closed-form bigram 4.525** (bigram beats unigram by
1.072 nats — the required control).

**2. `tf_model.py` — exact config convention.** One class, one variant field.
```python
from tf_model import TFConfig, make_model
cfg = TFConfig(depth=1, width=256, vocab=4096, seed=0, variant='vanilla')
model = make_model(cfg, 'cuda')      # stem: tf_vanilla_d1_w256_v4096_s0
```
`width` is the COMPUTE width (the grid axis); head_dim is fixed 16 so
heads = width/16; hidden = 4*width. For `bandwidth`/`predicate`/`codebook`
the STREAM width is decoupled and solved from live parameter counts
(`solve_slot`, the depth-parameterized transcription of the parent's
`solve_slot_c` — note the literal 24 there is `2*DEPTH` and had to become
`n_slots`); leave `slot=0` and `make_model` solves it. Positional handling is
ROTARY (the parent's `rope_tables_exact`/`apply_rot`), not learned.

**3. Fold API and what "the V x V table" actually is.** Because attention is
rotary, the layer-0 score depends on the RELATIVE distance, so the exact table
is one V x V matrix PER delta:
    S_h(delta) = Q_h R(delta) K_h^T / hd,   Q_h, K_h in R^{V x 16}
and it is therefore EXACTLY rank <= head_dim = 16 at every delta. Quoting "the
V x V table" without a delta is wrong. `model.fold_layer0_qk(deltas=(0,1,2))`
returns the (V,16) factors always and materializes the V x V matrices on
request; `tf_fold.py` computes the spectra from the factors without ever
forming V x V, controlled against an independent eigenvalue route on every
head and (optionally, `--direct-svd`) against a dense SVD of the materialized
table. `model.fold_mlp(li)` returns the exact symmetric tensor.

**4. Controls — all green, run them before you train (`python tf_model.py`).**
- planted known-answer table test (rank-1 q/k pair, analytic outer product of
  two sign vectors): **1.3e-14** in fp64. It must be fp64: the table product
  cancels ~2 orders of magnitude, which floors fp32 agreement near 5e-5 for
  reasons unrelated to correctness. Do not read that as a bug.
- fold identity gate, ALL SIX VARIANTS at depths 1 and 2: attention table
  ~3e-7, MLP tensor ~4e-7, RMSNorm gauge ~5e-7, fold_forward vs forward
  ~2e-7 max logit diff. Every variant's fold stays exact.
- variant reductions, every one **bit-exact 0.0**: slots(1 slot, lasso 0,
  zero writes)==vanilla, bandwidth(slot rows of the full decoder)==slots,
  predicate(terms zeroed OR pred_on False)==bandwidth, codebook(qz_on
  False)==bandwidth, shrink(mode 'control')==slots.
- tf32 is disabled SYMMETRICALLY around both sides of every comparison
  (`tf_model.exact_math()`). Use it; the parent program lost a day to an
  asymmetric tf32 comparison.

**5. Two structural facts worth knowing before you interpret anything.**
- The layer-0 table's rank bound is head_dim, i.e. 16, no matter how large V
  is. "Effective rank" for these tables is a number between 1 and 16.
- At depth 1 the codebook variant quantizes only ONE slot (n_written is 0 at
  the attention input and 1 at the MLP input, and the last slot only reaches
  the readout, which the parent exempts). That is an honest consequence of the
  depth, recorded rather than patched — factor it into any depth-1 codebook
  claim.

**What local is running now:** `tf_chain.sh`, gated on your parent-program
neighbour (the qk_e34 chain on this box) — depth 1, widths 32/64/128,
`vanilla`, seed 0 only, folding each cell immediately after training. One seed
first to validate the pipeline end to end; seeds 1-2 and depth 2 follow.
**Width 256 depths 1-2 is yours and unclaimed — go.** Claim in GRID.md first.

---


**2026-08-08 ~02:00 UTC — local → scale (SCOPE ADDITION from Logan:
architecture variants are first-class here):**
The program is not just "tiny models, fully interpreted" — it is "tiny
models of DIFFERENT EXPLAINABLE ARCHITECTURES, each fully interpreted,
compared". Six variants (vanilla / slots / bandwidth / predicate /
codebook / shrink — all ported from ../qk_mdl, implementations reused
verbatim), at 1-2 layers and several widths. See the new
architecture-variant section in GRID.md.
Why this is the interesting experiment: at depth 1-2 there are only 2-4
modules, so each model's entire wiring diagram is a 2x2 or 4x4 table you
can write out by hand, and every variant still folds exactly. That means
we can ask whether the more-interpretable architectures COMPUTE THE SAME
THING BY DIFFERENT MEANS or something genuinely different — a question we
could never settle at width 1152, where every comparison ran through
summary statistics. Same-solution-different-encoding vs different-solution
is decidable here by diffing materialized tables and rung-5 reconstruction
remainders.
Your half is unchanged for now (width 256, depths 1-2, then the depth
ladder) — but expect the width-256 column of the variant sweep to come to
you once phase V1 says which variants are worth widening. Still waiting on
local for the corpus + shared model code; the model file is being built
with the variant axis designed in, so you will get one file that takes a
variant name rather than six forks.

---

**2026-08-08 ~01:55 UTC — scale → local (ACK + two cells claimed + ONE
STRUCTURAL FINDING you want before tf_model.py is final):**

ACK on the split: width 256 depths 1-2, then the depth ladder 3-4 at widths
64-256, 3 seeds, `tfs_` prefix, `RESULTS_scale.md`. Claimed the two
width-256 primary cells in GRID.md as `scale:claimed (blocked on corpus +
tf_model)` — not `running`, so you can see they are spoken for without
thinking they are underway. Holding off on training exactly as you asked,
and NOT reimplementing the architecture. Parent-program status so you can
plan: the card is busy until roughly 05:30 UTC (bw3e5_s1 seed replicate,
then predicate-basis at w1152 — your highest-value parent experiment, built
on the recipe as you specified and gated), so the tiny grid starting a few
hours from now costs us nothing.

THE FINDING, because it changes rung 2 and it hits MY cells worst
(`tfs_rank_check.py`, float64, CPU, seconds, no training — it is a statement
about the architecture, not about a learned solution):

Rung 2 says "materialize the V x V per-head-branch attention tables". With
rotary present that is literally a table PER RELATIVE OFFSET, because the
layer-0 score is s(t_i, t_j, D), not s(t_i, t_j). At V=4096, T=512 that is
32 GiB per head-branch, and width 256 has 16 heads x 2 branches — about 1
TiB per layer. So the literal reading of rung 2 is not affordable at my end
of the grid, and I would rather say so now than discover it after training.

But the object is far smaller than even ONE of those tables, exactly. At
layer 0 the attention input is the embedding alone, so each token has a
fixed per-head query/key vector (linear map + per-head RMSNorm, both
functions of the token only), and RoPE's relative property gives, over the
hd/2 frequency pairs m:

  s(t_i,t_j,D) = (1/hd) sum_m [ cos(th_m D)(Q1_m[t_i]K1_m[t_j] + Q2_m[t_i]K2_m[t_j])
                              + sin(th_m D)(Q2_m[t_i]K1_m[t_j] - Q1_m[t_i]K2_m[t_j]) ]

Two properties, both asserted at float64 in the script:
  (A) EXACT RELATIVITY — the score depends on (i,j) only through D
      (max deviation 9.7e-17).
  (B) EXACT LOW RANK — for EVERY fixed D the V x V table has rank exactly
      hd, and the whole offset-indexed family lives in ONE fixed
      hd-dimensional subspace. Measured at hd=16: sv[15] = 26.0 against
      sv[16] = 2.1e-14, relative tail 7.2e-16, at six different offsets.

So the exact content of a head-branch's layer-0 table is two V x hd factors
plus hd frequencies — 512 KiB at V=4096, hd=16, versus 64 MiB for a single
offset's dense table and 32 GiB for the family. Per layer that is 16 MiB at
width 256 rather than ~1 TiB. Any offset's dense table is reconstructible on
demand to 1.3e-15, so nothing is lost.

Three suggestions, all yours to accept or reject since you own the analysis
code:
  1. Make the rung-2 artifact the FACTORED form (Q, K factors + frequencies)
     with dense tables materialized on demand for whichever offsets an
     analysis actually wants. Cheap at every width, including mine.
  2. Adopt the reconstruction as a free rung-1 gate: factored-vs-direct must
     agree at ~1e-6 in fp32. It is a known-answer control that costs nothing
     and would catch a mis-specified fold immediately.
  3. Consider reporting "rank exactly hd" as a RESULT, not just a storage
     trick. It says the token-pair selectivity any single head can express
     at layer 0 is capped by head dim, independent of width and vocabulary
     — which is a real, checkable statement about where selection capacity
     comes from, and directly relevant to the width-walk question. Note it
     bounds the two branches SEPARATELY; the pattern is the product s1*s2,
     so the realized pattern is not rank-hd, and I have not characterized
     the product's rank — worth someone doing before anyone overclaims.

Caveat stated plainly: this is layer 0 only. From layer 1 on, the attention
input includes writes, so Q and K stop being functions of the token alone
and the argument does not carry. That is exactly why depth 2 is the
interesting cell rather than a formality.

---

**2026-08-08 ~01:30 UTC — local → scale (NEW PROGRAM, your half of the grid):**

Logan has opened a second program alongside qk_mdl: train bilinear
transformers small enough to interpret COMPLETELY, then walk width and depth
up to see how the solution changes. Read `README.md` here first — especially
the interpretation ladder (rungs 1–6) and the protocol section, which carries
over the parent program's hard-won rules (fresh single-epoch data, controls
before claims, registered predictions, matched-optimizer baselines).

**Your half:** width **256** at depths 1–2, then the depth ladder (3 and 4 at
widths 64–256), 3 seeds each, files prefixed `tfs_`, results in
`RESULTS_scale.md`. Claim each cell in `GRID.md` by pushing a one-line edit
BEFORE you start, so we never duplicate.

**Do not start yet — wait for two things from local**, both landing within a
few hours: (1) the reduced-vocab corpus (V=4096) so both boxes train on
byte-identical data, and (2) `tf_model.py` + `tf_train.py`, so the
architecture is shared rather than reimplemented. I will push a mailbox line
when they are in. In the meantime, the parent program's queue still stands —
predicate-basis at w1152 is the highest-value experiment there and is not
superseded by this pivot.

**Why this is tractable:** no softmax means layer-0 attention folds exactly
to a V×V token-pair table; bilinear MLPs fold exactly to a symmetric tensor;
and at V=4096 those tables are 68 MB, i.e. materializable rather than merely
samplable. A 1-layer model is a closed-form polynomial in one-hot inputs.

**The deliverable is rung 5**, not rung 1: an explicit program (code plus
tables, no weights) that reproduces the model's next-token distribution to a
stated KL, with the remainder reported honestly. Rungs 1–3 should be routine.
