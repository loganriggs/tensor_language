# The first two layers of bilin18, decomposed: a consolidated account

*(Ticks 161–215. All numbers are held-out change in cross-entropy on the standard
307,200-prediction FineWeb audit unless noted; every claim cites its RESULTS section.
Companion interactive artifact: the "Layer-0 QK" page; master details:
`RESULTS_l0_mdl.md`; methodology of measures: its Section 8.)*

## 1. The one-paragraph story

The attention pattern computation of this 546-million-parameter bilinear-attention
transformer is, for its first two layers, a **small, explicit, causally calibrated
circuit**: per-token lookup tables compressed into sparse archetype dictionaries,
plus one low-dimensional context signal. Layer 0's pattern is *exactly* a token-table
computation (7,418 megabits raw, reproduced within +0.0023 nats by 493 megabits of
dictionaries and anchors, §3); its nine heads decompose into validated grammatical
archetype classes with a 128-fold spread in required capacity (§5). Layer 1's pattern
is ~99% token-identity-driven (static mean-residual tables cost +0.027 versus +2.70
for removing the pattern, §6a) and speaks a different archetype vocabulary —
boundaries, auxiliaries, quote pairing, and subword fragments (§6b). Between them
sits a Bilinear MLP that is **dense in every internal basis we and the literature
know how to test** (neurons, weight rank, composed token pairs — §7a, 7b, 7g, 7h),
yet whose functional product for the next layer's pattern is a **ten-dimensional,
mostly-token-identity signal** (§7c), priced end to end: a 16-dimensional-per-channel
oracle interface costs +0.0113; generating it from named codes reaches +0.032; and
the ungenerated remainder was proven to be **composed multi-token state** —
concentrated in one head's keys, on lexical-continuation text, at attention offsets
zero to two — that no token-identity lookup of any resolution can produce (§7k–7m).

## 2. The instruments (what made the claims trustworthy)

1. **One binding metric.** Every decomposition, of every object, ultimately answers
   to held-out ΔCE on one frozen audit set. Fit metrics (reconstruction R², moment
   residuals, CP relative error) are *gates and tools*, never conclusions — three
   separate decouplings proved they cannot be trusted alone: moment mass versus
   function (§5m ρ≈0.1), fit-R² versus audit (§7j), and code-width versus function
   (§7i).
2. **Known-answer tests before real data.** Every fitter (dictionary trainer, dense
   CP, sparse CP, asymmetric CP) passed planted-recovery gates before touching real
   objects; every failure in the program's history was caught by a control, never by
   inspection. Two solver families and one null statistic were rejected this way.
3. **The corrected null.** Sparsity/structure claims are scored by fitting on a
   measure-matched null and transplanting onto the real object (λ refit). The naive
   two-tensor comparison passed things it shouldn't (and failed things it shouldn't:
   the tick-180 verdict on heads 0/4 was overturned when the control head exposed the
   statistic, §5g).
4. **Measures are calibrated, not assumed.** Structure appears only under a measure
   (unigram, manifold); which measure predicts causal damage is itself
   object-dependent and was measured (§8): expected output magnitude for heads
   (ρ=0.87), plain weight fraction for channels (ρ=0.83), mechanism mass for nothing.
5. **Data-sufficiency ladders.** The claims that survived (manifold rank ≈ 10) were
   re-verified at 16× the estimation data; the ones that were sample artifacts
   (quiet-head rankings) were caught by exactly this check.

## 3. Layer 0 (function): the compression frontier

Because layer-0 attention reads only normalized embeddings, its entire query/key
computation folds into exact per-token factor tables. The frontier (exact-moment
training objective + exact rows for ~256 high-attribution anchor tokens): **+0.0023
nats at 493 megabits; +0.0008 at 1,393** against 7,418 raw — with the error
exposure-concentrated (top-50 tokens carry half), which the anchor mechanism exploits.
Six clean negatives shaped this frontier (reader co-adaptation, co-occurrence
weighting, coherent rotary, regrouping, tail weighting, and the original joint
training — later *reversed* by the true warm start, §5i, which now improves every
head it touches).

## 4. Layer 0 (mechanism): the archetype ledger

Per head, the third-moment tensor Σₜ pₜ·k¹ₜ⊗k²ₜ⊗vₜ — the exact object through which
a bilinear head's average effect factors — decomposes into nonnegative CP
**archetypes**: rank-one detector–writers over token classes. All nine heads validate
(corrected nulls, 3-seed stability 0.94–1.0, cross-corpus core cosines 0.98–0.99).
The vocabulary is case-invariant grammatical scaffold: {the}, {a/an}, {of}, {and},
punctuation and delimiter families, pronouns and copulas — with per-head minimal
capacities spanning **32 to 4,096 atoms** (a 128-fold range; two features per token is
the sweet spot) and the two big-inventory heads preferring the mode-separated
asymmetric form (shared feature space, unshared detectors, §5–5g). Causal
calibration: the archetype span carries 73–88% of whole-head causal load on the heads
that matter; head 3 alone is ~60% of the layer (+0.078); the archetype *directions*
are not privileged per unit of pattern removed (§5n) — the ledger's value is
descriptive, compressive, and predictive, not causal-geometric.

## 5. Layer 1: the port, and what attention becomes one level up

The token-identity approximation ports wholesale: mean-residual tables (with a
shrinkage estimator that cures rare-token moment noise, §6b) reproduce layer 1's
pattern within +0.027 of a +2.70 total — and layer 1's attention is ~27× more
causally important than all of layer 0's, with massive inter-head redundancy (single
heads sum to 5% of the joint effect) and measured *self-repair* (freezing the pattern
while ablating an upstream head more than doubles the damage, §6c). Its archetypes:
sentence and document boundaries, auxiliaries, quote pairing, determiners with number
words — and on the causally dominant head 1, **mid-word subword fragments**: a
word-completion head whose static part is trivial (32 atoms) and whose substance is
context. Layer-0 head 3's determiner signal is broadcast to every layer-1 head
(one-third of its effect through layer-1 pattern formation).

## 6. The Bilinear MLP: dense engine, narrow window

The layer is a written-in-weights CP tensor of rank 4,608. Inside it: nothing our
tools can grip — flat neuron usage (top 128 of 4,608 carry 6%), brutal prune frontier
(half the neurons cost +0.030), readers touching all neurons, mild weight-rank
compression at best, and composed token-pair tensors whose corrected nulls TIE the
real fits in both pure-weight and unigram measures (§7g–7h: the "measure is the
message" lesson, third occurrence). Outside it: a ten-effective-dimensional realized
channel into each layer-1 reader (verified at 524k positions), 84% embedding-squared
in the exact weight-space block decomposition, with layer-0 head 3 the top
cross-partner in 18 of 18 channels (§7f). The interface is priced by oracle
(+0.0113 at 16 dims/channel; +0.0009 at 64) and half-generated from named codes
(+0.032), with the generator search closed by a complete hypothesis walk (§7i–7m):
all architectures tie at matched parameters; the shortfall is input information;
window and bigram token-identity lookups are null; the remainder is **composed
multi-token state**, concentrated in head 1's keys on lexical continuations.

## 7. What we would tell someone starting layer 2

Expect the port to work (deeper layers accumulate more composition, so budget the
context share to grow); run the measure calibration before trusting any new fit
metric; expect the same triad — cheap static skeleton, small context interface,
dense-but-narrow mixers — and treat every "dense inside" verdict as an invitation to
find the narrow window rather than to decompose the engine. The methodology that
survived 55 ticks unchanged: fold exactly where the architecture permits, gate on
planted tests, validate on corrected nulls, calibrate measures against causal probes,
verify at data scale, and let the eighteen controlled negatives be findings.
