# A four-ledger per-layer decomposition of a no-softmax bilinear transformer

*Consolidation draft — 2026-07-30. Numbers are the post-adversarial-review figures only.
Sources: PLAN_per_layer.md, RESULTS_l0_mdl.md §32–§103, LOG.md. Left for parent review; not committed.*

## Abstract

We decompose **bilin18** — an 18-layer, 9-head-per-layer transformer whose attention is
softmax-free (each pattern is a product of two bilinear score branches, `(q1·k1)(q2·k2)/d²`,
causal and unnormalized) and whose feed-forward blocks are bilinear MLPs — layer by layer, on four
deliberately separated ledgers: **Representation** (can the layer be rewritten exactly as an
analytic tensor network?), **Substitutability** (can its computation be replaced, causally, by a
compact analytic interface without hurting the model?), **Function** (what job does it do?), and
**Meaning** (can that job be named as code that survives a held-out substitution gate?). The point
of keeping the four apart is that a layer can score well on one and fail another, and conflating them
manufactures false understanding; several of our own headline claims were retracted for exactly this
reason. The completed result (for the exhaustive four-ledger sweep, on the single model bilin18;
generality is shown separately for specific arcs and for the completeness boundary): **every layer 1–17
is representationally exact** (analytic gauges to ~1e-6, and its two attention branches are genuinely
two-factor — an exactness that is strictly bilinear, unavailable to a nonlinear model), **~99.9%
causally substitutable per layer at the margin** through PCA-bottlenecked analytic interfaces (a
*fidelity* statement, not a compression/description-length win — the interfaces reference the full
weight tensors as exact restrictions; the **per-layer marginal** cost —
replace one layer, all others left exact — is 99.95–99.998% of the uniform-ceiling headroom, every
layer, with paired standard errors and fair nulls; the honest **cumulative** whole-model cost of
replacing attention and feed-forward at *all* layers at once is ≈ +0.080 nats ≈ 98.95% of headroom,
~20× the marginal loss — whole-model attention bottleneck +0.0475 = 99.38%, MLP chain +0.0329 =
99.57%), **functionally
mapped** into three families with a per-head selection census and a feed-forward family map, and
**semantically characterizable as "nameable selection over spectral content"** — the model runs
nameable selection programs (only for the copy/induction/match head family) over a graded,
memorized, non-class-nameable content dictionary that is spectral at all 18 layers. Meaning is the
frontier and stays hard: functional content is even bounded-nameable at only four sites.

Beyond the per-layer sweep, we turn the exact decomposition into a **generator**: an unsupervised
discovery loop that follows single paths and, because a single linear proxy is unreliable, verifies
each candidate with a **type-specific causal test** — ten detectors spanning class-boost, copy,
suppression, redundant-distributed (via joint ablation), positional, byte-fragment, remap, and a
causal class-level detector. Two results discipline the picture. First, **completeness, measured**: a
coverage ledger shows the named circuits carry about eleven percent of the total causal headroom (about
forty-four percent of the single-path-expressible mechanism), the model is roughly three times
super-additive so most computation lives in combinations, and the largest single uncharacterized bucket
— the MLP-layer-1 hub — is **irreducibly distributed** (no single direction is nameable, three-quarters
of its effect appears only under joint removal, a signature that survives an adversarial control against
low-rank layers, random directions, and the neuron basis), marking the boundary where single-direction
interpretability stops. A bounded, a converged, and a ten-times-more-data sparse autoencoder on that
hub sharpen the boundary rather than crossing it. Reconstruction and causal explicability turn out to be
decoupled: with enough data a dictionary *does* reconstruct the hub well (held-out variance-explained
rising from ~0.72 to ~0.85) and *does* recover clean monosemantic *variance* features the orthogonal
view missed (up to 26 of 32 nameable versus zero of 32 for singular directions) — yet across every
reconstruction fidelity from 0.69 to 0.85 it never recovers the *computation*: no feature is individually
load-bearing and all of them together capture only about two percent of the hub's causal effect, with a
collective-encoding control intact. So dictionary methods *name* and *reconstruct* the early hub but do
not *explain* it; its causal mechanism is collective at every level and every fidelity tested. This whole boundary is
architecture-general: the super-additivity, the high-rank basis-aligned feed-forward tail, and the
irreducibly-distributed early hub all replicate on a second bilinear model and on a conventional softmax
SwiGLU transformer (two to three-and-a-half times super-additive, an early hub with zero-to-one of
thirty-two directions nameable) — and so does the dictionary decoupling itself: the identical sparse
autoencoder test on the softmax hub reconstructs at 0.85, names twenty-two of thirty-two features, and
still finds zero load-bearing (a dictionary explains about ten percent of that hub's causation versus two
percent on the bilinear one — a difference in degree, not kind). A final coalition search hardens the
verdict: joint ablation of searched feature groups up to five hundred twelve members — by energy, gradient
attribution, co-activation, decoder clustering, and alignment with the causally-sufficient subspace — never
concentrates even one percent of the hub's effect into a modest subset. A final characterization explains
why: the hub's output is a **redundant distributed code** — keeping half its amplitude costs one percent of
its effect, and deleting any random *half* of its dimensions costs under two percent, with breakdown only
past roughly ninety-percent removal. Every large fragment is sufficient; no small fragment is necessary. So
deletion-based attribution — which measures necessity — is structurally blind here, and its universal ~zero-
to-two-percent readings are a property of the code, not of the tools. The concrete methodological handoff:
explaining such hubs requires sufficiency-based analysis of what fragments carry, not necessity-based
ablation. Second, the discovered circuits
are **useful**: they generalize (the distributed-class-mover phenomenon reproduces on a conventional
softmax SwiGLU transformer), and a circuit found unsupervised and verified as a genuine algorithm — a
final-block capitalization selector — is a **calibrated, placebo-controlled control knob**, though its
selection logic sits upstream, so it resists surgical override. Throughout, every headline is held to a
held-out substitution/ablation gate with paired standard errors and an adversarial review before it
stands; the record includes roughly a dozen retractions and softenings that discipline enforced.
A final movement resolves the hub story in the architecture's own coordinates: in *sufficiency* mode the
hub is compact and hierarchical (its map restricted to a one-hundred-twenty-eight-times-smaller core
keeps 96.7 percent of function), its exact stream-pair terms give a five-term named anatomy with a
causally dead input row and a bigram-table term identified as such, and propagating perturbations
through folded compact cores repairs the unreliable linear proxy (rank correlation 0.43 to 0.81–0.93,
sign essentially perfect) — the failures of deletion-based attribution were a property of the hub's
redundant code, and the fold supplies the sufficiency calculus that reads through it. A closing
composition arc extends the fold across layers: folding components into their consumers makes the gating
partner of every conditional-looking component explicit — block 3 reduces to three of twenty-one exact
cross-layer terms, with fresh attention multiplying block 2's square as its context-supplying mixer;
block 1 becomes a readable token-indexed family of affine operators, 93.6 percent causal — and the
mid-stack distributed region, previously stateable only as a redundant collective code, now admits a
*partial* compressed model: a privileged message basis with three named message states, measured partial
redundancy (matched deletions about twenty-five percent cheaper than an unprotected code), and its gates
named in the fold algebra where they resolve at all. The super-additive whole-model compression bound's
missing structure is thereby identified as cross-layer product coupling — consumption — invisible to
per-layer rank allocation.

## Method

**The four ledgers.** *Representation* is settled by two architecture identities that hold for any
weights of this class, so they are method licenses rather than findings: a bilinear MLP folds exactly
into a symmetric third-order tensor and rms-norm folds out as a scalar gauge
(`MLP(rms(x)) = D·T(x,x)/‖x‖² + bias`); the attention pattern is a quartic multilinear numerator over
four norm gauges. Both verify at ~1e-7 and license a "stream algebra" in which a layer's pre-norm
input is split into analytic streams (embedding plus each upstream component's exact output).
*Substitutability* asks whether replacing a component with a truncated analytic interface is causally
free. *Function* assigns jobs via mean-ablation knockouts and a per-head selection-predicate census.
*Meaning* is the strict gate: a candidate name is written as code from independent knowledge, then
required to beat mean-substitution on a held-out slice — only then is a mechanism "nameable."

**The per-layer driver.** A single script (`qk_layer_decomp.py`) runs three ledgers on each layer:
the representation gauge, the substitutability cost of replacing the layer's attention
(projected onto per-head PCA-64 bases) plus its feed-forward block (composed analytic fold), and the
function census. The fourth ledger is swept separately by parametrized content- and selection-gate
drivers (`qk_content_gate.py`, `qk_selection_gate.py`) verified to reproduce byte-for-byte at a
reference layer before generalizing.

**Discipline.** All audits are on a held-back FineWeb slice **FW[448:600]**, disjoint from every
fitting corpus; cross-entropy in nats is the only headline metric. Every substitutability figure
carries a **paired standard error** and a **fair null** — for attention interfaces, a random basis
of the same head-span dimensionality; for the whole-model bottleneck, both random 576-dim subspaces
and random 64-of-128 within each head's own image. Costs are reported base-relative and against the
**mean-ablation floor** and the **uniform ceiling** (ln V = 10.83 nats) so that a small ΔCE is not
mistaken for a large fraction of anything. Substitutability additionally checks the attention
symbol-fold against an honest **positional-mean floor**, because zero-ablation was found to inflate
per-layer loads 10–60× (a standing §12q correction).

**Adversarial review as a first-class step.** Three dedicated red-team passes retracted or corrected
multiple headline claims — this is the reason to trust what survives:

- The joint-substitution "72.5% gap" (and a follow-on "knob-recovery") was a **λ-scaling bug**:
  deltas were injected with unit coefficients where the correct coefficients are products of
  downstream lambdas (a ~2300× overshoot). Corrected, joint substitution is essentially free (§33).
- The editing arc was reframed twice: from "clean repoint of an induction match" (too narrow — it
  needs no match) through "brute-force injection" (too broad) to the settled **copy-head
  commandeering** description (§37f–j).
- **Greater-of-two** was deflated three times, ending as an in-context copying prior, not a
  comparison circuit (§40).
- **Subject-verb agreement** and greater-of-two had their all-attention-ablation "zero prior"
  numbers flagged as **tautologies of a balanced design**, not evidence (§40/§42).
- Retracted phrasings on record: "exposure-bias mechanism," "fully-named analytic tensor network,"
  "composition supersedes data fitting," the "KEY_cap → capitals" selection name (§46), and the
  successor "token-pointer / format-free numeric identity" framing (§35).

## Results, ledger by ledger

**Representation — exact at every layer.** The composed-fold gauge is exact to ~1e-6 for all 17
layers; the licenses hold on three attention families (two-branch, normalized-squared, softmax). One
substantive finding beyond the identities: the two attention branches are **genuinely two-factor
everywhere** (§38). Across all 162 heads the per-head correlation of the two score branches has median
0.044, mean 0.006; **zero heads exceed 0.9** and 95.7% sit below 0.5. The most distinctive heads are
strongly *anti*-correlated (L15H1 −0.78, L0H7 −0.70) — difference/conjunction detectors — and even the
most redundant reach only ~0.70–0.78. So no head collapses to a single squared branch; both branches
must be carried when decomposing any layer's selection.

**Substitutability — near-total per layer at the margin.** The per-layer driver replaces each layer's
attention (PCA-64/head bottleneck) and feed-forward (composed fold) simultaneously on FW[448:600].
These are **per-layer marginal** figures — one layer is replaced while all others are left exact.
Marginal costs range from **+0.00014 nats (layer 12) to +0.0038 nats (layer 14)** — every layer between
**99.95% and 99.998%** of the uniform-ceiling headroom, each with a paired SE and a head-span null.
Null margins vary informatively: layer 5's random-basis null is 199× the true cost (attention very
load-bearing), whereas several layers have head-span nulls near 1× (L8 1.4×, L13 1.3×, L14 1.04×,
L16 1.1×) — where attention is near-dispensable. At those near-1×-null sites "substitutable" means
**near-dispensable on general cross-entropy**, not that a compact interface reproduces a rich
computation; and general-CE substitutability is **blind to rare-but-decisive capabilities**: layer 13
is ~99.98% substitutable on general CE, yet L13H8 is the causal router for the bracket/quote circuits
below. The marginal figures do **not** add up to a free whole model: replacing attention and
feed-forward at *all* layers at once costs ≈ **+0.080 nats ≈ 98.95%** of headroom — roughly 20× the
per-layer marginal loss. At the whole-model scale, projecting every attention output onto per-layer
PCA-64/head bases (with the residual itself truncated — the strongest test) costs +0.0475 versus base
(**99.38%** of headroom), a ~0.003/layer linear accumulation, against random-576-dim nulls 100× larger
and within-head nulls 20–30× larger; the architecture-general version holds on bilin12 (+0.116) and
bilinsm12 (+0.077). Replacing the entire MLP stack causally by the composed-fold chain costs +0.0329
(**99.57%** of headroom, own null 18×). The
attention symbol-fold beats the honest positional-mean floor at **15 of 16 layers**; the lone
exception is **layer 17**, whose near-output pattern is mostly positional (§43 — flagged as differing
from §12q's full-corpus mean, which localized the loss at layer 5; the per-minibatch vs full-corpus
methodology gap is noted for reconciliation).

**Function — three families, a per-head census, and a filled feed-forward map.** The circuit atlas
collapses a seven-task battery into three families: a **category-prediction engine** (subword /
punctuation / capitalization / digit / function-word, task correlation 0.98–0.999, 90–96%
MLP-driven, concentrated in MLP0–3 — a linear next-token-category probe jumps 0.527→0.611 across
MLP0–3 and collapses to 0.510 when they are ablated); an **induction / copy fabric** (28% head mass,
built on MLP1); and a **layout/newline** outlier that the category stack actively interferes with.
The selection census marks **23 of 162 heads programmatic** (predicate gain ≥5%) under the original
8-predicate library, or 30/162 under the 12-predicate census v2; and the predicate label predicts the
head's causal specialization (with the caution of §54 — a low-R² predicate LABEL like KEY_newline can be
a census artifact, not a real mechanism). Note these "programmatic" counts (per-head predicate gain) are
distinct from the §49 simultaneous-substitution "gated-nameable selection" criterion, which is stricter
and finds gated heads at every layer except the two diffuse ones (4 and 17). The feed-forward family map (§44) closes the largest
remaining Function hole: MLP0–3 is the category engine (with **MLP1 the hub — the only block serving
the two-branch match fabric**, +0.029 match-rate on ablation), MLP4–15 is **distributed
category-refinement with no distinct family** (each block removes ≤0.014 category accuracy, cost
≤0.11 nats), and MLP16–17 is a **lexical readout** near the output. Under the 8-predicate
function census three attention layers show no head clearing the 5% threshold — layers 4, 9, and 17 —
but **genuinely diffuse under *both* predicate libraries is layers 4 and 17 only**. Layer 9 is diffuse
only under the 8-predicate census: the fuller 12-predicate selection gate finds a gated **KEY_newline**
head there (L9H8, gain 0.062). And "diffuse" here means **no surface-predicate-nameable head, not no
computation** — layer 4's attention carries a 3.3× head-span null, so it is load-bearing.

**Meaning — nameable selection over spectral content, measured everywhere.** Running the content-
nameability gate at **every layer 0–17** settles the content axis. The real evidence is the class fit:
**no single class in an independent grammar/orthography/frequency library reaches R²≥0.8 under a
step-function name** at any layer (0–3 of 576 coordinates clear the bar, layer by layer; median
class-R² 0.014–0.038 throughout) — so the spectra are **not class-nameable in this library**, though
content could still be nameable under a different, untested ontology. The class-code and spike-code
substitution gates cost ~0.000 nats everywhere, but this is a **separate fact, not added evidence for
non-nameability**: with only 0–3 of 576 coordinates codable the gate is mechanically ~0 whether or not
content is nameable, and what it records is that content is spike-concentrated and low-sensitivity.
Crucially content stays **spectral** — graded and spike-concentrated, with no such class name — even
at the lexical-readout layers 16–17; it does *not* become class-nameable near the output. So a graded, memorized, non-class-nameable content spectrum is a
**universal** property of bilin18 (§48), not a layer-0 artifact. The selection axis is the
complement: gated-nameable selection heads exist at every layer **except the diffuse layers 4 and
17**, and they are **exclusively the copy/induction/match family** (MATCH_same / MATCH_prev / KEY_*
predicates). The strongest — L2H5 (gain 0.245) and L3H8 (gain 0.314) — are the induction MATCH
predicate, the one fully meaning-verified functional claim (held-out retention 98–111%). Functional
*content* is even bounded-nameable at only four sites — layer-0 selection, the block-3 category dial,
the layer-8 successor payload, the layer-13 opener flag — and each is a control dial plus an
extractable table over a bounded input set, **not a generalizing law**: the category directions are
steerable but causally deletable (+0.0003, ≈ random), the opener flag is type-blind and leaky after
closure, and the successor payload is a per-calibrated-element table whose four genuinely held-out
elements fail to generalize (follow 0.00–0.25). Capitalization, the cleanest fifth-site candidate,
**failed the gate** (§46): mean-ablating the KEY_cap cluster retains 101–102% of the
capital-vs-lowercase margin, so whether-to-capitalize is a static prior in the readout — the cluster's
real +0.046 effect is *within*-capital discrimination, not a capital gate.

## Algorithmic case studies

Alongside the per-layer sweep we probed hand-built algorithmic tasks with a fixed discipline:
verify the behavior → mean-ablation patch → **static-prior control** → minimal circuit → red-team.
The static-prior control is the decisive first filter, but it separated cleanly on its own in only
**2 of the 5** tasks (quote §50, increment §51); it needed a follow-up demo-swap control in
greater-of-two (the first reading was retracted), was a balanced-design tautology in subject-verb
agreement, and was prior-confounded for '(' in brackets.

- **Greater-of-two digits (a negative).** Behavior looks strong (accuracy 0.986), but with all
  attention ablated a fixed profile still solves 68/72 pairs (0.944). The decisive demo-answer-swap
  control (§40) showed the "prior" is **in-context copying of the few-shot demonstration answers**
  (the ablated profile peaks on the demo answers and moves with them; zero-shot it is flat, static
  accuracy 0.444 = chance). The genuine per-pair comparison signal is only **0.387 nats of margin /
  +0.042 accuracy** (~3 hard pairs), carried jointly and diffusely by attention and feed-forward in
  series with no separable single-head attribution. No small faithful circuit exists — this joins
  addition and sorting as a deflation of apparent numeric capability.
- **Bracket-type matching (a genuine circuit).** This one is a real attention circuit (§41),
  decomposed into three parts. The dominant, causal router is **L13H8 — the v1-router** — which
  attends back to the opener and copies its **layer-0 value payload** (removing it costs ~25% of the
  working margin; removing the layer-0 value mixing collapses `[`→`]` entirely). A forward value-swap
  is decisive: giving a working `[` host the `{` value breaks type-match 100%→0%, and the identity
  control reproduces the baseline exactly. The `{`→`)` "hole" is explained as a **missing
  value-cache entry**, upstream of the router, not a router failure. This confirms the v1-router
  principle (QK decides *where*, the layer-0 value decides *what*) on a fresh task — carried
  specifically by `[`, which provably needs attention to overcome the `)` prior.
- **Subject-verb agreement (a redundant router, number locus open).** Accuracy is 1.00 including 40
  incongruent-attractor items — genuine structural agreement, and attention is architecturally
  required. The dominant head **L11H3** reads the head-noun position (weight-share 0.35 vs 0.05 on
  the attractor) and carries ~46% of the incongruent margin, **but it is redundant** — ablating any
  single component keeps accuracy 1.00. Where the *number* lives is **not cleanly localized**: the
  layer-0 value cache is globally necessary, yet swapping one position's value moves only ~17% of the
  number swing and never flips the verb, so number is a redundant/distributed code, unlike bracket's
  swappable single-position payload.

## Unsupervised circuit discovery, indexed by circuit type

The exact decomposition is also a generator: following one set of paths through it — a head pathway
or a feed-forward singular-vector output direction — should isolate an algorithm. We built a
discovery loop that ranks all 162 head pathways and 72 feed-forward directions by a *cleanliness*
score (trigger-purity times effect-purity), then causally verifies the top candidates on the
held-back slice with mean-ablation and paired standard errors. The central methodological finding is
negative and load-bearing: the linear direct-to-logits *effect* proxy is unreliable — wrong in
magnitude, sign, token-case, and, critically, wrong about single-versus-joint attribution — so it
serves only as a candidate generator and every candidate earns its keep through a *type-specific*
causal test. A single verifier does not fit all circuits; different circuit types need different
tools, and an auto-clustering pass over the ranked paths surfaced twelve families and a map of which
types the default class-based pipeline could not express. We built a causally-verified detector for
each under-served type:

- **Class-boost** heads (fire on a token class, boost a class) — verified by whether the boosted
  class's logits drop under ablation.
- **Copy / value-router** heads, re-derived unsupervised — the source-token payload, not the logits.
- **Suppression** directions — negative-logit inhibition at clause boundaries, invisible to a
  boost-only proxy.
- **Redundant / distributed** circuits — the key tool is *greedy joint ablation* with a redundancy
  ratio (joint cost over sum-of-solos) and a same-size random-set control. It resolves a puzzle the
  proxy creates: a family of copy heads each near-null in isolation is in fact a single distributed
  circuit that collapses only jointly, while a superficially similar cluster is genuinely null.
  Single-head ablation cannot tell these apart; joint ablation can.
- **Positional / structural** heads — an offset-template-versus-content-residual decomposition, with
  a built-in honesty guard for the positional envelope every head inherits from the rotary embedding
  and causal mask. It establishes that the load-bearing positional heads are fixed-offset
  (previous-token and self). Its distance-since-boundary readout is, by an adversarial power check,
  underpowered against a saturating signal — the position-0 sink head itself shows a monotone rise in
  damage over the first ~15 tokens that a raw correlation scores as zero — so the tool licenses the
  positive positional attributions but *not* a strong negative claim about the absence of a
  distance-to-boundary circuit; that question is left open.
- **Byte-fragment / orthographic-trigger** paths — an orthographic-predicate library scored on the
  decoded trigger strings, with an out-of-sample purity check that acts as an artifact pre-filter
  (rejecting overfit rare-affix fingerprints before any causal budget is spent) and a conditional
  causal contrast that confirms genuine digit- and punctuation-triggered circuits.
- **Trigger-versus-output decoupling** ("remap") paths — fire on class A, boost a different class B.
  These are exactly where the proxy is most dangerous, so the decisive step is an output-side causal
  test (does ablation suppress class B specifically at active class-A positions against an
  inactive-class-A control). Of the top candidates a minority survive; the rest are proxy artifacts,
  including twin directions that split one-genuine-one-artifact and one with the causal effect in the
  reversed sign — a vivid demonstration that only causal verification separates them.

Per-circuit magnitudes, controls, and standard errors are tabulated in the results log; several
headline figures were subjected to a dedicated adversarial red-team (position-matched confounds,
independent reproduction, negative-claim power checks) before enshrinement.

**Discovered circuits, put through the full algorithmic arc.** Two of the cleanest discovered
circuits were then run through the same verify → minimal-circuit → red-team discipline used on the
hand-built tasks, and the pair illustrates why the causal step is not optional. The digit-attending
heads split, under a copyable-versus-non-copyable dissociation, into *two distinct algorithms*: one is
a verbatim digit value-router (it copies the number it attends to; ablating it where the attended
token *is* the target costs a quarter-nat and it contributes strongly to that token's logit), the
other a source-independent next-number predictor that boosts the correct next digit without copying
any attended token, and does its work precisely where no copyable referent exists. Both clear the
static-prior floor — next-number prediction is essentially fully attention-driven, not a bigram prior.
The capitalization circuit went the other way, an honest negative: the behaviour is real and
circuit-carried (two feed-forward directions, with the punctuation head as an upstream marker rather
than a capitalizer), but the generalization red-team refutes a dedicated "capitalize at sentence
start" mechanism — ablation suppresses mid-sentence proper-noun capitalization exactly as much as
boundary capitalization (specificity ratio one). Its boundary-selectivity is entirely on the trigger
side; the output is a generic, shared capital direction implementing the corpus prior.

**A blind spot, measured.** Because the discovery ranking is proxy-seeded, we asked directly whether
it only surfaces the easy circuits. A difficulty-stratified census measured causal importance
(mean-ablation delta cross-entropy at each path's own firing positions) *independently* of the
cleanliness score, across all two hundred thirty-four paths. The two are uncorrelated (Pearson
0.006). The consequence is not mild: the thirteen low-cleanliness, high-causal circuits the loop skips
are *more* important on average than the clean winners it finds, and the single largest single-path
effect in the model is one of the missed ones. Their shared, disqualifying property is a *distributed
output* — they push an entire token class (near-uniform over thousands of tokens), which the
top-token effect-purity signal scores near zero regardless of how load-bearing they are. The remedy is
a tenth detector — a causal, class-level effect ranking that describes outputs by whole-class logit
movement rather than sharp token sets. Its score correlates with causal importance at Pearson 0.986
(versus cleanliness's 0.006), it recovers the entire missed region, and the sign of the class-summed
movement is itself a new discriminator that separates class *pushers* from *suppressors* — correcting
one of the census's own provisional labels in the process.

**One real algorithm among the biggest effects, and it generalizes.** Put through the full arc, the
model's four largest distributed effects split by a decisive test: does a direction push its class
*when the class is due*, or flat? Only one, a final-block feed-forward direction, is a genuine
context-conditioned capital selector (it pushes capital more than three times harder where a capital is
due, and ablating it hurts eleven times more there); the other three are static class-frequency priors,
in fact anti-selective. The genuine selector was surfaced *only* by the class-level detector — the
token-level tools saw the generic boosters but never the algorithm, because its output is a distributed
class push. And the whole phenomenon is architecture-general: a conventional softmax SwiGLU transformer
independently develops the same near-uniform whole-class movers, in the same final block, with the same
push/suppress structure and the same class-push-tracks-importance correlation.

**Completeness, stated honestly.** A coverage ledger partitions the model's total causal headroom
(everything the eighteen layers do, measured by mean-ablation). The named-circuit fraction depends on
the denominator and is best given as a range: the named circuits carry about eleven percent of the
*total* headroom, but about forty-four percent (rising to a ~forty-six-percent ceiling once single-path naming is pushed exhaustively) of the *single-path-expressible* mechanism (the named
numerator is near-additive; the headroom denominator is nearly three times super-additive). The gap
matters: much of the remainder is not simply mechanism we failed to look at, but computation that single
paths cannot express by construction — about thirty-six percent of the headroom is feed-forward effect
below the leading singular directions, and the model is roughly three times super-additive overall, so
most computation lives in *combinations*. The honest one-line reading is that single-path naming has
captured a large share of what single paths *can* express, a small share of the whole, and that the
whole is predominantly hard, distributed, and combinatorial. Two structural facts frame the whole program: the model is roughly two-point-nine times
super-additive (ablating the full single-path basis costs almost three times the sum of the individual
ablations), so most computation lives in *combinations* and single-path naming inherently undercounts
it; and the largest effects we *do* name are the exception, not the rule, in a mechanism that is
predominantly hard, distributed, and partly superposed.

**Naming, reconstructing, and explaining are three different things.** The natural rejoinder to
"irreducibly distributed" is to train a sparse autoencoder on the hub, so we did — a small probe, a
converged run, and a run on ten times the data — and ran the same nameability and causal tests on the
dictionary that we ran on singular directions. The result separates three properties that are easy to
conflate. *Nameability*: a dictionary crosses it — up to 26 of its top 32 features are clean and
monosemantic (pure " the" at eleven-times enrichment, commas and periods at nine-times, capitals) where
single directions scored zero of 32. *Reconstruction*: also crossable, but only with data — held-out
variance-explained ceilings near 0.72 on a small corpus (a bound we first mistook for fundamental) and
rises to about 0.85 with ten times the data. *Explanation*: **not** crossed at any fidelity. Across
reconstruction quality from 0.69 to 0.85, no single feature is individually load-bearing (zero of 32
clear the causal bar), all features together capture only about two percent of the hub's causal effect,
and a control confirms the collective signature — removing the reconstruction and removing the residual
each leave a fully-compensating complement. So a dictionary *names* and, given data, *reconstructs* the
early hub, yet does not *explain* it: its causal mechanism is collective, and — the sharp point —
reconstruction fidelity and causal explicability are decoupled, because the nameable variance axis and
the causal axis are nearly orthogonal here. The decoupling is architecture-general: the identical test on
the softmax SwiGLU model's hub gives the same three verdicts (reconstruction 0.85, twenty-two of
thirty-two features nameable versus zero for singular directions, zero of thirty-two load-bearing — with
the one honest difference that a dictionary explains about ten percent of the softmax hub's causation
rather than two percent, a difference in degree, not kind). This is a concrete boundary on what
dictionary-learning interpretability recovers, measured rather than asserted.

## Fold-first attribution: the hub cracked in the architecture's own coordinates

The resolution came from turning the exact tensor structure — used all along for representation and
substitutability — onto the attribution question itself, in four steps.

**The hub has hierarchical structure that deletion could not see.** In sufficiency mode (keep a part,
delete the complement) the hub is compact at every level. Its output: the per-position mean plus its top
144 of 1,152 principal directions restores 98.5 percent of function, forty times better than a random
144. Its *map*: reading only the top 288 input directions costs under one percent, and the joint
restriction — input-288 by output-144, a core one hundred twenty-eight times smaller than the full
tensor — retains 96.7 percent of causal function. So the earlier "irreducibly distributed" verdicts were
true only of *necessity* attribution: the computation is orderly, compact, and hierarchical, carrying a
redundant code (any random half of the space suffices; breakdown only past roughly ninety-percent
removal) that makes every deletion-based reading come out near zero.

**The architecture's own terms give the interpretable split that learned bases never found.** Because
the feed-forward block is exactly bilinear and its input is an exact sum of upstream streams, its map
decomposes exactly (gauge at one part in a million) into pairwise interaction terms with provenance.
Five named terms — MLP-0-by-attention-1, attention-1 squared, MLP-0 squared, embedding-by-MLP-0,
embedding-by-attention-1 — restore essentially the full function (+0.0019 of 5.574 nats); every term
involving attention-0 is causally dead (confirming the old stream-level fact exhaustively); the four
same-stream terms alone are ten times worse than the six cross-stream terms, so the hub is an
*interaction* device mixing context, the layer-zero transform, and the current token; and the
embedding-by-embedding term is identified as a token-conditional bigram-table correction (the current
token alone explains ninety percent of its variance). The residual redundancy survives — but among ten
named objects rather than a thousand anonymous directions.

**The same structure fixes the program's most recurring failure.** The direct-to-logits linear proxy —
wrong in magnitude, sign, and case throughout the discovery arc — fails because it deletes downstream
computation. Propagating a candidate perturbation through the downstream layers' folded compact cores
instead lifts the proxy's rank correlation with ground-truth causal effects from 0.43 to 0.81 (0.93 at
double core rank, with essentially perfect sign agreement), recovers every early-layer case
linearization zeroes out, and is certified basis-specific: random bases of the same rank collapse back
to linear-proxy fidelity. Notably, the restricted model is a poor absolute predictor yet preserves
causal *ordering* — fidelity for attribution and fidelity for prediction are different properties.

**The model-wide anatomy.** The term census over all eighteen blocks (every layer's reconstruction gated
at one part in a million, every floor validated against the prior censuses to four decimal places) shows
the compact anatomy is an early-stack property inside a clean recency-to-history pipeline: the causally
heavy early layers are two-to-five named terms (layer 2 is almost literally "square the previous block's
output"), interaction terms dominate everywhere except the ends, and each layer's own attention goes
causally dead by layer fifteen as accumulated history takes over. The same pipeline — early recency, a
mid-stack crossover, late history-reading with the layer's own attention dead, and a tenfold floor jump
at an entangled readout — replicates on the softmax SwiGLU model under an input-group intervention that
needs no bilinearity, so the depth anatomy is architecture-general. The readout itself, first flagged as
the one spot resisting term-wise decomposition, resolved under a three-hypothesis test into a
**differential pair**: one arm writes broad generic capital-and-word mass, the other writes its
context-gated near-negation (class signatures opposite at cosine −0.97), and the layer's function is
their small conditional difference — removing both arms together is *cheaper* than removing either
alone, the causal fingerprint of computation-by-cancellation. This unifies the earlier findings that the
layer is a "lexical readout," that its one genuine algorithm is a context-conditioned capital selector,
and that the model's ubiquitous class priors are individually removable: their conditional retraction is
implemented here.

**Editing through the terms.** Term-targeted steering of the differential pair delivers
understanding-driven control rather than raw power. The prior arm is a saturation-free, perfectly
monotone capitalization dial across the full range (where the earlier direction dial reversed at high
gain), and a **contrast-strength knob** — scaling the pair's net write coherently — is a new edit type
not expressible as any single direction: at double contrast it costs only +0.053 nats globally while
*improving* the model at bracket-open decisions by −0.255 ± 0.040; scaling one arm alone at the same
strength blows the same decisions up (+1.30), a direct proof that preserving the cancellation structure
is what keeps the edit clean. Honestly: for raw single-behavior specificity the earlier top-direction
dial remains sharper at every matched effect size — and the fold explains why: that direction *is* the
pair's post-cancellation output axis, concentrating the functional degree of freedom each raw term
dilutes with its half of the cancelling mass. No amplitude edit, direction- or term-level, achieves a
surgical unconditioned override; the conditioning is encoded in the terms' own activation patterns, so
re-aiming it requires editing their upstream inputs — and a final experiment confirms that this works. At
the readout (where a per-position input edit affects only that position's prediction, so collateral is
localized by construction), transplanting boundary context into the feed-forward input at mid-sentence
positions produces the capital push the amplitude dials could not (+0.028 ± 0.003, nine standard errors,
graded and monotone under partial transplants), the reverse transplant suppresses it (−0.070, fifteen
standard errors), and the zero-collateral gate passes *exactly* — bit-identical logits at every
non-edited position in every run. The editing story therefore closes as a two-sided law: amplitude edits
scale writes and cannot re-aim conditioning; input edits re-aim it surgically. One refinement the
transplant alone could see: the forceable boundary context is carried predominantly by the accumulated
*feed-forward* history group (nine times the attention-history gain, specificity ten-fold over random
donors) — the attention accumulator's vectors are nearly parallel across positions, so their
distinguishing component is small even though the attention-involving term dominates the pair's energy.
Repeating the transplant mid-stack bounds the law's scope and unifies the program's two central
mechanisms: at layers eight through fifteen the edit remains almost free of collateral (after-position
costs statistically zero, with the perturbation demonstrably propagating) but the *gain* collapses to
one-to-six percent of the readout's — the redundant code re-derives the conditioning from the many
unedited sources and overwrites the injection, exactly as it compensated deletions. The boundary of the
editing law is washout, not collateral: the same redundancy that blinded deletion-based attribution
makes the model intrinsically robust to single-point mid-stack activation tampering, and makes the
readout — where compensation has run out — the model's one true control surface.

**Honest limits of the fold program.** The single-layer compression bargain does not globalize:
restricting all eighteen blocks at once costs +1.46 nats at one-hundred-twenty-eight-fold compression
against the exact chain's +0.033, and the compounding is in fact super-additive (half the joint cost is
cross-layer interaction). Non-uniform rank allocation does not rescue it: allocating by measured
per-layer need ties uniform allocation within noise, because per-layer needs are nearly flat, variance
concentration anti-correlates with functional rank, and no per-layer schedule addresses the interaction
term. The compression failure is structural; faithful whole-model compression, if achievable, runs
through shared cross-layer structure rather than better rank bookkeeping.

## How the model computes, start to finish

A depth-first series of algorithm arcs on the dashboard's matters-most/understood-least targets closes the
mechanism story into one connected narrative for the entire feed-forward stack, with every arrow measured.

**Block 0 is a token feature-table — exactly.** Its dominant term is a literal lookup table indexed by
current-token identity: variance explained by the current token is 1.000000 analytically once the shared
normalization scalar is removed (0.953 empirically against a shuffled control of 0.044). Its second term is a
bigram correction — the (previous, current) token pair explains 0.861 of its variance versus 0.706 for the
current token alone. The derived features are concrete: after a capitalized name plus comma the bigram
features boost dialogue verbs (" scoff", " glared", " sighed"); after a numbered-list item, title-case entry
starters; after a comma they suppress space-less continuations that would be illegal there. The block writes
nothing to the readout directly (removing its direct path costs −0.0000); blocks one through three consume
98.3 percent of its effect, and ablating it erases 74 to 82 percent of the category-code gain — it is the
category engine's input stage. The causal counterpart of the variance claim: replacing the block's entire
output with its token-conditional mean recovers 85.5 percent of its function (86.9 with the pair table).

**Blocks 2 and 3 are an iterated-squaring cascade.** Layer 2 is almost literally "square the previous
block's output," and the square is not confidence-sharpening — self-products are causally null — but a dense
quadratic expansion, predominantly cross-products with no privileged pairs (the largest single pair holds 0.2
percent of the variance; the high-rank tail beyond the top thirty-two directions recovers 88 percent alone).
Its consumer is preferentially layer 3's own square (freezing layer 3 clean removes 84 percent of the
damage — though a matched control shows freezing the next block rescues roughly two-thirds of *any*
perturbation, so the chain-specific excess is seventeen to twenty points early, shrinking to six by block
five: preferential next-block consumption within a broadly compensatory stack, the compensation being the
redundant code again), and layer 3's consumer is layer 4's (83 percent) — each stage building higher-order
products of the block-0 table features,
while a genuinely contextual mixer term (only 0.171 of its variance is current-token-determined) folds fresh
attention context in as topic-conditioned word-completion decisions ("Divine → Feminine", "Cylinder →
bearings/valve"). Category refinement diminishes down the cascade: the hub is essential, the table contributes
82 percent, then 43, then 21.

**The cascade dissolves; a three-region flow map with measured boundaries.** The next-square consumption
fraction decays smoothly — 84 and 83 percent at blocks 2 and 3, a plateau near 51 percent at blocks 4 through
6, then 39, 34, 29, 25, 24 — with no sharp stop, while the direct-to-readout fraction rises from essentially
zero to essentially all, crossing between blocks 8 and 9 (the interpolated blocks were subsequently measured; the smooth decay holds). The model's feed-forward stack is therefore three
regions: a cascade (blocks 0–6) where each block's output is food for the next square; a distributed region
(7–11), where the mid-stack heads are context-dependent semantic sharpeners with no crisp type (the largest
uncharacterized head failed every type test and delivers 88 percent of its effect through downstream
computation — real in aggregate at eleven standard errors, individually insignificant everywhere, and verified genuinely diffuse: the damage spans all activation deciles and one hundred twenty-five of one hundred fifty-two sequences, surviving removal of the top contributors); and a
readout region (12–17) writing directly to the logits — block 14's comma-inside-numbers deviation shapes the
digit-continuation distribution with no meaningful mediated remainder — terminating in the differential-pair
contrast stage. In one sentence: token table → essential hub → iterated squares of rising contextuality →
distributed semantic refinement → direct output writing → conditional contrast readout. Caveats stated with
the measurements: freeze-patching is first-order (late-block direct fractions read "approximately all"), the
region grid is coarse, and mean-ablation is one counterfactual among several.

## Cross-layer composition: folding components into their consumers

A final arc turns the fold outward. The program's most persistent residual impression is that components
look *conditional*: the same head helps at one position and hurts at the next, a block's write appears to
mean different things after different tokens, and every per-component summary carries an unexplained
"depending on context" clause. In a bilinear tensor network that clause need not stay informal. A
consumer's feed-forward block is an exact bilinear form on the sum of its input streams, so substituting —
*folding* — one component's exact output expression into its consumer expands the consumer, exactly, into a
sum of pairwise product terms, each naming two upstream contributors. Whatever was gating a component's
effect is then not a hidden context variable but an explicit multiplicative partner term. We ran three such
folds, and every decomposition is exact where it must be: reconstruction errors between one part in a
million and eight parts in ten million, and every fold's mean-ablation floor reproduces the earlier
censuses to four decimal places (block 3's floor 0.6163 both ways, block 1's floor 5.5744 exactly, the
mid-stack consumer's floor 0.0601).

**Block 3 is twenty-one exact cross-layer terms, and three of them are the layer.** Splitting the earlier
feed-forward stream into block 0's and block 1's separate writes, block 3's bilinear form expands into
twenty-one product terms with cross-layer provenance. Three carry it: block 2's output times itself (the
iterated square, energy share 0.099), fresh attention times block 2 (0.098), and block 1's hub output times
block 2 (0.061). Keeping only those three restores the layer to within +0.033 nats of full function against
a mean-ablation floor of 0.616; keeping the top eight restores it to within +0.0008 — and the named triple
is the best of all one hundred sixty tested triples, by eight standard errors under a permutation control.
In sufficiency the iterated square is the best single term (+0.248 nats kept alone, ahead of the mixer by a
significant −0.0081 ± 0.0020 paired difference), but in necessity it is co-dominant with the
fresh-attention-times-block-2 mixer: deleting the mixer costs +0.0077 ± 0.0010 against +0.0062 ± 0.0010 for
the square, a paired difference of +0.0015 ± 0.0013 — a statistical tie, not a ranking. The mixer is what
supplies the context: only 5.0 percent of its variance is determined by the current token (the iterated
square 17.3 percent; block-1-times-block-2, the most table-like, 24.1 percent). Cross-stream terms carry
97.4 percent of the layer (keeping only the cross-stream terms costs +0.0163 where keeping only the
same-stream diagonal costs +0.1386): the layer is an interaction device across *layers*, and the
conditionality is the cross-layer product itself. The same expansion settles routing: block 0's residual
write is causally dead as a direct input to block 3 (dropping all six block-0-involving terms costs
−0.0000 ± 0.0002, and a gauge-freed substitution of the direct input costs +0.0002 ± 0.0001) and equally
dead as a direct input to block 2 (at most +0.0005 at either of its ports). Block 0 reaches the cascade
through block 1 alone, which consumes it massively at both ports (+0.155 ± 0.005 at its feed-forward input,
+0.091 ± 0.003 at its attention input); blocks 2 and 3 then consume block 1's products.

**The token-table fold: block 1 becomes a token-indexed family of affine operators.** Substituting block
0's token-lookup write into block 1's bilinear form makes block 1, for each current token, an explicit
affine operator on the contextual rest (the two attention streams). The fold is explicit where the
token-blind view is not: the token-indexed linear family captures 41.7 percent of held-out output variance
where the token-blind linearization of the same block captures 0.1 percent (adding a quadratic term in the
rest reaches 69.2 percent). Causally, against a floor of +5.5744 nats, the token-blind linear substitute
costs +0.5366 while the folded family costs +0.3587 — 93.6 percent of the block's causal function — and
folded-plus-rest-quadratic reaches +0.1013 (98.2 percent, the whole remaining cost being the block-0 table
approximation). One coverage caveat must ride with the headline: the table is built from the training slice
and covers 80.5 percent of held positions, with unseen tokens falling back to the global mean of block 0's
write. On covered positions the fold recovers 95.3 percent versus 90.6 for the blind map; on the uncovered
19.5 percent of positions — which carry only 3.7 percent of the function — it is no better than the blind
map (59.7 versus 66.0 percent). The 93.6 percent headline is driven by, and understates, the covered-token
fold. The dissociation runs both directions — the blind linearization is 90 percent causal at 0.1 percent
variance — so causal recovery and explicitness are genuinely different facts. And the operators genuinely
differ by token: the median pairwise cosine between the two hundred most frequent tokens' effective maps is
0.18 — after different tokens, layer 1 is a different operator — with linguistic similarity structure
(' The' versus 'The' 0.938, ' is' versus ' was' 0.806, ' and' versus ' or' 0.711, '.' versus ' the' 0.177).
Concretely: after ' the' the map is a content-word amplifier — +5.4 million summed word-class logit units
and +3.9 million capital-class, while suppressing ' of' — and the substitution comparison favors it
directionally. After ',' the map is a clause-continuation device boosting concessive continuations
(' despite', ' although', ' even', ' though'), and this one passes its causal substitution gate cleanly: at
the six hundred forty held comma positions the folded comma-specific map beats the token-blind map at
paired z = 2.2. After the bare capital ' D' the map's first-order capital-class readout flips sign (−0.68
million versus +3.9 million after ' the') and pushes name and acronym completions — but causally this is a
small relative damping, not a behavioral flip: at the eleven held ' D' positions, substituting a
token-blind map raises capital-class probability by about one point (0.787 versus the folded map's 0.772,
base 0.777) and costs +0.065 ± 0.036 locally relative to the folded map — directionally consistent with the
readout, but weak. (The unembedding readouts here are first-order summaries; block 1's output is largely
consumed by the later squaring stages.)

**The mid-stack head's partner is real but distributed.** The same question for h.L7.0 — the section-100
head with no crisp type — is: who multiplies its write? Scanning downstream bilinear consumers, the largest
is block 7's own feed-forward (+0.00463 ± 0.00072; blocks 8 and 9 about +0.001 each; feed-forward
consumption accounts for roughly forty-five percent of the head's total +0.0170, the rest flowing through
downstream attention plus a twelve percent direct readout). Folding the head's write into block 7's form,
the leading cross term is nominally head-times-block-6-feed-forward (deleting it costs +0.00081 ± 0.00020)
— but it is statistically tied with the head's self-square (+0.00072 ± 0.00023) and
head-times-earlier-feed-forward (+0.00064 ± 0.00021; every pairwise z at or below 0.76). This is a
three-way tie: no single partner term is resolvable, the six single-term deletions sum to only sixty
percent of the joint effect, and the single best cross term is 4.8 percent of the head's total effect. The
mechanism explains why. The head's write is dominated by one global direction carrying eighty-one percent
of its mean write norm; the helping and hurting position centroids both sit at cosine 0.997 to that
direction, so the striking raw figure — helping and hurting writes aligned at cosine 0.9975 — collapses
once the shared component is removed: off that direction the two centroids align at only 0.50, and never
oppose. The head's write does not encode the help-or-hurt distinction at all; the partner re-aims it
position by position (cross-term within-set coherence 0.13 to 0.15), and no scalar gate exists. The sign
flips are real and text-legible — the term helps after "…widening US 66 to" (next token " four", +0.317
nats) and "…may not like your" (" purchase", +0.225), and hurts after "…Samford Intercollegiate. In"
("juries", −0.297) and "…returning from" (" vacation", −0.252) — but the aiming is distributed. Folding
names the gate decisively for the early stack; for the mid-stack it confirms the gating in kind (a bilinear
partner exists) but resolves no single gate — the collective code again.

**A compressed model of the mid-stack code: message plus partially redundant broadcast.** The complementary
experiment treats blocks 5 through 11 as one object. Replacing all seven blocks' outputs with per-position
means jointly costs +1.6528 ± 0.0156 nats — 3.79 times the sum of the seven per-block floors (0.4366), the
strongest quantitative statement yet that the blocks are mutually redundant carriers of a shared message,
and the super-additivity is already present at pair scale (blocks 7 and 9: joint +0.1921 against a sum of
+0.1164, ratio 1.65). The message has a strongly privileged compact head and a long tail: sixteen shared
principal directions of the joint write restore half the function (49.4 percent, ten times over a random
basis), but ninety percent of the function needs 576 of the 1,152 residual dimensions — a head-plus-tail
shape that is robust to basis grain (the shared summed-deviation basis is the best of five allocation
schemes at every size; per-block equal allocation recovers 54, 62, and 70 percent at 144, 288, and 576
dimensions where the shared basis recovers 72, 82, and 93). Hunting for the variable that conditions the
message, current-token identity is the best single conditioning variable at group R-squared 0.47
(shuffled-label control 0.046) — and this is not a frequency artifact: excluding the twenty most frequent
tokens, which cover twenty-nine percent of positions, *raises* it to 0.52 (control 0.083). The mid-stack
write is roughly half of a richer token-conditioned table — far below block 0's 0.953, but led by the same
variable. Three message dimensions earn names, all next-token syntactic-register variables: a
**boundary-reset state** (R-squared 0.753), at its extremes on end-of-text markers and sentence-final
closing quotes — after `…we tried to incorporate all of that into this location."` it pushes " Although",
"This", and " While" and suppresses determiners; a **clause-continuation state** (0.499), at its extremes
on content words deep in run-on sentences ("…may choose to do difficult or substantial graphics changes"),
pushing period, comma, " with", and semicolon; and a **discourse-turn expectation** (0.487), pushing
" But", "Others", and " However". On redundancy, the honest statement after correction: under *matched*
half-subspace deletions the real code costs about twenty-five percent less than an unprotected code
(deleting the same half-subspace from every block costs +0.540 to +0.580 where an unprotected surrogate
costs +0.727 to +0.729), and deleting independent per-block halves costs less than deleting a shared one
(+0.465 to +0.485 versus +0.54 to +0.58) — direct evidence that the surviving blocks partially cover for
deleted subspaces — but the code is far from a fully protected repetition code, and the near-perfect
any-half robustness measured earlier per block does not persist at seven-block joint scale. So
"low-dimensional message plus redundant broadcast" holds as a head-plus-tail statement with named states
and measured partial redundancy, not as a clean factorization.

**What the composition arc says about compression.** The whole-model compression experiments had ended on a
structural negative: restricting all blocks at once is super-additive — half the joint cost is cross-layer
interaction — and no per-layer rank schedule addresses it. The folding results name the missing structure.
It is cross-layer *product coupling* — consumption: a layer's function lives largely in products between
another layer's output and its own streams (97.4 percent cross-stream at block 3), and such a term is
invisible to per-layer rank allocation, which compresses each factor separately and cannot see what the
product needs. The tensor-network route to faithful whole-model compression is therefore to compose folded
cross-layer terms — keep the named products, not per-layer subspaces.

## Honest limitations

1. **Substitutability buys fidelity, not compression — and the exactness is what the bilinear form
   uniquely provides.** The composed analytic interfaces reference the *full* weight tensors as exact
   restrictions; the cores are measured to be incompressible by naked rank truncation. So "~99.9%
   substitutable" is a faithfulness statement, not a description-length win — the compression lives on
   the selection/program side (data-fit programs are ~27× smaller but less faithful). A direct
   decomposition quantifies what the exact fold buys: a generic, architecture-agnostic rank-matched
   linear surrogate recovers about three-quarters of the floor-relative substitutability gain, but it
   leaves the model badly broken (residual delta cross-entropy +4.9 nats, more than doubling the loss),
   whereas the exact fold reaches +0.034 — roughly one hundred forty times more faithful. The same
   generic surrogate on a softmax SwiGLU model is stuck at the same broken level with no fold available;
   the per-layer reconstruction is exact to one part in a million for the bilinear MLP and only about
   sixty percent for the surrogate on either architecture, and full rank does not close it — the residual
   is genuinely quadratic. So folding is not a cosmetic last mile: it is the entire difference between a
   rough approximation and a faithful identity, and that exactness is the one substantial thing a
   nonlinear model structurally cannot offer. This is a fidelity-vs-compression frontier, stated as such.
2. **Meaning is genuinely hard, and that is a finding.** Spectral, non-class-nameable content is the
   *rule* at all 18 layers; nameable selection is confined to the copy/induction/match family; and
   even functional content is bounded-nameable at only four sites, none of them a generalizing law.
   The complete description of the content side remains the exact weight-derived spectrum itself.
3. **Single model for the per-layer decomposition.** Generality is shown for the composition arc, the
   category-engine/induction split, and copy-head commandeering across four models spanning three
   attention families — but the exhaustive four-ledger, per-layer decomposition is bilin18 only.
   Scale transfer is out of ledger (not tested here).
4. **Algorithmic circuits are single-template, single-seed.** The bracket and agreement results each
   rest on one sentence template, small n, and one seed, without error bars; generalization is
   plausible but not established, and the number locus in agreement is open.
5. **Two open methodology reconciliations.** The positional-mean floor localizes the substitutability
   loss layer differently under per-minibatch vs full-corpus means (L17 vs L5); the editing arc's
   in-the-wild transfer is demonstrated only in controlled, strong-clean-induction conditions.
6. **Unsupervised discovery is proxy-seeded and single-model.** The discovery ranking is generated by
   the known-unreliable linear proxy, so it can only surface candidates the proxy scores highly — a
   circuit invisible to the proxy is invisible to the loop; the causal verification is sound but the
   *coverage* is not exhaustive. Some verified remaps also lack a specificity control where the
   trigger class saturates (for example a newline-triggered direction that fires on essentially all
   newlines), so those verdicts rest on the raw output-side effect rather than an active-vs-inactive
   contrast. All of this is bilin18 only.
