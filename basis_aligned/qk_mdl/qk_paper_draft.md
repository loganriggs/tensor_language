# A four-ledger per-layer decomposition of a no-softmax bilinear transformer

*Consolidation draft — 2026-07-30. Numbers are the post-adversarial-review figures only.
Sources: PLAN_per_layer.md, RESULTS_l0_mdl.md §32–§75, LOG.md. Left for parent review; not committed.*

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
interpretability stops and sparse-dictionary methods would be needed. This whole boundary is
architecture-general: the super-additivity, the high-rank basis-aligned feed-forward tail, and the
irreducibly-distributed early hub all replicate on a second bilinear model and on a conventional softmax
SwiGLU transformer (two to three-and-a-half times super-additive, an early hub with zero-to-one of
thirty-two directions nameable). Second, the discovered circuits
are **useful**: they generalize (the distributed-class-mover phenomenon reproduces on a conventional
softmax SwiGLU transformer), and a circuit found unsupervised and verified as a genuine algorithm — a
final-block capitalization selector — is a **calibrated, placebo-controlled control knob**, though its
selection logic sits upstream, so it resists surgical override. Throughout, every headline is held to a
held-out substitution/ablation gate with paired standard errors and an adversarial review before it
stands; the record includes roughly a dozen retractions and softenings that discipline enforced.

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
