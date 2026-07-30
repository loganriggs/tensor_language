# A four-ledger per-layer decomposition of a no-softmax bilinear transformer

*Consolidation draft — 2026-07-30. Numbers are the post-adversarial-review figures only.
Sources: PLAN_per_layer.md, RESULTS_l0_mdl.md §32–§49, LOG.md. Left for parent review; not committed.*

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
reason. The completed result: **every layer 1–17 is representationally exact** (analytic gauges to
~1e-6, and its two attention branches are genuinely two-factor), **~99.9% causally substitutable**
through PCA-bottlenecked analytic interfaces (marginal replacement cost 99.95–99.998% of the
uniform-ceiling headroom, every layer, with paired standard errors and fair nulls), **functionally
mapped** into three families with a per-head selection census and a feed-forward family map, and
**semantically characterizable as "nameable selection over spectral content"** — the model runs
nameable selection programs (only for the copy/induction/match head family) over a graded,
memorized, non-class-nameable content dictionary that is spectral at all 18 layers. Meaning is the
frontier and stays hard: functional content is even bounded-nameable at only four sites.

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

**Substitutability — near-total, every layer.** The per-layer driver replaces each layer's attention
(PCA-64/head bottleneck) and feed-forward (composed fold) simultaneously on FW[448:600]. Marginal
costs range from **+0.00014 nats (layer 12) to +0.0038 nats (layer 14)** — every layer between
**99.95% and 99.998%** of the uniform-ceiling headroom, each with a paired SE and a head-span null.
Null margins vary informatively: layer 5's random-basis null is 199× the true cost (attention very
load-bearing), whereas layers 8 and 14 have nulls near 1× (attention near-dispensable there). At the
whole-model scale, projecting every attention output onto per-layer PCA-64/head bases (with the
residual itself truncated — the strongest test) costs +0.0475 versus base, a ~0.003/layer linear
accumulation, against random-576-dim nulls 100× larger and within-head nulls 20–30× larger; the
architecture-general version holds on bilin12 (+0.116) and bilinsm12 (+0.077). Replacing the entire
MLP stack causally by the composed-fold chain costs +0.0329 (99.56% of headroom, own null 18×). The
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
The selection census marks **23 of 162 heads programmatic** (predicate gain ≥5%), and the predicate
label predicts the head's causal specialization. The feed-forward family map (§44) closes the largest
remaining Function hole: MLP0–3 is the category engine (with **MLP1 the hub — the only block serving
the two-branch match fabric**, +0.029 match-rate on ablation), MLP4–15 is **distributed
category-refinement with no distinct family** (each block removes ≤0.014 category accuracy, cost
≤0.11 nats), and MLP16–17 is a **lexical readout** near the output. Three attention layers are
**diffuse** — no head clears the 5% threshold: layers **4, 9, and 17**.

**Meaning — nameable selection over spectral content, measured everywhere.** Running the content-
nameability gate at **every layer 0–17** settles the content axis: per-head value spectra are **not
class-nameable at any layer** (0–3 of 576 coordinates class-nameable, layer by layer; median class-R²
0.014–0.038 throughout), and class-code and spike-code substitution gates cost ~0.000 nats
everywhere. Crucially content stays spectral even at the lexical-readout layers 16–17 — it does *not*
become nameable near the output. So a graded, memorized, non-class-nameable content spectrum is a
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
The static-prior control is the load-bearing step, and it splits the tasks honestly.

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

## Honest limitations

1. **Substitutability buys fidelity, not compression.** The composed analytic interfaces reference the
   *full* weight tensors as exact restrictions; the cores are measured to be incompressible by naked
   rank truncation. So "~99.9% substitutable" is a faithfulness statement, not a description-length
   win — the compression lives on the selection/program side (data-fit programs are ~27× smaller but
   less faithful). This is a fidelity-vs-compression frontier, stated as such.
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
