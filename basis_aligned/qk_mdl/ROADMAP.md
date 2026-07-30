# Program roadmap: interpretable decomposition of bilin18 (2026-07-30)

## 0. The thesis, stated so it can fail

**Claim under test:** with the current toolset, *most of bilin18 can be decomposed in an
interpretable way* — where "interpretable" is not a vibe but a graded, measured property with four
distinct senses (§1). The strong form — "the whole model is understood" — is almost certainly false
in the naming sense (§1.4; layer-0 content is provably not class-nameable). The defensible strong
form is: **the model is exactly represented, ~99% causally substitutable through compressed analytic
interfaces, functionally mapped into a small number of verified circuits, with the nameable/
un-nameable boundary measured rather than assumed.** That would be a large result. The job of the
next N ticks is to establish it at whole-model scope and make it survive repeated red-teaming.

## 1. What "understood" means here (the four ledgers — never conflate)

1. **Representation** — exact rewriting as a tensor network + gauge scalars. Status: COMPLETE
   (architecture identity; holds by construction, verified 1e-7).
2. **Substitutability** — causal ΔCE cost of replacing a component with a compressed analytic
   surrogate, vs its mean-input floor and vs the uniform ceiling. Status: MLP stack ~99% joint;
   whole-model PCA/head bottleneck +0.047; attention booked as reconstructibility margin. This is
   the strongest ledger and the one to push to whole-model, with nulls.
3. **Function** — which components causally implement which behaviors (atlas families, minimal
   circuits, selectivity nulls). Status: 3 families + several circuits, 4-model general.
4. **Meaning** — component content named as independent code, passing the substitution/steering
   gate. Status: SELECTION nameable, CONTENT spectral at layer 0; higher layers under test now.
   This is the hardest ledger and the one most prone to overclaim — every name goes through the gate.

Reporting rule: a headline must state WHICH ledger. "82.5% substitutable" and "3/576 nameable" are
both true and about different things.

## 2. Purpose — what the decomposition is FOR (each capability, its evidence bar)

- **Algorithm extraction** — pull a task's mechanism OUT as runnable code (predicate/table/circuit)
  that reproduces the model on held-out data. Bar: standalone artifact matches model behavior +
  generalizes to inputs it was not built on. DONE: induction predicate (100.5% shuffled), successor
  tables + increment (agents finishing), bracket predictor (agent). This is the flagship deliverable.
- **Editing / control** — turn a named channel, get the predicted behavior change with bounded
  collateral. Bar: monotone dose-response + natural-CE flat + placebo control. DONE: capability dial
  (induction match channel, CE within 0.002). NEXT: category steering, opener dial, successor dial.
- **Generalization prediction** — the decomposition predicts behavior on new distributions. Bar:
  a functional claim forecasts held-out/OOD/period-shifted behavior. DONE: induction period-96 pass
  / period-32 + shuffled fail (honest); importance maps FineWeb->Pile 0.85-0.91; 4-model transfer.
- **Jailbreak / safety-relevant** (NEW, Logan-raised) — if selection is nameable and content is a
  spectral bus, the leverage points are the SELECTION programs (what gets attended/matched), not the
  content. Concrete test to design: can a named selection edit induce/suppress a targeted completion
  (a controlled "steer the copy target" demo) — the interpretability-grounded analogue of a prompt
  injection, measured for reach and collateral. This is where selection-nameability becomes useful.
- **Red-teaming** — every arc gets an adversarial reviewer before enshrinement; the record already
  shows this catching a real bug (lambda-scaling) and 3 framing inflations. Standing, not optional.

## 3. Coverage plan to whole-model (the remaining work, by ledger)

Substitutability is near-complete; Function is patchy above layer 8; Meaning is the frontier.
Structure the sweep as: for each layer L (bottom-up, the dependency order), establish (a) attention
pattern reconstructibility with named-basis margin over random null, (b) MLP composed-fold interface
%, (c) which atlas-family/circuit it participates in, (d) any nameable selection channel via the
gate. Most of (a)/(b) are already whole-model; the gaps are (c)/(d) for layers 8-17.

### TODO — sequenced

- [ ] **T1 Process the 3 running semantics agents** (opener L13, successor L8, category block-3):
      integrate, then dedicated red-team pass on the batch before RESULTS §35. (in flight)
- [ ] **T2 Selection-channel census, all 18 layers.** The nameable objects are selection-side; run
      the QK-side principal-angle + third-moment-CP scaffold extraction per layer (mechanism ledger),
      gate each named cluster by substitution. Expect a per-layer table: how many selection programs,
      what they match. This is the systematic version of the induction/opener/successor findings.
- [ ] **T3 Whole-model substitutability with SEs + fair nulls + uniform-ceiling denominators.**
      One clean script: every layer's attention+MLP replaced by its reviewed surrogate simultaneously,
      base-relative ΔCE with per-token SE, head-span null, reported against both floor and ln V.
      The single defensible whole-model number.
- [ ] **T4 Circuit atlas for layers 8-17 behaviors** not yet covered (the top-MLP register gains,
      the L13+ readout, any second induction/copy paths) — minimal circuits + selectivity nulls.
- [ ] **T5 The content-spectrum question at depth** (answers Logan's L>0 question): are higher-layer
      content channels FUNCTIONALLY nameable (task-tables) even though lexically un-nameable? The
      successor/opener agents are the first probes; generalize to a depth curve.
- [ ] **T6 Jailbreak/steer demo** (§2) — design a named-selection edit that redirects a targeted
      completion; measure reach + collateral; red-team for artifact.
- [ ] **T7 Consolidation** — paper_atlas sync (§33/§34/§35), artifact refresh, a single
      whole-model figure with the four-ledger status. Do this only on reviewed numbers.
- [ ] **T8 Second-model replication of the composition arc** (bilin12/swiglu18) — generality of the
      analytic-chain result, not just the atlas. Gauges + composed fold should port; measure.
- [ ] **Standing:** every arc -> adversarial reviewer -> retract/rerun until defendable; held-back
      slice + SEs on all headlines; never conflate ledgers; no top-token naming without a gate;
      Pythia HELD.

## 4. Honest risks to the thesis (track these; they are how it could be less than claimed)

- Substitutability at +0.047 is real but the coordinates are mostly un-named — "decomposed" in the
  substitutable sense, not the meaning sense. The write-up must not let one borrow the other's glory.
- The composed fold references full weight tensors (no compression win); the compression story lives
  entirely in the streams/selection side. State the description-length ledger every time.
- Content-spectrum un-nameability may be the RULE, not a layer-0 quirk. If T5 confirms it at depth,
  the honest headline becomes "named programs over unnameable dictionaries" — still a big result,
  but a different one than "fully interpretable," and we say so.
- Selectivity of edits (jailbreak/steer) may be low if redundancy buffers everything (induction
  dial had modest range for exactly this reason). Report reach honestly.

## ═══ PROGRAM PIVOT 2026-07-30: FULL PER-LAYER DECOMPOSITION (Logan) ═══
GOAL: fully decompose EVERY layer 1..17 on all four ledgers (representation / substitutability /
function / meaning), bottom-up starting at layer 1, to the reviewed standard (held-back FW[448:600],
paired standard errors, fair nulls, substitution gates, adversarial review before enshrinement). Option-2
algorithmic-capability arcs run IN PARALLEL (verify -> patch to minimal circuit -> red-team). Layer 1 is
UN-held. Cadence: 10-minute cron (job 0b62fec1).

PARALLELIZATION ARCHITECTURE:
- GPU is one shared ~15GB card. ALL heavy decomposition GPU work is SERIALIZED through the qkqueue
  daemon (QUEUE.txt, one script/line); keep it stocked every tick.
- Subagents run everything GPU-free in parallel: per-layer gap audit, writing the next batch of
  layer/algo scripts, adversarial red-teaming, paper drafting. GPU-touching agents use batch<=8,
  expandable_segments, <4GB footprint.
- Reusable driver qk_layer_decomp.py (arg = layer L) runs the four-ledger battery per layer; queue
  `qk_layer_decomp.py 2`, `... 3`, ... bottom-up.

PER-LAYER STATUS TABLE: [to be populated by the audit agent — layers 1..17 x 4 ledgers, DONE/PARTIAL/
MISSING with evidence]. Known starting point (from RESULTS): Representation + Substitutability largely
whole-model/done; Function patchy above layer 8; Meaning is the frontier; layer 1 has §6a-c (token-
identity port, 9-head mechanism ledger, h3-reading) — build on it.

SETUP TICK (2026-07-30): dispatched 3 agents — (A) per-layer four-ledger audit -> master status table +
prioritized next-10 experiments; (B) algorithmic-capability scout -> verify scripts for new tasks
(qk_algoverify_*.py); (C) qk_layer_decomp.py template author. Queue intentionally idle this one tick so
agent B has GPU headroom; stock from agent outputs next tick.

## ═══ PER-LAYER DECOMPOSITION PROGRAM — COMPLETE + DEFENDED (2026-07-30) ═══
STATUS: the pivot goal ("fully decompose every layer") is DONE for all layers 1-17 on all four ledgers,
capstone-adversarially-reviewed (5 framing corrections applied), and consolidated (RESULTS §32-52,
qk_paper_draft.md, artifact f27aeab4). Generality: whole-model substitutability across 4 models (§32b);
content-spectral confirmed architecture-general on swiglu18/softmax (§52). Five algorithmic circuits
decomposed + reviewed (2 v1-routers, 1 bounded successor, 1 redundant router, 1 in-context prior).
~12 over-claims retracted across the effort (editing arc, greater-of-two 3x, sv-agreement, capstone 5).

REMAINING (diminishing-returns or scope-expansion — awaiting Logan's steer, not auto-run to avoid churn):
- Full per-layer 4-ledger sweep on a SECOND model (bilin12/swiglu18) — big commitment, scope expansion.
- Selection-nameability generality on a 2nd model — likely reconfirms the KNOWN negative (taxonomy is
  model-specific; families general — census-generality already showed this).
- Open mechanistic threads: sv-agreement number locus (§42), L8 successor range limit (§51), KEY_newline
  mechanism (open since anchor falsified).
- Paper polish toward a shareable write-up; artifact title/abstract refresh (still says "layers 2-17").

## GENERALITY PICTURE COMPLETE (2026-07-30, §52/§55): all four ledgers' headlines tested on a 2nd
model (swiglu18/softmax) + multi-model. Repr exact-all-arch; Subst general (§32b, 4 models); Function
family-geography general §55 (hub bilin18-specific = 2-branch artifact); Meaning content-spectral general
§52. Four-ledger STRUCTURE architecture-general; head TAXONOMY model-specific (census-generality neg).
No clean in-scope work remains without Logan's steer -- holding (brief no-op ticks) until redirected.

## UNSUPERVISED CIRCUIT-DISCOVERY TOOLBOX COMPLETE (2026-07-30)
Following one set of paths through the exact decomposition yields an algorithm; different circuit TYPES need
different detectors. Every under-served type from the §58 auto-cluster tool-gap map now has a working,
CAUSALLY-VERIFIED detector (held-back FW[448:600], mean-ablation, paired standard errors):
- §56 class-boost head (trigger→boost a token class) — 5 verified
- §57 composition (feed-forward→head, QK-steering edge-patch)
- §58 auto-cluster taxonomy (12 families + tool-gap map)
- §59 suppression (negative-logit inhibition) — mlp.L17.d1 clause-boundary generic-word inhibition
- §60 copy / value-router (source-dependent) — re-derived L8H3/H7, L5H5 unsupervised
- §61 redundant / distributed (greedy joint ablation + redundancy ratio + random control) — copy family
  ratio 3.86 distributed circuit; diffuse newline cluster genuinely null
- §62 positional / structural (offset-vs-content-residual + distance-since-newline bucketing) — 54/162
  positional; NO distance-to-newline circuit (line structure is lexical)
- §63 byte-fragment / orthographic trigger (predicate library + out-of-sample purity pre-filter + conditional
  causal contrast) — digit heads h.L8.7/h.L8.3, punctuation head h.L13.8; rejects overfit affix fingerprints
- §64 trigger-vs-output decoupling / remap (trigger+output histograms → output-side causal test) — 3 genuine
  remaps, 3 proxy-artifacts; strongest mlp.L15.d2 punctuation→capital
RECURRING HEADLINE LESSON (documented across all nine): the linear direct-to-logits proxy is UNRELIABLE in
magnitude, sign, case, and single-vs-joint — every circuit type earns its keep only through a type-specific
CAUSAL test. Catalog: TECHNIQUES_unsup_discovery.md. Next: adversarial red-team (§61-§64 in flight) then fold
into paper draft + artifact.
