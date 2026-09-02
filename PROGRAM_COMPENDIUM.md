# PROGRAM COMPENDIUM — everything we tried, in one place
*Compiled 2026-09-02 from a full sweep of the repo, the archive, theseus-bench, and all 6,800+ commits.
Requested by Logan: "find all the things we did, even if they didn't work, and organize them together."*

**How this is organized.** Four complementary views of the same history:
- **Part I** — the timeline: every era at a glance.
- **Part II** — era chapters: goal → what happened → how it ended → what got dropped.
- **Part III** — cross-cutting registries: the graveyard, the dropped-threads list, the claims that stand,
  the retraction log, method families traced across eras.
- **Part IV** — the bilin18 by-module map (dossier digest) and the ledger/§-namespace appendix.

Citation convention: `§N` = BILIN18_CONNECTION.md ledger entries (§804–§2611 live there; earlier §s in qk_mdl files
and early-program history — see Appendix). "FINDING n" = basis_aligned/RESULTS.md. "F n" = per-directory findings.

---

# PART I — TIMELINE OF ERAS

| Dates (2026) | Era | Territory | Headline |
|---|---|---|---|
| 06-01 → 07-05 | Cycle task (founding) | root, archive/ | bilinear attention CAN learn cycles; 2L·d64 generalizes; softmax beats it OOD |
| 07-05 | Graph-walk tracing + geometry | root, archive/ | topology-agnostic circuit; 3-layer collapse diagnosed (lerp halves the stream) |
| 07-05 | "Why neighbors nearby" program | archive/ | reversibility pins organization sign; 4-coef reconstruction r=.954 |
| 07-05 | LLM replication + GPT-2 circuit | archive/ | locality↔organization r=+.60 across 144 heads; induction heads are NOT the map builders |
| 07-05 → 07-06 | k-hop depth ladder | root | 3 attn layers unlock hop-2/3; "k+1 layers" FALSE; 4th layer = seed lottery (1/3) |
| 07-08 → 07-09 | Deeper circuits on text (PLAN.md sprint) | root | induction reverse-engineered (XNOR signature); smooth formation (no phase change); hysteretic scaffolding |
| 07-09 → 07-10 | mechdecomp (weight-action SAE) | mechdecomp/ | "a better basis, at best a weak feature finder"; single-atom ablation has no power |
| 07-10 → 07-13 | jacclust (Jacobian kernel) | jacclust/ | real-MLP NULL (G isotropic); real-attention 15–25× positive; rotary sign bug caught |
| 07-11 → 07-13 | tensor-sim transcoders | root subdir | the metric decides whether a backdoor survives compression (ASR .016 vs .999) |
| 07-13 → ~08-04 | basis_aligned e1–e11 | basis_aligned/ | sparsity is gauge; THE METRIC DECIDES WHAT EXISTS (e4); +0.26 nats @ 49× objects (e7b) |
| 07-14 → 08-08 | qk_mdl tiers + four ledgers | qk_mdl/ | 546M → +0.256 nats @ ~12.7 MB readable structure; marginals don't compose (+0.534) |
| 07-20 | tn_gauge (DMRG bridge) | basis_aligned/tn_gauge | interior gauge DOF = 0; QK-used-subspace 28% @ +0.06; DMRG iteration is a no-op (F39) |
| 07-28 → 08-03 | ECG/medical teachers | code_teacher/ | teachers validated (LBBB .998); program not continued here |
| 08-04 → 08-08 | qk_e scale box (w264→w1152) | qk_mdl/ | partition cost halves with width; EVERY w264 structural win flips sign at w1152 |
| 08-08 → 08-10 | tiny_full_interp | basis_aligned/ | THE FOLDABILITY TAX: softmax×GELU interaction carries induction (74%); tax grows with width |
| 08-08 → 08-11 | memorization_post (Parts 1–3) | basis_aligned/ | KKT unlearning exact (r=1.000000); D≈−I refuted; LP edits are 50–100× noise-fragile |
| 08-11 | editability/capacity week (P1–P16) | memorization_post | one-layer margin-LP zero-collateral at 100 facts; capacity curve N*=1200/H*=40 |
| 08-14 → 08-16 | toy Part A/B (planted answers) | runs_gen/, ledger head | A1–A6, B0–B3; 47/47 theory checks; "what the toys say about bilin18" carry-over doc |
| 08-16 → 08-21 | bilinear_quotient early era (§~100–803) | bilinear_quotient/ | census machinery, carriers, regulator arcs, canary lineage (per commit log) |
| 08-21 → 09-02 | THE LEDGER ERA §804–§2611 | bilinear_quotient/ | frontier +2.6735; sealed protocol; equality circuit; rungs 400–490; three claimed positives |
| 08-25 → 08-31 | theseus-bench | /workspace/theseus-bench | anchors FROZEN 198/198; first verified plank mlp1 .9507 @ 96.6 Mbit; M0-partial |

Commit-volume peaks: 07-29/30 (106/197), 08-17 (331), 08-23 (360), 08-27–09-01 (458/516/537/472/198/552).

---

# PART II — ERA CHAPTERS

## Chapter 1 — The founding toys and the geometry program (06-01 → 07-08; now archive/)
**Goal.** Can softmax-free bilinear attention learn structured tasks, and why do in-context graph
representations organize (Park et al. ICLR 2025)?
**What happened.**
- *Cycles*: 1 layer can't; smallest solver 2L·d32·1head (18.7k params); smallest generalizer 2L·d64 with
  grokking-like late rise; softmax baseline better OOD (0.98/0.97). Residual encodes phase on a clean circle.
- *Lattice walks*: grid<cylinder<torus difficulty; grid-trained transfers to cylinder/torus at 0.99; the 3-layer
  collapse traced to the `lerp` residual halving the stream per layer (fix: plain add or RMSNorm).
- *Six-family generalist*: ~1.00 legal on all + zero-shot torus/ER; mixture HELPS. Organization sign is
  architecture-dependent (+.66 bilin-lerp vs −.80 softmax-add on identical data).
- *The "why"*: walk REVERSIBILITY pins the sign (directed rings −.55…−.80, biased rings +.38…+.67);
  4-coefficient reconstruction of organization r=.954; projecting out own-token content moves every model positive.
- *Real LLMs (11 models)*: all organize positively (+.02…+.49), no scale needed; head LOCALITY predicts
  organization r=+.60 (local attention = one step of Laplacian smoothing); induction heads solve the task but
  are NOT the map builders (r=−.03); the 16 most-local GPT-2 heads are worth +.93 nats on wikitext.
- *Causal verdict*: "the content is used, the geometry is its shadow" — a 180° automorphism patch (geometry
  perfectly preserved) is the WORST intervention (legal .278).
**How it ended.** Formally closed 07-13 (README rewritten); code swept to archive/.
**Dropped.** 7-ring readout inversion (unexplained, flagged "worth keeping"); GPT-2-with-numerals does the task
better with NO map (competence/organization dissociation, left as curiosity); toy_query/toy_nlmix P5/P8
pre-registered but verdicts never recorded; grid+cylinder "not pinned" boundary case.

## Chapter 2 — k-hop ladder and the deeper-circuits sprint (07-05 → 07-09; root)
**Goal.** Find the "next induction head": token categories needing >2 layers; then map what each depth unlocks
on real text.
**What happened.**
- *k-hop*: 3 attention layers unlock hop-2/3 as a class (attn·attn·attn .96/.82; MLPs cannot substitute);
  mechanism = per-hop pointer advance in a rotated basis; "hop-k needs k+1 layers" FALSE; the 4th layer buys
  nothing (1/3 seeds both ways — an optimization lottery); curriculum BACKFIRED (0/4).
- *Deeper circuits (24h autonomous sprint)*: induction = 0.91% of val tokens, reverse-engineered as
  L0H3→{L1H2,L1H1} with an XNOR (pattern×OV agreement) signature, verified 3 ways; depth-3/4 unlocks
  higher-order n-gram/lexicon circuits (67% predictable from 3 corpus tokens); two architecture axes
  (context circuits need attention LAYERS; statistics circuits need bilinear-MLP capacity); heads non-monotone
  (h4 best at 0.47). Dynamics: bilinear induction forms SMOOTHLY (no phase change — a negative vs Singh et al.);
  scaffold mixtures install it dose-dependently and HYSTERETICALLY (remove scaffold → synthetic collapses,
  natural persists). Program-A retraction: the copy-burst lever installs a POSITIONAL copier, not induction.
**How it ended.** Explicit PAUSE POINT mid-loop (block4 fingerprints computed, clusters never labeled);
program pivoted to mechdecomp on Logan's direction.
**Dropped.** block4 cluster labeling; both *_gated_depth3 reports (referenced by nothing); OWT differential
gates shipped with a self-declared non-monotone sanity failure, never re-derived; the atlas ("one page per
circuit") stopped at one page; runs_lm/ gitignored so most numbers are unreproducible from the repo.

## Chapter 3 — mechdecomp: the weight-action SAE (07-09 → 07-10)
**Goal.** Learn a dictionary whose rank-1 mechanisms W·ddᵀ sparsely reconstruct a map's ACTION (not activations).
**What happened.** Closed-form rank-r theorem verified 1.8e-10; synthetic recovery .96–.98; on Pythia-410m
down_proj beats a strengthened SAE (R² .4591 vs .4291, z=−8.9) — the spec premise holds modestly. Central
negative: atoms only weakly mechanism-like (uniqueness 1.79× random vs a true generator's 4.06×);
identifiability bounded by sqrt(rank/d_in). Methodological gold: SINGLE-ATOM ABLATION HAS NO POWER (1.54× on a
known-true generator) — "ablate a feature, watch behavior" invalidated as dictionary evidence; irreplaceability
separates (8.4×/34×). The Pythia postmortem is the best failure doc in the repo: the SCISSORS (low-rank site
reconstructs but can't separate; high-rank separates but can't reconstruct — no site gives both).
**How it ended.** Paused by Logan for jacclust; 18 retracted claims tabulated, including a retraction of a retraction.
**Dropped.** Gemma tier (401 gated-repo block); irreplaceability under matched-R² for GPT-2 OV; the bilinear
quartic with no matched-L0 solution.

## Chapter 4 — jacclust + tensor-sim transcoders (07-10 → 07-13)
**Goal.** Cluster datapoints by the OPERATION applied (exact per-datapoint Jacobian kernel ⟨J(x),J(x')⟩=xᵀGx');
separately, train transcoders whose faithfulness is enforced in WEIGHT space (Gaussian tensor inner product).
**What happened.**
- jacclust: toys perfect (kernel exact 1e-14, ARI 1.000, the advantage law ρ=.83); REAL MLPs NULL (G eff-rank
  .86–.94 ≈ isotropic → ties controls); real ATTENTION the one positive (J beats random 15–25×; the causally
  useful object is the query readout Wq·x, ~5×). 9 retractions incl. the rotary SIGN BUG (inflated 5–16×→~2×)
  and the naive_squared_attention normalization bug (CE 7.5→3.5) — both fixes baked into tt_model.py, which
  became the loader backbone for the entire bilin18 era.
- tensor-sim transcoders: the metric-decides flagship — a planted backdoor's survival under compression is set
  ENTIRELY by the audit metric (ASR .005 blind vs 1.000 ridged; a 1% identity ridge flips .016→.999). F10
  ("real bilinear MLP not low-rank") self-retracted: with the right metric, rank 8 = 0.17% of hidden units
  reproduces .807. No 2-layer compositional structure found (honest negative).
**Dropped.** Nameable roles for winning heads; a non-derived causal target for attention ("would make it
publishable"); the SVHN backdoor variant and the two-real-layer composition test (both explicitly asked of
Logan in the LOG, never answered); jacclust_dgpA.html parked at root, never extended.

## Chapter 5 — basis_aligned: e1–e11, tn_gauge, tiny_full_interp, memorization (07-13 → 08-11)
**Organizing thesis (the era's discovery):** weight sparsity is basis-dependent and only meaningful in FOLDED
bases; and WHICH COMPUTATION "EXISTS" IS DECIDED BY THE AUDIT METRIC (MSE vs ε-accuracy vs CE).
**The e-series (FINDINGS 1–13, RESULTS.md authoritative):**
- e1: unfolded sparsity is a gauge artifact (zeros 87.5%→0% under function-identical rotation).
- e2: L1-sparsifying only L,R,D is COSMETIC (rotation hides in free E,U); full sparsification reaches the
  exact solution's 8.3% only from rotated-handcoded init (protocol exactly undoes the rotation).
- e3: NO superposition under MSE for squares — closed-form rank bound verified to <0.7%; pruning to 1.3% of
  weights flat, endpoint literally the dedicated solution.
- e4 (the pivot): SAME architecture, swap loss — MSE computes 18 features, ε-surrogate computes 128;
  MSE-finetuning DESTROYS a handcoded superposition solution; ε preserves it.
- e5: CE behaves like ε-accuracy (128/128); ReLU readout beats the linear rank bound; sign-pairing hypothesis
  refuted (it's distributed superposition denoised by ReLU+bias).
- e6: embedding compression fails on raw weights (k-means 25.6k costs +1.45 nats); FVU mispredicts damage —
  SUBTRACTION ≫ ADDITION (deletion 10–18× worse than noise at matched FVU); hierarchy priors underperform.
- e7/e7b/e7c: Pareto dictionaries + CE-through-frozen-model: n=1024/L0=64 → +0.26 nats (from +2.11), a 49×
  object reduction; KL-distillation reproduces it (+0.23) so it's FAITHFUL compression, not repair.
  (Partly overturns e6's "the tokens are the objects".)
- e8/e8b: semantic ordering beats random by a stable ~.03 FVU at every TT rank; BPE = random (token-ID
  adjacency carries no hierarchy); TT-SVD near-optimal; a param-efficiency claim CORRECTED (budget mislabeled).
- e9: BatchTopK better FVU, WORSE ΔCE (adaptive L0 follows reconstruction difficulty, not behavioral
  importance); headline robust (+0.28 ≈ +0.26).
- e10 (NEGATIVE): the reader-Gram weight metric is ≈isotropic; NO quadratic form can capture the
  subtraction/addition asymmetry in principle; proposed successor e12 NEVER BUILT.
- e11 (NULL): control ladder all passes (the global-rank floor made explicit); learned ordering by 0.5M swaps
  is a no-op — every accepted swap was pad⟷pad; semantic ordering is 2-swap-optimal.
**tn_gauge (F1–F39):** the DMRG framing died on a toy FIRST (residual stream is one shared bond; interior gauge
DOFs = 0; QK rotation broken by RoPE Δlogit 18.7; MLP hidden basis pinned by ⊙ — sparsity is a masking problem,
not a rotation problem). Exact rotation gauges EMPTY (F1–F8); code propagation a Pareto trade not a free win
(+0.59 nats). The productive line: the QK USED-SUBSPACE (~128-dim activation-weighted) → all-18-layers QK at
28% of raw for +0.06 nats; code 82% current-token; composed classes decode to syntactic dependencies. F39
closes it honestly: the DMRG re-fit iteration is a NO-OP (+0.241→+0.234→+0.241) — compounding is irreducible.
**tiny_full_interp (F1–F24):** THE FOLDABILITY TAX — a matched softmax+GELU transformer inducts where every
foldable arm is null (9.8×/12.8×); held-CE tax 0.052–0.125 nats, positive 9/9 cells, growing with width;
factorial: NEITHER softmax nor GELU alone — the INTERACTION carries 74% (refuting the registered headline);
the conjunction dissolves as models grow (91.4%→19.9%). Compression negative: the model is still the shortest
description of itself. F14's clean "one octave per layer" threshold retracted by F16 (property of the
detection criterion). Stopped by priority override for qk_mdl Tier 2.
**memorization_post (Parts 1–3 + editability week):** Gram-predicted collateral r=.943; KKT unlearning EXACT
(predicted-vs-measured r=1.000000, ~9× less collateral than naive); D≈−I REFUTED; blind rank-1 fact extraction
fails completely (0–1% recovery); storage not disjoint per layer (degree-3/4 composed terms ~82% of ablation);
margin-LP editor certified zero-collateral at 350/600/900 facts BUT masking battery shows margin collapse
(LP-edited models 50–100× more noise-fragile than a retrained oracle); relearn-speed and resurrection
signatures refuted. Part 4 (capacity vs structure fraction) NEVER BUILT (listed in README, file absent).
**Dropped (era-wide).** e12; e11's structural-vs-annealing verdict; memorization Part 4; unscored
tiny_full_interp ladders (X5, vanilla-w192) + the stopped seeds3–5 chain + STANDALONE §8 items; thread-4
A5/A6/Part B of bilinear-quotient-experiments.md; TENSOR_NET_EXPLAINER §5's four never-run cells; the
balanced-gauge canonicalization of e-series statistics (acknowledged, unexecuted); bio_medical field map
(orphan strategy doc); e6b/e6c/e7b/e7c have no dedicated results JSON (fragile provenance).

## Chapter 6 — qk_mdl: pricing description of a real model (07-14 → 08-08)
**Goal.** Price the BITS of description a real bilinear LM needs, audited by ΔCE — never reconstruction.
**Four sub-programs in one directory:**
- *Tier ladder*: tier-0 exactness gates (folding reproduces the model 1e-13–1e-15); bilin18 layer-0 joint
  frontier all-9-heads vq256 = 165× compression at +0.0084 nats; 7/9 heads individually zeroable ≤.011 but
  JOINTLY +0.534 — MARGINALS DON'T COMPOSE (the era's most-reused lesson). sqrd12 is ~15× less compressible.
  Tier-3 path-folded lookups NEGATIVE. Consolidated headline: 546M → +0.256 nats ≈ 12.7 MB readable structure;
  layer-0 trained grand codebook −0.019 (better than exact); the full-stack wall (+0.757) cracked by windowed
  code propagation to +0.059 (W=6). Superadditivity: rulebooks ≤+0.008/layer but +0.190 composed (6×).
  bilin18's unnormalized product attention degrades past ~512 context → ALL audits pinned to T=512.
- *tier2_model/tier2_folding*: THE loader that gets everything right (normalization conventions, RoPE sign,
  rms-then-rope order, v-lerp, bf16 tables, local-files fallback) — imported by ~20 later files and the
  polynomial_causal facade; folded V×128 factor tables that never materialize the forbidden 50304² matrix.
- *Merge/dictionary/SAE line*: sparse dictionary over folded QK factors survives at ~zero cost (6.1% of raw
  bits, wide ΔCE ≈ 0; matched-bits SVD +0.023); atoms READABLE (music/film/surname/-ed morphology). Two-stage
  merge→dict RETRACTED (small-audit overfitting). Residual-stream SAE NEGATIVE three rounds: 0/32 features
  clear the causal bar; the dictionary NAMES the hub but does not causally explain it (2.2% of ablation ΔCE).
- *Four-ledgers whole-model decomposition*: (1) representation complete (identity 1e-7); (2) substitutability —
  whole-model bottleneck +0.047 nats = 99.4% of uniform-ceiling headroom (a lambda bug caught and retracted);
  (3) function — 30/162 heads programmatic; the v1-ROUTER principle (late heads route the layer-0 value cache
  via v-lerp; QK decides WHERE, layer 0 decides WHAT — increment & successor converge on L8H3/L8H7);
  (4) meaning — the boundary: nameable SELECTION over spectral non-generalizing CONTENT; gate-passing names
  cover only 9/180 components (5.2% of solo causal mass; ~11% of headroom — computation is combinational,
  superadditivity 2.87×).
- *qk_e scale box (w264→w1152, rented GPUs, MAILBOX coordination)*: slot-partition cost halves with width
  (+0.234→+0.124); Muon default; prox-lasso ~free at scale but buys no readability; honest recipe premium
  +0.167 nats; §S6: EVERY w264 structural win flips sign at w1152 — "decompositions survive; absolute wins don't."
**Dropped.** Empty QUEUE.txt; §S4 not-yet-run-at-scale list; IDEAS_arch_slate S1–S4+A2 (all proposed, none
run); BRAINSTORM_STATE items 3–7 + parked; ROADMAP T4–T8 open; "Pythia HELD" standing; four spec/proposal
docs with no results file; tier-3 abandoned after its negative.

## Chapter 7 — the toy bridge: Part A/B and "what the toys say" (08-11 → 08-16)
The editability week (P1–P16: one-layer margin LP zero-collateral at 100 facts; P16b refuted; capacity
N*=1200/H*=40, expressivity-limited) ran against memorization_post; then the planted-answer toy program
(A1–A6, B0–B3, runs_gen bilin-add/lerp): A2 crystallisation arrest + withdrawn-column restoration; A3
identifiability axis settled; factor ablation NOT implementable (all three schemes fail soundness); B3 path
routing recovers from composed weights, within-path factor split does NOT; B0 sharpening signature fires at
random init (deflating it as evidence). 47/47 theory checks; 3 retractions in review. It ends in the
"What the toys say about bilin18" doc (now the ledger's prologue): MLP architecture match EXACT, attention
NOT (two-QK product differs) — the carry-over contract for the main program.

## Chapter 8 — THE BILIN18 PROGRAM (08-16 → present) — see Part IV and the arc tables below
[filled from the ledger sweep — §804–§2611, rungs, frontier, claims]

## Chapter 9 — theseus-bench (08-25 → ; /workspace/theseus-bench)
**Goal.** Mech-interp as VERIFIABLE COMPONENT REPLACEMENT ("glass planks"): swap modules for simple surrogates,
verify against frozen optimal-ablation anchors, price simplicity in Mbit; Pareto frontier is the object of record.
**Built & verified.** Anchors FROZEN 198/198 (median opt/mean .9964); contract/complexity/priorities machinery;
priority board by unexplained CE (top: mlp1 .181, mlp0 .062, mlp2 .041); FIRST VERIFIED PLANK — mlp1 tiered
token table + rank-128 ridge = fid .9507 @ 96.6 Mbit (vs 255); worked examples: head 5.7 the attention-sink
bias-head (one fixed vector ≈ .985), head 12.4 the pronoun announcer (selectivity 347×); registry seeds incl.
227 named circuit certificates; a separate gauge-aware tensor-program pricing prototype.
**Standing lessons.** LOCAL FIDELITY DOES NOT COMPOSE (first all-18-attention composite failed); no partial
replacement preserves the 62 circuits (0/62 across four configs) — registry rows are aggregate-CE objects.
**Status.** M0-partial and drifting into a results sink for the parent program: baselines/ and submissions/ are
stubs; verifier/sandbox/splits/CI absent; complexity grammar admittedly a proxy; the name itself still TBD.

## Chapter 8 — THE BILIN18 PROGRAM (08-16 → present; bilinear_quotient/ + polynomial_causal/)
The main event: reverse-engineering bilin18 (Elriggs/gpt2-bilinear-sqrd-attn-18l-9h-1152embd, 546M, D=1152)
into a predictive, manipulable, composable tensor program. Ledger BILIN18_CONNECTION.md §804–§2611 (1,811
entries); ~390 backlog rungs + rung-numbered experiments to 490; 2,612 result receipts; 154 preregistrations;
165 hourly strategic reviews; 114 numbered lessons.

### 8.0 The machinery (how the program runs)
- **Facade**: bilin18_observed_model_facade.py — checkpoint pinned three ways (revision, SHA-256, byte size),
  config validated field-by-field, no network I/O, dispatchers contractually checked. tt_model.py (jacclust) +
  tier2_model.py (qk_mdl) are the load-bearing ancestry.
- **Queue discipline**: bqrunner pops queue.txt; enqueue.sh runs the fast suite + static gate + BQLIB_DRYRUN
  plan-preflight before anything reaches the GPU; preflight.py carries 8 advisory checks, each purchased by a
  named incident; idle cycles run the canary (4-atlas integrity + leverage law + a bit-level determinism
  fingerprint).
- **Epistemic discipline**: preregistered pred_a/b/c with measured bars; score-as-written; failures preserved
  as receipts (failed-instrument receipts retained alongside repairs; the largest polynomial_causal dir is a
  preserved FAILURE lifecycle, 32MB); Möbius/interchange/commutant/projector instruments in ops/.
- **LESSONS.md**: rules 1–10 are scientific (mean-ablate never zero-ablate; only subtraction bites; census
  members are damage-tails; two-signed policies everywhere; "mechanism" has grades); 11–114 are incidents —
  verification theatre, control design, serialization, and honestly-logged self-repetition ("I rebuilt a
  metric whose defect I had already written down").

### 8.1 Condensed arc map (merged from the two ledger sweeps)
**§804–§1035 — Understanding gpt2-family + bilin18 early:** DC-bias correction cascade (§805); "~93% of the
early layers understood" (§809); keep-magnitude RETRACTED as a rank artifact (§836); 77% of loss is word-choice
within the correct class (§829); bilin18 = grammar machine + content machine (§863); 78% of loss is
first-mention content, universal (§879); the whole-model benchmark born at 81% and cut to 0.29 held-out
(§900/§901) → stabilized at two benchmarks 0.41/0.72 (§914); content = bag-of-embeddings running average
(§932); front-nonlinearity REVERSAL (§992: the front interaction is load-bearing); squared attention drives
the distributed-cooperative structure, the bilinear MLP dampens it (§969); injection wave validates mechanisms
generatively (induction +8.56 nats, §1025); north star stabilizes at 0.32±0.06 (§1014).
**§1036–§1160 — Structure and transport:** deep middle = CONTEXT×CONTEXT (§1041); content is functionally
interchangeable across independently trained siblings (96%/93%, §1061–66); CAPSTONE NULL — per-module
understanding does not compose, whole-model stand-ins recover 12% (§1070); L5H7's 0.88-nat function is ONE
fixed vector (§1089; later flagged as a re-derivation of §429–432 — the dedup failure); class is a PACKAGE
(§1103); content = sparse skeleton k≈8 + dense causal tail (§1114/15); single API features are not causal
variables (§1129); transport law — coverage is everything, depth irrelevant (§1151–54).
**§1161–§1320 — The locality mega-arc and the named heads:** ALL 162 attention patterns replaced by
weights-only 128-token window functions for +0.0141 nats (§1166); all 18 MLPs ≤64-token windows (§1185); copy
regime = source-matchers vs successor-fetchers (§1215) with a three-sibling constant ~3.2 nats@W64 (§1212);
THE MATCHER AXIS — one projected-out axis costs 1.001 nats, 1250× the random null, restoring it recovers 95.3%
(§1249/50); the anti-criterion sign incident (§1238–40: flipping a pattern sign costs MORE than blinding it);
the named-head boom — closer 13.8 (96.5% of its layer), increment 8.7 ("+1 is in the weights", 8/8 digits),
question 10.5, comparative 8.1, openers 1.1/1.8, exclamation twins 17.2+17.3; "the twin is dormant" (14.4
perfect in weights, causally inert, §1279); no pivot-free fuzzy induction — the null is the finding (§1298).
**§1321–§1499 — Kits, THE BILL, and the glass ship:** THE BILL — 22-head commons + 6 specialists = 24.8M
params = 4.5% of the model buys 24.2 capability-nats (§1369); capitalized is a COMMITTEE (7→13 members,
§1397→§1418); the mids_reads INSTRUMENT RETRACTION (stale under λ-mixing, §1426) and the corrected diets
(§1427); mlp2 is 92% a linear map of [attn2,mlp1] — the biggest single prediction beat (§1437); the composite
fails 0/3 three times then lands — all 18 attention layers simultaneously +.162 CE (§1477); TOTAL GLASS =
5.5347 (§1499) → unit truncation → 3.88±.04 certified (§1542).
**§1500–§1613 — Circuit registry and the share-statistic war:** weights-only circuit discovery BEATS
data-driven (§1505/07); close_paren at 363×, pronouns 347×; the '?'-component of mlp11 is a rank-2 slice at
183× (§1573); the gates are interpretable axes (the-gate = determiner axis AUC .92; pronouns-gate = the
she↔he gender axis, §1583); then the writer-share crisis — random subspaces "find" the same top writers;
"72% in four components" declared uninformative (§1609); the §1612 retractions applied with Logan's approval
(§1653); the one survivor: the eigenvalue-ratio predictor replicates out of sample ρ=.678 (§1647).
**§1659–§1900 — Compiling the model:** mlp0 ~90% a current-token table (§1661); best MLP-stack program 56.29%
(§1670); 36 compiled sites reproduce 50.94% / 53.69% held-out (§1694/§1701); the only cheaply compressible
object in bilin18 is attention's ROUTING (§1692); the Down_bias retraction + the 3× parameter-count error
(§1714–§1723); named circuits do NOT beat arbitrary same-size sets (§1724); the causal-mask leak withdrawal
(§1729–31: future tokens in the induction mask); OAT and leave-one-out site importance are UNCORRELATED
(Spearman .026, §1737) and OAT INVERTS in program context (−.66, §1738); the 339× table-cost correction
(§1754); the standalone program that never calls a native module (§1777); the matched-rank law map_rank ≥
table_rank+1 (§1881); the program and the model are wrong IDENTICALLY at 6.1–6.5× chance (§1895) — settled
after a retraction-of-a-retraction (§1898→§1901→§1903).
**§1901–§2135 — Frontier bootstrap and the SIGN EPISODE:** tilt-axis era priced and closed (premise 2.7 nats,
recovered 2.4%, §1977); the converged-build SELECTION RETRACTION (§2037: fresh window loses by 11.8 mnat —
"the arc was selection"; registry audit 29→53 of 111 claims in the selection-noise regime, §2046/§2049);
gating rung strong null (§2081–84); metric-units positive certified label-free (§2116/§2124); **THE
SIGN-CONVENTION EPISODE (§2125–§2135)**: L2 = CE added above the real model, LOWER is better; §2128/29/33/34
mass-retracted; rung 41's K-0 null caught it; §2125's frontier (+2.6735 at norm-2304) reinstated.
**§2136–§2298 — Re-reading in the correct convention → the QK-rank grammar:** c8+c9@288 best (§2144); attn16
is FOUR heads (§2153); the pair census — every expensive attention function is an adjacent-block same-position
duo (§2182); the certificate chapter's strong null — "the elephant was the base": front tables alone cost
+1.924 (§2196/§2198); the CP-front grammar (m2's own top-K bilinear units crush the table 29×, §2203);
frontier +2.0553 → +1.8765 → +1.6599; the QK-rank grammar — all-QK-16 "the largest single move in program
history" (§2283); frontier +0.6412; THE FIRST CERTIFICATES (2/62, §2297).
**§2299–§2492 — Pareto set, manipulability, the context-metric era, tiers:** 11/62 at census +0.0553 (§2347);
the FALSE FLOOR correction — one legacy component a1v, not recompute structure (§2386); the honest storage
bill rejects both headline claims, 539.60M adopted after shifted-corpus OOD (§2392–94); contextual input
covariance opens the MLP0 frontier (§2422); the QK96→48 gated ladder; MDL — every fully gated Q/K rung is
MDL-optimal somewhere (§2455); the 1.025GB two-byte artifact adopted (§2459/61); full-stack shared bilinear
input-rank law (§2463); three-tier dial complete (§2475); the user's dossier memory recovers the 14,984-value
L16 program that dominates a 2.07M Tucker core by 138× (§2486); FIRST SUB-500M PROGRAM ADOPTED (§2492).
**§2493–§2557 — Exact algebra + attention0:** exact MLP0 context attribution to 2.88e-13 with interaction the
largest role (§2508); token identity 97% causally sufficient but linear transport fails (§2496); the coupled
continuous QK1×QK2×OV block preserves nearly the whole held-out downstream computation — "whole heads are not
the right basis" (§2544/45); head census closed via excess statistics (heads 7/4 lead, §2553).
**§2558–§2611 — Sealed protocol, equality circuit, and the endgame:** the sealed learned-simplicity bank built
but the rule NOT fit (§2558–74; vocabulary family strong null preserved un-rescued); THE EQUALITY CIRCUIT
(§2575–94) — flat interchangeable matcher (L5H5→L8H4 transplant), three-MLP correction group {8,9,12} at
cosine .992, two-site suppressor, 1,358 exact product terms causal on held-out code — then closed honestly:
code-selected indices do NOT transfer to natural text; composition claims are frame-relative (§2592);
native-coordinate/gauge nulls (§2595–98: products, sparse mixtures, commutant algebra all die); the TEMPORAL
REPRODUCIBILITY BREACH found, contained, and characterized (§2599/§2602: six processes bit-identical, cause
eliminated-to-unknown, 1.218×/layer depth law, in-run baselines now standing rule); attention0's
gauge-stable-but-semantically-unstable directions (§2600); MLP0 branches task-unselective at circuit grain
(§2601); tangent readers inadequate (§2604); categorical context tables close (§2607/08); and the finale —
the 487→490 claim-guard-refine cycle: midpoint interchange validates held-out (§2609), its T/I-specific
reading FALSIFIED by the registered confound guard (§2610), and the branch-resolved law VALIDATES
prospectively (§2611): T/I share a native-state reader (.97/.98) that excludes C, with strict correction
ordering T>I>C — A/B/C/D all true, the ledger's final entry.

### 8.2 What the program has claimed and kept (as of 09-02)
1. **The compiled-program frontier**: §312-lineage norm-2304 at +2.6735 (L2 = CE added above native, lower
   better); mixed-spectrum Pareto set to 11/62 certificates; three-tier dial; the adopted 1.025GB two-byte
   artifact and the first sub-500M program (§2492); byte tiers 62/62@1.0918GB → 17/62@495.8M scalars.
2. **The guarded projector claim**: on code, under projector-form removal, the three-MLP query interaction is
   source-aligned (.9389 pooled; halves .9692/.8743) — frame-stable because the equality query channel is
   effectively ONE-DIMENSIONAL per position (§2606's discovery: orthogonal component ~5e-7 everywhere).
3. **The branch-resolved MLP1 law** (§2611, prospective validation): T/I share a native-state response
   approximation (.97–.98) that does not cover C; finite corrections strictly ordered T>I>C (twelve
   inequalities, both quarters).
4. **The equality-circuit parts list** (code register): matcher + correction trio + suppressors, swap-tested.
5. **Laws and constants**: locality/window law; the ~3.2-nat copy-read constant across three siblings; the
   matcher axis (1250×); marginals-don't-compose (many independent measurements); compression lives in task
   geometry, never native coordinates (§2560, confirmed from four directions); the noise-floor depth schedule.

### 8.3 The program's biggest self-caught incidents (in one place)
- **The sign-convention episode** (§2125–§2135): an evening of "frontier improvements" that were certified
  damage; caught by a registered K-0 null; mass retraction; the convention now travels inline with every claim.
- **The selection retraction** (§2037–§2049): a converged build beaten on a fresh window; 29→53 of 111
  registry claims re-labeled selection-noise.
- **The causal-mask leak** (§1729–§1731): future tokens in an induction mask; withdrawal + autopsy ("four
  controls, none of which touched the thing that was wrong").
- **The instrument retractions**: mids_reads stale under λ-mixing (§1426); Down_bias omission (§1714); the
  339× table-cost accounting (§1754); the 3× parameter-count error (§1720).
- **The temporal reproducibility breach** (§2599–§2606): rung 474's own code stopped reproducing its own
  bundle (.084 nat); six processes later bit-identical; every named cause eliminated; contained by in-run
  baselines + a canary fingerprint tripwire; the Lyapunov depth law (1.218×/layer) retrodicts the old .015
  tolerance at depth 0.
- **Numbering defects (know before citing)**: §899 never written; duplicate §1075/§1614/§1616/§1715/§1716/
  §1729/§1771/§2386; §2034/§2050 never exist; §2036/§2040 cited but never written; backlog rung numbers
  duplicate (150 ×3); RUNG482 prereg missing from an otherwise contiguous 458–490.

---

# PART III — CROSS-CUTTING REGISTRIES

## R1. THE GRAVEYARD — closed directions, with receipts (do not retry without a NEW object)
**Toy/prehistory era:** curriculum for hop-3 (backfired 0/4); "hop-k needs k+1 layers"; copy-burst as an
induction lever (installs a positional copier); softmax-induction at small scale (descoped).
**Dictionary/SAE family:** activation SAEs as causal explainers (qk_mdl: 0/32 features clear the causal bar,
three rounds, two models); mechdecomp single-atom ablation (no power); two-stage merge→dict (retracted);
Archetypal-SAE on attention0 responses (strong null, 0-for-6 across geometries); BatchTopK for behavioral
compression (FVU better, ΔCE worse).
**Rank/decomposition family:** raw-weight embedding compression (e6); TT/HT beyond the ordering effect (e8/e11
null); dense coefficient-Frobenius Tucker at MLP2 (all mode ranks ≈ full); whole-layer R4k2 at L17 (rung 415:
0/62); Tucker rank-tuning at L16 ("do not tune"); native-channel selection at MLP2 (every rule worse than
deleting the layer); v1 factorization; metric-constructed bases/spans; half-price/K-reduction on the frontier
(§2118; §2133/34 retracted); conditioning on cfgE (§2132, zero).
**Gauge family:** exact rotation gauges (empty, tn_gauge F1–F8); DMRG re-fit iteration (a no-op, F39);
QK rotation under RoPE (broken by construction); the commutant/block route for the equality trio (§2598);
native product coordinates and sparse mixtures as circuit units (§2595/96).
**Tables/categorical family:** per-token content tables at the front (§1005-era); the eval-fitted bigram
comparison (retracted §1790–94, replaced leak-free); current-token tables for downstream branch effects
(rung 485: negative improvement); previous×current bigram tables (rung 486); tier-3 path-folded lookups.
**Circuit-selectivity family:** named circuits vs arbitrary same-size sets (§1724); per-module composition
(§1070: 12%); local fidelity composing (theseus-bench composite; §1477 needed joint training); mode-conditioned
stand-in selection (blocked, §2090); the m16 cheap interface (§2127); sink-head scalar (§2126); c6–c9
reordering (§2131); gating (§2081–84); the tilt axis (§1973–77).
**Bases:** eigenbasis-vs-neuron competition ended "confound unresolvable" (§2317/§2319).

## R2. DROPPED & FORGOTTEN — the things you asked about (ranked by how findable/valuable they look)
**Era-level, likely still worth something:**
1. **memorization_post Part 4** (capacity vs structure fraction) — designed, listed in the README, never built.
2. **e12** — the second-order read-Gram weight metric proposed by e10's negative; never built.
3. **tiny_full_interp leftovers** — X5 + vanilla-w192 ladders TRAINED but UNSCORED; the seeds3–5 fix-round
   chain stopped before training; STANDALONE §8's five-item list.
4. **thread-4 A5/A6/B0-third/B3/B4** (bilinear-quotient-experiments.md) — the original program's second half;
   receipts for a1–a8/b0–b3 exist but A6/B3/B4 "not started" per RESULTS.md.
5. **qk_mdl idea slates** — IDEAS_arch_slate S1–S4+A2 ("the model datasheet"); BRAINSTORM_STATE items 3–7 +
   parked; ROADMAP T4–T8; §S4 not-run-at-scale list; "Pythia HELD".
6. **The block4 fingerprints** (prehistory) — computed for 3 seeds, cluster labeling never ran (the explicit
   pause point); plus both *_gated_depth3 reports referenced by nothing.
7. **MLP dossier debt** — 12 of 18 MLPs have no consolidated dossier; MLP1 is the glaring one (11 preregs,
   ~20 receipts, second-largest family, no dossier); MLP4/MLP11 have real adopted results stranded in ledger.
8. **The five equal-price MLP2 programs** comparison (DOWN/FULL/RANDOM/CONTINUE/ROBUST-512) — explicitly
   paused for attention0; also MLP17 has NO current-harness Tucker result while its mode spectra sit unused.
**Questions raised and visibly dropped in the ledger:** §839 cross-model class steering (paused by user);
§1109/§1134 the gatherer's mass driver (bounded unknown); §1302 comparative screen; §1351 open-quote; §1363
atlas3 "the map's edge"; §2100/§2127 m16's per-document coefficient axis; §2161 the a16 constructive pause
(never lifted); §2321–2332 the index-table lane (stops mid-ladder); §2427 the sub-p448 MLP0 lane; §2533 the
.92359GiB near-miss artifact; §2551's open taxonomy question; §2556's +29,968-byte repair lane.
**Small but real:** explanation_0428.md unindexed; explanations README structural drift + duplicate-basename
links; PENDING_RETRACTION_S1612.md still sitting at root (applied per §1653 — file never moved); the empty
qk_mdl QUEUE.txt; ko_a14/a17_cfgmean superseded-unrun; das_m16_* scripts built never run; covcache.py +
scoring.py shipped with zero adoption; the ECG .out stubs whose scripts were never committed; tensor-sim's
two unanswered asks (SVHN; two-real-layer composition); the 7-ring inversion; GPT-2-numerals-without-a-map.
**Provenance hazards:** 469 orphan receipt stems (245 bilin18_* with zero scripts — the entire early bilin18
era is receipts-only); runs_lm/ gitignored (prehistory numbers unreproducible); e6b/e6c/e7b/e7c mutate shared
JSONs; the §1–§803 'gap' is mostly a deliberate numbering jump (probe 09-02: §200–§803 never existed in any commit; §100-era = qk_mdl).

## R3. WHAT STANDS — the positive results worth building on (repo-wide)
**Mechanistic findings:** induction reverse-engineered with the XNOR signature; the locality law (+0.0141 for
all 162 patterns as window functions); the matcher axis (1250×, 95.3% restoration); the named-head roster
(closer/increment/question/comparative/openers/exclamation twins/sink 5.7/announcer 12.4); the v1-router
principle; the copy-read family constant; L5H7's one-vector function; the equality-circuit parts list; the
continuous QK1×QK2×OV block; attention0's two gauge-stable downstream-fixed directions; the one-dimensional
equality query channel; the branch-resolved MLP1 law; the site-graded T/I filtration (.51→.62→.79).
**Program/pricing results:** THE BILL (24.8M params → 24.2 nats); total glass 3.88; 36-site compilation at
53.69% held-out; the QK-rank grammar and 11/62-certificate Pareto point; the three-tier dial + sub-500M
program; MDL-optimality of the gated Q/K rungs; the theseus anchors (198/198) + mlp1 plank .9507@96.6Mbit.
**Meta-scientific machinery (arguably the most durable output):** the preregister/score-as-written/preserve-
failures pipeline; Möbius/interchange/commutant/projector instruments; the noise-floor depth schedule; the
canary + determinism fingerprint; LESSONS.md; the four-ledger and dossier disciplines; the k·u² deployed-dtype
bar convention (vindicated by independent validation in rung 488).
**Era findings that still frame everything:** THE METRIC DECIDES WHAT EXISTS (e4/e5, tensor-sim F13, qk_mdl
FVU-vs-ΔCE, e6's subtraction≫addition); sparsity is gauge (e1/e2); marginals don't compose (everywhere);
the foldability tax; "decompositions survive, absolute wins don't" (scale box); nameable selection over
spectral content (the meaning boundary); reversibility pins representation-organization sign (prehistory).

## R4. RETRACTION & CORRECTION LOG (highlights; the full trail is in the ledger)
By type — **Selection/leakage:** §901 (81%→0.29), §913, §2037–49 (the registry audit), §1729–31 (causal-mask).
**Sign/convention:** §1238–40 (anti-criterion), §1563/§1565/§1587/§1605, §2125–35 (the episode), §1907.
**Instrument:** §1426 (stale ledger), §1148 (blind readout), §2225/§2235/§2238 (void runs), §1838 (VOID),
§2371/72, rung 487's BF16 bars (repaired by derived k·u²), the b-variant's extreme-value bar (mine).
**Accounting:** §1714–23 (Down_bias + 3×), §1754 (339×), §2386 (a1v false floor), §2488 (literal-price),
§2531 (bf16 storage), theseus's +2.77/+2.82/+2.84 sign-error frontier notes (reverted).
**Overclaim trimmed:** §836, §805-era cascade, §873→§874, §1008, §1152, §1175 (published-claim correction),
§1858, §1878, §1898→§1901 (retraction un-retracted), F10 (tensor-sim), F14→F16 (tiny_full_interp), e8's
param-efficiency, jacclust's rotary-sign inflation, my D×A misread (M-led, corrected by Codex), my §2603
geometric reading (killed by my own mirror test at 5 orders), my 488 over-broad interpretation (guarded by
Codex's §2609, falsified by 489).

## R5. METHOD FAMILIES ACROSS ERAS (what we tried, wearing different hats)
- **Dictionaries/SAEs**: mechdecomp (action-SAE) → jacclust operator-SAE → qk_mdl folded-factor dictionary
  (the one success: ~zero-cost at 6.1% bits, readable atoms) → residual SAE causal nulls → Archetypal null.
  Verdict pattern: dictionaries reconstruct and name; they explain causally only when the object is folded
  weights, not activations.
- **Low-rank/tensor decompositions**: e6 SVD/k-means → e8 TT/HT → tn_gauge used-subspace (the success:
  QK 28%@+0.06) → CP-front grammar (§2203) → QK-rank grammar (§2283) → Tucker nulls at MLP2/16/17.
  Verdict: rank works on ROUTING/selection maps; it fails on content and whole layers.
- **Gauge/invariance**: e1/e2 → balanced_gauge spec → tn_gauge → the §2592 frame-relativity discovery →
  projector/commutant instruments → the one-dimensional channel resolution. Verdict: gauge questions ended
  as measurement discipline (state the coordinate or quotient it out).
- **Tables/categorical codes**: front token tables → class packages → the compiled table+map programs
  (§1746–1883, the standalone program) → token/bigram closure at branch grain (485/486). Verdict: tables
  carry the front and the compiled program's spine; they do not explain downstream USE.
- **Causal factorials/interchange**: DAS era (§2054+) → interchange instrument (§2257) → Möbius subset
  factorials (472–474) → secant factor interchange (487/488) → the claim-guard-refine template (488→490).
  Verdict: the program's current core method; where all three standing claims came from.
- **Benchmarks/accounting**: MDL tiers (qk_mdl) → the four ledgers → frontier/certificates → theseus-bench.
  Verdict: pricing keeps everyone honest; "unexplained CE" ranking is the reusable artifact.

---

# PART IV — THE BILIN18 MODULE MAP (dossier digest, 2026-09-02)
Native price per MLP: 15,926,400 stored numbers. Dossiers exist for 6 of 18 MLPs (see R2 for the debt).
- **MLP0** — quadratic feature generator, NOT a token map. Exact grammar G = μ+T+C+I+S (2.88e-13); held-out
  Shapley: I=1.538, T=1.498, C=.418, S=.067 (interaction leads). C512 replaces Down at 72% smaller. Rank-64
  causal output subspace ≈80% of effect. Early MLPs strongly non-additive (surplus .458). Token-only anatomy
  closed (M+L+Q; keep exact Q). Branch effects task-UNSELECTIVE at circuit grain; consumers read T/I nearly
  identically at MLP1-direct (.9998 shape) but with uncorrelated magnitudes.
- **MLP1** — the highest-stakes un-dossiered module: Δ_opt 7.25 nats (theseus), token table ~.93, glass plank
  .9507@96.6Mbit; the secant/interchange arcs (487–490) live here; T/I native-state reader law validated.
- **MLP2** — native-channel selection CLOSED (every 512-rule worse than deletion); dense Tucker closed;
  learned mixed 512 works (FULL512 .0515) but RANDOM512 nearly ties (no privileged atoms); five-program
  comparison paused. 92% linear in [attn2,mlp1] (§1437).
- **MLP8/9/12** — the equality correction trio: group cosine .992 with the four-context correction, ~47% of
  magnitude, source-stable .994; term indices register-specific (code≠natural); representationally individual
  at every tried grain (products, mixtures, blocks); functionally composable.
- **MLP16** — 14,984-value quadratic surrogate (corrected price), clean-split R² .82, census +.039, 27/62;
  rank-1 activation-conditioned Down recovers 90.2% of benefit (sentence-ending direction); part of the
  adopted lower-fidelity tier; "do not tune Tucker ranks here."
- **MLP17** — output concentrated (rank-4 Down = 83%) but the low-variance tail is functionally important;
  the L16-style 14,984-value object FAILS here (R² −29.7, 0/62) — the historical result was overlap-
  contaminated; no current-harness Tucker result.
- **Attention0** — continuous 6/6/32 interface (99.03% edge signal, +.0002 nat CE) with two gauge-stable
  downstream-fixed score-branch directions (eigengaps 5.7/4.9) whose circuit semantics don't transfer;
  QK normalizers broad (r90≈101), closed three ways; sparse global vocabulary document-stable.
- **Attention heads (named)**: 5.7 sink/bias-head (one vector .985); 13.8 closer; 8.7 increment; 10.5
  question; 8.1 comparative; 1.1/1.8 openers; 17.2/17.3 exclamation pair; 12.4 pronoun announcer (347×);
  L5H5/L8H4 equality matchers; 14.4 the dormant twin.
- **attn5** — the price cliff: its write is where compile-depth breaks (85% of norm on head 5.7; a 159× gain,
  not dispersal; the fall actually at layer 1 per §1829).

## Appendix A — Ledger/§ namespace map
- **BILIN18_CONNECTION.md** (bilinear_quotient/): §804–§2611, 1,811 entries, started 08-21 (§804 = gpt2-medium
  mlp0 diagnosis). Prologue: "what the toys say about bilin18". §1–§803: RESOLVED (git-pickaxe probe 09-02):
  §200/§300/§400/§500/§700/§803 never existed in ANY commit; the §100-era lived in qk_mdl/RESULTS_l0_mdl.md
  (07-31). The range was never a contiguous ledger — §804 was a NUMBERING JUMP continuing qk_mdl's ~§100-max
  namespace with an offset. Internal citations to §429–432/§513/§546/§649 refer to entries that exist only in
  the 245 orphan bilin18_* receipts + git history of since-trimmed files; most of the "missing" range is
  deliberate offset, not lost content. Rung numbers: backlog rungs 1–390 (duplicates exist) and experiment rungs
  →490 (RUNG482 prereg missing).
- **qk_mdl**: separate small namespaces — RESULTS_l0_mdl.md §32–§39d (89 entries), LOG.md §21–§43,
  redteam §40–41, RESULTS_scale_draft §S1–S6. Commit messages 07-29/30 reference §50–§80.
- **theseus-bench** priorities cite S-numbers (S1437, S1474...) = this ledger's § namespace.
- **AGENT_BOARD.md** (+ archive 08-31, 18,381 lines): the two-agent (Claude/Codex) coordination channel.
- Key knowledge stores: polynomial_causal/explanations/ (90+ indexed explanations, 8 eras), the 5+index
  module dossiers, 154 preregistrations, 165 hourly + 32 mathematical reviews, RESULTS.md files per era.

## Appendix B — Repo map (where things live)
/workspace/tensor_language — main repo (6,800+ commits, 06-01→). Root = prehistory + spillovers; archive/ =
frozen geometry program; mechdecomp/, jacclust/, tensor_sim_regularized_bilinear_transcoders/ = July method
programs; basis_aligned/ = e-series + tn_gauge + tiny_full_interp + memorization_post + qk_mdl +
bilinear_quotient (engine room: 2,270 scripts, 2,612 receipts, ops/ toolchain, ledger, backlog, LESSONS) +
polynomial_causal (knowledge base: facade, dossiers, preregs, explanations, reviews).
/workspace/theseus-bench — separate repo (379 commits, 08-25→): SPEC, anchors, priority board, registry.
Toy runs: runs/ (L1 sweeps), runs_gen/ (bilin-add/lerp), runs_geo/, runs_llm/, runs_hop/, runs_markov/,
runs_owt/; runs_lm/ was gitignored and is absent.

*Compiled by Claude (red-team/ops lane) from eight parallel repo surveys + the commit history + the live
session record. Corrections welcome on the board; § citations verified against the ledger where quoted.*
