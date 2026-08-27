# bilin18 — consolidated findings index

The working index. The full ledger (BILIN18_CONNECTION.md, 1090+ sections) is an append-only
lab notebook, not a place to keep up with results. **This file is the ~10 things that matter,
their confidence, and what's open.** Update this in place; don't let it grow past ~2 pages.
CONSOLIDATED 2026-08-23 (was 29 items; §1092 exposed that the index had dropped the sink arc,
causing a 4-experiment re-derivation — every major arc is indexed here now).

Confidence: **HIGH** = causal test + control + null, reproduced. **MED** = solid but one caveat.

**PER-MODULE DOSSIERS: `modules/` — self-contained doc per module (facts+numbers+§refs,
stand-in status, module-specific gotchas, open questions). READ THE RELEVANT DOSSIER BEFORE
DESIGNING AN EXPERIMENT on a module, and update it in the same commit as the ledger writeup.
CIRCUIT WORK: `CIRCUIT_REGISTRY.md` is the one status board (pipeline stage per circuit +
candidate pool + generators) — check it before opening a circuit thread, update it when a
circuit moves stage.**

## The established results

0. **THE CIRCUIT-KIT ARC (§1311-1372, 2026-08-24/25) — capabilities as kits on a commons.** Behavior-first
   screens + a route-grain extraction template produced: ~8 named specialist FUNCTIONS (closer 13.8
   [brackets+quotes, 107.9% solo], comparative 8.1, question 10.5, expressive pair 17.2/17.3, numeric
   pair 8.3/8.7 [copy/fresh split], matchers 1.1/1.8, sink 5.7, router 0.3); a 22-HEAD ANNOTATION
   COMMONS that serves every kit BETTER than dedicated bands (§1367-68); three kits at 0.66-0.88
   recovery for ~25M attention params (4.5% of model, §1369); laws: criterion scope predicts annotator
   depth (4-for-4, §1358), owners are annotator-conditional (4-for-4), width inversions (6x — kits
   have interior optima), never gate a constant (§1367), ablation names roles / construction prices
   kits (§1347). TRANSFER (§1373-82): the CLOSER replicates across FOUR sibling models
   (2 score fns x 2 MLP types) as one surgical owner head at mid-late depth with net-negative
   neighbors — existence/grain/depth are TRAINING-PROBLEM properties, strength is
   architecture-modulated; the extraction grammar's laws (query-side dominance,
   owner-as-liability) replicate on bilin12. BOTTLENECK: the generic middle is ONE redundant service with TWO implementations
   (mid attention + mid MLPs) — extraction removes either half, never both (§1370-71); attention
   compresses, the MLP side does not. Dossiers: CIRCUIT_REGISTRY.md, MDL_BILL.md. **HIGH.**


1. **THE TWO MACHINES — the organizing fact.** bilin18 = a GRAMMAR machine (front L0-1, low-rank,
   local, token-driven, near-linear per layer, ~90% understood as smooth per-token functions §905/§941,
   §1088: L1 tok-only held-out 0.93 of a 6.5-nat stake) + a CONTENT machine (deep-middle L5-14,
   high-rank topic/register manifold, long-range, multiplicative) + INDUCTION (front chain, third
   mechanism §952-954/§1025). Grammar = the easy ~23% of loss, content = the hard ~77% (§829-831,
   universal). Readout L15-17 = near-linear read (§1046); grammar owns the head of the distribution,
   content the tail (§1033/1034; the merge is additive-linear, logit-delta vs −W·c cos 0.77 §1082,
   but argmax fragility is magnitude-generic, not content-specific §1086). Errors are content errors:
   class-correct 2/3, calibrated hedging to function words, graceful topic-neighborhood misses
   (§972-980). Division of labor: early ATTENTION gathers content seed (§1074), front MLP writes
   grammar (orthogonal), deep-middle MLPs MULTIPLY content (context×context §1041), readout reads.
   Depth story in MLP terms: smooth tok→context gradient, NO binding band (cross-terms flat ~0.2
   §1084); L4 = first true context MLP (§1084/§1088); variance-share tables are in-sample-only
   (§1090), held-out CE is the standing per-layer fact (§1088). mlp16: 82% token-variance output but
   the 4%-variance dev part carries ~87% of its CE value (§1090). **HIGH.**

2. **THE CONTENT FRONTIER — located, named, causal, universal, register-adaptive; bounded by real
   high-dimensionality (§1036-1081).** One shared, drifting, load-bearing content subspace across
   L6-14 (overlap 0.58 vs 0.05 null §1049; low-rank bottleneck anywhere starves the band §1051),
   born gradually L3-5 (§1052), a cumulative residual-stream object (not authored by middle attention
   §1053). NAMED: high-dimensional semantically-organized topic/register manifold, top axes
   interpretable, top-10 PCs only ~12% var (§1055); genuine semantic topic not surface register
   (~2% §1064); multi-scale, predominantly running local context (§1065). CAUSAL: ablation
   catastrophic + privileged over random across the whole spectrum (§1056); activation patching
   TRANSPORTS topic (62×/22× random @K16/64, §1059-1060; replicated 4th-decimal on fresh rows §1150).
   Transport is POSITION-BOUND: shuffling which position gets which content vector kills all excess
   (goes below random); a per-sequence average carries zero excess; rank-8 slice already carries 78%
   of rank-16's alignment (§1150 — in-protocol resolution of the §1146-49 nullified sub-thread).
   UNIVERSAL across the family: same info
   (CCA 0.95-0.97 §1061), functionally interchangeable cross-arch and cross-width (93-96% of
   within-model §1062/§1063/§1066); supports content/topical-word prediction (clean K=16: 7.6× rare
   vs frequent §1068, correcting §1067's destructive-regime artifact). ORIGIN: not a raw-word bag
   (R²~0 §1072); progressively linearly built from the PROCESSED stream (0.38@L1→0.88@L7 §1073);
   content = pooled bag of block-0's STATIC per-token c_v values broadcast by the value-residual
   (§1076). REGISTER (4 registers, §1079-1112): grammar
   consistently MORE register-general than content (1.6-2.1× all pairs — no fixed constant, §1111
   retracted the 2.05 'law'); content band prose-specialized; value-residual the most
   register-robust channel; front tables register-CONTEXTUAL w/ breadth asymmetry (§1112).
   **SKELETON/SCRATCH (§1113-1118, the naming capstone):** the content code = ~8 STABLE features
   (cross-seed 0.76-0.81; 3.6× PCA; 6/8 human-named: topic/register/discourse mix) + dense tail.
   Reads are SPARSE everywhere (tail only 10-15% of read-CE at deep MLPs / readout MLPs / final
   logit path — which cross-checks §1082); construction is MIXED (stream-level: skeleton-removal
   97%, tail-removal 46% — the tail works in transit, read by nothing). Skeleton = the API,
   tail = the scratch space; 'irreducible high rank' re-scoped to SIMULATING construction, not
   READING the result (§1056 reinterpreted as construction disruption). **HIGH.**

3. **ATTENTION — fully resolved at head level. One constant + a few routers + redundant collective
   pooling.** (a) THE SINK (§429-432 + sink arc; §1083-1092 re-derived — see dedup lesson): head 5.7
   costs 0.92 nats, 8× the next head; it reads POSITION 0 for 99.8% of queries; the value parked
   there is manufactured by MLP4 (norm 155k, direction cos 0.998 across documents); its output is a
   CONSTANT (mean-replacement free; donor text's mean works); the constant IS the stream's baseline
   (cos 0.99 with mean residual L6-11, 62-72% of magnitude; top coords on the massive/gain dims);
   partial removals/truncations are WORSE than deletion (being the baseline is not a local property;
   §1091 sparse-truncation, §1087). (b) BIAS MAP (§1091): 62% of summed per-head causal weight is
   bias-value (L5H7 = 86% of it); TRUE dynamic value is thin: L0H3 0.079 (a genuine router; = a
   bigram LOOKUP TABLE, exact at zero cost — prior sink arc), L2H5, L1H1, L6H3… nothing else >0.02.
   (c) COLLECTIVE (§1093): all-heads-static costs 3.67 nats = 5.7× the per-head sum — the middle's
   pooling is collectively load-bearing, per-head redundant (no indispensable courier; §931/§952-954
   super-additive theme); stale biases are worse than no attention (all-zero 3.42 < all-const 3.67).
   (d) CRITERION: QK = signed conjunction of two bilinear scores (anti-heads possible §981-984);
   routing modes are RECENCY and INDUCTION, not content-similarity (§983; §1085: middle pool
   positional-first r −0.39..−0.61, content-sim second-order but growing with depth, causal 1.8-4×);
   ~44% of heads factorize positional×content (§684-685); focality from the product (§681-682).
   (e) WINDOWS: every genuine content read fits a 4-token window except the position-0 constant
   fetch (17/18 layers <0.1 nats; prior sink arc). Induction heads exist but are redundant
   (L2h5 top, L5h5, L8h3/4; single-head max +0.12 vs collective +5.2 §953-954).
   (f) FOLD CAPSTONE (§1161-1166): attention SELECTION is a bounded-window weights function
   MODEL-WIDE — 162-head map: every layer's pattern argmax 0.8-1.0 predictable from weights over
   the last ~128 tokens (sink 0.998; exact-prefix fix after a pad-clamp artifact); CAUSAL: run the
   model with ALL 162 patterns replaced by window-folded reconstructions → +0.0141 nats (front-5
   +0.001; wrong-text null +1.513). Pattern-locality is log-local (~+0.06 hit/doubling of W; loss
   4-local but written code ~64-128-local — two-scale front law §1162). Long-range behavior lives
   in VALUES/content, never selection — PRICED (§1180-81, family-constant to 3rd decimal): whole
   model on 128-token windows costs 0.082 nats (selection 0.014 + values ~0.07); smooth decay, no
   plateau, zero at trained context (§1182). EVERY MLP is a ≤64-token window function (§1183:
   deep entries 0.002-0.006, decreasing with depth — MLPs process, attention accumulates).
   GRAND STACK (§1184): 12 reductions jointly (162 folded patterns + sink constant + 9 n-gram
   MLPs) = 0.0385 nats, sub-additive at all six composition steps. Front n-gram width is
   REGISTER-DEPENDENT (§1175-76 correction to writeup 480: k=2 prose / k=8-16 structured; full
   fold exact on all registers). COPY REGIME (§1204-12, the one place locality breaks — 30×):
   read-grain map on verbatim-repeat rows = SERIAL chain, not prose's redundant crowd; reader
   STATIONS localize — bilin18: heads 2.5 + 3.8 (L3's ENTIRE read; complement 0.004) + pair
   8.3/8.4 = 69% necessary / 59-41 partition vs distributed tail (§1207/09); stations do double
   duty on prose (22% of the 0.176 budget, §1211). THREE-FAMILY LAWS: total copy read price
   ~3.2 nats @W64 in all siblings (3.200/3.206/3.231 — invariant to arch AND induction strength
   4.3-vs-11.8: text-set, not machine-set §1212); ~2 dominant stations each. MECHANISM
   (§1215-18): one algorithm, two implementations — bilinear-scored models (both) MATCH the
   source by direct long-range reads at o=128 then FETCH the successor at o=127 mid-stack;
   the softmax model has NO matcher (all stations o=127 fetchers; local key-composition,
   textbook) — the SCORE FUNCTION decides, normalization doesn't (bilin12 = the deciding
   case). Fingerprints: station depth + sharpness (bilin18 single heads > bilin12 pairs >
   crowds); bilin12's late band NOT local (17%); L3H1 = matcher's auxiliary, toxic without
   it (§1213). **HIGH.**

4. **ARCHITECTURE MECHANISMS — how the two machines are grounded in bilin18's oddities.**
   (a) VALUE-RESIDUAL (v = ½v + ½v1, block-0 values everywhere): the content-aggregation substrate;
   ablation +3.3 nats, content-tilted (rare/freq 2.7); block-0 value = EXACTLY static per-token
   content (§985-987/§1075-76; §1075 was a partial dup, corrected: both channels content-heavy,
   x0 only RELATIVELY grammar-tilted). (b) x0 RE-INJECTION (λ₁≈8 every block): keeps the embedding
   ever-present — current token linearly recoverable at the FINAL residual (R² 0.73 §690); why class
   is re-derived every block; ablation +2.3 nats. (c) MASSIVE ACTIVATIONS (dims 645/990/981…) = the
   rms-norm GAIN CONTROLLER, not attention sinks (no softmax; §676-680), written by the multiplicative
   gates (§688-691), delivered from position 0 by mlp4→head-5.7 (item 3a); they host w_freq (88%).
   (d) BILINEAR GATE: front MLPs = genuine two-distinct-factor conjunction (force self-product
   catastrophic +2.4); deep-middle gate diffuse; mlp0 = mix of ~1% self-square class units + a
   conjunction majority (§1077-1078, reconciling §842). (e) Residual rescale x = λ₀x + λ₁x₀; front
   λ₀ near-zero resets; logits = 30·tanh(lm_head(rmsnorm(x))/30); MLP outputs are exact quadratic
   forms; mlp17 variance-rank 8 ≠ functional rank (§615/§660). **HIGH.**

5. **HOW MUCH DO WE UNDERSTAND — the benchmark (0=mean-ablate, 1=full).** Honest unit = the VARIABLE
   (read/write subspace), verified by INTERCHANGE (class IIA 0.25 §892; topic +0.70 nats §894) —
   steering fails because read≠write. Whole-model simultaneous held-out: **0.32 ± 0.06**
   (draw-sensitive; topic term drives spread; §1013-1014); progression 0.81-insample(retracted)
   →0.30→0.42(map). Per-module-in-isolation is the VALID measure — per-module understanding does NOT
   compose (whole-model greedy substitution 12%, compounding-dominated; content passthrough triples
   to 0.39 §1070-1071). Per band: front grammar ~0.90, readout ~0.56-0.9, middle content ~0.10 —
   the middle's gap is the content's REAL high rank (three-way confirmed §1000/§1038/§1042), not an
   instrument limit. Linear ceiling: all-MLPs-linearized costs +1.55-1.59 nats of content (§1000-1003);
   front content splits ~18% token-lookup / 29% context-linear / 54% context-multiplicative (§1005).
   L5H7 is ~98.5% understood as a constant (§1089/§1091). **HIGH.**

6. **CLASS+POSITION PROGRAM — with the §836 correction that guards it.** The front sorts what it
   writes by token-class on a canonical seed-free token-conditional-mean SUBSPACE (necessary 268×
   random, nameable: lexical for MLP, structural for attn §767-772); class steering is class-specific
   and drives grammatical sequencing (§823/837/838); class = computed collapse (~132→24 eff-dim,
   sharpened 1.8× §780-782); position = coarse 2-dim RoPE readout. Cross-model common (6 models).
   **CORRECTION (§836): every keep-only recovery magnitude (0.78/0.92) is a RANK/CONSTRUCTION
   ARTIFACT — a shuffled-label matched-rank subspace recovers the same. Never quote "the model is
   78% class+position"; the low-rank structure is class+position by NAMING and STEERING, but the
   keep-only fraction is unattributable.** Right keep-only null = shuffled-label matched-rank.
   Stack shape: barbell with a super-additive class+position-MAINTENANCE middle, universal across
   6 models (§809-820); per-component keep is MEAN-dominated (§821-822). **HIGH (with correction).**

7. **DISTRIBUTED-COOPERATIVE COMPUTATION + read≠write — bilin18's signature and the method laws.**
   Single components near-free, ensembles load-bearing (super-additive: content MLPs 4.9×, attention
   3.5×, induction, middle bands ~4×; OPPOSITE of GPT-2's sub-additive redundancy §956/§965) — why
   single-unit ablation comes up empty; honest units are ensembles/subspaces/variables. One cleanly
   isolable linear knob exists (frequency-calibration w_freq, rank-1, ~6% of loss-benefit §650-651);
   everything conditional/stateful has NO removable linear carrier even when decodable (quote-parity
   AUC 0.83 causally inert §668; parenthesis depth §669) — decodability ≠ causality; probes read,
   unembedding rows write, ~orthogonal (§619-622). SAE atoms are the wrong unit (seed-unstable);
   the canonical token-mean subspace is the right one (§750-772, item 6). **HIGH.**

8. **NAMED CIRCUITS (each causally verified by output selectivity, not firing).** Newline: `.`
   trigger 28× → front attention discriminates line-ends (AUC 0.81→0.51 chance on ablation §728) →
   block-17 calibrates. Article: attn=choice, mlp=magnitude, prep→the (§729). Digit: continuation vs
   initiation, two circuits in one class (§641-642). Frequency calibration: block 17 dominant, rank-1
   w_freq, distributed component in 5 layers (§624-662). Induction: strong (score 11.8, scales with
   size §880-885), distributed-cooperative with identifiable-but-redundant heads (item 3e), asymmetric
   coupling: content amplified ~1.5-2× when induction fires, induction independent of content
   (§1027-1032). Circuits bottom out in embedding trigger-geometry routed by context (§637-644).
   Loss budget: first-mention 78% / seen 20% / inductable 1%; invariant across the family (§876-885).
   2026-08-24 additions (behaviour-first SOP, all invisible to magnitude screens): 13.8 =
   delimiter closer (§1270-74), 8.7 = general successor head, four lexicons, "+1" in W_v·W_proj,
   dormant weights-twin 14.4 certified vestigial at 10x data (§1275-83); 10.5 = question head
   (§1282-89). ANNOTATION SERVICE (§1289-98): heads 1.1 (identity mark, local) + 1.8 (context
   signature, global) — either sufficient on natural text, THREE consumers (matchers, 13.8, 10.5),
   redundant beyond the pair (§1299 caveat: keep-none baselines are ~70% generic front-band
   damage — source-specific excess ~1.25 nats; within-baseline rankings stand), one-armed double dissociation (identity generalizes to context-free
   repeats 110% vs 33%; signature is NOT an independent pathway — no pivot-free fuzzy induction,
   base=chance §1298; signature actively harmful off-distribution §1297). Opener corruption poisons
   via mlp4 re-encoding (94% restore when blocked §1288). HEAD-PARTITION LAW (§1290-94): task-parts
   are weights-derivable at BOTTLENECK VARIABLES (verdict axis, successor maps), not at TRANSPORT
   (identity is full-rank, depth-recoded); partition variables, not pipes; long tail = identity
   crowd (1.1 nats, 60:1-null instrument). GOAL-3 LOOP (§1307-08): the matcher is a STEM
   matcher, read from weights (collisions = inflections), certified causally on natural text
   (variant targets 78% of identity strength, inside the registered band). **HIGH.**

9. **GENERATIVE (input-side) VALIDATION.** Injecting a topical word early boosts its topic-neighbors
   at distance with the full predicted property set: dose-linear, additive multi-topic superposition
   without dilution, recency-weighted broad receptive field, content/grammar separability 77×,
   architecture-general; grammar validated symmetrically (adjacent determiner → P(noun), far does
   nothing); induction validated (AB…A→B +7-8 nats) (§1016-1026). **HIGH.**

10. **METHOD LAWS (each learned the hard way — check before designing).**
   (a) DEDUP before building: grep the ledger with MULTIPLE vocabularies (sink/constant/cost map/
       dotted X.Y head notation — old arcs predate § numbering); §1092 = 4 re-derived experiments.
   (b) Registered predictions + controls + nulls; the redteam catches ~half the headlines
       (§1067→1068, §1082→1086, §1084→1088, §1089→1091).
   (c) In-sample per-token means LEAK on singletons (~20% of positions) — held-out means for any
       tok/dev decomposition (§1088); variance-share tables in-sample are contaminated (§1090).
   (d) OFF-MANIFOLD partials: removing part of a load-bearing constant/baseline costs MORE than
       removing all of it (§1087/§1091/§1093 stale-bias inversion) — never read partial-ablation
       fractions as shares.
   (e) Banding vs output-ablation FLIP on redundant parts (§1008-1009); keep-only needs
       shuffled-label matched-rank nulls (§836); firing ≠ function (§726-727); variance rank ≠
       functional rank (§617/§660); destructive-regime artifacts at high K (§1067-1068);
       benchmark stand-ins must match the removal point/scale (§1066).
   (f) Per-head/per-component sums ≠ collective cost (5.7× gap §1093) — measure both granularities.

11. **COMPRESSION: IT HELPS, BUT NEVER SPECIFICALLY AT CIRCUITS (§1593-96 + the 2026-08-27
    polynomial track).** Four independent probes, three different senses of "compressible",
    pointing three different ways. (a) COMPRESSION IS EXTRACTION (§1594-95): the rank-32
    whitened-QK background alone preserves **97%** of question class function (class rise .190
    vs 5.996 under optimal constants) and substituting the exact circuit heads back adds
    NOTHING (−.077, i.e. trivially worse); on pronouns the exact heads are ANTI-extraction
    (rec −.29). A good compression already IS the extraction. (b) BUT IT IS NOT CLASS-SELECTIVE
    (§1596, the load-bearing negative): across r ∈ {4,8,16,32} the class rise ≈ the GLOBAL rise
    at every rank (.99/1.03, .49/.49, .19/.16). The circuit does not break before the model
    does — it degrades in lockstep. There is no privileged fragile core to cut out and no
    privileged robust core either. r8 is explicitly insufficient ("needs the full rank-32 tier").
    (c) DISCOVERY DEGRADES FASTER THAN FIDELITY (output_slice_audit, 2026-08-27): a rank-8
    class-trained output basis recovers **exactly half** of oracle head-recall (97/30 vs 97/15 —
    an exact-arithmetic tie, its bar passed at 0.000 margin) while delivering only **13.5%** of
    oracle removal damage, though at 8.6× random selectivity. It finds heads that are clean but
    not the heads that carry the damage; it misses head 13.8 entirely (oracle 1.000, winner
    0.000) — the §1515 single-head close_paren circuit at 363×. (d) THE ALGEBRAIC SAVINGS ARE
    REAL BUT SMALL AND GEOMETRY-BLIND: a real quadratic with inertia (p,q) needs max(p,q)
    products, not rank(S) — verified both bounds, so the certified rank-2 question form is **ONE**
    multiplication, a true 2×. Yet a single SQUARE, **32% wrong** in scalar reconstruction,
    retains **99.3-99.6%** of that slice's behavioural effect: the hyperbolic geometry that makes
    the theorem interesting is behaviourally irrelevant, and the saving is in the product COUNT,
    not the sign structure. Against that, MLP product rank is FULL (1152/1152 at layers 0,1,2,11,17,
    ≤4.21× bound) — but under GAUSSIAN probes, which measure coefficient-space rank, not
    natural-activation fidelity, and (a) falsifies the naive reading empirically. And a learned
    paired-product content decoder is DOMINATED by a plain linear map on both axes (heldout R²
    .542 vs .639 at 75,840 vs 73,792 params). **Bottom line for the bench: compression buys real
    fidelity cheaply, buys a bounded 2× on quadratic slices, and buys nothing circuit-specific —
    class and generic function move together at every rank tested.** Dossiers: registry
    `_extraction_asymmetry`, `_discovery_compression_ranking`, AGENT_BOARD 2026-08-27. **HIGH**
    for (a),(b),(d-inertia); **MEDIUM** for (c) — one harness, one rank, bar met at zero margin.

12. **Null-methodology arc (§1623–§1638): a matched-rank random arm drawn from ONE basis is not a
    control, and the class-type of the control decides the answer.** Twelve registered runs on the
    eigen-slice separation statistic, all on the canonical `.rowcache` FineWeb tensors, corrected
    quantity (forward stops at the site, upstream components, site-relative coefficients).
    **(a) Published figures all replicate exactly:** §1597's .718 → .7179, §1598's .482 → .4823,
    §1597's 20:1 head-grain → 20.00. What changed across this arc is INTERPRETATION, never a
    published measurement. **(b) One random basis is a sample of size one.** Seed 1729 is
    unrepresentative in a direction that varies by SITE (high at mlp11, low at mlp17) and by CLASS
    (−.0568 to +.0141 within mlp11 alone), and it UNDERSTATED both published effects. Use ≥20
    independent bases; report a FRACTION for identity claims and a MARGIN for magnitude claims.
    **(c) Identity is nearly free, magnitude is informative:** attn10 sits in a random top-4 in
    53.3% of 60 trials, attn9 88.3%, mlp16 100%, head 10.5 in 100% — while λ-vs-random SHARE
    separates cleanly. Naming a component "rule-specific" because it appears in a top-K is not a
    control. **(d) But separation margins are cell-dependent and NOT generalisable from certified
    cells** — 60/60 at two published cells, 54/60 at a fresh one. **(e) AMENDED BY §1639 — a rank-2 TOP-4 phenomenon that does NOT
    transfer.** At rank-8 TOP-6 the separation count SATURATES: six cells spanning 17–59 at rank-2
    TOP-4 all land at 57–60, and ` at`@mlp11 goes from 17/60 (no signal) to 57/60 on nothing but a
    change of rank and TOP. As measured AT rank-2 TOP-4, depth profiles are CLASS-TYPE dependent —
    function words bottom out at mlp11 (4/4 strict minima), punctuation peaks near it (period
    58/60), capitalised tokens rise monotonically — but none of this is visible at rank-8 TOP-6
    because nothing is.
    **(f) The control-matching rule this produced:** a control must match the claim on CELL
    (§1634), on CLASS TYPE (§1637) and on CONFIGURATION (§1639), not merely on rows, seeds and
    statistic.
    §1633's headline margin of 13/60 collapsed to 2/60 once punctuation was compared against
    punctuation. **(g) THE BRIDGE FAILS FOR THE GAP, BUT NOT FOR THE EIGENVALUE RATIO (§1643-§1647).**
    §1647: |λ1/λ2| — computable from WEIGHTS ALONE — predicts relative CE rise at rho +.678,
    permutation p .019, on twelve held-out classes with the hypothesis registered in advance.
    That is the first significant result of the arc and it is NOT yet promoted: its rho and p
    sit almost exactly on §1614's (.6727, p .0192), which §1616 then refuted, and its
    out-of-sample rho exceeds in-sample, suggesting the relationship may hold within function
    words only. A second held-out set spanning types is required before any claim. For the
    SEPARATION GAP the bridge does fail:  Mean-ablating a class's own mlp11 slice costs that
    class CE in 11/12 classes — solid — but the separation gap does NOT predict how much:
    rho +.056 at permutation p .43 on twelve classes, while rho(n_positions, CE rise) = −.580,
    ten times stronger. §1643's +.657 on six classes was an n_positions artifact and is
    withdrawn. **The statistic describes where attribution mass concentrates and has NO
    demonstrated causal consequence** — anything using it to select or prioritise circuits must
    establish that first. **Bottom line for the bench: every separation or membership claim needs ≥20 bases
    and a control matched on cell, class type AND configuration; margins measured at certified cells
    must not be generalised; and the separation count must be checked for saturation before any
    margin is read from it — at rank-8 TOP-6 it has no dynamic range left.** Two withdrawals of §1612 are DRAFTED AND UNAPPLIED pending Logan
    (`PENDING_RETRACTION_S1612.md`); §1597's own statistics are untouched. Dossiers: registry
    `multiseed_null_methodology`, LESSONS 21–23. **HIGH** for (a)–(c) and (f); **MEDIUM** for (e) —
    seven classes, one rank, one TOP.

13. **THE FULL-DEPTH MLP DOSSIER (§1326, surfaced 2026-08-27) — and only SIX of eighteen modules are
    measurable at all.** stake = mean-ablation CE cost (nats); ceiling = 50k token-table recovery;
    elbow = k16/ceiling.

    ```
      mlp0   .799  .863  .43      mlp6   .079  −.060  --      mlp12  .039   .005  --
      mlp1  6.997  .945  .43      mlp7   .057  −.148  --      mlp13  .035   .059  --
      mlp2   .760  .716  .14      mlp8   .048  −.099  --      mlp14  .028   .039  --
      mlp3   .631  .593  .40      mlp9   .051  −.067  --      mlp15  .035   .141  .78
      mlp4   .104 −.252  --       mlp10  .040  −.060  --      mlp16  .150   .494 1.10
      mlp5   .083 −.036  --       mlp11  .043  −.032  --      mlp17  .381   .497  .84
    ```

    **(a) EVALUABILITY IS THE HEADLINE.** Only **mlp0-3, mlp16, mlp17** are evaluable. The twelve middle
    modules have stakes of .028-.104 nats, so every recovery figure divides by a near-zero denominator;
    their mostly NEGATIVE ceilings are the instrument's noise floor, not a module property. **The
    instrument is uninformative below ~0.15 nats of stake.** The middle of this model is UNMEASURED, not
    measured-as-small — a distinction the raw table does not make and that anyone quoting it must.
    **(b) THREE REGIMES.** Front (mlp0-3) largely token-tableable, ceiling .59-.95. Middle, unmeasured.
    Top (mlp16/17) genuinely half-contextual, ceiling ~.49. Front-minus-back gap .471 vs .206.
    **(c) mlp0 SPECIFICALLY,** for the early-block compiler work: stake .799 nats, token-table ceiling
    86.3%, un-tableable residue .110 nats, per-token mean table effective rank 22.7 against the
    embedding's 132.4 (§780), and that mean only 44% linearly predictable from the embedding — a standing
    prediction that any affine read of mlp0 caps out low.
    **(d) THREE INCOMPATIBLE PROTOCOLS EXIST FOR THE SAME MODULE.** mlp0 reads .799 (ladder stake), 2.195
    (scoreboard benefit), and a third value under exact-restoration-on-a-frozen-ship. §1324 warns
    explicitly they are "not directly comparable". Never mix them; always name the denominator.
    **Why this entry exists:** the ladder lived only in BILIN18_CONNECTION.md, appeared in neither
    FINDINGS nor the registry, and on 2026-08-27 that cost a duplicated GPU run — I began re-measuring
    mlp0's stake and ceiling from scratch before Logan pointed out it was already done. Dossiers:
    registry `_mlp_module_dossier`, `opt_ablation_consts_all.pt` (optimal constants, all 198 components),
    `bilin18_scoreboard_mp_results.json` (per-component, all 36). **HIGH** for the six evaluable rows;
    the middle twelve are NOT a result.

## Open / focus
- **A. CLOSED at read grain (§1222):** no compact carrier ensemble exists — best 12-head set
  = 43% of prose pooling; nameable edge = copy/induction core (2.5/3.8/5.5, 23%); rest is a
  ~150-head collective. Criterion question (content-sim grows with depth §1085) still open.
- **B. CLOSED (§1094):** L4 consumes a NON-content context variable (l4_function ran; stale
  entry corrected this pass).
- **C. CLOSED for the front band (dedup pass):** §1096 (prose tables ~50% on code), §1111
  (markdown, ratio-law retracted), §1112 (matched-vocab: register-contextual with breadth
  asymmetry) — the dossier marks the register thread closed; no re-run needed.
- **D. Middle attention non-local remainder** (§1069, low-stakes) and per-layer cross-model
  transfer maps (incremental). Census name-circuit reconciliation (induction dossier) — now
  easier with stations named (2.5/3.8 matchers).
- **E. FINDINGS hygiene:** keep this file ≤ 2 pages; add every closed arc as it lands.

## 14. mlp0 is a current-token table plus an attn0-delivered correction — and the measurement that shows it took three tries (§1659–§1661)

**The finding.** In the running model, mlp0's per-token lookup table recovers **90.27%**
of its 0.855-nat mean-ablation stake on positions the table covers. Freezing attn0 at its
optimal constant raises that to exactly **100%**. So mlp0's un-tableable residue — about
0.083 nats, the quantity `_mlp0_dossier` listed as open — is **entirely context that attn0
wrote into the stream**. mlp0 itself computes no context-dependence. It is a token table
plus a correction it inherits.

**Correction (§1662), applied here rather than left in the ledger.** The sentence above
originally read that *100% of the residue is attn0's write*. That is architecture, not a
finding: attention is the only module in a transformer that moves information between
positions, so an MLP with its attention inputs frozen is token-deterministic by
construction and its covered table is exact necessarily. The frozen arm is the instrument
check; the empirical content is the live 90.27%. **Which** attention delivers the residue
is a real question and is not answered by this run.

**The front band, same protocol (§1662).** Live covered ceilings are not ordered by depth:
mlp0 90.27%, mlp1 96.01%, mlp2 76.98%, mlp3 67.55%. mlp1 is the anomaly twice -- the most
tableable of the four despite sitting above mlp0, and carrying a 7.005-nat stake against
0.86/0.77/0.62 for its neighbours. Its 3.99% residue is therefore 0.279 nats, the largest
absolute un-tableable quantity in the band. The instrument check passes at all four sites.

**The instrument check, and why it matters more than the number.** With attn0 held
constant, the residual stream below mlp0 is embedding + constant, and MLPs are
position-wise — so mlp0 is a deterministic function of the current token and a covered
table must reproduce it exactly. The frozen ceiling therefore has a value known BEFORE the
run: 1.0. Observed: 1.0, with `ce_table = ce_live = 3.50924` equal to five decimals.

Two earlier versions of this measurement failed that check and **neither was reported**:

| | substitute at | score on | frozen ceiling (true: 1.0) |
|---|---|---|---|
| v1 | all positions | all positions | 49.37% |
| v2 | all positions | covered only | 55.83% |
| v3 | covered only, mlp0 live elsewhere | covered only | **100.00%** |

v1's headline was mlp0's ceiling *falling* 25 points under the freeze — the opposite of
the hypothesis, which reads as a strong negative result rather than as a bug. Without a
pre-derivable answer to check against, it would have gone into this file.

**The general lesson (LESSONS 27).** Excluding positions from a CE average does not
isolate a substitution applied at those positions. The forward pass still ran with wrong
values there, and attention in the layers above mixes them into the retained positions.
Interventions must be restricted in the forward pass — `torch.where(valid, sub, out)` —
not repaired in the score. On a quantity whose answer was known, the difference was 44
points.

**Scope of the correction.** Every table ceiling in this project fitted with an
unseen-token fallback and substituted at all positions is contaminated the same way and is
therefore **understated**. On mlp0 the understatement is 15.9 points (74.42% → 90.27%).
Entry 13's dossier levels are flagged in the registry rather than silently revised: only
mlp0 has been remeasured, and while the bias is probably similar at every site (so the
ordering across modules likely survives), the levels are not trustworthy until remeasured.
