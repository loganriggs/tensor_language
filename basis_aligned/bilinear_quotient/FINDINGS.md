# bilin18 — consolidated findings index

The working index. The full ledger (BILIN18_CONNECTION.md, 1090+ sections) is an append-only
lab notebook, not a place to keep up with results. **This file is the ~10 things that matter,
their confidence, and what's open.** Update this in place; don't let it grow past ~2 pages.
CONSOLIDATED 2026-08-23 (was 29 items; §1092 exposed that the index had dropped the sink arc,
causing a 4-experiment re-derivation — every major arc is indexed here now).

Confidence: **HIGH** = causal test + control + null, reproduced. **MED** = solid but one caveat.

**PER-MODULE DOSSIERS: `modules/` — self-contained doc per module (facts+numbers+§refs,
stand-in status, module-specific gotchas, open questions). READ THE RELEVANT DOSSIER BEFORE
DESIGNING AN EXPERIMENT on a module, and update it in the same commit as the ledger writeup.**

## The established results

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
   4.3-vs-11.8: text-set, not machine-set §1212); ~2 dominant stations each. Fingerprints:
   station depth (front / mid / spread-to-late), secondary-station width (pair vs crowd);
   bilin12's late band NOT local (17%). **HIGH.**

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
   **HIGH.**

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

## Open / focus
- **A. The content seed's real carriers:** L5H7 retired (constant); §1074 says early attention
  gathers content — WHICH of the redundant poolers, and by what criterion (content-sim grows with
  depth §1085)? Needs an ensemble-level (not per-head) intervention design.
- **B. L4:** first true context MLP — what does its dev×dev consume (content subspace vs other)?
  l4_function queued.
- **C. Register/OOD axis:** grammar-machine causal transfer to code; more registers.
- **D. Middle attention non-local remainder** (§1069, low-stakes) and per-layer cross-model
  transfer maps (incremental).
- **E. FINDINGS hygiene:** keep this file ≤ 2 pages; add every closed arc as it lands.
