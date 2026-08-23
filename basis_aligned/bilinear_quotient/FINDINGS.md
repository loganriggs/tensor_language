# bilin18 — consolidated findings index

The working index. The full ledger (BILIN18_CONNECTION.md, 630+ sections) is an append-only
lab notebook, not a place to keep up with results. **This file is the ~10 things that matter,
their confidence, and what's open.** Update this in place; don't let it grow past a page.

Confidence: **HIGH** = causal test + control + null, reproduced. **MED** = solid but one caveat.
**LOW/known-limit** = suggestive or resolution-limited.

## The established results (most important first)

1. **Redundancy is universal, and decodability ≠ causality — one isolable knob + a distributed
   remainder.** Computation is diffuse over MLP units, depth bands, and attention heads; the
   strongest localization is "these matter more than chance," never "this is the circuit" (§610–616,
   §633, §644, §648). The isolation taxonomy (§650–668): **additive/subtractive biases isolate to a
   removable linear direction** (the frequency calibration = rank-1, §650–651) — but **every
   conditional/predictive/stateful computation** (newline routing §652–653, article magnitude §654,
   content-writing §658, quote-parity register §668) has **no removable linear carrier**, even when
   the feature is strongly *decodable* (quote-parity AUC 0.83 yet causally inert, §668 — read≠write).
   Net: the model is ~**one cleanly-isolable linear knob** (the frequency-calibration bias, ~0.43 of
   the ~7.48-nat loss-benefit over uniform, ~6%) plus a ~94% **distributed remainder** with no
   linear carrier. **HIGH.**

1b. **The right decomposition of the front is a canonical token-class SUBSPACE, not learned
   parts (§737–772).** [§836 CORRECTION: every keep-only RECOVERY magnitude quoted in this item and 1c
   (0.78, 0.92, cross-model 0.64–0.92) is a RANK/CONSTRUCTION ARTIFACT — a shuffled-label subspace of
   matched rank recovers the same (§836). Read those numbers as "rank-r compressibility", not
   "class+position's specific share". Class+position is real via NAMING (§825/826) and STEERING (§823),
   not via keep magnitude.] Chasing a good decomposition: reconstruction-optimal bases (SVD/A-SVD) are
   CE-catastrophic at low rank (front-load loss-irrelevant massive-activation energy, §737/748); a
   learned overcomplete **sparse dictionary** (weight-action SAE) reconstructs faithfully (§750/759,
   needs a reconstruction anchor or pure-CE training destroys weight-faithfulness §758/762) and its
   cross-layer coupling is **weight-only / data-invariant** (§754, A-SVD's is not) — but the SAE
   *atoms* are the **wrong unit**: seed-unstable (recur 0.40, §763), redundant (super-additive §761),
   and monosemanticity is orthogonal to causal importance/stability (§763); co-activation groups
   don't rescue it (§766). The interpretable+causal+stable structure the SAE fails to be **is a
   seed-free SUBSPACE** you compute directly: the **token-conditional-mean** directions of a
   component's output. That subspace is **canonical/data-stable** (0.82 vs atoms 0.40), **necessary**
   (remove top-64 → +1.34 nats, 268× a random subspace §767), **sufficient** (keep only top-64 →
   92% of the layer's loss §770), **low-rank** (64/1152), and **human-nameable**: the front sorts
   what it writes by **token-class** — MLP by LEXICAL categories (determiners/punct/numbers/verbs
   §768), attention by STRUCTURAL/discourse markers (conjunctions/clause-boundaries §771). It is
   **universal** across both component types (§771) and **multiplexed** into **near-orthogonal**
   per-component subspaces (attn0|mlp0 overlap 0.254 vs 0.201 floor §772). *Ground-truth-free recipe
   for "a good component": don't fit a dictionary and hope — compute the canonical token-mean
   subspace, verify it causally (ablate-vs-random AND keep-only-vs-random), read its named axes.*
   **HIGH.**
   **GRAND CAPSTONE (767–795):** keeping ONLY class+position at ALL 36 components at once recovers
   **78%** of the whole model's 9.1-nat contribution (front alone 84%; random subspace 4%),
   **uniformly** across every token-class (function words 0.80, content 0.77 — gap 0.04). Class is a
   computed grammatical-class collapse (identity ~132 eff-dim → class ~24, sharpened ~1.8× vs the
   embedding, nonlinear R²0.44 §780/782); position is a coarse ~2-dim early/late readout of RoPE
   (RoPE ~18 eff-dim → ~2 §786/788). Written to near-orthogonal channels (§772), read amortized by
   later components (§783/784). **Cross-model COMMON across ALL SIX models (§807/§808, corrected
   mean-preserving per-component nat-weighted):** class+position share is bilin18 0.92, gpt2-small 0.91,
   gpt2-large 0.92, pythia-160m 0.81, pythia-410m 0.74, gpt2-medium 0.64 — band 0.64–0.92, every model
   >> its random null (0.02–0.28); robust across bilinear/absolute/rotary and 124M–774M. (bilin18's
   "0.78/four-fifths" is the stricter SIMULTANEOUS metric — all 36 components at once; per-component,
   apples-to-apples with the others, it is 0.92.) **GPT-2-medium was NOT an exception** — §802/803's
   "genuine isolated exception" (0.12) was a DROPPED-MEAN artifact of the centered keep metric (§804/805):
   its mlp0 is 91% constant bias by norm; keep goes −0.13 → +0.63 once the mean is preserved. Large
   constant DC biases are a COMMON early-layer feature (§806: 7/10 early components >0.5), and the
   §800 centered numbers were all slight underestimates; corrected, gpt2-medium is in-family (0.64, the
   least-clean case) with NO genuine exception. (The
   *simultaneous* whole-model metric is bilin18-specific — needs its 30·tanh output clamp — §799).
   The class-SHARPENING mechanism is the one model-specific piece (bilin18/Pythia yes, GPT-2 no §789/796).
   Also generalising to GPT-2/Pythia at the subspace level (§778/781). The distributed remainder is
   **content that makes confident predictions confident** (§798: class+position recovery rises 0.50→1.00
   with token difficulty — the skeleton is all that's left when the model is uncertain). The
   ~22% remainder is a **uniform, diffuse, distributed** computation with no low-rank or class handle
   (§795). So the 546M model is ~4/5 a class+position machine. Per-HEAD concept-naming FAILS (heads
   are broad multi-function; the fold names QK geometry not causal function, §792) — the robust unit
   is the VARIABLE a component reads/writes, not a head label. **HIGH.**

1c. **Whole-stack shape = an amortized compute→maintain→read pipeline, UNIVERSAL across 6 models
   (§809→817).** By loss-benefit the stack is a barbell in PER-COMPONENT ablation (front layers 0–5 =
   81% of benefit, 93% class+position; back MLPs 15–17 read out, 93%; middle low per-component). BUT
   per-component ablation UNDERSTATES redundant regions — CORRECTION §813: ablating the whole middle
   (6–11) at once costs 1.93 nats, ≈4× the 0.49 per-component sum (super-additive = distributed/redundant,
   each component compensable). The middle is class+position MAINTENANCE: simultaneous keep-only
   class+position on 6–11 recovers 0.65 of its collective benefit (random null −1.06, §814). REFINED
   §818: since the residual stream is additive, the middle's cost means it ADDS class+position — and it
   does so in mostly NEW directions (middle output keep-own 0.65 vs projected onto the front's subspace
   only 0.37, overlap 0.29), so ~57% reinforces the front's directions and ~43% is new class+position
   structure; that new content leans toward finer CLASS over position (§819: net token gain +0.18 vs
   position +0.10, both ≫ matched random nulls). So the middle CONTINUES computing class+position,
   chiefly refining grammatical class (not mere refresh). Pipeline: FRONT computes class+position →
   MIDDLE keeps computing MORE class+position (redundantly, mostly finer class) → BACK reads it out. This three-band structure with a SUPER-ADDITIVE class+position-maintenance middle
   REPLICATES in ALL SIX models (§815/§817: middle compounding 1.3–3.9×, class+position keep 0.59–0.78;
   every band ≫ random) — bilin18, GPT-2 small/medium/large, Pythia-160M/410M; bilinear-softmax-free,
   absolute, rotary; 124M–774M. The redundant class+position-MAINTENANCE middle is the robustly universal
   part. Front/back detail is architecture-specific and varies more than n=3 implied (§817): front
   compounding sub-additive in bilin18/pythia, super in GPT-2; pythia-160m is back-heavy; and gpt2-large's
   FRONT is only 0.37 class+position under the SIMULTANEOUS band metric (which compounds — per-component it
   is 0.92, §808 — so its front has strong cross-component interaction structure). Middle is ~half-compressible
   (§816: drop 3 of 6 middle layers for 0.27 nats; front convex/irreplaceable). VALIDITY (§820): the
   subspace GENERALIZES to held-out FineWeb (within≈cross keep, gap ≤0.005; not overfit), BUT per-component
   mean-preserving keep is MEAN-dominated (§821: driver is the mean, NOT redundancy) — substituting a
   component's constant mean μ alone recovers 0.66–0.91 of its single-ablation benefit, so even a random-
   orthonormal subspace (which re-adds μ under mean-preserving keep) recovers 0.84–0.95; a random projection
   WITHOUT the mean recovers only 0.04–0.29. bilin18 HAS large per-component means (ratio up to 0.97;
   §808's "biases not loss-critical" was wrong — centered≈mean-preserve because the real cp subspace
   CONTAINS μ). The class+position VARIATION adds a modest but specific increment (0.05–0.14 over mean+random;
   centered-random low → specific). So the per-component headline (0.92) is an UPPER BOUND; the honest
   whole-model share is the SIMULTANEOUS CENTERED metric (§794: 0.78, random 0.04) — "~4/5 of the model". **HIGH.**

   **CAPSTONE — class+position program (§767→836), cross-model, WITH A MAJOR §836 CORRECTION on the
   quantitative headline.** WHAT: class = grammatical categories (determiners, pronouns vs numbers,
   punctuation, conjunctions, prepositions, be-verbs/auxiliaries; §825/826, named & universal, shuffled
   control incoherent), position = a logarithmic early/late scale + first-token landmark (§827). CAUSAL
   (all keep-only-free, so unaffected by the §836 retraction): class steering shifts predictions toward the
   target class and is class-SPECIFIC — toward the injected class, away from others (§823/837; random
   directions do not; steering the embedding corrupts §824); and it DRIVES grammatical sequencing —
   steering→B shifts the predicted next-class toward what-follows-B, far better than random (§838). USED AS
   grammatical sequencing — predicted next-class matches the empirical class-bigram to KL 0.009 (§828). LOSS SPLIT (universal, pure chain rule): predicting the class is the
   EASY ~23-25% of the loss; the word within it is the HARD ~75-77% (77% bilin18 / 75% gpt2 / 76% pythia,
   §829/831), partly context-reducible (~1.1 nats) but mostly an irreducible ~2.4-nat entropy floor (§830).
   **HOW MUCH — CORRECTED (§836): RETRACTED as a class+position-specific number.** The keep-only "78%/0.92
   class+position" (§794/807/808) is a RANK/CONSTRUCTION ARTIFACT: a shuffled-LABEL subspace of matched
   rank 96 recovers the same (0.805/0.760 vs real 0.811/0.761; specific gap +0.006). Keep-only measures
   rank-96 compressibility, not class+position specifically (class+position ≈ top-PCA ≈ shuffled-label
   subspace). So do NOT say "the model is 78% class+position"; say "each component's output is ~78%
   recoverable at rank 96, and that low-rank structure is — by NAMING and STEERING — grammatical class +
   position." The right keep-only null is a shuffled-LABEL matched-rank subspace, not random-orthonormal
   (§821 was far too weak). ROBUST regardless: naming (§825/826), steering (§823), loss split (§829-831),
   and ablation-based shapes (barbell/redundant-middle §812/813). ONE-LINE (corrected): the model computes
   a low-rank representation, organized and causally used as grammatical class + position, that nails the
   easy grammatical quarter of the loss; the hard three-quarters is high-entropy lexical choice — but the
   *fraction* of the model attributable to class+position is not measurable by keep-only and was withdrawn.

1d. **Bottom-up LAYER MAP (mechanism + geometry, keep-only-free; §841→858).** FRONT (0-5): mlp0 = bank of
   bilinear SHARPENING self-product class-detectors (grammatical class), attn0 = COPY-SOURCE (writes prev-token,
   decode 0.86); mlp1 RE-EXPANDS the class-collapsed geometry (eff-dim 20→47) folding in class+position+modest
   prev/fine-token; L2-4 refine; attn5 = the early CONTENT head AND the induction GATE (§877/§878 — ablating L5
   collapses ~all induction: synthetic 2nd-copy 1.29 vs 13.09; natural inductable 0.69→3.64 while control L11 does
   nothing; fed by the attn0 prev-token head in a front chain L0-L5). NOT pure induction: L5 also aids first-mentions
   (+2.15), copy-specific effect ~0.8 nats. Explains seen-token cheapness (inductable bigram 0.68 vs 3.55). MIDDLE (6-14): mechanism NAMED (§869/§870) — ATTENTION is the TOPIC AGGREGATOR (topic-decode rises
   0.37→0.85 across depth, gains ALL from attention: every layer's attn-increment +, middle +0.205 vs MLP
   −0.016), MLPs are TOPIC READERS/word-generators (read topic-organized input 3-4× at all depths but don't
   raise topic decodability). Matches the geometry (re-inflates eff-dim to peak 51 by L9 into content dims;
   topologically stable RSA 0.95-0.98, §813). READOUT (15-17): reads
   class(13×)+position(6×) hard (§851), semi-distributed content-write with some coherent category units
   (proper-name unit, capitalized-word units), collapses the residual's VARIANCE to ~3-4 eff-dim (grammar/boundary; §891) — but PREDICTION is high-rank (~256 dims for 88% of CE gain, content in the low-variance tail). WHOLE-STACK
   geometry: 2 expansions (L1, L6-9) + 3 collapses (L0, L5, L16-17); re-clustering at front+readout, middle
   near-frozen. CONTENT MACHINE NAMED (§866): the diffuse content is a TOPIC/semantic-DOMAIN tracker — the content residual
   clusters cleanly by subject (education, Orthodox religion, tech/business, geopolitics, sports, vaccines, aviation,
   medicine, prison; distinctive-token read) and predicts topically-coherent words. Reconciles the diffuseness: topic
   space is HIGH-DIMENSIONAL (no low-rank handle) but STRUCTURED, not noise (§810 sharpened). Attention feeds it context
   (§862). Topic causal support (§868/§887 CORRECTION): topic-direction steering gives a weak-but-specific gain (own +0.042
   vs off −0.004, every topic) — BUT this is a weak test: steering the class read-direction the same way is only ~2× stronger
   at matched alpha (§888, grammar own +0.079 vs topic +0.042, both weak <0.1; grammar modestly more, consistent with lower
   rank), because read≠write (item 2) limits single-direction steering for BOTH machines. Topic IS CAUSAL, confirmed by
   INTERCHANGE INTERVENTION (§894, Geiger-style — patching the topic subspace base←source shifts prediction toward the
   source topic +0.70 nats vs random, flip 0.36→0.49; interchange works where steering fails because it respects read+write
   jointly). Also supported by content-word gist via context edits (§872), first-mention floor (§876), replicable geometry
   (§874). Class likewise causal via interchange (§892, IIA 0.25 vs 0.06). The two machines differ by STRUCTURE (grammar low-rank/local/context-free;
   content high-dim/long-range/topic), not steering strength. grammar = easy 23%, content = hard 75%. Full content
   chain (§870): ATTENTION aggregates context → topic representation → MLPs read topic → readout emits topic-coherent word.
   Aggregation NATURE (§871/§872): topic needs LONG context (content-CE drops 1.53 nats to C=256 vs grammar 0.35, 4.4×) and is
   a CONTENT-WORD, ORDER-INVARIANT gist (function-mask retains 82% of topic, content-mask 16%, order-shuffle 61% while
   grammar-CE rises +0.70). Grammar = local (few tokens), token-driven; content = long-range content-word gist.
   FRAMING (§873/§874): the topic structure is CONTINUOUS/high-dimensional — the 12 clusters are a VISUALIZATION, not 12
   discrete categories. Discrete labels don't replicate across splits (expected for a continuum), but the CONTINUOUS geometry
   DOES: content-subspace overlap A↔B 0.53 = 25× random chance (§874). Finding rests on the content-word
   gist §872 + held-out decode §870 + replicable geometry §874, not on discrete buckets. HIERARCHY (§927):
   the content is subject matter at MULTIPLE GRANULARITIES — re-clustering within a topic yields coherent
   sub-topics (religion→worship/saints/titles/doctrine; academic→medicine/peer-review; tech→security/design),
   so the high dimensionality is a coarse→fine topic hierarchy, not noise (caveat: the sibling-disjoint metric
   is weak; the evidence is the sub-topics' interpretability). Beyond topic, only weak non-topic structure is
   nameable (§926: entity-repeat +0.05, prev-class +0.06). **HIGH.**

1e. **LOSS BUDGET + irreducible floor + generality (§876→885).** Where every nat goes (bilin18, unigram baseline 7.17 →
   3.24 = grammar 0.76 + content 2.48): by position type — FIRST-MENTION (new token type) 60% of tokens / CE 4.24 = **78% of
   the loss** (almost all content); SEEN-OTHER 34% / CE 1.94 = 20%; INDUCTABLE (repeated bigram) 6% / CE 0.69 = 1% (induction
   nearly free). Grammar is a cheap uniform tax (~0.8 every bucket); content tracks novelty (0.39 copied → 1.10 seen → 3.48 new).
   IRREDUCIBLE floor (§876): first-mention content is 87% shared with gpt2-large (per-token loss corr 0.90) — topic-narrowed but
   open new-word choice, only shrinks with scale. SEEN discount (§881) = soft-copy salience (falls with recency+count), L5 the
   long-range copy. GENERALITY (§880/§883/§884/§885): the budget (78/20/1) and 23/77 split are INVARIANT across model (gpt2/large),
   MLP form (swiglu18), attention (sqrd12), and scale (bilin12); induction STRENGTH scales with size (18L 11.8 ≫ 12L 4.3) and is
   stronger with normalized attention (sqrd12 8.6 > bilin12 4.3). **HIGH.**

1f. **VARIABLE-LEVEL causal abstraction + how much we understand (§892→906, user-directed).** The honest unit
   is the VARIABLE (read/write subspace), not the head. INTERCHANGE INTERVENTION (Geiger) verifies causality where
   mean-STEERING failed (steering pushes the read direction, read≠write, item 2): CLASS is causal (patch flips
   predicted part-of-speech 4× a random-subspace swap, §892); TOPIC is causal (patch shifts prediction toward the
   source passage's subject +0.70 nats vs random, §894). The class variable's subspace ROTATES write→read across
   depth (front-write overlap 0.29→0.02, readout-read 0.07→0.18, §897) — so weights identify a variable only at the
   layer that acts; activations/DAS are the ground truth (DAS≈activation, §895). HOW MUCH WE UNDERSTAND (replace
   components with named-variable stand-ins; 0=mean-ablate, 1=full; certified on HELD-OUT data — in-sample overfits,
   e.g. 0.81→0.30, §901): with a smooth low-rank MAP of named variables (token+prev+continuous-topic), ~**40%** of
   the whole model held-out (§906; table 0.30, null −0.03). Splits by depth: FRONT grammar ~0.90 (a smooth
   generalizing function of the current token — SOLVED, §905), READOUT ~0.56, MIDDLE content ~0.10 — the middle is
   NOT a function of these variables (genuinely high-dim, needs longer-range/interaction variables), yet is causal
   (§894). So grammar is a writable+causal variable; content is causal but not yet tabulatable — the frontier.
   BENCHMARK REFRESH (§939): give the token variable a SMOOTH per-token LINEAR MAP (not a discrete table) and the
   held-out number rises to **0.42** (token-map alone 0.32 beats the whole old table pipeline 0.315; shuffled-map
   null −0.19). Progression: 0.81 in-sample (retracted) → 0.30 held-out table → **0.42** held-out map. DEPTH
   LOCALIZATION (§940): genuine (shuffle-corrected) understanding front 0.72 > middle 0.60 > back 0.58; the middle
   has the LARGEST contribution yet worst raw reconstruction → the residual frontier. [§1013/§1014 CAVEAT: the
   held-out understanding number is DRAW-SENSITIVE — the table-based simultaneous-held-out config is **0.32 ± 0.06**
   (4-draw mean±std, genuine-vs-shuffled 0.74 ± 0.15); the swing is the TOPIC term (train/eval topic overlap varies),
   the token term is the tightest (~0.15). So ALL single quotes here (0.30 / 0.40 / 0.42) are POINTS on a
   draw- and config-variable distribution; report the understanding fraction as a range with ±0.06 draw-uncertainty,
   not a stable point. The residual ~0.68 is dominated by the irreducible context-multiplicative content (§1000).]
   WHY (§941): a
   linear→multiplicative→linear depth arc — MLPs are 90–98% linearly-recoverable at the front (incl the dominant
   writer MLP1), ~38% in the middle (i.e. ~60% genuine bilinear MULTIPLICATION, irreducible to any linear/table
   stand-in), 85% at readout. CONTENT-MECHANISM ARC (§928→941): topic = order-invariant bag-of-word-embeddings
   gist (§932/§934, universal §937) organized as a coarse→fine hierarchy (§927), but only a MODEST causal slice —
   a topic-centroid stand-in recovers 6%@K128 / 17%@K1024, a continuum not a finite set (§930); no single handle
   dominates (token-identity ~10%, cross-token topic ~7%, §938); the bulk is the middle multiplicative computation.
   Grammar reads the local current token; content reads the whole-context bag (double dissociation §934). **HIGH.**

1g. **GRAMMAR MACHINE — complete bottom-up account (§915→919).** (1) mlp0 writes the current token's MULTI-AXIS
   SURFACE class — capitalization, determiner, punct, number, space-prefix are separately decodable near-ORTHOGONAL
   axes (|cos| 0.10), a token is several at once ("The"=determiner+capital; §915); my 8-way label is a coarse
   single-label readout of the ~24-dim code. (2) ATTENTION across depth aggregates context to build PREDICTIVE
   next-token-class grammar (next-class decode 0.53→0.67, every attn step +, total +0.11; §918) — the SAME
   distributed heads that build topic (per-head grammar/topic corr 0.31, mostly shared + mild specialization,
   §919): attention is a SHARED context-engine for grammar-prediction and content. (3) Middle MLPs CONTEXTUALIZE
   the class (context-derived, NOT maintenance: token-R² 0.40 < context-gap 0.46; §916), shifting the code from
   describing the current token toward predicting the next (§917). (4) Back MLPs CONSUME class into word logits
   (negative next-class increments; §918). Re-derived not carried because the residual is leaky (λ₀ rescale) +
   token re-injected each block (λ₁·x₀). Causality via interchange, not steering (read≠write, §892). **HIGH.**

1h. **FULL-STACK COMPUTATIONAL ACCOUNT — three registers by depth (§939→950).** Read off the layers, bilin18
   computes in THREE registers: (a) FRONT (L0-5) — near-LINEAR per layer (MLP linear-recoverable R² 0.90-0.98,
   §941/§942), writes grammar+token; localized/load-bearing (mlp1 mean-ablate 6.4 nats §933) and tightly
   SEQUENTIALLY COUPLED (can't be co-linearized — §950). (b) MIDDLE (L6-15) — the high-rank MULTIPLICATIVE content
   computation (~40% nonlinear certified, §941/§950; universal middle-nonlinearity dip incl GELU models §942);
   multiplies TOKEN and CONTENT, NEVER class (§946); DISTRIBUTED-COOPERATIVE (each MLP ~0.04 alone, all-10
   super-additive 4.9× §948); the irreducible frontier. (c) READOUT (L16-17) — does the BULK of output formation
   (logit-lens CE 5.8@L15→3.26 final, §944) via a ~95% LINEAR rotation into the token basis (§945, certified
   nonlinear-fraction 0.16 §950). BENCHMARK: replacing all 36 components with named-variable stand-ins recovers
   **0.42** held-out (§939, smooth per-token map > discrete table 0.315; progression 0.81-insample→0.30-table→
   0.42-map); the residual ~0.58 is the middle multiplication. Depth localization (§940): middle is the biggest
   contributor + hardest to reconstruct. Figure: stack_map.png. CAUGHT+CORRECTED: skip-confound §947→§948
   (residual-rescale), front-linearize artifact §949→§950 (uncertified stand-in). FAMILY-GENERAL (§965→§967): the
   whole two-machine account generalizes across the Elriggs family (bilinear AND SwiGLU, 18L AND 12L) on three
   axes — STRUCTURE (grammar⊥content separability, ratio 1.16-1.5 §966), MECHANISM (content=bag-of-words,
   bag-current +0.41-0.49 §967), DISTINCTIVE STYLE (distributed-cooperative super-additivity 1.6-3.2 vs GPT-2 0.71
   §965, training-driven not MLP-specific); key properties universal via GPT-2 (§925/§937/§942/§880). DISTINCTIVE STRUCTURE (§956):
   bilin18's computation is DISTRIBUTED-COOPERATIVE — single components near-free, ensembles load-bearing
   (super-additive: content §948 4.9×, attention §956 ratio 3.52); this is the OPPOSITE of GPT-2 (sub-additive
   ~0.72, individually-critical redundant layers), so it is a genuine bilin18 property, not generic compounding —
   and why single-unit ablations came up empty and the honest units are ensembles/subspaces/variables. **HIGH.**

1i. **OUTPUT BEHAVIOR & ERRORS — the two machines at the readout (§972→980).** bilin18's mistakes are CONTENT
   mistakes, not grammar: top-1 is class-correct ~2/3 of the time (§972); when wrong it either picks the wrong
   content word or HEDGES to a frequent function word (grammar-error top-1 is function-class 78% vs base 44%, true
   token rare, §973) — family-wide (§975). The hedge is the FRONT's high-frequency default that the READOUT
   overrides toward content on easy positions but not hard ones (§976); frequency de-biasing is DISTRIBUTED across
   the stack (readout dominant, §977). The model is EXCELLENTLY CALIBRATED (token ECE 0.009, class ECE 0.022):
   confident+accurate on grammar (~0.72), appropriately uncertain on content (~0.38) — so the hedge is CALIBRATED
   deferral, not blind failure (§979). And content errors are GRACEFUL: the wrong word is in the true word's TOPIC
   40% vs 15% class-matched-random (§980) — the content machine narrows to the right subject even when it misses
   the word. CAPACITY buys CONTENT not grammar (bilin18 vs bilin12: content-CE 2.53 vs 2.68, grammar-CE ~0.8 both;
   §978): grammar solved at any size, content the capacity sink + slow-yielding frontier. Method caveat: the §974
   within-top-class entropy metric was confounded (report class-entropy result only). **HIGH.**

1j. **QK ROUTING — the softmax-free squared attention reverse-engineered (§981→984).** The attention weight is
   an UNNORMALIZED PRODUCT of two bilinear score matrices: pattern = (q.k/D)*(q2.k2/D), causally masked. (1) Heads
   SPECIALIZE by routing mode; (2) the two QK factors are NON-REDUNDANT (differ 6/9 heads), so the pattern is a
   SIGNED CONJUNCTION of two distinct criteria — positive where they agree, NEGATIVE where they disagree, giving
   SUPPRESSIVE "anti-heads" that route AWAY (impossible for non-negative softmax) (§982/981). (3) The range-robust
   routing modes are RECENCY and INDUCTION, NOT content-similarity — the apparent content-similarity was a recency
   confound, retracted via a long-range control (§983). (4) INDUCTION routing is FRONT/MID-peaked (peak L5, gone by
   L11+, §984), confirming §954 at the pattern level. (5) Since attention does NOT route by content, the topic
   EMERGES from broad ~uniform/recency POOLING (unifies §932 bag-of-words: routing is not content-selective). Exact
   pattern captured via verbatim-copy monkeypatch. **HIGH.**

1k. **ARCHITECTURAL MECHANISMS — value residual & x0 re-injection ground the two machines (§985→987).** The model
   keeps the original token available at every layer via two re-injections, both heavily used (learned weights
   large): (1) the VALUE RESIDUAL mixes the first block's ORIGINAL token values into every block's attention
   (lamb -4..+4.6); ablating it DOUBLES the loss (+3.33 nats) and damages CONTENT 3.8x more than grammar — it is
   the bag-of-words CONTENT-aggregation substrate (§932), genuine and graded (§986). (2) The x0 EMBEDDING
   RE-INJECTION (lambda1~8, saturated) DOMINATES each block's input (~8/9 embedding), the reason class is
   re-derived from the ever-present token every block (§962); ablating it costs +2.34 nats. Both are CONTENT-HEAVY;
   x0 is only RELATIVELY more grammar-weighted (ratio 2.96 vs value residual 3.84) — a relative shift, NOT a clean
   grammar-vs-content dissociation (my dissociation hypothesis corrected, §987). Caveat: full ablations are
   off-distribution (both graded, so genuine, with a near-zero nonlinear tail). **HIGH.**

1L. **BILINEAR-MLP INTERNALS + CONTENT-MACHINE MECHANISM/LOCALIZATION + the multiplicative ceiling (§988→1003).**
   *Bilinear MLP:* the multiplicative interaction term (u·w) DOMINATES the down-projection's raw output variance at
   EVERY layer incl the front (~90%+, §990); it is NOT gated (both factors vary, §989) nor projected out. The FRONT
   interaction is LOAD-BEARING (deleting it at L0 costs 1.7 nats, worse than deleting the layer, §992) yet the front
   output is reproducible by a BEST-FIT linear map (89–98% loss recovery, §941/§993) because the front interaction is
   linearly-SHAPED in the input; the middle's is genuinely nonlinear (best-fit ~27–38%). The front interaction is a
   SUPER-ADDITIVE cooperative cascade (joint≫sum, §994); deep-middle interaction is cheap even jointly (redundant
   band, §940/§994). Three of my interpretations were corrected in place along the way — "constant gate" (§989),
   "Down projects out interaction" (§990), "front interaction inert" (§992) — runs were clean, fixes were better
   controls. *Content machine (§995→1002):* content = broad, long-range, order-invariant, content-word-weighted
   (80% vs 51%) BAG-OF-WORDS (§995/996), UNSATURATED even at 256 tokens (~4× more context-hungry than grammar);
   GATHERED by attention concentrated in L3-5 (§998), pooled into each position's residual across the early/middle
   stack and READ OUT LOCALLY by the late layers (§997, reconciling §936's "content local at L15" as post-pooling
   readout). The irreducible MULTIPLICATIVE content is FRONT-LOADED in L0-2 (§1001, ~44% of it), mostly local
   word-sense with a secondary pooling-substrate role (§1002; clean local/topic split not achievable — entangled
   interventions, stated plainly). *Ceiling:* forcing all MLPs to their best COMPOSITIONAL linear approx costs
   **+1.55–1.59 nats of content** (§1000, draw-stable §1003) — the honest floor on what any linear/table/bag
   named-variable stand-in can reconstruct, and why the whole-model benchmark's content term is capped. In ABSOLUTE
   nats the front costs more than the middle (writes most, §933) though it is ~linear per-layer RELATIVE (§941). Any
   band linearized hurts content ~4× more than grammar. **HIGH.**
   *Ceiling refined + benchmark (§1004→1005):* a content-word bag does NOT beat an all-token bag as a benchmark
   content stand-in (both ≪ current-token+centroid, §1004) — §996's content-word finding is a MECHANISM fact, not a
   stand-in recipe. A per-token TABLE is WORSE than a linear map (§1005: front 1.55 vs 1.01) because it discards
   context → the front content is CONTEXT-dependent, not word-sense. Clean three-way split of front MLP content: ~18%
   current-token-lookup / 29% context-linear / 54% context-MULTIPLICATIVE (the irreducible floor). *Single content
   head, corrected (§1006→1009):* by attention-window banding, the L3-5 content-gathering is dominated by ONE head,
   **L5 head 7** (broad-pooling content head, ratio 3.9; §1006/1007) — BUT output-ablation of h7 costs only 0.01 nats
   (§1008), so h7 is the PREFERENTIAL but REDUNDANT content head, NOT a load-bearing single part; the "rare single
   nameable component" reading was retracted. Methodological lesson (§1008/§1009): banding vs output-ablation FLIPS
   with granularity — banding overstates a redundant SINGLE head (sibling compensation + local-pool interference),
   output-ablation overstates a whole BAND (removing early attention is foundational/off-distribution). Neither is
   uniformly right: banding isolates the long-range FUNCTION (correct for "where is content gathered", §998 stands),
   output-ablation measures total-output importance. **HIGH** (mechanism); **MED** (single-head localization: real
   but redundant, instrument-dependent).

2. **Read ≠ write direction.** A supervised probe decodes a feature; the *unembedding row* (write
   axis) steers it; the two are ~orthogonal (cos≈0). Pushing the probe does not steer (even
   reverses). To decode, fit a probe; to intervene, push the write axis. **HIGH.** §619–622.

3. **Block 17 is the DOMINANT frequency calibrator (not the sole one), isolated to a rank-1
   direction.** It suppresses tokens ∝ log-frequency (corr +0.64); the only *net* calibrator by
   whole-block CE (−0.17/+0.69 → +0.43 nat trade). But rank-1 `w_freq` removal finds a calibration
   *component* in **five** layers — L4/5/6 and L16/17 — block 17 dominating 5–10× (§662); calibration
   is distributed across two bands, diluted below net-calibration in the others by their writer roles.
   The calibration = one direction `w_freq` (removing it
   kills it, random doesn't; §650–651), ~40% aligned with the unembedding log-freq axis (cos 0.61,
   cos²≈§627's R² 0.41; §656). Its function→content mass shift (§629) is the SAME w_freq direction —
   "boost rare content" = "suppress frequent function tokens" (removing w_freq also drops rare
   capitalized-writing; §657). Subword-writing is a separate, preserved function. **HIGH.** §624–657.

4. **Depth division of labor (with mechanism).** FRONT (0–2) decides the next-token *class* —
   MLP-dominant, token-local, from the embedding trigger (§634). MIDDLE (7–15) refines *which token
   within the class* (esp. the open content-word slot) — attention+MLP balanced, more context-
   dependent than the class decision (§665); a GENERAL content-word predictor (serves novel words as
   much as repeats — copying/induction is a sub-component, not its defining feature, §666). BACK
   (16–17) calibrates frequency. **MED** (front's +7-nat magnitude inflated by error-compounding;
   middle/back claims clean). §630–632, §634, §665–666.

5. **Circuits bottom out in embedding trigger-geometry, not computed triggers.** Skip all 18 blocks
   and a `.`→newline / prep→the lean is already in embedding∘unembedding. BUT the direct path is a
   *poor* LM (CE 12.65 > uniform 10.83) — it's a relative lean, off-distribution alone; the blocks
   do essentially all real predictive work (+7.48 nats over uniform). **HIGH.** §637, §640.

6. **The blocks' job = context-discrimination/routing of a context-blind trigger.** The `.` bigram
   fires identically at every period; the blocks route among {newline, capitalized, continuation}
   by context, done ~80% by FRONT ATTENTION. Trigger → route → calibrate. **HIGH.** §638–639, §643–644.

7. **Newline circuit (flagship, fully traced; causally verified §728).** `.`/`!`/`?` embedding
   trigger (28× lift) → front attention discriminates real line-ends (0.47) from mid-paragraph (0.21)
   → block-17 calibration. **Causal AUC test (§728):** line-end discrimination AUC 0.806 collapses to
   **0.510 (chance)** when front attention is ablated (vs 0.789 random) — front attention carries ~all
   the discrimination. **HIGH.** §635, §637, §639, §643–644, §728.

8. **Article circuit (traced; corrected 614; causally verified §729).** be-verb→a/an; **preposition→the**
   (was wrongly "a/an"); punctuation→the. Front attention carries the a/an-vs-the *choice*; front MLP
   carries the *magnitude*; block 17 calibrates "the". **Causal AUC test (§729):** the-vs-a/an AUC 0.870
   → 0.703 when front attn ablated (drop 0.167, 24× random) > front-mlp 0.737 — **confirms attn=choice,
   mlp=magnitude**, but the choice is more DISTRIBUTED than newline (front-attn ablation ≠ chance). **HIGH.**
   §636, §640, §729.

*Circuit-verification method (§726–729):* every named circuit is now checked by **causal output
selectivity** (ablate → which behavior's CE/AUC collapses), not firing. This verified items 7–8 and
BROKE the false "boundary circuit" (§726–727: only mlp16 causally boundary-selective; block1.attn fires
at boundaries but writes open-vocab continuation). See method note.

9. **A token "class" can hide two circuits.** Digit: *continuation* (prev digit→digit) vs
   *initiation* (first digit after $/page/word). Initiation is computed (9.4×); the average misleads.
   **MED.** §641–642.

10. **Induction/copying — ALREADY MAPPED in the census (name circuit attn0+attn1 build the copy
    source; "induction-target" motif). This run re-derived it and added:** natural-text induction is
    rare-token-dominant (P 0.33 for rare vs 0.08 frequent) and distance-robust; reader heads L5.H5
    (z+3.99)/L8.H4/H6/L10.H8 attend to the copy-source, BUT under ablation the causal copying is
    **distributed across ~the whole attention stack** — top-16 pattern-heads = 19% of the effect,
    all-attention = 87% (§649). So attention-pattern salience ≠ causal contribution; copying is NOT
    a localizable head-set. **MED / overlaps prior work.** §645–649. ⚠ opened without checking it
    was done — a tracking miss.

11. **Stateful context registers exist but are read-correlates.** The model tracks
    counting-based context state — quotation parity (probe AUC 0.83, peaks mid-network then decays;
    §667) and parenthesis depth (AUC 0.92 from block 2, behaviorally 600×; §669) — decodable and
    behaviorally used, but removing the decodable direction does nothing (read≠write): the causal
    mechanism is distributed, per item 1. A capability dimension beyond the token-class/frequency
    machinery. **HIGH.** §667–669.

12. **Massive activations = the rms-norm gain controller (not attention sinks).** A few residual
    dims (persistent 645/990/981) grow to 20–60× the median by block 17 and dominate ~85% of the
    residual sum-of-squares, so their large DC offset *sets the rms-norm scale* for the readout —
    removing that offset costs +1.58 nats (§680). They are NOT token/position sinks (uniform across
    both — this model has no softmax, so no sink mechanism; §678), and they host the frequency-
    calibration direction (88% of `w_freq`; §676). **HIGH.** §676–680.

13. **Softmax-free attention = a two-criterion multiplicative conjunction (mostly positional ×
    content).** Attention is focal (~0.23 eff-keys vs 0.64 random; §681) despite no softmax, because
    each head multiplies TWO QK scores (`pat = s1·s2`): each alone is diffuse (~0.54), the product
    focal (more focal than both in 100% of cases; §682). The two QK circuits are complementary
    (corr ~0, 0/162 redundant; §683) and 44% of heads factorize into one positional (distance-
    selective) × one content QK — nearly all use positional selectivity in ≥1 QK (§684–685). So the
    model does lookup-style attention by AND-ing a positional and a content criterion. **HIGH.** §681–685.

14. **Embedding-dominant residual — the current token is kept present to the readout, distributed.**
    The residual rescale is `x = λ₀x + λ₁x₀` with the embedding re-injected at **λ₁≈8 at every
    block** (a systematic gain, not decay), while λ₀ **resets** the running residual in the front
    (L1 λ₀=0.013, L5=0.064 nearly zero it) and accumulates in the back (§689). Functional confirm:
    the current token's identity stays **linearly recoverable from the FINAL residual** (log-freq
    probe R² 0.91→0.85→**0.73** across depth, slow decay not transformed-away; shuffled null −0.43;
    §690) — unlike a normal transformer that transforms the current token into context. The
    embedding is dimensionally **flat** (per-dim RMS peak 1.5×), so it carries identity in a
    distributed way; it is **NOT** the source of the massive dims — those (peak 58×) are built by
    the blocks (overlap 2/10, corr 0.14; §691). So embedding-dominance (item 14) and the massive-dim
    norm controller (item 12) are **independent** mechanisms sharing the residual stream. Blocks add
    context *on top of* an ever-present embedding. **HIGH.** §689–691.

15. **RSPD functional-rank map of the components (data-conditioned A-SVD, CE-priced).** Decomposing
    each component's decoder map on real activations, priced by held-out cross-entropy (r80 = smallest
    rank recovering ≥80% of the layer's loss-benefit): **attention c_proj maps are very low-rank**
    (block0 r80=2, block1 r80=1 — one direction = a boundary→continuation writer, block2 r80=8), and
    **mlp0/mlp17 are low-rank** (8, 4). But **mlp1/mlp2 are globally high-rank** (r80=128, 256; low-rank
    surrogates worse than ablation). r80 is **data-robust** (mlp0=8 identical across 3k–24k tokens,
    §699). The high-rank layers (mlp1/mlp2) are **genuinely high-rank**. Clustering tokens by
    decoder-output direction and giving each cluster its own low-rank subspace gives a **modest, real
    advantage** over one global subspace at matched low rank (cluster>shuffle>global, data-robust §707),
    but does **NOT** dissolve the high rank: at a fair 80% recovery bar, 7/8 mlp1 clusters still need
    rank ~128 (§709). The strong "union of low-rank circuits" reading of §704–705 was **overstated and
    is corrected** (§708–709): the effect is real but small (even 32 clusters at rank-8 recover ~35%).
    Scope: this is the **decoder map conditioned on real activations**, not the full bilinear-MLP rank.
    Fast A-SVD (normal-equations right-inverse) = 17.5× over the library, N-linear (§700). **HIGH**
    (method validated, controls+nulls; over-optimistic conclusion corrected §709). §694–709.

16. **Decomposition metrics: a learned overcomplete sparse dictionary beats SVD/A-SVD.** SVD/A-SVD are
    faithful (exact at full rank) but **dense** (~120/256 components per datapoint, §740) and **CE-
    catastrophic at low rank** — A-SVD orders by response *energy* (massive-activation directions, §737),
    so its rank-4 reconstruction is worse than ablation (§737, §748). Metrics genuinely differ (weight-SVD
    wins efficiency/composability/monosemanticity, A-SVD marginally wins raw parsimony). The right family
    is a **learned top-k sparse dictionary**: per-datapoint it recovers far more CE than SVD at the same k
    (activation-SAE 86% vs SVD −233% at k=8, §748), validated on ground-truth toys (recovers planted atoms
    perfectly, respects shared/hierarchical structure, doesn't hallucinate; §743–745, §747). The **novel
    weight-action** form — factor `W ≈ D·E`, codes `E·gate` top-k sparse — is faithful to the *weight* and
    CE-faithful on the real mlp1.Down (87% at k=8, §750; soft-L1 failed §749, hard top-k works). Orthogonal
    rotation can't sparsify (§741); overcompleteness is required. Real layers differ: mlp16 (rank-1) is
    genuinely simple (SAE≈SVD), mlp0/mlp1 have rich sparse structure the SAE finds (§746). Metric for
    "right decomposition" = atom-recovery + code-length-matching-true-k on toys; k tuned by the recovery
    peak (§745). **HIGH** (toys + real, controls+nulls). §737–750.

### Architecture facts worth keeping
- MLP = `Down[(Lx)·(Rx)] + b`: every output dim is an exact **quadratic form** `xᵀMₖx`. mlp17's
  *output* is rank-8 by **variance** (§615), but its **functional (loss) rank is higher** (§660):
  ~4 quadratic functions recover 75% of its loss (the "~4" answer holds at that level), but the
  top-8 variance dirs recover only 78% — the low-variance tail (last 5% of var) carries ~22% of the
  loss. Variance rank ≠ functional rank (extends §617's variance basis ≠ functional basis). So
  "mlp17 = 4 quadratic functions" is a good ¾-approximation, not an exact reduction. (Q3)
- Residual is rescaled every block (`x = λ₀x + λ₁x₀`); a writer 12 layers back arrives ×∏λ₀ ≈ 2e-4
  (front L1/L5 λ₀ near-zero reset the stream; embedding re-injected at λ₁≈8 every block — item 14).
- Logits are `30·tanh(lm_head(rmsnorm(x))/30)`.

## Open / focus (hierarchical — go deeper on any)

- **A. Finer component isolation — METHOD WORKS, WITH A SCOPE (§650–652).** Behavior-conditioned
  low-rank + REMOVAL isolates **additive/subtractive** components to rank-1: block-17's calibration
  = one direction `w_freq = cov(mlp17 out, log-freq)` — removing it kills it (103%), random removal
  0–2% (§650–651). But it FAILS on **conditional/routing** computations: removing the rank-1
  `w_route` does nothing to the newline routing (§652), which is distributed in front attention
  (§644) and whose correlational direction is a decode-not-cause readout (read≠write). Rule: isolate
  by (behavior direction + removal + random control); expect rank-1 for biases/calibrations, NOT for
  context-conditional routing. **Scope confirmed (§653):** routing has NO removable linear carrier
  at any rank (top-32 removal = 0%); it's computed by attention + read nonlinearly. So finer-grained
  isolation is answered by component *type*: rank-1 for additive/subtractive biases, not for
  conditional routing (the wall there is nonlinearity/distribution, not redundancy). Arc §650–654,
  now confirmed on **three** behaviors: calibration = rank-1 isolable; newline routing & article
  magnitude = conditional, no low-rank carrier.
- **B. ~~head-SET localization of induction~~ — ANSWERED NO (§649): copying is distributed across
  ~all attention; pattern-selection can't isolate it. → reinforces A (need subspace method).
- **C. Systematic circuit discovery** vs the current opportunistic depth-first tracing (see method note).
- **D. Middle's within-class refinement mechanism** — hit the redundancy wall (§633), unlocalized.
- **E. Reconcile induction reader-heads (L5.H5) with census name-circuit source-builders (attn0/1).**

## Method note (how ideas are generated — honest)
Opportunistic, depth-first: pick a behavior, trace it output→input causally, follow each result to
the next question, generalize/contrast, turn contradictions into experiments. **No systematic
enumeration and no dedup against prior work** — which is how induction got re-run. Fix: consult this
index before opening a "new" thread.

**Circuit-naming rule (§726–727):** name a circuit by its **causal OUTPUT selectivity** (ablate it,
measure which behavior/token-category the CE-increase concentrates on), **not** by its **firing
pattern** (which input tokens make its coefficient spike). Firing tells you *when* a component
activates; only causal ablation tells you *what it does*. Over-reading firing as function created the
false "boundary→continuation circuit at 3 layers" (§726): of block0.attn-dir1 / block1.attn-rank1 /
mlp16-rank1, only **mlp16** is causally boundary-selective; **block1.attn fires at boundaries but does
general open-vocab continuation** (§727). This is the **fires≠contributes** face of read≠write
(items 1–2). Verify every named circuit this way before trusting the name.
