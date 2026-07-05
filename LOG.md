# Experiment log: multi-family graph tracing

## Session 4: causal tests of the copy-mechanism story + does COMPUTATION on the graph force a map?

User's four new ideas (2026-07-05), kept as the working list:

1. **Graph-computation queries (try first).** If *computation* on the graph is what
   matters, interleaving questions about the map into the walk documents should force a
   positive map. Concretely: append question blocks to walk documents using reserved
   special tokens — `[QDIST] u v ANS_d` (what is the graph distance between node u and
   node v?). Modular addition ("Sunday + 2 = Tuesday") is well-defined on the DIRECTED
   ring (undirected rings have no canonical direction), so `[QADD] u ANS_k v` there.
2. **Self-loop grid.** If the copy mechanism is truly the organizing principle, a grid
   where a walk can repeat the current node (self-edges) should causally install the
   positive mode — recurrence pressure inside the task family itself, no burst data.
3. **Boosted backtrack.** Same idea, milder: make the transition matrix return to the
   previous node with 2× the default weight.
4. **Natural-language co-training.** Mix natural-language documents (reduced ~5k vocab)
   with walk documents — does NL alone install the map in the toy softmax stack?

Plus: brainstorm which OTHER question types should require positive self-organization
(and which should not — those are the controls).

### Pre-registered predictions (before any session-4 training)

- **P5 (queries).** softmax-add-3L trained on grid+dring WITH distance queries flips
  positive (baseline grid+dring is reliably anti; new softmax baseline trained this
  session, bilinear baseline −0.55/−0.70/−0.72). Reasoning: the anti-map (neighbors
  antipodal) supports next-token elimination but distance is monotone-decreasing in
  neighbor similarity, so a metric query rewards neighbors-nearby coordinates.
  Falsifiable alternative: the model answers queries by a separate in-context lookup
  circuit (like induction solves the walk task in GPT-2) and organization stays anti.
  Query accuracy is reported so "never learned the queries" is distinguishable.
- **P6 (self-loop).** grid-selfloop+dring kills the anti mode for both archs (predict
  org ≥ +0.2, vs bilinear baseline −0.55..−0.72), because repeat-of-current-node is
  exactly the burst ingredient placed inside the family.
- **P7 (backtrack-2x).** grid-bt2x+dring moves positive but more weakly than P6
  (backtrack is already present in uniform grid walks; 2× is a dose increase, not a new
  ingredient). Predict org > baseline but possibly still ≤ 0 given dring's pull.
- **P8 (NL mix).** NL co-training alone (no burst, no queries) makes softmax-add-3L
  non-anti on the walk families (natural text = maximal re-predict pressure, per
  session 3's attribution result).

### Session-4 log

- [launched] toy_query.py (grid+dring with distance queries, both archs × 2 seeds) and
  toy_recur.py (softmax grid+dring baseline; self-loop grid + backtrack-2× grid, each
  +dring, both archs) running concurrently. Smoke tests: query docs well-formed
  (distances 0–6, sensible histogram), boosted walks backtrack 51% vs 34% uniform,
  self-loop walks repeat the current node 26% of steps.

- **[P5 FALSIFIED — appended distance queries do NOT install the map]** softmax-add-3L
  on grid+dring with interleaved `[QDIST] u v ANS_d` blocks stays reliably anti at long
  context (org256 −0.63 / −0.72 across two seeds; baseline −0.77), PC1/PC2 harmonic corr
  ~0.05/0.22 (no map). Bilinear likewise −0.54. The models DO answer distance queries
  (top-1 acc 0.30–0.38 vs a 0.29 always-guess-the-mode marginal), so they compute a weak
  distance signal WITHOUT a neighbors-nearby geometry. **Caveat:** queries are appended
  AFTER the walk, so metric pressure lands on the query-block reps, not the walk-node
  reps that mean_reps aggregates — the model can keep "walk mode" and "query mode" as
  separate circuits. The sharper test (queries interleaved throughout the walk, forcing
  the running node-rep to carry position-on-map) is logged as future work; the
  self-loop result below made it lower priority.

- **[P6 CONFIRMED — self-loops flip the sign; P7 FALSIFIED — backtracking does not]**
  Trained on grid+dring (the pairing that reliably pins ANTI):
  - grid baseline (softmax): org256 **−0.77**, no map (PC corr 0.05/0.07). Reproduces −0.80.
  - **gridSL (self-loop) softmax: org256 +0.53**, clean grid map (PC corr **0.97/0.76**). FLIP.
  - **gridSL (self-loop) bilinear: org256 +0.21** (baseline −0.55..−0.72). FLIP, both archs.
  - gridBT2 (backtrack-2×) softmax: org256 **−0.80**, no map (PC corr 0.03/0.08). NO flip.
  - gridBT2 (backtrack-2×) bilinear: org256 **−0.42** (baseline −0.55..−0.72). NO flip (within noise).
  - grid+dring+NL (natural-language co-train): **arch-dependent.** softmax org256 **−0.51**,
    no map (PC corr 0.08/0.23) — anti mode SOFTENED (from −0.77) but did NOT flip.
    bilinear org256 **+0.46**, clean map (PC corr 0.89/0.87) — FLIPPED (baseline −0.55..−0.72).
    So NL pushes both archs toward positive; the push suffices to flip the weakly-anti
    bilinear but not the strongly-anti softmax stack. (Consistent with session 3: bilinear
    is positive-prone, softmax anti-prone; NL is a moderate positive nudge, self-loops a
    strong one.)

  **The discriminating variable is LAG-1 adjacent-token repetition, not reversibility.**
  Measured repeat rates: grid lag1 0.00 / lag2 0.34 (anti); gridSL lag1 **0.26** / lag2
  0.26 (positive); gridBT2 lag1 **0.00** / lag2 0.51 (anti). Organization tracks lag1
  (0→0.26→0, i.e. anti→positive→anti) with a monotone relation, and is UNRELATED to lag2
  (0.34→0.26→0.51). Backtracking at 51% (A→B→A, tokens never adjacent-equal) does nothing;
  self-repetition at 26% (A→A, adjacent-equal) flips it. This refines session 3's
  "recurrence" into: **the organizer is adjacent duplicate tokens in the node stream the
  reps are measured on** — exactly previous-token/copy (induction-precursor) pressure.
  It also explains the NL null: wikitext supplies adjacent duplicates only in the SEPARATE
  text-token stream (node tokens live in a reserved id range), so it merely softens, not
  flips — the copy pressure has to be ON the node tokens.
  Confirmatory dose-response launched (toy_stutter.py): a pure lag-1 stutter, p ∈
  {0.00, 0.15, 0.30, 0.50}, graph left as plain grid.

- **[DOSE-RESPONSE — lag-1 stutter installs the map, confirmed and quantified]** A pure
  lag-1 stutter (duplicate the current node token with prob p; graph UNCHANGED plain
  grid, no self-edges), grid+dring, softmax-add-3L:
  - p=0.00 (control): org256 **−0.77**, no map (PC 0.07/0.11) — reproduces baseline exactly.
  - p=0.15: org256 **+0.43**, clean map (PC **0.91/0.82**), legal 0.94.
  - p=0.30: org256 **+0.54**, map 0.96/0.73, legal 0.62.
  - p=0.50: org256 **+0.55**, map 0.90/0.79, legal 0.42.
  Threshold-then-saturate: even 15% adjacent repetition flips the sign to a clean map;
  the effect saturates by p≈0.30. Because the graph is untouched, this isolates the
  SEQUENCE STATISTIC — the map is a property of token co-occurrence, not graph topology.
  Legal rate falls with p (the model spends more prediction budget on "repeat current
  token"), so p≈0.15 is the sweet spot: flipped map AND high task accuracy.

### Session-4 verdict summary

| manipulation | softmax @256 | bilinear @256 | map? | verdict |
|---|---|---|---|---|
| grid+dring baseline | −0.77 | ≈−0.63 | no | reliably anti |
| + self-loops (lag-1 ~0.26) | **+0.53** | **+0.21** | yes | **P6 ✓ flips** |
| + backtrack-2× (lag-2 0.51, lag-1 0) | −0.80 | −0.42 | no | P7 ✗ no flip |
| + natural language | −0.51 | **+0.46** | bilin only | P8 arch-dependent |
| + distance queries (appended) | −0.63/−0.72 | −0.54 | no | P5 ✗ no map |
| + lag-1 stutter p=0.15 (graph unchanged) | **+0.43** | — | yes | dose-response ✓ |

**Synthesis.** Session 3 said "recurrence installs the map, natural language is maximal
recurrence." Session 4 sharpens *which* recurrence: it is **lag-1 adjacent token
repetition** — the immediate previous-token / copy signal that induction heads are built
on — measured on the very token stream whose representations we read. Reversibility
(backtracking, A→B→A) is NOT it: lag-2 returns at 51% do nothing. On-graph computation
(distance queries) is NOT it: the model answers metric questions through a separate weak
lookup without ever drawing the map. And the pressure must land on the node tokens
themselves: natural language in a disjoint token range only nudges (flips the
positive-prone bilinear, not the anti-prone softmax). The map is thus a shadow of a very
specific predictive pressure — "the token you just saw is worth predicting again" — and
its strength is dialable (15% suffices, saturates by 30%). This directly answers the
user's opening puzzle: real softmax LLMs organize positively because natural-language
pretraining is saturated in lag-1 repetition (function words, names, syntactic echoes)
applied to the same tokens they represent; the from-scratch toy softmax, fed only graph
walks with zero adjacent repeats, had no such pressure and was free to anti-organize.
- [figure] figures/toy_lag1.png (dissociation bars, lag-1-vs-lag-2 rates, dose-response).

- **[seed replication — self-loop flip is NOT a lottery]** gridSL+dring across 3 seeds:
  softmax +0.53/+0.46/+0.53 (all clean maps, PC ~0.9); bilinear +0.21/+0.32/+0.50.
  Reliably positive both archs. (runs_gen/selfloop_seeds_results.json.)

## Session 5: finding the "next induction head" — depth-gated token categories

Goal (user): induction is the canonical 2-layer algorithm — a category of tokens
(repeated-context continuations) that a 1-layer model can't do and a 2-layer model can.
Are there categories that need MORE depth (3+)? We care about depth (longer sequential
algorithms), not breadth. Method: a task with per-token category labels spanning depth
requirements; sweep depth; find categories that light up only with more layers; then
causally check whether a depth-gated category recruits >2 attention heads.

Architecture ladder (bilinear attn + bilinear MLP `y=D(Lx⊙Rx)`, polynomial, norm OFF to
stay a tensor — tensorized RMSNorm to be slotted into deep_model.make_norm when supplied):
`attn·attn` (baseline) → `attn·MLP·attn` (2 attn + 1 middle bilinear MLP) → `attn·attn·attn`.
Param counts: 206k / 403k / 304k (the 4× bilinear MLP makes attn·MLP·attn the largest —
a param-matched control is planned if the MLP model wins).

Task (hop_data.py): in-context k-hop retrieval. Each doc defines a random E-cycle f on
E=32 entities, shown as bindings `[e, f(e)]`, then queries `[Q, e, H_k, a]` with
a=f^k(e), k∈0..3. Score the answer prediction bucketed by hop count:
k=0 copy (floor) · k=1 one lookup ≈ INDUCTION · k=2/3 chained lookups (need depth).

### Pre-registered predictions (session 5)

- **P9.** attn·attn solves k≤1 (copy + induction) but fails k≥2 (hop-2 accuracy near
  chance-above-copy). Depth-2 ceiling = the induction ceiling.
- **P10.** attn·MLP·attn and/or attn·attn·attn solve k=2 (and partially k=3): the extra
  sequential stage composes two lookups. Hop-2 answers are the depth-gated category —
  the "next induction head."
- **P11.** A depth-gated category recruits >2 attention heads (single-head ablations
  show ≥3 load-bearing heads for hop-2, vs 2 for hop-1).

### Session-5 log

- [pilot v1, seed 0, lerp residual, E=32, answer-only loss] attn·attn final acc by hop
  **[1.00, 0.955, 0.20, 0.19]** — solves copy and induction, FAILS hop-2/3 exactly at the
  depth-2 ceiling (**P9 supported**). But attn·MLP·attn STALLED on a copy-only plateau
  (loss ~2.54, hop-1 at chance through step 12k).
- [optimization note] Switching the whole ladder to add residual made it WORSE, not
  better: add-attn·attn also stalled on the copy-only plateau (hop-1 at chance through
  step 18k) where lerp-attn·attn had already reached 0.68. So in this k-hop setup the
  induction phase-transition is fragile and lerp escapes it while add does not (opposite
  of the deep-grid-walk collapse, where add was the fix — the interaction is task- and
  depth-specific). Robust setup adopted: **lerp attention residual, E=24 (stronger match
  signal), dense full-sequence CE (how induction heads actually form), 40k steps.** Pilot
  v2 running.
- [pilot v2, dense loss] BACKFIRED — dense loss stalled even attn·attn on the copy-only
  plateau (the random binding-VALUE positions are unpredictable, so dense CE floods the
  gradient with high-entropy noise and drowns the answer signal that drives induction
  formation). Reverted to **answer-position-only CE** (v1's working choice). Kept E=24.
- [pilot v3, answer-only, E=24, lerp, parallel per-spec] Escapes the plateau fast now:
  attn·attn hop-1 0.75 by step 6k; attn·MLP·attn escaped by step 12k (hop-1 0.92). BUT
  attn·attn·attn (3-layer attn-only, lerp) **DIVERGED** — loss → 1e21 by step 12k. Deep
  norm-free bilinear stacks explode (each bilinear layer ~squares the polynomial degree,
  so 3 layers ≈ degree 8; unstable at lr 1e-3 with no normalization — the exploding twin
  of the earlier vanishing collapse). Fix: **gradient clipping (max-norm 1.0)**, applied
  uniformly, all three restarted. (This instability is exactly what a tensor-compatible
  RMSNorm would cure; grad-clip is the stopgap until it's added.)
- **[DEPTH-GATING FOUND — the "next induction head" is a 3rd ATTENTION layer]** With
  grad clipping, at step 12k (seed 0):
  - attn·attn (2 attn):        hop [1.00, 0.77, **0.26**, 0.26] — induction ceiling.
  - attn·MLP·attn (2 attn+MLP): hop [1.00, 0.91, **0.26**, 0.26] — SAME ceiling.
  - attn·attn·attn (3 attn):    hop [1.00, 0.85, **0.86**, 0.75] — hop-2/3 SOLVED.
  The chained-lookup category (hop-2, f(f(e))) is unlocked ONLY by the third attention
  layer. The middle bilinear MLP does NOT substitute — because each hop is a
  content-based lookup (a match-and-copy), which needs an *attention* layer; a
  position-wise MLP cannot do content-based retrieval. So the depth ladder is literally
  a count of ATTENTION layers: hop-k needs k+1 attention layers (induction=hop-1=2;
  hop-2=3 = the next induction head). **P10 supported (via the 3rd attn, not the MLP);
  strong steer for P11.** Watching whether attn·MLP·attn forms hop-2 late by step 40k
  (so far flat). Full seeded sweep + causal head check next.
- [seed-0 finals] attn·attn [1.00, 0.79, 0.26, 0.26] · attn·MLP·attn [1.00, 0.97, 0.28,
  0.26] · attn·attn·attn [1.00, 0.94, 0.93, 0.86]. Clean: the 2-attn ceiling is at hop-1
  for BOTH 2-attn models regardless of the MLP; the 3rd attention layer alone lifts
  hop-2/3.
- **[P11 CONFIRMED — causal head recruitment, hop_ablate.py on attn3-seed0]** single-head
  zero-ablation, load-bearing = per-hop top-1 drop > 0.10 (baseline [1.00,0.94,0.93,0.86]):
  - hop-0 (copy): 4 heads, mostly layer 0 (shallow).
  - hop-1 (induction): 8 heads, layers 0–1 (+L2H1).
  - hop-2 (chained): **8 heads spanning ALL THREE layers** — all of L0, plus L1H0/L1H1,
    plus **L2H0/L2H1** (the third-layer heads, which barely affect copy).
  - hop-3: 9 heads across all three layers.
  Biggest single effect: ablating L0H3 drops hop-2 0.93→0.09 (the shared retrieval/
  prev-token substrate); the deeper layers stack the successive lookups on top. So the
  depth-gated categories causally recruit >2 heads and specifically use the 3rd attention
  layer — the algorithm genuinely occupies the extra depth, not just correlates with it.
  (runs_hop/ablate_attn3-seed0.json.)

### Query-type taxonomy (answer to "what else would require positive self-organization?")

A question forces a map only if its answer is a function of the graph's METRIC (many
hops), not of local neighborhoods (few hops). Sorted by predicted map-pressure:

**Should require a map (global / metric):**

1. **Distance** `d(u, v)` — implemented this session. Distance is monotone-decreasing
   in map proximity; spectral coordinates answer it directly, in-context adjacency
   lookup would need iterated composition (BFS), implausible in 2–3 layers.
2. **Closer-of-two** — "which of u, w is closer to v?" Ordinal distance: needs the
   metric but never an exact count, so it may be learnable where exact distance is
   hard. Good dose-response follow-up.
3. **Modular addition on the DIRECTED ring** — "Sunday + 2 = Tuesday". Needs a
   coordinate on the cycle (a rotation in the spectral plane), not just the metric.
   Note: undirected rings have no canonical direction, so this is dring-only (my
   earlier grid analogue of "how many away" is exactly the distance query, #1).
4. **Geodesic continuation on the grid** — given an adjacent pair u→v ("a direction"),
   name the node continuing straight. Requires a translation-invariant coordinate
   FRAME, the strongest form of map. Same family: vector arithmetic u + (v − w).
5. **Midpoint of u, v** — needs metric + betweenness; unique only on rings with even
   distance, so ring-only.
6. **Farthest node from u** (eccentricity) — global extremal query.

**Should NOT require a map (local — these are the controls):**

7. **Adjacency** — "are u, v neighbors?" One-hop lookup; the induction-style circuit
   that already solves next-token prediction suffices.
8. **Degree of u** — local count (on the grid this is corner/edge/interior
   classification).
9. **Common-neighbor count of u, v** — two-hop, still local.
10. **"Did u appear?"** — pure memory, no geometry at all.

Predicted dose-response: organization gain from queries should rank
adjacency/degree ≈ 0 < distance ≈ closer-of-two < addition/geodesic. If distance
queries flip the org sign but adjacency queries (same token budget, same format)
do not, the causal story is clean — it's the METRIC content of the computation,
not query formatting or extra supervision, that installs the map. Adjacency-query
control is the planned first ablation once the main result is in.

## Session 3 (6h): do REAL pretrained LLMs have this geometry — and what circuit makes it?

User's question: real LLMs use softmax attention and (per Park et al., on Llama-3.1-8B)
organize positively — but our from-scratch toy softmax model anti-organizes despite
perfect task performance. Can small pretrained models (GPT-2 scale, for fast iteration)
do the task on random-word walks, do they organize, and what exactly are they doing?

### Plan (user's directives, kept as the working list)

1. **Survey** small models — find ones where it works AND ones where it doesn't
   (knowing where it fails is explicitly wanted). Compare organization values to the toy
   models (toy bilinear multi +0.66, toy softmax −0.80).
2. **Circuit analysis** of a real model that does it (GPT-2 small): which components
   (heads/MLPs/layers) build the positive map.
3. **Causal test 1 — what creates the organization:** activation patching with changed
   sequences.
4. **Causal test 2 — is the mid-stack structure USED downstream:** corrupt the organized
   subspace at e.g. layer 6 and measure the behavioral hit (with a matched
   random-subspace control, to avoid the "same as changing the tokens" confound).
5. **Feed back into the toys:** use the circuit findings to identify what architecture or
   training-data change would make the toy softmax organize — and possibly make the
   bilinear-lerp toy even MORE self-organizing.
6. **Stretch — data attribution:** if the circuit is identified, look for which
   pretraining data enforces it (text whose low loss relies on that structure).

### Session-3 log

- [pipeline, llm_reps.py] Park protocol ported to pretrained LLMs: nodes = random common
  single-token English words (one fixed labeling per model), walks fed as plain word
  sequences (400 words, 96 walks); organization measured at EVERY layer × context;
  behavior (legal top-1 among node words); ownU/nbrU content coefficients in the model's
  own unembedding basis. Graphs: 4×5 grid, 12-ring, 7-ring.
- [first ladder: gpt2, pythia-410m, Qwen2.5-1.5B/3B, Qwen2.5-7B(8-bit)] ALL do the task
  in-context (grid legal top-1 0.82–0.99, rings ~1.00) and ALL organize the grid
  positively — Pythia/Qwen at the last layer (+0.42…+0.47), GPT-2 mid-stack (+0.34 at
  layer 11 of 12, decaying to +0.02 at the end). Park's Theorem-5.1 test PASSES for all
  five (grid harmonics in PC1/PC2, corr 0.71–0.97; top-2-PC Dirichlet energy 0.57–0.86 vs
  ~2 random). The phenomenon needs neither 8B scale nor Llama.
- [why real softmax ≠ toy softmax] The toy account transfers: every LLM's final layer
  carries big positive neighbor evidence (nbrU +1.4…+33 — the prediction itself) AND
  positive own-token content (grid ownU +0.18…+5.38). No LLM writes own-token suppression
  into the stream — natural text rewards predicting recent tokens again (the same pressure
  that makes induction heads copiers), so pretraining sits deep in the "reversible" regime
  of our reversibility account. The toy softmax's anti-map was an available implementation
  choice for tiny from-scratch stacks, not a property of softmax attention.
- [anomaly, logged] The smallest ring (7 nodes) INVERTS at the final layer in every LLM
  (−0.25…−0.55) at perfect task performance — the toy "7-star" readout mode exists in real
  LLMs too, on graphs small enough that the recent past covers most of the graph.
- [report] results_llm.md + figures/llm_{org,maps,coeffs}.png.

### Phase E1 (exploration, ~45 min): survey + in-context probes

- [survey, 11 models total] Every model that can DO the task organizes somewhere in the
  stack. pythia-70m is the "can't do the task" case (grid legal 0.10 — and no real
  geometry either, +0.19). The others: gpt2 +0.34 (mid-stack), gpt2-medium +0.46 (mid),
  pythia-160m +0.32 (mid, weak), pythia-410m +0.44, Qwen2.5-0.5B…7B +0.42…+0.47,
  OLMo-1B +0.37 (in layers 1–3! earliest organizer), opt-125m +0.49 (strongest small
  model, organized through the last layer, and the only model whose 7-ring is not
  negative at the readout). WHERE the map lives varies wildly by family: OLMo builds it
  immediately, GPT-2 mid-stack and tears it down at the end, Qwen carries it to the end.
- [in-context reversibility battery on GPT-2 — the toys' training result does NOT
  transfer to context] ring-12 walks: uniform (50% backtracks) best +0.40; biased-7:1
  +0.50; fully DIRECTED (never backtracks) +0.52; directed-k2 +0.55. In-context walk
  direction does not flip the map — if anything the more predictable walks organize
  better. The reversibility effect in the toys is a TRAINING-time (weight-learning)
  phenomenon; a pretrained model's in-context map-building mechanism is
  direction-agnostic. (This also cleanly separates "data pins the mode at training" from
  "data pins the mode at inference" — only the former is real.)
- [token-type dissociation — competence WITHOUT geometry, in one model] Node labels =
  random NUMERALS instead of words: GPT-2 does the task BETTER (legal 0.91 vs 0.82) but
  builds NO map anywhere in the stack (best +0.01, vs +0.34 for words). First version of
  this probe assigned numerals in row-major order and showed +0.48 — that was GPT-2's
  numeric-order prior faking the result (grid neighbors were numerically consecutive);
  random assignment kills it. Rare BPE word-tokens behave like common words (best +0.30,
  legal 0.79) — so it's not embedding "richness", it's something specific about numeral
  embeddings (hypothesis for the exploit phase: the numeral subspace's strong static
  structure swamps or replaces the walk-induced map; test by projecting out static
  numeral-embedding structure and remeasuring).
- [shuffled control] Time-shuffling the walk kills organization (+0.02) — the map comes
  from transition statistics, not token identity. Good null for the whole pipeline.

### Phase X1 (exploit, ~45 min): the GPT-2 circuit that builds the map

Method: pre-LN residual stream decomposes exactly as embed + Σ attention-layer writes +
Σ MLP writes; attribute the organization (Gram–adjacency covariance at the map's peak,
layer 11) over components, then heads; then ablate (`gpt2_circuit.py`,
`gpt2_localheads.py`).

- [who writes organized content] Almost every attention layer's windowed-mean output is
  positively organized (attn2 strongest at +0.56); at head level, layer-2 heads lead
  (2.11 +0.67, 2.3 +0.62, 2.2 +0.59). Layer 2–4 is where GPT-2's known previous-token
  heads live.
- [the predictor of head organization is LOCALITY] Across all 144 heads, attention mass
  on offsets 1–3 predicts the head's output organization at **r = +0.60** (offsets 1–12:
  +0.61). The most local heads are the textbook previous-token heads (4.11 mass 0.97,
  2.2 mass 0.76). Mechanism: **a local attention window applied to a walk IS one step of
  graph message passing** — the head's output at node v is a blend of v's and its
  graph-neighbors' token embeddings, because walk-adjacent tokens are graph-adjacent.
  Stacked local mixing = Laplacian smoothing = the spectral map. This also explains why
  the map is direction-agnostic in context (a window doesn't care which way the walk
  runs) and why every pretrained family has it (they all have local heads).
- [what induction heads do: the task, NOT the map] Classic induction scoring finds the
  textbook GPT-2 induction heads (6.9 0.89, 5.5 0.87, 7.10 0.84) — and induction score is
  UNCORRELATED with head-output organization (r = −0.03); same-token match heads
  (0.5, 3.0, early) slightly anti-correlate (−0.21). The task-solving retrieval circuit
  and the map-building circuit are different heads within the same model — the toy
  "competence ⊥ organization" dissociation, reproduced componentwise inside GPT-2.
- [amplifier layers] Late attention (layers 9–10) is NOT local (offset-1–12 mass ~0.05)
  but carries the largest organized variance into the final map (covariance attribution:
  attn9/attn10 self- and cross-terms dominate; ablating them: map +0.35 → +0.11).
- [inheritance, causal] Mean-ablating the 16 most-local heads (layers 0–7) collapses the
  late heads' own output organization (attn9 +0.36 → +0.12, attn10 +0.31 → +0.13) and the
  residual map (+0.35 → +0.15), while only costing some behavior (legal 0.83 → 0.65 for
  the top-8 version). The composers are the local heads; layers 9–10 inherit and amplify.
- [circuit summary] local/previous-token heads write neighbor-blended content (message
  passing) → mid/late diffuse heads aggregate and amplify it → induction heads separately
  retrieve the answer. Organization is a *byproduct of the local-copy machinery natural
  text installs*, not of the task solver. This is the architecture-level ingredient the
  toy softmax stack lacks: its layer-1 previous-token information is used only as a
  K-composition pointer (matching), never blended positively into the value stream.

### Phase E2 (exploration): the numerals mystery + how robust is the operator

- [numerals: the map still FORMS, then gets buried] With numeral labels, the local
  attention layers still write organized content (attn2 +0.49, attn9 +0.43 — the
  message-passing operator is content-agnostic, as the mechanism predicts). What changes
  is the rest of the stream: GPT-2's late number-processing MLPs write large per-numeral
  features (centered norms 60–100) with zero-to-negative organization (mlp9 −0.11,
  mlp10 −0.05; with words these are +0.07…+0.12 and mlp5 is +0.37), and these bury the
  map in the residual Gram (resid +0.13 vs +0.34 for words). Projecting out the STATIC
  numeral-embedding subspace does not recover it (+0.10…+0.15) — the dilution is dynamic
  (MLP features), not the embedding number-line. Also noted: numeral organization varies
  with the random assignment (+0.01 vs +0.13 across two labelings) — labeling-lottery
  noise is larger for numerals; flagged.
- [comma-separator probe: the operator is content-aware, not strict-offset] Interleaving
  ", " between walk words: attn2's output organization goes UP (+0.56 → +0.65) and the
  residual map survives (+0.34 → +0.24). The local mixing behaves like "blend the recent
  content words", not "blend positions q−1..q−3" — robust to tokenization noise, which is
  presumably why the map shows up in natural settings at all.

### Phase X2 (exploit): is the mid-stack map USED? (user's causal question)

Method (`gpt2_usetest.py` + patches): define the map subspace at layer 8 = top-2/4 PCs of
the clean node-mean representations (verified: that content alone is the organized part,
+0.28; it holds 40% of node-mean variance). Delete or replace it in the residual stream
at every position from step 200 on; measure behavior at steps 350–400.

- [deletion] Removing the top-4-PC map subspace: legal 0.83 → 0.59, neighbor-mass
  0.46 → 0.26, and the downstream map goes NEGATIVE (−0.23). Matched random 4-dim
  deletion: no effect (0.81). So the subspace is causally load-bearing…
- […but so is unorganized node content] Deleting PCs 5–8 (same provenance, map left
  intact at +0.38) costs almost as much behavior (legal 0.63). The damage tracks
  "node-identity content deleted", not "map destroyed".
- [own-mean patch] Replacing the subspace content at every position with the node's OWN
  clean mean: behavior intact (0.822 vs 0.826). The class-mean content of the map
  subspace is sufficient — within-class variation is unused.
- [permutation patch] Same replacement with a RANDOM permutation of node means
  (nodes stay perfectly distinguishable; placement scrambled): legal 0.64.
- [automorphism patch — the decisive control] Permuting node means by a 180° grid
  rotation (a graph automorphism: the GEOMETRY is exactly preserved, organization still
  +0.28; only identities are relabeled, with no fixed points) is the MOST destructive
  condition: legal 0.28. If downstream read the arrangement, this patch would be
  harmless; instead damage scales with how far each node's content moved.
- **[verdict on "the structure is used & useful, right?"]** Not as structure. What is
  causally used is the CONTENT that happens to be arranged geometrically — node identity
  plus neighbor-evidence (the prediction). Preserving the geometry while relabeling gives
  zero protection; deleting equal-sized unorganized node content hurts nearly as much.
  The adjacency-respecting arrangement is the *shape* that useful content necessarily
  takes (neighbors' prediction-contents overlap through each other — the toy account),
  not a data structure the model consults. Park-style maps in GPT-2 are a readable
  byproduct of prediction-carrying representations, exactly as in the toys.

### Phase X3 (exploit): feed the circuit finding back into the toys

Pre-registered (toy_burst.py): if LLMs organize because natural text installs *positive
local copying* (recurrence), then adding a "burst" family — complete-graph K16 walks
where 50% of steps repeat one of the last 3 tokens; pure recurrence, NO graph structure
— to grid training should flip the toy softmax stack positive.
P1: softmax-add-3L grid+burst goes positive (it is −0.80 on the six-family mixture,
−0.55…−0.72 on grid+dring). P2: bilin-lerp-2L grid+burst positive too.

- [P1, seed 0 — CONFIRMED] softmax-add-3L grid+burst: grid organization **+0.51 (ctx 8)
  → +0.38 (ctx 256)**, grid legal **0.999**. The architecture that anti-organized on
  every previous mixture organizes positively at full competence once the training data
  contains recurrence. Stochastic *graph* diversity was never the active ingredient for
  softmax — copy-the-recent-past pressure is. (Two measurement bugs found and fixed
  before trusting this: icl_reps globally disables grads at import — re-enabled — and
  the legal-rate check indexed the NEXT node's neighbor mask, which on a bipartite grid
  is disjoint from the correct set, reading 0.00 for a perfect model.)
- [P1 seed 1 — supported with nuance] softmax-add-3L grid+burst seed 1: +0.53 (ctx 8) →
  +0.11 (ctx 256), legal 1.00. Neither burst seed is anti (vs 5/5 anti seeds for
  dring-partnered and mixture softmax); one holds +0.38, one decays to weakly positive.
  Burst removes the anti mode; long-context magnitude varies by seed.
- [P2 — CONFIRMED, and burst is the cheapest pinning partner known] bilin-lerp-2L
  grid+burst: **+0.65, +0.60** (2 seeds; legal 0.78/0.96). One structureless recurrence
  family pins the positive mode as strongly as the entire six-family mixture
  (+0.55…+0.66) — compare grid+cylinder (unpinned ±0.24) and grid-only (lottery).
  Recurrence is the active ingredient, matching the LLM data-attribution result.
- [mixture+burst — tug-of-war, as the reversibility account predicts]
  softmax-add-3L on six-family+burst: +0.62 (ctx 8) → **−0.33** (ctx 256), legal 1.00 —
  burst moves it a long way from −0.80 but cannot fully overcome the irreversible dring
  family at long context; recurrence pressure and irreversibility pressure literally
  compete, landing in between. bilin-lerp-2L on six-family+burst: +0.74 → **+0.62**,
  legal 0.99 — holds the mixture ceiling; burst does not push bilinear beyond +0.66.
  The positive mode saturates in data-space; "even more self-organizing" would need an
  architecture change (a dedicated local value-blending layer, mirroring GPT-2's
  previous-token heads) — noted as future work.

- [pre-registered, toy_localmix.py — the ARCHITECTURE version] Hard-wire the ingredient
  instead of training it: LocalMixModel adds a parameter-free causal local average
  (x += 0.5 · mean of previous 3 positions) after the embedding — one built-in
  message-passing step, the toy analogue of GPT-2's previous-token heads. Trained on the
  PLAIN six-family mixture (no burst). P3: softmax-add-3L+localmix flips positive
  (baseline −0.80/−0.67). P4: bilin-lerp-2L+localmix ≥ +0.66 with a cleaner Park
  spectrum.
- [P3 — PARTIAL] softmax-add-3L+localmix: legal 1.00, org +0.37 (ctx 8) → +0.05
  (ctx 256), and PC1 carries a true grid harmonic at 0.97. The anti mode is gone
  (from −0.80!) — but long context erodes to neutral, same tug-of-war as mixture+burst
  (dring's suppression pressure remains in the data). Wiring removes the anti default;
  it does not out-pull irreversible data.
- [P4 — FALSIFIED] bilin-lerp-2L+localmix: org +0.55 (below the +0.55…+0.66 baseline),
  legal only 0.68, train loss stuck ~2.9 (vs ~1.7) — the fixed pre-blur interferes with
  the bilinear model's content-matching. Hard-wiring the ingredient is a blunter tool
  than data pressure: burst data beat localmix wiring on both organization and
  competence. Closing synthesis: the map-building ingredient is best INSTALLED BY DATA
  (recurrence) and merely PERMITTED by architecture.

### Session-3 verdict summary

| question (user's list) | verdict |
|---|---|
| Can small pretrained LLMs do it? | Yes, all task-capable ones, down to GPT-2 124M; Theorem 5.1 passes everywhere; org +0.32…+0.49 vs toy bilinear +0.66 / toy softmax −0.80 |
| Where does it NOT work? | pythia-70m (can't do the task); GPT-2 with numeral labels (task 0.91, map +0.01 — buried by number-MLP features); every LLM's final layer on the 7-ring (readout inversion) |
| What circuit builds it? | Local/previous-token heads = walk message passing (locality↔org r=+0.60); layers 9–10 amplify (inheritance shown); induction heads solve the task, uncorrelated with the map |
| What causes it (patching)? | Ablating 16 local heads collapses the map (+0.35→+0.15) and late-head org; random heads don't; time-shuffled walks and in-context direction changes bound what the sequence must supply (transition statistics only) |
| Is the structure USED? | The content yes (deletion hurts, random-dim control clean); the arrangement no (automorphism patch preserving geometry is the WORST condition) — geometry is the shadow of prediction-content |
| Architecture for the toy softmax? | Not architecture — DATA: adding a structureless recurrence family removes the anti mode (grid+burst +0.38/+0.11, was −0.80); on the full mixture it's a tug-of-war (−0.33) |
| Data attribution? | All natural text funds the local-copy machinery (+0.93 nats uniformly on wikitext when ablated; no recurrence-bucket concentration) |

### Phase E3 (exploration): which pretraining data funds the map-builders (stretch goal)

- [wikitext ablation attribution] Mean-ablating the 16 local map-building heads on real
  text: +0.93 nats of loss EVERYWHERE (16 random heads: +0.09). Pre-registered guess —
  damage concentrates on recently-recurring tokens — NOT supported: repeat-gap-1–4
  tokens +1.01, novel tokens +0.91, no gradient. Verdict: the local-copy machinery is
  not funded by a niche data subset (lists, coreference); local context is predictive
  for essentially every token of natural text, so *all* data trains it. That is why
  every pretrained family has the machinery and the map is universal — and why no
  dataset-slice attribution can isolate it at head granularity (limitation logged).

## Session 2 (8h autonomous): WHY are geometric neighbors stored nearby?

Central question: what makes the multi-family model place graph-adjacent nodes near each
other in representation space, when single-family models (usually) don't?

### Pre-registered hypotheses

- **H6 (composition account).** The per-node mean rep decomposes as
  `h_v ≈ α·e_v + β·Σ_{u∈N(v)} e_u (+ γ·Σ_{2-hop} e_w + residual)` — own token plus a signed
  blend of *neighbor token embeddings*, because the stream at node v must carry the
  prediction (the neighbor distribution) into the unembedding. With near-orthogonal random
  embeddings this implies sim(u,v adjacent) ≈ 2αβ and sim(dist-2) ≈ β²·|common neighbors|,
  so **organization sign = sign(αβ)**. Predictions:
  P1: the regression explains most Gram structure (high R²) and sign(αβ) matches measured
  organization for every model — multi (+), softmax (−), grid+dring (−), and crucially the
  *seed-lottery* gridonly models should split by their β sign (seed2 + vs seed3 −).
  P2: distance profile on triangle-free grids — positive models show sim>0 at both d=1
  (2αβ) and d=2 (β²·common), anti models sim<0 at d=1.
  P3 (decisive intervention): a function-preserving reparameterization that flips β should
  flip the measured organization without changing behavior — establishing the sign is
  functionally free and *training data selects it*. If no such reparameterization exists,
  identify the circuit constraint that forces the sign.
- **H6b (why the data pins the sign).** Stochastic families reward hedged output states
  (β>0: write neighbor evidence the unembedding reads off directly); deterministic copying
  rewards suppression-style solutions (β<0). Test with families of *tunable neighbor-set
  overlap*: k-nearest-neighbor rings (adjacent nodes share k−1 neighbors) and the
  user-suggested **widening rings** (concentric rings 4→8→16 with radial spokes).
- **Deliverable measure:** representation-similarity-by-graph-distance profile
  (mean normalized inner product of mean reps at graph distance d) per model × family.

### Session-2 log
- [setup] Live progress artifact published; push notifications at milestones.
- [path decomposition, geometry.py] The residual stream decomposes exactly over paths
  (embed / o1 / o2 [/ o3]); attribution of the Gram–adjacency covariance over path pairs,
  14 models: **the o2 (prediction-write) self-term is POSITIVE in every bilinear model,
  including the anti-organized ones** (grid+dring: o2 +1.07 while total −0.54). Anti
  organization comes from cross-terms (embed×o2, o1×o2, embed×o1). The seed lottery is a
  cancellation balance (gridonly-seed3: o2 +8.3 vs embed×o2 −6.0 → net +0.04). Distance
  profiles: positive models decay monotonically (d1 +0.29 > d2 ≈ +0.08 > d3 −0.17), anti
  models alternate (d1 −0.33, d2 +0.11 — bipartite checkerboard).
- [content regression] Path contents on {own, nbr, 2-hop}×{embed, unembed} bases:
  **nbrU (neighbor evidence in unembed basis) is positive in o2 for every model** —
  functionally forced, it IS the prediction. The free knob is **ownU**: multi writes
  +0.97·u_own (ownU·nbrU > 0 → positive map); softmax-add-3L writes −3.69·u_own (→ anti);
  dring-trained models move own/recent-token suppression into the write paths
  (grid+dring o1 ownU −0.74 vs multi +0.09). Both implementations give the same behavior —
  suppression-in-writes (anti-map) vs suppression-in-readout (positive map) is an internal
  degree of freedom, which is exactly why single-family training leaves it to seed luck.
- [validation] Reconstructing every path from just 4 coefficients (own/nbr × embed/unembed)
  reproduces the measured organization across all 14 models with **r = 0.954** (systematic
  negative offset — off-basis residual content adds positive similarity; noted).
- [widening rings, zero-shot] Built the user-suggested structure (rings 4-8-16 + radial
  spokes, n=28, degrees 3-5). All three key models handle it zero-shot (legal 0.96-1.00,
  mass 0.64-0.74) and the organization split holds on a structure NONE of them ever saw:
  multi +0.65, softmax −0.75, specialist −0.37. Added as the first panel of the 3D viewer.
- [own-token intervention] Projecting each node's own-token directions {e_v, u_v} out of its
  rep moves EVERY model toward the positive map (−0.47→−0.09, −0.22→+0.08, −0.76→−0.29;
  positive models barely move). The anti-map lives in the own-token component. (An extended
  projection incl. OV-transformed own directions over-projects — kills positive org too —
  and was discarded as uninterpretable.)
- [reversibility battery — DECISIVE] New two-family runs, grid + X:
  dring-k2 (stochastic 1 bit, never backtracks) **−0.70**; dring-k3 (1.6 bits, never
  backtracks) **−0.38**; biased-7:1 (only 0.54 bits, backtracks 12.5%) **+0.67**;
  biased-3:1 (backtracks 25%) **+0.66**. Double dissociation from entropy: what pins the
  anti mode is **irreversibility** — whether the walk can return to its recent past.
  Coefficients close the loop: irreversible partners induce huge L1 own-token suppression
  (o1 ownU −2.17 / −1.50) vs reversible (−0.17 / −0.07); o2 evidence identical for all.
- [18-model scatter] Σ ownU over write paths vs organization: r = 0.76 (softmax an
  x-magnitude outlier with the right sign). Figure: figures/geo_why.png.

- [weight-edit test] Injecting own-token suppression into the multi model's OV writes
  (OV ← OV − λ·Σ u_x ê_xᵀ) dials organization down monotonically (+0.66 → +0.16 at λ=0.4)
  but degrades behavior with it (legal 0.995 → 0.336) — there is NO static reparameterization
  connecting the modes (the injected content is input-dependent at the logits). The two
  implementations are retraining-compensable alternatives, not a gauge symmetry: training
  data selects the mode; it cannot be flipped after the fact by a weight symmetry.

- [battery seed-2 replication] dring-k2 −0.80 (anti replicates), biased-3:1 +0.65 and
  biased-7:1 +0.64 (positive replicates), dring-k3 +0.05 vs seed0 −0.38 — k=3 is a boundary
  case: dose–response, the irreversibility pressure weakens as out-degree grows. Core
  dissociation unchanged (k=1,2: 5/5 seeds anti; biased: 4/4 seeds positive).

- [two-family seed-2, closing the last single-seed points] grid+ring **+0.67** (seed 0 was
  +0.38) and grid+tree **+0.65** (seed 0: +0.41) — the positive pinning replicates for both.
  Surprise: grid+cylinder **−0.24** (seed 0: +0.24) — pairing grid with its structural
  near-copy pins *nothing*; that condition behaves like grid-only training (an init lottery).
  Refinement recorded in the ANSWER below: an irreversible partner reliably pins the anti
  map; a genuinely *different* reversible partner reliably pins the positive map; a partner
  that is almost the same graph family adds no constraint at all.

### ANSWER: why geometric neighbors end up nearby (all links measured)

**Vocabulary used here and in everything below** (the log leans on these constantly):

- **Node, token, u, v.** Every training document is one random walk on one graph. For each
  document, each graph node is assigned a random token (a "word" from the 100-word
  vocabulary), so within a document "node v" and "v's token" are interchangeable. `u` and
  `v` are simply names for two nodes; "neighbors" are nodes connected by an edge, and
  N(v) is the set of v's neighbors.
- **Representation of a node (h_v).** Run the model on a walk. At every position where v's
  token appears, take the model's internal state vector (the residual stream after the last
  layer), and average those vectors — over a 50-token window and over many walks. That
  average vector is what we call "v's representation."
- **Organization** (the number reported everywhere, also "corr"). Compute the similarity
  (dot product) between every pair of node representations, and correlate those similarities
  with the graph's adjacency (1 if two nodes share an edge, 0 if not). Positive = nodes that
  are neighbors in the graph get *similar* representations — "neighbors stored nearby," the
  Park-et-al-style map. Negative = neighbors are actively pushed *apart* — the "anti-map."
  "+0.66 at ctx 256" means: correlation +0.66, measured after the model has read 256 tokens
  of the walk.
- **Embedding e_v and unembedding u_v.** Two different vectors for the same token. e_v is
  the *input* vector (the row of the embedding table that enters the network when v's token
  is read). u_v is the *output* direction: the model's score (logit) for predicting v's
  token next is the dot product of the final state with u_v.
- **Writes: embed / o1 / o2.** These models have no MLPs and no normalization layers, so
  the residual stream is an *exact* sum of three parts: the token's own embedding
  ("embed"), what layer 1's attention added ("the o1 write"), and what layer 2 added
  ("o2"). Every representation can therefore be split exactly into these parts, and the
  organization number can be attributed exactly across them.
- **nbrU and ownU.** For each write at node v, we ask (by regression) what recognizable
  content it contains. **nbrU** = how much of "the sum of v's neighbors' unembedding
  vectors" the write contains; a positive nbrU means the write directly raises the
  predicted scores of v's neighbors — i.e., it literally is the model's prediction.
  **ownU** = how much of v's *own* unembedding vector the write contains (positive = the
  write boosts v's own token's score; negative = the write suppresses it).

With that vocabulary, the answer in five steps:

1. **The state must contain the prediction.** Standing at node v, the model's job is to
   raise the scores of v's neighbors' tokens. The only way to do that is for the final
   write to contain the neighbor unembedding vectors with a positive coefficient — positive
   nbrU. Measured: positive in the layer-2 write of all 18 models tested, including every
   anti-organized one. This ingredient is functionally forced; it *is* the output.
2. **The prediction alone already places neighbors nearby.** If u and v are adjacent, then
   u's prediction-content contains v's token direction (v is one of the things u predicts),
   and v's prediction-content contains u's. Two vectors that each contain the other's token
   direction have positive overlap. So this forced ingredient, by itself, always produces
   the positive map. Measured: the layer-2 write's own contribution to organization is
   positive even in the anti-organized models (grid+dring: +1.07 while the model's total
   is −0.54). Whatever makes a model anti-organized has to come from somewhere else.
3. **The "somewhere else": own-token content, whose sign behavior does not fix.**
   Representations also carry the node's own token (and recently visited tokens). The model
   must make sure it doesn't *predict* tokens that can't come next — but there are two ways
   to do that with identical outputs: (a) write **negative** own-token content into the
   stream ("suppression in the writes"), or (b) write **positive** own-token content and
   cancel it at the readout stage (the direct embedding→unembedding route contributes a
   fixed negative score for the token currently being read). Same behavior, opposite
   internal sign — an internal degree of freedom.
4. **Total geometry = forced prediction-content + free own-token content.** Positive
   own-token content adds to the neighbor overlap (my state contains "me"; my neighbor's
   prediction also contains "me") → neighbors nearby. Negative own-token content subtracts
   exactly that shared piece → anti-map. Measured: across all 18 models, the summed
   own-token write coefficient predicts organization at r = 0.76, and rebuilding every
   model's geometry from just four coefficients (own & neighbor content, in embedding &
   unembedding bases) reproduces the measured organization at r = 0.954. Projecting the
   own-token directions out of every node's representation moves every anti model toward
   the positive map and barely moves the positive ones — the positive map is always
   underneath.
5. **What the training data pins: reversibility of the walks.** If the training walks can
   *never* return to a recently visited node (directed rings — the walk only moves
   forward), then recently seen tokens must be actively pushed down, that suppression lands
   in the writes, and the map inverts. If backtracking happens at *any* rate, the recent
   past is a legitimate part of the prediction and the map stays positive. The amount of
   randomness (entropy) is irrelevant — a nearly deterministic walk that backtracks 12.5%
   of the time organizes positively (+0.67, +0.64), while a genuinely random walk that
   never backtracks organizes anti (−0.70, −0.80). Two caveats to the pinning: it requires
   the second training family to genuinely differ from the first (pairing grid with its
   near-copy cylinder pins nothing — the sign reverts to init luck: +0.24 / −0.24 across
   seeds), and single-family training pins nothing at all (a pure seed lottery). Softmax
   induction-style models choose suppression-in-writes regardless of the data — that
   default is described, not explained.

## Session 1 (earlier): does multi-family training create the geometry at all?

*(Recorded before session 2; session 2's question came out of these results.)*

Goal: models trained on a *single* task learned task-specific tricks — the cycle model
learned "never predict any of the ~3 most recently seen tokens" (elimination), the grid
model learned "boost the token we just came from" plus two copying routes — and they stored
graph structure with neighbors pushed **apart**, the opposite of the Park et al. picture in
LLMs. Question: does training **one model on many graph families at once** force a more
general relational circuit — and with it, Park-style neighbors-nearby representations?

## Hypotheses (stated before running)

- **H1 (organization).** Multi-family training flips the organization measure positive
  (neighbors stored nearby) for architectures that anti-organized when trained on a single
  family. *Test:* the organization measure on grid documents for the mixture-trained
  bilinear-lerp 2-layer model vs its single-task twin (which scored −0.57). *Falsified if*
  the mixture model still shows ≤ −0.3 after 256 tokens of context. *Refined if* it flips
  for some architectures only.
- **H2 (shortcut removal).** The task-specific tricks conflict across families and should
  be abandoned. Concretely: (a) "suppress the last ~3 tokens" is *wrong* for undirected
  rings, where stepping back is legal — so the mixture model's layer-1 behavior should
  change, and removing layer 2 should collapse cycle accuracy (the single-task model kept
  0.88 accuracy with layer 2 removed, because the suppression trick lives entirely in
  layer 1); (b) "boost the backtrack" is *wrong* for directed rings — so on grid documents,
  removing layer 2 should leave much less than the single-task model's 0.84 legal rate.
  *Falsified if* the same trick signatures reappear with similar ablation numbers.
- **H3 (generalist competence).** One model reaches near single-task performance on every
  training family AND transfers zero-shot to held-out graph types (torus, Erdős–Rényi) and
  held-out sizes. *Benchmark:* the single-task grid model already transferred to torus at
  0.99 legal rate. *Falsified if* the mixture model scores < 0.9 legal on torus, or lags
  any training family's single-task ceiling by more than 0.1.
- **H4 (architecture).** Based on single-task results: softmax attention ≥ bilinear on raw
  performance; positive organization most likely for the bilinear 3-layer additive model
  and for softmax early in context.

## Design

- **Data** (`graphs.py`): uniform random walks on token-labeled graphs — at each step the
  walk moves to one of the current node's neighbors, chosen uniformly at random. One graph
  per document, 256 tokens per document, 100-token vocabulary. Training families: undirected
  ring (sizes 5–20), **directed ring** (the walk only moves one way — this is the original
  cycle task; 5–20), grid, cylinder (a grid wrapped around in one direction; 3×3 to 4×5),
  random tree (8–16 nodes), random 3-regular graph (10–16). Held out ENTIRELY: torus (a
  grid wrapped in both directions), Erdős–Rényi random graphs. Held-out sizes: ring 30,
  directed ring 27, grid 6×6, tree 24, 3-regular 24. Graph structures are pre-sampled;
  token labelings and walks are drawn fresh every batch.
- The mixture is *self-disambiguating in context*: e.g. a ring document and a directed-ring
  document are distinguished only by whether the walk ever steps backward.
- **Architectures** (all width 128, 1 attention head, no MLPs, no normalization layers).
  Naming: "bilinear" = attention scores are a product of two query–key dot products with
  **no softmax**, so attention weights can be negative; "softmax" = standard attention.
  "lerp" = each layer *averages* its input with its attention output (new stream =
  0.5·old + 0.5·attention output); "add" = the standard residual sum (new = old +
  attention output). "2L/3L" = number of layers. The five contenders: bilinear-lerp-2L
  (the original recipe), bilinear-add-2L, bilinear-add-3L, softmax-lerp-2L, softmax-add-3L.
  Training: 24k steps, batch 128, Adam lr 1e-3 with cosine decay.
- **Metrics**, defined once: **legal rate** = fraction of late positions (position ≥ 128 in
  the document) where the model's top-choice next token is actually a neighbor of the
  current node. **neighbor mass** = total probability the model assigns to the legal next
  tokens. For the deterministic directed ring both are computed against the unique correct
  token.
- Then: representation organization (the measure defined in the vocabulary above) per
  family × architecture; circuit analysis (attention-offset profiles, ablations) comparing
  the mixture-trained bilinear-lerp-2L against its single-task counterparts.

## Log

- [setup] Wrote `graphs.py`. Pool sizes: tree 360, 3-regular 360, ER 180 structures; the
  lattice/ring families enumerate their few possible structures. Next: validate the data
  generation (are walks valid? are legal-move masks right? family statistics), then a
  600-step pilot for learnability and speed.
- [validation] All families pass: every walk transition is a real edge, the next token is
  always inside the legal mask, and the number of legal moves equals the node's degree.
  Generating a training batch takes 11 ms (not a bottleneck).
- [pilot] bilinear-lerp-2L, 600 steps at 71 steps/sec: grid legal rate 0.02 → 0.49, tree
  → 0.53, torus (never trained on) → 0.42 already. Learnable; 24k steps ≈ 6 min per
  architecture.
- [launched] Full sweep: 5 architectures × 24k steps, all width 128, 1 head. ETA ~35 min.
  While training: writing the representation-organization and performance analysis
  (`analysis_general.py`).
- [interim, architecture 1/5 at 10k steps] bilinear-lerp-2L: grid legal rate 0.99, torus
  (unseen family) 0.98, training loss 1.71. Early support for H3 — and notably *faster*
  than the single-task grid model (which needed 20k steps to reach 0.99). Mixture training
  appears to help, not hurt.

### Sweep results (24k steps, legal rate; figures gen_perf/gen_training)

- **bilinear-lerp-2L: 0.99–1.00 legal rate on ALL eval sets**, including torus 0.99,
  Erdős–Rényi 0.99, and held-out sizes 0.94–1.00.
- **softmax-add-3L: 1.00 everywhere** (neighbor mass 0.75–0.96).
- bilinear-add-2L: stuck around 0.75 (loss plateaus at 2.4). bilinear-add-3L: **diverged**
  (loss ~1e20 — the additive-residual fix that worked in single-task training explodes on
  the harder mixture; the lerp recipe's halving of the stream was accidentally acting as a
  stabilizer). softmax-lerp-2L: 0.83–0.93 on the stochastic families but **fails the
  deterministic directed ring (0.10)** even though its single-task twin solved cycles.

### Hypothesis verdicts

- **H1 (organization) — CONFIRMED for the target architecture, falsified as a universal
  rule.** The mixture-trained bilinear-lerp-2L, measured on grid documents: organization
  **+0.66**, stable across context (its single-task twin: −0.57). Its top principal
  components draw a visibly local map — on grid, and on torus (+0.68) which it never
  trained on — and its representation of a 7-cycle is a clean heptagon with adjacent
  phases adjacent. BUT softmax-add-3L, the *other* model with perfect scores, is the
  strongest anti-organizer measured (grid −0.80, torus −0.84), and its 7-cycle
  representation is a **7-pointed star** — adjacent phases placed nearly opposite each
  other. Organization is decoupled from competence. softmax-lerp-2L repeats its
  organize-early-then-flip trajectory (+0.73 early in context → −0.49 late).
  Controls launched: a fresh-seed replication, plus grid-only training through the
  identical new pipeline (prediction: grid-only anti-organizes; if it *also* organizes
  positively, the flip would be an artifact of the pipeline, not the mixture).
- **H2 (shortcut removal) — CONFIRMED (a), NUANCED (b).** (a) The cycle elimination trick
  is gone: with layer 2 removed, cycle accuracy at length 5 was 0.88 for the single-task
  model, **0.00** for the mixture model — no layer-1-only fallback remains. (b) The grid
  backtrack-boost survives (layer 2 removed: legal rate 0.89) — reasonable, since stepping
  back is legal in most families; only the elimination trick (actively harmful on
  undirected rings) died. Rewiring beyond the predictions: layer 2's reliance on reading
  layer-1 output through its keys ("K-composition" — layer 2 finds where to attend by
  matching against what layer 1 wrote at each position) became all-or-nothing (cutting it:
  0.00 on cycles); layer 2's reliance on *copying* layer-1 output through its values
  ("V-composition") **flipped from load-bearing to unnecessary** (cutting it barely hurts
  grid, 0.91/0.80, and even *improves* cycle accuracy at lengths ≥ 10); and layer-2
  attention at the induction offsets is now positive on cycle documents (it was negative).
  One unified relational circuit: layer 1 = a short negative window over the previous few
  tokens; layer 2 = positive content-match retrieval plus raw-token copying.
- **H3 (generalist competence) — CONFIRMED, exceeded.** No training family lags; zero-shot
  torus/Erdős–Rényi at 0.99–1.00 legal; the mixture model even *beats* the grid specialist
  on grid documents (legal 0.995 vs 0.989, neighbor mass 0.811 vs 0.762). Weak spots:
  directed ring at unseen length 27 is 0.00 for the failing architectures
  (bilinear-add-2L, softmax-lerp-2L) while both champions hit 1.00.
- **H4 (architecture) — partially confirmed:** softmax-add-3L is the strongest performer,
  but the guess about who organizes positively was wrong: the positive organizer is
  bilinear-lerp-2L, and softmax-add-3L anti-organizes hard.

### Controls (H1 validation)

- **Fresh-seed replication of bilinear-lerp-2L**: performance again ~1.00 everywhere;
  grid organization **+0.60 → +0.55** across context. The positive flip replicates.
- **Grid-only control** (identical pipeline, mixture removed, same training steps): grid
  organization +0.00 early → **−0.14** late — non-positive, trending negative. So the flip
  is caused by the family mixture, not by the new pipeline. (Weaker anti than the old
  specialist's −0.57; same direction.) Side finding: even grid-only training transfers
  zero-shot to torus/ER at 0.98 legal — broad transfer is nearly free; what the mixture
  distinctively adds is the deterministic family (grid-only scores 0.63 on directed-ring
  length 27) and the representation flip.
- **softmax-add-3L, fresh seed**: 1.00 legal on all 13 eval sets again; grid organization
  +0.36 (after 8 tokens) → **−0.67** (after 256). Anti-organization at long context
  replicates; the early-context part of the trajectory is seed-dependent (seed 0 was
  negative throughout; seed 1 organizes mildly then flips, like softmax-lerp). The
  competence-vs-organization decoupling stands.

### H5 (pre-registered): which ingredient of the mixture flips the organization?

Candidates: (a) **conflict** — one family that punishes the anti-circuit's tricks is
enough (the directed ring punishes backtrack/recency heuristics); (b) **diversity** — many
families are needed regardless of conflict. Test: train bilinear-lerp-2L on two-family
mixtures — grid+directed-ring, grid+ring, grid+tree, grid+cylinder (identical pipeline and
steps) — and measure grid organization after 256 tokens. *Predictions:* if conflict-driven,
grid+directed-ring flips positive and grid+cylinder (a structural cousin of grid, minimal
new demands) stays ≤ 0. If diversity-driven, all two-family mixtures stay ≤ 0.

### H5 results — prediction FALSIFIED, mechanism refined (now 2 seeds per condition)

Two-family mixtures, bilinear-lerp-2L, grid organization after 256 tokens:

| mixture | organization (each seed) | verdict |
|---|---|---|
| grid + directed ring | −0.55, −0.70, −0.72 (3 seeds) | the "conflict" family *entrenches* the anti-map |
| grid + ring | +0.38, +0.67 | reliably positive |
| grid + tree | +0.41, +0.65 | reliably positive |
| grid + cylinder | +0.24, **−0.24** | **not pinned** — behaves like grid-only (init lottery) |

The conflict hypothesis had it exactly backwards: pairing with the deterministic directed
ring *preserves* anti-organization, while a genuinely different stochastic undirected
family (ring, tree) reliably flips it positive. The seed-2 runs also corrected an early
single-seed impression: the structural near-copy (cylinder ≈ grid with one wrap) pins
*nothing* — with almost no new demands, the sign stays at the mercy of initialization,
just like single-family training. Refined picture: **organization sign tracks the
algorithmic mode** — deterministic token-copying selects induction-style circuits, which
anti-organize; predicting *neighborhoods as sets* across genuinely diverse stochastic
families builds the positive, Park-style map. The full six-family mixture (directed ring
included) is strongly positive (+0.66): enough stochastic diversity overrides the directed
ring's pull. Session 2 later grounded all of this in the reversibility mechanism (see the
ANSWER above).

### Mech on the anti-organizing champion (softmax-add-3L)

- Layer 1 = a textbook "previous-token head": 97% of its attention goes to the position
  immediately before the current one, on both document types.
- Layer 2 = an induction head: it attends to the positions right after earlier occurrences
  of the current token (so it can copy what followed last time); removing it destroys the
  cycle task.
- **Layer 3 is nearly idle**: knocking it out keeps cycle accuracy at 0.94–0.97 and grid
  legal rate at 1.00 — and *raises* grid neighbor mass (0.895 vs 0.851), echoing the
  Q-composition calibration paradox from the single-task models.
- So the perfect softmax model is a classic 2-layer induction stack — and it anti-organizes
  even when trained on the full stochastic mixture. Circuit type, not data alone, sets the
  sign. Caveat: both circuit types use K-composition matching, so "retrieval needs
  separable neighbors" does not by itself explain the sign difference — open question
  (session 2's answer: the induction stack implements recent-token suppression in its
  writes; *why* softmax defaults to that implementation remains open).

### Seed batteries — MAJOR REVISION of H1 (user's skepticism vindicated)

Grid organization after 256 tokens, bilinear-lerp-2L unless noted:

| condition | each seed | verdict |
|---|---|---|
| grid only | −0.14, +0.67, −0.08, +0.16 | sign is a seed lottery (matches user's "prolly just init") |
| full six-family mixture | +0.66, +0.55, +0.62 | consistently positive, tight |
| softmax-add-3L on the mixture | −0.80, −0.67 | consistently anti |

Restated H1: multi-family training does not flip a predetermined sign — it **stabilizes
the positive mode that single-family training reaches only occasionally by luck**. The old
single-task grid model (−0.57) was one draw from a high-variance lottery. All two-family
conditions were subsequently re-run at 2–3 seeds (table in the H5 section above). What
survives regardless: (i) the mixture models have *genuine* Park-style geometry — the graph
appears in the **top two principal components** (correlations 0.81/0.80 with the graph's
spectral coordinates, top-2-PC Dirichlet energy 0.49 vs ~2 for random, the lattice moved
from PC12 to PC1) — a position-in-the-spectrum difference that no sign or rotation
symmetry can produce; (ii) the architecture effect (the softmax induction stack is
consistently anti at equal competence).

### Final verdict summary

| hypothesis | verdict |
|---|---|
| H1: the mixture flips organization positive | Confirmed for bilinear-lerp-2L (+0.66; replicated across seeds; the grid-only control stays ≤ 0, so the mixture is causal). NOT universal: softmax-add-3L anti-organizes (−0.80 / −0.67 across seeds) at equal competence |
| H2: single-task tricks die | The elimination trick is dead (accuracy with layer 2 removed: 0.88 → 0.00); the backtrack-boost survives (it is legal in most families); the circuit is unified: K-composition all-or-nothing, V-composition abandoned, attention sign flipped positive |
| H3: generalist competence | Confirmed and exceeded: two architectures at ~1.00 legal everywhere including zero-shot torus/ER; the mixture model beats the grid specialist on grid. But most raw zero-shot transfer comes nearly free even from grid-only training |
| H4: architecture ranking | softmax-add-3L strongest (perfect); recipe fragility extreme: bilinear-add-2L plateaus, bilinear-add-3L diverges, softmax-lerp fails only the deterministic family |
