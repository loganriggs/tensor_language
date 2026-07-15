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
- **[seeded sweep, 3 seeds — depth-gating robust]** hop-2 accuracy: attn·attn 0.26±.00,
  attn·MLP·attn 0.28±.01, attn·attn·attn **0.96±.02**; hop-3: 0.26 / 0.26 / 0.82±.09.
  Tiny variance, enormous gap — the 3rd attention layer is necessary and sufficient for
  chained retrieval; the middle MLP never helps on any seed. Figure figures/hop_ladder.png,
  report results_hop.md. **Session-5 verdict: the "next induction head" above induction's
  2-layer circuit is a 3-attention-layer chained-retrieval circuit; depth = a count of
  ATTENTION layers (one content-based lookup each), and MLPs don't substitute.**
- **[SURPRISE — adding RMSNorm DESTABILIZES hop-2 into a seed lottery]** Re-ran the ladder
  with standard RMSNorm (pre-norm, affine-free), no grad clip. It trains stably (norm cures
  the divergence, as hoped) and hop-1 IMPROVES (attn3-rms hop-1 ≈ 1.00 vs 0.94 no-norm), but
  the chained-retrieval circuit becomes fragile: **attn·attn·attn hop-2 = 0.26 / 0.25 / 0.94
  across seeds 0/1/2 — solved only 1 of 3**, versus no-norm (grad-clip) which solved 3/3
  (0.93/0.97/0.99). attn·MLP·attn and attn·attn stay at the 0.26 ceiling as before. So
  RMSNorm is NOT a neutral stabilizer for this algorithm — it turns a reliable outcome into
  a lottery, plausibly by rescaling away activation-magnitude information the intermediate
  retrieval uses (affine-free RMS division), or by a placement effect (norm feeds q/k/v but
  the residual stream still carries magnitude). The no-norm + grad-clip result stands as the
  primary depth-gating finding. OPEN follow-ups: affine RMSNorm; norm on q/k but not v; more
  seeds/steps to map the lottery rate. Runs tagged -rms (runs_hop/*-rms-*/acc.json).

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

---

# Program change (2026-07-08): deeper circuits — finding the next induction heads

New direction (Logan): train bilinear (tensor) attention ladders of increasing depth on
**natural text**, track which validation datapoints differentially improve as depth is
added (strong threshold + hysteresis), cluster gated datapoints by causal circuit, then
isolate each circuit in a toy task and study its training dynamics following Singh et al.
2024 (`induction_head_training_dynamics_paper.txt`). Roadmap + gotchas: `PLAN.md`.
Prior hop-ladder results (chained retrieval = depth-3 circuit, pointer-advance mechanism,
seed lottery, curriculum backfires 0/4) stand in `results_hop.md` and become the
isolation-stage template.

## Session 1 (2026-07-08): infrastructure + first ladder

- **Corpus decision (Logan): natural text from the start** — TinyStories, byte-level
  BPE V=1024, n_ctx 256. Built `data_text/`: train 616M tokens, frozen val 15M tokens
  (val split + 30k held-out train stories; held-out stories excluded from train.bin).
- Repo reorganized: legacy projects → `archive/`, raw run logs → `logs/`; active core at
  root (`model.py`, `deep_model.py`, hop_*, new lm_* pipeline).
- New pipeline: `text_data.py` (corpus), `lm_train.py` (depth ladder, identical data
  order across depths at fixed seed), `lm_eval.py` (per-token CE on frozen val),
  `differential.py` (depth-gated datapoint finder + report with decoded examples).
- Smoke test: bilinear attn-2 LM trains stably on text (no softmax; RMSNorm on):
  106 steps/s, 40k-step run ≈ 7 min, loss 6.9→3.9 in 300 steps.
- RUNNING: depth ladder attn1–attn4 × seeds 0–2 (two parallel workers).
- Heartbeat cron active (20 min, expires 2026-07-09 12:31 UTC).

### Questions for Logan

- Corpus resolved per your message (natural text). Recipe defaults I picked: TinyStories,
  BPE-1024, d_model 128, 4 heads, 40k steps — veto anytime.
- After the bilinear ladder: planning a softmax control ladder (same depths) to separate
  "bilinear-specific" from "depth-generic" gates. Default: yes.

## Descope (2026-07-08, Logan): drop the softmax comparison entirely

Focus is the TENSOR NETWORK program only. Softmax controls killed (TinyStories softmax
results stay in results_deeper.md as archive; partial OWT softmax run deleted). GPU
reallocated to the bilinear ladder on OWT: attn1-4 (dense ckpts on seed0 of attn2/attn3)
plus "full bilinear transformer" specs — blockN = N x [bilinear attn + bilinear MLP]
(block2 dense + 3 seeds queued, then attn4). The dynamics story is now purely internal:
how do bilinear circuits form, and what does each added attn layer vs bilinear MLP buy.

### Questions for Logan (mechdecomp, 2026-07-09)
- Tier 1.1/1.2 DGP semantics: independent-coefficient ρ=1 pairs are separable by
  nonneg-L1 alone (method recovers pure atoms — arguably a good surprise). Redesigning
  1.1 with tied coefficients (variation only along e0+e1) as the true merge case;
  1.2's mechanism-side-rescue claim needs the L1×gain geometry derived first — flag if
  you had a specific construction in mind.
- Tier 1.2 falsified as constructed: gain-anisotropic W does NOT rescue co-occurring
  features (basis-degeneracy survives; SAE comparison shows no advantage, 3 seeds).
  Salvage candidates in results_mechdecomp.md — gain-weighted L1 is my first pick.
- Mechdecomp Tier 1.5: contraction on GENERIC activations recovers L0H3→L1H2 in K2 but
  not K1 (L0H0's generic writing dominates K1). Weight-space check confirms it's real,
  not a method bug. The causal circuit is data-conditioned (source positions); plan to
  re-run contraction conditioned on the induction datapoints. Confirm this is the
  intended §1.5 reading for circuit discovery (I think yes).
- Mechdecomp Tier 1.5 KEY FINDING: mean contraction magnitude does NOT recover selection
  circuits (L0H0 dominates K1 by magnitude but is causally inert — position-constant
  baseline; L0H3 is the selective/causal one). Fix: score contraction by DISCRIMINABILITY
  (variance across candidate keys), not norm. Substantive refinement of spec §1.5's
  "circuit = matrix entry" for selection circuits. Confirm the reading.
- Mechdecomp Tier 2 blocker: on dense real OWT activations the kmeans+lasso solver gets
  R²<0.3 though the parameterization is COMPLETE (identity init + c=1 = exact). Fix =
  SVD/identity init (spec init (b)); kmeans (init c) only suits sparse toy data. Confirm
  intended regime for real layers: overcomplete-sparse vs near-complete-basis.
- Mechdecomp Tier 2 resolved: R²<0.3 was SITE (dense token-embedding activations, no
  sparse features), not init/solver (identity+dense = R²1.0). Pivot: sparse-recon Tier 2
  → Pythia/Gemma MLP-hidden activations (where superposition makes features sparse). Our
  tiny models served their role as Tier 1.5 circuit ground truth. Confirm the pivot.
- Mechdecomp Tier 2 Pythia-70m: NEGATIVE — worse than rank-21 dense (R² 0.49 vs 0.55 at
  L0 66) and atoms incoherent. Confounded with model size (Pythia-70m tiny). DECISION
  NEEDED: (a) jump to Gemma-2-2B (spec target, good features known), (b) retry Pythia on
  RESIDUAL STREAM first (cheap site test), (c) Pythia-160m/410m. I recommend (b)→(a).
- Mechdecomp Gemma BLOCKED: google/gemma-2-2b is GATED (401, no HF token with access).
  NEED FROM LOGAN: an HF_TOKEN with Gemma access (set in ${WORKSPACE}/.env as HF_TOKEN),
  OR approve a non-gated substitute for the interp/SAE-comparison target. Meanwhile
  testing Pythia-410m (non-gated, 6× larger than 70m) to check if interp improves with
  scale — confirms/refutes the "weak features" diagnosis before Gemma.
- Mechdecomp FULLY CHARACTERIZED: method needs data sparse in a W-aligned basis. Toys have
  it (perfect recovery); real Pythia activations don't (dense: low-rank map can't separate,
  high-rank can't sparsely reconstruct). Gemma residual stream is the DECISIVE test — it's
  SAE-validated-sparse (the exact required property). Still need HF_TOKEN for gemma-2-2b,
  OR approve GPT-2-small (has SAEs). This is the one experiment that resolves the method's
  real-world value. Natural handoff point.

### Mechdecomp RESOLVED (2026-07-09) — Gemma question now MOOT
Unblocked the crux without Gemma: used sae_lens + GPT-2-small res-jb SAE (non-gated,
24576 known-good features). DECISIVE RESULT: method atoms do NOT align with validated SAE
features (max-cos 0.16 ≈ random 0.15), and every real-model run hit distinct solver
instabilities. HONEST VERDICT (results_mechdecomp.md): method is exact/sound on toys and
circuit-contraction, but real-model feature discovery is unsupported as implemented (fragile
optimization + finds PCA-like not sparse directions). No Gemma access needed anymore — the
underlying question is answered. Validated contribution = toy recovery + circuit contraction;
real-model feature discovery = open problem (needs robust solver + objective reformulation).
- ATLAS CAVEAT WITHDRAWN (2026-07-09, same day): I flagged the atlas as containing a wrong
  "magnitude fails / selectivity recovers" claim. It does NOT — that claim lived only in
  results_mechdecomp.md's Tier-1.5 scoring. The atlas says "raw Frobenius-norm composition does not
  single out the causal edge; composed with the matched token's embedding (what L0 actually writes
  when it reads ' j') both K branches light up on L0H3", and closes with "weight products need the
  right input direction before they reflect the circuit." That directional composition uses the
  CORRECT input (a previous-token head at the source reads the previous token), and its caution
  literally anticipates the wrong-write bug I later made. **No atlas correction needed.**

## Questions for Logan (2026-07-09)
1. **Your experiment #2 is answered: "features are not optimal for this objective."** On a toy with a
   known generator, a correct refinement leaves the true dictionary untouched (Δ −0.0002, cos 0.9998).
   On GPT-2 the SAE feature basis moves: +0.033 held-out R²(Wx), atoms rotate ~60°. Features are still
   the best available init (beats random-init refinement by +0.023).
   *Caveat I want your read on:* "SAE features aren't GPT-2's generators" is close to tautological —
   an SAE with R²≈0.99 leaves structured residual. Is the informative content just the magnitude
   (+0.033 / cos 0.48)? Or is there a version of this test you'd consider load-bearing?
2. **The identifiability metric is dead.** OMP at the SAE's own L0, over the SAE's own decoder, beats
   OLS-on-the-SAE's-support (0.9328 vs 0.8807) with only 23% support overlap. The SAE's active set is
   not the reconstruction optimum at its own sparsity, so overlap-with-SAE-features can't adjudicate
   *any* reconstruction objective — including the older max-cos 0.16-vs-0.15 result you flagged.
   I've made "no SAE-overlap oracle" a standing rule. Ground truth must come from toys/Tier-1.5.
3. **Solver hygiene, for the spec:** OMP-support + M-step alternation is not guaranteed descent (the
   support is re-selected each round). With the fixed M-step it is monotone in practice at N=5.9k.
4. Program A (circuits) still paused. Tier 1.5 (bilinear models, defined ground truth) is the natural
   next mechdecomp step now that the SAE-overlap metric is retired — it's the only place an
   identifiability claim can actually be tested. Proceeding there unless you say otherwise.

## Questions for Logan (2026-07-09, later)
5. **Tier 1.5 passes — on the right quantity.** Atom-reconstructed write ablation recovers the causal
   L0H3→L1H2 edge at 4.07×. But the spec §1.5 per-branch contraction is *ill-posed*, not just weak:
   `|G|` names L0H0 for K1; signed-selectivity names L0H2 for K2. No variant gets both branches, and
   the answer changes with the statistic. Proposed spec amendment: for gated layers, replace "read the
   contraction matrix" with "ablate the atom-reconstructed write, measure Δ on the downstream score."
   Your standing gate ("must recover the edge in both K branches") presumes a separability that
   `Δs = a₁b₂ + a₂b₁ − b₁b₂` denies — I've restated it as the joint-Δŝ gate. Flag if you disagree.
6. **Tier 1 recovery is healthy**: the objective recovers the true dictionary from random init at
   K/d up to 4 (0.96–0.98 vs chance 0.35). The 4×-overcomplete case just needs 60 rounds, not 20.
7. **The lasso E-step is fine after all** — my "it fails" reading was λ mis-calibration (L0 28 vs
   ktrue 4). At matched L0 it ties OMP. So spec §3 needs a λ-calibration note, not a solver swap.
   The GPT-2 gap (OMP 0.90 vs lasso-debias 0.665 at matched L0) is real but is about real-data
   coherence, not a generic L1 defect. Recommend: calibrate λ to target L0, per dictionary.
8. **Weak-signal caveat worth your view:** my induction batches are random-token repeats, base match
   −0.022 vs −0.434 on natural text. Ordering reproduces. Worth re-running Tier 1.5 on natural
   sequences before this goes in a writeup?

## Questions for Logan (2026-07-09, tick 3)
9. **Tier 1.5 closed, both flagged items resolved without needing your call:**
   - Natural-text induction sites (mined, not synthesised; base match −0.186 vs −0.022 random-token):
     joint-Δŝ gate PASSES at **8.81×** (was 4.07×). Δŝ +0.087 tracks true Δs +0.104.
   - The uncitable "lasso 0.06–0.46" is retracted and re-measured. At matched L0=8 on L0H3-OV:
     OMP 0.9749, lasso 0.8278, lasso+OLS-debias 0.9131. The old figures were λ≥0.2 ⇒ L0≤0.9.
10. **res-jb has 45 live duplicate feature directions** (|cos|>0.999, e.g. 979↔2039 at cos=1.000000,
    clustered around feat 316). Not dead-feature collapse — the top-2048-by-usage dictionary still has
    mutual coherence 1.0. This is a fact about a widely-used SAE that may interest you independently.
    An exactly duplicated pair is unidentifiable for any solver; L1 splits mass between the twins,
    which is where it loses R² at matched L0. I now think the OMP-vs-lasso gap is *mostly* this.
11. **My "W's null space merges features" hypothesis was wrong** — the collinear pairs are collinear
    *before* W (null-space energy 2.8%). Mentioning because it was the intuitive story and it failed.
12. Next: Pythia tier with the validated (Gauss-Seidel) M-step and λ calibrated to target L0. The
    old Pythia runs used the un-validated solver, so I plan to re-run rather than cite them.

## Questions for Logan (2026-07-09, tick 4)
13. **RETRACTION of my item 10.** I claimed the OMP−lasso gap is "mostly" L1 splitting mass between
    duplicate feature twins. Tested and false: deduping the 45 twin pairs changes the gap by 0.0002
    (0.1219 → 0.1217). The twins form a clique around feat 316, so dedupe removes just 9 of 2048
    atoms — it could never have explained 0.12 R². The gap is ordinary L1 shrinkage + support
    selection on a coherent dictionary. The duplicate-features fact about res-jb stands on its own;
    it is simply not the cause. Apologies for asserting it before testing.
14. **Tier 2 (Pythia-410m) re-run, validated solver.** `down_proj` L3 (1024×4096, rank 1024).
    Held-out R² 0.6020 at L0 32 from random init; svd-init gives no lasting advantage (0.5985).
    Floor check: PCA-32 0.3509, PCA-64 0.4231, OMP-32-over-PCA-64 0.3897 — the learned sparse
    dictionary dominates all of them, so sparsity *and* learning each buy real margin.
    **This retracts the program's old "finds PCA-like, not sparse, directions" claim** — that was
    the un-validated M-step talking. Absolute quality is still mediocre (0.60 vs 0.97 on attn2),
    and 25% rowspace visibility means down_proj atom interpretations are rowspace-only claims.
15. **A concrete identifiability number for the spec:** for `W: R^d_in → R^d_out`, atoms are
    recoverable only up to `row(W)`; cos(true, identifiable) = sqrt(rank/d_in). Pythia down_proj:
    0.50. GPT-2 OV: 0.993. Worth stating in §1.4 — it decides which layers are even decomposable.

## Questions for Logan (2026-07-09, tick 5) — TWO PROGRAM-A RETRACTIONS
16. **"10% copy-burst mixture installs induction 3/3 seeds" is WRONG.** The bursts are `[u[:128]; u]`
    with a CONSTANT period of 128 (lm_train.py:90-96). In that distribution, content-matching and a
    fixed −127 offset are exactly equivalent, so the model learns a **positional copier**. Vary the
    period: mix10 scores 0.9036 at P=128 and 0.0001–0.0003 at P=150/100/64 (chance). Fix is one line:
    randomise the repeat period per row. Want me to retrain the mix10 ladder with random periods?
17. **tiny attn2-seed0 is not an induction head model.** On the exact training-burst distribution it
    copies arbitrary repeated tokens at 1.2× chance (mix10: 4609×). Its L1H2 is a repeated-bigram
    match-and-copy circuit on natural text — causally real, wrongly named. Program A's "induction
    recovered unsupervised from depth-2 gates" needs the same relabelling.
    **Tier 1.5's mechdecomp gate is unaffected**: it recovers a causally-verified edge, whatever we
    call it. The atlas artifact and results_deeper.md both use "induction" and should be corrected.
18. **I got the induction-head criterion wrong first** by selecting the most-negative match — your
    XNOR point from the atlas review, which I failed to apply here. Behavioural ablation (guarded to
    reproduce L1H2 on tiny) is what I use now.
19. **Two of my random-repeat probes failed their positive control** (mix10 scored the same as dense).
    Only the third — matched to the training distribution — detects induction. Logged, not hidden.
20. **Gemma is hard-blocked:** `GatedRepoError: 401` on config.json, no HF_TOKEN configured. The last
    tier needs you to accept the license or supply a token.

## Questions for Logan (2026-07-09, tick 6) — corrections to my own tick-5 retraction
21. **My period-sweep probe was unsound below P=128** (`[u;u]` leaves positions q>2P−1 unpredictable).
    Under a sound *tiled* probe, mix10 scores **0.58 at P=64**, not chance as I reported. The
    retraction survives — P=96/85/150/180 are chance under the sound probe — but two of the numbers
    I showed you were wrong.
22. **"Positional copier" is falsified as a mechanism.** It predicts copying iff P|128; P=43 gives
    0.2144 and P=127 gives 0.0586. Attention analysis shows a *mixture*: at P=96 the head puts mass
    17.4 on the fixed key q−127 (wrong token, copying fails); at P=43 it puts 19.7 on the
    induction-target keys (copying partially works). **Mechanism unresolved.** I used |attn| and the
    bilinear scores are signed, so this bounds rather than pins it.
23. **NEW, and I think the real finding:** randomising the burst period (tiled, P~U[42,128]) means
    **no copying is learned at all** — 10% and 30% mixture, 30k steps, chance at every period — while
    the matched fixed-period control reaches 0.72 in 20k. The lever worked *because of* the period
    regularity. Whether content-based induction can form here at all is now an open question, and
    it's the one I'd most like your steer on: worth a long (200k-step) random-period run?
24. `--randperiod` added to lm_train.py (tiled bursts). Fixed path RNG untouched → old runs reproduce.

## Questions for Logan (2026-07-09, tick 7)
25. **I was wrong to say "mechanism unresolved" (tick 6).** The fixed-period copier IS positional; my
    prediction assumed a single attended offset. It attends **−127 and −128**, so it copies iff
    `P | 128` or `P | 129`. This predicts the *untrained* P=129 → observed 0.4013, and chance at
    96/85/126/63/44. 129 = 3·43 is exactly why P=43 copied and broke my naive rule.
26. **Content-based induction CAN form here** (positive control: 100% tiled random-period bursts →
    0.96–0.99 on trained periods, 0.88/0.92 at untrained P=32/16). So the 10%/30% failure was signal
    sparsity, not architectural capacity. I ran this before the long training run you'd have paid for.
27. **Mixture ladder is non-monotonic**: 0.5 gives strong content-based induction, 0.7 nearly none,
    1.0 near-perfect. Could be formation variance (your basin-competition result) or a bad seed.
    **I am not reporting a threshold** until seeds 1–2 land.
28. **We finally have a real induction circuit**: mix50-rp has redundant L1H0/L1H3 copy heads reading
    from required L0H1/L0H0. Verified at P=96 where positional copying scores chance. I propose
    promoting this model to Tier-1.5 ground truth — it is induction in the Elhage/Olsson sense,
    unlike attn2-seed0 (repeated-bigram) and mix10 (positional).
29. **RESOLVED — no mixture threshold exists.** mix 0.5 seed0 = 0.7483 (hit), seed1 = 0.0011 (miss),
    identical settings. Formation of content-based induction is **stochastic** at 30k steps, which is
    the basin-competition signature you found for the old lever, now shown for the real capability.
    So the ladder in tick 7 must not be read as a threshold, and I withdraw the framing. A proper
    study reports **formation rate over seeds**, not a copy score. Worth doing?

## Questions for Logan (2026-07-09, tick 8) — the Tier-1.5 gate is vacuous
30. **I ran the gate on the real induction circuit and it passed — then I controlled it and it is
    meaningless.** A *random unlearned dictionary* reproduces Δŝ to 0.0003 and passes with a LARGER
    margin (25.59×) than the learned one (8.81×). On mix50-rp, simply ranking heads by ‖write‖
    reproduces the causal ranking with no decomposition at all. **All Tier-1.5 gate PASSes are
    retracted as evidence about the method.** The causal circuits remain real.
31. **I built a gate that can fail (atom localization) and the method fails it.** Ablating atoms by
    contribution energy from L0H1's write: learned needs 32 atoms to halve P(copy), random needs 32,
    and the learned curve is slightly worse at every point. No localization advantage.
32. **My explanation for that was refuted by measurement.** I predicted a flat OV spectrum
    (isometry ⇒ nothing to find). In fact OV eff/rank is 0.10–0.43 while Pythia down_proj — where the
    method *does* beat PCA — is 0.80, i.e. flatter. Cause unresolved. Leading candidate: 128 atoms in
    128 dims against a ≤32-dim target means k=8 doesn't bind. Next test: 1024 atoms, k=2.
33. **What survives:** Pythia Tier 2, where the learned dictionary beats PCA-32 (0.6020 vs 0.3509) at
    matched L0 — a control that could have failed and didn't. That is currently the *only* evidence in
    this program that the decomposition does something a trivial baseline doesn't.

## Questions for Logan (2026-07-09, tick 9) — why the gate was vacuous
34. **Resolved, and it is a scope limit rather than a bug.** The learned-vs-random tie on OV maps has
    two parts. (i) Sparsity never bound: at k=8 the learned−random R² gap is +0.0026; at k=1 it is
    +0.2299. Pythia's k=32 against a 1024-dim target *does* bind (random 0.2587 vs learned 0.6020).
    (ii) More importantly, **there is no k that is sparse AND faithful AND localizing**: where the code
    is faithful the dictionary is irrelevant; where the dictionary matters the code destroys 40–70%
    of the behaviour.
35. **Strongest gate, causal-ranked atoms, faithful regime: learned ≡ random.** All 512 atoms live,
    top single-atom effect 1.4% of base. The induction write is "copy the embedding" — a dense map on
    a 32-dim subspace. **Its mechanism is distributed; there is nothing sparse to find.** The
    decomposition is not failing here, it is inapplicable.
36. **My flat-spectrum explanation was wrong** (I said so last tick, confirming here): OV eff/rank
    0.10–0.43 vs down_proj 0.80. The map's spectrum does **not** predict decomposability.
37. **Proposed spec addition — a decomposability pre-test.** Sweep k; record the learned−random R² gap
    and the faithfulness of the reconstructed output. If no k gives both, the map has no sparse
    mechanism. Copy-head OV fails it; Pythia down_proj passes. This would have saved the whole
    Tier-1.5 gate line. Do you want it in §2 as a gating step before any tier?

## Questions for Logan (2026-07-09, tick 10) — Pythia survives the audit
38. **I ran the Tier-1.5-killing controls against Pythia and it passes both.**
    - Decomposability pre-test: at k=8 the learned−random R² gap is **+0.4952** (on OV maps it was
      +0.0026), while a k=32 sparse code splices back into the model for only +0.049 CE.
    - Behavioural calibration: zeroing down_proj costs +0.2247 CE, so the learned code recovers
      **78.4%** of the layer's total causal contribution; a random dictionary recovers 60.0%.
    - Localization, causal-ranked: **one learned atom = +0.0112 CE vs +0.0009 for the best random
      atom (12×)**; ~10× after normalising by each dictionary's explained CE.
39. **Caveats I want on the record.** Destroying the whole MLP costs only 0.22 nats, so one atom is 5%
    of *the layer*, not the model. And effects saturate hard (top-1 atom +0.0112, top-128 +0.0239), so
    the code is redundant — "one atom = one mechanism" is **not** supported by this.
40. **The decomposability pre-test earns its place in the spec.** It separates the two regimes cleanly
    (OV: gap +0.0026, dense mechanism; down_proj: gap +0.4952, localizable) and would have saved the
    entire Tier-1.5 gate line. Recommend §2 gates every target on it before any tier work.
41. ~~Program status ... the method works on MLP down-projections and has nothing to say about
    attention OV maps of copy heads.~~ **RETRACTED next tick — see items 42-44.**

## Questions for Logan (2026-07-09, tick 11) — I retract item 41, and my pre-test is confounded
42. **"Attention ⇒ not decomposable" is FALSE.** I only ever tested the pre-test on the two maps whose
    answer I knew. On two new maps: pythia L3 `attention.dense` gap **+0.606**, GPT-2 L6 OV gap
    **+0.365** (vs pythia down_proj +0.703), all at k=8, K=1024. Attention maps have large gaps.
    Item 41's one-line summary is retracted.
43. **My pre-test is confounded by K/rank(W).** On attn2's OV map the gap grows from +0.0026 (K/rank
    16) to +0.0271 (K/rank 1). A map tested with K < rank would be rated "decomposable" for free.
    Comparisons must be at matched K/rank, and the gap alone is insufficient — it must be paired with
    behavioural faithfulness *and* a localization advantage.
44. **What still stands.** At matched K/rank=1, attn2 OV gap +0.027 vs pythia attention +0.606 — the
    maps really do differ, so that isn't overcompleteness. And attn2's atoms-to-halve is exactly K/4
    for learned *and* random at every K, which is as clean a "nothing is localized" signature as one
    could ask for. Pythia down_proj remains the only map passing all three criteria.
45. **Next experiment, and it decides the method's scope:** do pythia `attention.dense` and GPT-2 OV
    *localize*, or do they merely have large reconstruction gaps? If they localize, the method is not
    MLP-specific and Tier 1.5's negative was about a 128-dim rank-32 toy map, nothing more.

## Questions for Logan (2026-07-09, tick 12) — the localization criterion was worthless
46. **The `down_proj` 12× does not replicate.** It compared max-over-sampled-atoms ΔCE across two runs
    with different training subsets. Clean matched re-test (same seqs, same atoms, full distribution):
    learned mean ΔCE −0.00005, random +0.00030. **Random is marginally higher.** 12× retracted.
47. **Worse: the criterion has no power.** On the toy where `D_true` IS the generator, single-atom
    ablation gives true/random = **1.54× mean, 1.00× top1**; the conditional variant 1.57×. It cannot
    detect localization where localization provably exists. With codes fixed, dropping any used atom
    removes ~1/k of the reconstruction regardless of whether it is a real factor.
48. **So every localization claim I made is retracted as untestable**, including two I reported to you
    as findings: Tier 1.5's "the OV mechanism is distributed, nothing sparse to find", and
    `attention.dense`'s "reconstructs but localizes nothing". Neither was measured by a valid test.
49. **A criterion that works: irreplaceability** — drop the atom from the *dictionary* and let OMP
    re-select. On the toy: true/random = **8.01× mean, 4.24× top1** (vs 1.54×/1.00× for ablation).
    This is what I will run on the real maps next.
50. **What survives is narrower than I said:** the learned−random reconstruction gap and behavioural CE
    recovery are real and replicated. Whether any atom is a *mechanism* is **open**, not negative.

## Questions for Logan (2026-07-10, tick 13) — first valid answer on "are atoms mechanisms?"
51. **Irreplaceability validated, including its confound.** Matched-R² control on the toy: random dicts
    get *more* replaceable as k rises, so matching is conservative. True vs matched-random = **34× mean,
    21× top1** (single-atom ablation gave 1.54×/1.00× — no power).
52. **Real maps, matched-R²:** `down_proj` **1.84× mean / 5.18× top1**; `attention.dense` **1.46× /
    2.27×**. Both > 1× (real), both far below a true generator (34×).
    ⇒ **The learned atoms are much closer to an arbitrary basis than to generative factors.**
53. **The tail is the interesting part.** `down_proj`'s most irreplaceable atom costs 0.0079 R² — 20×
    its own mean. A minority of atoms behave like mechanisms; the bulk do not. `attention.dense` has
    almost no tail (2.27×).
54. **Calibration caveat I want to flag rather than bury:** the toy is an exactly-sparse generative
    model, an upper bound no real activation distribution can hit. So 1.84×-vs-34× is not "the method
    fails" — the defensible claim is only "> 1× against matched random, with a heavy tail".
55. **Next:** inspect down_proj's tail atoms. If the few highly-irreplaceable ones are interpretable,
    the method's value lives in the tail, not the bulk. That would be a usable, honest finding — and it
    is the first question in this program I can now ask with a criterion that has demonstrated power.

## Questions for Logan (2026-07-10, tick 14) — the tail atoms, with controls
56. **Irreplaceability weakly predicts token purity.** Over 96 probed `down_proj` atoms:
    Spearman **+0.231**; top-10 irreplaceable have mean purity 0.347 vs bottom-10's 0.120, against a
    *measured* chance level of 0.081. Some tail atoms are crisply monosemantic — two are pure `'\n'`
    detectors (purity 1.000, entropy 0), one is an article detector (`' a'/' an'`, 0.725).
57. **I inflated it first, then caught it.** My n=5 table read TAIL 0.600 vs BULK 0.150 — but the top-5
    happened to include the two purity-1.0 newline atoms. The correlation over all 96 is the honest
    number, and it is 4× smaller in effect. Quoting +0.231, not 4×.
58. **Consistency check the criterion could have failed.** Atoms 264 and 537 are both pure newline
    detectors. Near-duplicates would substitute for each other, so *neither* could be irreplaceable —
    that would have broken the metric. Measured cos = **−0.2475**. Not duplicates; two distinct
    directions that both fire on newlines. The criterion survives.
59. **Not monotone:** the single most irreplaceable atom (27× the dictionary mean) has purity 0.225.
    Irreplaceability and monosemanticity are correlated, not identical.
60. **Caveat I want stated in any writeup:** purity uses token identity only and ignores context, so it
    is blind to context-dependent features — i.e. to most of what an SAE would call a feature. The
    result is "irreplaceable atoms are modestly more token-pure", not "the method finds interpretable
    features".

## Questions for Logan (2026-07-10, tick 15) — the spec's central premise fails its first direct test
61. **Weight-aware does NOT beat an activation-only SAE.** Matched K=1024, k=16, same activations,
    held-out. R²(Wx): masked-projector **0.4919**, SAE decoder **0.4654**, random 0.0893. But the SAE
    *wins* on irreplaceability (top1 0.0093 vs 0.0061) and purity (0.385 vs 0.347) — and my SAE is
    weak (x-R² 0.273 after 3k steps), so the test is generous to the masked projector.
    ⇒ Optimising for `Wx` instead of `x` buys nothing measurable here. This is the premise of the spec.
62. **I retract tick 14's "the method's value is in the tail".** A 2×2 (dictionary × selection) shows
    purity tracks the SELECTION RULE: a *random* dictionary's top-irreplaceable atoms have purity
    0.347 — identical to the masked projector's — vs 0.160 for its unranked atoms. Ranking any
    dictionary by irreplaceability surfaces token-pure atoms, because a few token types (newlines,
    articles) sit in isolated activation directions. It is a fact about the data, not the method.
    My tick-14 comparison (learned tail vs unranked random) was apples-to-oranges.
63. **What still stands:** the objective is learnable (R²(Wx) 0.49 vs random 0.09); the closed-form
    theorem; toy recovery. Everything about atoms *being mechanisms* is now either weak or shown to be
    non-method-specific.
64. **Before I call this final** I want a seeded replication with a distribution-level test rather than
    max statistics — the irreplaceability top1 gap (0.0093 vs 0.0061) is exactly the kind of max-over-
    sample number that already burned me once (the retracted "12x"). Running that next.
65. **Honest bottom line for the spec, unless the replication overturns it:** the method learns a
    better-than-random basis for a map's action, is not better than a plain SAE at doing so, and its
    atoms are not more mechanism-like than an SAE's.

## Questions for Logan (2026-07-10, tick 16) — I was wrong last tick; the premise holds
66. **REVERSAL of item 61.** The seeded, distribution-level replication (3 seeds, identical probes,
    stronger 8k-step SAE) says the masked projector **wins**:
    - `R²(Wx)`: **0.4591 ± 0.0003** vs SAE **0.4291 ± 0.0042** (random 0.0887).
    - irreplaceability **median**: **0.000134** vs SAE **0.000016** — 8×. Mann-Whitney z = −8.88,
      P(SAE > MP) = 0.238.
67. **Why I got it backwards:** the SAE's loss distribution is heavy-tailed with a *low median* — its
    typical atom is more replaceable than a random direction, while a few are very irreplaceable. On
    one seed those outliers lifted its **mean** and **max** above the masked projector's. I used mean
    and max, on one seed. Across seeds the max swings 0.0026–0.0093.
    This is the **third** time a max-over-sample statistic has misled me here. I have stopped quoting
    them; medians + rank tests + seeds only.
68. **So the spec's premise is supported, modestly but robustly:** optimising for `Wx` yields a basis
    whose atoms are more uniformly load-bearing for the map's action than an SAE's.
69. **Unchanged:** purity tracks the *selection rule*, not the dictionary (tick 15's other finding —
    a random dictionary's top-irreplaceable atoms are as pure as the method's). Atoms-are-mechanisms
    is still weak in absolute terms (1.84× matched-random vs 34× for a known-true dictionary).
70. **Not to be cited yet:** the purity comparison (SAE 0.385 vs MP 0.347) was single-seed top-10 —
    exactly the statistic class that just failed. Replicating with seeds before it means anything.
71. **Tick 15's SECOND claim also reverses.** Seeded purity replication:
    - *Dictionary effect is real*: purity median MP **0.100** > SAE 0.050 = random 0.050 (chance 0.070);
      Mann-Whitney SAE vs MP z = −6.16, random vs MP z = −7.60. So MP atoms ARE more token-pure.
      Tick 15's "SAE 0.385 vs MP 0.347" was single-seed top-10 and inverts.
    - *Selection effect is NOT universal*: top-10-irreplaceable vs random-10 within each dictionary —
      MP 0.192/0.140 (1.4×), **SAE 0.477/0.110 (4.3×)**, random 0.205/0.152 (1.35×). My claim that a
      random dictionary's tail is as pure as the method's was one seed's noise.
72. **Coherent story from the two replications:** the `Wx` objective spreads the map's action across
    atoms (uniformly irreplaceable, uniformly mildly pure); the `x` objective concentrates structure in
    a minority (mostly replaceable + impure, but a crisp, very pure, very irreplaceable tail).
    Neither dominates — it depends whether you want a faithful basis for the map or a few clean features.
73. **Standing rule now enforced program-wide:** no single-seed / top-k / max-over-sample statistic may
    be reported as a finding. Medians, distributions, rank tests, ≥3 seeds. Three claims have been
    reversed by this rule so far ("12×", tick-15 irreplaceability, tick-15 purity).

## Questions for Logan (2026-07-10, tick 17) — the mechanism-likeness headline collapses; findings doc rewritten
74. **"Atoms are weakly mechanism-like (1.84× matched-random)" was a single-seed MEAN.** Re-run under the
    standing rule (3 seeds, medians, rank test): at **matched R²(Wx)** the masked-projector's atoms are
    **1.09× random, Mann-Whitney z = −1.00 — not significant.** At equal sparsity they win 4.71×
    (z = −12.05). The toy's *true* dictionary passes the matched-R² control at **34×**. MP does not pass it.
75. **The control has a confound I am not hiding:** matching R² forces random to k=96 (6× denser codes,
    6× per-atom usage), and irreplaceability grows with usage. Three views: equal-sparsity 4.71×,
    matched-R² 1.09× (n.s.), per-usage 6.6×. The toy's true dict scores 8.4× / 34× / 216× on the same
    three. **Under every control MP is 1–2 orders short of a generator.** Open problem: a control matched
    on both R² *and* usage would settle it.
76. **Unaffected by this** (same k, comparable R², so no k=96 confound): MP still beats the SAE on
    irreplaceability median (0.000134 vs 0.000016, z = −8.88) and purity median (0.100 vs 0.050,
    z = −6.16), 3 seeds. And the objective is learnable (R²(Wx) 0.459 vs random 0.089).
77. **`mechdecomp_findings.md` rewritten** against replicated evidence, with provenance on every number
    and 18 retractions tabulated. Bottom line for the spec: **the `Wx` objective is a better basis than
    random or an activation-only SAE — but it is not a feature finder.**
78. This is the fourth headline reversed by the no-single-seed rule. Every one of them was a positive
    result that I wanted to be true. The rule is doing real work; I'd keep it in the spec (amendment 5).

## Questions for Logan (2026-07-10, tick 18) — open problem #1 closed; §2.5 is now a clean negative
79. **The usage confound dissolves once you notice `usage = k/K` for ANY dictionary** (each datapoint
    picks exactly k atoms). So equal-k is already usage-matched, and the right scale-free statistic is
    **uniqueness = loss / (base_R²/K)** — the fraction of an atom's fair share of explained variance
    that is uniquely its own.
80. **It is not K/d-invariant** (random: 0.129 at K/d=4, 0.299 at K/d=0.25), so I re-validated it at the
    *real* proportions — K/d=0.25, rank/d_in=0.25, exactly Pythia's 1024×4096: **TRUE generator 0.9558
    vs random 0.1961 (4.88×)**. The criterion has power where it is used.
81. **Verdict, 3 seeds, medians, rank test:** masked-projector uniqueness **0.2990** vs random k=16
    **0.3280** (z = +0.95, **n.s.**) and random k=96 matched-R² **0.2717** (z = −1.12, **n.s.**).
    **The learned atoms have exactly the uniqueness of a random basis.** MP's raw 4.71× irreplaceability
    advantage is fully explained by its 5.2× larger base R². SAE is *more* redundant than random (0.0448).
82. **§2.5 upgraded from "not demonstrably mechanisms" to a clean negative**, with usage, R², and K/d all
    controlled and the metric's power demonstrated at the exact experimental setting. This is the
    strongest evidential footing anything in this program has had — and it is a negative result.
83. **The one-line verdict for the spec is unchanged but now well-founded:** the `Wx` objective yields a
    better *basis* than random or an activation-only SAE; it does **not** carve a map's action into
    independent mechanisms.

## Questions for Logan (2026-07-10, tick 19) — I have to correct last tick's negative
84. **"MP atoms have the uniqueness of a random basis" is retracted.** I measured it at **K/d = 0.25**
    (undercomplete — the regime where an overcomplete feature code *cannot exist*) with only 12.5
    firings per atom. Redone at N_EVAL=4000 with probes restricted to atoms firing ≥5 times:
    K/d 0.25 → 1.16× (n.s.); K/d 0.50 → 1.53× (marginal); **K/d 1.00 → 1.79×, z = −3.33, significant.**
85. **A measurement failure I nearly reported as a finding.** `usage = k/K`, so at K=8192 with 600 eval
    points each atom fires **1.2 times** and most probed atoms are never selected — loss exactly zero.
    That run gave medians of 0.0000 and a "ratio 0.04×". It measured dead atoms. Void.
86. **Corrected verdict:** the `Wx` objective yields atoms **modestly but significantly more
    irreplaceable than random once the dictionary is at least critically complete** — and still less
    than half as unique as a known generator (1.79× vs 4.06× at the same K/d). Below critical
    completeness the effect vanishes.
87. **The standing rule needed an addition:** ≥3 seeds + medians + rank tests is not enough. With sparse
    codes the eval set must scale with K, or per-atom medians are computed over atoms that never fire.
    Always report firings/atom and dead-probe fraction.
88. **Fifth reversal in this program, and the first of a NEGATIVE result.** Worth saying plainly: the
    controls are not biased toward pessimism. They are biased toward whatever the data says — which is
    the only property I actually want from them.

## 2026-07-10 — DIRECTION CHANGE (Logan): mechdecomp paused, Jacobian clustering started
89. **Your trace question: yes, and it's the identical number.** `<A,B>_F = tr(A^T B)`, so Frobenius
    cosine `tr(A^T B)/(||A||_F||B||_F)` *is* cosine on the flattened matrices (verified to 12 dp).
    The writeup already means this. The real content of the lemma is that for one bilinear layer you
    never build `J`: `tr(J_i^T J_j) = x_i^T G x_j` exactly (verified to 1e-14, plus gauge/Euler/autodiff).
90. **P2 is FALSE as stated, and DGP-A is why.** The exact Jacobian of the hand-coded layer is
    `J = [A_g | (1/eps)·F(c)]`. The gate-column block `∂y/∂s` is (a) `O(1/eps)` — 56.7× the content
    block at eps=0.1 — and (b) **bit-identical across gates** for fixed content (diff 0.00e+00), since
    it depends on `c` alone. So the full Jacobian recovers **content** (ARI 1.000) and the gate at
    **chance** (−0.002). Restricting `J` to content columns recovers the gate at ARI 1.000.
91. **The eps sweep has no dissociation window.** Full-J sees the gate only at eps=10, where
    `||J_gate||/||J_cont||` finally drops below 1 — and at that eps the raw input sees the gate too.
92. **So a choice is needed, and it is yours:** (a) redesign the DGP so the gate is not an input
    coordinate (multiplicative gate / separate stream ⇒ no `∂y/∂s` block), which is the honest test of
    P2; or (b) keep the content-restricted Jacobian, but then state that choosing the input subspace
    already encodes the structure we claim to discover. I've queued (a) as next tick's first task.
93. Minor: the writeup assumes random orthogonal experts are near-Frobenius-orthogonal. Max off-diag
    expert cosine is **0.474 at d_c=6**, 0.097 at d_c=16. Use d_c ≥ 16 or enforce exactly.
94. **DGP-A′ survives** — the restricted-Jacobian embedding morphs with the expert family: orthogonal
    → spikes, continuous rotations → ring, hierarchical → nested blocks. That's the centrepiece figure
    and it's in the HTML I sent.
95. Cron `96d16410` runs every 2h with the mechdecomp standing rules carried over (≥5 seeds, measured
    chance ARI, matched-dimension random-projection control, no single-seed/top-k statistics).
96. **Your mechdecomp cron was still firing on the paused program.** Retired it (`1b6ea46c`); the
    jacclust cron (`96d16410`, every 2h) is the live one.
97. **The DGP-A fix you'd want isn't available.** "Make the gate multiplicative so no `∂y/∂s` block
    exists" cannot work while the gate is *any* coordinate of `x`: differentiating it yields
    `Σ_g A_g c w_gᵀ`, summed over all experts, hence gate-independent. Euler (`J(x)x = 2y`) says `J`
    always carries the output as a rank-1 shadow, and the gate derivative is what carries it.
98. **I predicted J would be dominated by that shadow. Wrong.** On the trained bilinear MLP the Euler
    shadow is only **3.7%** of `||J||²_F` (median 0.0366). So `J` does carry real information beyond
    `cos_x·cos_y`. Kernel exact on trained weights (1.2e-15). P3 identity reproduced at 1.0000.
99. **The real obstacle is different: on this model `corr(cos_J, cos_x) = +0.944`.** `G`'s spectrum is
    nearly flat (effective rank 118.5/128), so `xᵀGx' ≈ const·x·x'`. Its top eigenspace is nearly
    *orthogonal* to the data PCA (cos 0.04–0.31), so it isn't whitening — it just isn't anisotropic.
100. **Controlled, and it's layer-dependent.** Against `G_rand` (same eigenvalues, random
    eigenvectors), at MLP#0 the true `G` agrees with raw-cosine clustering exactly as much as the
    control (0.858 vs 0.872 — eigenvectors carry nothing); at MLP#1 it departs far more than the
    control (0.464 vs 0.789 — weight structure does real work). Your "effective rank of G across
    depth" summary statistic looks worth computing: 118.5 vs 110.5 already tracks this difference.
101. **Necessary ≠ sufficient**, and this is the crux for the agenda: "G-clusters ≠ x-clusters" is not
    "G-clusters recover mechanism", and real models have no ground-truth mechanism labels. DGP-C
    (two-layer, joint `(g₁,g₂)`) is the only design in the writeup that can bridge that, so it's next.

## 2026-07-10 — jacclust tick 3
102. **P2's failure is a theorem, not a quirk of DGP-A.** For any single bilinear layer with the gate
     a linear readout `p_g·c`: `J = Σ_g A_g c p_gᵀ + Σ_g (p_g·c) A_g`. The first term sums over *all*
     experts (gate-independent, norm ≈ √k_g‖c‖‖A‖); the signal is ≤ ‖c‖‖A‖. The gate-independent term
     always dominates. No "multiplicative gate" redesign escapes it. So the content-restricted
     Jacobian is the object — and in DGP-C that restriction is architectural, not a peek at the answer.
103. **DGP-C confirms the compositionality claim, with controls.** 5 DGP seeds × 5 k-means seeds,
     chance ARI measured at ≈0.000. J₁ restricted → g₁ 1.000 (g₂ at chance); J₂ → g₂ 1.000; **end-to-end
     J → joint (g₁,g₂) at 1.000**, content at chance. Input and a **matched-dimension random projection**
     both recover content (1.000) and gates at chance — the control could have failed and didn't.
104. **Sketched VJP: your k≈10–20 is right for the kernel, but clustering needs k=1.** Kernel
     correlation 0.83 (k=1) → 0.973 (k=10) → 0.993 (k=50), while ARI(joint) is 1.000 at every k.
     Quote the kernel error when justifying probe counts; ARI saturates and hides the cost.
105. **I got DGP-B wrong first.** I negated content only and claimed `J` is odd — residual 1.46, because
     the restricted `J₁ = A_{g₁}` is *constant* in `c`. The right statement: bilinear layers are degree-2
     homogeneous, so negating the **whole** input gives `f(−x)=f(x)` and `J(−x)=−J(x)` (both exact to 0).
     Then `cos` on J gives ARI 0.175 and `|cos|` gives **1.000**. Fix `|cos|` globally for all methods,
     as you specify — applied to only one method it is a confound.
106. Next: P4 (principal angles, G's top eigenspace vs the gate/content subspaces), then DGP-A′
     geometry recovery metrics (ring: correlation of recovered angle with θ, not ARI), then the
     real-model question that actually matters — with no ground-truth mechanism labels on a trained
     model, what would even count as validation? DGP-D (train on a task with known modules) is the
     writeup's answer and I think it is the right one.

## 2026-07-10 — jacclust tick 4
107. **P4 is sharper than your prediction.** The top `k_g` eigenvectors of `G` are the gate subspace
     **exactly** (principal-angle cos = 1.0000 at every eps), with content strictly below. Not "gate ⊕
     content directions" — gate first, content after. Same fact as P2's failure, read off the spectrum.
108. **DGP-D succeeds, on modules that TRAINING found.** Bilinear MLP trained from scratch on
     `y = A_g c` (val rel. error 0.003–0.005, 5 seeds). Content-restricted `J` recovers the mechanism at
     **ARI 1.000±0.000**; input 0.002, hidden `Lx⊙Rx` 0.002, output 0.214, rank-1 M 0.331, matched-dim
     random projection 0.002. Chance measured at ≈0.000. This is the "DGP-D success" criterion in your
     §7 — mechanism structure invisible to activation clustering at either end.
109. **The impossibility theorem survives training.** The *trained* net's full `J` recovers content at
     1.000 and the gate at chance, exactly like the hand-coded layer. Gate-derivative domination is a
     property of the bilinear form, not of our construction.
110. **The gap that remains is the whole agenda.** DGP-D's content/gate split is supplied by the task's
     input layout. A real LM has no such split, and tick 2 showed the full-`J` kernel there is nearly
     plain input cosine (`corr = +0.944`, G eff. rank 118.5/128) at one MLP and meaningfully different
     at the other. So: **the method provably reads mechanism when a content/control split exists, and
     we do not yet know what it reads when one doesn't.**
111. Suggestion for where this goes. The honest paper-shaped claims available now are (a) the exact
     kernel + the collapse lemma, (b) DGP-D, and (c) the *negative* structural result — for a single
     bilinear layer, the gate-derivative block always dominates, so the full Jacobian cannot separate
     mechanisms that are selected by the input. (c) is novel and cheap to state, and it predicts
     exactly the real-model behaviour we saw. Worth its own section rather than a footnote.

## 2026-07-10 — jacclust tick 5 (all four listed priorities now complete)
112. **DGP-E answers tick 4's open question.** No control stream: the gate is a *quadratic* feature
     `(w_g·c)²` computed by layer 1 from the content itself. Then the tick-3 impossibility theorem no
     longer applies — same content, different gate gives `cos_F(J_a,J_b) = 0.043` (DGP-A: identical).
     **Depth + a nonlinear gate restores mechanism signal to the FULL Jacobian.**
113. **But only partially, and I can say exactly why.** Column-block decomposition: the operator block
     `J[:, v-cols]` recovers `g` at **1.000**; the gate-derivative block is **6.1× larger in norm** and
     only a 0.355 signal, diluting the full `J` to 0.389. `layer-2 J` alone is at chance on the gate —
     its own gate-derivative block is gate-independent, so the single-layer theorem reasserts layer-wise.
114. **A real dissociation window exists** (unlike DGP-A, where it was empty): for gate_amp ≲ 6 the full
     `J` reads mechanism at ≈0.45 while input cosine is at −0.003 and content at 1.000. Caveat worth
     stating: rank-1 `M` already gets 0.25–0.38, so `J`'s margin over the cheap baseline is ≈0.19.
115. **A new, weights-only recipe, derived from your P4.** Since `G`'s top eigenspace *is* the gate
     subspace, project `J`'s columns off the top-r eigenvectors of the layer's own `G`. No labels, no
     data, no supplied split. Result **0.654±0.176** vs FULL 0.389, matched-dim random subspace 0.388,
     `G`-bottom 0.348, oracle 1.000. It beats both controls. I think this belongs in the writeup: §2.2
     notes "top eigenspace of G = where the layer is most input-dependent" as a *diagnostic* — it is
     also a **filter**.
116. All four cron priorities are complete (P2 impossibility, DGP-B, DGP-C, P4). I've rewritten the
     cron's priority list: (1) does the G-top projection help on the real bilinear MLPs from tick 2,
     where `corr(cos_J, cos_x) = 0.944`? (2) DGP-A′ ring/hierarchy with manifold metrics, not ARI;
     (3) scale to the 500M tensor-transformer, one layer, with the projection recipe.

## 2026-07-10 — jacclust tick 6
117. **A label-free validation for real models.** If Jacobian clusters group "same map applied" points,
     one linear map per cluster should predict held-out outputs. Cluster on train, fit `y≈A_c x`, assign
     held-out by nearest centroid, score held-out R². Controls that could win: raw `x`, spectrum-matched
     `G_rand`, random clusters, single global map.
118. **My `G_P` derivation was wrong and the identity test caught it.** `J(x)P = D[diag(Lx)RP +
     diag(Rx)LP]` keeps the gates `Lx,Rx`; `gram(D,LP,RP)` projects them too. Correct form puts `P`
     inside the middle Grams. Verified 1e-13, with `r=0` reproducing plain `G`.
119. **My first surrogate harness was broken and I'm reporting it.** Unregularised per-cluster least
     squares gave R² = −319 (raw cosine) and −139 (G) while **random clusters scored −0.78** — random
     beat everything. It measured fit instability. Ridge + a `k=1`-must-equal-global identity check fixed it.
120. **Your §9 "parked" open question is actually load-bearing.** `‖J(x)‖²_F = xᵀGx` exactly, so the gain
     is free from the kernel. Cosine throws it away, and on real MLPs that is the difference between
     **−0.25 and +0.22** surrogate R² (MLP#0, k=8). Gain-only is a failing control, so it's direction
     *and* gain. **§8's "decide cos vs |cos|" should be "decide cosine vs Euclidean" — keep the gain.**
121. **First real-model result that survives a control that could have won.** Under a fair rule
     (Euclidean, gain kept, 5 seeds), the G-metric beats raw `x` and spectrum-matched `G_rand` in **5 of
     6 (layer,k) cells**, margin growing with k (+0.09 at k=32). ⚠ The exception: MLP#0 k=16, where
     `G_rand` wins (0.3425 vs 0.3107). Reported, not dropped.
122. `G_P` (tick-5's weights-only decontamination) ≈ `G` on real MLPs — no gain, unlike DGP-E. Consistent
     with P4: a real layer has no clean gate subspace for the top eigenvectors to occupy.
123. Still open, and still the crux: these clusters are better "same-map" clusters, but nothing shows
     they are *interpretable* roles. The intervention test (patch within- vs across-cluster) is next.

## 2026-07-10 — jacclust tick 7 (open problem 4: intervention validation)
124. **Built and validated the intervention test.** Replace the MLP output with a per-cluster ridge
     surrogate `A_c x`: within-cluster vs across-cluster vs one global map. Harness identities exact:
     `clean` mode reproduces the unmodified CE (0.00e+00) and `k=1` within == global (0.00e+00).
     Random-cluster control gives **exactly zero differential** (−0.002…+0.003), which is the null it
     must give. The instrument works.
125. **But the differential — the statistic your §7 asks for — is saturated and I won't quote it.**
     Uniform CE is ln(5120) = 8.54 nats; across-cluster CE is 8.5–13.9, i.e. at or beyond uniform. The
     model is confidently wrong there, so metric-to-metric differences measure how badly a wrong linear
     map fails, not mechanism. **Report within-cluster CE instead** (equivalently tick 6's surrogate R²).
126. **By within-CE, the Jacobian metric's real-model advantage is modest and non-uniform.** It wins 2 of
     4 cells clearly (MLP#0 k=8: 7.785 vs raw 8.407, G_rand 8.052; MLP#1 k=16: 5.125 vs 5.189, 5.215),
     ties one (MLP#1 k=8 — a 0.0014-nat margin is nothing), and **loses MLP#0 k=16 to the
     spectrum-matched control** (7.286 vs 7.428).
127. **The losing cell is the same (layer, k) that lost in tick 6.** Two independent tests — one offline,
     one causal — agreeing on *where the method fails* is stronger evidence than either agreeing on where
     it wins. I'd trust that more than any single headline.
128. Replacing the whole MLP with one linear map costs +5.99 nats (MLP#0) / +1.47 (MLP#1), so there is
     real dynamic range; the within-CE differences (0.06–0.62 nats) sit well above seed noise (sd ≤0.18).
129. **Still not established:** that G-clusters are recognisable computational roles. Both tests show only
     that they are better *same-map* clusters. That is weaker than §6's Phase-1 goal, and I think the
     honest framing for a writeup is exactly that distinction.

## 2026-07-10 — jacclust tick 8 (scale to the real tensor-transformer)
130. **Loaded the real 500M-style bilinear transformer** (`Elriggs/gpt2-bilinear-12l-6h-768embd`, the
     §6 preferred target: bilinear MLP, softmax attn). My hand-rolled forward gave CE 8.45 (broken
     rotary) — caught by the CE check, no bad activations used. Extracted the actual modded-nanogpt
     classes; loads with zero missing keys, CE 3.385. Faithful.
131. **`G` is nearly full-rank at EVERY depth (eff rank 677–718 / 768).** corr(cos_J, cos_x) = 0.62–0.80
     across all 12 layers. A near-isotropic G ⇒ the Jacobian metric ≈ input cosine. This is a
     weights-only diagnostic and it predicts the method won't help here.
132. **The real-model advantage does NOT survive to scale.** Surrogate R² (k=16, 5 seeds): G ties raw x
     and the spectrum-matched G_rand at every layer, AND — the sharper negative — **a single global
     linear map beats every clustering** (+0.45…+0.51 vs all clusterings negative). Clustering this
     layer into locally-linear pieces makes held-out prediction *worse*.
133. **Reinterpreting ticks 6–7.** The modest MLP#1 signal on the 1.9M toy is best read as a
     small-model / low-eff-rank artifact, not a scaling property. The standing rules did their job: the
     one positive real-model result was on a single 128-dim layer and did not replicate at 768-dim × 12L.
134. **Honest program status for a writeup:** the exact kernel + collapse lemma + P2 impossibility
     theorem + DGP-A..E are solid, novel, publishable as *method and theory*. The real-model
     application is a **negative** at scale — G is too isotropic for the reweighting to matter. I'd
     write it as: "the method reads mechanism whenever the layer's operator is anisotropic over the
     data (toys, by construction); trained bilinear MLPs at scale are close to isotropic, so it reduces
     to input-space clustering there." Effective rank of G is the cheap weights-only test for when to
     even bother.
135. Question for you: is there a bilinear model / layer you *expect* to be anisotropic (a
     task-specialised or smaller-vocab model, or the bilinear-ATTENTION variant where per-position maps
     may be sharper)? That is where the method would actually earn its keep, and I'd target it next.

## 2026-07-10 — jacclust tick 9 (I retract tick 8's negative)
136. **Tick 8's scale-negative was my own harness artifact — retracted.** "Global map beats clustering"
     and "G ties raw x at scale" both came from per-cluster ridge overfitting: 768 features, 500
     pts/cluster, λ=0.1. The SAME fit-instability I caught at 128-dim in tick 6 and failed to re-check
     when d went 128→768.
137. **Audit (clustering fixed, sweep data+λ):** at 3000 pts/cluster or λ≥3, per-cluster clustering
     BEATS the global map by up to +0.13, at every setting. So the layer *is* better modelled
     piecewise-linearly; "clustering hurts" was false.
138. **Corrected real-model result (48k train, λ=3, k=16, 5 seeds):** clustering beats global everywhere
     (+0.10–0.12); the G-metric beats BOTH raw x and spectrum-matched G_rand at layers 2 & 6 by ~2–3σ
     (0.6150 vs 0.6094/0.6103; 0.6071 vs 0.6011/0.5992), ties at layer 10. **A weak but real positive
     that survives to 500M scale** — the opposite of what I told you last tick.
139. **Magnitude, honest:** the margin is ~0.006 R², not the toys' dissociation. The G eff-rank story
     (~690/768, still true) explains why it's small — G is nearly isotropic, so little reweighting is
     available. Small, real, and predicted by a weights-only quantity, all at once.
140. **Rule added:** per-cluster surrogate needs pts/cluster ≫ d_in; re-verify the fit harness whenever
     the dimension changes. I violated my own "check the harness with a known point" rule by not
     re-checking it across a 6× dimension jump.
141. Net for a writeup, unchanged in shape from tick 8 but corrected in sign: method+theory+toys solid;
     real-model effect is **small-but-real and control-beating at early/mid layers**, its size predicted
     by G's effective rank. Still want your steer on an anisotropic target (bilinear-attention variant?)
     where the effect might be large rather than marginal.

## 2026-07-10 — jacclust tick 10 (real-model regime, characterized)
142. **G is near-isotropic on every trained bilinear MLP measured** — eff rank/d = 0.86–0.94 across 14
     layers of 2 independent models. This is weights-only and uniform. The method's leverage is
     1−(eff rank/d), so it is small everywhere, predictably.
143. **G-clusters ≈ x-clusters on the 768 model** — ARI(G,x) = 0.50–0.61, vs the k-means self-consistency
     floor 0.60–0.68. The difference is about a seed's worth of noise.
144. **But it's a PRINCIPLED nudge, not nothing.** G stays *closer* to x than the spectrum-matched G_rand
     does (0.597 vs 0.564), yet clusters slightly better (tick 9). So the Wx-aware metric is a small
     weight-informed perturbation of cosine — which is exactly why it beats a random-spectrum control by
     2–3σ while barely moving the assignment. Ticks 9 and 10 reconcile precisely.
145. **Real-model verdict, complete and honest:** small but real, control-beating, effect size predicted
     by G's effective rank (~0.90 → small). Not degenerate with cosine, but close to it, for a reason
     you can read off the weights before touching data.
146. **The pre-screen that falls out:** compute G eff rank from weights; only bother clustering where it
     is far from full. Every trained bilinear MLP I've measured is ~0.90 (near-full).
147. **Sharpened anisotropy question for you:** effect ∝ 1 − eff_rank(G)/d. Where is G anisotropic?
     Candidates: bilinear ATTENTION per-position maps (Phase 3), small-vocab/task-specialised bilinear
     models. If you point me at one you expect to be sharp, that is the decisive "does this method ever
     matter a lot" test. Otherwise the honest writeup is: exact lemma + toys + theory, plus a real-model
     effect that is small-by-a-measurable-weights-property.

## 2026-07-10 — jacclust tick 11 (DGP-A' ring, priority 2 done)
148. **DGP-A' ring quantified with manifold metrics.** Continuous rotation family A_θ=R(θ): the
     restricted-Jacobian 2-D embedding recovers the mechanism circle at |circular corr| **0.999±0.000**,
     trustworthiness **1.000**, while input x (0.083), output y (0.028), and a matched-dim random
     projection (0.069) are all at chance. Construction verified to 0.00e+00 first.
149. **The toy side is now complete and rigorous:** discrete experts→clusters (ARI 1.0), rotation
     family→circle (circ corr 0.999), hierarchy→nested blocks. Each verified with the metric matched to
     its geometry — ARI only where clusters exist, circular-correlation + trustworthiness for the
     manifold. This is the paper's centerpiece and it's solid.
150. **Program status, whole:** exact kernel + P2 impossibility theorem + DGP-A..E + the ring/geometry
     result are all solid and publishable as method + theory + toys. The real-model application is
     small-but-real (tick 9), degenerate-ish with cosine because G is near-isotropic on every trained
     bilinear MLP (tick 10, eff rank 0.90). G eff rank is the weights-only pre-screen.
151. **The one open lever is still anisotropy** (bilinear-attention per-position maps, or a
     task-specialised/small-vocab model). Everything else on the priority list is done. Awaiting your
     steer on whether to build the Phase-3 bilinear-attention machinery or whether the current
     characterization is the intended stopping point.

## 2026-07-11 — jacclust tick 12 (the organizing law, tested)
152. **Tested (not asserted) the tick-10 organizing claim.** Controlled knob: rank r of the L,R row
     subspace sets eff_rank(G). Sweeping r=2→48 moves eff_rank(G)/d from 0.04→0.92; the G-metric's
     surrogate advantage over cosine falls 0.886→0.068 monotonically. **corr(1−eff_rank/d, advantage) =
     +0.913 (vs cosine), +0.911 (vs G_rand).** It's a law, and it could have failed.
153. **Controls behave:** raw cosine flat at ~+0.04 across the whole range (blind to the operator);
     G_rand tracks cosine (random eigenvectors). So the advantage is the anisotropic eigen-STRUCTURE of G
     specifically, not its spectrum or dimension.
154. **This locates the real models exactly.** Trained bilinear MLPs are at eff_rank(G)/d ≈ 0.90 = the
     r=48 row, advantage ≈ +0.068 — matching the +0.006–0.07 measured on the 768 model. The real-model
     effect is small because it's the law evaluated where LM MLPs sit, not because the method fails.
155. **Program is complete and internally consistent.** Kernel+lemma, P2 theorem, toys (clusters/ring/
     hierarchy), DGP-E depth + G-top projection, real-model characterization, and now a weights-only law
     predicting effect size. Every real-model number is on the curve. The pre-screen (eff_rank(G)) has a
     quantified payoff curve behind it.
156. **The only thing that would add a large real-model result** is finding/So training a bilinear layer
     with anisotropic G. LM MLPs don't have it. The bilinear-ATTENTION per-position map is the remaining
     untested candidate (Phase 3 machinery). Still awaiting your steer, Logan; absent it, tick 12 is a
     natural stopping point — the method is fully characterized and its real-model payoff is predicted by
     a law, which is itself the publishable result.

## 2026-07-11 — jacclust tick 13 (I overclaimed in tick 12; correcting)
157. **Tick 12's "real models sit on the law" is retracted.** My own numbers were 10x off (+0.068
     predicted vs +0.006 measured) and I glossed it. Swap-one-thing test on real G (eff_rank/d 0.893):
     isotropic Gaussian input +0.066 (= the law), real activations **+0.0012**, and vs the spectrum-
     matched control real activations are **−0.0012** (negative). The law is ISOTROPIC-INPUT-only.
158. **eff_rank(G) is necessary but NOT sufficient.** Effect needs G anisotropic AND data populating G's
     anisotropic directions. Matched-covariance Gaussian keeps +0.037; the collapse to +0.001 comes from
     the NON-Gaussian structure of real activations (heavy tails, token clusters), not from covariance.
159. **The real-model positive (ticks 6–10) was over-read.** On a like-for-like isolation (same G, swap
     only input), the real-data G-vs-G_rand advantage is essentially zero/negative. The honest statement:
     on real bilinear-MLP activations the Jacobian metric ≈ cosine, advantage within noise of zero.
160. **Corrected program bottom line:** kernel + P2 theorem + toys + geometry are solid; the organizing
     law is real but conditional on input isotropy; the real-model application is a clean negative
     (weight metric ≈ input cosine on real data). That is the publishable, honest result — a fully
     characterized method with a precise, tested account of when it works (anisotropic operator +
     anisotropy-populating data) and a demonstration that trained LM MLPs fail the second condition.
161. This is the natural, honest stopping point. Four ticks of "awaiting steer on bilinear attention"
     with no reply; the science is complete and the real-model verdict is now correct rather than
     optimistic. If you want the large-effect regime, it needs a model whose activations populate an
     anisotropic G — not something I can manufacture without a target from you.

## 2026-07-11 — jacclust tick 14 (Logan live Q&A, recorded)
162. **Decoder/linear-transformation (rank-1 M) clustering:** beats activations on the TOY (0.355 vs
     0.002, captures the input×output joint that reveals the gate) but LOSES to raw x on the real model
     (M 0.49/0.52/0.42 vs x 0.50/0.53/0.53). Output-similarity is the wrong grouping for a same-map
     surrogate; no gate → no joint structure to exploit. Consistent with "nothing beats cosine on real
     bilinear-MLP activations".
163. **Outlier fold-out (Logan's LLM.int8 analogy):** flips G-vs-control from −0.0028 to +0.0066 (right
     mechanism) but tiny — the 124M model is post-RMSNorm (no norm-outlier tokens; dim outliers only
     1.6–1.8× median). Available bilinear models are all normed, so no dramatic outliers to fold. Idea
     sound, blocked by model availability.
164. **Impossibility result recorded precisely** (symmetry irrelevant / tensor≡CP / novelty honest) in
     results.md for the eventual writeup.
165. **Consolidated real-model picture across all of Logan's levers:** G-metric ≈ cosine, rank-1 M <
     cosine, outlier-fold-out marginal. Every lever works on gated/structured toys and collapses on
     real activations. The real-model application is a well-characterized negative; the toys + kernel +
     P2 theorem + geometry recovery are the solid deliverables.

## 2026-07-11 — jacclust tick 15 (joint weight×data anisotropy sweep, Logan's request)
166. **Built the 2D sweep Logan asked for.** Weight anisotropy (rank r → eff_rank G) × data alignment
     (G-top/random/G-bottom/isotropic). The method's advantage IS correlated with anisotropy, and the
     best predictor is DATA-AWARE: 1 − corr(cos_J,cos_x), ρ=+0.83; weights-only 1−eff_rank/d is ρ=+0.76.
167. **Three findings:** (a) weight anisotropy necessary not sufficient (isotropic G → +0.03-0.07 for any
     data); (b) given anisotropic G, data swings advantage 2.4× (+0.32→+0.76); (c) my naive "aligned data
     → big effect" was WRONG — ISOTROPIC data gives the max (+0.76), because the data must EXPOSE G's
     eigenvalue spread, not sit inside a locally-flat top/bottom region.
168. **This explains the real-model null exactly:** real activations have corr(cos_J,cos_x)=0.62-0.94 →
     G≈cosine on that data → small advantage. Real data sits where G is flat. Unifies ticks 12-13.
169. **Complete honest deliverable:** the method beats cosine iff anisotropic operator AND data populating
     its spread; predictor = 1−corr(cos_J,cos_x) on real data (cheap, ρ=0.83). Recorded truth-determination
     limits for real LLMs (surrogate + intervention proxies verify same-map clusters, not meaning; ground
     truth only in toys).

## 2026-07-11 — jacclust tick 19 (synthesis)
170. Wrote jacclust/SUMMARY.md — authoritative synthesis of the whole program. 14 solid results (S1-S14)
     with provenance; the MLP negative + attention positive characterized; 7 retractions tabulated with
     cause; the weights-only pre-screen recipe; 4 open questions. This is the reference doc; results.md
     remains the chronological log.
171. Held off launching another attention sweep autonomously: I have an open question to Logan (is the
     modest head-dependent control-beating positive enough, or does he want full characterization?) and
     I just corrected an over-read (tick 17->18). Consolidating was the responsible move rather than
     risking a third premature claim.
172. Honest program state: method+theory+toys SOLID; MLP real-model = characterized null (data-driven,
     predicted by corr(cos_J,cos_x)); attention real-model = modest control-beating positive, needs a
     non-derived causal target to be publishable (open #1,#4). Awaiting Logan's steer on direction.

## 2026-07-11 — jacclust tick 23 (attention resolved)
173. **The attention Jacobian is causally near-useless; the QUERY readout Wq·x is the causally-meaningful
     object.** Causal switch test, 2 causally-important heads (screened), 5 seeds: query 0.009-0.010 vs
     J 0.001-0.002 vs random 0. Resolves Logan's "are clusters meaningful + causally important" question:
     the JACOBIAN clusters aren't, the QUERY clusters are.
174. Third single-seed over-read caught by the 5-seed rule (query 0.0194 single -> 0.0090 mean). Direction
     robust, magnitude was inflated.
175. Program theme confirmed: the useful object is a weight-derived READOUT that isolates the operation
     (content-restricted J for MLPs, Wq·x for attention); the full Jacobian is contaminated both times.
     Net: Jacobian-clustering has no real-model causal win, but the method is a useful LENS that points to
     the right weight-derived readout.

## 2026-07-13 — basis_aligned tick 1 (new program, Logan's spec: fold E/U into the bilinear layer)
176. **New folder `basis_aligned/`** — weight sparsity vs functional sparsity when a bilinear layer sits
     between embed/unembed. Folded objects: L̃=LE, R̃=RE, D̃=UD; invariant per-class form B_c. e1 hand-coded
     demo: inserted rotation keeps function+folded weights bit-identical, unfolded 87.5%→0% zeros.
177. **e2 (block task, 3 seeds, Logan's iterated L1→prune→revert→finetune protocol):** (a) dense training
     leaves 84-87% of |B| mass on never-probed cross-block entries (OOD FVU 2-5) — the backdoor moral in
     the minimal setting; (b) sparsifying L,R,D ONLY is cosmetic: 94% unfolded zeros, junk unchanged —
     L1 shoves the rotation into free E,U; (c) sparsifying through E,U is functional (junk 0.85→0.45,
     near-minimal weight count) but does NOT recover ground truth from a junky scratch solution;
     (d) from rotated hand-coded init the protocol EXACTLY undoes the rotation (block 1.000, junk 0,
     stops at the hand-coded weight count).
178. **e3 (squares-in-superposition): honest NEGATIVE with theory.** Linear readout of d_h hidden functions
     ⇒ MSE ≥ tail of target Gram spectrum = (m−d_h)(p/5−p²/9)/m — NO computation-in-superposition gain is
     possible for pure bilinear+linear readout under MSE (Vaintrob/Mendel gains need a nonlinear readout).
     Closed form verified vs 2M-sample MC (<0.7%); trained nets land ON the bound (+0.3-1.2%), computed
     features = d_h exactly. Sparsifying trades the last ~2% (shared-mean) for a LITERAL dedicated circuit:
     13 units, one input each, 1.3% of weights. Thread 3 (real LLM embedding structure) awaits Logan's spec.

## 2026-07-13 — basis_aligned tick 2 (Logan's challenge on the CiS post; RETRACTION + FINDING 5)
179. **Logan pasted Vaintrob/Mendel/Hänni and asked if their quadratic U-AND has a mistake, given e3's
     rank bound. It does not — MY explanation had the mistake.** Their §1.5 construction is a bilinear
     layer + LINEAR readout (same architecture as e3), so "their gains need a nonlinear readout" is
     retracted. The true discriminator is the ERROR METRIC: the rank bound is an MSE statement; their
     construction optimizes ε-accuracy and its per-target MSE sits far ABOVE predict-zero (they say
     this themselves in §2). Ran their Gram through the bound: per-AND MSE floor ≈ p²(1−p)² ≈ predict-zero
     — no contradiction, superposition is just worthless under MSE and cheap under ε-accuracy.
180. **e4 (controlled demo, same task/arch/data as e3, m=128, d_h=32):** hand-coded low-coherence
     superposition computes ALL 128 squares to ε=0.20 with 32 hidden dims while its p-sparse MSE is
     12× WORSE than predicting zero. Training with MSE → 18 computed (= dedicated, on the rank bound);
     training with L8 (their own §2 ε-surrogate) on the SAME data → 128 computed. MSE-finetuning
     destroys the hand-coded superposition (128→4); L8-finetuning preserves it. The loss alone flips
     superposition on/off, both directions. Echoes tensor-sim FINDING 13: the metric decides what exists.

## 2026-07-13 — basis_aligned tick 3 (e5: CE and nonlinear readouts, Logan's "where does CE sit?" question)
181. **CE lands at the ε-accuracy end, not the MSE end.** Same bilinear toy (m=128, d_h=32), outputs as
     logits, task = predict the active feature: CE computes ALL 128 features (100% acc) with 32 hidden
     dims. Label smoothing 0.9 shrinks logit margins ~90× (57→0.65) but the computed SET stays all-128 —
     peakedness sets margins, not capacity. Mechanism: softmax only prices relative logits (threshold-like).
182. **A ReLU(Dh+b) readout under plain MSE BREAKS the linear rank bound**: 1-active MSE 7.7e-4 < the
     1.17e-3 floor binding all linear readouts, computing 64 = 2·d_h features to ~zero error. My first
     structural guess (sign-pairing, 2 features/unit) was REFUTED by inspection (2/32 units; dominant unit
     holds ~9% of a feature's mass): it's distributed superposition denoised by ReLU + learned negative
     bias, computing 64 perfectly and dropping 64 entirely. This is the TMS mechanism isolated, and it
     retro-justifies the retracted 'nonlinear readout' sentence as the correct explanation FOR TMS (not CiS).
183. Synthesis for LLMs: two independent routes to interference tolerance — threshold-like metric (CE/
     softmax) and downstream clipping nonlinearities — and real transformers have both, so heavy
     superposition graded by feature frequency is the prediction; MSE+nonlinearity SELECTS a subset
     (denoise-and-drop), ε-flavored objectives compute everything at nonzero interference.

## 2026-07-13 — basis_aligned tick 4 (e6: thread 3 opened — real embedding "objects", dual audit)
184. Logan specced thread 3 with a 4-class map of "minimal representation" (rank / MED / sparse-dictionary
     / tensor-network). e6 operationalizes classes 1+3 on pythia-410m embed_in with the program's dual
     audit: weight FVU vs behavioral ΔCE (embedding swapped into the live model, pile-10k).
185. **The tokens are the objects.** At 51% float budget: kmeans-25.6k (half the vocab as centroids)
     +1.45 nats, rank-512 +1.95; at ≤10% budget everything is FVU ≥ 0.74, ΔCE ≥ +4.3. No large
     "fewer-objects" reduction exists on raw weights. Hierarchy priors (RQ, tree) UNDERPERFORM at
     matched budget — Park-style hierarchy is real but carries little mass. Clusters that form are
     morphologically clean; many are singletons.
186. **FVU wildly mispredicts behavior — subtraction ≫ addition.** Additive gaussian noise at FVU 0.75:
     +0.43 nats. SVD deletion at the same FVU: +4.33 (10×; 18× at FVU 0.32). Noise must reach 3× the
     total embedding variance to match deletion at FVU 0.68. Random-basis and shuffled-assignment
     controls are 1–3.4 nats worse at equal budget. The model reads token identity across the whole
     1024-dim space; near-orthogonal corruption is cheap, deleted subspaces are fatal.
187. Next-tick options recorded in RESULTS: CE-trained codebooks (is the +1.45 metric mismatch or true
     incompressibility?), unembedding/softmax-bottleneck side, TT-ordering test, learned matryoshka arm.

## 2026-07-14 — basis_aligned tick 5 (e7: the Pareto frontier Logan asked for; FINDING 7a partly overturned)
188. Logan: matched budgets don't matter, Pareto-frontier the spectrum (SVD = few features/dense codes ↔
     original = V objects), one-time compute is fine. Also asked how much compute the "hierarchical SAE"
     took — honest answer: it was recursive k-means, ~2 GPU-min; nothing learned. e7 is the learned version.
189. **Stage A (signed top-k dictionaries on E's rows, n_atoms × L0 grid):** learning dominates every
     unlearned e6 arm. 16k atoms @ L0=64: dCE +0.11 (vs kmeans-25.6k +1.45). FVU mispredicts within
     learned dicts too: equal-FVU configs differ 7× in dCE (distributed codes >> hard quantization).
190. **Stage B (CE-through-the-frozen-model training of atoms+coeffs, supports frozen):** first run in
     fp16 DIVERGED (train CE rose — no loss scaling); caught by inspecting train curves, discarded,
     rerun in bf16 + grad clipping + bf16-consistent baseline. Fixed results: n=1024 atoms @ L0=64 goes
     +2.11 → +0.26 nats. "The tokens are the objects" (tick 4) was an artifact of unlearned Frobenius
     fits: 2% of vocab as learned atoms ≈ the whole embedding, behaviorally. k-means corner +1.47 → +0.87
     splits e6's damage: ~40% metric mismatch, ~60% genuine (escaping it needs L0>1).
191. Caveats logged: CE-training data only 262k tokens (~19 epochs; floor ~+0.08 partly data-limited;
     mild overfit at n=32k where train CE 1.95 < baseline 2.79). Open: identify the 1024 atoms
     (hierarchy? Park simplices?), unembedding side, TT-ordering.

## 2026-07-14 — basis_aligned tick 6 (e7c KL-faithfulness + MDL graph + e8 tensor-train, Logan's requests)
192. **MDL graph built** (Logan: "CE vs MDL or something") — every representation on one axis (fp16
     floats + index bits): figures/e7_mdl.png. Honest note: at L0=64 coefficients dominate bytes, so
     n=1024 is 49× fewer OBJECTS but ~8× fewer BYTES (12 vs 98 MiB).
193. **e7c answers Logan's sharpest question (data vs model-distribution training): the compression is
     FAITHFUL.** KL-distillation to the original model matches data-CE training (n=1024: +0.23 vs +0.26)
     and drives KL(orig‖comp) 2.10 → 0.25 nats; residual data-CE ≈ residual KL everywhere. Not repair.
194. **e8 (class 4, TT-SVD = linear-tree HT, vocab reshaped 16⁴, three orderings):** semantic ordering
     (balanced recursive k-means) beats random by a stable ~0.03 FVU at every rank — the vocabulary is
     measurably but WEAKLY hierarchical; native BPE ordering is indistinguishable from random (no
     block-locality in token IDs). TT+semantic beats SVD per-param on FVU; CE-finetuned TT cores land
     ON the MDL Pareto envelope at the small end (+2.01 nats @ ~5 MiB).
195. Program-level synthesis holding across all of thread 3: structure exists (semantic clusters, weak
     hierarchy, atom sharing) but the operative lessons are (a) learn the representation, (b) train/audit
     it under the behavioral metric, (c) never trust FVU.

## 2026-07-14 — basis_aligned tick 7 (Logan's HT questions: explainer, correction, gradient fits)
196. **CORRECTION to FINDING 10 (Logan's "isn't 0.85 FVU bad?" caught it):** the rmax=256 TT is 4.7% of
     E's floats, not 2.4%; at matched params plain SVD (0.836) slightly beats semantic TT-SVD (0.848).
     "TT beats rank per-param" retracted; the ordering GAP survives as the real class-4 measurement.
197. **e8b (weights-only gradient fits — Logan: "optimize HT to match the matrix directly; tensor cos-sim
     with scale"):** gradient TT improves on TT-SVD by only ~0.009 (sequential SVD near-optimal);
     rank sweep to rmax=1024: FVU 0.45 @ 43% of E, SVD slightly ahead throughout; balanced-tree HT
     (r=16,r2=48) underperforms chain at matched params BUT its semantic-vs-random gap is 2× larger
     (0.05 vs 0.028) — block-hungry topologies reward semantic ordering more.
198. Wrote TENSOR_NET_EXPLAINER.md (Logan's request): the token-90 worked example (digits select which
     small matrices to multiply), all hyperparameters (ordering / base&digit-count / topology / per-edge
     ranks / solver), param formulas, all tables, and the unswept grid. e9 (BatchTopK robustness check
     of the +0.26 headline) queued.

## 2026-07-14 — basis_aligned tick 8 (e9/e10/e11: BatchTopK, weight-metric negative, control ladder + ordering null)
199. **e9 (Logan's BatchTopK question):** adaptive L0 gets better FVU and WORSE dCE at every config
     (allocation follows row-norm, not behavioral importance) — the metric divergence now inside the
     sparsity allocation. CE-finetuned endpoint +0.28 ≈ fixed-k +0.26: headline robust.
200. **e10 (Logan's "weight-based tensor-sim should predict behavior"): honest NEGATIVE with a theorem-
     shaped reason.** The linear-reader metric M = Σ WᵀW over all 49 readers is ≈ isotropic; weighted
     FVU ≈ plain FVU on all 16 measured perturbations (Spearman 0.685→0.691); M-weighted dictionary fit
     is behaviorally worse. Structural point: NO quadratic form in ΔE can separate noise from deletion —
     the asymmetry is nonlinear robustness (LN+softmax denoise incoherent perturbations). Proposed e12:
     second-order weights-only objective — preserve the read-Gram ‖ÊMÊᵀ − EMEᵀ‖ (Isserlis-exact on the
     bilinear checkpoints).
201. **e11 (Logan's control ladder + learned ordering):** ALL controls pass (L0 grad 5.7e-12; (2,32768)
     full-rank exactly 0 — "factorizing is free" confirmed; r₂=256 = SVD-256 floor 0.5496 exactly, by
     both TT-SVD and gradient). Global-rank floor now explicit: FVU(TT@rmax) ≥ FVU(SVD@rmax) for ANY
     ordering. Learned ordering by pairwise swaps: NULL — every accepted swap was pad⟷pad; ZERO improving
     real-token swaps; semantic ordering is 2-swap-locally-optimal. Gap to floor needs block-level moves
     or is structural.
202. Two OOM/bugs caught by inspection this tick (e9 unchunked gather 18GB; e11 Gram on the tall side
     16GB) — both fixed, both re-run to completion. Logan's next program: qk_mdl (spec read; see tick 9).

## 2026-07-14 — qk_mdl tick 0 (program armed per Logan's instruction)
203. Read Logan's spec (basis_aligned/qk_mdl/qk_mdl_spec.md). Verified §0 assumptions from source:
     A1 pre-RMSNorm affine-free (no γ; lerp-0.5 residual halves the layer-1 direct path); A2 rotate-half
     RoPE, all dims, 16 bands, both branches; A3 **NEITHER spec case: NO softmax at all** — pattern =
     (q1·k1)(q2·k2)/d² × causal mask, fully polynomial. Consequences logged in qk_mdl/LOG.md: no softmax
     gauge (don't row-center), JS metric inapplicable → QUESTION FOR LOGAN + provisional metric
     (relative pattern MSE + downstream ΔCE). G-tie passes (4 separate Linears).
204. Cron job armed (every 2h): one spec-ordered tick per firing, anti-drift rules restated in the
     prompt, gates before claims, Tier 0.4 planted-structure battery = the ground-truth-MDL component.
     Note: cron is session-scoped with 7-day expiry — re-arm in future sessions if program continues.

## 2026-07-14 — qk_mdl tick 1 (Tier 0.2–0.3: exactness gate PASSING)
205. folding.py + tier0_gate.py. Gate first FAILED at 1e-10 — root cause in model source (Rotary builds
     fp32 trig tables), not the algebra; gate now uses the model's own tables via exact difference
     identities → PASS on 3 checkpoints (pattern err ~2e-15, branches ~1e-13, gauge 9e-16). Analytic-ω
     deviation from deployed models quantified (~1e-4) for ε calibration. Descriptive band profiles:
     strong mid-band concentration (one branch 57% in a single band). Next: Tier 0.4 metric + planted
     ground-truth-MDL battery.

## 2026-07-15 — qk_mdl tick 2 (Tier 0.4: conventions frozen, planted battery PASS 3/3)
206. mdl_accounting.py (DL in bits, distortion conventions, ε rule — FROZEN), codebooks.py (svd /
     bicluster / toeplitz), tier04_battery.py. Battery caught a real solver bug on first run (random-init
     biclustering lost its own plant to SVD; fixed via spectral init) — the ground-truth battery works.
     Final: SELECTIVITY PASS 3/3; SVD pays 45× true DL on the bicluster plant (computational ≪ spectral
     MDL, on a plant); bicluster solver known gap: k=16 vs planted k=8 (2.4× true DL), logged not hidden.
     Pending: conjunction plant + sparse-bilinear codebook, HODLR/tree (tick 3), then Tier-1 real heads.

## 2026-07-15 — qk_mdl tick 3 (conjunction codebook + plant; battery PASS 4/4)
207. fit_conjunction (bicluster ⊙ Toeplitz gate, alternating weighted LS) + conjunction plant. Battery
     caught the random-init solver bug a SECOND time (conjunction failed its own plant at k=64 and the
     pure-bicluster plant outright); spectral init on the gate-whitened matrix fixed it → SELECTIVITY
     PASS 4/4, conjunction wins its plant 33× over SVD, loses to plain bicluster by exactly the
     constant-gate overhead. Gaps logged: k=32 vs planted 8 (7× true DL); positive-gate identifiability
     limit documented (real pipeline has branches, never blind). Next: Tier 1.1 real layer-0 MDL table.

## 2026-07-15 — qk_mdl Tier 2 (Logan's directed push, ~4h of 10h budget): Elriggs models done
208. CE gate: sqrd12 3.37✓; bilin18 3.23 @T=512 but 5.50 @T=1024 — context degradation past ~512 found
     and characterized (unnormalized score-product row mass grows with key count); audits frozen @T=512.
209. bilin18 (546M, two-branch): layer-0 QK = 884 MiB of folded factors; joint vq256 (256 token-classes
     per head-branch) = ΔCE +0.0084 at 165× compression. Marginals don't compose (7 individually-free
     heads cost +0.53 jointly). Pattern-MSE useless (vq16 at MSE 0.95 ≈ free; H3 vq16 IMPROVES CE).
     Classes are readable token types (digits/punct/morphology/nouns/determiners).
210. sqrd12 (162M, one-branch normalized): ~15× less compressible at matched DL ratio (+0.116 @ vq256);
     no free head ablations. Contrast logged with candidate explanations open. TIER2_RESULTS.md is the
     deliverable; all gates passing.

## 2026-07-15 — qk_mdl tick 4 (Tier 1.1: tiny-model layer-0 table)
211. Reference forward bit-exact (0.0 logit diff); baseline 4.634 ≈ recorded. FINDING T1-1: tiny model
     is the structural opposite of the 546M — rank-compressible (svd16 free everywhere, svd1 near-free
     on 5/8 head-branches) but NOT token-clusterable (vq1 +0.24–2.19; joint vq256 +2.73 vs bilin18's
     +0.008). All-zero layer-0 QK +16.7. Depth story: 2-layer models need fine token identity in L0;
     18-layer models route coarse token types. Non-monotone joint-vq flagged (kmeans seed variance).
     Next: pre-registered L1H2 conjunction test (Tier 1.2).

## 2026-07-15 — qk_mdl tick 5 (Tier 1.2: pre-registered target missing; null + positive control)
212. attn2-seed0 (the spec's L1H2 retention-table model) no longer exists in runs_owt → deviation logged
     + QUESTION FOR LOGAN. Nearest substitute attn2-dense-seed0: real NULL (no match-and-copy behavior,
     no identity structure in any path, all at chance). Positive control on the genuine induction model
     (mix50-rp): screens recover the documented circuit (L1H0/L1H3 copy pair 25-30× chance, L0H1 prev-
     token) — machinery validated, null is checkpoint-specific. Design lesson: branch-zeroing is not
     branch-specific in product attention (s1·s2) — interventions must replace, not zero. Conjunction
     test re-anchored to the rp model for tick 6.

## 2026-07-15 — qk_mdl tick 6 (conjunction test, re-anchored: PARTIAL PASS, sharper than hypothesized)
213. Genuine-induction model, guard ✓ (P(copy) 0.7467 ≈ 0.7483). The conjunction is REAL and
     branch-specific but CROSS-HEAD REDUNDANT: each copy head carries token identity in exactly one
     branch (key-fed by L0H1 alone: H0.b1, H3.b2 — opposite branches!); destroying identity in one
     head costs ~0 (twin covers), destroying it in both identity branches collapses copying
     (−0.487/−0.517 total; diffuse branches only −0.138). Pre-registered single-head criterion fails
     for the documented redundancy reason; circuit-level conjunction confirmed. Weight-space identity
     signal partial (H3.b2 via L0H0 at 380× chance) — the known generic-vs-data-conditioned gap.

## 2026-07-15 — qk_mdl tick 7 (Tier 1.3 negative + joint-svd frontier; Tier 1 complete)
214. Positional-head sweep (32 branches, 2 tiny models): ZERO behaviorally-positional branches at
     |dCE| ≤ 0.01 — the spec's positional DL collapse is falsified for this zoo. Key distinction found:
     pattern-positionality ≠ score-positionality (rp L0H1 attends Δ=1 but loses −0.74 P(copy) when its
     scores are positional-averaged). mix10 joint-svd frontier: half-rank +0.054 (mildly non-additive).
     Tier 1 complete (1.1 ✓, 1.2 partial-pass re-anchored, 1.3 negative); attn2-seed0 question open.

## 2026-07-15 — qk_mdl tick 8 (data-conditioned identity metric: prediction CONFIRMED, Tier 1.2 → PASS)
215. Conditional-mean q/k by token on induction data, key side decomposed by L0 source: identity
     structure at 2200× chance in EXACTLY the two causal identity branches (H0.b1 0.444, H3.b2 0.423
     gauge-corrected), exclusively via L0H1; all other cells at chance. Generic-vs-conditioned gap
     resolved (generic said L0H0; conditioned matches the causal ablations). Sign flip between heads =
     pure branch-sign gauge. Conjunction test now PASS at full strength on the re-anchored model.

## 2026-07-16 — qk_mdl tick 9 (results/ folder; CE/KL codebooks; Tier-3 negative)
216. results/ subfolder with per-experiment MDs + figures + examples (Logan's request). CE-trained joint
     codebooks BEAT the original 546M layer-0 at every k (vq64: +0.015 → −0.032 at 500×); KL-distilled
     variant proves faithful compression alone reaches parity (−0.007) — 64 token-classes ≥ 884 MiB of
     trained weights. Tier-3 lookup codebooks: informative negative (structure-visible ≠ computation-
     sufficient; tables carrying 0.44 identity hit rate still destroy the circuit when substituted).

## 2026-07-16 — qk_mdl tick 10 addendum (OV CE-trained)
217. OV vq CE-training recovers only ~38% (vq1024 +0.92→+0.57) where QK went NEGATIVE — the
     selection/content dichotomy is genuine. Next: top-k sparse coding for OV (the e7 move proper)
     + the V×V cross-block codebook.

## 2026-07-16 — qk_mdl tick 11 (methods explainer, unified comparison graph, codebook pattern display)
218. results/00_methods.md (code+intuition per codebook); fig_methods_compare.png (all families, one
     object — new joint arms: svd16 +0.0045 @12.5%, svd64 negative @50%, positional +1.47 → layer-0 QK
     ≈ 1.0 nats positional + 1.5 token-selective, classes capture the selective part at 20× less DL than
     rank); fig_pattern_display.png (patterns FROM the vq256 CE-trained codebook vs original, token-
     labeled; 48% pattern MSE yet better CE — the dissociation visible).

## 2026-07-16 — qk_mdl tick 12 (methods explainer v2, annotated display, shared-registry finding)
219. 00_methods.md rewritten per Logan (factors defined, full code incl. helpers, conjunction
     step-by-step, FAQ); pattern display re-rendered with token·class labels. Shared-registry test:
     selection robust to partition choice (+0.008→+0.051), content tolerates none (own +1.38, QK's
     +1.81, global +2.47), both-global +2.78 — FINDING SR-1: token-interchangeability is
     circuit-specific; no single privileged coarse structure on the embedding.

## 2026-07-16 — qk_mdl tick 13 (OV sparse coding confirmed)
220. Top-k sparse coding on OV v-tables: +0.034 L2-fit (vs vq256's +1.38), CE-trained −0.019 — better
     than original. Refined dichotomy: selection = hard classes, content = sparse combinations; both
     layer-0 circuits now compress to better-than-original under behavioral training.
