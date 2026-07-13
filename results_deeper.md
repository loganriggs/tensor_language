# Deeper circuits on natural text: results log

## EXECUTIVE SUMMARY (2026-07-09, end of the 24h autonomous sprint)

**Program**: train bilinear (tensor) attention/MLP ladders on natural text, find the
datapoints each architectural increment unlocks (differential per-token CE, seed-median
hysteresis thresholds), identify the circuits behind them, isolate each in a toy task,
and characterize formation dynamics (Singh et al. 2024 as the mold). Corpora:
TinyStories (V=1024, discovery+mechanism) and OpenWebText (V=5120, dynamics).

### The circuits found
1. **Induction (depth-2 gates, 137k tokens, 64% strict bigram-copy).** Fully reverse-
   engineered on a real example: a 3-head circuit — L0H3 (previous-token head) writes
   the matched token into the source's residual; L1H2 (primary) and L1H1 (secondary)
   both read that key at the final token and copy the source value. Verified three
   ways: position-specific ablations (head × token causal map), directional weight
   composition (both bilinear key branches of L1H2 select L0H3; raw norm composition
   fails — needs the right input direction), and match-weight retention (only L0H3@src
   collapses the match, −.434→−.031). Sign structure: pattern×OV AGREEMENT (XNOR) is
   the head signature, not pattern sign (L1H2: −×− = strongly positive).
2. **Higher-order n-gram/lexicon circuits (depth-3/4 gates, ~33k tokens).** NOT copies:
   targets appear earlier at base rate; 67% are >0.5-predictable from 3 corpus context
   tokens (e.g. " m|ak|es| bu"→"b" needs exactly 3 tokens: P .105/.322/.997). Path:
   L0/L1 heads gather offsets −1/−2; an L2 head composes and emits.
3. **Chained k-hop retrieval (toy only).** Depth-3-gated, per-layer pointer advance in a
   rotated basis; absent from natural-text demands at this scale.

### The architecture laws (all 3-seed)
- **Two families, two axes**: context circuits (copy/induction/k-hop) need attention
  LAYERS (MLPs can't substitute); statistics circuits (n-gram/lexicon) need bilinear
  MLP capacity (block1 beats attn4 on order-2 Markov 0.19 vs 0.51; attention depth only
  weakly emulates it). Text confirmation: block1 solves the statistics examples attn2
  fails, but not the copy example; block2 (2 attn + 2 MLP) unlocks induction+composition
  together on fresh tokens (139k gates, 32% induction, 14%/1% overlap with attn gates).
- **Heads are a sweet spot, not redundancy**: attn2 gated-token CE by heads (d128):
  h1 1.52 / h2 0.64 / h4 0.47 / h8 1.56. Too few = missing complementary components;
  too many = d_head too thin for the bilinear match.
- **CE ladder (tiny, medians)**: block2 1.88 < block1 2.14 < attn4 2.17 < attn3 2.22
  < attn2 2.26 < attn1 2.43.

### The dynamics laws
- **Bilinear induction forms SMOOTHLY — no phase change** (250-step-dense replay,
  4.63→0.27 nats without plateau). Clamping the PT substrate (paper optogenetics)
  speeds mid-formation 1.5× → the subcircuit DEPENDENCY exists, but no saddle geometry.
- **Basin competition decides what forms**: on OWT no config forms induction unaided
  (d128 40k/120k; d256 slowly by 40k) though copy is perfectly learnable in isolation
  at V=5120. Circuit formation is dose-dependent (mix10 forms by ~9k, 3/3 seeds; mix5
  ~22k; mix3 not in 40k), expressivity-respecting (attn1+mix10 stays at chance), and
  HYSTERETIC: removing the scaffold at 15k collapses the synthetic behavior (overshoots
  past chance) while the natural induction capability persists and keeps improving
  (3.19 final vs plain 3.44). Anneal protocol nets positive overall (4.560 vs 4.562).
  Contrast: easy-first curriculum entrenches plateaus (hops 0/4); mixing-in pure
  circuit demand installs machinery the target distribution then maintains.

### Negative/corrected results (kept honest)
Softmax formed no induction on TinyStories in ≤104k steps (both residuals) — descoped
by Logan. Single-seed gates 4× inflated (lottery). Raw norm composition misleading.
k=3 Markov capacity-bound (not composition). "Hard threshold" dose → diverging delay.
config.json residual bug invalidated early softmax-add analyses (caught, re-run).

**Artifacts**: interactive circuit atlas https://claude.ai/code/artifact/a4364846-bbb7-469d-a491-48a5938b2df6
(tabs per circuit; click-to-ablate 3-head wiring; full bilinear attention patterns with
axes+colorbar; causal head×token map; output bar charts; k-hop worked examples;
two-axes example table). Reports: differential_report_{tiny,owt}.md, PLAN.md (gotchas),
this log (chronological below).

---

Program: PLAN.md. Corpus: TinyStories BPE-1024, frozen 15M-token val stream.
Ladder: bilinear attn-only, RMSNorm, d128 h4 ctx256, 40k steps, identical data order
across depths per seed.

## Session 1 (2026-07-08)

### Ladder val CE (final, quick-val)

| run | val CE |
|---|---|
| attn1-seed0 | 2.302 |
| attn2-seed0 | 2.16 (29k) → final pending |
| attn3-seed0 | 2.07 (38k) → final pending |

Monotone so far (gotcha #1 gate passing).

### Induction on text — the mold analysis (attn2-seed0 vs attn1 null control)

`induction_dynamics.py`; induction-predictable token := its bigram completion appeared
earlier in the window (11–12% of val tokens).

- **Depth gate confirmed at the category level**: attn1 plateaus at **1.56** nats on
  induction tokens (n-gram statistics only); attn2 reaches **1.22**. Non-induction
  tokens differ far less (2.43 vs 2.26). The differential-CE machinery sees the same
  category the induction-head literature predicts.
- **No sharp phase change at 2500-step checkpoint granularity** — the induction-token
  CE curve is smooth from ckpt 2500 on. Either the transition happens before 2500 steps
  (likely: TinyStories induction forms early) or bilinear circuits form gradually.
  → queued: attn2 rerun with dense early checkpoints (every 250 steps to 5k).
- **A negative-score induction head**: per-head progress measure (attention score from
  q to induction target j+1) shows L1H2 developing a strong NEGATIVE score (−0.11)
  while L1H1 is mildly positive (+0.03). Bilinear attention has no softmax, so signed
  scores are legal and the value path can flip sign back. The knock-out ablation agrees:
  removing L1H2 is the most damaging single knock-out (1.22 → 2.07).
- **No head redundancy — unlike the paper's softmax models** (their §3.1 found additive,
  redundant induction heads). Here every layer-1 knock-out hurts (→1.6–2.1) and every
  solo head is much worse (2.8–3.6). Bilinear induction appears *distributed across
  heads* (superposed, possibly with sign cancellation), not additive. This is a genuine
  architecture difference to track for every deeper circuit we find.

Figure: `figures/induction_dynamics_attn2-seed0.png` (+ attn1 null control).

### Infra

- 106 steps/s per run; full 40k-step run ≈ 7–12 min. Ladder attn1–4 × seeds 0–2 running.

### First differential report (PRELIMINARY — attn2/attn3 single-seed, medians pending)

Full-stream mean CE: attn1 2.449 (2 seeds) > attn2 2.301 > attn3 2.233 — monotone ✓.
Thresholds: learned < 0.5 nats, unlearned > 1.5 (hysteresis). Base rate of
induction-pattern tokens (bigram completion seen earlier in window): 11.7%.

| gate | count | % of stream | induction-pattern overlap |
|---|---|---|---|
| depth-2 (attn1→attn2) | 175,883 | 1.17% | **48.6%** (4.2× enrichment) |
| depth-3 (attn2→attn3) | 122,908 | 0.82% | **23.6%** (2.0× enrichment) |

- Decoded depth-2 examples are textbook induction: mid-word/name completions of words
  seen earlier in context ("m→ot" after 'motorcycle', " Ben" after an earlier Ben).
  The unsupervised differential-CE pipeline recovers the induction category on its own.
- Decoded depth-3 examples look like DIFFERENT categories: morphological completions
  ("search→ing", "storekeep→er"), story-convention name introduction (" Sp[ot]" for a
  new dog, "Lily" as first-mentioned girl), syntax ("teach you → how"). Only 23.6%
  match the bigram-induction pattern → mostly a new algorithm class. These are the
  "next induction head" CANDIDATES, pending: (a) 3-seed medians (lottery gotcha #2),
  (b) attribution fingerprints + clustering (task #5).
- `runs_lm/is_induction.npy` caches the induction mask for the full stream.

### Bilinear induction forms SMOOTHLY — no phase change (dense replay, 250-step ckpts)

`attn2-dense-seed0` replays attn2-seed0's exact trajectory (same init + data order) with
checkpoints every 250 steps to 5k. CE on the pure depth-2-gated token set (n=1437 in the
first 500 val windows, tokens attn1 provably never learns):
smooth exponential 4.63 → 0.265 nats across 40k steps (fastest stretch 7.5k→17.5k, but no
plateau-then-cliff anywhere; `runs_lm/attn2-dense-seed0/gated2_ce_curve.json`).

The canonical softmax induction story (Olsson et al.; the paper's Fig 3) is a loss
plateau then an abrupt phase change from interacting subcircuits (their a·b·c saddle
toy). In the bilinear parameterization that discontinuity is ABSENT on identical data.
Candidate explanation: the paper's saddle comes from products of small co-evolving
subcircuit strengths; bilinear attention replaces softmax sharpening with polynomial
score products, changing the geometry. RUNNING: softmax control (attn2-softmax-dense,
same data/init schedule) — if it shows the plateau+cliff, "smooth vs phase change" is an
architecture-level dynamics difference and a headline result for the program.

### Softmax control I (softmax-lerp): induction never forms — bilinear finds it FASTER

`attn2-softmax-dense-seed0` (same data order/schedule, softmax attention, lerp residual):

- On the depth-2-gated tokens it plateaus at **1.50 nats** (vs bilinear 0.27) despite
  BETTER overall val CE (2.10 vs 2.16) — it compensates with local statistics.
- Progress measure: max per-head attention weight to the induction target is **0.017**
  at 40k steps (real induction heads carry 0.2+), rising only slowly (L1H1/L1H3). No
  induction circuit; no phase change reached within the horizon.
- Interim picture: on this data, bilinear attention discovers the induction circuit
  dramatically earlier than softmax(-lerp); the classic plateau→phase-change may simply
  live far beyond 40k steps for softmax here. CONFOUND to resolve: lerp residual is the
  bilinear-tuned recipe; prior graph-program found softmax prefers `add`.
  RUNNING: `attn2-softmax-add-dense-seed0` (canonical add residual) to adjudicate.
- If softmax+add also fails/slow-forms: headline = tensor/bilinear attention changes
  induction from a delayed phase change to fast smooth learning (optimization-geometry
  claim, testable via the paper's subcircuit-clamping method — subcircuit products may
  no longer gate each other).

### Softmax control II (softmax+add, CORRECTED) + the delayed-phase-change test

BUG CAUGHT: lm_train wrote `residual: "lerp"` into every config.json regardless of the
actual residual; all post-hoc loads of attn2-softmax-add-dense-seed0 were silently wrong
(a lerp-wired forward over add-trained weights). Patched config; reran analyses. (Gotcha
added: verify load_model reproduces the run's final history val CE before analyzing.)

Corrected softmax+add (canonical recipe, best overall val CE 2.011):
- depth-2-gated tokens: smooth 4.27 → **1.15** nats — partial, far above bilinear (0.27).
- broad induction-category CE 1.23 ≈ bilinear's 1.22 (mostly statistics).
- induction attention score flat at ~0.017 through training → **no crisp induction
  heads formed by 40k steps** in either softmax variant.

Interim summary of the dynamics story (single seed each; caveat noted):
| model (attn2) | overall val CE | gated-token CE | induction heads? |
|---|---|---|---|
| bilinear-lerp | 2.16 | **0.27** | yes (distributed, signed, fast, smooth) |
| softmax-add | **2.01** | 1.15 | not by 40k steps |
| softmax-lerp | 2.10 | 1.50 | no |

Bilinear attention finds the induction circuit far faster; softmax buys better overall
CE from statistics instead. The classic softmax phase change may be DELAYED past 40k
steps here → RUNNING: attn2-softmax-add for 160k steps (dense early ckpts) to see if/when
the phase change arrives. If it does: same circuit, radically different discovery speed —
an optimization-geometry difference between tensor and softmax attention.

### Multi-seed differential (attn1 ×3, attn2 ×3, attn3 ×2 seeds) — gates sharpen

| gate | count (vs single-seed) | induction-pattern overlap |
|---|---|---|
| depth-2 | 136,740 (was 175,883) | **63.7%** (was 48.6%) |
| depth-3 | **29,117 (was 122,908 — 4× shrink)** | 20.9% |

Seed-median filtering removes lottery noise: 3/4 of the single-seed depth-3 "gates" were
seed idiosyncrasies (gotcha #2 in action, quantified). The surviving 29k depth-3 tokens
(0.19% of stream, only 21% induction-pattern) are the real next-circuit candidates.
Monotonicity holds. attn3-seed2 + attn4 seeds pending for the final cut + depth-4 gates.

### What depth 3 actually buys on text: NOT retrieval chains — feature composition

Characterization of the (2-seed-median) gate sets vs 6000 random baseline tokens:

| set | target seen earlier in window | current-token seen earlier | strict bigram-induction |
|---|---|---|---|
| base rate | 49.6% | 49.2% | 11.6% |
| depth-2 gated | **82.7%** | 78.6% | **64.8%** |
| depth-3 gated | **51.0% (= base!)** | 45.7% | 21.3% |

Depth-2 = in-context copying (induction), as established. But depth-3 gated targets are
NOVEL tokens in their window at base rate — the 3rd layer's differential value on
TinyStories is NOT longer in-context retrieval (contra the k-hop toy extrapolation).
Examples say it is FEATURE COMPOSITION: BPE morphology completing English word forms
("mater→i[als]", "scre→en", "real→iz", "y→ell[ow]"), plus story-schema/syntax tokens
(" B[ob]" after "a boy named" at a story boundary, "until it is → time").

Trajectory clustering v2 (`trajectory_clusters.py`, 6k tokens × attn3 seeds 0,1):
clusters separate by FORMATION TIME (half-formation step 5k→15k), all reaching CE
0.24–0.52, but examples are semantically similar across clusters → the depth-3 set looks
like one broad capability forming at difficulty-graded rates, not discrete subcircuits
(or the lens is still wrong — next: cluster on target-token linguistic type and
context-feature dependence rather than dynamics).

Honest divergence from the program's founding hypothesis: on natural text the "next
induction head" (first depth-gated class above induction) is NOT chained retrieval —
chained retrieval may simply be absent from TinyStories' distribution. The k-hop circuit
remains real but toy-task-specific. The discovery pipeline is working as designed:
it found what the data actually gates on depth.

## OWT (BPE-5120) session: 40k steps is NOT enough — no depth differential yet

First OWT completions (d128 h4 ctx256, 40k steps): attn1-seed0 val 4.588,
attn2-dense-seed0 val 4.562 — gap only 0.026 nats. Induction measures agree:
attn1 ind-CE 3.508 vs attn2 3.444 (Δ=0.06; TinyStories Δ was 0.34 with gated tokens
→0.27), attn2 induction-attention scores ≈0.018 max. At this scale/horizon on OWT the
models are pure statistics — no induction circuit, hence no depth-2 gates to find.
(OWT induction-predictable base rate: 7.4% of val tokens.)

Action (steps calibration, one knob at a time): killed the queued 40k-seed workers
(wasted compute given zero differential); attn3-dense + block2-dense allowed to finish
as depth references. RUNNING: attn1 + attn2 at 120k steps, dense ckpts, seed0
(tags *-s120k-dense-seed0). Decision rule: if the attn1↔attn2 induction gap opens by
~60-80k, rerun the ladder grid at the calibrated horizon; if not, next knob is width
(d_model 256) — capacity, not time, would be the binding constraint.
Early note: block2 (attn+MLP ×2) val 4.24@18k ≪ attn-only anything — bilinear MLPs
carry large statistics capacity on OWT; keep attn-only vs block ladders separate when
defining gates.

### TinyStories grid COMPLETE (archive): full ladder attn1-4 × 3 seeds

Median mean-CE ladder: 2.428 → 2.261 → 2.217 → 2.168 (monotone ✓). Final gate sets
(3-seed medians): depth-2 136,740 (63.7% induction-pattern), depth-3 32,806 (19.4%),
depth-4 33,180 (23.5%). Depth-3/4 gates share the non-copy profile (feature
composition). TinyStories work is now frozen as the pipeline-validation archive
(differential_report_tiny.md); active corpus is OWT pending the 120k steps calibration.

### Unifying lens (tiny archive): the depth axis is word-form completion; hypothesis = n-gram order

Target linguistic type (vs 20k-token base sample: 29% word-internal / 58% word-initial /
14% punct):
  depth-2 gated: 86% word-internal   depth-3: 82%   depth-4: 82%
ALL depth gates live inside multi-BPE words (3× enrichment). Depth-2 = copy the
word-form from context (induction, 64% strict-bigram). Depth-3/4 = complete word-forms
with NO in-context copy (corpus lexicon in weights). HYPOTHESIS (testable, cheap):
depth-d gates are tokens requiring order-d context to predict — each attention layer
composes one more previous-token feature into the query (n-gram order ladder;
generalizes "attention builds skip-grams"). Test: train-corpus conditional entropies —
depth-3 gates should be low-P(tgt|prev) but high-P(tgt|prev2,prev); depth-4 gates
should need the 4-gram. If true: the "next induction heads" on statistical text are
N-GRAM-ORDER circuits, and the toy isolation task is an in-weights lexicon of
multi-token words disambiguated only at prefix length d-1.

### N-GRAM-ORDER HYPOTHESIS: CONFIRMED for depth-3 (tiny archive)

Train-corpus conditional probabilities of gated targets (4000-token samples, single
616M-token targeted scan; "o-gram" = o context tokens):

| set | P(tgt|1): med / >0.5 | P(tgt|2) | P(tgt|3) |
|---|---|---|---|
| base | .043 / 4.9% | .121 / 22.6% | .197 / 32.4% |
| depth-2 | .031 / 0.2% | .120 / 18.6% | .238 / 36.3% |
| depth-3 | .040 / 0.7% | **.349 / 38.8%** | **.787 / 66.7%** |
| depth-4 | .046 / 1.4% | .318 / 34.5% | .674 / 60.2% |

Two circuit FAMILIES on statistical text:
1. depth-2 = in-context copy (induction): corpus n-grams DON'T predict them (≈ base at
   every order) — the model must read the answer from context.
2. depth-3/4 = higher-order n-gram / lexicon circuits: 67% of depth-3 gates are
   >0.5-predictable from 3 context tokens in the corpus. "Feature composition" made
   precise: the 3rd layer buys ORDER-3 statistics (composing ≥2 context-token features
   into the prediction). Empirically 2 bilinear attn layers do NOT learn trigram
   statistics here — an architecture-specific expressivity/optimization fact worth its
   own toy study. Depth-4 is the same family (harder tail; not separated up to 4-grams —
   may need order-4/5 counts or it's just difficulty-graded).

TOY ISOLATION TASK (designs itself): order-k Markov chains in-weights (no in-context
structure): which depth learns order k? Prediction: bilinear attn-d learns order ≤ d-1.
Plus the induction family stays separate (copy vs statistics = two axes: context-reading
circuits vs in-weights composition circuits).

### Calibration mid-flight read (dense ckpts, matched steps): time alone is NOT opening the gap

OWT induction-token CE, attn1 vs attn2 (s120k runs):
step 10k: gap .007 | 20k: .032 | 40k: .026 | 60k: .020 | 75k: .041
Trend is a slow creep, nowhere near a formed circuit (TinyStories formed gap: 0.34 on
ind tokens, gated set → 0.27 nats absolute). Hypothesis shift: WIDTH is binding —
d128 embeddings over V=5120 leave no subspace for match-and-copy machinery beyond
statistics (TinyStories worked at V=1024 = 4× more dims per token). RUNNING: d256
attn2 (dense) + attn1 at 40k steps (tags *-d256-*). If the gap opens at d256/40k →
rerun ladder at d256; if not → next suspect is the bilinear match at large vocab
(score resolution), testable with a synthetic in-context copy task at V=5120.
Markov toy note: deeper models underfit even order-1 at 12k fixed steps (attn4 gap
0.19) — read order-k results RELATIVE to each depth's k=1 baseline.

### Width does NOT rescue OWT induction; copy-isolation launched

d256 attn2 (OWT, dense): statistics improve (val 4.452@18k < d128's 4.562@40k; ind-CE
3.30 vs 3.53 at matched 17.5k) but induction-attention scores stay FLAT (~0.011-0.015)
— no match-and-copy circuit at d256 either. Every bilinear config at V=5120 has failed
to form induction; V=1024 TinyStories formed it easily. Two live explanations:
(a) bilinear product-score match degrades with VOCAB SIZE (architecture limitation);
(b) statistics basin outcompetes the circuit on OWT (optimization).
`copy_isolation.py` (running): pure [u;u] copy task, iid uniform tokens (no statistics
to hide in; floor 0 nats), bilinear attn2, grid V∈{1024,5120} × d∈{128,256}. If V=5120
fails HERE, (a) is confirmed — a core tensor-network limitation finding.

Markov k=2 so far: attn1 gap 1.49 (fails, predicted), attn2 gap 0.86 (well off floor —
either order-2 needs 3 layers, or slow optimization; attn3-k2 next decides).

### COPY ISOLATION: vocab hypothesis REFUTED — OWT failure is BASIN COMPETITION

Pure [u;u] copy (iid uniform, floor 0): V1024-d128 CE 0.0005, V1024-d256 0.0000,
V5120-d128 **0.0059** (slower start — 0.75 at step 2k — but fully solved by 8k).
The bilinear match-and-copy circuit is expressible AND findable at V=5120 in isolation.
⇒ OWT's missing induction is OPTIMIZATION/DATA competition: the statistics basin keeps
paying (unlike TinyStories where stats saturate early), and SGD never finds the circuit
within 120k steps. Same phenomenon class as the hop-task seed lottery (induction
plateau) — circuits lose to easier basins. THE program question is now: what data/
training lever makes circuit formation win? (Curriculum failed for hops; next lever:
INJECT in-context-copy structure into the stream — burstiness à la Chan et al.)

### Markov k=2: NO crisp depth gate — in-weights order-2 is hard at every depth

gaps: attn1 1.49 / attn2 0.86 / attn3 0.61 / attn4 0.51 (vs k=1 baselines 0.01-0.19).
Depth helps GRADUALLY; nothing "unlocks" at 3 layers. So the tiny-corpus depth-3 gates
(order-3 statistics) are NOT explained by raw expressivity thresholds — likely
optimization-rate differences on skewed natural n-grams (uniform Dirichlet tables may
also be unnaturally hard: no low-order backoff structure). Toy follow-ups queued:
longer steps (is it time?), skewed/backoff tables (is it distribution shape?).

### Markov k=2 complete — the composition engine is the bilinear MLP, not attention depth

gaps (k=2): attn1 1.49 | attn2 0.86 | attn3 0.61 | attn4 0.51 | **block1 (1 attn + 1
bilinear MLP) 0.19** | block2 0.30. One bilinear MLP beats FOUR attention layers by
2.7× on in-weights order-2 statistics. TWO FAMILIES, TWO AXES (the program's clean
architectural statement so far):
- CONTEXT circuits (copy/induction/k-hop): need attention LAYERS (cross-position
  composition); bilinear MLPs cannot substitute (hop result); form fast in isolation;
  gated by basin competition on rich data.
- STATISTICS circuits (n-gram/lexicon): need bilinear MLP capacity (within-position
  feature composition); attention depth is a poor substitute — explains block2's OWT
  dominance (4.14 vs attn2 4.56) and reframes the tiny-corpus depth-3 gates: the 3rd
  attn layer was weakly emulating an MLP.

### Basin competition is quantitative: mixture slows circuit discovery

attn2-mix10 (10% copy-burst rows in OWT): copy-row CE still at CHANCE at step 5k,
though the identical task in ISOLATION clicks by ~2k. The OWT gradient crowds out
circuit formation; watching whether it forms by 40k. Handle: time-to-circuit vs
mixture fraction (next: mix30 if mix10 stalls).

### MIXTURE RESULT: copy-bursts install a TRANSFERABLE induction circuit on OWT

attn2-mix10 (10% [u;u] rows) vs plain attn2, matched steps, natural OWT val tokens:

| step | mix10 ind-CE | plain ind-CE | mix10 max ind-score | plain |
|---|---|---|---|---|
| 10k | 3.857 | 3.711 | 0.027 | 0.015 |
| 20k | 3.459 | 3.530 | 0.086 | 0.012 |
| 30k | 3.233 | 3.449 | 0.137 | 0.012 |
| 32.5k | **3.214** | 3.438 | **0.143** | 0.013 |

- Copy-row CE: chance (8.7) at 5k → 1.87 at 10k → 0.42 at 32.5k. The circuit forms at
  ~7-9k under 10% mixture (vs ~2k in isolation): basin competition DELAYS discovery ~5×
  but 10% signal suffices.
- TRANSFER: the circuit trained on synthetic bursts fires on NATURAL induction tokens —
  ind-CE advantage 0.22 nats and growing, induction-attention score 0.143 and rising
  (plain flat at 0.013). Early cost (10k: mix worse — 10% of data budget is synthetic)
  is overtaken by 20k.
- CONTRAST with hop curriculum (backfired 0/4): scaffolding by EASIER-VERSION-FIRST
  entrenches plateaus, but MIXING-IN pure-structure examples of the target circuit
  installs it. Data mixture = circuit installation; the lever for "what forms" is the
  presence of undiluted circuit-demanding examples in the stream, not ordering.
Next: (a) let mix10 finish 40k + eval depth-2 gates vs plain; (b) dose-response
(mix1/mix3/mix30?) if worth it; (c) does the installed circuit IMPROVE overall val CE
(is induction actually worth nats on OWT at this scale)? — overall val mix 4.64 vs
plain-40k 4.56 currently reflects the synthetic-row budget; needs matched-step compare.

### Markov k=3 is capacity-bound (honest caveat) + mixture robustness launched

ALL architectures fail order-3 equally (attn1-4 gaps 2.37-2.60, block1 2.43): the k=3
table has 262k contexts — beyond the ~0.5M-param models' storage regardless of wiring.
k=3 tests CAPACITY, not composition; the two-axes claim rests on k≤2 (where block1's
advantage is clean). Mixture finals: attn2-mix10 40k val 4.637 (vs plain-40k 4.562 —
the synthetic-row budget costs ~0.08 overall while installing induction; whether the
circuit PAYS overall needs longer horizons or better mixtures). Launched: mix10 seeds
1-2 (robustness) and mix03 dense (dose-response).

### REVISION: width accelerates OWT induction after all (late-formation, missed at 18k)

d256 40k finals: attn1-d256 4.446, attn2-d256 4.309 (depth-2 gap 0.137 vs d128's 0.026).
Decomposition: ind-CE 3.347 → 2.958 (0.39 gap on induction tokens) and the induction
score GREW 0.011 (18k) → 0.047 (40k) — the circuit was slowly forming; my 18k check
called "flat" prematurely (gotcha: don't judge slow formation mid-run). Corrected
picture: circuit formation speed on OWT scales with BOTH width (d256 ≫ d128) and
injected copy-signal (mix10 ≫ plain); basin competition is the frame, and these are its
dose knobs. d128-plain never forms it in 120k; d256-plain partially by 40k; d128-mix10
fully by ~30k.

### Dose-response is a THRESHOLD + XNOR head structure measured (atlas v3)

- mix03 (3% copy-bursts): copy-row CE flat at chance for ALL 40k steps — the circuit
  never forms. vs mix10: forms by ~9k. Basin escape is a PHASE BOUNDARY in signal
  fraction (between 3% and 10%), not a smooth speedup.
- XNOR structure (Logan's correction, measured): in bilinear attention what matters is
  pattern×OV sign AGREEMENT, not pattern sign. L1H2 on the jewelry example: pattern
  −0.43 × OV −4.20 = +1.83 (supports correct token). Aggregate over 2,898 induction
  positions: L1H2 mean product +0.485, 68% positive (other heads ≈0/51-55%). "Negative
  attention score" framing retired; results and atlas updated.
- N-gram examples now carry full evidence: tokenized context, per-order corpus P with
  counts (e.g., " m|ak|es| bu"→"b": P=0.105/0.322/0.997 — true 3-gram), and the head
  path: L0/L1 heads gather offsets −1/−2, an L2 head composes and emits (ablate → P≈0).
  Golf example: order-2-sufficient statistics STILL not learned by attn2 (P=0.01) —
  learning, not expressivity.
- Atlas artifact v3: per-circuit tabs, interactive vertical induction view (click-to-
  ablate heads, live output bar, attention highlighted on the ribbon), visible-repeat
  examples, k-hop pointer-advance vertical view.

### Mixture updates: seed-robust at 10%; 5% still at chance by 15k

mix10-seed1 final copy-row CE 0.796 (formed; seed0 0.42) — 2/2 seeds install the
circuit at 10%. mix5 at 15k: still chance (8.58) — threshold (or strong slowdown)
persists above 5%; run continues to 40k. Anneal run passed the 15k mixture-off point;
transience readout at 25-40k.

### Mixture installation is RELIABLE: 3/3 seeds at 10%

mix10 final copy-row CE by seed: 0.42 / 0.80 / 0.61 — all form the circuit. Contrast
with the hop-task seed lottery (1/3 at any depth): sufficient mixture signal makes
circuit formation reliable, not merely possible. The unreliable regime is
sub-threshold signal (hops: rare in-task demand; OWT plain: zero pure-copy demand).

### TRANSIENCE TEST: circuit formation is HYSTERETIC — scaffold to create, nature maintains

mix10-until-15k (mixture removed at 15k, pure OWT after), dense ckpts:

| step | copy-row CE | natural ind-CE |
|---|---|---|
| 15k (mix off) | 0.816 | 3.570 |
| 17.5k | 10.43 | 3.514 |
| 25k | 11.96 | 3.309 |
| 40k | 12.27 | **3.188** |

- The SYNTHETIC behavior collapses within 2.5k steps of removal and overshoots past
  chance (8.54 → 12.3): pure OWT anti-learns iid-uniform copying (off-distribution).
- The NATURAL induction capability persists and keeps improving without the scaffold:
  final ind-CE 3.188 beats plain attn2 (3.438) and matches always-mixed (≈3.19).
- Interpretation: HYSTERESIS in circuit space. Natural data cannot form the circuit
  (basin competition) but CAN maintain it once formed (7.4% of tokens exercise it).
  Practical: a temporary mixture phase installs a permanent natural capability — no
  need for permanent data contamination. Refines the curriculum picture: scaffolds
  work when they install machinery the target distribution then rewards; they fail
  (hop curriculum) when they entrench a competing basin.

### CORRECTION on dose-response: not a hard threshold — a steeply diverging delay

mix5 copy-row CE: chance @15k → 6.77 @20k → 2.13 @30k → 1.73 @40k (formed, late).
Full dose curve: 10% forms ~9k; 5% forms ~20-25k; 3% not by 40k (likely delayed beyond
budget, not impossible). Formation TIME grows superlinearly as signal fraction drops;
within a fixed compute budget it presents as a threshold. (Earlier "phase boundary"
phrasing corrected.)

### Overall-CE payoff + mix10 LADDER launched

Overall val CE at 40k: anneal (mix off @15k) 4.5598 < plain 4.5622 < always-mixed
4.6373. The installed circuit repays its installation budget (+0.002 net) — small
absolute payoff, consistent with 0.25 nats × 7.4% induction tokens. Always-mixed pays
a 0.075 synthetic-budget tax with no extra natural benefit → anneal is the right
protocol.

RUNNING: the mix10 LADDER — attn1-mix10 ×3 (control: 1 layer cannot implement copy
even with the scaffold; mixture should NOT help) and attn3-mix10 ×3. With induction
installed at depth 2, the OWT differential (attn2-mix10 vs attn3-mix10, 3 seeds each)
asks what the 3rd layer buys on natural text once the context-circuit bottleneck is
removed — the OWT analogue of the tiny-corpus depth-3 gates.

### Control: mixture cannot install what depth cannot express

attn1-mix10-seed0 copy-row CE 9.07 ≈ chance after full training with the same 10%
scaffold that installs the circuit 3/3 in attn2. The lever respects expressivity —
it changes WHICH basin optimization finds, not what the architecture can represent.
(Depth gates remain real gates; the mixture removes only the optimization barrier.)

### OWT mix-ladder differential (attn1/2/3 × mix10, 3 seeds): depth-3 buys most; depth-2 nets ~zero at d128

Overall natural-stream CE (median): attn1-mix10 4.667 ≈ attn2-mix10 4.671 (**monotone
WARNING**) < attn3-mix10 4.615. At d128 + scaffold, attn2's copy circuit costs about as
much statistics capacity as its natural induction pays — net ~0 overall (though its
specific induction tokens do improve: 4,845 depth-2 gates, 18.4% induction-pattern vs
7.7% base). attn3 has slack for both → clear gain and 10,286 depth-3 gates (16.4%
induction-pattern). Examples: word completions ("Mediter→r") AND long-range syntax —
closing "]" after a long parenthetical (delimiter tracking: possible new circuit class
worth isolating). Caveats: gate sets are small because learned<0.5 nats is a much
higher bar at OWT CE levels (~4.6); consider per-corpus thresholds. The tiny-corpus
conclusions (two families) remain the cleaner discovery set; OWT adds the capacity-
cost/payoff tradeoff at fixed width.

### Delimiter lens: enriched but minor; heads ladder launched

Closing-delimiter targets in OWT gates: 3.0% (depth-2) / 2.8% (depth-3) vs 1.3% base —
~2.2× enriched, but a small component; the bulk of OWT depth-3 gates remains the
statistics family. Not pursuing a delimiter toy yet.

RUNNING (task #3, heads axis): tiny attn2 × heads {1,2,8} × 3 seeds. Question: bilinear
induction at h4 concentrates in ONE head (L1H2 carries the XNOR product; no
redundancy) — does h1 suffice for induction? Does h8 form redundant copies (softmax-
style) or stay concentrated?

### Heads axis (partial): 1 head/layer degrades induction badly despite the circuit
concentrating in one head at h4

attn2-h1 (full d_head=128/head): ind-CE 1.33/1.41, gated-token CE 1.06/1.56 (2 seeds,
high variance) vs h4's 1.22/0.27. The h4 circuit's XNOR product lives in ONE head, yet
single-head models can't replicate it — the "minor" heads are complementary circuit
components (matches no-redundancy ablations: parts, not copies). h2/h8 pending for the
full table.

### Heads axis is NON-MONOTONE: h4 optimal at d128 (gated-token CE)

| heads (d_head) | h1 (128) | h2 (64) | h4 (32) | h8 (16) |
|---|---|---|---|---|
| gated-token CE | 1.06/1.52/1.56 | 0.89 (seed0) | **0.27** | 1.85/1.56 |

Too few heads: missing complementary circuit components (routing/substrate pieces).
Too many at fixed d_model: d_head=16 too thin for the bilinear product match at V=1024.
Bilinear induction has a head-count sweet spot — neither "more heads = more redundancy"
(softmax picture) nor "one big head suffices". Remaining h2/h8 seeds pending for
medians.

### Heads axis COMPLETE (3 seeds each, median gated-token CE)

| h1 (d_head 128) | h2 (64) | h4 (32) | h8 (16) |
|---|---|---|---|
| 1.522 | 0.638 | **0.466** | 1.563 |

Non-monotone sweet spot at h4 confirmed across seeds. (Footnote: h4's earlier 0.27 was
measured against the single-seed gate set; the 3-seed-median set is stricter — same
model, apples-to-apples table above.) Task #3 (depth × heads ladder) complete: depth,
heads, and MLP axes all quantified.

### Clamping (paper optogenetics, bilinear): dependency without the saddle

PT-attend clamp (L0H0 pattern fixed to perfect prev-token from step 0), gated-token CE
vs the unclamped replay: identical to 5k, FASTER 7.5k-15k (10k: 0.92 vs 1.45 — the
match circuit builds sooner when the key substrate is free), WORSE after 20k (final
0.446 vs 0.265 — the frozen pattern costs capacity). Conclusion: bilinear induction has
the paper's subcircuit DEPENDENCY (L1 match waits on L0 substrate) but no saddle
GEOMETRY — formation stays smooth under every condition tested. The softmax phase
change is a property of the parameterization, not of the circuit's dependency
structure. Task #6 (paper-mold study) complete: loss split, progress measures,
knockout/solo ablations, XNOR head structure, position-specific wiring, clamping.

### Atlas v4 (all Logan's viz requests) + block1 text confirmation

block1 (1 attn + 1 bilinear MLP) on tiny: val 1.975 — best tiny model, beats attn3
(2.071). On the shared examples: solves all four statistics examples (0.57-0.96) where
attn2 fails (0.01-0.10), but only 0.45 on the copy/induction example where attn2 gets
0.93 — the two-axes dissociation on ACTUAL text predictions. Atlas v4: tight token
ribbon (no spacing/margins), two-head circuit wiring with SVG arrows and position-
specific ablations (L0H3@src → L1H2@q), k-hop concrete document + resolution chains,
two-axes tab led by the example table.

### Threshold sensitivity (closes task #4): OWT gates robust to τ

| (τ_learned, τ_unlearned) | d2 gates / ind% | d3 gates / ind% |
|---|---|---|
| (0.5, 1.5) | 4,845 / 18.4% | 10,286 / 16.4% |
| (0.75, 2.0) | 4,911 / 19.3% | 10,536 / 15.2% |
| (1.0, 2.5) | 4,489 / 20.9% | 9,438 / 15.0% |
| (1.5, 3.0) | 7,804 / 21.3% | 15,307 / 14.2% |

Gate structure is threshold-stable across a 3× range; --tau flag added to differential.py.
Task #5 (clustering) also closed: delivered via knockout fingerprints (v1), formation-
trajectory clustering (v2), structural lenses (word-internal / n-gram order / target-seen-
earlier / delimiter), and position-specific circuit wiring — "same algorithm ⇒ same
circuit" established at the family level (context vs statistics), with head-level wiring
for induction.

### Deep induction accounting (atlas v6): 3-head circuit, composition verified

- Causal head×token map (zero each head at each single token): L1H2@q ΔP 0.80,
  L1H1@q 0.36 (secondary parallel path), L0H3@src 0.15; all others ≤0.02. The circuit
  is THREE heads: L0H3 → {L1H2 primary, L1H1 secondary}.
- Weight-level composition (bilinear analogue of Elhage QK/OV, per branch Q1/K1/Q2/K2/V):
  raw Frobenius-norm composition FAILS to identify the causal edge (ranks L0H0→L1H2
  above L0H3→L1H2). Composed with the matched token's embedding (directional), BOTH key
  branches of L1H2 select L0H3 (K1 .147, K2 .110 — column maxima). Causal zeroing is
  decisive: only L0H3@src collapses L1H2's match (−.434→−.031); the other L0 heads
  leave it intact (retention table 4×4). Lesson: in bilinear attention, weight products
  reflect the circuit only in the right input direction.
- Atlas v6: output distribution as grouped bar chart (full vs ablated, top-10 tokens),
  clickable causal-map heatmap (8 heads × 40 tokens) synced to ribbon underlines,
  3-head wiring, composition + retention tables.

### Note: both induction paths share the key substrate

Retention table detail: L1H1's match weight (base +0.027) also collapses when L0H3 is
zeroed at src (→ −0.004) — the secondary path rides the SAME previous-token write as
L1H2. One substrate, two parallel readers. (block2 tiny at 20k: val 1.788 — block depth
strongly beats attn depth; differential when seeds finish.)

### Block ladder differential (bilinear-MLP depth axis, 3 seeds): a broader, mixed gate set

Mean CE: block1 2.135 → block2 1.880 (both far below all attn-only models). Gates
(block1→block2): 138,898 tokens (0.92%) — as large as the induction set — but MIXED:
31.8% induction-pattern (block2's second ATTENTION layer makes copy expressible; block1
sat at 0.45 on the jewelry example), yet only 14% overlap with the attn-ladder induction
gates and 1% with the statistics gates. The second block buys a combination the attn
ladder never shows: induction + deeper feature composition simultaneously, on largely
different tokens (block1's MLP had already absorbed most of what attn3 gates on).
Naming note: block gates now saved as gated_block2.npy (spec-length naming had clobbered
gated_depth4.npy; attn gates regenerated).

## Session 2 (2026-07-09, Logan's directive): deeper blocks, strict gates, component clusters

- block3 (6 layers: [attn,MLP]×3), 3 seeds: mean CE 1.755/1.761/1.763 (median 1.739) —
  0.12 below block2; ladder not saturating.
- STRICT differential (tau 0.25/1.5 = p>0.78 solved, 3-seed medians): block2→block3
  gates = 17,422 tokens (0.12%).
- Component fingerprints (12 heads + 3 MLPs, knockout ΔCE, 2,500 sampled gates):
  mean |ΔCE| — MLPs dominate (L1MLP 8.9, L3MLP 8.1, L5MLP 4.0 nats), heads 0.2-0.9.
  8 k-means clusters + load-bearing counts written to cfp_report_gated_block3_*.json.
- Labeling subagent launched: cluster labels, circuit hypotheses, proposed causal
  interventions (to be executed on return).

### Subagent cluster labels + first causal verdicts (block3 gates)

Labels (cluster_labels_block3.md): C0-C3 = one MLP-stack lexicon circuit split by
bottleneck ratios; C4/C5 = sibling induction circuits with SPECIALIZED layer-2 heads
(L2H0 names-across-sentence, L2H3 article+fragment nouns); C6 = candidate
agreement/inflection circuit; C7 = candidate "fuzzy retrieval with re-inflection"
(new-circuit priority).

CAUSAL TEST (C7 vs C0 control, 40 tokens each): zeroing L4H0 at the prediction position
ONLY: C7 median ΔP 0.783 — equal to zeroing it everywhere (entire effect at q);
control C0: 0.097. CONFIRMED: C7 = single-deep-head circuit acting at the final token,
8× dissociated. NOT yet confirmed: the fuzzy-lemma-retrieval semantics — L4H0's top
attention on sampled C7 tokens lands mostly at short offsets (4-6), suggesting
short-range semantic-context integration rather than long-range lemma matching
(sign-of-weight not meaningful alone — XNOR caveat applies). Next: offset distribution
over all 123 C7 tokens + OV-direction analysis + prefix-overlap test between attended
word and target word.

### C7 RESOLVED: not fuzzy retrieval — a DEEP LOCAL READ-OFF circuit

Signed per-source contribution analysis (pattern×OV toward target logit, all 123 C7
tokens): top contributor at offset median 1 (IQR 1-2), >20 tokens away only 2%, lemma-
substring overlap 11% (control 6%) → the fuzzy-lemma-retrieval hypothesis is REFUTED.
True circuit: L4H0 reads the immediately-preceding position's deeply-processed residual
(2 attn+MLP rounds of context baked in) and emits the continuation — a final
local-composition step structurally unavailable to block2 (its last attention sees only
1-MLP-deep features). Explains the "semantic" look without retrieval. Confirmed causal
profile: single head, single position, ΔP 0.78.
Method lesson (again): attention-weight eyeballing misleads; signed pattern×OV
attribution per source is the reliable readout in bilinear attention.

### Specialized induction confirmed (C5) — three circuit signatures, one instrument

Signed pattern×OV contribution profiles: C5/L2H3 median offset 32, 66%>20 tokens, 28%
source==target → genuine long-range induction specialized for article+fragment nouns.
C4/L2H0 mixed (median 3, 35%>20, 7%) — name-piece induction partially confirmed,
needs source+1 refinement. C7/L4H0 local (1, 2%, 2%) — deep read-off. At depth,
induction FRAGMENTS into token-type-specialized heads while a new deep-local circuit
family appears; the MLP lexicon stack carries the rest. Discovery cycle (task #11):
complete for 3 priority clusters; open: C6 agreement minimal-pairs, C4 refinement.

### C4 refinement: NOT classic induction — an unresolved fuzzy name-reader

Induction-wiring test (top contributor preceded by the current token, x[s-1]==x[p]):
C4/L2H0 7% (vs C5/L2H3 30% — which plus its 28% source==target makes C5 solidly
inductive; C7 1%). C4's head reads long-range name information WITHOUT exact-token
anchoring → relabeled "fuzzy name-context reader, mechanism unresolved" — the genuine
feature-matching candidate the subagent hypothesized, but on C4 rather than C7. Open
follow-ups (queued in LOG): C4 source-content analysis, C6 agreement minimal pairs,
atlas pages for the block3 circuit family.

### C6 CONFIRMED: MLP-stack agreement circuit (minimal pairs)

Subject-number flip pairs (identical local suffix "...and run"): block3 mean P("s")
0.433 singular vs 0.027 plural (gap +0.41; strong pairs flip 0.85→0.02); block2 weaker
and unreliable (+0.22, one pair no-flip). Zeroing ANY of L1/L3/L5 MLP destroys the gap
(L3MLP → 0.00). Agreement is carried by the MLP stack jointly — a genuine syntax
capability beyond n-gram statistics, strengthened by the third block. (Caveat: 6
crafted templates, 3-4 informative; a fuller template battery would firm the rate.)

BLOCK3 CIRCUIT FAMILY — final table:
| cluster | circuit | verdict |
|---|---|---|
| C0-C3 | MLP lexicon stack (order-2/3 + re-tokenization) | one circuit, k-means split |
| C4 | fuzzy name-context reader (L2H0, long-range, no token anchor) | UNRESOLVED mechanism |
| C5 | specialized induction: article+fragment nouns (L2H3) | confirmed (30% match+1, 28% src==tgt) |
| C6 | agreement/inflection via MLP stack | confirmed (minimal pairs) |
| C7 | deep local read-off (L4H0 reads 2-block-deep prev position) | confirmed; retrieval refuted |

### C4 mechanism (partial): fuzzy ENTITY re-mention retrieval

Decoded top-contributor sources on long-range C4 cases: ~half read the earlier mention
of the entity being completed — names ("A→l" ← "Jimmy and Alice", off 41) AND objects
("the m→ic" ← "saw a microp[hone]", off 36) — with NO exact-token anchoring (7% match+1).
So L2H0 implements feature-anchored (fuzzy) entity retrieval, the capability the
subagent originally hypothesized — living in C4, evidenced by source decoding, but
noisier than C5's token-anchored induction (other half of sources opaque). Status:
mechanism identified at the family level; per-example attribution remains partly
distributed. block4 ×3 seeds training (ladder continues per Logan's directive).

### PAUSE POINT (Logan, 2026-07-09): discovery loop paused; pivot to mechanism decomposition

block4 (8 layers, 3 seeds): median CE 1.693; strict gates block3→block4 = 5,743 tokens
(3× shrink from block3's 17,422 — ladder approaching saturation). Component fingerprints
computed (20 components; cfp_report_gated_block4_*.json) but cluster labeling NOT run —
program paused here. Discovered models/circuits become toy testbeds for the
weight-activation SAE method (mechanism_decomposition_spec.md).
