# Experiment log: multi-family graph tracing

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

### ANSWER: why geometric neighbors end up nearby (all links measured)

1. The state at node v must carry the model's prediction — positive neighbor-token evidence
   in the unembed basis (+nbrU in the o2 write of all 18 models; functionally forced).
2. That evidence alone IS the positive Park map: adjacent nodes' predictions overlap
   through each other (u's evidence contains v and vice versa), so states of neighbors
   align. o2 self-terms are positive even in anti models; removing own-token content moves
   everyone toward positive.
3. States also carry own/recent-token content whose sign is behaviorally underdetermined:
   "don't predict what can't follow" can be implemented as suppression **in the writes**
   (−ownU) or **in the static readout** (the E–U diagonal), with the same logits.
4. The total geometry = coupling of that own-token content with the neighbor evidence:
   +own → neighbors nearby; −own (suppression-in-writes) → anti-map.
5. Training data pins the choice through **reversibility**: walks that can never revisit
   their recent past force adaptive suppression of recent tokens (→ anti); any nonzero
   backtrack rate makes the recent past part of the prediction (→ positive). Entropy is
   irrelevant. Single-family training leaves the sign underdetermined (seed lottery);
   softmax induction stacks default to write-side suppression regardless of data.

Goal: the single-task 2-layer models learned *task-specific* circuits (cycle model: "never
predict the last ~3 tokens" elimination; grid model: backtrack-boost baseline + two copy
routes) and organize in-context representations opposite to Park et al.'s LLMs. Question:
does training **one model on many graph families at once** force a more general relational
circuit — and Park-style neighbors-together representations?

## Hypotheses (stated before running)

- **H1 (organization).** Multi-family training flips the Gram–adjacency correlation of
  windowed mean reps positive (Park-like) for architectures that anti-organized when
  single-task trained. *Test:* Park protocol on grid docs for the multi-task
  bilinear-lerp-2L vs the single-task grid_L2_d128_long (corr −0.57). *Falsified if* the
  multi-task model still shows corr ≤ −0.3 at ctx 256. *Refined if* it flips for some
  architectures only.
- **H2 (shortcut removal).** The task-specific shortcuts conflict across families and are
  abandoned. Concretely: (a) the "suppress last 3 tokens" hack is *wrong* for undirected
  rings (backtracking is legal there), so on directed-ring docs the multi-task model's
  layer-1 offset profile should differ from the cycle model's, and no-L2 accuracy at L=5
  should be ≪ 0.88; (b) the backtrack-boost is *wrong* for directed rings, so on grid docs
  no-L2 legal rate should be ≪ 0.84. *Falsified if* the same shortcut signatures reappear
  with similar ablation numbers.
- **H3 (generalist competence).** One model reaches near single-task performance on every
  train family AND transfers zero-shot to held-out families (torus, Erdős–Rényi) and sizes.
  *Benchmark:* single-task grid model transferred to torus at 0.99 legal. *Falsified if*
  multi-task torus legal < 0.9 or any train family lags its single-task ceiling by > 0.1
  legal at matched steps.
- **H4 (architecture).** Based on single-task results: softmax ≥ bilinear on performance;
  positive organization most likely for bilinear-3L+add and early-context softmax.

## Design

- Data (`graphs.py`): uniform random walks on token-labeled graphs, one graph/doc,
  n_ctx 256, vocab 100. Train families: ring (5–20), **directed ring** (≡ cycle task, 5–20),
  grid, cylinder (3×3–4×5), random tree (8–16), random 3-regular (10–16). Held out
  ENTIRELY: torus, ER graphs. Held-out sizes: ring 30, dring 27, grid 6×6, tree 24, kreg 24.
  Structure pools pre-sampled; labelings + walks fresh per batch.
- Note the mixture is *self-disambiguating in context*: e.g. ring vs directed ring docs are
  distinguished only by whether backtracks ever occur.
- Archs (all d=128, 1 head): bilinear-lerp-2L (the original recipe), bilinear-add-2L,
  bilinear-add-3L, softmax-lerp-2L, softmax-add-3L; 24k steps, batch 128, Adam 1e-3 cosine.
- Metrics: legal rate + neighbor mass on tail positions (≥128), per family. For directed
  ring, mass = probability on the unique correct token.
- Then: Park-protocol rep analysis per family × arch; mech battery (offset profiles, class
  stats, ablations) on the multi-task bilinear-lerp-2L vs its single-task counterparts.

## Log

- [setup] Wrote `graphs.py`. Pool sizes: tree 360, kreg 360, ER 180 structures; lattice/ring
  families enumerate their few structures. Next: validate data generation (walk validity,
  legal-mask correctness, family statistics), then a 600-step pilot for learnability/speed.
- [validation] All families pass: every walk transition is an edge, next token always in the
  legal mask, legal-count == degree. `train_batch(126)` = 11 ms (not a bottleneck).
- [pilot] bilin-lerp-2L, 600 steps @ 71 it/s: grid legal 0.02→0.49, tree →0.53, torus
  (never trained) →0.42 already. Learnable; 24k steps ≈ 6 min/arch.
- [launched] Full sweep: 5 archs × 24k steps (bilin-lerp-2L, bilin-add-2L, bilin-add-3L,
  softmax-lerp-2L, softmax-add-3L), all d=128 · 1 head. ETA ~35 min.
  While training: writing rep-organization and performance analysis (`analysis_general.py`).
- [interim, arch 1/5 @ 10k steps] bilin-lerp-2L: grid legal 0.99, torus (unseen family) 0.98,
  loss 1.71. Early support for H3 — and notably *faster* than the single-task grid model
  (which needed 20k steps for 0.99). Mixture training appears to help, not hurt.

### Sweep results (24k steps, legal rate; figures gen_perf/gen_training)

- **bilin-lerp-2L: 0.99–1.00 on ALL sets** incl. torus 0.99, ER 0.99, OOD sizes 0.94–1.00.
- **softmax-add-3L: 1.00 everywhere** (mass 0.75–0.96).
- bilin-add-2L: flat ~0.75 (loss plateau 2.4); bilin-add-3L: **diverged** (loss ~1e20 — the
  additive fix that worked single-task at 8k explodes on the harder mixture; the lerp's
  halving was accidentally stabilizing); softmax-lerp-2L: 0.83–0.93 on stochastic families
  but **fails deterministic dring (0.10)** despite its single-task twin solving cycles.

### Hypothesis verdicts

- **H1 (organization) — CONFIRMED for the target arch, falsified as a universal rule.**
  Multi-family bilin-lerp-2L on grid docs: corr **+0.66** stable across context (single-task
  same arch: −0.57). Its top-PC map is Park-like (visibly local edges), on torus (+0.68,
  never trained) too, and its cycle representation is a clean phase heptagon. BUT
  softmax-add-3L, the other perfect model, is the strongest anti-organizer measured
  (grid −0.80, torus −0.84), and its cycle representation is a **7-pointed star** (adjacent
  phases antipodal). Organization sign is decoupled from competence. softmax-lerp-2L
  repeats its organize-early-flip-late trajectory (+0.73 → −0.49).
  Controls launched: seed-1 replication + grid-only training through the identical pipeline
  (prediction: grid-only anti-organizes; if it also organizes positively, the flip is the
  pipeline, not the mixture).
- **H2 (shortcut removal) — CONFIRMED (a), NUANCED (b).** (a) The cycle elimination
  shortcut is gone: no-L2 accuracy at L=5 was 0.88 single-task, **0.00** multi. (b) The
  grid backtrack baseline survives (no-L2 legal 0.89) — as expected, since backtracking is
  legal in most families; only the elimination hack (harmful on undirected rings) died.
  Circuit rewiring beyond the predictions: K-composition became all-or-nothing (cutting it:
  0.00 on cycles); **V-composition flipped from load-bearing to unnecessary** (cutting it
  barely hurts grid 0.91/0.80 and *improves* cycle accuracy at L≥10); L2 attention is now
  positive at induction offsets on cycle docs (was negative). One unified relational
  circuit: L1 = short negative prev-window; L2 = positive content-match retrieval +
  raw-token copy.
- **H3 (generalist competence) — CONFIRMED, exceeded.** No train family lags; zero-shot
  torus/ER at 0.99–1.00; the multi model *beats* the grid specialist on grid docs
  (legal 0.995 vs 0.989, mass 0.811 vs 0.762). Two weak spots: dring at unseen n=27 is
  0.00 for the failing archs (bilin-add-2L, softmax-lerp-2L) while champions hit 1.00.
- **H4 (architecture) — partially confirmed:** softmax-add-3L is the strongest performer,
  but "positive organization most likely for 3L-add / early softmax" was wrong at depth:
  the positive organizer here is bilin-lerp-2L; softmax-add-3L anti-organizes hard.

### Controls (H1 validation)

- **Seed-1 replication of bilin-lerp-2L**: performance again ~1.00 everywhere; grid corr
  **+0.60 → +0.55** across context. The positive flip replicates.
- **Grid-only control** (identical graphs.py pipeline, mixture removed, same steps): grid
  corr +0.00 → **−0.14** — non-positive, trending negative. The organization flip is caused
  by the family mixture, not the new pipeline. (Weaker anti than the old geodata specialist's
  −0.57; direction consistent.) Side finding: even grid-only training transfers zero-shot to
  torus/ER at 0.98 — broad transfer is nearly free; the mixture's distinctive effects are the
  deterministic family (gridonly dring-27: 0.63) and the representation flip.
- **softmax-add-3L seed-1**: 1.00 legal on all 13 sets again; grid corr +0.36 (ctx 8) →
  **−0.67** (ctx 256). Anti-organization at long context replicates; the *early-context*
  trajectory is seed-dependent (seed 0 was negative throughout, seed 1 organizes mildly
  then flips, like softmax-lerp). The competence⊥organization decoupling stands.

### H5 (pre-registered): which ingredient flips the organization?

Candidates: (a) *conflict* — one family that punishes the anti-circuit's shortcuts (dring
punishes backtrack/recency heuristics) suffices; (b) *diversity* — many families needed
regardless of conflict. Test: train bilin-lerp-2L on two-family mixtures grid+dring,
grid+ring, grid+tree, grid+cylinder (identical pipeline/steps), measure grid corr at ctx 256.
*Predictions:* conflict-driven ⇒ grid+dring flips positive, grid+cylinder (structural cousin,
minimal new demands) stays ≤ 0. Diversity-driven ⇒ all two-family mixtures stay ≤ 0.

### H5 results — prediction FALSIFIED, mechanism refined

Two-family mixtures, bilin-lerp-2L, grid corr at ctx 8 → 256:

| mixture | corr | verdict |
|---|---|---|
| grid + dring | −0.29 → **−0.55** | the "conflict" family *entrenches* the anti-map |
| grid + ring | +0.31 → **+0.38** | flips positive |
| grid + tree | +0.43 → **+0.41** | flips positive |
| grid + cylinder | +0.71 → **+0.24** | positive, decaying (partial) |

The conflict hypothesis had it backwards: pairing with the deterministic directed ring
*preserves* anti-organization, while any second *stochastic undirected* family flips it
(even the structural cousin cylinder, partially; grid alone was −0.14). Refined picture:
**organization sign tracks the algorithmic mode** — deterministic token-copy/induction
circuits anti-organize; diverse stochastic families force "predict the neighborhood as a
set", which builds positive local geometry. The full 6-family mixture (dring included) is
strongly positive (+0.66): enough stochastic diversity overrides dring's pull.

### Mech on the anti-organizing champion (softmax-add-3L)

- L1 = textbook previous-token head (0.97 attention mass at offset 1 on both doc types).
- L2 = induction head (cycle attention at kL−1 successor slots; kills cycles when removed).
- **L3 nearly idle**: knockout keeps cycles at 0.94–0.97 and grid legal at 1.00 — and
  *raises* grid mass (0.895 vs 0.851), echoing the Q-composition calibration paradox.
- So the perfect softmax model is a classic 2-layer induction stack — and it anti-organizes
  even when trained on the full stochastic mixture. Circuit type, not data alone, sets the
  sign. Caveat: both circuit types use K-composition matching, so "retrieval needs separable
  neighbors" does not by itself explain the sign difference — open question.

### Seed batteries — MAJOR REVISION of H1 (user's skepticism vindicated)

grid corr @ ctx 256, bilin-lerp-2L:

| condition | seeds | verdict |
|---|---|---|
| grid only | **−0.14, +0.67, −0.08, +0.16** | sign is a seed lottery (matches user's "prolly just init") |
| full mixture | **+0.66, +0.55, +0.62** | consistently positive, tight |
| softmax-add-3L mixture | −0.80, −0.67 | consistently anti |

Restated H1: multi-family training does not flip a determined sign — it **stabilizes the
positive mode that single-family training reaches only occasionally**. The old single-task
geodata grid model (−0.57) was one draw from a high-variance lottery. Consequence: all
single-seed two-family H5 numbers are unreliable; grid+dring re-running with seeds 2–3.
What survives regardless: (i) the mixture models have *genuine Park geometry* (Theorem-5.1
test: PC1↔z2 0.81, PC2↔z3 0.80, top-2-PC energy 0.49 vs ~2 random; lattice moved from PC12
to PC1) — a spectrum-position difference no sign/gauge symmetry produces; (ii) the
architecture effect (softmax induction stack consistently anti at equal competence).

### Final verdict summary

| hypothesis | verdict |
|---|---|
| H1 mixture flips organization positive | Confirmed for bilin-lerp-2L (+0.66; seed-replicated; grid-only control stays ≤ 0 ⇒ mixture is causal). NOT universal: softmax-add-3L anti-organizes (−0.80 / −0.67 across seeds) at equal competence |
| H2 shortcuts die | Elimination hack dead (0.88 → 0.00); backtrack baseline survives (legal in most families); circuit unified: K-comp all-or-nothing, V-comp abandoned, attention sign positive |
| H3 generalist competence | Confirmed+: two archs ~1.00 everywhere incl. zero-shot torus/ER; multi beats the grid specialist on grid. But most of the raw zero-shot transfer comes free even from grid-only training |
| H4 architecture | softmax-add-3L strongest (perfect); recipe fragility extreme: bilin-add-2L plateaus, bilin-add-3L diverges, softmax-lerp fails only the deterministic family |
