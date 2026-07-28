# Lesson-plan proposal: Tensor networks for understanding neural computation

*Proposal for Logan, 2026-07-28. Draft to react to — not the lessons themselves. Language setting.*

## 0. The one idea that unifies everything

**A trained network is a tensor network.** Every layer is a multilinear map (our bilinear
attention and bilinear MLPs are *exactly* multilinear — no softmax, so no gauge we can't remove).
Drawn as a diagram, the whole model is **nodes** (the tensors that do the computing) wired
together by **bonds** (the edges that carry information between them).

That picture makes your two "meaningful things" the two axes of the same object:

- **Thing 1 — breaking down the models = simplifying the NODES.** Can we rewrite one tensor in a
  small, privileged, *sparse* basis so it becomes a few interpretable pieces instead of a dense
  blob of weights?
- **Thing 2 — sparse communication = thinning the BONDS.** How much information actually flows on
  the wire between two components? A fat bond means dense communication; a thin bond means the two
  modules talk through a narrow, sparse code.

And the extremes you want to illustrate fall right out of this:

| | Node is **low-rank / decomposable** | Node is **dense / full-rank** |
|---|---|---|
| **Bond is thin (sparse comm.)** | the easy, clean case | **dense-but-sparsely-communicating** (Thing 2's headline) |
| **Bond is fat (dense comm.)** | rare — usually simplifiable | the genuinely hard, opaque case |

The whole course is: teach people to look at a network and ask, node by node and bond by bond,
*"is this thing decomposable, and is this channel sparse?"* — and to know what each answer means.

## 1. What we've done, organized by the two things

Everything in the language program is one of these two moves (or the honest failure of one):

**Thing 1 — breaking down nodes**
- The **exact fold**: bilinear layers fold to polynomial interaction tensors (embedding folded in,
  gate error ~1e-7). The weights, once folded, live in privileged input/output bases.
- The **gauge lesson**: the neuron basis is an arbitrary CP factorization; two different neuron
  bases give the identical observable tensor. So "neuron X does Y" is often reading noise.
- The **right basis**: a sparse overcomplete **dictionary under the weight-induced metric**, not the
  neuron basis. Atoms turn out semantic (topics + morphology suffixes) and reusable (**archetypes**:
  {the}, {a/an}, punctuation families).
- **Minimal circuits & honest limits**: behavioral rank (a "192-neuron" MLP is really rank ~a few
  dozen); induction head H5; the mlp16 register gains that are *exact weight quadratic forms* and
  decode to document register. And the honest counter-result: readout-sparsity ≠ causal-sparsity —
  some nodes are genuinely redundant and won't break down.

**Thing 2 — thinning bonds**
- **Windowed-D**: every residual *read* is the embedding stream + cheap token-keyed tables for old
  context + a small live window — i.e. layers communicate mostly a **sparse, token-identity code**,
  not the full residual.
- **Manifold collapse**: the block-0 MLP is *dense inside* (uses ~all its neurons) but its output,
  as read by the next attention, **collapses to effective rank ~10** — a dense node with a thin bond.
  This is the cleanest real example of your Thing-2 extreme.
- **Typed opaque blobs**: the memory pipeline moves a first-mention entity through the stack as a
  payload that later layers route without unpacking — a narrow, typed channel over a dense store.
- **TN-gauge / code-propagation**: the toy program that asks whether a shared sparse dictionary can
  serve as the bond between layers (result: the residual bond is pinned by the boundaries; naive
  shared-Φ is too lossy end-to-end) — the direct study of *how thin can the bond get*.

## 2. Proposed lesson sequence

Each lesson: **one concept → a toy example you can see → the image → the real LLM instance.**

**Lesson 0 — A network is a tensor network.** The diagrammatic language: nodes, bonds, contraction.
- *Toy:* a single bilinear attention head on a 6-token cycle (we already have these smallest models).
- *Image:* the model as a tensor diagram; hover a node to see its tensor.
- *Real:* bilin18 drawn as its diagram, with the pieces we've named highlighted.

**Lesson 1 — The exact fold.** A bilinear layer *is* a polynomial; fold the embedding in and read the
interaction tensor directly.
- *Toy:* a 2-feature bilinear map computing a soft-AND; show `out = xᵀ T x` and the 2×2×2 tensor `T`.
- *Image:* weights (opaque matrices) → one small tensor heatmap; the "fold" as an animation.
- *Real:* the QK head as a mixed third moment `C = Σ_j k1(x)k2(x)⊗W_o v`; verify fold error ~0.

**Lesson 2 — The gauge trap.** The neuron basis is arbitrary; only the tensor is real.
- *Toy:* one tensor, two different neuron factorizations (relabel/rotate the hidden units), identical
  output; then the antisymmetric part that cancels behaviorally.
- *Image:* same output, two totally different neuron-activation heatmaps, side by side; a slider that
  morphs one basis into the other while the tensor stays fixed.
- *Real:* permuting the bilinear-MLP neurons leaves the folded tensor identical (our §30 check).

**Lesson 3 — The right basis (metric-aligned + sparse).** Fit a sparse dictionary in the geometry the
weights actually read, not the neuron basis.
- *Toy:* a computation that's *dense in neurons* but *one sparse atom* in the right basis; show the
  dense neuron heatmap collapsing to a sparse code.
- *Image:* dense heatmap → sparse bars; the metric as a whitening of the space.
- *Real:* the layer-0 dictionary atoms (topics + suffix morphology), and archetypes reused across
  positions.

**Lesson 4 — Minimal circuits and the honest limit.** How few pieces reproduce the behavior — and
when the answer is "no few pieces do."
- *Toy:* a **low-rank** map (rank-2, breaks down to 2 features) vs a **redundant** map (full-rank,
  random directions work as well as chosen ones — won't break down).
- *Image:* two rank-vs-fidelity frontier curves — one that plateaus at low rank, one that climbs
  linearly; the random-vs-ranked control as the diagnostic.
- *Real:* mlp16 causally sufficient at rank-16 (breaks down) vs a redundant block where our own
  baselines showed random ≈ chosen (doesn't).

**Lesson 5 — Communication channels (bond dimension).** Zoom out: information between two nodes lives
on a bond, and the bond has a width.
- *Toy:* two composed bilinear layers; sweep the bond (intermediate) dimension and watch behavior —
  find the width below which it breaks.
- *Image:* the tensor-network wire with a dial for bond dimension; a curve of fidelity vs width.
- *Real:* the windowed-D result — layers need token identity on the wire but almost no live context.

**Lesson 6 — Sparse codes and typed blobs on the wire.** The bond can be *narrow* (few dimensions) or
*sparse* (few active symbols), and it can carry an *opaque typed payload*.
- *Toy:* a two-module system where the wire carries only k active symbols from a shared dictionary;
  contrast with a payload the downstream routes without unpacking.
- *Image:* a sparse code lighting up a few atoms on the wire; a "sealed envelope" payload icon.
- *Real:* the memory pipeline's typed blobs; the shared dictionary reused across layers.

**Lesson 7 — The two extremes, side by side (the payoff).** Put a decomposable node next to a
dense-but-sparsely-communicating node and show how differently you understand each.
- *Toy:* Extreme A = a clean low-rank node with interpretable atoms. Extreme B = a dense associative
  lookup (full-rank, resists decomposition) that nonetheless emits a rank-10 signal downstream.
- *Image:* the money figure — two nodes, one transparent-and-decomposed, one opaque-with-a-thin-wire,
  annotated with what you *can* and *can't* say about each.
- *Real:* mlp16 (Extreme A) vs the block-0 MLP → attention interface / manifold collapse (Extreme B).

## 3. Toy models to build (2 total, reused throughout)

1. **The cycle head** — a single bilinear attention head trained to move around a small cycle/grid
   graph (we have these). Its interaction tensor is genuinely low-rank and geometric, so it is the
   running example for *decomposable nodes* (Lessons 0–4).
2. **The two-layer bond model** — two composed bilinear layers on a planted task where the true
   intermediate code is k-sparse. It's the running example for *bonds* (Lessons 5–7): we can dial the
   bond width and show the decomposable-vs-dense-but-sparse-comm extremes on one system.

Both are small enough to train in minutes and to render exactly.

## 4. Delivery format

Interactive, self-contained HTML artifacts (SVG/Canvas), one per lesson, in a shared visual system:
- tensor **diagrams** you can hover;
- **heatmaps** for weights → tensors → sparse codes;
- **sliders/dials** for the fold animation, the gauge morph, and the bond-width sweep;
- **frontier plots** for rank-vs-fidelity and the random-vs-ranked control.
Theme-aware, no external deps. The toy panels can embed real computed numbers from the two toy models.

## 5. What I'd want your steer on before building

1. **Audience/level** — interpretability researchers (assume linear algebra, tensor diagrams), or
   broader (build up the diagram language slowly)? It changes Lesson 0's depth.
2. **Scope** — all 8 lessons, or start with a tight 3-lesson core (fold → right basis → the two
   extremes) and expand?
3. **The two-things names** — I've used "breaking down nodes" / "thinning bonds"; if you have
   preferred terms (decomposition vs communication? substrate vs channel?) I'll standardize on them.
4. **First build** — I'd suggest I build **Lesson 7 (the two extremes)** first as a single figure,
   since it's the thesis and will tell us if the visual language works before we do all eight.

Once you react to this, the natural next step is: train the two toy models, then build Lesson 7's
figure as the proof-of-concept, then the rest.
