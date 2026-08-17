# A two-track interpretability benchmark grounded in the bilin18 program

Draft spec (2026-08-17), from the user's proposal: turn the program's instruments
into a benchmark with a semantic track and a structural track, scored by an
explicit fidelity-vs-simplicity trade-off that is not quantization.

## Track 1: Semantic (explanations, scored causally)

Unit of submission: a natural-language explanation of a component (a layer, a
head, a span of a layer's output, a functional).

Scoring, in increasing order of stringency:
1. **Correlational (Bills-et-al style):** an evaluator (LLM simulator or probe)
   predicts the component's activation on held-out text from the explanation
   alone; score = correlation vs truth, minus the same evaluator's score on a
   shuffled explanation (the null this program always required).
2. **Causal (the program's contribution):** the explanation must predict
   *intervention outcomes* — given the explanation, the evaluator predicts which
   held-out tokens' loss moves (and in which direction) when the component is
   mean-ablated. Score = rank correlation with the measured per-token deltas.
   Motivation: this program found correlational names verify while causal token
   stories fail (§§66–81); an explanation that only matches activations has not
   explained the computation.
3. **Guards** (from the corrections ledger): per-token difficulty splits are NOT
   valid evidence (flattening relief and gain amplification both fake them,
   §§140–142); loud-component fidelity must be offered gain-frozen (§§116–117).

## Track 2: Structural (decomposition under a fidelity-complexity Pareto)

Unit of submission: a **replacement program** for the model — per component, a
stand-in from a typed ladder, plus a claimed interaction-edge list.

Stand-in ladder (each rung has an explicit parameter count):
constant mean (0 free params beyond d) → rank-r REFIT linear (2dr; sequential
refit is the frontier lever) → full component. The compact-quadratic rung is
measured DEAD in this family (§160: −0.009/+0.000 over linear at L16/L9 —
the nonlinear residues are diffuse in every compact basis, §113), so submissions
claiming quadratic rungs must beat that null. §155's measured reference:
bilin18's L9 = constant (+0.031 nats), L16 = rank-8 linear (+0.059),
L1 = irreducibly quadratic (full linear still +0.29).

Scoring:
- **Fidelity:** held-out ΔCE of the *jointly installed* replacement (never the
  sum of per-component costs — composition drifts superadditively, §104), with a
  gain-frozen variant reported alongside (§116). Interaction fidelity: the
  replacement must reproduce measured interaction excesses within a tolerance
  (the composition law's grid, §123).
- **Complexity:** total parameter count of all stand-ins, computed at the
  **balanced gauge point** (balanced_gauge_spec.md) so it cannot be gamed by
  per-unit rescalings. Not bits/quantization — structural rank and sparsity.
- **Metric:** the (log params, ΔCE) Pareto curve; headline numbers = ΔCE at
  fixed budgets (1%, 5%, 20% of original params). Measured reference points
  (§§157-158): naive assignment (4 constants + 8 rank-8) = +2.68 at 0.15M;
  sequentially-REFIT assignment (rank-16) = +1.66 at 0.29M; all-full-linear
  same layers = +1.26 at 15.9M. Sequential refit is the frontier lever (36%
  cost reduction free); the traced refit curve is FLAT — rank 4/16/64 give
  +1.81/+1.66/+1.54, so the competitive region is below 0.1M params and closing
  the last ~1.3 nats needs a different computation class, not more rank;
  per-layer ladder in §155. Note: with the λ-mixing instrument fixed, naive
  joint composition drift is modest (+1.26 vs the sum ~+0.6) — score jointly
  anyway, but the drift penalty is real, not catastrophic.
- **Edges:** claimed interaction graph scored against transplant-measured edge
  strengths (the program's instruments: full-write transplants, dilution
  shares, span patches).

Measured floors (§162): base-loss predicts fingerprints at only 0.13 median
|Spearman| (publish beside every score; >0.2 = real signal); position 0.01.

Ground-truth asset: `bilin18_fingerprints.pt` — per-token ablation deltas for
12 components on a fixed held-out set (fingerprints mutually near-orthogonal,
pairwise Spearman 0.04; deterministic, so scoring runs on this fixed set and
cross-set generalization is the explanation's burden).

## Graph formalization (nodes, edges, and amortized semantics)

Responding to the design question of node complexity and compounding semantic
work. A submission is a GRAPH, and each of its three cost axes is explicit:

- **Node** = a component with a declared INTERFACE: the input variables it reads
  (a subspace/watch-list) and the output variables it writes. Node complexity =
  interface size (dims in + dims out) plus the stand-in class parameters needed
  to reproduce its input-output behavior. Internal wiring is free — a node is
  simple if its *behavior* needs few variables, however it computes them. (The
  program's measurements say interfaces are naturally small here: watch-lists
  ~8-dim, score filters ~5-dim — but §133 warns that any single 8-dim channel
  carries little causal load alone; the fidelity axis keeps submissions honest
  about that.)
- **Edge** = a declared dependency: node B reads variables node A writes. Edge
  claims are verified by the program's cut instruments (project A's write out of
  B's watch-list; §§131–134); *unclaimed* edges must be verified inert the same
  way. Edge complexity = the DESCRIPTION LENGTH OF THE DEPENDENCY'S FUNCTIONAL
  FORM, not the wire count: the algebra is {k specific coordinates | a summary
  statistic (norm, mean, share) over a declared set | a low-rank map | opaque}.
  A dense connection that only transmits a norm is one variable wide and priced
  accordingly. All three non-opaque types are measured in this model: L17's
  span dependence is 82% energy->gain (summary-typed, §116); generic tail edges
  are share-of-stream typed (§93); the L5->L6 cargo edge is coordinate-typed
  (~8 dims, §170). Typed-edge claims are verified both ways: cut the claimed
  channel (effect should vanish) and keep ONLY the claimed channel (effect
  should survive) — the §117 gain-frozen instrument is the reference
  implementation for norm-typed claims. The measured macro-graph
  here is the relay (MLP → attention-transport → MLP), so sparse type-level
  graphs are achievable.
- **Semantic cost with amortization** = total description length of the
  explanations, where referencing an already-defined variable is nearly free:
  explaining node B as "gates A's x-feature by recency" pays only for its
  *marginal* content. This is exactly the compounding the design wants — sparse
  edges + small interfaces make each successive node's explanation short, so a
  good decomposition *reduces* total semantic work superlinearly. Score the
  composed explanations causally (fingerprint prediction) at the graph level,
  so credit flows only through explanations that actually predict interventions.

The trade surfaces honestly: tiny interfaces are cheap on the complexity axis
and weak on the fidelity axis (within-type diffuseness); the Pareto rewards
finding the interface size where understanding actually lives.

## Why this is not quantization

Quantization preserves the computation and shrinks the numbers; this benchmark
rewards replacing computation with *simpler computation classes* (constant,
linear, low-rank quadratic) and knowing *where* each class suffices. The score
moves only when structural understanding does: a submission that knows L9 is
functionally constant but L1 is irreducibly quadratic beats one that quantizes
both.

## Assets this program already provides

- Ground-truth reference Pareto (§155 ladder; §105 refit pipe).
- Instrument suite with known failure modes and their controls (twelve ledger
  corrections — each one is a trap a naive benchmark would fall into).
- A second model (bilin12) with verified transfer of the structural laws (§154)
  AND of token-level causal responsibility (§163: analogous components'
  fingerprints correlate 0.34 cross-model vs 0.05 non-analog) — the model-level
  train/test split, with `bilin12_fingerprints.pt` as the held-out target.
- The canary regression script for scoring-environment stability.
