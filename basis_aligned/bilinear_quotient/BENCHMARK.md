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
constant mean (0 free params beyond d) → rank-r linear (2dr) → diagonal/scale →
rank-r quadratic in a declared basis → full component. §155's measured reference:
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
  for train/test split at the *model* level.
- The canary regression script for scoring-environment stability.
