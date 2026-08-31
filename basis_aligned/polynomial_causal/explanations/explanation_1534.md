# Plain-English update — 2026-08-31 15:34Z

(Yardstick: damage = extra prediction error above the real model; LOWER IS BETTER.)

## The circuits program, one afternoon in
Following the operator's directive, we measured the 62 known circuits three ways:
1. **How big is a circuit?** Typically FIVE components (of 16 measured): knocking out just the top one gives
   only ~40% of the damage; you need five to reach 90%. Not minimal labels, not everything-everywhere —
   mid-sized mechanisms on a heavily shared base.
2. **What's the right surgical tool?** Swapping a component's output between examples WITHIN a circuit's own
   positions ("counterfactual interchange") is ~250x more selective than deleting the component. Ablation
   was the wrong knife all along.
3. **Where does a circuit live inside a component?** Not in the full 1152-dim output: a 32-dimensional
   subspace (found by plain PCA) carries about half the causal effect, monotonically in rank. These are the
   "counterfactual circuits with small rank" the operator asked about — first versions found by the cheap
   method; the learned-search version (DAS) is running after one optimizer bug was caught and fixed.

## Running right now
- Does LEARNING beat PCA at finding these subspaces? (v2, with the optimizer sanity-checked)
- Is the subspace NECESSARY (does the rest of the space carry little)?
- Is it SHARED — one subspace serving all 16 circuits of the a8 family? If yes, the repertoire compresses
  from 62 subspaces to a handful of causal variables — the best possible shape for downstream use.

Everything accumulates into circuits/REPERTOIRE.json: per circuit — components, refs, minimal set size,
removal profile, which simplifications preserve it, and (soon) its low-rank carrier.
