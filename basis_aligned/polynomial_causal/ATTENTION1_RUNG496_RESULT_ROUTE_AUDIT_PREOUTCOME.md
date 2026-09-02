# Pre-outcome audit of rung496's pass and null routes

**Written:** 2026-09-02 17:31 UTC

**Status:** CPU reasoning completed while rung496 is live. This file does not inspect a rung496 result, change its
registration, or authorize a scientific claim. Its purpose is to prevent either result branch from turning into an
ambiguous intervention or another rank/reconstruction sweep.

## The candidate produced by rung496

For head `h`, MLP0 branch `b`, and factor `i` in `{Q1,K1,Q2,K2,V}`, rung496 produces three complete residual-write
changes:

- `phi[h,b,i]`, the Shapley allocation of every finite five-factor interaction involving `i`;
- `first[h,b,i]`, changing only `i` while all four partners remain branch-absent; and
- `last[h,b,i]`, changing `i` after all four partners are already normal.

These are gauge-invariant **output contributions** of one factor in specified partner backgrounds. They are not the
private 128-dimensional Q or K vectors themselves. A downstream-use match therefore proposes that later circuits
treat two factor-attributed output changes similarly. It does not yet supply a map between the two heads' private
Q/K coordinate systems.

## Pass-route audit: what can and cannot be interchanged

A direct command such as “put head `p`'s Q vector into head `q`” is undefined. Each head has a private coordinate
gauge: `Q -> GQ` and `K -> G^{-T}K` leaves its score unchanged. Coordinates from two heads cannot be crossed without
a separately defined transport map.

Two finite tests are legal, but they answer different questions:

1. **Allocated-output interchange.** Replace the selected factor's complete raw-write contribution with the matched
   contribution at the raw attention1-write site, then run the real normalization and suffix. This tests whether the
   two rung496 objects are finite downstream substitutes. It advances operational grouping and manipulation, but it
   must be named as output-contribution interchange rather than Q/K-vector interchange.
2. **Whole-score interchange.** Replace one head's scalar `score1[i,j]` or `score2[i,j]` array with the other head's
   scalar score array while retaining the recipient's other score, value, and output map. This is gauge-invariant and
   tests reusable attention patterns, but it returns to a whole score and cannot by itself prove that only Q or only
   K is shared.

A genuine Q-side or K-side interchange needs a transport map fixed without using final outcomes. One possible later
construction is to fit a map using the factor's scalar contractions against a frozen bank of natural partner keys or
queries, then require the map to predict unseen partners, documents, and circuits. That map needs its own
identifiability, held-out, and control analysis; it cannot be invented after seeing a rung496 pair.

Therefore a rung496 pass should first preregister bidirectional allocated-output interchange with complete suffix
recomputation, position and noncandidate controls, target preservation, unrelated-circuit preservation, and a clear
scope label. Calling that test “input-side interchange” would overstate what is manipulated. Only after it succeeds
should the private-coordinate transport problem be opened.

## Null-route audit: do not repeat older predictive-state and rank tests

Several older routes already showed why a generic “predictive-state quotient” is underspecified:

- clustering raw MLP0 token states did not establish interchangeability across consumers and backgrounds;
- local Jacobian/Fisher or balanced-truncation directions can miss large finite interactions;
- a delimiter predictive-state representation failed live shuffle controls; and
- there is no registered string-concatenation algebra that would make an arbitrary response matrix a weighted-
  automaton Hankel matrix.

The null successor must therefore not fit another low-rank response basis, cluster a single 62-number fingerprint,
or reshape results into a Hankel matrix. Those would repeat closed screens and would not advance circuit extraction.

## A narrower intervention-conditional causal quotient

Use actual candidate computations as states and actual legal interventions as actions. For state candidate `s`,
action `a`, downstream circuit `c`, and document `x`, record the signed finite response

`R(s,a,c,x) = circuit_effect after applying action a to candidate s`.

Examples of candidate states are the exact MLP0 T/C/I/S contributions, exact attention1 factor-attributed writes,
and named downstream equality corrections. The first action alphabet should contain only already implemented finite
operations: removal, same-site restoration, matched donor substitution where it is legal, and selected pairwise
compositions. A missing action/state cell is missing evidence, not zero.

Two candidates may be placed in the same proposed state only if:

1. their response vectors agree on discovery documents for every available action and every registered consumer;
2. the same relation predicts held-out documents and held-out circuit families without reselection;
3. applying an action to either candidate leads to response states that remain equivalent under the remaining
   actions; and
4. finite bidirectional substitutions preserve target effects and unrelated circuits better than matched controls.

Condition 3 is partition refinement: equivalence must remain closed under the operations we want the compiled program
to support. It is the key difference from clustering one response vector. Conditions 2 and 4 prevent a post-hoc
partition from being called a circuit.

## Cheapest executable first step after a null

The first step is CPU-only: construct a provenance table of existing receipts and bundles with columns

`candidate, site, action, documents, circuit tags, source corpus, exact/approximate, available per-example data`.

Then report which action-by-candidate cells actually exist for at least two document splits and which are missing.
Do not cluster, select a rank, or impute missing cells in this audit. Its decision is whether the current archive can
support a prospective partition-refinement experiment or whether one small balanced GPU collection is needed.

If coverage is sufficient, freeze an action-indexed discovery/confirmation split and a held-out refinement test. If
coverage is not sufficient, preregister the smallest collection that fills the missing cells. Either path directly
advances cross-module grouping, held-out prediction, composition, and selective manipulation.

## Result-dependent continuation

- If rung496's instrument fails, repair only the instrument; neither route above is licensed.
- If its shared-side screen passes, start the allocated-output finite interchange with the scope correction above.
- If the screen fails lawfully, start the CPU action-coverage table for the intervention-conditional quotient.
- Do not use rank, sparsity, storage, reconstruction, or average CE as the discovery criterion in either branch.
