# Rung515 preregistration: finite downstream quotient of exact attention11 and MLP11 terms

**Frozen:** 2026-09-03 00:58 UTC, after rung514 scoring and before any rung515 model outcome.

## Decision and circuit target

Rung514's exact and planted-identifiable search found zero passing fixed factor allocations or signed two/three-term
programs among113,568 candidates. That closes a small linear program in the exact consumer-write basis. It does not
test the user's central operational definition of sameness: two different writes should count as one variable when
the later model uses them in the same way.

Rung515 therefore changes the observation rather than widening the program. It physically removes exact attention11
and MLP11 interaction terms and lets the actual nonlinear layers after the edited site read them. It asks whether a
term under one equality-score implementation has the same finite task and circuit effects as the same or a different
term under another implementation, and then tests whether the two term tensors can be substituted in both directions.

The targeted circuit properties are cross-action and cross-term grouping, within-consumer splitting, held-out
prediction, selective manipulation, reuse across MLP10 branches, and stable identification. Native attention-head
and MLP boundaries are not treated as semantic units. There is no rank, variance, activation reconstruction, SAE,
quantization, or parameter-compression objective.

## Exact nodes and nonlinear downstream outcomes

Keep rung513's six MLP10 branch subsets, four calibrated score actions `N/P/Z7/Z8`,31 exact attention11
`Q/K/Q2/K2/value` interaction terms, and3 exact MLP11 Left/Right/joint terms. For branch subset `b`, action `a`, site
`s`, and exact term `i`, define one node

`u = (b, a, s, i)`.

There are `6 * 4 * (31 + 3) = 816` nodes. Construct each exact term by the already validated32-corner attention or
four-corner MLP inclusion-exclusion calculation. Then patch the running model at its native site:

`write_removed(u) = write_intact(a,s) - term(b,a,s,i)`.

Recompute every later layer and compare cross-entropy to the intact action. This finite difference includes all
nonlinear downstream interactions; it is not a gradient or write-space cosine.

For each document half, node `u` has:

- a four-coordinate equality-task effect `T_h(u)`, consisting of removed-minus-intact CE on near matches, far
  matches, positions with one earlier match, and positions with multiple earlier matches; and
- a circuit effect `C_h(u,c)` for every circuit family `c`, defined as the removed-minus-intact CE effect on that
  circuit's member tokens minus the same effect on its matched slice-control tokens.

Use the fixed32 circuit tags from rung510 for discovery and keep its other30 tags unopened until confirmation.
Discovery documents500:748 are split500:624 and624:748. Confirmation documents752:1000 are split752:876 and
876:1000. Documents748:752 remain unused so batches never cross a boundary.

## Fixed pair space

Test only the three source relations inherited before rung513: `N--Z7`, `N--Z8`, and `P--Z7`. A pair must use the
same MLP10 branch subset and native consumer site, but its exact term names may differ. Thus a query-only term under
one action may match a query-key interaction under another. The fixed pair count is

`6 branch subsets * 3 action relations * (31^2 attention pairs + 3^2 MLP pairs) = 17,460`.

Same-term pairs remain included because nonlinear downstream use could agree even though rung513's write vectors did
not. Cross-site attention11-to-MLP11 pairs are excluded from this rung because a donor inserted before MLP11 is acted
on by MLP11 whereas the reverse donor is not; a single substitution rule would not compare the same causal object.
This restriction is about causal location, not an assertion that the two modules are semantic units.

## Discovery rule

For pair `u` under left action and `v` under right action, fit one scalar only on the first discovery half's32-circuit
vector:

`beta(u <- v) = <C_0(v), C_0(u)> / <C_0(v), C_0(v)>`.

Do not fit term weights, directions, or a neural reader. The actual model suffix supplies the nonlinear reader. A
pair passes only if all clauses hold:

1. Both nodes have pooled circuit-effect RMS at least`.0005` nat and pooled four-task norm at least`.00025` nat.
2. `0.25 <= abs(beta) <= 4`.
3. On the32 circuit effects, cosine is at least`.90` and relative residual at most`.35` in half0; with beta frozen,
   cosine is at least`.80` and residual at most`.50` in half1.
4. On the four task effects, the same beta gives cosine at least`.70` and residual at most`.65` in both halves.
5. The reciprocal scalar passes the same finite and materiality checks in the reverse direction.

Keep every passing pair without ranking. Exactly1--16 pairs may open confirmation. Zero is a registered null; more
than16 is a non-identifiable result and does not permit selecting the best16.

## Multiplicity and planted recovery

Sixteen fixed controls use seeds51510:51526. For each non-native score action, apply one fixed permutation of the32
circuit coordinates to every term at that action, preserving relationships among its terms and using the same
permutation in both document halves. Task coordinates are unchanged. Run the full17,460-pair detector. The real
candidate count must be strictly larger than the largest control count as well as lying in1--16. Controls cannot
alter the absolute gates.

Before any model outcome counts, eight synthetic cases with seeds51500:51508 must each contain exactly one planted
cross-action proportional pair, possibly with different term indices, plus nuisance nodes. The complete discovery
detector must recover exactly that pair and no other. Failure makes prediction A false and routes only to an
instrument repair before model interpretation.

## Held-out prediction

For every discovery pair, keep node identities and beta frozen. On documents752:1000 and the30 unopened circuit
families, require pooled and per-half circuit cosine at least`.75` and residual at most`.55`, task cosine at least
`.70` and residual at most`.65`, and the same materiality floors. No refit, reselection, threshold change, or use of
the held-out circuits in pair construction is allowed.

## Bidirectional physical substitution

For a confirmed same-site pair `u=(b,a,s,i)` and `v=(b,a',s,j)`, capture both exact term tensors on each confirmation
batch. To emulate removal of `u` in action `a`, substitute the scaled donor removal at the same consumer boundary:

`write_sub(a) = write_intact(a,s) - beta * term(b,a',s,j)`.

Compare this with the real target removal `write_intact(a,s)-term(b,a,s,i)`. Run the reciprocal intervention in
action `a'` using `1/beta`. In both document halves and directions, the substituted finite response must match the
target-removal response with circuit cosine at least`.75`, circuit residual at most`.55`, task cosine at least`.70`,
and task residual at most`.65`. Absolute off-target CE difference between substitution and target removal must be at
most`.002` nat. Every requested patch must be nonzero and replay the requested residual write.

A pair is an operational equivalence only after both directions pass. Report a **cross-term grouping** when `i != j`.
Report **branch reuse** only when the same `(site,i,j,action relation)` passes physical substitution for at least two
different branch subsets. Connected components require every internal pair to pass and fitted scale products around
every cycle to lie within25% of one; otherwise report edges without transitive closure.

## Registered predictions and routes

- **A — exact live and identifiable instrument:** all hashes, score calibrations, source relations, term identities,
  corner replays, suffix calls, masks, finite removals, substitution patches, and eight unique planted recoveries pass.
- **B — small downstream relation:** exactly1--16 real pairs pass discovery and the real count strictly exceeds every
  coordinate-permutation control count.
- **C — held-out identification:** at least one B pair predicts both halves and all30 unopened circuit outcomes with
  identities and beta frozen.
- **D — causal interchange:** at least one C pair passes both physical substitution directions and off-target guard.
- **E — nontrivial decomposition:** at least one D pair has different exact term names or the same exact mapping is
  reused across at least two MLP10 branch subsets.

`strong_null = not (A and B and C and D)`. E is the stricter grouping/reuse result.

Frozen routes:

- A false: repair only the named instrument clause; preserve an invalid receipt namespace.
- B false with zero pairs: exact consumer terms are not operationally proportional even under the real nonlinear
  suffix; leave the MLP10-consumer descent and move to a task-defined state transition or a different program gap,
  not wider term sums, rank, or threshold relaxation.
- B false because controls or more than16 pairs match: strengthen the observation with independently specified
  circuit tasks; do not select a best subset.
- C false: preserve discovery screens only and identify which held-out circuit outcomes break them.
- D false: the downstream-effect correlation is not an interchangeable computation; localize the first later site
  that separates the two removals.
- D true, E false: retain same-term action portability only and validate it on fixed out-of-distribution code.
- D and E true: validate the grouping/reuse on fixed out-of-distribution code, then price its executable interface.

No outcome permits cross-site substitution in this rung, support widening beyond rung514, continuous term weights,
post-outcome nearest-neighbor selection, rank, SAE, reconstruction, quantization, or calling a discovery correlation
a circuit.

## Literal price

One248-document phase has62 batches. Each batch uses one direct calibration, one score-absent capture, four intact
action captures,24 MLP10-branch-removed consumer captures, and816 exact-term removals:

`62 * (1 + 1 + 4 + 24 + 816) = 52,452 full forwards`.

Discovery stops there if B is false. Confirmation costs another52,452 forwards if B holds. If `q <= 16` pairs pass
confirmation, the physical pass recollects30 baseline/corner states and runs two substitutions per pair per batch:

`62 * (30 + 2q) = 1,860 + 124q`.

Maximum total cost is`104,904 + 3,844 = 108,748` full forwards,0 backward passes, at most16 fitted scalars,0 deployed
parameters added, and0 parameters saved. Local corner contractions and all requested patches are counted separately.
These are research costs, not deployment savings.

## Frozen evidence

- rung514 result SHA256: `864f7834bd15f8dda591a5aa8e925b4af6b757cdaa3cce54f1e02e56271c00ec`;
- rung514 bundle SHA256: `6e4d1037ef64563001907da1af6ec2ffa4e4ccc581c59a26f02ce9801d82b7b1`;
- rung514 source SHA256: `4248ce6e14789a6d0ee0d907626d4ae3b884c06ae5bbf82520d9ec0e62dcd28a`;
- rung514 preregistration SHA256: `602e167697e1eda8099ee8e52037cb3bf844f793722bba6da463b89cb0fd7957`;
- rung514 preflight addendum SHA256: `30e3635ecc31ffc764b41d65edad426671fac3bf1651ac04317983b32cf3f0c7`;
- rung510 downstream-quotient preregistration SHA256:
  `e344760333af378ea5604c211c259a27d9ff030b60bad8054ca962d465f46055`;
- rung510 result/source SHA256:
  `16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177` /
  `7901aa5d9c7c39bf5666e0f081bfe08047f23c73eec08b12508c601def7b967a`;
- checkpoint weights SHA256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
