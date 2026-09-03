# Rung513 preregistration: exact attention11 and MLP11 factor interactions

**Frozen:** 2026-09-03 00:10 UTC, after rung512 scoring and before any rung513 model outcome.

## Decision and circuit target

Rung512 found18 fixed same-subset action pairs whose complete MLP10 branch writes are proportional on both discovery
halves, but none remains proportional after attention11. MLP11 and its question-mark form do not restore any relation.
This is evidence that the first downstream consumer uses differences hidden by the MLP10-output comparison. It is not
evidence that the whole of attention11 is one semantic unit.

Rung513 splits that consumer by the tensor contraction it actually computes. The targets are within-module splitting,
cross-action grouping of shared subcomputations, held-out prediction, and selective physical substitution. Rank,
reconstruction error, parameter count, and activation variance are not discovery objectives.

## Frozen source relations

The source-level rule and rung512 result fix exactly18 relations. The six branch subsets are

`L`, `R`, `L+R`, `L+LR`, `R+LR`, and `L+R+LR`.

For each subset the three action relations are `N<->Z7`, `N<->Z8`, and `P<->Z7`. The `LR`-only subset and all other
action pairs are excluded because they did not pass the already-frozen source rule. This is outcome-conditioned routing
from rung512, not a new search; the exact list is hash-pinned before rung513 execution.

## Exact attention11 interaction terms

At attention11, after its learned projections, per-head normalization, and rotary position transform, the five
factors are `Q`, `K`, `Q2`, `K2`, and the mixed value `V`. The deployed unnormalized attention write is

`A11 = W_O [ ((Q K^T)/128) * ((Q2 K2^T)/128) V ]`,

with the causal mask applied to the product of the two score matrices.

For one exact MLP10 branch intervention, let factor0 be the branch-removed value and factor1 the intact value. Evaluate
all32 corners obtained by choosing0 or1 for each of the five factors. Möbius inversion on this five-dimensional Boolean
cube gives31 nonempty finite interaction terms. For example:

- `{Q}` uses the changed Q with removed-run `K,Q2,K2,V`;
- `{Q,K}` is the extra joint effect not assigned to either single change;
- `{Q,K,Q2,K2,V}` is the five-way interaction.

Their sum equals the full attention11 response apart from an explicitly stored numerical remainder. All corners use
the deployed bfloat16 attention contraction and output projection. The removed and intact corners must replay the
captured attention11 writes exactly; the numerical remainder must have RMS at most1% of the full response RMS and is
never a candidate. This decomposition is exact finite interaction accounting, not a Taylor expansion or head basis.

## Exact MLP11 interaction terms

For the same intact and branch-removed MLP11 inputs, evaluate its four deployed bilinear corners and define the exact
Left-only, Right-only, and Left-by-Right joint output changes as in rung511. Their sum must equal the deployed MLP11
response with relative squared error at most`1e-12`. These three terms are tested separately from the31 attention
terms, giving34 fixed term names.

## Fixed tests; no ranking

For each of six branch subsets and34 terms, test all three frozen source relations:612 relation-by-term tests. A term
is material only if its response RMS is at least10% of the complete response RMS at that consumer in both document
halves. The10% floor was fixed before outcomes to exclude numerical fragments: it is larger than the equal-norm
`1/31` attention share and does not select a top-k set.

For each relation and term, fit one signed scale on documents500:624 and test it without refitting on624:748. It
passes when both responses are material,`.25 <= |beta| <= 4`, cosine is at least`.85`, and both directional relative
residuals are at most`.55` in both halves.

A `(branch subset, term)` becomes a discovery candidate only if the same named term passes all three fixed action
relations. All204 possible groups are reported and every passer is retained. The complete attention11/MLP11 response
is retained as a negative control: the candidate is meaningful only because the whole response already failed in
rung512.

For each relation, the exact mismatch

`complete_left - beta_source * complete_right`

is also decomposed into its term mismatches plus the numerical remainder. Signed inner products with the complete
mismatch are reported for diagnosis, but do not select candidates.

## Held-out prediction and physical term substitution

Every discovery candidate is recomputed on documents752:1000 with its discovery scales frozen. All three relations
must remain material, have cosine at least`.75`, and have both directional residuals at most`.65` in both halves.

Every confirmed group is then tested physically for all three action relations and both directions. For a target term
`t_x`, donor term`t_y`, and frozen scale beta, keep the target MLP10 branch intact and replace only the consumer term:

`consumer_x -> consumer_x - t_x + beta * t_y`.

Attention terms patch the complete attention11 residual write; MLP terms patch the complete MLP11 residual write.
For comparison, removing`t_x` alone is run once per target action and reused across relations. A group passes physical
substitution only if every direction, in both confirmation halves:

- has a nonzero term-removal effect with copy-task norm at least`.00025` nat and held-out-circuit RMS at least`.0005`
  nat;
- substitution damage has at most`.50` of the corresponding term-removal norm for both the four copy-task cells and
  the30 held-out circuit effects; and
- absolute off-target CE damage is at most`.002` nat.

These are downstream causal requirements. A stable factor tensor without this test remains an algebraic screen.

## Registered predictions and routes

- **A — exact live instrument:** hashes/checkpoint match; rung512's18 source relations reproduce exactly; all branch
  removals, factor captures, corner replays, Möbius/MLP closures, call counts, and any physical patches pass; attention
  numerical remainder is at most1% RMS.
- **B — shared factor term:** at least one of204 fixed `(branch subset,term)` groups passes all three discovery
  relations.
- **C — held-out identification:** at least one B group passes all three relations on new documents with no refit.
- **D — causal interchange:** at least one C group passes every bidirectional physical term substitution.
- **E — reuse:** at least one D-passing term name passes for at least two different MLP10 branch subsets.

`strong_null = not (A and B and C and D)`. E is a stricter reuse claim.

Frozen routes:

- A false: repair only the named factorization or patch instrument.
- B false: exact singleton factor interactions do not reveal a shared subcomputation; use the registered signed
  mismatch decomposition to preregister multi-term combinations with a planted identifiability test, not rank.
- C false: preserve discovery screens only and test why the factor relation changes across documents.
- D false: the factor similarity is not operational equivalence; split at the first downstream reader that rejects
  the substitution.
- D true, E false: retain the identified branch-specific factor circuit and test it on fixed OOD code.
- D and E true: validate the reusable factor vocabulary on OOD code, then compose the passing terms jointly.

No outcome permits lowering thresholds, choosing the best few terms, fitting an unconstrained latent dictionary, or
claiming that a native head, attention module, or MLP is itself the final circuit unit.

## Literal price

There are24 action-by-branch nodes: four actions times six branch subsets. Each248-document discovery or confirmation
phase keeps rung512's2,108 full-model-forward schedule. In addition, each phase evaluates
`62*24*32 = 47,616` attention corner contractions and`62*24*4 = 5,952` MLP corners; these are local module evaluations,
not full model forwards.

If `q` of at most204 groups confirm, physical execution uses per batch one absent arm, four intact arms,24 branch-
removed factor captures, four reusable term-removal arms per group, and six directional substitutions per group:

`62 * [29 + 10q] = 1,798 + 620q` full forwards.

The stop prices are2,108 full forwards after B false,4,216 after C false, and at most
`4,216 + 1,798 + 620*204 = 132,494` full forwards. There are0 backwards, at most612 discovery-fitted scalars,
0 deployed parameters added, and0 parameters saved.

## Frozen evidence

- rung512 result SHA256: `118d28d4d3b106df6b9d20d165a955ace2bfc07ee35b07e9ea748ecb9d6d877e`;
- rung512 bundle SHA256: `504b7d8e892009cfe2c88462f99db53132108adb6b13b21521b0bb3dbf350113`;
- rung512 source SHA256: `ed66fc329b6ad6ce0e6e4b843bbef0046a53d9dbb229f0f7c99604e75ef96f9b`;
- rung512 preregistration/addendum SHAs:
  `b72ab252feb82132602f1f674f594eded0eb419d54319aa7801ad2293f17daf8` /
  `d31e4fa39273f91afd7c09872f995d285133d7759f7f4bf0700870a5d64d68bd`;
- checkpoint weights SHA256: `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
