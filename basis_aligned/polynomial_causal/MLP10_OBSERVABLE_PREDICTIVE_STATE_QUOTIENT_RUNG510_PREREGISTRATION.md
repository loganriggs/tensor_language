# Rung 510: observable downstream-equivalence pairs inside MLP10

Status: prospectively frozen after rung509's synthetic identifiability gate failed and before any rung510 model
outcome is computed.

## Why this changes the object

Rung509 tried to infer eight hidden groups whose Left-source assignments, Right-source assignments, and downstream
effects were learned jointly. On a ground-truth toy with eight distinct 99.77%-pure observed anchors, the repaired
archetypal fit still failed all four recovery requirements: minimum response cosine `.3536 < .90`, minimum
assignment cosine `.6715 < .80`, minimum anchor weight `.000993 < .90`, and only `37/48` anchor identities correct.
No model checkpoint was loaded and no rung509 model outcome was opened.

This rung makes no hidden-group claim. Its objects are directly observed interventions. It asks whether two exact
MLP10 input-pair terms can be treated as the same downstream variable because a scale learned on one document set
predicts their relative causal effects on new documents and previously unused circuit families, and because their
actual output changes can be substituted for one another in the running model.

This targets within-MLP splitting, grouping across source pairs and score implementations, held-out prediction, and
selective manipulation. It is not a rank, quantization, reconstruction, variance, or parameter-count experiment.

## Exact nodes and the 62 circuit families

Use the same 22 exact normalized inputs to MLP10: embedding `E`, attention outputs `A0..A10`, and earlier MLP outputs
`M0..M9`. The bilinear expansion has 253 unordered terms. Under each of the four already calibrated equality-score
implementations `N/P/Z7/Z8`, removing one exact term and recomputing layers11--17 defines one **observable node**

`u = (score implementation a, exact source pair p)`.

There are `4 * 253 = 1,012` nodes. For every node, measure finite cross-entropy changes rather than gradients:

- four copy-task coordinates: near match, far match, one earlier match, and multiple earlier matches;
- member-minus-matched-control effects for 32 fixed circuit families in discovery; and
- the same member-minus-control procedure for the other 30 circuit families in confirmation.

Thus all 62 existing circuit families are used, with 32 allowed to propose a relation and 30 kept unopened until
confirmation. Discovery uses documents500:748, split at624. Confirmation uses documents752:1000, split at876.
Documents748:752 remain unused so no batch crosses a boundary.

## Discovery relation

For node `u`, let `T_h(u)` be its four task effects and `C_h(u)` its 32 discovery-circuit effects in document half
`h`. For every unordered pair of distinct nodes `(u,v)`, fit exactly one scalar on half0:

`beta(u <- v) = <C_0(v), C_0(u)> / <C_0(v), C_0(v)>`.

This scalar records sign and scale; it is not a learned direction. The relation is tested in both orientations by
using `beta(v <- u) = 1 / beta(u <- v)`. A pair is a discovery candidate only if all clauses hold:

1. Both nodes are materially active: pooled circuit root-mean-square effect is at least `.0005` nat and pooled
   four-task norm is at least `.00025` nat. These are the same conservative effect scales used in rungs506--508.
2. `0.25 <= abs(beta) <= 4`; this symmetric interval prevents a nearly dead node from explaining a live one.
3. On circuit coordinates, predicting `u` as `beta*v` has cosine at least `.90` and relative residual at most`.35`
   in half0; without refitting, it has cosine at least`.80` and residual at most`.50` in half1.
4. On the four task coordinates, the same frozen scalar gives cosine at least`.70` and relative residual at
   most`.65` in both halves.
5. Both directions satisfy the preceding criteria. This is algebraically redundant in exact arithmetic for cosine,
   but the explicit check catches zero denominators and implementation asymmetry.

Keep every passing pair; do not rank or take nearest neighbors. Exactly1--16 candidates opens confirmation. Zero is
a registered null. More than16 means this observation basis does not identify a small quotient and is also a null;
the best16 may not be selected. The cap is an interpretation and intervention budget, not a searched rank.

As a descriptive chance control only, repeat the detector after independently permuting the 32 circuit coordinates
of every node with 16 hash-fixed seeds. This control cannot promote or remove a real candidate and no threshold will
be changed from its result.

## Held-out prediction

For every discovery candidate, freeze its node identities and scalar. On documents752:1000 and the 30 held-out
circuit families, require:

- circuit cosine at least`.75` and relative residual at most`.55` in pooled data and each document half;
- task cosine at least`.70` and relative residual at most`.65` in pooled data and each half; and
- both nodes again clear `.0005` circuit RMS and `.00025` task norm.

The scalar is never refit on confirmation. Retain every passer. Zero held-out pairs is a null; one through16 opens
physical substitution.

## Bidirectional physical substitution

Response similarity is still only a prediction. For a held-out pair

`u=(a,p), v=(b,q)` with `R(u) approximately beta R(v)`, capture the exact score-dependent MLP10 output changes
`deltaY[a,p]` and `deltaY[b,q]` on confirmation documents. Then perform two new interventions:

1. in the actual `a` background, subtract `beta * deltaY[b,q]` where `deltaY[a,p]` would have been removed;
2. in the actual `b` background, subtract `(1/beta) * deltaY[a,p]` where `deltaY[b,q]` would have been removed.

Recompute the real layers11--17 suffix after each substitution. In each direction, the substituted finite response
must predict the native term-removal response with circuit cosine at least`.75`, relative residual at most`.55`, task
cosine at least`.70`, and task residual at most`.65`, pooled and in both halves. Both directions must pass. This tests
the downstream readers directly; merely similar stored vectors do not count.

A passing edge with `p != q` is evidence that different exact MLP10 input interactions are one downstream variable.
A passing edge with `a != b` shows that the same variable can survive a change of equality-score implementation.
Report these separately. Connected components are called quotient groups only when every within-component pair
passes physical substitution and the products of fitted scales around every cycle differ from one by at most25%.
Otherwise report passing pairs without taking a transitive closure.

## Predictions and routes

- **A — exact live instrument:** all source, factor, float32, deployed-output, patch-count, support, replay, and score-
  calibration checks pass; every requested singleton and substitution edit is nonzero.
- **B — small discovery relation:** exactly1--16 pairs satisfy every discovery clause without ranking.
- **C — held-out relation:** at least one frozen pair predicts new documents and all 30 held-out circuit coordinates.
- **D — physical downstream equivalence:** at least one held-out pair passes both physical substitutions.
- **E — nontrivial grouping:** at least one D-passing pair uses different exact source pairs, and any reported
  multi-node quotient group passes the complete-graph and scale-cycle rules.

`strong_null = not (A and B and C and D and E)`.

Routes are frozen:

- A false: repair only the named instrument clause before any interpretation.
- B false with zero pairs: close pairwise proportional equivalence at this observation scale and test registered
  multi-term signed combinations; do not relax bars or rank pairs.
- B false with more than16 pairs: add independently defined downstream tasks before intervention; do not select16.
- C false: close the observable response-pair grouping and test consumer-specific nonlinear readouts.
- D false: treat response similarity as non-interchangeable and localize which downstream consumer separates it.
- E false: retain only same-term/action portability; do not claim a new within-MLP grouping.
- A--E true: validate the quotient pairs on the fixed OOD code set, then price an executable replacement.

## Literal price and stopping points

One complete exact-term phase costs

`62 batches * [1 direct + 1 score-absent + 4*(1 intact capture + 253 removals)] = 63,116 forwards`.

- If B is false, stop after discovery: `63,116` forwards.
- If B passes but C has zero pairs, stop after confirmation: `126,232` forwards.
- If `q` pairs pass held-out prediction, physical substitution costs
  `62 * [1 direct + 1 absent + 4 intact captures + 2q substitutions] = 372 + 124q` forwards.
- Maximum at `q=16`: `128,588` forwards.

There are zero model backward passes, at most `C(1,012,2)=511,566` CPU pair comparisons, at most16 fitted scalars,
and zero deployed parameters added or removed. The fit-failed rung509 model output namespace remains absent.

## Frozen identities

- rung509 identifiability result SHA256:
  `7a10e97a41328b97008d1e1b81a70de77977bdd2fb615dd701878ee9d26a3d1a`
- rung509 implementation SHA256 at that result:
  `f346b78ab47006c68d522d153d441603e627e60233e6cba3dd703e7225ef6ec3`
- rung509 identifiability-repair SHA256:
  `381988395edd4d54c1d08ba99bef336ed0ca708fc48497dc479887e0d647f5bf`
- rung508 result SHA256:
  `05060565f25a5b59a233f5b336ee9882e330ea3e39f8d7f6b27e715aab5825ba`
