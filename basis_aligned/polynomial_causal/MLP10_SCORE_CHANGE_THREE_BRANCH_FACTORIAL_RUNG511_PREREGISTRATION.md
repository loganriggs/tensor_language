# Rung 511: exact three-branch decomposition of MLP10's score-induced change

**Status:** prospectively frozen after rung510 returned zero observable singleton pairs and before any rung511 model
outcome is computed.

## Question and relation to the goal

Rung510 tested all511,566 pairs among1,012 individual action/source-pair interventions. Its exact instrument passed,
716 nodes were materially active, and zero pair passed the frozen discovery relation. Thus no individual exact term
can presently be merged with another as the same downstream variable. The registered route is to test signed sums
whose form is fixed by the bilinear computation, not to rank near-miss pairs or fit another hidden basis.

This rung asks whether MLP10's complete response to the equality-score edit separates into reusable left-input,
right-input, and joint-change computations. It also measures the causal interactions among those three parts rather
than treating single-branch activation patching as an isolated effect. A positive result would split one native MLP
by computation and group the same computation across different upstream realizations of the equality score.

This is not a rank, reconstruction, quantization, or parameter-compression experiment. All branches and all tested
combinations are fixed before outcomes, and every passing relation must survive new documents, different circuit
families, and an actual substitution through layers11--17.

## Exact deployed three-branch identity

For one token, let `z0` be MLP10's normalized input when the L8H4 equality score is absent, and let `za` be its input
under one of the four calibrated score implementations `a in {N,P,Z7,Z8}`. The bilinear MLP, omitting the common
output bias, is

`f(z) = Down[Left(z) * Right(z)]`.

Write `L0=Left(z0)`, `R0=Right(z0)`, `La=Left(za)`, and `Ra=Right(za)`. In exact real arithmetic,

`f(za)-f(z0) = Down[(La-L0)*R0] + Down[L0*(Ra-R0)] + Down[(La-L0)*(Ra-R0)]`.

The three terms are called:

1. `L`: change the left input while holding the right input at the score-absent value;
2. `R`: change the right input while holding the left input at the score-absent value;
3. `LR`: the extra output caused by changing both inputs together.

The actual model executes BF16 linear maps and multiplication. To preserve that deployed computation exactly, run
the four corners `f(L0,R0)`, `f(La,R0)`, `f(L0,Ra)`, and `f(La,Ra)` with the model's real modules. Define

`L = f(La,R0)-f(L0,R0)`,

`R = f(L0,Ra)-f(L0,R0)`,

`LR = [f(La,Ra)-f(L0,R0)]-L-R`.

Thus `L+R+LR` equals the deployed MLP10 score-induced write change by construction. Also compute the float32 ideal
three terms above and report their discrepancy from the deployed branches. This makes BF16 rounding visible without
turning floating-point compression into an interpretation claim.

The branch input `z` is captured by a temporary hook on MLP10 during the already-audited explicit forward. The hook
must fire exactly once per capture and is removed immediately afterward. It may not alter the forward.

## Fixed branch combinations and causal interactions

There are exactly seven nonempty subsets of `{L,R,LR}`:

`L`, `R`, `LR`, `L+R`, `L+LR`, `R+LR`, and `L+R+LR`.

No subset is selected or ranked. Removing subset `S` under action `a` means subtracting its exact output vector from
the deployed MLP10 write and recomputing layers11--17 normally.

For a token mask `M`, let `F_a(S;M)` be the finite cross-entropy change caused by that removal relative to intact
action `a`. Positive means that branch combination helped those predictions. The exact factorial interaction assigned
to a nonempty subset `T` is the Möbius difference

`I_a(T;M) = sum_{S subseteq T} (-1)^(|T|-|S|) F_a(S;M)`, with `F_a(empty;M)=0`.

For two branches this is the joint removal minus the two single removals. For all three it is the part not explained
by any singleton or pairwise interaction. These are direct finite effects, not gradients. All seven `F` values and
all seven `I` values are reported even if no relation passes.

## Data and downstream measurements

Use the same four calibrated score actions and masks as rungs506--510.

- Discovery: documents500:748, split at624, with the same32 circuit families used by rung510 discovery.
- Confirmation: documents752:1000, split at876, with the other30 circuit families that rung510 never opened.
- Documents748:752 remain unused.

For every action and subset, report four copy-task effects: near match, far match, one earlier match, and multiple
earlier matches. Retain the all-copy and off-target masks separately for score calibration and selective-effect
checks. For every circuit family, report member-minus-matched-control effect. Circuit and task vectors are tested
separately.

Reusing rung510's discovery documents is permitted because the seven branch subsets follow from the frozen bilinear
identity and are not chosen from rung510 outcomes. Confirmation circuit families and documents remain unopened when
the relation and all fitted scales are frozen.

## Discovery relations: all42 same-subset action pairs

For each of the seven fixed subsets, test all six unordered pairs of the four score actions:42 relations total. Do
not compare different subsets at this stage and do not rank the42 relations.

For target node `(a,S)` and donor `(b,S)`, fit one signed scalar on the32 discovery-circuit coordinates in document
half0:

`beta(a <- b,S) = <C_0(b,S),C_0(a,S)> / <C_0(b,S),C_0(b,S)>`.

The relation is a discovery candidate only if all conditions hold in both directions:

- each node has pooled circuit-effect RMS at least`.0005` nat and pooled four-task norm at least`.00025` nat;
- `.25 <= abs(beta) <= 4`;
- discovery half0 circuit cosine is at least`.90` and relative residual at most`.35`;
- with the scalar frozen, half1 circuit cosine is at least`.80` and relative residual at most`.50`;
- task cosine is at least`.70` and task relative residual at most`.65` in both halves.

Retain every passer. Zero is a scientific null. There is no upper-count selection gate because all42 fixed
relations fit inside the literal physical-intervention budget.

As a descriptive chance check only, independently permute circuit coordinates within every node using16 hash-fixed
seeds and report candidate counts. The control cannot add, remove, or choose a real relation.

## Held-out prediction

Freeze action identities, subset identity, and `beta`. On documents752:1000 and the30 held-out circuit families,
both nodes must again be material, and pooled plus both document halves must have:

- circuit cosine at least`.75` and relative residual at most`.55`;
- task cosine at least`.70` and relative residual at most`.65`.

Retain every passer; never refit the scalar.

## Bidirectional physical substitution

For every held-out pair `(a,S) <-> (b,S)`, perform both substitutions in the actual running model:

- in action `a`, subtract `beta` times action `b`'s exact branch-combination output;
- in action `b`, subtract `1/beta` times action `a`'s exact branch-combination output.

Recompute layers11--17. In each direction, the substituted response must match the native target-combination removal
with circuit cosine at least`.75`, circuit relative residual at most`.55`, task cosine at least`.70`, and task
relative residual at most`.65`, pooled and in both confirmation halves. Both directions must pass.

This is the grouping test: similar CE fingerprints alone do not make two computations the same.

## Predictable composition and selective effect

A physically portable subset with at least two branches counts as a distributed computation only if its causal
composition is also predictable. Freeze the following rule on discovery, in this order:

1. `additive` if both the circuit-vector norm ratio and task-vector norm ratio of every higher-order Möbius term
   inside `S` are at most25% of the corresponding joint-effect norm;
2. otherwise `interaction-stable` if each higher-order term whose circuit RMS is at least`.0005` nat and task norm
   is at least`.00025` nat has cosine at least`.70` across the two actions and the same fitted scalar predicts it
   with relative residual at most`.65` in both discovery halves;
3. otherwise no composition claim.

The frozen rule must pass on confirmation pooled and in both halves with the same thresholds. For selective effect,
the absolute all-copy joint-removal effect must be at least`.002` nat and at least three times the absolute off-target
effect in both actions and both confirmation halves.

Singleton portability is reported but cannot satisfy the distributed-computation prediction. The complete
`L+R+LR` subset is also reported as the exact whole-change control; by itself it cannot establish a nontrivial split.

## Frozen predictions and routes

- **A — exact live instrument:** score calibration, direct replay, hook count, four-corner replay, three-branch sum,
  seven-subset construction, call/patch counts, mask support, and every interpreted edit pass their exactness and
  nonzero checks. The deployed branch sum has relative-squared error at most`1e-12`; the float32 ideal discrepancy is
  reported rather than selected on.
- **B — discovery portability:** at least one of the42 fixed same-subset cross-action relations passes discovery.
- **C — held-out portability:** at least one frozen relation predicts new documents and the30 unopened circuit
  families.
- **D — physical portability:** at least one held-out relation passes bidirectional branch-combination substitution.
- **E — distributed interpretable computation:** at least one D-passing relation has a two-branch subset, passes a
  discovery-frozen composition rule on confirmation, and has a selective all-copy effect. `L+R+LR` alone cannot
  satisfy E.

`strong_null = not (A and B and C and D and E)`.

Routes are frozen:

- A false: repair only the named instrument clause.
- B false: close global score-action portability for these three exact branches and localize their responses at the
  first downstream consumer, testing the already-known MLP11 question-form interface explicitly.
- C false: preserve the discovery screen only and test consumer-specific nonlinear readouts.
- D false: response similarity is not interchangeability; localize the first layer11--17 consumer that separates it.
- E false: retain any singleton portability but make no distributed-computation or MLP split claim.
- A--E true: validate the passing branch program on the fixed OOD code set, then decompose its stable branches by the
 22 named earlier writes and price an executable replacement.

No outcome permits a rank sweep, reconstruction objective, latent dictionary, threshold relaxation, best-subset
selection, or treating the native MLP boundary as the semantic answer.

## Literal execution price

Batch size is four, so each248-document phase has62 batches. Per batch, collect one direct replay, one score-absent
capture, four intact action captures, and seven subset removals for each action:

`62 * [1 + 1 + 4 + 4*7] = 2,108 forwards` per phase.

Discovery plus confirmation costs4,216 forwards. If `q` relations pass confirmation, physical substitution costs

`62 * [1 direct + 1 absent + 4 intact + 2q substitutions] = 372 + 124q` forwards.

At all42 relations, the maximum is`4,216 + 5,580 = 9,796` full forwards,0 backwards,42 fitted scalars,0 deployed
parameters added, and0 parameters saved. This rung earns circuit evidence, not compression credit.

## Frozen inputs

- rung510 result SHA256: `16d100e7b92152fc70939b000934699882605c30c513c570f6c519b80f943177`
- rung510 bundle SHA256: `a8832624c94e3e9aa491d26290e55a14f94aa103eb7cddc3df3a0e1b34c3eed7`
- rung510 source and preregistration hashes are inherited from its result and must be rechecked by the implementation.
