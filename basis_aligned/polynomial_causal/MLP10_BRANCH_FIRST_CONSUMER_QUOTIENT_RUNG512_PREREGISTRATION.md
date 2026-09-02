# Rung512 preregistration: exact MLP10 branches at the first downstream consumers

**Frozen:** 2026-09-02 23:46 UTC, after valid rung511 scoring and before any rung512 model outcome.

## Decision and circuit target

Rung511 found that all28 fixed `(score implementation, branch subset)` interventions are live, but none of the42
same-subset cross-action relations is proportional in the combined copy-task plus32-circuit observation. Eight of42
relations nevertheless have copy-task cosine at least`.70` in both document halves. The registered B-false route is
therefore to change the observation, not the decomposition: keep the exact `L`, `R`, and `LR` branches and ask where
the first real downstream consumers treat two implementations as the same variable.

This rung can advance four circuit targets:

1. split MLP10 by downstream use rather than by native units or rank;
2. group different score implementations when an actual consumer cannot distinguish their branch effects;
3. predict that relation on fresh documents; and
4. substitute one local consumer response for another without changing the target task or held-out circuits.

It provides no compression credit by itself. No rank, singular-value, sparse-autoencoder, dictionary, reconstruction,
gradient ranking, nearest-neighbor ranking, or post-outcome subset choice is allowed.

## Fixed computation

For each score action `a` and MLP10 branch subset `s`, rung511 supplies the exact deployed output change
`delta10[a,s]`. It is physically removed at MLP10 and the model is recomputed. For any captured downstream quantity
`g`, define its finite response

`Delta_g[a,s] = g(intact action a) - g(action a with delta10[a,s] removed)`.

This is a finite causal response, not a gradient. The three primary consumer outputs are fixed before execution:

- `A11`: attention11's complete1152-number residual-stream write;
- `M11`: MLP11's complete1152-number residual-stream write;
- `Q11`: MLP11's already-published question-mark quadratic scalar.

`Q11` is rebuilt directly from frozen model weights. Let `u?` be the unit-normalized mean unembedding vector of GPT-2
tokens matching `^\?$| \?$`. For MLP11 input `z`, form

`S? = sym(Left^T diag(u?^T Down) Right)`.

The two fixed eigenvectors with largest absolute eigenvalues have archived eigenvalues about`+144.8641` and
`-73.8464`. If their coordinates are `c+` and `c-`, then

`Q11(z) = 144.8641 c+^2 - 73.8464 c-^2`.

This scalar is exactly the selected two-direction part of MLP11's output projected onto `u?`. Eigenvector signs do
not matter because the coordinates are squared. The source response `delta10` is also measured as a fixed control:
a relation that first appears at `A11`, `M11`, or `Q11` is consumer-induced rather than already present in the MLP10
write.

For `A11`, `M11`, and `delta10`, comparison vectors concatenate every response coordinate at the fixed
`all_positive` copy-task positions. For `Q11`, the primary vector contains its scalar response at those same positions.
The scalar response on actual question-token positions is an additional fixed semantic test, not the sole selector.
All other valid positions are reported as an off-task diagnostic.

## Relation test with no ranking

The seven branch subsets and six action pairs per subset give the same42 relations as rung511. Every relation is
tested independently at all three primary consumers, for126 possible `(relation, consumer)` candidates. None is
ranked; every passer is retained, including zero or all126.

On discovery documents500:624, fit the one scalar

`beta = <Delta_left, Delta_right> / ||Delta_right||^2`.

Without refitting, a consumer relation passes discovery only when all of the following hold on both documents500:624
and624:748:

- both response root-mean-squares are at least`1e-4` of the corresponding intact consumer root-mean-square and are
  strictly nonzero;
- `.25 <= |beta| <= 4`;
- cosine is at least`.85`; and
- both directional relative residuals are at most`.55`.

For `Q11`, the same beta must additionally give cosine at least`.70` and both directional relative residuals at most
`.65` on the question-token responses in both halves; each half must contain at least20 question tokens. The observed
support was checked without running the model: discovery halves contain40 and57 question tokens. The archived
question basis is a live prerequisite, not a fitted rung512 outcome.

Each candidate is typed in advance:

- `transported`: its `delta10` source response also passes the same discovery rule;
- `consumer_convergence`: its source response fails but the consumer response passes.

This distinction prevents ordinary similarity already present in MLP10's output from being misreported as a
downstream-created basis.

## Confirmation and physical substitution

Every discovery candidate, with its discovery beta frozen, is tested on documents752:876 and876:1000. Both responses
must remain material; cosine must be at least`.75`; and both directional residuals must be at most`.65`. A `Q11`
candidate must also pass its question-token test at cosine`.65` and residual`.75`; those halves contain26 and36
question tokens. No relation or consumer is selected after confirmation.

Every confirmed candidate is then substituted in both directions at the named consumer while keeping the target
MLP10 branch intact. For target `x` and donor `y`, the consumer write becomes

`g_x,intact - Delta_g[x] + beta * Delta_g[y]`.

At `A11` and `M11`, this patches the complete residual write. At `Q11`, only the question direction of MLP11's write
is changed by `(beta Delta_Q11[y] - Delta_Q11[x]) u?`. This isolates whether the suffix distinguishes the proposed
consumer variable; it does not erase MLP10's direct residual path.

A bidirectional substitution passes when, in both confirmation halves:

- the norm of its four-cell copy-task damage is at most`.50` of the corresponding full MLP10 branch-removal damage;
- the norm of its30 held-out-circuit damage vector is at most`.50` of the full branch-removal damage;
- off-target CE damage is at most`.002` nat, twice the largest parent calibration offset rounded upward; and
- every requested patch is nonzero and every call/capture/patch count is exact.

For `Q11`, the question-token true-logit damage and CE damage are also reported relative to full branch removal. They
do not override the primary copy/circuit substitution rule.

## Registered predictions and routes

- **A — valid instrument:** frozen hashes and checkpoint match; direct native replay is exact; the R511 branch
  identity/calibration and all call/capture/patch counts pass; the rederived question eigenvalues match the archived
  values within`1e-3`; all fixed masks have the registered support.
- **B — local relation:** at least one of126 fixed consumer relations passes discovery.
- **C — held-out prediction:** at least one B relation passes confirmation with its discovery beta unchanged.
- **D — causal interchange:** at least one C relation passes bidirectional physical substitution at its consumer.
- **E — question interface:** at least one `Q11` relation passes discovery and confirmation, independently of whether
  it passes D.

`strong_null = not (A and B and C and D)`. E is a separately scored semantic hypothesis because the equality-score
branches need not use the question channel.

Frozen routes:

- A false: repair only the named instrument clause; preserve a distinct invalid namespace.
- B false: close output-level consumer equivalence and split the exact branch response inside attention11
  (`Q/K/Q2/K2/value`) and MLP11 (`Left/Right/product`) with finite interventions, not rank.
- C false: preserve discovery screens only; seek a task-conditioned nonlinear consumer response with a planted
  identifiability test.
- D false: response similarity is not interchangeability; use the first consumer whose physical swap fails as the
  splitting boundary.
- D true, E false: retain the identified consumer-local variable without attaching question semantics; test it on the
  fixed OOD code set and compose it with the other confirmed local variables.
- D and E true: validate the consumer-local question-channel relation on OOD code, then trace the stable response back
  through the22 named earlier writes.

No outcome permits threshold relaxation, a best-k relation list, another MLP10 rank/dictionary sweep, or treating an
entire attention/MLP module as the final semantic unit.

## Literal price

Batch size is four, so each248-document phase has62 batches. Discovery reuses rung511's exact schedule:

`62 * [1 direct + 1 absent + 4 intact + 4*7 removals] = 2,108 forwards`.

Confirmation costs another2,108 forwards if B is true. Physical substitution first recollects one absent, four intact,
and28 removed branch arms per batch, then runs two substitutions for each confirmed `(relation, consumer)`. If `q`
of at most126 candidates confirm, physical cost is

`62 * [33 + 2q] = 2,046 + 124q forwards`.

The stop points are2,108 forwards after a B false result,4,216 after a C false result, and a maximum
`4,216 + 2,046 + 124*126 = 21,886` forwards. There are0 backwards, at most126 fitted discovery scalars,0 deployed
parameters added, and0 parameters saved.

## Frozen evidence

- valid rung511 result SHA256: `39a6afc592ceea8ed3f79d2928333eb70442ca63f5d147f61635649e57fca6d4`;
- valid rung511 bundle SHA256: `16a70cb757ba97a6bc72b1b5bf2a35eaae4b7c5538b474254cad4beabb377a6e`;
- rung511 source SHA256: `6d07301b253c1216ea24e310eb82e1deab5c18baa3ce120b590cfa7fdba95031`;
- rung511 preregistration SHA256: `95a296478a5adc21ef0ef9bf8a1762ddd86e8f1312258733ce1de2eb2d9b4cd4`;
- archived question-writer result/source SHAs:
  `f3394570e3122f8fee84f9e30b51574367549a68489a6c9505c67785b72b3cde` /
  `0b534adeceabd1cedec6977470c277a4dfafa786f787323f8e1f9fba2b0a04ee`;
- archived one-product result/source SHAs:
  `f8f58fd96b37eb23f95dc69b140b7b1c5edf9d708f247c3935e502d3ce03a2f5` /
  `4ff0fd56983818dc129d13244db092bf3aa3522f818fd5df71e4f310de5b2f9a`;
- checkpoint weights SHA256 inherited from rung511:
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.
