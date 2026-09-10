# The query is background computation; test the value path next

Follow-up: the [native value-fold test](mlp4_value_folding_and_intervening_computation_2026-09-10.md)
has now run. The direct fold is numerically valid but misses most of the
MLP4-to-value effect; the intervening computations must be investigated.

10 September2026. **The new experiment closes the strong query-carrier
hypothesis at the shared layer9 heads.** Their contextual queries matter
for normal execution, but swapping those complete queries between paired
examples transfers very little of the present/past change. We should stop
looking for the tense carrier by decomposing those query inputs further.

The earlier [source-interaction report](query_source_interaction_math_2026-09-10.md)
found that single query sources were insufficient and separate cuts did
not add to joint removal. Those observations remain valid. The new evidence
limits their interpretation: normalization can create large interactions
without revealing a shared semantic operation.

## What was run and what it found

The managed experiment used the same72 previously opened has/had and
is/was pairs. For each endpoint, it ran the native model, an identity
replacement, and a replacement of the complete query input with its paired
counterpart. Only queries Q/Q2 of heads1/4 at layer9 changed. Keys, values,
other heads/tokens and the receiving residual context stayed native.
The model's query normalization and position rotation were recomputed.

It took24 full forwards,432 sequence evaluations and2.55 seconds inside
the executor. All instrument checks passed; identity full-logit errors
were at most1.53e-5. No model weights were fitted or changed on disk.

The table reports signed projection of the query-swap logit change onto
the native paired change, in percent. A negative number means the swap
moved against that paired direction. This is not task accuracy.

| Panel | Base receives donor query | Donor receives base query |
|---|---:|---:|
| has/had original construction | -0.27% | -1.15% |
| has/had A2 construction | -0.30% | -0.35% |
| is/was original construction | -0.68% | -0.44% |
| is/was A2 construction | +0.40% | +0.51% |

The corresponding answer-margin projections also have magnitude below1.2%.
Every panel misses the registered10% carrier threshold in both directions.
The swaps do change outputs, so this is a live but weak task-transfer
result. It does not prove the queries are irrelevant, or exclude a joint
query/key intervention that was not tested.

## Where the source interactions came from

For a source and the remaining sources, we compared four local attention
reads: both present, source alone, remainder alone, and neither. Their
interaction is I=read(both)-read(source)-read(rest)+read(neither).

The derived read is a quadratic numerator divided by two query-normalization
factors. We separated I into the numerator cross term evaluated at the full
query's normalization, plus a normalization remainder. Both terms were
mapped through the native output projection into the same residual-write
coordinates; no comparison mixes local reads with final logits.

The numerator cross term alone has124–134% relative error across the
endpoint and paired comparisons. The normalization remainder projects
121–131% along the actual interaction, while the cross term opposes it.
These are cancelling vectors, not percentages of explained variance or
additive causal shares. The partition uses an explicit full-query norm
as its reference; it is not a unique attribution of nonlinearity.

An additional CPU audit checked the saved native coefficient banks. Halving
or doubling ALL source gains left the local write identical at FP64
precision. Setting all gains to zero gave a zero read. The exact formula
explains this: the numerator scales quadratically and its normalization
almost cancels that scale. The tiny epsilon terms break perfect invariance;
the largest analytic uniform-gain derivative was1.25e-20 relative to the
native write norm on these banks.

This makes a zero-input baseline special. A large joint-cut effect alongside
small individual cuts need not mean that we have found semantic cooperation.
We should test task-changing interchanges before pursuing that interpretation.

## The next weight-based computation is already implemented

Earlier native MLP4 experiments identify a better-supported route: much of
their restricted effect on the shared heads passes through the layer9 local
value content. The reported local-value retention fractions were .870/.954
for has/had and1.156/.982 for is/was across the two construction families.
Those fractions describe each earlier restricted path effect, not all tense
behavior; the older total-mediation null remains unchanged.

The next distinction is whether MLP4's write reaches that value reader
directly through residual additions, or whether intervening attention/MLP
computations transform it. Only the former can be folded by a simple
composition of the two weight maps.

Let u be the original residual input to layer9 and let delta_m be a change
to MLP4's output. Its direct transported contribution is delta_u=gamma*delta_m,
where gamma is the product of the intervening residual coefficients.
With s0=RMS_scale(u) and s1=RMS_scale(u+delta_u), the value change is exactly

    delta_value = W_V delta_u / s1
                  + W_V u * (1/s1 - 1/s0).

The first term carries changed content. The second records how normalization
rescales the pre-existing content. Neither term may be silently discarded.

MLP4 has output Down[(Left n)*(Right n)]+bias. For two normalized inputs
n_base,n_donor, set middle=(n_base+n_donor)/2 and delta=n_donor-n_base. Then

    delta_feature = (Left middle)*(Right delta)
                    + (Left delta)*(Right middle)
    W_V delta_m = (W_V Down) delta_feature.

This is an exact bilinear difference identity. It folds the downstream value
reader into the MLP's output weights, while leaving its input dependencies
explicit. The constant output bias cancels in the MLP difference; it remains
part of the original residual context and its normalization.

The [implementation](../../mlp_value_lineage_fold.py) passes five CPU controls,
with errors below5.7e-14, including a live normalization term. **This next
fold has not yet been validated on the native MLP4-to-value path.** It does
not include the later writes induced by an actual MLP4 intervention. Comparing
those two effects is the next discriminating native test.

No new circuit is identified or adopted yet. All545902902 native parameters
remain charged. The saved query bank contains947936 conditional entries
and occupies4034332 compressed bytes; it supports further local calculations
but does not independently execute from tokens.

Primary evidence: [native query experiment](../../BILIN18_L9_QUERY_PARTITION_NATIVE_V1_RESULT.json),
[uniform-gain audit](../../QUERY_UNIFORM_GAIN_V1_AUDIT.json), and
[value-fold controls](../../MLP_VALUE_LINEAGE_FOLD_V1_CONTROLS.json).
