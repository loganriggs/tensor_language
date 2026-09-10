# The value response needs a mixed chain and a nonlinear explanation

10 September2026. Two further native experiments sharpen the result from
the [direct weight-fold test](mlp4_value_folding_and_intervening_computation_2026-09-10.md):
**both attention and MLP computation are needed, and their combined finite
response is not predicted accurately by the derivative at the receiving
example.** We have not yet recovered a simpler reusable circuit.

The tests concern the existing partial MLP4-to-layer9-local-value path in
heads1/4, on72 previously opened has/had and is/was pairs. They do not
establish whole-task sufficiency or new-text OOD prediction. The earlier
has/had full-logit carrier miss remains failed.

## First test: which intervening computations are needed?

While swapping the MLP4 output between paired examples, we held either
attention outputs or MLP outputs at their receiving-native values in
layers5–8. All unfrozen modules recomputed on their current inputs. We also
tested both frozen and neither frozen. Each resulting layer9 local-value
field was tested through the same receiving-native heads and suffix.

| Allowed intervening response | Full-logit relative error versus full path |
|---|---:|
| MLPs only | 34.6–45.6% |
| Attention only | 72.5–80.1% |
| Direct residual transport only | 81.0–88.0% |

Both new hypotheses miss the registered10% error bar. These are errors
in the vector of centered-logit changes, not accuracy changes. The modules
interact through actual recomputation, so this is stronger than removing
one source term after computing the remaining terms through that source.

The instrument passes: identity clamps and parent effect replay are exact;
holding both module types fixed recovers the independently calculated
direct value field within1.40e-5. The run used80 full forwards,1440
sequence evaluations and3.88 seconds inside the managed executor.

## Second test: can a receiving-context derivative predict the finite response?

Needing both module types does not itself prove nonlinear dependence on
the changed source. Even the linear expression

    (I + M)(I + A) delta = delta + A delta + M delta + M A delta

has a mixed attention/MLP path. Here A and M stand for linear response maps;
the final term requires both maps while remaining linear in delta.

We therefore tested the combined derivative directly. A **Jacobian-vector
product** computes the derivative of the local-value field in one specified
MLP4-change direction, without constructing a giant Jacobian matrix. The
prediction was

    predicted_value = native_value + Df(native_MLP4) * delta_MLP4.

The derivative used only the receiving trajectory. The donor supplied the
source intervention vector, but none of its later intermediate states were
used to construct the derivative. Both attention and MLP operations remained
in the calculation, through the unchanged native model backend.

The actual finite-response test fails: full-logit effect errors are37.5–48.0%,
and margin-effect errors16.1–44.0%. Local-value errors are47.2–54.9%.
Every panel/direction misses at least one required bar. Some answer-direction
projections look much better, but they do not establish vector fidelity.

The derivative machinery passes its checks. On a small model using the
original backend, it agrees with a central finite difference to1.81e-11
maximum absolute error. On the trained model its primal value field,
identity outputs and parent reference effects replay exactly. The run used
48 backend forwards, including8 forward-mode derivative evaluations,
864 sequence evaluations and3.39 seconds in the executor. Derivative work
adds compute; these counts are not an acceleration claim.

## A scalar adjustment cannot repair this result

Let p be the predicted centered-logit effect vector and f the reference
effect. Even allowing the best possible multiplier separately for every
panel and direction, the minimum relative error is

    min_a ||a p - f|| / ||f|| = sqrt(1 - cos(p,f)^2).

The saved norms and projections determine this bound. The resulting floors
are27.6–46.4%, all above the10% bar. Thus a gain adjustment cannot make
this predictor pass. This is a bound on these finite-cohort vectors, not
a bound on nonlinear/vector corrections or future data. No scalar was
installed, fitted for deployment or promoted.

## The next mathematical distinction

The failed derivative linearizes normalization as well as the products
inside attention and MLPs. Those are different sources of nonlinear behavior.
The next candidate keeps every RMS normalization, projection and position
rotation evaluated on the actual changed input. It simplifies only the
products of the resulting factors.

For an MLP with factors L and R, the exact feature change is

    L_base * delta_R + delta_L * R_base + delta_L * delta_R.

The candidate omits only the last term. The inputs to L and R still use
the actual changed normalization. For attention, the analogous expression
retains first-order changes in each of its five normalized factors:
query, key, second query, second key and value. It preserves the causal
mask and the actual shared-first-value payload.

This is not the failed fixed derivative: its normalizers remain nonlinear
functions of the changed state. It tests whether those normalizers are
enough, or whether interactions between changing product factors are also
essential. The same combined rule will apply to both module types in all
four intervening layers; no posthoc source or module subset is nominated.

The [factor-response implementation](../polynomial_factor_response.py)
passes five controls, including exact finite single-factor attention
changes, a live joint-factor remainder and the MLP product identity.
The [native protocol](../BILIN18_MLP4_NORM_PRESERVING_RESPONSE_V1_PREREGISTRATION.md)
is registered; **native integration and execution remain pending**.

All545902902 native parameters, receiving context and source generation
remain charged. None of these results is an adopted reduction or a circuit
meeting all four requested properties.

Primary evidence: [module-dependency experiment](../BILIN18_MLP4_INTERVENING_CHAIN_V1_RESULT.json),
[finite tangent experiment](../BILIN18_MLP4_VALUE_TANGENT_V1_RESULT.json),
[direction-error bound](../VALUE_TANGENT_DIRECTION_V1_AUDIT.json), and
[factor controls](../POLYNOMIAL_FACTOR_RESPONSE_V1_CONTROLS.json).
