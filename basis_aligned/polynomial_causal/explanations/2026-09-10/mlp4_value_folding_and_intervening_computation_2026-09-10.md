# The MLP4 value path needs intervening computation

Follow-up: [the intervening-chain and derivative experiments](mixed_chain_and_nonlinear_response_2026-09-10.md)
have now run. Neither module type can be dropped, and the receiving-context
derivative also fails to predict the finite partial-path effect.

10 September2026. **The direct weight fold works algebraically, but it does
not reproduce the trained model's MLP4-to-value computation.** It projects
only12.5–20.5% onto that path's centered-logit effect, with81–88% relative
effect-vector error. The attention and MLP modules between layers4 and9
cannot collectively be replaced by their direct residual connection.

This is progress in specifying the computation, rather than a newly
identified reusable circuit. The next test separates the contributions of
the intervening attention computations from the intervening MLP computations.

## What the experiment compared

We used72 previously opened has/had and is/was pairs, retaining all native
errors. We replaced MLP4's output with its paired counterpart at every valid
token position, in both directions. This is broader than the earlier
selected-source-bank MLP4 experiments; their numerical results are not being
reinterpreted as results of this new intervention.

The reference effect comes from allowing the later modules to recompute,
capturing the resulting layer9 local values, and inserting those values
into the receiving model's heads1/4. The receiving queries, keys, shared
first-layer values and residual background remain native; the suffix is
recomputed. This isolates the effect through those local value inputs.

The candidate instead transports the MLP4 output change directly through
the residual connections. It recomputes the resulting value normalization
but leaves the other source contributions at their receiving-native values.
The transport coefficient is gamma=.03257659278460778. This coefficient
alone is not a causal importance score.

The tested weight fold was

    delta_value = gamma * (W_Value Down4) delta_feature / s_new
                  + W_Value u * (1/s_new - 1/s_original).

The first term carries the changed MLP content. The second accounts for
normalization rescaling the pre-existing content. The full original
MLP-output difference and residual still supply the norm inputs, and their
computation remains charged. We do not claim to have removed Down4 or the
native prefix merely by folding its value reader.

## Results and checks

| Task | Direct/full value-effect signed projection | Full-logit relative error | Margin relative error |
|---|---:|---:|---:|
| has/had | 19.4–20.5% | 81.0–83.0% | 70.8–77.8% |
| is/was | 12.5–13.0% | 87.5–88.0% | 84.8–87.2% |

Ranges cover the two construction panels and both swap directions.
Projection measures alignment and magnitude relative to the reference
effect; it is not task accuracy or an additive fraction of a circuit.
The registered maximum error was10%, so the candidate fails clearly.

The reference local-value path itself passes the10% full-logit/margin
carrier screen for is/was, but misses the full-logit part for has/had.
The latter still has11.0–16.7% signed margin projection, but only5.7–7.8%
full-logit projection. That failed screen remains recorded. Preserving this
partial path in a later decomposition would not establish whole-task
sufficiency.

All numerical instrument checks passed:

- Folded normalized content versus independent FP64 contraction: maximum
  absolute error9.88e-15.
- Folded versus directly recomputed native value field:1.59e-5.
- Their complete downstream logits:1.34e-5.
- Identity value replacement: exact full-logit replay.
- Read-only residual recurrence: bitwise normalized-input replay, including
  runs with the MLP4 intervention. Hooks and unselected values were preserved.

The managed run used48 full forwards,864 sequence evaluations and2.86
seconds inside the executor. All545902902 native parameters remain; the
folded reader has1179648 coefficients and achieves no actual weight saving.

## Mathematical tool for the intervening chain

An ordinary derivative describes a very small input change. These paired
interventions are finite changes, so we derived an exact **secant operator**:
a linear map specific to the two endpoints that reproduces their difference.
It is a tool for accounting for a known change, not a predictor independent
of those endpoints.

For RMS normalization n(x)=x/s_x, let m=(x+y)/2. Then

    n(y)-n(x) = T(x,y)(y-x)
    T(x,y)v = .5*(1/s_x + 1/s_y)*v
               - 2*m*(m^T v)/(D*s_x*s_y*(s_x+s_y)).

D is the residual dimension, and each scale includes the model's epsilon.
This applies T using a vector scaling and one inner product; it never
materializes a1152-by1152 matrix.

Combining it with the exact midpoint difference of a bilinear MLP gives
an exact finite-change operator for a normalized MLP. Five CPU controls
pass with errors below3.6e-14. The receiving-point tangent gives a large
error on the same planted finite step, so replacing this secant with a
local derivative would not be an exact computation.

The [secant implementation](../../normalized_bilinear_secant.py) can help trace
an identified response through several MLPs. A reusable task-independent
operator would still need to be discovered and tested across new contexts.

## Next experiment is registered and its intervention tool is tested

While applying the same MLP4 change, hold layers5–8 attention outputs at
their receiving-native values; alternatively hold their MLP outputs fixed;
also test both fixed and neither fixed. Every unfrozen module recomputes
on its current input. Compare the resulting local-value effects through
the same receiving heads.

This tests actual computational dependencies. Deleting an attention source
term after computing all the MLP changes through that attention would not
show that the MLP computation can dispense with it.

The [protocol](../../BILIN18_MLP4_INTERVENING_CHAIN_V1_PREREGISTRATION.md) fixes
80 forwards and1440 sequence evaluations. The [clamp tool](../../intervening_write_clamp.py)
passes five controls, including preservation of the first-value payload
and recovery of pure direct transport when both intervening module types
are held fixed. **The native chain experiment has not run yet.**

Primary evidence: [native folding result](../../BILIN18_MLP4_VALUE_LINEAGE_NATIVE_V1_RESULT.json),
[secant controls](../../NORMALIZED_BILINEAR_SECANT_V1_CONTROLS.json), and
[clamp controls](../../INTERVENING_WRITE_CLAMP_V1_CONTROLS.json).
