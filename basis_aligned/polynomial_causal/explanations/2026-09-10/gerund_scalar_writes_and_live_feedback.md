# Shared grammatical writes work; the scalar-only predictor does not

**10 September, 16:06 UTC.** Applying the same grammatical direction inside
attention and MLP outputs is stronger than editing it only at the final readout.
The distributed swap recovers **94%/102%** of the bare-verb versus -ing cue effect
in the two tested frames. Its registered preservation and removal tests pass.
However, a program that predicts the edit using only the accumulated scalar
misses **65–67%** of the full-vocabulary effect.

The distinction matters: the scalar is a useful distributed intervention, but
later native computations respond in other directions. Those responses remain
part of the explanation we need to recover. We have not extracted an independent
one-scalar circuit or reduced the model.

We checked the prior writer-accounting work and module dossiers, derived the
exact backward readout weights, ran three groups of live swaps and removals,
then audited their uncertainty and source contributions on CPU. The experiment
reuses the fixed grammatical direction and the previously opened 16 verbs and
control rows. It makes no new-data or broader OOD claim.

## How the readout folds through the residual chain

For block l, write its actual computation as

    x_(l+1) = a_l*x_l + b_l*x0 + A_l + M_l.

A_l and M_l are the attention and MLP output vectors. The learned coefficients
a_l and b_l mix the running state with the normalized token embedding x0.
They are not generally one. Define

    beta_l = product(a_j for j>l),
    gamma = product(all a_j) + sum_l beta_l*b_l.

Then the final residual and its grammatical scalar satisfy

    h = gamma*x0 + sum_l beta_l*(A_l+M_l),
    e^T h = gamma*e^T x0 + sum_l beta_l*e^T(A_l+M_l).

The direction e is unchanged from the eight unembedding bare/-ing pairs in the
[previous experiment](shared_gerund_component_from_unembedding.md). This is a
backward fold of that reader through every residual addition. The result is
an identity on actual module outputs, not an algorithm for producing them.
The existing v185 gate analysis already used this kind of weighted accumulation;
the new test concerns the grammatical reader and its live output-port edits.

Here gamma is 145.12. The earliest attention/MLP writes receive a final weight
of about .000357, while several middle writes receive weights above one, up to
2.084. Ignoring these trained coefficients would distort the source accounting.
Each cue pair has the same final input token, so its entire direct x0 re-entry
term cancels in the cue difference. Routing from other tokens can still change.

## The three distributed interventions

We test all attention outputs, all MLP outputs, and both groups together. At
each eligible module's final semantic position, a swap replaces only its
projection on e:

    y_new = y_live + e*(e^T y_donor - e^T y_live).

The donor value is saved from the natural donor sentence. The recipient value
is recomputed on the current edited trajectory. All other output directions,
other token positions and attention's separate first-value cache remain live.
Removal sets that output projection to zero instead. It does not delete the
whole attention head or MLP.

The scalar-only predictor adds up the changes that would occur on the original
background, propagates them with beta_l, and applies them at the final residual.
It then runs the real final RMS normalization, unembedding and tanh softcap.
Its limitation is not an omitted final nonlinearity: it omits how earlier edits
change later native computations outside the chosen scalar direction.

## Results

| Scalar writes swapped | First-frame cue recovery | Report-frame cue recovery |
|---|---:|---:|
| Attention outputs | 20.0% | 17.5% |
| MLP outputs | 78.7% | 89.6% |
| Both groups | **93.6%** | **101.7%** |
| Earlier terminal-only scalar swap | 58.6% | 62.1% |

Recovery is the change in the native answer/foil margin divided by the native
cue-induced margin difference. It can exceed 100% by overshooting the donor
margin; this is not more than 100% of the model explained. The paired 95% intervals
for both-group recovery are 89.8–97.4% and 97.9–105.4%.

Both-group swaps preserve the controls under the registered criteria: mean
absolute correct-token CE changes are .0404 for can/may and .00962 for the
unrelated either/not behavior. The latter's cue recovery is .00128.

Zeroing both groups raises target CE by .977/1.268 nats, averaged across both
native grammatical endpoints. Intervals are [.878,1.086]/[1.146,1.404]. The
unrelated control's mean absolute CE change is .0433, interval [.0257,.0630],
below the registered .10 limit. These are narrow preservation results.
The can/may states still use the grammatical distinction being removed, and
their removal damage is .985 nats; removal does not preserve those states.

## Why this is not a self-contained scalar program

The frozen-background predictor's both-group swap errors are .654/.670 of the
live full-vocabulary effect norm, with intervals [.649,.658]/[.667,.673]. All
three groups have large prediction errors on both target frames. The registered
10% prediction-error bound fails decisively, even though distributed
sufficiency and selective removal pass.

For both-group swaps, the final scalar actually equals the donor's scalar up
to numerical precision: every module's output scalar was prescribed and the
direct embedding term matched. That is an algebraic consequence of the edit.
It does not fix the remaining 1,151 residual coordinates. The real trajectory
changes those coordinates, and they materially affect the final readout.

A two-coordinate example makes the issue explicit. Suppose the first operation
writes a=1 and the next writes a² into another coordinate, giving state (1,1).
Changing the first write to a=2 produces (2,4). A predictor that edits only the
final scalar gives (2,1). Both have the correct scalar 2, but the predictor misses
the consumer's changed output 3. The CPU audit executes this counterexample.

The native source accounting attributes 84–87% of the raw scalar cue difference
to MLP writes and 13–16% to attention writes. These are sums of realized native
contributions, not independently removable causal shares. For example, editing
attention scalar writes induces additional scalar response from the live MLPs.
That is exactly why the source table cannot replace the causal experiment.

The instrument checks hold: 48 body forwards over 768 sequence instances, all 64
native answer/foil pairs correct, state reconstruction relative error below
1.40e-7, readout error below 4.01e-5, and maximum all-port scalar identity error
.00073. Managed executor time was 2.04 seconds. All 545,902,902 native parameters
remain; 36 output sites share the direction, but their producers are not thereby
replaced by one computation.

The next explanation must account for the context-dependent responses outside
e, and test stronger construction/control transfer before treating this as a
general grammatical circuit. No rank increase, gain adjustment, selected layer
subset or relaxed sufficiency threshold was used here.

Evidence: [preregistration](../../GERUND_SCALAR_NETWORK_V1_PREREGISTRATION.md),
[native result](../../GERUND_SCALAR_NETWORK_V1_RESULT.json),
[paired audit and counterexample](../../GERUND_SCALAR_NETWORK_AUDIT_V1_RESULT.json),
and [MLP17 dossier](../MLP17_CURRENT_UNDERSTANDING.md).
