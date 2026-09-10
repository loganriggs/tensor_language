# Shared grammatical writes work; the scalar-only predictor does not

**Latest result — 10 September, 17:17 UTC:** both token-specific score changes
and final normalization are needed for this grammatical MLP response. Supplying
the correct norm does not repair the old two-reader writer. A different, exact
local program that retains the actual answer/foil readers and the shared norm
polynomial does predict their scores and successive edits accurately. Its
initial coefficients still depend on the native model.

## Token readers and a shared normalization computation

The final score of a token is obtained by reading the residual with that token's
unembedding vector, dividing by the residual's RMS magnitude, and applying the
model's softcap. The first quantity is its numerator; the second is a denominator
shared by every token. Existing calibration work already studied these two uses.
The new test applies that split to the specific grammatical input edit whose
two-reader output replacement just failed.

We compared the actual edit with changing only its token numerators, changing
only its final RMS, or giving the old two-reader writer the true edited RMS.

| Retained change | Full-vocabulary effect error, frame 1 | Frame 2 |
|---|---:|---:|
| Token numerators only | .476 | .585 |
| Final RMS only | .935 | .980 |
| Old two-reader numerator plus true RMS | .879 | .916 |
| Full numerator and RMS computation | 1.04e-5 | 9.77e-6 |

All three simplification hypotheses fail the .10 criterion. For the old writer
with true norm, paired intervals are [.873,.884] and [.911,.922]. Therefore
missing normalization alone cannot explain its failure; omitted token-specific
numerator information also matters. These error norms are not additive causal
percentages. The numerator/RMS endpoint interaction is .045/.062 of the full
effect norm on these rows, but their producers have not been separated into
independently executable circuits.

## A local score program with the required readers

The bilinear response gives the final state along this one edit direction as

    h(delta) = h0 + delta*b + delta^2*a.

Here h0 is the native final state, b is the context-dependent linear response,
and a is the weight-derived quadratic response. Folding a token vector U_t
backward gives three numerator coefficients:

    n_t(delta) = U_t*h0 + delta*(U_t*b) + delta^2*(U_t*a).

The squared RMS is a degree-four polynomial rho(delta). Its five coefficients
come from h0·h0, 2h0·b, 2h0·a+b·b, 2b·a, and a·a, divided by the residual width;
the native epsilon is added to the constant. The predicted score is

    score_t(delta) = 30*tanh(n_t(delta)/(30*sqrt(rho(delta)))).

Thus two token numerators and one shared norm polynomial require **11 numbers
per initialized context**. The token-specific coefficients preserve differences
between words, while the same normalization computation is reused by both.
Shifting the polynomials' argument updates the state for the next edit.

On all64 native test rows, maximum answer/foil score error is8.03e-6 and maximum
margin error8.73e-6. Sequential-edit composition error is5.33e-15. A separate
CPU reload test on saved states also passes. The two scores determine the
answer-versus-foil log-odds, but not full-vocabulary probabilities or CE: those
still require the other scores. This is a conditional local score program,
not a model that produces its initial state from a sentence. h0, b and their
required context moments are still computed using native dependencies.

The successful managed run used12 forwards/192 sequence instances in1.37seconds.
Its3,010,885-byte state cache enables further CPU analysis. An initial run saved
states but failed on an artifact-size reporting call; its bytes and log are
preserved. The repaired run's states match them exactly. No scientific bars or
intervention choices changed. All native weights remain required.

The next extraction question is how to compute the needed token/context readers
and shared norm moments from upstream operations. This result identifies the
required readout information; it does not justify another fitted output rank.

Evidence: [protocol](../../GERUND_READOUT_FACTORIAL_V1_PREREGISTRATION.md),
[successful native result](../../GERUND_READOUT_FACTORIAL_V2_RESULT.json),
[paired and saved-state audit](../../GERUND_READOUT_FACTORIAL_AUDIT_V1_RESULT.json),
and [hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-10_1714.md).

## Earlier native context result

**10 September, 17:04 UTC:** the small program correctly
predicts its two selected readouts on native sentence states, but it does not
yet provide a useful replacement for the intervention. Context substitution
fails, and even the exact two-reader output leaves almost all of the full
vocabulary effect unexplained.

## Testing the program on native contexts

The intervention now changes the grammatical scalar at the last MLP's
normalized input to its natural donor value. The input complement stays at the
recipient value, and the model computes the actual MLP output and final logits.
This is a local MLP17 intervention, distinct from the earlier all-layer edits.

We separate the context gate from the scalar itself:

    s = e^T u;
    tau = K u - 2*a*s;
    response = delta*(tau + 2*a*s) + delta^2*a.

The test compares native tau with two fixed alternatives: the gate from the
first old base sentence, or the next verb's gate in the same new frame. Neither
uses a fit or a chosen best donor. The initial scalar s remains native in all
cases; even a successful gate replacement would not extract its producer.

| Response context | Selected-reader error, frame 1 / 2 | Full-vocabulary effect error, frame 1 / 2 |
|---|---:|---:|
| Exact native context | 4.6e-7 / 3.7e-7 | .990 / 1.009 |
| One old reference context | .261 / .250 | .996 / 1.001 |
| Next verb in the same frame | .198 / .086 | .991 / 1.010 |

The context criterion required at most .10 error in both target frames. Both
substitution hypotheses fail. Reference-context intervals are [.220,.293] and
[.226,.270]; the first-frame cyclic interval is [.111,.290]. No new gate, donor,
rank, or threshold was chosen to repair the result.

For the full-vocabulary comparison, the two predicted reader changes are
converted into their fixed minimum-norm physical output vector. That vector
preserves those two linear readouts but omits all other output directions. It
then passes through the real final normalization, unembedding and softcap.
The exact-context row shows that fixing the context alone cannot rescue this
writer: its full-effect intervals are [.987,.994] and [1.008,1.011]. An error
near1 means about as much error norm as the effect being predicted, not a
percentage of incorrect tokens.

The actual local intervention changes correct-token CE by +.715/+.420 nats and
recovers .098/.074 of the full native cue-induced task margin. The exact-reader
writer's CE prediction errors are .775/.468 nats. Matching the chosen internal
readouts therefore misses behaviorally important changes. On the agreement
control, even exact-context full-effect error is .871. Its earlier selectivity
failure remains unchanged.

All64 native pairs are capable, native bridges pass, and the exact FP64 formula
passes its registered checks. The managed experiment used13 body forwards over
193 sequence instances, including one fixed reference, in1.33seconds. The CPU
paired audit is complete. This validates the local equation while rejecting
the proposed contextual simplifications and behavioral replacement. The next
question is which omitted token-specific outputs and normalization effects
carry the discrepancy, using the existing calibration dossiers as prior art.

Evidence: [native protocol](../../NATIVE_RESPONSE_GATE_V1_PREREGISTRATION.md),
[result](../../NATIVE_RESPONSE_GATE_V1_RESULT.json), and
[paired audit](../../NATIVE_RESPONSE_GATE_AUDIT_V1_RESULT.json).

## Earlier mathematical result

**10 September, 16:54 UTC:** we now have a saved,
small executable program for the last MLP's response along two chosen output
readers. It predicts repeated edits along the grammatical direction and composes
them exactly, once two context scalars are initialized. This is a local response
program; equal-state counterexamples show that it cannot predict all other
output directions.

## What the mathematical cycle produced

For the two readers, collect the weight-derived context maps into a matrix K,
and their squared-term coefficients into a two-entry vector a. Initialize the
two context values q=Ku from the normalized MLP input. For an input edit of
size delta along e, the whole response program is

    predicted change = delta*q + delta^2*a;
    updated context state = q + 2*delta*a.

The second equation makes repeated same-direction edits compose: applying two
commands in sequence predicts the same response as their combined command.
The program predicts changes in the two selected MLP readouts, not absolute
MLP outputs, normalized token probabilities, or a sentence's answer. It still
needs the initial full input to compute q; those upstream producers have not
been extracted.

The trained MLP17 test matched direct evaluation within 1.01e-13 relative error
and composition within 1.59e-13. A separate reload test used only the saved
program, with new synthetic contexts and commands, and also passed. It stores
2,306 float64 coefficients:18,448 raw tensor bytes, or20,613 bytes in its saved
file. These are local program costs, not a reduction in the full model's bill.

The same test constructed context pairs with identical q, identical grammatical
scalar, and identical norm. Their other output responses still differed. Any
program using only those shared features must leave at least 11–24% relative
full-output error on these pairs. The contexts are synthetic internal states,
not natural-text OOD examples. Thus local selected-reader extraction and
composition are established, while full-output closure is explicitly rejected.

This mathematical cycle supplied both a reusable executable operation and a
test that prevents overstating its scope. The [review and proof](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1649.md)
compare exact reduction methods with the actual normalized model. The
[native-weight result](../../SELECTED_READER_RESPONSE_V1_RESULT.json),
[saved program](../../SELECTED_READER_RESPONSE_V1_PROGRAM.pt), and
[independent reload check](../../SELECTED_READER_RESPONSE_V1_RELOAD_AUDIT.json)
record what ran. The next missing part is producing the context state and
handling the remaining downstream readers.

## Earlier native consumer result

**10 September, 16:42 UTC:** MLPs read the grammatical scalar
and write consequential changes into other directions. Holding those scalar
reads at their original values reduces target recovery by **29 percentage
points in both constructions**. It also removes about **70% of the agreement
damage on the tested base endpoint**. This identifies an important use of the
shared scalar, while the remaining prediction error still prevents a closed
scalar-only explanation.

## What the MLP consumer test changed

We kept the same distributed output swaps and zeros. At each MLP's final token
position, we additionally held its grammatical input scalar at the natural
recipient value. Everything else in that MLP input remained live. This operation
acts after the model's RMS normalization; we did not renormalize the edited
input. It therefore tests a specific internal read, not a naturally occurring
residual state. Attention and normalization-induced changes to other coordinates
can still transmit effects.

Let u be that normalized MLP input, e the fixed grammatical direction, and
delta the change needed to restore its original scalar. The exact bilinear
output change is

    Delta m = D[delta*((Le)*(Ru)+(Lu)*(Re)) + delta^2*(Le)*(Re)].

Here * multiplies corresponding coordinates. The first term combines the scalar
change with the current context; the second is the scalar's squared contribution.
We add this weight-computed correction to the actual MLP output, then perform
the original output-scalar intervention. Because that final operation fixes the
output coordinate along e, differences between the two arms travel through the
other output directions. The earlier v184/v185 work already derived this kind
of expansion; the new evidence is this grammatical consumer intervention.

| Measurement | New frame 1 | New frame 2 |
|---|---:|---:|
| Original distributed swap recovery | .952 | .863 |
| Recovery with MLP scalar reads held fixed | .660 | .573 |
| Recovery lost | **.291** | **.291** |
| Original scalar-prediction relative error | .649 | .645 |
| Error with MLP scalar reads held fixed | .153 | .084 |

The loss intervals are [.256,.324] and [.262,.319]. Both clear the registered
.20 necessity threshold for this intervention. The prediction-error criterion
required at most .10 in both frames, so it **still fails**: frame 1 has interval
[.149,.156]. This smaller error compares the predictor with an altered model
whose scalar reads were blocked; it is not an improved predictor of the
unchanged model's original intervention.

For agreement, base-endpoint zero-removal CE damage falls from .387 to .118
nats. The reduction is .269, interval [.248,.291], or 69.6% of its original
signed mean. The remaining damage is still above .10. This experiment tested
the base endpoint only; the earlier .548 figure averaged base and donor, so
these numbers must not be compared as if they were the same measure. Blocking
also changes target zero effects; it is not a selective repair of agreement.

## Shared scalar, different contextual reads

Folding an output reader v through the same correction gives a context reader

    k_v = L^T[(D^T v)*(Re)] + R^T[(D^T v)*(Le)] = 2 Q(v)e.

For u=se+z, with z perpendicular to e, the scalar-dependent contribution to
that output is s*(k_v^T z) + s^2*a_v, where
a_v=(D^T v)^T[(Le)*(Re)]. Thus a shared scalar can combine with a different
contextual quantity for each token or structured output reader.

The direct weight maps for output e and the runs/run agreement contrast have
per-layer cosines from −.020 to .736, median .205. They are generally distinct
context readers. This is a diagnostic of immediate MLP output functions, not
proof that two complete behavioral circuits are separate; later layers still
transform their writes. No layer was selected from these values.

The instrument passed: 28 forwards over 448 sequence instances, all 64 native
pairs capable, exact replay of previous swap/base-zero results, and maximum
relative direct-versus-folded MLP error 1.60e-6. Executor time was 1.72 seconds.
All native weights and contextual input producers remain necessary. The next
missing piece is predicting these context reads and their downstream use without
depending on the rest of the original model.

Evidence: [consumer preregistration](../../GERUND_MLP_CONSUMER_V1_PREREGISTRATION.md),
[native result](../../GERUND_MLP_CONSUMER_V1_RESULT.json), and
[executed paired audit](../../GERUND_MLP_CONSUMER_AUDIT_V1_RESULT.json).

## Earlier fresh-transfer result

**Fresh-transfer result — 10 September, 16:31 UTC:** the fixed distributed direction
transfers to new verbs and grammatical cues, recovering **95.2%/86.3%** of the
target effect. However, a closer subject–verb agreement control fails both
preservation and selective removal. The earlier positive selectivity result
therefore remains limited to its original control. This is useful shared
grammatical information, but it is not an isolated gerund circuit.

## The two unembedding paths

The token path follows an individual output vector u_t backward to ask what
produces that token's score. The structure path first writes the same vector as

    u_t = sum_j a_tj*s_j + r_t,

where s_j are shared directions, a_tj are token-specific coefficients, and r_t
is the remaining token-specific vector. A cluster mean is one special case;
hierarchical parent/child differences and overlapping grammatical directions
are other possibilities. Retaining r_t makes this an exact decomposition.
Deleting it is a separate hypothesis that needs testing.

For a bilinear MLP m = D[(Lx)*(Rx)] + b, a reader v has quadratic form

    Q(v) = sym(L^T diag(D^T v) R),
    v^T m = x^T Q(v) x + v^T b.

Because Q is linear in v, both views fold through the same calculation:

    Q(u_t) = sum_j a_tj*Q(s_j) + Q(r_t).

This exposes shared input products and their token-specific differences.
Earlier residual paths and attention can then be expanded with their actual
normalization and routing retained. The decomposition is additive before the
final softcap, given the shared normalized state; applying the softcap separately
to each summand would be incorrect. Shared weight terms still require causal
tests before they count as reusable circuits.

The coarse cluster-mean test already failed. The grammatical direction below
is a more specific shared component, and the new agreement failure shows why
its consumers and task-specific branches need to be distinguished.

## New verbs, new cues, and a closer control

The next experiment kept the direction and all 36 output sites fixed. Sixteen
new verbs were disjoint from both the eight weight-construction verbs and the
sixteen earlier test verbs. Cues changed to might/were and would/was, with
different sentence frames. A new control used he/they with runs/run agreement
across the same sixteen preceding contexts. It tests one agreement contrast,
not sixteen different agreement readouts. The tokenizer-driven correction to
that control was recorded before any model execution.

| Measurement | New frame 1 | New frame 2 | Agreement control |
|---|---:|---:|---:|
| Donor cue recovery | .952 | .863 | .133 |
| Swap mean absolute CE change | 2.549 | 3.264 | .108 |
| Zero-removal mean CE damage | .668 | 1.149 | .548 |

Target swaps intentionally change the answer, so their positive CE change is
expected. Control preservation required absolute CE change at most .10 and
absolute mean recovery at most .10: agreement fails both. Zero removal damages
agreement by .548 nats, far above its .10 preservation limit. We retain it as a
failed control rather than rename it as a target after seeing the result.

Paired 95% intervals are [.926,.978]/[.831,.898] for target recovery,
[.129,.138] for agreement recovery, and [.514,.583] for agreement removal
damage. The agreement swap-CE interval [.084,.132] crosses .10, but this does
not overturn its registered point-estimate failure or the other failed bars.
The modal-paraphrase and unrelated either/not controls pass swap preservation;
either/not also retains its earlier removal result.

All 96 native pairs have positive intended-answer margins at both endpoints.
Instrument checks and the old intervention replay pass. The managed run used
36 forwards, 576 sequence instances, and 1.76 executor seconds. No new fitting,
selected sites, relaxed thresholds, or independently extracted producers were
introduced. New experimental text is not proof of pretraining OOD.

The next question is whether shared grammatical information has distinct
context-dependent consumers. A decomposition must explain those branches
before it supports separate removal or predictable composition.

Evidence: [frozen new-data plan](../../GERUND_FRESH_TRANSFER_V1_PREREGISTRATION.md),
[native result](../../GERUND_FRESH_TRANSFER_V1_RESULT.json), and
[executed paired audit](../../GERUND_FRESH_TRANSFER_AUDIT_V1_RESULT.json).

## Earlier experiment: distributed writes and scalar feedback

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
