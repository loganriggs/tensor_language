# Shared grammatical writes work; the scalar-only predictor does not

**Latest scientific result — 10 September, 18:13 UTC:** the lexical/form
interaction needs the grammatical scalar AND the changed complementary state.
Its token-specific scalar readout alone is insufficient, even when supplied with
the true changed norm. This directs the next test toward the complementary
change's producers, rather than another readout approximation.

## Can fixed scalar-reader coefficients explain the cross-task effect?

From the saved final state, write h=s*e+h_perp. A token numerator is

    U_t*h = a_t*s + c_t,
    a_t = U_t*e,   c_t = U_t*h_perp.

Its score is30*tanh((a_t*s+c_t)/(30*rho)), where rho is the final RMS. The a_t
coefficients are fixed by weights, while c_t depends on the remaining context.
We tested the eight choices of base/edited s, c and rho. A separate physical
scalar-only curve keeps h_perp fixed and predicts its own norm from
rho(s)^2=(||h_perp||^2+s^2)/1152+eps32. No model run or checkpoint load was needed.

| Predicted change | Frame1, original / cyclic verb | Frame2, original / cyclic verb |
|---|---:|---:|
| Physical scalar-only | .832 / .902 | .951 / .950 |
| Scalar plus actual changed norm | .818 / .876 | 1.045 / 1.072 |
| Complement plus actual changed norm | .878 / .949 | 1.357 / 1.322 |
| All three changed quantities | 2.7e-6 / 2.6e-6 | 3.7e-6 / 3.1e-6 |

Entries are relative errors in the two lexical-margin changes under F. All
three simplifying hypotheses fail the registered .20 bound. The reference norms
are4.24–6.03, so this is not a zero-effect denominator. Native-score bridges are
within6.3e-6; a separately reloaded physical curve reproduces its base score.
The physical scalar-only curve also leaves35%selected-four-score effect error.
Its640 stored coefficients are a conditional readout summary, not an extracted
context producer or a repair of the earlier command failures.

A further CPU check makes the cancellation explicit. Before softcap, the change
splits exactly into a scalar term, complementary numerator term and norm term:

    delta_p = a*delta_s/rho1 + delta_c/rho1
              + (a*s0+c0)*(1/rho1-1/rho0).

Multiplying each token's terms by its endpoint softcap secant gives exact final
score accounting. This secant is delta_score/delta_p, using the derivative when
delta_p is zero. After taking lexical margins, the scalar and complement terms
partly oppose one another: cosine ranges from-.30 to-.62. The sum of their norms
and the norm term is1.91–2.67times the norm of the net change. The secant uses both
native endpoints, so this is an accounting identity, not an independently
predictive or separately intervened set of causal effects. It cannot rescue the
failed scalar/complement-only tests.

Receipts: [registered test](../../LEXICAL_FORM_READOUT_FACTORS_V1_PREREGISTRATION.md),
[CPU result and paired intervals](../../LEXICAL_FORM_READOUT_FACTORS_V1_RESULT.json),
[cancellation audit](../../LEXICAL_FORM_NUMERATOR_CANCELLATION_V1_RESULT.json),
[physical-curve reload](../../LEXICAL_FORM_READOUT_RELOAD_V1_RESULT.json).

The [18:14 hourly review](../../HOURLY_STRATEGIC_REVIEW_2026-09-10_1814.md)
records four native receipts and14.95-minute median spacing. Publication and
restoration were taking too much time. The active startup guide is now215lines;
all4335old lines are preserved with an identical hash in session_history.
Results now live once in this explanation, with short links elsewhere.
Next reviews: hourly19:14 UTC, mathematical19:49 UTC. The full goal remains open.

## Previous result — 18:03 UTC

**Result — 10 September, 18:03 UTC:** the fixed lexical and grammatical
commands show substantial reuse, but fail the complete separation/composition
tests. The lexical command mostly preserves form margins. The form command
changes lexical margins too much, and their joint command chooses the intended
word/form in only 14/16 and 13/16 cases. The predefined interaction reader
accounts for much of the measured cross-task drift; its producer remains open.

## Testing two reusable commands and their joint effect

For each pair of verbs, we evaluated four native contexts: original or cyclic
next verb, each with a bare-verb or an “-ing” grammatical cue. The four scored
tokens are the two verbs in both forms. The lexical command L changes only the
primed token's first-value vector. The form command F sets all 36 output e
scalars to the original verb's ing-cue values. We reuse that SAME F command
with the second verb; we do not supply a second set of context-specific scalars.
The joint command installs L and F together on the original base context.

| Test | Frame 1 | Frame 2 |
|---|---:|---:|
| L lexical recovery, bare / ing cue | .842 / .841 | .809 / .785 |
| F form recovery, original / cyclic verb | .952 / .950 | .863 / .864 |
| L drift in form margins, relative error | .090 / .098 | .050 / .046 |
| F drift in lexical margins, relative error | .115 / .117 | .119 / .169 |
| Joint four-token effect error versus native joint change | .177 | .198 |
| Joint correct choices among four tokens | 14/16 | 13/16 |
| Error from adding singleton full-vocabulary effects | .123 | .083 |

The recovery bars were .80; the preservation bars were .10. All L preservation
point estimates pass, but lexical recovery in the second frame's ing cue misses
.80. All F recovery point estimates pass, but every F preservation case fails.
The joint test additionally required all 16 correct choices and full-vocabulary
addition error at most .10. It fails. None of these thresholds was relaxed.

Native capability also limits the claims. Frame 1 has 16/16 correct four-token
choices at every corner. Frame 2 has 15/16 at each ing corner: the shop context
prefers traveling over shopping, and a laugh context prefers traveling over
laughing. These are retained failures; the positive bare-cue capability result
from the previous run did not establish ing-cue capability. Consequently all
three combined reuse/selectivity/composition predicates fail as registered.

## What the structured readers reveal

For four unembedding rows ordered [word0-bare, word0-ing, word1-bare, word1-ing],
we use the exact decomposition

    U(word,form) = mean + sign_word*lexical + sign_form*form
                       + sign_word*sign_form*interaction.

The components are fixed averages and contrasts, with no fitting. Each folds
through MLP17 to a reader of its native products. Reconstructing the original
rows is exact; reconstructing their folded product readers has maximum error
1.16e-14. Normalization and softcap remain explicit. Applying the same contrast
basis to FINAL scores measures final-score interaction, which can include
nonlinearity at readout as well as earlier computation.

This makes the preservation failure more informative. Write the four final-score
coordinates as mean, ell, phi and iota. The two lexical margins are
2*(ell-iota) and 2*(ell+iota). Therefore their squared change is exactly

    8*(change_in_ell^2 + change_in_iota^2).

The paired CPU audit finds that iota accounts for **58–81% of the form command's
squared lexical-margin drift**, depending on frame and starting verb. The rest
is change in the common lexical coordinate. This is orthogonal accounting of
observed scores, not a percentage causally mediated by an identified module.
It favors investigating an explicitly coupled readout operation over assuming
two independent semantic effects. The arbitrary upstream computation is still
charged and unexplained.

## Composition and the agreement control

On the four selected scores, adding the singleton effects predicts the ACTUAL
joint intervention with .047/.041 relative error. That is much better than the
full-vocabulary result, but it predicts the imperfect joint command, not the
native joint lexical/grammatical change. We must retain its 14/16 and 13/16
choice failures. A CPU diagnostic adds the saved residual-state changes and
then recomputes exact normalization/softcap; errors are .046/.065. It does not
provide a normalization-only repair or an independently initialized program.
All required singleton states still come from the native model.

We now measured the previously missing runs/run readouts. The lexical value
swap changes their margin by **.233 nats in average absolute value**, exceeding
the new .10 margin limit; its paired interval is [.160,.310]. The original
probability-preservation failure also replays at .705 nats. Thus there is a real
change in grammatical-form preference under this intervention, in addition to
word-probability movement. The natural cyclic context change moves that margin
by .466 nats, reported for scale; it does not change the registered verdict.

All numerical instruments pass. The managed run finished18:03:20 after23batched
forwards/368sequences and1.56seconds of executor time. Final states, selected
scores and folded readers were saved (7,476,061bytes). The paired audit,
readout/state composition diagnostic and drift-component accounting were all
executed on CPU with no checkpoint load or extra model forwards. The full
four-property goal remains open: reusable partial commands are not sufficient
independent circuits, and selected-score accuracy is not full-model prediction.

Receipts: [preregistration](../../LEXICAL_FORM_INTERCHANGE_V1_PREREGISTRATION.md),
[native result](../../LEXICAL_FORM_INTERCHANGE_V1_RESULT.json),
[paired/state audit](../../LEXICAL_FORM_INTERCHANGE_AUDIT_V1_RESULT.json),
[drift components](../../LEXICAL_FORM_DRIFT_COMPONENTS_V1_RESULT.json).

## Previous result — 17:48 UTC

**Result — 10 September, 17:48 UTC:** the shared first-layer value stream
transfers most of the primed-word preference (84%/81%) across the two frames,
but neither supplies the full context gate nor passes probability preservation.
This is partial lexical transfer through a known shared input channel, not an
independently extracted or selectively removable circuit.

## A shared input used across attention layers

The previous test found no sufficient single-module source. We next tested a
source that crosses module boundaries: the first attention layer produces a
value vector for each token, and attention layers 1–17 reuse it. This channel
and its exact token-only producer were already documented in the module dossiers
and earlier v173/v174 work. The new question is whether it carries the lexical
context required by the current token readers.

Every cyclic pair differs at exactly one token—the primed verb at position 3.
We separated its influence into two ports: the ordinary token stream and the
shared first-value cache returned after attention 0. Four combinations give
native base, native donor, base with donor values, and donor with base values.
Attention 0's own output stays with the ordinary stream; later routing, local
values and MLPs recompute normally. An own-cache no-op is the fifth run.

| Measurement | Frame 1 | Frame 2 |
|---|---:|---:|
| Shared-value-only lexical recovery | .842 | .809 |
| Remaining-stream-only lexical recovery | .174 | .208 |
| Shared-value-only context-gate transfer | .486 | .607 |
| Shared-value-only context-gate error | .557 | .567 |
| Remaining-stream gate-change magnitude | .534 | .599 |
| Full-vocabulary factorial interaction | .365 | .328 |

Lexical recovery measures the movement from preferring the original primed
bare verb to preferring the donor's bare verb. All 16 pairs in each frame prefer
their own verb at both native endpoints. Its paired intervals are [.779,.895]
and [.748,.871]. These are opened, authored panels; this is not broad OOD or
a claim that the population recovery exceeds .80 with confidence.

The context-gate source hypothesis fails: each branch retains substantial
influence. Shared-value-only full-vocabulary effect errors are .480/.560, and
the factorial interaction shows that adding branch effects also misses important
response. A program could model this interaction explicitly, but we have not
extracted the contextual consumer program.

The combined lexical-transfer/selectivity prediction also fails. The agreement
control's mean absolute correct-token CE change is **.705 nats**, versus a .10
limit (paired interval [.402,1.078]). Seven examples worsen, nine improve:
mean harm .310 and mean improvement .395 give signed average -.085. Improvement
on average does not satisfy preservation.

There is an attribution limit here. Correct-token CE is -z_correct+logsumexp(z),
whereas agreement log-odds is z_correct-z_foil. The former can change when the
latter is fixed. For example, logits [2,1,0] becoming [1,0,2] increase the first
token's CE by exactly 1 nat while leaving its margin over the second token at 1.
This CPU witness does not explain the native result: edited foil scores were
not saved, so the native grammatical-margin effect remains unmeasured. The
registered probability-preservation failure stands. The next useful test must
separate lexical identity from grammatical form with explicit token readers
and shared inflection contrasts, rather than infer either from one token's CE.

All instrument checks hold: exact token-to-first-value reproduction, own-cache
no-op and saved-state replay; all 48 base and 48 rotated grammatical endpoints
are correct. The managed job ran 17:48:20–17:48:23, with 15 batched forwards,
240 sequences and 1.43 seconds of executor time. All native parameters remain.
The paired CPU audit and the signed-control/counterexample analysis were both
executed after the result.

Receipts: [registered test](../../TOKEN_CONTEXT_BROADCAST_V1_PREREGISTRATION.md),
[native result](../../TOKEN_CONTEXT_BROADCAST_V1_RESULT.json),
[paired audit](../../TOKEN_CONTEXT_BROADCAST_AUDIT_V1_RESULT.json),
[control interpretation](../../TOKEN_CONTEXT_BROADCAST_CONTROL_CE_V1.json).

## Previous result — 17:37 UTC

**Result — 10 September, 17:37 UTC:** none of the 36 single-module
output swaps supplies enough of the token-dependent context signal to pass
in both grammatical frames. All controls pass. This is a source-localization
null, not a rejection of distributed circuits. The saved-reader CPU audit also
shows why both unembedding views must retain token-specific information.

## The two backward-unembedding paths

The user’s distinction is the working plan:

1. **Individual tokens.** Start from each token's unembedding vector u_t, or a
   specified contrast u_answer-u_foil. Fold that reader backward to expose the
   products and earlier state it reads. Compare different tokens after folding.
2. **Shared structure.** Decompose the same vectors into clusters, hierarchy
   contrasts, shared directions and token-specific remainders. Fold each piece
   backward and test whether the shared pieces correspond to reusable operations.

For example, a hierarchy can represent a token reader exactly as

    u_t = root_mean + sum(child_mean - parent_mean along its path) + remainder_t.

For a bilinear MLP m(x)=D[(Lx)*(Rx)]+b, a reader v gives

    v·m(x) = x^T Q(v) x + v·b,
    Q(v) = sym(L^T diag(D^T v) R).

Here sym(A)=(A+A^T)/2. Crucially, Q is linear in the reader v. Therefore every
shared component and remainder can be folded separately and summed exactly.
The same principle applies to a fixed-input attention output reader through OV;
its QK routing and normalization remain explicit. Further backward substitution
can expose common products across readers, but exact rewriting alone does not
establish fewer independent computations.

The previous fixed 16-leaf means-only experiment failed. That closes its
particular approximation, while shared subterms and hierarchy contrasts remain
open. We retain remainders so a cluster does not silently erase a word's actual
computation. The circuit test is whether a shared piece has explicit producers
and consumers and passes held-out prediction, extraction, selective manipulation
and composition—not whether words merely cluster near one another.

## Which earlier outputs produce the required context signal?

The local token-score program below needs context-dependent coefficients.
For its fixed normalized grammatical direction e, the token contrast's linear
MLP17 response reads k_v=2Q17(v)e from the MLP's normalized input u. Remove the
part along e to obtain

    a_v = e^T Q17(v)e,
    k_perp = k_v - 2*a_v*e,
    tau = k_perp·u.

Tau is the context signal: it contributes delta*tau when we edit the input by
delta*e. We changed the primed verb while keeping each grammatical frame fixed,
using the next of the 16 existing base contexts as donor. The recipient's token
reader stays fixed when applied to both contexts.

We restored each of the 36 attention/MLP outputs individually at the final token
position, then let the later model recompute. A candidate had to transfer at
least .50 of the natural gate change, leave at most .50 relative gate error in
both target frames, and change agreement-control CE by at most .10 nats on
average in absolute value. **No site passed.** Even the largest target transfers
were only .359 and .370; the smallest errors were .657 and .711. These are
internal gate measurements, not recovered lexical behavior.

All 48 base grammatical contrasts were correct. Restoring every cached output
reproduced the donor context gate exactly. Own-cache restoration changed no
logits, and changing MLP17's output had exactly zero effect on its own input
gate. The managed GPU run used 117 batched forwards, 1,872 sequences and 3.51
seconds of executor time. Every native weight remains required.

The CPU follow-up reports paired intervals for all sites, with no selected
site promoted. It also splits the saved token-specific readers into their
panel mean and individual remainder. Keeping only the mean leaves .649/.905
relative gate-change error in the two target frames; keeping both reconstructs
the gate changes within 2.3e-13. These are descriptive results on already opened
rows, not a new held-out cluster discovery. Their cross terms even have opposite
signs between frames, so component magnitudes are not additive causal shares.
The agreement panel uses one fixed token contrast, so its zero remainder is
expected by construction, not extra evidence of shared computation.

This leaves distributed production and shared subterms as the next questions;
it does not license a post-hoc best-layer subset or identify any whole module
as a semantic circuit. The exact MLP16 product-reader coefficients were saved
for reuse, without claiming MLP16 is the producer.

Receipts: [registered test](../../TOKEN_CONTEXT_SOURCE_V1_PREREGISTRATION.md),
[native result](../../TOKEN_CONTEXT_SOURCE_V1_RESULT.json),
[CPU audit](../../TOKEN_CONTEXT_SOURCE_AUDIT_V1_RESULT.json).

## Previous result — 17:17 UTC

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
