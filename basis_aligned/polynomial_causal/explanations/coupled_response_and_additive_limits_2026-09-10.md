# Why the current response needs a joint computation

Updated September 10, 2026, 09:12 UTC.

The next source-reuse test is also complete. Swapping the full MLP8 mixed source
between original and fronted sentence layouts fails both simple reuse rules in
all 16 pairs. The same source displacement has different effects in the two
recipient contexts. An executed bound puts the unavoidable error of a single
context-free effect prediction at 14.5–22.4% of the larger native source effect.
This still permits one shared nonlinear operation with explicit context inputs;
the new section below explains that distinction.

The mathematical review gives a useful exclusion: **we cannot explain the measured
response as two independent additive output branches, and final normalization does
not account for the missing interaction.** We still have not extracted a reusable
semantic circuit satisfying the four requested properties.

I returned to the original [handoff](bilinear_circuit_reconstruction_codex_handoff.md)
and [pilot report](bilinear_reconstruction_pilot_report.md). The pilot established
faithful execution but found no new shared computation. Its recommendation to study
joint read–route–write operations is more relevant here than another search for
duplicate weights or a smaller activation basis.

The current example concerns how object number and the kind of a third noun jointly
affect reflexive predictions. It is no longer the is/was experiment. The interventions
start at MLP8, a bilinear feed-forward layer, and examine how later attention and
MLP computations respond. These are controlled response measurements with native
background retained; they are not an independently executable language algorithm.

## What the two new tests establish

First, the [directional-cut test](../MLP8_COUPLING_DIRECTIONS_V1_RESULT.json) compares
four cases: independent response banks, attention-to-MLP response enabled,
MLP-to-attention response enabled, and both enabled. A bank is a set of module outputs
from a specified counterfactual run, installed to block a particular dependency.
Attention9 stays native in every case. The source edit and sentence bank stay fixed.

Neither one-way explanation passes in any of the 32 lexical/layout groups. The
interaction has 42.1–67.2% of the directional coupling's centered mixed-vocabulary
norm. Here a norm measures the size of an output-change vector; this is not a
percentage of all model behavior explained. All replay checks pass exactly.

Second, I derived and tested whether the final output reader creates this interaction.
The [native test](../MLP8_COUPLING_READOUT_V1_RESULT.json) says it does not: the reader
contribution is at most 0.601% of the measured interaction across the registered
readouts and table types. For the centered mixed-vocabulary measurement it is
0.131–0.289%. The internal-state term passes the 10% fidelity requirement in every
group. The complete final reader remains essential for computing logits; the result
concerns where this particular interaction originates.

The first test used 512 forwards over 8,192 sequence instances in 9.06 executor
seconds. The second used the same forward count plus 192 decoder batches over
6,144 states in 19.03 seconds. Both ran through the managed GPU runner. These times
exclude scientific design, implementation and review. No model training occurred.

## Separate internal interaction from nonlinear reading

Let x00, x10, x01 and x11 be final residual vectors under the two intervention
switches. The residual is the vector passed to the final normalization and vocabulary
projection. Independent additive changes would predict

\[
\bar x_{11}=x_{10}+x_{01}-x_{00}.
\]

The actual decoder is

\[
D(x)=30\tanh\left(\frac{W x}{30\sqrt{x^T x/d+\epsilon}}\right),
\]

where W contains the vocabulary output weights and d=1152. The square-root
denominator is root mean square normalization; it depends on every residual
coordinate. Its general definition is described by
[Zhang and Sennrich](https://arxiv.org/abs/1910.07467); this experiment uses the
checkpoint's actual implementation and epsilon.

Write Zab=D(xab). Using the same sign convention as the directional test,

\[
\begin{aligned}
I &= Z_{10}+Z_{01}-Z_{00}-Z_{11},\\
I_{reader} &= Z_{10}+Z_{01}-Z_{00}-D(\bar x_{11}),\\
I_{internal} &= D(\bar x_{11})-Z_{11}.
\end{aligned}
\]

Thus I=I_reader+I_internal. The reader term can be nonzero even when the internal
state is perfectly additive. The CPU control deliberately constructs that case.
For the trained model, however, the internal term accounts for this interaction
within the stated tolerance. This is an attribution relative to the chosen baseline
and synthetic endpoint edit, not a unique decomposition under every representation.

The synthetic state was computed in FP64 and cast once to native FP32. Its norm
was recomputed. Independently decoding the five captured native/counterfactual
states reproduces full-vocabulary outputs within 2.01e-5 maximum absolute logit
error and passes the registered relative tolerance. The original intervention
outputs replay exactly. No tolerance was changed after seeing results.

## A sharp limit on any additive explanation

Suppose we allow arbitrary refitting, but require the output model to have the form

\[
\widehat Z_{ab}=B+F(a)+G(b).
\]

B is background; F depends only on the first switch; G only on the second. Such a
model has zero four-corner interaction. If eab=Zab−Zhat_ab, the triangle inequality
therefore gives

\[
\|I\|\leq\sum_{a,b}\|e_{ab}\|\leq4\max_{a,b}\|e_{ab}\|.
\]

At least one corner must have error at least ||I||/4. This bound is sharp: subtract
one quarter of the signed interaction at each corner, with alternating signs,
and all four errors attain that magnitude while the fitted table becomes additive.
The [executed control and measurement](../ADDITIVE_SWITCH_LOWER_BOUND_V1_RESULT.json)
verify closure and attainment within 8.89e-16 on 32 random vector-valued fixtures.

The result also holds after any fixed linear measurement, including vocabulary
centering and the object-number × noun-kind interaction projector Q. Applying it
to the saved native measurements gives a minimum worst-corner error of
**10.52–16.80% of the directional coupling's norm**, across all 32 groups.
Consequently every group excludes a uniformly 10%-accurate additive output model,
even with freely chosen branches. This is a numerical conclusion about this fixed
intervention domain, not a theorem about arbitrary language inputs.

## Consequence for weight folding and shared modules

Weight folding is still legal and useful. For a linear consumer C of a bilinear MLP,

\[
C M(u)=(C D_{MLP})[(Lu)\odot(Ru)]+C b.
\]

It exposes exactly which products the consumer reads. A subsequent nonlinear
consumer needs its normalization and interactions retained. The new evidence rules
out treating later attention and MLP responses as two independent final-output
circuits in this example. A useful candidate must represent their joint operation,
with explicit inputs and distinct consumer uses.

This does **not** imply that a whole attention group and a whole MLP group should
be merged into one named circuit. That would retain the unexplained computation.
Nor does the lower bound rule out additive internal writes followed by some learned
nonlinear shared operation. The native test specifically rejects the existing
final decoder as a sufficient explanation of the coupling.

The next scientific object is therefore the internal joint operation and its
weight-defined products, rather than another architectural split. No head, layer,
rank or intervention-amplitude sweep is licensed by these results. All 545,902,902
native parameters and counterfactual input dependencies remain charged; structural
saving is zero. Fresh/OOD prediction, independent extraction, selective removal and
reuse of an identified computation remain open.

## Full-source interchange reveals different consumer responses

The directional switches above are interventions on module execution. A clamped
MLP uses recorded output rather than evaluating its bilinear function on its current
input. Thus interaction across those switches must not be directly labeled a newly
formed bilinear input product. Even two gated linear identity maps can produce the
switch function a*b. This is consistent with the earlier linear-path control.

I therefore tested a source interface directly before attempting another product
attribution. The [full-source interchange](../MLP8_SOURCE_STAGE_INTERCHANGE_V1_RESULT.json)
uses the MLP8 object-number × noun-kind component C in both sentence layouts. It
maps three positions: the first token where both factors are available, `to`, and
the final action. The first position has different noun roles in the two layouts;
this map specifies information timing, not equal grammatical roles. The mixed
source is exactly zero before those positions in every tested group.

For recipient r, remove its own component C_r from the native MLP output, leaving
background B_r. Then install either its own component or the mapped donor C_p.
Every downstream attention and MLP computation responds normally. The effect is

\[
E_{pr}=z_r(B_r+T C_p)-z_r(B_r).
\]

The earlier COMMON_INTERCHANGE test swapped a partial attention9 output. This
experiment instead delivers the MLP8 source to **all** consumers. It used 256
forwards over 4,096 sequence instances in 5.43 executor seconds, with zero fitting.
Native outputs, prior full-source removals and self-interchanges replay exactly.
The shared first-value state stays unchanged. All instrument checks pass.

Neither effects following the producer alone nor effects following the recipient
alone meet the 10% criterion in any of the 16 pairs, including the task-only checks.
Full-vocabulary producer-following errors are 27.5–41.0%; recipient-following errors
are 86.8–160.2%. Source–recipient interaction is 28.9–44.8% of the larger native
source-effect norm. We do not promote the producer rule merely because its errors
are smaller. These are previously opened layouts, not fresh OOD evidence.

## What the second consumer tells us

The four measurements isolate the effect of the *same* source displacement C0→C1
in each recipient:

\[
d_0=E_{10}-E_{00},\qquad d_1=E_{11}-E_{01}.
\]

Their difference is exactly the measured source–recipient interaction. Changing
the removal-reference offset in either recipient cancels out of these differences.
For any common context-free effect prediction h,

\[
\max(\|d_0-h\|,\|d_1-h\|)\geq\tfrac12\|d_1-d_0\|.
\]

The midpoint h=(d0+d1)/2 attains the bound. The
[executed analysis](../SOURCE_CONSUMER_EDIT_OBSTRUCTION_V1_RESULT.json) verifies
offset cancellation and midpoint attainment within 4.45e-16 on 32 random fixtures.
On the native centered mixed-vocabulary effects, the bound is **14.47–22.42% of
the larger native source-effect norm**, in every pair. It excludes a uniformly
10%-accurate context-free effect prediction at this interface. It does not exclude
a context-aware program or a different justified interface.

Crucially, different effects do not imply different arithmetic. One shared square
operation F(b,c)=(b+c)^2 gives source effect 2bc+c^2 relative to F(b,0). With c going
from 1 to 2, its effect is 3 when b=0 and 9 when b=3. The same operation is reused;
the context operand differs. This exact counterexample is included in the control.

That is the stronger target for the next decomposition: identify both the source
operand and the context operand of an internal shared operation, then account for
its separate consumers. Calling an activation portable, or insisting its effect
be context-free, is insufficient. No position, gain, head or rank sweep follows
from this null. All native weights and counterfactual inputs remain charged, and
the four-property circuit goal is still incomplete.
