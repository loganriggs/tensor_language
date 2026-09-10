# Why the current response needs a joint computation

September 10, 2026, 09:00 UTC.

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
