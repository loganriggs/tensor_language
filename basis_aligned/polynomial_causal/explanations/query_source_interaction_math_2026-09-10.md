# What the mathematics says about the shared attention circuit

10 September 2026. The original report is
[bilinear_reconstruction_pilot_report.md](bilinear_reconstruction_pilot_report.md),
alongside the [handoff](bilinear_circuit_reconstruction_codex_handoff.md).
Its important conclusion is that faithfully rewriting attention as tensors
does not make previously unknown circuits emerge automatically.

I followed its suggestion to examine the joint computation that reads
information, routes it and writes a result. **The new result is that the
shared attention heads need a combination of upstream query inputs on the
tested examples. No single input contribution suffices.** I also derived
and tested a small tensor expression for how those inputs interact, keeping
normalization explicit. This gives us a way to examine the combination,
but it has not yet produced a smaller independently executable circuit.

## What happened

1. Returned to the pilot's distinction between an execution rewrite and
   discovery of a reusable operation. The recent fixed-query and raw
   embedding-query simplifications had both failed their native screens.
2. Derived an expression for the attention read as a function of19
   upstream contributions. It has190 pairwise source products rather
   than requiring a giant expansion of the entire model.
3. Implemented the expression and checked it against separate direct
   calculations, first on synthetic inputs and then in the trained model.
4. Tested every source alone and every source removed, on both has/had
   and is/was. The managed GPU run used336 full forwards on6048 sequence
   evaluations and took13.18 seconds inside the executor.
5. Used the saved results to calculate each source's interaction with
   all remaining sources. No further model execution was needed.

These are72 previously opened prompt pairs across four panels. The A2
panels change the construction. They are useful transfer checks, but this
atlas does not supply fresh, independently selected OOD evidence.

## What is a query source?

An attention head forms a **query**, a vector that it compares with **keys**
at earlier positions. Those comparisons determine the weights applied to
the **values**, the information it reads. This model multiplies two
query/key comparisons and does not apply softmax to the resulting scores.

We examined heads1 and4 in layer9, which participate in both task paths.
Their query input is a sum of19 contributions: the original token embedding,
plus the attention and MLP writes at each of layers0 through8. An MLP is the
feed-forward module; here it multiplies two linear projections of its input.
Each contribution includes its subsequent residual scaling coefficients.

The experiment changes these contributions only where the two heads read
their query input. All upstream outputs, keys and values stay fixed; the
later model is recomputed. Removing an upstream module globally would
change other computations too, so this result does not describe that edit.

## The tensor computation, with normalization included

Call the19 contribution vectors f_a. A gain z_a specifies whether a
contribution is present (1), absent (0), or rescaled. The query input is

    u(z) = sum_a z_a f_a.

For either query projection W_i, collect its projected source vectors as
rows of P_i: P_i[a,:]=W_i f_a. Also collect the source inner products in
G_ab=f_a^T f_b. This matrix records how sources reinforce or cancel in
the norm; their individual lengths alone do not suffice.

RMS normalization divides a vector by the square root of its mean squared
length plus a small epsilon. Combining the model's input and query RMS
normalizations yields

    H_i = P_i P_i^T / 128 + epsilon_head G / 1152
    d_i(z) = z^T H_i z + epsilon_head epsilon_input
    query_i(z) = R_t P_i^T z / sqrt(d_i(z)).

Here R_t is the model's actual stored position rotation. We retain its
rounding; we do not assume it is perfectly orthogonal.

Because each head multiplies TWO query/key comparisons, each component
of its value read has the form

    read_component(z) = z^T N z / sqrt(d_1(z) d_2(z)).

N is a symmetric matrix formed from the projected source/key comparisons
and their value contributions. Its diagonal describes same-source
products; its off-diagonal entries describe cross-source products.
There are19*20/2=190 distinct quadratic monomials. Linear output maps can
be folded into this numerator. Native normalization and nonlinear later
layers still have to be represented and executed.

This follows the weight-quadratic approach of
[Pearce et al.](https://arxiv.org/abs/2410.08417) and the associative attention
contraction described by
[Katharopoulos et al.](https://proceedings.mlr.press/v119/katharopoulos20a.html),
applied here to a restricted domain of source edits. The full derivation
and theorem limits are in the
[mathematical review](../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0149.md).

## What the trained-model test found

No source alone passed the complete fidelity screen in any of the four
panels. Conversely, removing one source usually made a relatively small
change. These results are consistent with a distributed combination;
they do not by themselves distinguish redundancy from cooperation.

| Panel | Single sources passing | Largest individual omission effect* |
|---|---:|---:|
| has/had original construction | 0 of19 | 7.63% — MLP7 |
| has/had A2 | 0 of19 | 10.47% — MLP7 |
| is/was original construction | 0 of19 | 9.36% — MLP8 |
| is/was A2 | 0 of19 | 8.93% — MLP8 |

*The change in centered endpoint logits, divided by the native paired
logit-effect norm. This is not a percentage of task accuracy or a share
of an additive circuit. These largest entries are descriptive selections.

The preregistered comparison required an omission effect of at least10%
in BOTH panels for a task. Neither task had such a source. The comparison
therefore failed to establish a shared strong dependency set; it does
not establish that the tasks have different dependencies or none at all.

All instrument checks passed. The source tensor and independent direct
FP64 read calculations differed by at most4.97e-14. Compiled full-model
unity logits differed from native logits by at most1.53e-5. A separate
raw query-input implementation of the MLP8 omission agreed with its
compiled intervention, and hooks were restored.

## Why adding separate removal effects is misleading

For one source s, split the inputs into s and the remaining18. Let F denote
the paired change in the answer-versus-foil logit margin, measured after
the full model. The exact four-case interaction is

    I_s = F(all) - F(only s) - F(all except s) + F(none).

If this is zero, those two blocks have additive effects on that observable.
Otherwise the effect of s depends on whether the rest is present.
The identity closes exactly on the saved results.

For example, the MLP6-versus-rest interaction norm is61–89% of the total
query-dependent paired-margin effect across the four panels. That does
not make MLP6 a61–89% circuit: it measures dependence on the other inputs.
Normalization, attention multiplication and the nonlinear suffix can all
contribute to this interaction.

Adding all19 separate leave-one-out margin effects predicts the joint
query removal poorly: relative errors are73–81%. Thus a list of individual
module importances would miss much of the joint behavior. These numbers
concern paired answer margins, not full output distributions. They are
descriptive analysis of opened data, not a population bound.

## What this changes for the goal

Weight-based methods are feasible here, including folding downstream
linear maps into explicit interaction tensors. The useful object is the
shared source computation together with its distinct consumers and norm
dependencies. A factorization of one module alone cannot establish that
its factors are reusable circuits.

This work improves the computational specification and gives a verified
tool for joint source interventions. It does not yet establish fresh OOD
prediction, independent extraction, selective global removal or reuse of
a newly identified operation. All545902902 native parameters and the
upstream initialization remain charged; actual weight savings are zero.

The next identification target is a shared operation within the combined
sources, tested through both task consumers. The large interactions argue
for preserving and explaining that combination before proposing another
single-source simplification.

One further mathematical control is already executed: a small valid query
example has **zero bilinear cross-source numerator** but a nonzero interaction
of -.74235 in its local read, caused entirely by normalization. Thus large
interaction alone cannot identify a shared bilinear feature. The new
[partition tool](../query_partition_norm_math_v1.py) separates those two
contributions in the same local read coordinates. Applying that distinction
to the trained model is the next unrun localization test.

Primary receipts: [native atlas](../BILIN18_L9_QUERY_SOURCE_ATLAS_V1_RESULT.json),
[source/rest analysis](../SOURCE_REST_QUERY_INTERACTIONS_V1_AUDIT.json), and
[implementation](../source_gain_attention.py). The preregistration's02:04
header is a timestamp typo: the managed log records execution at02:00:05–
02:00:21 UTC, after hash-bound preflight. Its original bytes are preserved.
