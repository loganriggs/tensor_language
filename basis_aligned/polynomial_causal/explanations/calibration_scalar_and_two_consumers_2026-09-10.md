# A weight-derived calibration scalar with two coupled uses

This is a positive result for a bounded component. A scalar extracted from the
last bilinear MLP has a repeatable calibration effect on held-out text, and we
can compute it directly from a folded weight tensor. The same scalar affects
two parts of the final readout: vocabulary scores and their shared normalization
scale. Neither use alone reproduces its full effect.

It is not yet the full decomposition we want. The scalar still reads an input
produced by the original model, its quadratic matrix is dense, and the remaining
model supplies the background. There is no whole-model parameter saving or
independently executable token-to-answer circuit here. The useful advance is an
explicit producer with two tested uses, rather than only an activation direction.

We began with earlier evidence for a frequency-correlated MLP17 output direction.
Removing that direction had a specific effect, but the proposed identification
as a simple log-frequency bias failed a sanity test. Large activation dimensions
controlling RMS gain were also already known. The new experiment separates the
two uses of this particular scalar, folds its producer through the weights,
and tests it outside the rows used to choose the direction.

The sequence was: freeze data and predictions; verify the algebra on CPU; fit
one output direction on 48 FineWeb rows; run the native model on 42 different
FineWeb rows and 16 Pile documents; evaluate separate and joint consumer edits;
then audit row-level uncertainty. An initial run reached result assembly but
failed on a file-size reporting typo. Its source and producer were preserved.
The corrected run changed only reporting and output filenames. The two runs'
producer tensors are bitwise identical.

The valid run made 30 complete model-body forward calls covering 118 sequence
instances of length 256, including replay and online-intervention checks. It
took 9.36 seconds inside the experiment. Later readout evaluations reuse the
captured states and perform additional vocabulary projections; they are not
additional model-body forwards.

The direction w is proposed from the covariance of MLP17 outputs with the log
frequency of the next token, using FIT rows only. The 20 most frequent FIT
tokens define the frequent class; all remaining targets form the rare class.
Evaluation labels only score the results. They are never inputs to the scalar
producer or its intervention.

| Effect of removing the full scalar contribution | FineWeb held-out rows | Pile corpus shift |
|---|---:|---:|
| Rare-token loss change | +0.500 nats | +0.573 nats |
| Frequent-token loss change | -0.287 nats | -0.213 nats |
| Mean absolute rare-token effect of three random-axis controls | 0.00059 nats | 0.00040 nats |

Positive loss change means worse prediction. Thus this component helps rare
tokens while imposing a cost on frequent ones. The sign pattern holds in every
one of the 58 evaluation rows. Resampling whole rows gives 95% intervals of
[0.461, 0.539] and [0.500, 0.651] nats for rare-token damage. FineWeb's original
document grouping is unknown, so its interval is a row-bootstrap diagnostic.
Each Pile row comes from a separate document. Pile is a corpus shift; we do not
claim these documents were absent from the model's training.

The weight fold makes the producer explicit. Write the bilinear MLP as

    M(u) = D[(L u) * (R u)] + b,

where u is its native RMS-normalized input and * means elementwise
multiplication. The coefficient of its output along w is

    q(u) = w^T M(u) / (w^T w) = u^T Q u + beta,
    Q = sym(L^T diag(D^T w / (w^T w)) R),
    beta = w^T b / (w^T w).

Here sym averages a matrix with its transpose. Q collects the input-pair
interactions that produce the scalar. This is a direct contraction of learned
weights, with no fitted tensor approximation or rank selection. The folded
formula reproduces the native scalar with approximately 2.1e-7 relative error
on both cohorts. The small difference comes from floating-point reassociation.
It does not establish that the scalar literally counts token frequencies.

Now let h be the final residual vector, and e=q*w the proposed contribution.
Removing e has two consequences. It subtracts q*(U w) from the unembedding
numerator U h, where U contains the vocabulary readout weights. It also changes
the RMS denominator from

    rho0 = sqrt(mean(h^2) + epsilon)
    to rho1 = sqrt(mean((h-q*w)^2) + epsilon).

We test these uses separately with

    logits[n,d] = 30*tanh((U h - n*q*(U w)) / (30*rho[d])).

n and d indicate whether the numerator and denominator use is removed.
The four settings are native, numerator-only, denominator-only and both.
The native epsilon and final softcap remain explicit. The joint expression
matches actual native and compiled-producer removals with maximum logit
differences below 4.3e-5 and relative errors below 5.9e-7.

Both uses matter. Numerator-only removal leaves about 52–53% relative error
in the centered full-vocabulary effect; denominator-only leaves 93–95%.
The registered sufficiency limit was 10%. Their loss effects also interact:
for rare tokens, adding the separate loss changes underestimates the joint
change by 0.132 nats on FineWeb and 0.136 on Pile. The interaction is positive
in every evaluation row, with bootstrap intervals [0.122, 0.141] and
[0.113, 0.159]. We therefore retain the two uses and their nonlinear interaction.

This is the kind of decomposition the handoff motivates: one computed quantity
feeds multiple consumers, and interventions on those uses have different,
predictable effects. The narrower claim supported here is a conditional scalar
producer and its two final-readout uses. Donor interchange, independence from
the original upstream model, stability across independent direction fits, and
a lower structural description cost remain open. We have not discovered a
complete collection of reusable semantic circuits.

The stored producer has 1,328,257 scalars for Q, w and beta. Its artifact is
about 10.66 MB, including the diagnostic random axes. All 545,902,902 native
parameters remain in the current executor; this standalone producer size is
not a whole-model saving.

Evidence: [registered plan](../CALIBRATION_TWO_READERS_V1_PREREGISTRATION.md),
[native result](../CALIBRATION_TWO_READERS_V2_RESULT.json),
[row-bootstrap audit](../CALIBRATION_TWO_READERS_BOOTSTRAP_V1_RESULT.json),
[producer weights](../CALIBRATION_TWO_READERS_V2_PRODUCER.pt), and
[implementation](../calibration_two_readers_v1.py).
