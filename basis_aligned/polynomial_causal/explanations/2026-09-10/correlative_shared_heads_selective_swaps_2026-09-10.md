# Same heads, separately swappable information, incomplete independence

The latest experiment gives a positive answer to part of the shared-module
question: the same26 attention heads contain two distinguishable sets of
information for two behaviors. The saved correlative directions transfer the
both/neither behavior; their orthogonal remainder transfers either/not.

But this is not yet a decomposition into independently removable circuits.
Replacing either part with its saved mean can disturb the other behavior, and
the effects of changing both parts do not add. We now have direct evidence
that a successful swap partition is weaker than the four properties we want.

## What changed in this test

The earlier work established a14-scalar correlative interface across26heads.
Weight folding made its read/write operation explicit, and separate tests
showed that its local/first value sources and its routing/value operands could
not individually replace the coupled operation.

Here we kept the entire operation and asked how two behaviors use the same
heads. No new directions, heads, ranks or gains were fitted. For each block,
the saved unit direction q defines a projector P=q qᵀ: it extracts the component
along q. R=I−P retains everything else. P+R=I exactly by construction; the saved
q has orthogonality error below2.4e−7. Across blocks, P has14 coordinates and
R has3314. The large remainder is still opaque, not a newly explained subroutine.

We tested the existing16-group bare and report both/neither frames and16groups
of the disjoint either/not frame. Their intended answers are and/nor and or/but,
respectively. These are reused test rows, not a fresh out-of-distribution sample.
All48 pairs passed correct native answers at both endpoints.

At each block, a donor swap writes P(donor−live), R(donor−live), or the full
selected-head difference. Each later block uses the current edited recipient.
Mean replacement uses the same operations with the original saved FIT mean
instead of a donor. It is the registered removal operation, not a claim of
universal feature erasure.

The managed run used30 forwards/480 sequences,1.584 seconds of recorded executor
time, at14:17:37–14:17:41 UTC. Native and projector-versus-folded execution
bridges passed. The experiment's instrument, native capability and double
interchange predicates passed; selective mean removal and endpoint addition failed.

## Swaps distinguish the two uses

Recovery measures the fraction of the native cue-induced answer-margin change
reproduced by the swap. It is an answer-score measure, not full-model equality.

| Behavior/frame | Saved directions P | Remainder R | Full selected heads |
|---|---:|---:|---:|
| both/neither, bare | .969 | −.004 | .973 |
| both/neither, report | .899 | .052 | .966 |
| either/not | −.025 | .982 | .966 |

The registered target threshold was.8 and cross-task absolute recovery threshold
was.23. All pass. This is a double dissociation under donor swapping: each part
changes one task strongly and the other weakly. It supplies within-module
splitting evidence, even though the native module boundaries are shared.

This does not mean P reproduces everything the full heads write. Its full-
vocabulary effects can differ while its selected task margin transfers well.
Nor does it identify the computation contained in the3314-coordinate remainder.

## Mean replacement does not preserve the other behavior

Cross-entropy is the negative log probability of the correct base-answer token,
measured over the complete vocabulary. A positive change means damage; a
negative change means improvement. For preservation, we registered the mean
of each example's **absolute** change, so opposing changes cannot cancel.

| Mean-replaced part | Behavior/frame | Mean signed CE change | Mean absolute CE change |
|---|---|---:|---:|
| P | both/neither, bare | +1.289 | 1.289 |
| P | both/neither, report | +1.392 | 1.392 |
| P | either/not | +.051 | **.172** |
| R | both/neither, bare | −.118 | **.136** |
| R | both/neither, report | −.014 | .080 |
| R | either/not | +3.182 | 3.182 |

All intended-removal damage thresholds of.5 nats pass. Preservation requires
mean absolute change<=.1 nats. P on either/not and R on bare both/neither fail.
The latter is mostly improvement, but the preservation claim requires the
other behavior to remain stable, not merely improve on average.

For P on either/not,10 of16 examples worsen and6 improve. A paired bootstrap
over those authored groups gives a signed-mean interval[−.044,.139], which
includes zero, but an absolute-change interval[.124,.217], above.1. This is why
the small signed mean is misleading as evidence of preservation. The R-on-bare
absolute interval[.081,.194] crosses the threshold; its registered point
criterion still fails. These are finite-panel intervals, not population claims.

## Why orthogonality is insufficient

At one block, the two changes add exactly to the full head-space change. The
rest of the model is nonlinear, however, and changing earlier blocks changes
the inputs to later ones. Orthogonality of coordinates does not imply separate
downstream consumers or additive final effects.

The measured quantity was

    ||center(z_full - z_P - z_R + z_base)||
    / ||center(z_full - z_base)||,

where z is the full prediction-position vocabulary vector and center subtracts
its vocabulary mean. The interaction was.207/.266/.119 in the three panels,
all above the.10 additive-effect threshold. Bootstrap intervals were
[.198,.215], [.252,.280], and[.111,.126].

Also, swapping a coordinate between two naturally related sentences and
replacing it by a global FIT mean are different interventions. Small changes
between the two donors do not imply that the coordinate is irrelevant to the
other task. It may carry a shared contextual contribution that both donors
retain. This is a possible explanation, not an identified native consumer map.
The earlier handoff/pilot distinction between shared computation and independently
editable consumer state remains applicable; orthogonal projection does not
resolve it automatically.

## What we have and what is missing

The positive result is a fixed, causally tested split of information within
shared heads under the specified donor swaps. Selective removal and additive
composition do not follow, and both failed their direct tests. Independent
execution, semantic explanation of both producers and consumers, fresh OOD
prediction and reduced structural description cost remain missing.

All545902902 native parameters remain required. The large remainder should not
be called one simple circuit merely because it has one name. The next useful
explanation must account for which downstream uses are private to each task
and which depend on shared context; fitting another direction or changing the
removal mean would not establish that explanation by itself.

Receipts: `CORRELATIVE_COMPLEMENT_SPLIT_V1_RESULT.json` and
`CORRELATIVE_COMPLEMENT_SPLIT_AUDIT_V1_RESULT.json` in the parent directory.
The hourly review is `HOURLY_STRATEGIC_REVIEW_2026-09-10_1414.md`; its next
checkpoint is15:14 UTC, with the mathematical review due16:49 UTC.
