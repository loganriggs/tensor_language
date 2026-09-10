# A replayable correlative interface and its weight-folded program

10 September 2026, 13:22 UTC.

**We now have a saved, replayable circuit interface and a weight-derived program
that reproduces its joint interventions in the native model.** This is a concrete
advance beyond the preceding mathematical control. The program uses26heads across
14layers, computes one scalar per layer, and applies one corresponding residual
writer per layer. It reproduces both donor swaps and FIT-mean replacements over
the full vocabulary at the scored prediction position.

The main limitation remains substantial: the program still calls the original
model to produce its contextual inputs, routing and background. It is a conditional
executable description of an existing interface. Independent extraction, new OOD
prediction, selective removal and structural savings remain incomplete.

The original [handoff](bilinear_circuit_reconstruction_codex_handoff.md) and
[pilot report](bilinear_reconstruction_pilot_report.md) continue to govern this
direction. This work moves from the normalization-aware mathematical diagnostic
to an existing behavior with positive causal evidence; it does not restart is/was
or tune the closed calibration hypotheses.

**The behavior and starting evidence.** Correlative constructions pair words such
as “both … and” and “neither … nor.” One tested prompt pair is:

- “The ranger praised both the manager” → expected continuation “and”.
- “The ranger praised neither the manager” → expected continuation “nor”.

The existing v505 study found a distributed attention interface whose activation
swaps transfer much of this distinction. Its report omitted the chosen head IDs
and fitted direction tensors. Those omissions prevented executing or folding that
specific saved interface from the report alone.

We reran the unchanged recipe once under a new name, preserving its original
selection, fitting and evaluation rules. The new artifact saves the ordered heads,
per-layer directions, FIT means, fitting history, actual row sets and source hashes.
It reproduces the four reported comparison metrics exactly at their saved
precision, and all five original predicates pass. Loading the new directions from
disk gives zero measured difference in the replayed margins. The native, head-patch
and MLP-patch reference bridges also give zero difference.

This is explicitly a **new fitted artifact**. Agreement with old rounded metrics
cannot prove identity with the missing historical tensors. Reconstruction took
34.29executor seconds,1,420body forwards and22,636sequence evaluations, including
the original greedy selection and120fitting steps. Only the interface was fitted;
the model weights stayed fixed.

**What “rank one” means here.** A direction is fitted separately at each selected
attention layer. There are14such layers, numbered3through16. Consequently the total
interface dimension is14, not one. The26selected heads contribute3,328head-output
coordinates to these directions. The saved directions contain3,328coefficients;
the FIT means contain another3,328.

For layer \(l\), concatenate its selected head outputs into \(z_l\), and let
\(q_l\) be its saved direction. Its scalar is \(s_l=q_l^Tz_l\). The original
interchange changes the live head output by

\[
\Delta z_l=q_l(s_l^{\rm donor}-s_l^{\rm live}).
\]

The later layer uses its current live value after earlier edits have propagated.
It does not repeatedly add a difference computed from the untouched base model.

**The computation can be folded through the weights.** Split \(q_l\) into one
128-dimensional reader \(q_{lh}\) per selected head. A head computes a weighted
sum of source values. Its value is the model's actual mixture of a local value
and the first-layer value, using coefficient \(\lambda_l\). Define

\[
r_{lh}^{\rm local}=(1-\lambda_l)V_{lh}^Tq_{lh},\qquad
r_{lh}^{\rm first}=\lambda_lV_{0h}^Tq_{lh}.
\]

Then the same scalar is

\[
s_{l,t}=\sum_{h\in H_l}\sum_{j\le t}p_{lh,tj}
\left[(r_{lh}^{\rm local})^Tx_{l,j}
 +(r_{lh}^{\rm first})^Tx_{0,j}\right].
\]

Here \(p\) is the complete native two-QK routing score, including its head
normalizers, rotary tables and causal mask. \(x_l\) is the normalized input to
that attention layer, and \(x_0\) is the normalized input to first-layer attention,
not an unexplained substitute for a raw token embedding. The operation reads two
value features per source, weights their sum by routing, and adds across sources
and selected heads. Negative and greater-than-one mixing coefficients are retained;
for example, layer4 uses \(\lambda=-4.1875\), and layer14 uses4.625.

Fold the head direction through its output projection as well:

\[
w_l=\sum_{h\in H_l}O_{lh}q_{lh},\qquad
\Delta r_l=w_l(s_l^{\rm donor}-s_l^{\rm live}).
\]

Thus one residual writer per layer implements the complete selected-head edit.
The same equations implement FIT-mean replacement by substituting the fixed mean
scalar for the donor scalar. The current remainder and all later computation remain
live. These are consequences of linearity around the existing routing computation.
They expose an operation and its ports without assigning new semantic meaning to
the14individual scalars.

The CPU weight audit checked every block using fixed synthetic continuous source
streams. The largest scalar discrepancy was5.26e-13 and writer discrepancy1.25e-14.
It saved26pairs of local/first-value readers and14writers, totaling76,032core
coefficients. That is more numerical content than the original direction artifact;
there is no claimed storage improvement over the original implementation.

**The native joint test passed.** The compiled producer reads native routing and
normalized inputs directly. It computes the scalar without using the selected
head outputs as its producer; those outputs are read separately to check the
identity. The native head computations remain active to supply the unchanged
background. All14writer interventions are installed together and later layers
recompute from their changed inputs.

Across four panels of16rows and seven executions per panel:

| Check | Result |
|---|---:|
| No-edit compiled capture versus native reference | Exact agreement |
| Joint donor/mean edit maximum full-vocabulary discrepancy | 1.53e-5 logits |
| Largest full-vocabulary relative L2 discrepancy | 6.08e-7 |
| Largest answer-margin discrepancy | 7.63e-6 logits |
| Worst scalar error divided by its registered tolerance | 0.1893, below1 |
| Body forwards / sequence evaluations | 28 / 448 |
| Executor time | 1.64seconds |

All five frozen predicates pass. The full-vocabulary comparison covers the50,304
possible next tokens at each scored prediction position. This does not establish
an end-to-end replacement preserving every output on arbitrary text.

**What the behavior measurements mean.** A subsequent CPU audit recomputed the
interchange effects from saved row margins. Recovery measures how far the patched
model moves from its native preference toward the donor's preference. It is not
classification accuracy. Answer-preserving and disjoint controls instead measure
absolute margin movement divided by the original construction's native separation.

| Panel | Example change | Folded donor effect |
|---|---|---:|
| Original construction, A1 | both → neither in the praise frame | 0.9676 recovery |
| Second construction, A2 | both → neither in a report frame | 0.9022 recovery |
| Answer-preserving, P | ranger → worker, retaining both | 0.01910 normalized movement |
| Disjoint, C | either/or → not/but | 0.02038 normalized movement |

The old report's1.001 and0.935 values divide recovery by the selected full-head
set's recovery. They are different quantities from the raw0.9676 and0.9022 above.

Mean replacement increases correct-next-token cross-entropy by1.794nats on A1
and1.594nats on A2, with positive damage on all16rows in each panel. On disjoint C,
the mean increase is0.0633nats; a4,000draw authored-group bootstrap gives
95%interval[-0.0196,0.1486]. This descriptive result does not establish a sufficiently
small unrelated-removal effect. The compiler protocol registered correspondence,
not a new behavioral selectivity threshold.

P has exactly the same16base prompts as A1. Its donor changes only the reporter.
Replacing the target variable by a mean therefore damages P's base answers too.
That is not evidence of unrelated collateral: those prompts still need the target
computation. This corrects an overbroad interpretation in the earlier progress
commentary. The disjoint construction provides the relevant unrelated readout here.

The original rows carry the author's FIT namespace. Our audit verifies that the
recipe's internal fit and held subsets have no shared row IDs or group IDs. Each
evaluated panel has16groups, but A1/A2/P share the same group identities. These
panels should not be pooled as independent replications. All have been reused in
prior experiments, so none supplies pristine OOD confirmation. The mean-replacement
numbers above use base sides only; the earlier removal report averages both sides.

**What remains for the four goals.** Prediction currently covers reused controlled
constructions and the original interface's effects. Extraction is conditional on
native contextual inputs and routing. Removal has an executable correspondence,
with unrelated selectivity unresolved. Composition is verified for all14edits
together; shared semantic reuse across independent tasks remains unestablished.
All545,902,902native parameters are still required. No simpler overall model or
previously unspecified language algorithm has been demonstrated.

The next scientific step should explain an input operation of these frozen ports
and test it on unseen factor combinations. The saved interface and reusable
executor make that possible without another direction fit. The hourly review
also calls for finishing this shared machinery before opening unrelated candidates.

The canonical record is
[subroutine.correlative.block_scalar_interface](../../bilinear_quotient/circuits/subroutine_correlative_block_scalar_interface.json),
with conservative status `site_live`: executable conditional correspondence,
without circuit adoption. Sources and exact reproductions are in the
[interface result](../CORRELATIVE_REPLAYABLE_INTERFACE_V1_RESULT.json),
[weight-port audit](../CORRELATIVE_WEIGHT_PORTS_V1_RESULT.json),
[native protocol](../CORRELATIVE_FOLDED_PROGRAM_V1_PREREGISTRATION.md),
[native result](../CORRELATIVE_FOLDED_PROGRAM_V1_RESULT.json),
[effect/split audit](../CORRELATIVE_FOLDED_EFFECTS_V1_RESULT.json), and
[folded executor](../correlative_native_folded_executor_v1.py).
