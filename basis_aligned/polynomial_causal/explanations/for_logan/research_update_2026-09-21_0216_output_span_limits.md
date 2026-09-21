# Conditional error comes from both output omission and imperfect computation

21 September 2026, 02:16 UTC.

An exact-output projection control splits the large conditional error into two roughly equal parts. Some required output directions are absent from the graph; even inside its retained output space, its computed values are imperfect. Neither “just raise input rank” nor “just improve the output coefficients” is a complete explanation.

## The oracle test

We construct an orthonormal basis $Q$ for each graph's available output directions in the vocabulary-centered metric. This uses exported weights only. It does not use diagnostic outcomes to select directions.

For exact weighted intervention $y$ and graph prediction $\hat y$ in that span,

$$
\|y-\hat y\|^2
=\underbrace{\|(I-QQ^\top)y\|^2}_{\text{omitted output directions}}
+\underbrace{\|QQ^\top y-\hat y\|^2}_{\text{error inside retained span}}.
$$

We evaluate the exact projected native intervention, not a fitted approximation to it. The Pythagorean identity holds to relative error below $2\times10^{-15}$ across the recorded documents.

The baseline exposes264 output directions; the8- and32-feature corrections expose272 and296. These exported spans can contain directions not independently attainable by the actual product readout, making the lower bound valid but potentially loose. The projection oracle uses native target values and is not an executable compressed replacement.

## Corrected graph: context-only linear errors

| Domain | Actual graph error | Exact output-span floor | Squared error due to output omission |
|---|---:|---:|---:|
| FineWeb | 58.78% | 42.14% | 51.41% |
| Related code | 42.04% | 28.99% | 47.55% |

The prediction that omitted directions account for at least half the squared context error in **both** domains fails: code is below half. The instrument and lower-bound checks pass. Preserve this miss rather than relabel the hypothesis as confirmed.

The practical implication is still strong. For these FineWeb interventions and this linear metric, **no output contained in the current296-dimensional span can achieve30% relative error**, even if its values were computed exactly. This is a conditional fixed-span bound, not a theorem against Tucker/HT, all296-dimensional spaces, or native-logit error after nonlinearities.

The remaining48.6% of FineWeb squared error and52.4% of code squared error occur inside the output span. Those terms can reflect input directions, product structure, fitting objective or fitted coefficients. This experiment does not separate them.

## A simple amplitude adjustment is insufficient

A separate CPU oracle optimally scales the already-computed centered native logit-effect arrays by one scalar. For the corrected context effects, the optimal scales are0.9944 on FineWeb and1.0036 on code. Relative errors barely change:55.820% to55.818%, and40.099% to40.098%. The missing effect therefore is not repaired by a uniform amplitude adjustment.

This is a diagnostic scaling of final effect arrays, not scaling residual writes through the nonlinear model and not an independently validated fit.

## Consequence for the decomposition direction

The initial output basis was selected to represent broad folded-output variation. The current tests isolate a conditional interaction. We should test whether an output basis selected from the **original centered bilinear operator** preserves that interaction better at the same width, before spending more effort refitting a space with a known floor.

That would still leave the inside-span error to address through decomposition or graph structure. A better subspace is a candidate representation, not a semantic circuit; stable identity, selective interventions, extraction, composition and external OOD prediction remain required.

Evidence: `MIDPOINT_OUTPUT_SPAN_CONTROLS_V1.json`, `MIDPOINT_OUTPUT_SPAN_NATIVE_V1.json`, its raw records, and `MIDPOINT_CONTEXT_SCALE_ORACLE_V1.json` under `direct_tensor_match`. The projection construction is in `export_midpoint_output_spans.py`.
