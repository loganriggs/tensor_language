# Calibration interaction basis versus tensor-energy basis

A data-informed output basis fits calibration interactions closely but transfers poorly. It does not rescue the head1.8 mean/remainder sparse-circuit candidate. This is a negative control on a restricted output-basis family, not a lower bound on every tensor decomposition.

## Design

At the same output ranks32/128 and full128-dimensional head-input span, compare: joint coefficient-tensor energy basis; uncentered covariance eigenspace of native normalized cross-interaction vectors; and a fixed-seed random orthogonal basis. All cores are dense, so core sparsity cannot explain failure. The empirical basis uses only8separate calibration documents (skip80 rows96:104),128positions each, before any evaluation. It is a closed-form fitted basis, not weights-only discovery. No model parameters are updated.

Each candidate projects only the conditional quadratic branch. Background and background-linear terms and RMS denominators remain exact; their generation is not included in the branch price. Errors below compare the mean/remainder cross term as well as the entire normalized four-corner interaction, so exact background terms cannot conceal a failed cross approximation.

## Results

| Output frame | Rank | Calibration cross error | Code cross error | Arithmetic cross error | Repetition cross error | Opened prose cross error | Max normalized interaction error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| weight_energy | 32 | 0.546 | 0.741 | 0.659 | 0.767 | 0.784 | 0.672 |
| weight_energy | 128 | 0.457 | 0.522 | 0.580 | 0.597 | 0.609 | 0.525 |
| calibration_interaction | 32 | 0.082 | 0.776 | 0.235 | 0.798 | 0.695 | 0.700 |
| calibration_interaction | 128 | 0.020 | 0.614 | 0.180 | 0.679 | 0.541 | 0.594 |
| random | 32 | 0.983 | 0.987 | 0.983 | 0.986 | 0.986 | 0.864 |
| random | 128 | 0.937 | 0.950 | 0.938 | 0.939 | 0.941 | 0.823 |

All candidates fail the requirement of <=10% cross and normalized-interaction error in every family. Native coordinate/fold replay passes. The calibration-interaction frame at rank128 achieves2.0% calibration error but18–68% cross error elsewhere; it does not identify a stable low-dimensional dictionary across these contexts. A single random frame is a capacity control, not a statistical null distribution.

65native forwards:1calibration batch plus64evaluation forwards; no gradient fitting or model updates. Existing evaluation families are opened after prior experiments, so this is not fresh confirmation. There is no claim that the small calibration sample exhausts all data-informed approaches. More training rows might change results; that possibility does not turn the current failed transfer into a success.

## Target reassessment

The mean-head route has now failed direct output fidelity, signed removal-effect prediction, shared-core simplification, and cross-context basis transfer. Low average CE of a routing replacement is insufficient motivation for further rank or support sweeps on this target. Retain it as a negative baseline and the exact normalized-head-coordinate executor as reusable algebra.

The existing circuit registry points to subject_number_l11h3_sparse_graph_v1 as a more task-specific lineage. Its manifest reports OOD, selective removal and composition evidence, and its executor constructs ports from token IDs. However, inspection shows that it still runs exact checkpoint prefixes and ten suffix corners; its nine graph edges do not imply small computational cost. Those historical claims need receipt verification before adoption. This is a handoff target, not a new validated circuit claim.

The subject-number templates inspected place the subject at the prediction endpoint, so the shared source/readout position in that executor is consistent with those templates. Broader syntax would need separate positions and new evidence. This check avoids misdiagnosing a deliberate restricted task boundary as an implementation bug.

[Native receipt](../bilinear_quotient/circuits/followups/head_shared_core_v641_result.json) · [dense/sparse baseline](HEAD_SHARED_CORE_BASELINE_2026-09-20.md) · [subject-number package](extracted_circuits/subject_number_l11h3_sparse_graph_v1/README.md)
