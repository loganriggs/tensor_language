# Mechanism controls and transfer of the frozen output component

This continues the [output-sharing component screen](OUTPUT_SHARED_COMPONENT_2026-09-20.md).
The256-feature program is unchanged throughout these experiments. It receives
the actual normalized MLP17 input; upstream state production is still native.

## Bias omission caught and corrected

v622's instrument failed at .002439 relative error. The native bilinear MLP
returns D[(Ln)(Rn)]+Down_bias. The full projected output therefore includes a
constant, whereas the discovered C,A,B tensor describes only the quadratic
term. The projected bias is2256.152344. Keep v622 as a failed instrument receipt.

v623 subtracts this scalar from the observation before quadratic comparison
and removal, while retaining the bias in the native model. Maximum replay
error drops to5.61e-7, including both baseline suffix and actual edited-model
CE checks. Earlier v620/v621 prediction scores compare a quadratic predictor
with the full affine component; they remain recorded as such, not exact
quadratic identities. The corrected256-feature errors on the same two panels
are .045923 and .041652.

## The feature is not the isotropic constant

For alpha=n^T Q n, the weights-only isotropic term is
trace(Q)*||n||²/1152. No constant is fitted. Its value is approximately573396,
whereas observed alpha means are approximately-923000 and-916422, with standard
deviations503006 and516016. Isotropic prediction errors are1.502 and1.499.

Even the optimistically fitted constant on each panel has relative error
.4782 and .4903 (derived from the recorded mean and sample variance over512
positions). This is a lower bound for any constant on those same positions,
not a heldout fitted-baseline result. Both comparisons strongly favor the
input-dependent quadratic feature over a constant explanation.

## The measured removal effect is primarily direct

All arms use the real final softcap. The direct arm removes the component
while holding final RMS normalization fixed; the full arm recomputes it.

| Panel | Full removal delta CE | Fixed-norm delta CE | Additional norm effect | Common-only fixed-norm delta CE | Centered-only fixed-norm delta CE |
| --- | --- | --- | --- | --- | --- |
| skip1200 | .161668 | .163819 | -.002151 | .005217 | .139932 |
| skip11000 | .202638 | .202596 | .000041 | .001095 | .179243 |

Changing final normalization contributes little here. Despite87.7% of writer
energy lying in a common vocabulary direction, most of the behavioral effect
survives in its token-varying part. Common and centered effects do not add:
their softcap interaction contributes .018670 and .022259 CE. This is an
explicit path-dependent intervention decomposition, not unique causal shares.

Removing the traceless component alone increases CE by .380541/.495603;
removing the isotropic term alone increases it by .097433/.103033. These need
not sum to full removal because the signs cancel and the suffix is nonlinear.

## Frozen transfer beyond the prose panels

v624 specifies eight code, eight arithmetic and eight repetitive prompts in
the runner before evaluation. No refitting, feature reordering or rank
selection occurs. These are synthetic input-family shifts relative to the
prose calibration distribution; they are not asserted absent from pretraining.

| Family | Relative quadratic prediction error | Exact-removal CE | Predicted-removal CE | Difference |
| --- | --- | --- | --- | --- |
| code | .018852 | 4.871780 | 4.862448 | -.009332 |
| arithmetic | .023135 | 3.855190 | 3.857765 | .002574 |
| repetition | .018508 | 2.074328 | 2.093516 | .019187 |

All registered gates pass: replay4.32e-7, prediction<=.10 per family, removal
prediction within .02 CE per family. The repetition intervention result is
close to its threshold and should not be generalized beyond this sample.
Full per-prompt results, program hash and literal prompts are in the receipt.

## Requirement audit and next action

* Prediction under distribution shift: demonstrated for this conditional
  component on these three predefined synthetic families; broader OOD unknown.
* Extraction: saved256-feature executable at normalized MLP17 input. It still
  relies on native upstream states, so full prompt-to-output extraction is open.
* Removal: exact and approximate effects replay through native suffix; one
  earlier random-direction control exists. Semantic selectivity remains open.
* Reuse: the same frozen program transfers across input families, but useful
  reuse in a distinct downstream computation or transplantation is untested.
* Simplicity:256 square features/346624 stored values for a single component,
  including both writers. The <=64-feature hypothesis failed; no globally
  economical full-tensor replacement was found.

Next, characterize the input-dependent feature and its upstream producers,
using counterfactuals and multiple matched direction controls to test selective
behavior. Do not replace that work with more aggregate CE measurements. A
shared bilinear/DAG refactor should target those producers while preserving
the established component-prediction and intervention checks.

Authoritative JSON receipts in `../bilinear_quotient/circuits/followups/`:
`output_component_mechanism_v622_result.json` (failed instrument),
`output_component_mechanism_v623_result.json` (corrected), and
`output_component_shift_v624_result.json` (frozen transfer). All managed jobs
finished. v622/v623 each use6 native forwards and32 suffix evaluations; v624
uses24 native forwards and72 suffix evaluations. The full goal is not complete.
