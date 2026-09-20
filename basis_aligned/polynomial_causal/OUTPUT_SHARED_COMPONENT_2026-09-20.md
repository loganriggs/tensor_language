# New spectral features and a conditionally extracted component

This follows the failure of small shared-input Tucker ranks and fixed native
atom pruning. Unlike pruning, v619 constructs new input features. The full
target is still C=UD, A=LE, B=RE with all vocabulary outputs and source ports.

## Output-sharing block baseline

Use the full output-mode eigenframe W. Each output feature has quadratic form
Q_a=sum_k (W^T C)_ak sym(A_k outer B_k). Diagonalize Q_a independently:
Q_a=sum_j lambda_aj p_aj p_aj^T. The resulting program is a signed sum of
squared linear features with shared output directions. There is no iterative
fit. Largest lambda² entries give optimal support *in this fixed orthogonal
dictionary*, not across possible output rotations or arbitrary DAGs.

All1152 output directions are processed in chunks of8; no full order-three
tensor is materialized. The block-energy identity agrees within4.54e-8 and
the leading block eigen-reconstruction replays within2.15e-6 relative error.

| Square products | Active output features | Full tensor relative error | Expanded stored values |
| --- | --- | --- | --- |
| 64 | 9 | .971666 | 895,168 |
| 512 | 30 | .945279 | 5,048,576 |
| 2048 | 148 | .928362 | 21,602,816 |
| 4608 | 301 | .912917 | 46,996,608 |
| 16384 | 552 | .863592 | 141,030,400 |
| 65536 | 916 | .733951 | 499,128,832 |

The registered native-product-budget fidelity hypothesis fails. This family
does not produce an economical global replacement. Cross-output input-feature
sharing or output-frame optimization remain untested.

## Frozen leading component and native-model validation

The strongest output block has a709-square truncation retaining99.0073% of
its own weight energy, corresponding to9.28196% of full tensor energy. This
component was selected from weights, before behavioral evaluation.

v620 constructs exact adapters so its input is the actual normalized MLP17
input. Input-adapter relative error is7.94e-7 and vocabulary-to-residual writer
replay is8.22e-7. These adapters permit in-model component removal while
preserving the actual final RMSNorm and softcap.

On8 documents each from two fixed FineWeb cache panels, first64 tokens:

| Panel | Component relative prediction error | Removal delta CE | Equal-norm random direction delta CE |
| --- | --- | --- | --- |
| skip1200 | .009622 | .161518 | .052802 |
| skip7000 | .009594 | .190178 | .026289 |

The control uses one frozen random residual direction, matched to the
component writer norm. It is not a null distribution. These are aggregated
document results, not semantic selectivity or domain-OOD evidence. Removing
a component and increasing loss does not alone establish an interpretable
circuit. In particular,87.7142% of the vocabulary writer's energy lies in
the common logit direction. Final normalization and softcap make that direction
potentially behaviorally relevant; do not discard it as a softmax gauge.

## Simplification with separate rank validation

v621 keeps the frozen weight-energy feature order and chooses the smallest
prefix reaching<=.05 component prediction error on skip1200. It then tests
that choice on8 fresh skip11000 documents. No coefficients or directions are
refitted. Selection is calibration-dependent, so only the second panel is a
new rank-validation test; the cache offsets do not establish a domain shift.

The selected prefix is256 squares: .047086 calibration error, .042716 fresh
document error.64 squares give .136020/.127523. The <=64-feature simplicity
hypothesis fails. The256-square artifact stores346,624 floats including the
native-input readers, coefficients, residual writer and vocabulary writer.
This price excludes upstream computation producing the normalized MLP input.

## Evidence and remaining requirements

Two focused tests verify dense tensor/error accounting and signed-component
execution with an explicit RMS denominator. Managed runs v619/v620/v621 all
completed; v620 uses12 native batched forwards, v621 uses4.

Receipts and artifacts in `../bilinear_quotient/circuits/followups/`:

* `output_shared_blocks_v619_result.json` and `_component.pt` — full dictionary
  frontier and the frozen folded-input709-square component.
* `output_component_validation_v620_result.json` and `_adapter.pt` — native
  normalized-input adapters and removal/control results.
* `output_component_rank_v621_result.json` and `_program.pt` — calibration rule,
  selected-row hashes, fresh rank-validation results and256-square program.

The executable primitive is `output_shared_quadratics.execute_component`.
For the native-input adapter, pass the already-normalized MLP17 input and omit
`h`: applying RMS normalization again would be incorrect. For residual-space
removal, use `residual_writer`; for its unembedding contribution use the saved
`vocabulary_writer`. Neither includes the final normalization/softcap by itself.

Next: separate the exact native quadratic into isotropic and traceless parts
and measure their contributions. The isotropic term can become almost constant
on normalized states, so a constant/amplitude-calibration explanation is an
essential competing model. Also split removal effects into fixed-final-norm
and changed-final-norm paths before attributing semantics. If a nontrivial
feature survives those controls, test domain-shift prediction, selectivity,
interchange/reuse and upstream extraction. The full circuit goal is unachieved.
