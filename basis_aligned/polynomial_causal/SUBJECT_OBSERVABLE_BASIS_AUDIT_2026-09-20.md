# Subject-number response spaces: preserve observables, not just state variance

v648 repairs the v647 instrument without changing rows, bases, width or gates.
Recomputing projected baseline-write ports from float64 polynomial numerators
reduces maximum absolute local replay error from.00412 to3.64e-10. The original
invalid run is retained. With this valid instrument, the state-SVD8 chain fails:
conditional effect error16% on calibration,12–17% on new preposed prompts and
57–73% on postposed prompts. The latter prompts are now opened, not prospective.
The random8 control predicts essentially none of the response (error about100%).

## v649: locate the representation failure

Use the same frozen bases in two additional controls. Initial-projection-only
passes the projected block11 response through the **dense** six-MLP conditional
chain. Final-projection-only projects the true final response before the readout;
it is an oracle, not an executable predictor from the declared starting inputs.

| Conditional effect error | Calibration | Preposed | Postposed |
| --- | --- | --- | --- |
| Repeated state-SVD8 projection | .161 | .120–.172 | .573–.733 |
| Initial projection only | .077 | .039–.068 | .601–.759 |
| Final projection oracle | .120 | .106–.139 | .465–.598 |

Thus missing input/output-relevant directions already cause large failures;
repeated projection is not the sole explanation. Eight principal state directions
discard19–32% of calibration response-vector norm, but the behavioral failure is
the decisive criterion. This does not bound all width-eight nonlinear dictionaries.

## v650: matched-size answer-reader anchors

Reserve two orthonormal directions spanning the fixed is/are unembedding rows,
then choose six principal calibration response directions orthogonal to them.
The basis still has width8 and the same513,030 fixed values. The U-row span is
checked to1e-10; local joint contractions replay to3.20e-10 maximum absolute error.

| Conditional effect error | Calibration | Preposed | Postposed |
| --- | --- | --- | --- |
| Reader-anchored chain | .0458 | .0219–.0647 | .0761–.1665 |
| Initial projection only | .0313 | .0253–.0440 | .1025–.1519 |
| Final projection oracle | .0032 | .0020–.0042 | .0011–.0030 |

This is substantial improvement at unchanged capacity, but both registered
all-cell gates still fail. Including raw answer readers closes the final numerator
loss; exact final RMS still depends on the state norm. The remaining error is
upstream. No gate was relaxed and the width was not increased.

## Targeted literature search and next executed consequence

Actual queries: “Benner Goyal balanced truncation quadratic bilinear systems model
reduction arxiv”; “Rowley balanced proper orthogonal decomposition 2005 snapshot
adjoint model reduction”. Opened primary arXiv abstracts1705.00160 and1610.03279,
then the full HTML of1705.00160.

[Benner and Goyal](https://arxiv.org/html/1705.00160v1) formulate QB reduction using
reachability and observability information, rather than trajectory variance alone.
They discuss the input dependence of trajectory-based reduction. Their continuous
control-system framework does not directly give a guarantee for our finite,
layer-dependent rational chain with live RMS and softcap. We have not established
the required mapping or stability assumptions, and make no balanced-truncation
error-bound claim here.

[Benner, Goyal and Gugercin](https://arxiv.org/abs/1610.03279) optimize a truncated
Volterra-kernel H2 objective for QB systems. That norm is not our finite intervention
error. This provides a neighboring algorithm family, not an off-the-shelf solution.

Our inference is narrower: test output-sensitive basis selection before discarding
small response spaces. v651 keeps the two fixed reader anchors, replacing the six
remaining state-variance directions by principal **backward-reader** directions
computed on calibration baseline and edited endpoints. The forward model still
keeps its complete quadratic core and exact RMS; local derivatives are used only
to select a basis. The same48 test rows and gates remain unchanged.

The exact local reader pullback is implemented in
[bilinear_chain_readers.py](bilinear_chain_readers.py). For upstream reader q,
P0=D[(Lh)*(Rh)], s=mean(h^2)+eps, the MLP contribution is

```
(L^T[(D^T q)*(Rh)] + R^T[(D^T q)*(Lh)])/s
    - 2*h*(q dot P0)/(d*s^2).
```

Add q for the residual route and multiply by lambda0 for the preceding state.
The final reader explicitly differentiates RMS and both separate softcaps. An
independent three-block float64 autograd test passes, including biases and saturated
readouts. v651 is registered and queued; no native outcome is claimed yet. This
targeted search does not reset the scheduled three-hour review clock.

Subsequent v651–v653 results and exact finite-reader/dual-space constructions are
recorded in [finite readers and dual spaces](FINITE_READERS_AND_DUAL_SPACES_2026-09-20.md).
The numerical instruments pass; the reduced models still fail the all-cell gates.
