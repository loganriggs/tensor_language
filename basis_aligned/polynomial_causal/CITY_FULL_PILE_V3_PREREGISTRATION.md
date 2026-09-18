# Precision-faithful reference repair on the opened Pile V2 panel

V2 fails pred_a: FP64 analytic delta differs from native FP32 response by
.00012054468673731246 relative, above.0001. V2 outcome remains failed.
Its final exact-reentry readouts were within3.82e-6absolute; behavioral b–f
numerically passed but did not authorize corpus adoption.

Repair only the positive-control reference. Preserve native operation order:
g0=mixed+attention; g1=mixed+(attention+delta), each in FP32. Independently compute
block_end(g)=g+(Down(Left(RMS(g))*Right(RMS(g)))+Down_bias), with FP32 epsilon,
exact native weights and addition order. Reference is block_end(g1)-block_end(g0)
subtracted in FP64. Do not call the model's MLP method or use observed edited
block9 state as the reference. Keep the old analytic delta error as a diagnostic.

Same40V2sequences/20documents; same candidateformula, tablebytes, baselines,
nullseed17093100+1000*k+cell, strength.5,support, fourreaders;800forwards/300s.
Original gates a–f remain numerically unchanged, but a now measures the explicit
native FP32 reference contract. Add pred_g: all non-reference arms (native,
native8,candidate,and16nulls) replay V2<=1e-5absolute; independent native baseline
block_end(g0) replays observed native x9<=1e-5absolute. Any behavior change blocks
promotion. Data are nowopened for instrument repair, not new independent fresh
confirmation. No retroactive erasure of V2a; report both protocols/receipts.

The mathematical fold's floating-point error is not recast as a passed1e-4gate.
This repairs a reference implementation, not the approximate candidate. Natural
preposition-filtered contexts and artificial city counterpart/readout limitations
remain. No token-only, fullstrengthremoval, orcomposition claim.
