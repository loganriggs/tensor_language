# Native replay of the reduced head9 odd-value interface

Opened prospective40sequences. Native, original midpoint, reduced-interface
midpoint =120forwards/300seconds. The same head8 delta is used in both arms.
Original: changed=RMS(lambda90*(x9+delta8)+lambda91*x0), routing from native z9,
subtract current-value projections. Reduced: raw9=lambda90*x9+lambda91*x0,
changed=RMS(raw9+lambda90*delta8), native=RMS(raw9), project their difference.
Inherited-first values cancel exactly. Reassociation may change FP32 results.

pred_a: each head9 write relativeL2<=1e-4, finite and outside-head8-mask zero.
pred_b: all readout original/reduced score differences<=1e-5 maxabs AND<=1e-6
relativeF; original scores replay saved native/midpoint<=1e-5maxabs.
pred_c: each target/control effect relativeL2<=1e-3 with original norm>=1e-6.
Null: cancellation/reassociation loses native behavioral precision.
Save native raw9,delta8,lambda90,mask and original delta9 fixtures for CPU replay.
CPU pred_d: original delta9 relative error<=1e-4 and zero upstream write gives
zero result, using only exported head9 weights and copied source code.

This closes an inherited-value input algebraically and combines reentry inputs.
The head9 delta still requires raw9 and delta8; the combined head8/head9 program
requires current8,donor-city-state (or earlierg7),raw9 plus tokens/masks. Three
native-state arrays over the larger boundary, not a two-port whole model. Count
only used head9 arrays; first-value maps and scalar even-branch adapters are not
needed. Native suffix after the head9 injection remains external.
