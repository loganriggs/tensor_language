# Closed feature program: known-answer multi-layer control

Construct a fixed seeded18-layer bilinear residual program, ambient64, shared
orthonormal reader/writer frame8,12products/layer and96arbitrary output readers.
Left/Right, Down and bias lie in the shared frame. Actual residual reentry is
included at every layer. NativeFP32epsilon is retained in FP64 RMS calculations.
The reduced executor receives only initial8features, complement squared norm,
initial complement readout and small model coefficients; no intermediate native
state or native-generated norm. The initial readout cache is explicitly priced.

Test64 independent inputs at each of scales0.3,1,3. Compare dense and reduced
18-layer execution for baseline, factor(5,1)removal, factor(12,2)removal, andboth.
Compare their nonlinear joint interaction, not just additive single effects.

A: full-logit relative errors <=1e-11, feature and squared-norm errors <=1e-11
for every case. B: all single/joint signed-logit-effect errors <=1e-8 and effect
norms >1e-5. C: nonlinear interaction norm >1e-9 and its relative replay error
<=1e-6; dropping the retained-complement norm must produce >=1e-3 relative logit
error somewhere (live negative control). No optimizer or native checkpoint.

This is a positive extraction/composition instrument control, not native-model
evidence or linguistic OOD. The invariant common-frame assumption is restrictive
and must be tested or imposed and validated in any learned native candidate.
Price is measured for weights and per-input cache; runtime state and arbitrary
full-output complement information are not silently declared free.
