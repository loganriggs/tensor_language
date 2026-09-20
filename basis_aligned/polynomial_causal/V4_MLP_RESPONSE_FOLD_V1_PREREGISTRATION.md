# Fold mixed-edge writers through normalized MLP11

Contract the existing27-writer interface with MLP11 Left/Right and jointly with Down. Background is the FP32 additive singleton postattention11 state at the later query. Compile residual response, not absolute state. If P(h)=D[(Lh)*(Rh)], s(h)=mean(h²)+eps and delta=Wz, then

    delta + [P(h+delta)-P(h) - (P(h)/s(h))*(s(h+delta)-s(h))] / s(h+delta)

is exact and vanishes at z=0. The numerator has27linear and378packed symmetric quadratic coefficients per output. Off-diagonal coefficients include both orders. The quadratic RMS denominator remains explicit. Bias cancels. This is the existing projected_bilinear_response identity prepared directly from background projections, avoiding its large output×input×background mixed array.

CPU independent secant comparison2.49e-14, zero exactly0, planted normalized-constant cancellation1.42e-14. Native same openedv4 contexts/selectors;12prefix16full24joined suffix,12explicitMLP11calls. Old compiled-edge and baseline joins retained; new program adds response to native additive-background MLP11 state, then native suffix12–17. Local FP64 response replay<=1e-8relative; original effects replay<=1e-8; weak effect gates.1number/.05controls and precise.001 unchanged. Native coefficient generation and remaining suffix are charged. Export only predeclared first row of first context for portable CPU replay, avoiding a full48context coefficient dump.

Price canonical program498,070values/context. Implicit conditioned factors290,710/context plus sharedDown5,308,416values; canonical is smaller for few contexts but larger across48. No universal compression claim. This exact baseline precedes any sparse Tucker approximation; native behavioral fidelity is required, not coefficient error alone. No fresh OOD or context-independent circuit claim.
