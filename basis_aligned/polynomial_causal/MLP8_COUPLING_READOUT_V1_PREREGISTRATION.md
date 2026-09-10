# Does the final reader explain directional-switch interaction?

Use exactly the 32 worlds, source edit, A9 clamp, response banks and directional
switch cube of MLP8_COUPLING_DIRECTIONS_V1. No new fit, row selection or tuning.
The parent is valid and rejects both one-way dominance hypotheses. Its output
interaction alone cannot distinguish internal mixed state from reader curvature.

For final raw states x[am,ma], define a=x10+x01-x00, constructed in FP64 and cast
once to native FP32. Decode it with native RMS (native epsilon), complete vocabulary
weights and 30*tanh softcap. This is a synthetic endpoint edit, not a new native
attention/MLP pathway. No norm trace is frozen. Let Z=D(x) and Za=D(a).

I=Z10+Z01-Z00-Z11; I_reader=Z10+Z01-Z00-Za;
I_internal=Za-Z11. Thus I=I_reader+I_internal exactly in real arithmetic.
This baseline-anchored decomposition is not a canonical attribution under all
reparameterizations. The old THIRD_NOUN_MIXED_STATE test used language-factor
corners on natural runs; these axes are directional intervention switches.

A instrument: all parent native/four-cube three-reader replays, native identity
full-vocabulary replay, and offline decoding of native plus four raw endpoint
states have maxabs<=1e-3 AND relative Frobenius<=1e-5. Incoming source and frozen
writes bitwise, first-value unchanged, finite outputs, hook cleanup. Decomposition
closure maxabs<=1e-10 in FP64. CPU reader-only/internal-only and Gram controls pass.

B final-reader sufficiency: ||I_internal||/||I||<=.10 for correct margin,
centered three readers and centered full vocabulary, in both Q_oh mixed and full
tables, in EVERY world. C internal sufficiency: same rule for ||I_reader||/||I||.
Scientific zero denominators fail. Record task-only summaries separately if useful;
do not promote them. Measure raw state nonadditivity diagnostically, not as a
replacement for effect fidelity. Neither candidate promoted when its gate fails.

512 native forwards / 8192 sequence instances, plus 192 endpoint decoder batches /
6144 states, zero fits, 600-second cap, managed bqrunner only. All 545902902 native
parameters and counterfactual states retained; zero structural saving. If B fails,
final-reader folding alone cannot explain the coupling: stop architectural switch
subdivision and study explicit internal joint operations. If B passes it licenses
a conditional reader explanation, not independent extraction or semantic discovery.
