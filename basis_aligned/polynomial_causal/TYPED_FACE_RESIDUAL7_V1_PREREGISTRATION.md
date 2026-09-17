# Two-state residual7 boundary

Own block8 reentry using exact token initial states andlambda8, fullattention8,
and coupled head8.2/MLP8 response. Native current8, donor_city8, rawcontext and
rho8 all generated internally. Inputs residual7[B,T,1152],donor_city7[B,1152],
token IDs,city,destination and half-strength. Capture residual7 DIRECTLY at
block8 entry, not by reversing rounded scaled sources.74token table, no fitting.
Full price24,060,931floats,38,016state scalars atT32,two native arrays.

40opened native8 fresh-panel sequences. Native/native8/compiled=120bodyforwards,
plus40localattention8 generator calls,300seconds.
pred_a: native/half-write reference replay<=1e-5abs, native jointdelta error
<=1e-4relative, outside zero, finite,120bodyforwards.
pred_b: compiled readouts<=1e-4maxabs AND<=1e-5relativeL2 versusnative8.
pred_c: target andfourcontrol effect errors<=1e-3relative in everyfamily.
pred_d: isolatedcopiedpackage reproduces40native jointdeltafixtures<=1e-4relative,
zero strength zero,unknown token rejected. No relaxed gates. No whole suffix,
fresh OOD, causalcomposition or simplicity certification from this replay.
