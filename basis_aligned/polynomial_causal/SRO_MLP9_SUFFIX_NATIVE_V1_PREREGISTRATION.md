# Rational head9 component → MLP9 state through the native suffix

Fixed regional rows 0/24/48 and natural rows 0/16 from the existing confirmation
panel. Strengths are the eight binary S/R/O removal masks, plus (-1,.5,1.5),
(.25,.75,1.25), and (2,-1,0). No target-based fitting or changed source weights.

For each prefix, the first pristine native pass prepares the exact quadratic
MLP numerator and RMS denominator from its actual z9 and three source writes.
Direct arms remove writes at attention9 output and run the real MLP9. Compiled
arms replace block9's output with the evaluated rational post-MLP state and retain
native first values plus the entire downstream suffix. Earlier prefix computation
is unchanged. Count 5×11×2=110 body forwards, 120s, managed lane1 only.

A: each post-block9 state and full-vocabulary score vector <=1e-5 relative error.
B: baseline-subtracted selected readout discrepancies <=1e-5 + 1e-4|reference
effect| for every case. Regional endpoints are the cue and unrelated token
contrasts; natural endpoints are newline CE and newline/comma margin.
C: zero-strength effects exactly zero and 110 recorded body forwards.

The absolute floor explicitly handles tiny FP32 effects; it is not a behavioral
selectivity gate. Charge full original MLP weights, source/context generators
and all prepared-state bytes. This validates conditional compilation across a
module boundary, not autonomous extraction, new OOD behavior or a speed claim.
