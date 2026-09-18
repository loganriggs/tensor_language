# Does the local quadratic term improve downstream removal prediction?

The full-strength test beat twice the observed half-strength logit effect, but that
does not separate local quadratic response from nonlinear suffix propagation.
Opened CPU fixtures show full_write−2*half_write is 1.5–14% of full-write norm.
This is half of the full-strength quadratic term, not the entire quadratic term.

For frozen candidate d(a)=a*A+a²*B, evaluate three block9 writes at strength1:
full=A+B; secant=2*d(.5)=A+.5*B; linear=4*d(.5)−d(1)=A.
All three are propagated through exactly the same complete native suffix. Use all
40 opened Pile sequences, six spelling probes and four control readers. Saved
native full-removal effects from CITY_FULL_STRENGTH_V1 are the target. No fit,
new data, row filtering, new null confirmation or independent-composition claim.

Predictions: a, full arm replays saved full-candidate readouts <=1e-5 absolute and
<=1e-5 relative, all outputs finite,120body forwards. b, full relative target error
<=.8*linear error. c, full error<=.8*secant error. b/c test whether retaining the
local quadratic term materially improves accuracy; failures remain informative.
For each arm also report existing screen bars: error<=.35, collateral<=.5 for all
four readers, positive attenuation>=.90 among native margin>=.1, mean>=.02.
These are descriptive lean-candidate screens, not promotions: no new random
nulls, new corpus confirmation, extraction certification or removal adoption.

Price:120body forwards,120second cap, existing saved writes and one native model.
Model loading and execution only through the managed runner. Effective units20
documents. All writes, rows, reference results and this protocol are hash-bound.
