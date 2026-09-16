# CrossFirst Hessian finite-scale curve V1

## Question

The complete 18-stage causal Hessian closes exactly as a derivative, but on the
new-endpoint four-stage panel it leaves `40.3%` child-relative error for
`licence/license`.  Determine whether this is a real higher-order finite-edit
effect or an implementation/attribution error.

This is an opened-panel diagnostic, not a new OOD test.  Reuse the exact frozen
CrossFirst child/remainder ports and the hash-bound complete-Hessian artifact.
For scales `s in {0.25, 0.50, 0.75, 1.00}`, physically evaluate

`I(s) = f(s(a+b)) - f(sa) - f(sb) + f(0)`

through the native suffix.  Compare it with `s^2 H`, where `H` is the complete
mixed Hessian from the prior exact allocation.  Preserve every prompt and both
target/control readers.

## Frozen predictions

- **A -- replay:** all scale-one physical arms replay the bound four-stage
  artifact within `2e-6` relative error.
- **B -- local Hessian validity:** at `s=0.25`, `s^2 H` has target relative L2
  at most `0.15` for every family and every endpoint concept.
- **C -- higher-order growth:** for `licence/license`, target relative error of
  `s^2 H` is nondecreasing over the four scales with `0.02` slack, and the
  scale-one error exceeds the quarter-scale error by at least `0.15`.
- **D -- material native-scale curvature:** for `licence/license`, the change
  between `I(1)` and the quadratic extrapolation of `I(0.25)` has norm at least
  `0.20` of `I(1)`.
- **E -- cubic diagnostic:** define `g(s)=I(s)/s^2`.  Estimate a per-row cubic
  coefficient from `g(0.50)-g(0.25)` divided by `0.25`, without using larger
  scales.  The resulting `s^2(H+s*C3)` prediction at `s=1` improves the
  `licence/license` complete-Hessian relative L2 by at least 30%.  This is an
  oracle diagnostic, not a reusable circuit or deployable fit.
- **F -- audit:** exact scales, physical arms, Hessian predictions, residuals,
  and cubic diagnostic are serialized for every prompt and both readers.

If A/B fail, investigate numerical precision or implementation before reading
the curve.  A--D passing establishes that the native-scale failure is genuine
higher-order curvature.  E says whether third order is already a useful local
description; failure points to fourth-and-higher terms.  No outcome changes the
previous fresh null or licenses a standalone component.

## Price

One checkpoint load; 48 opened prefixes; one native upstream trace and 16
physical suffix arms per prefix (four scales times four corners); no JVPs,
backward passes, parameter updates, or stage/rank search.
