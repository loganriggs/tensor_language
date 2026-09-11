# Latest research update

**11September05:17UTC:** [A measured optimizer bottleneck and equivalent repair](explanation_2026-09-11_0517.md).

The first exact-curvature run remained unconverged after24outersteps. A local
matrix-association repair matches its Hessian within2.3e-17 and speeds up the
measured host kernel14.7–44.3x. Thread capping alone did not speed it up.
These are kernel measurements, not whole-fit speedups.

The longer600-second managed continuation is live; check
BLOCK_TRUST_REGION_THIN_V1_RESULT.json and the runner for its final status.
No convergence result was available when this update was written.

[Current weight-only methods and receipts](../../WEIGHT_ONLY_METHODS_INDEX.md) ·
[Previous exact-curvature explanation](explanation_2026-09-11_0503.md).
