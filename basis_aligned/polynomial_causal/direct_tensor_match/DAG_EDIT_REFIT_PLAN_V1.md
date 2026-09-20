# First approximate graph-edit/refit control — 2026-09-20 19:30 UTC

Extend exact graph rewrites with one approximate edit: delete a reachable product globally, simplify zero/one arithmetic, refit all remaining learned coefficients, then compare whole-graph J. This is not a general topology search yet.

Two opposing planted cases, degree<=2, Gaussian squared function loss integrated exactly by the existing5point4dim rule. Initial graph outputs are mixtures of two products.
1. Near-duplicate: products ab and(2a+.05c)(.5b), target outputs ab,.4ab. Removing one product and refitting should recover target with one product.
2. Independent: target outputs ab,cd with full-rank readout mixtures initially. One retained scalar product cannot span both outputs; removal should be rejected.

Adam .01,500steps per baseline/candidate; J=relative squared error+.005products+1e-5additions+1e-5coefficients. Selection only by exact Gaussian loss and whole-graph cost. Fresh4096Gaussian probes for final diagnostic.

pred_a: positive case accepts one-product graph, fresh relative error<.01.
pred_b: independent case rejects deletion, or any accepted edit fails this hypothesis even if J improves.
pred_c: exported accepted graph matches differentiable program<1e-10; no degree>4; price counts global reachable computation once.
