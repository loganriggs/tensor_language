Exact quartic CP profile controls — 2026-09-21 22:14 UTC

Before native CP512, test sum_a c_a product_s(p_sa^T x), four independently normalized factor matrices, exact symmetrized coefficient Gram and cross inner products. Solve ridge readout exactly; optimize factor directions with the envelope gradient. Teacher constant omitted only in optimization, included in dense toy error.

Five known rank-three families at d=4, output=2: independent factors, shared first inputs, shared output direction, fourth powers, and cancelling duplicate atoms. Compare Adam and Muon, two paired starts each, 400 cosine-scheduled steps, initial rate .03, floor1%, ridge1e-8. Equal steps are not equal computation costs; record runtime. Select by exact regularized objective and evaluate full dense coefficients. This bounded rate does not establish an optimizer optimum.

Predictions: (a) implicit/dense Gram, loss and gradient relative errors <1e-8, including profiled/full-solve gradients, with planted error <1e-4; (b) at least one optimizer achieves <1% in at least 8/10 random starts; (c) both optimizers improve median error relative to their initial fits. Preserve all failed predictions. Known-solution success and random-start failure mean optimization difficulty, not insufficient expressivity. Cancellation is allowed to lower target rank and must not make its nonzero target trivial.

CPU only. The planned native 512 atoms are not executed or approved by these toy results; they require their own registered native cross-contraction smoke, memory and fit plan. This is an exact coefficient baseline, not a monosemanticity or intervention test.
