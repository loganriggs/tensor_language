# Retain optimizer history: matched native controller comparison

The completed spectral-original20-minute fit restarts L-BFGS every200acceptediterations. It remains unconverged and slightly trails the older output-projected solver. The measured fixed-core input metrics have median condition32.92and maximum389.75; none reaches the registered1e4extreme-conditioning bar. That does not exclude coupled reduced-Hessian conditioning, but it favors testing the restart schedule before implementing a new manifold optimizer.

This arm begins from the identical spectral pilot, with the identical initial conditional core optimum,64groups/rank16, full folded-unembedding coefficient objective and0.01group-energy penalty. It reuses the unchanged bounded controller with `iterations_per_cycle=1000000`, replacing200. L-BFGS history survives until an earlier solver termination or the1200-second soft budget. At every such boundary, it still emits/re-encodes the function and evaluates the fresh gradient and progress. Raw bounds remain[-1,1]; no old scaling loophole is reopened. Maximum100cycles and1800-second process alarm remain.

- A: initial CPU replay, final graph replay, reduced-objective identity, maximum re-encoding error and accepted objective increase≤1e-8; true inner normal residual≤1e-10; raw entries≤1+1e-10.
- B: fresh packed gradient≤1e-7 and20-step relative objective progress≤1e-6. Time exhaustion is not convergence.
- C: final normalized penalized objective at least1e-4below the completed200-step-restart baseline,0.8841870347451684. This is a different C from the earlier arm's gain-over-initial bar; both results retain their own registered predicates.

Report capture, elapsed time, accepted steps, restart cycles and fresh convergence. The null is no substantial practical advantage; one timed comparison does not establish universal optimizer superiority, and both fits may remain unconverged. Same model family and1,254,400floats/1024readers/1024products; no new inference cost, corpus fitting, or model-body execution. Stable circuit identity remains unproven. Better optimization could make subsequent parent/group comparisons more informative; lower loss alone is not a circuit discovery.

The wrapper/helper, original25dependencies, protocol and completed baseline receipt are frozen in SHARED_READER_RETAINED_HISTORY_V1_BINDING.json. Existing model-free native preflight and bounded-controller checks are reused. Queue through managedlane1 behind existing work; do not change live source or reset other jobs. Further optimization depends on the actual result.
