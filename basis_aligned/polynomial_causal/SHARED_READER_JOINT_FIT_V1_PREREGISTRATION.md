# Matched native joint reader fits with bounded coordinates

Four separately managed20-minute arms compare the original64rank16LL1 groups with the compatible shared-parent graphs, using both spectral and native-product pilot starts. Every arm optimizes input spaces and output directions while solving all symmetric interaction cores jointly. The target is the complete folded unembedding coefficient tensor; the group-energy penalty is0.01. No corpus fitting or model-body forwards.

The original-space arms start from the same pilots used to construct the shared graphs and eliminate their cores before optimization. The shared arms start from their completed conditional core fits. Their initial objectives must match the CPU reference to $10^{-8}$ before spending the optimization budget.

Every raw reader/private/output entry is bounded in[-1,1], mapped through the packed numerical scales. Every normalized representation is feasible under this bound. After at most200acceptediterations, or any earlier solver termination, emit and re-encode the same graph and check the fresh gradient. A raw solver success flag alone cannot end the search as converged. Maximum100cycles,1,200softseconds per arm and1,800second process alarm.

Per-arm registered bars:

- A: initial CPU objective agreement, final graph RMS replay, reduced-objective identity, maximum re-encoding objective discrepancy, and accepted objective increase each at most $10^{-8}$; inner normal residual at most $10^{-10}$; raw entries at most $1+10^{-10}$ in absolute value.
- B: fresh maximum packed gradient at most $10^{-7}$ and relative objective change over20acceptedsteps at most $10^{-6}$. A time limit is not convergence.
- C: normalized penalized objective improves by at least $10^{-4}$.

After all arms finish, compare each shared graph to its equally optimized original-space arm: coefficient capture gap at most0.001 while saving at least1% of stored floating coefficients, in both starts. Count graph incidence and variable products separately. Cross-start parent/function agreement is descriptive at this stage. The common whitener, native unembedding, bias and full model background remain required.

The checked native kernel takes about0.28seconds per evaluation and less than0.4GiB allocated GPU memory. This supports substantial optimization, but does not prove global recovery. Bounded planted independent-start recovery remains1/4 versus the registered2/4bar. Two fresh-coordinate misses show no corroborated Hessian direction below $-10^{-4}$ under the executed check. These limitations are preserved; the native experiment is a matched local-refinement test of weight-proposed warm graphs.

Sources and inputs are frozen in SHARED_READER_JOINT_FIT_V1_BINDING.json. Four wrappers allow coordination at20-minute job boundaries. No direct GPU process or hidden follow-up is authorized by the script itself; further runs depend on actual results.
