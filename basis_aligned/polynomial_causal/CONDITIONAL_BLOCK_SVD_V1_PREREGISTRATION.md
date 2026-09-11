# Exact conditional block update

Weights only, full unembedding metric, last bilinear layer, latest frozen
BLOCK_TRUST_REGION_THIN_V1 checkpoint. Same 16 overlapping blocks, 16 readers
per block, four output/core pairs; lambda=.01; 377344 coefficients plus native
unembedding and background. No body forwards, data access or new representation.

For orthonormal input frame E_g, project the residual excluding block g into
its symmetric quadratic core space. Whiten the residual-output metric U^T U.
Call the resulting 1152-by-136 coefficient matrix R_g. Every rank<=4 matrix A
is representable by the four writer/core pairs. Minimizing their energy penalty
at fixed A gives ||A||_*^2/4. Hence the exact conditional problem is

$$
\min_{\operatorname{rank}(A)\le4}
\|R_g-A\|_F^2+\frac{\lambda}{4}\|A\|_*^2.
$$

Von Neumann's trace inequality aligns singular vectors with R_g. For its top
four singular values s_i, solve the strictly convex scalar problem

$$
\min_{t_i\ge0}\sum_{i=1}^4(t_i-s_i)^2+
\frac{\lambda}{4}\left(\sum_i t_i\right)^2.
$$

For k active values, threshold tau=(lambda/4)*sum_{i<=k}s_i/(1+k*lambda/4),
and t_i=max(s_i-tau,0). Find the consistent active set among four possibilities.
The discarded singular values contribute a constant. Hadamard rotation gives
a factorization attaining the original component penalty; normalize cores and
transfer their scales to the writers. Thus this optimizes the EXISTING objective,
not an unreported replacement penalty. The squared nuclear mapping is local
to a fixed input block, not the whole nonlinear model.

Related primary literature: [column-energy penalty](https://arxiv.org/abs/1710.05092)
and [singular-value thresholding](https://web.stanford.edu/~candes/svt/index.html).
The rank-capped squared-nuclear formula above is our direct derivation, not
an assertion that ordinary nuclear-norm thresholding is identical.

CPU controls compare packed and dense target/cost, an independent bounded
scalar solve at four penalties, and planted rank-three recovery with zero
penalty. Maximum error <=1e-9. These already pass. Then native cyclic updates
run <=100 sweeps /180 fit seconds, with full objective and conditional-gap
checks every five sweeps and at the end. Keep frames fixed throughout.

Pred_a: initial native loss/residual replay and maximum normalized local/global
objective increase <=1e-8; frames unchanged and unit cores <=1e-8.
Pred_b: maximum possible single-block objective decrease, divided by native
energy and captured fraction, <=1e-6 at final state. This is conditional
block optimality only, not all-parameter convergence.
Pred_c: capture >=1.05 times .08634383041327387, preserving same metric/price.
Null: exact joint content updates give little benefit; input-frame search,
capacity or structural assumptions remain the higher-value questions. A failed
conditional-gap bar requires further optimization diagnosis, not that inference.

Report input/core manifold gradient norms under the existing reduced objective
after the update. A remaining input gradient is evidence against calling the
whole fit converged even if all conditional block solves settle.
Save the new small checkpoint in /dev/shm; previous states and bound source
bytes remain unchanged. Managed lane1 only, hash-bound dry-run before enqueue.
