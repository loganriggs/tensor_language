# Rung 479 preregistration — gauge-aware blocks of equality downstream readers

Registered after rung478's strong null and before collecting any new reader or state statistic. The30 odd-root
circuits and documents500:1000 remain reserved. This is a discovery-only algebraic screen, not an intervention or
adoption claim.

## Question

Rung478 showed that sparse weighted sums of native MLP products can interpolate one32-circuit response view but do
not transfer even to the other discovery-document half. The native product coordinates and their sparse mixtures are
therefore closed as stable circuit units.

The exact MLP quadratic computation nevertheless has a coordinate-free downstream-reader form. For an MLP with
input `x`, left/right matrices `L,R`, output matrix `D`, and a downstream loss gradient `g` at the MLP write,

`Q_m(g) = sym(L^T diag(D^T g) R)`

satisfies `g^T MLP_quadratic(x) = x^T Q_m(g) x`. A family of such symmetric matrices has a common block structure
exactly when there is a non-scalar matrix `C` that commutes with every member: `C Q = Q C`. The eigenspaces of `C`
then define projectors onto blocks. Rotating coordinates inside a block does not change its projector, so a block
claim avoids the native-product gauge that defeated rungs477--478.

Rung479 asks whether the circuit-specific reader families of any pair among MLP8, MLP9, and MLP12 share such a block
structure on a fixed equality-state subspace, and whether that structure transfers across matcher source and document
half.

## Frozen data and sufficient statistics

Use exactly the corrected rung477b discovery authority:

-32 even-root circuit tags;
- documents0:500 split at row250;
- native and transplanted matcher sources; and
- member and matched in-slice control masks.

For each half, source, MLP, circuit, and mask, sum the gradient of selected next-token CE with respect to every MLP
write position and then average by the same mask count as rung477b. The circuit-specific output reader is

`g[m,v,c] = mean_gradient(member) - mean_gradient(control)`,

a1,152-number vector. In the same source forwards, capture the exact MLP input states. Relative to the equality-absent
forward, accumulate the covariance of the state change `delta_x = x_source - x_absent` over all causal positions in
each fixed view. Save only gradient sums, counts, and1,152×1,152 covariance sums; save no rows, tokens, logits, or
hidden states.

Replay, factor reconstruction, member/control support, batch-boundary allocation, forward/backward counts, covariance
symmetry/positive-semidefiniteness, and the equality-absent/source call paths are live instrument checks. The
rung477b member-minus-control product responses must also be reproduced by contracting these gradients with the
already computed equality product changes on a fixed checksum subset.

## Frozen32-dimensional equality-state restriction

Fit only the native-source documents0:250 view. For each MLP pair `(a,b)`, divide each endpoint's state-change
covariance by its trace, add the two matrices, and take its top32 eigenvectors `U_ab`. Thirty-two is fixed because
there are32 discovery circuit coordinates; it is not selected by variance retained, rank, or outcome. All claims are
explicitly restricted to this subspace, and the captured state-change variance is reported in every view.

For every circuit reader, form the exact projected matrix without materializing the full1,152×1,152 reader:

`A_mvc = sym((L_m U_ab)^T diag(D_m^T g_mvc) (R_m U_ab))`,

which is32×32. Verify it against direct quadratic evaluation on16 frozen random vectors. Normalize each matrix by its
Frobenius norm only for the block-structure calculation; retain the unnormalized matrices for circuit-labelled
response profiles.

## One frozen approximate-block calculation

For each MLP pair, use its64 normalized fit matrices (two MLPs ×32 circuits). Define the commutator loss operator

`cal_L(C) = mean_A ||C A - A C||_F^2`.

The identity matrix always has zero loss. The next-smallest eigenvalue `lambda2` measures how close the family is to
having an additional common block projector. Use the lowest-loss symmetric direction orthogonal to identity,
eigendecompose it, and make exactly one two-block split at its largest interior eigenvalue gap subject to both blocks
having dimension at least2. No block count, subspace dimension, rank, tolerance, or split is swept.

For16 frozen seeds, conjugate every projected matrix of the second MLP by one seed-specific Haar-orthogonal matrix,
leaving circuit labels, individual spectra, and each MLP's internal family intact while destroying a common input
block alignment. Repeat the identical calculation. These controls determine whether a small `lambda2` is evidence of
cross-MLP structure rather than generic smoothness of each family.

For the proposed two projectors, report the median and90th-percentile fraction of each reader matrix lying outside
the two diagonal blocks. Apply the fit projectors unchanged to the other three source×half views.

For block `P`, the basis-invariant signed circuit profile is

`profile[m,v,P,c] = trace(P A_mvc)`.

Center the32 values across circuits. Compare corresponding MLP profiles in every view, and repeat after omitting each
of the six even top-level families. This is a reader-structure screen; because it averages gradients before combining
them with states, it does not inherit rung471's paired-response or causal-intervention evidence.

## Frozen predictions

### A — lawful collection and exact projected algebra

- all preregistration, rung478 result/source, rung477b result/bundle/source, row, mask, and checkpoint hashes match;
- rung478 A/B are true, C/D/E false, and its strong null remains true;
- replay, factor reconstruction, projected quadratic identities, checksum contractions, support, batch allocation,
  finite-value, covariance, and exact call-count checks pass; and
- validation-family and SEALED outcomes remain unopened.

### B — a nontrivial shared algebraic direction beats conjugated controls

For at least one MLP pair, the real `lambda2` is at most one quarter of the5th percentile of the16 independently
conjugated control `lambda2` values, and the candidate direction is symmetric and non-scalar to numerical tolerance.

### C — the fit-view direction defines an approximate two-block family

Both proposed blocks have dimension at least2. On the64 fit matrices, median off-block Frobenius fraction is at
most`.20` and the90th percentile is at most`.35`.

### D — the same blocks survive source and document shifts

In each of the three non-fit views, median off-block fraction is at most`.30` and the90th percentile at most`.50`.
The top32 equality-state subspace independently recomputed in that view has squared principal-overlap at least`.70`
with `U_ab`.

### E — at least one block has a stable circuit-labelled reader profile

One of the two blocks has cross-MLP profile cosine at least`.70` in every view, exceeds the95th percentile of the16
conjugated-control minimum-view cosines by at least`.15`, and retains at least`.60` minimum-view cosine after omitting
at least five of six discovery roots.

## Strong null and routing

The strong null fires if A fails, every pair fails B, or every proposed block has minimum non-fit circuit-profile
cosine at most`.30` or fails to beat the conjugated-control95th percentile. A+B+C+D+E is only a screen: it licenses a
new preregistered pass that keeps gradient and state paired and then applies exact projector-defined block
interventions on the reserved odd-root families. It does not itself identify, remove, or compress a circuit.

With A and a strong null, retire the simultaneous-block route for this equality-MLP trio at the registered task
resolution. Do not change32, loosen the block bars, sweep ranks, or refit sparse products. Switch the active
decomposition test to the independent attention0/attention1 Q/K/output tensor direction, using downstream
discrimination and held-out interventions rather than native head boundaries.

## Price

GPU discovery collection plus CPU algebra. Zero deployed parameters saved or added. Report forwards, backwards,
peak memory, covariance storage, projected matrix storage, block metadata, control seeds, runtime, and all hashes.
