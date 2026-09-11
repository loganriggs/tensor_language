# Shared linear parents from LL1 input subspaces

This proposal method searches the input spaces of whole quadratic groups, rather than matching their individual eigen-square directions. It is motivated by the exact uv/uw counterexample: the shared parent u lies in both input spaces even when no square axes agree.

For each pair of rank16 input spaces, principal vectors with cosine at least0.95 propose their normalized midpoint u. For each proposed unit reader, let

$$
\alpha_g=u^TQ_gu,\qquad v_g=Q_gu-\alpha_g u.
$$

The exact portion of group g that touches this reader is

$$
c_g\left[\alpha_g(u^Tx)^2+2(u^Tx)(v_g^Tx)\right].
$$

Subtracting this from the group's quadratic is equivalent to evaluating that group at the input projected perpendicular to u. Since v_g is perpendicular to u, squared-reader and mixed-product coefficient terms are orthogonal. The mixed term's individual energy is twice ||c_g||^2 ||v_g||^2. Cross-group cancellation is included when scoring a combined candidate.

Candidate consumers require squared membership in the group's input space >=0.95 and mixed-parent energy >=1% of that group's energy. Parent proposals are clustered by complete-link absolute reader cosine >=0.99. The combined mixed part must carry >=10% of its parent component's energy and >=1e-4 of whole native tensor energy. All thresholds were registered before the census.

The spectral/native pilots yield15/17 retained parent proposals. Only1/0 have at leastthree consumer groups, missing the prediction of at leastfour. However10/9 groups have multiple proposed parents, with maximum degrees4/3; that prediction holds. All tested coefficient and execution identities pass. [Receipt](LL1_SUBSPACE_PARENTS_V1_AUDIT.json), [implementation](ll1_subspace_parents_v1.py).

These are input-incidence proposals from unconverged fits, not identified circuits or a jointly executable DAG. Two parent components can contain the same interaction. Adding them naively would double-count it. A joint implementation needs one shared quadratic core or appropriate overlap corrections; its storage and computation must be priced. Multiple parent membership alone does not prove causal reuse or semantic hierarchy.

The ongoing projected LL1 run addresses a separate uncertainty: whether the parent groups stabilize after better optimization. Its equilibrated conditional output solve is exactly equivalent to the original group normal equations and has an explicit condition bound. [Registered run](PROJECTED_LL1_CONVERGENCE_V1_PREREGISTRATION.md). Native results must be interpreted before carrying these proposals over to a new fit.

The proposal readers and incidence records are saved in LL1_SUBSPACE_PARENTS_V1_PROPOSALS.pt. This is a proposal artifact, not a runnable model replacement. SHA256: 2191d9b073d29385448d5284cbe2e687005dd88871e6fb8ebc73f482cba1dc1a.
