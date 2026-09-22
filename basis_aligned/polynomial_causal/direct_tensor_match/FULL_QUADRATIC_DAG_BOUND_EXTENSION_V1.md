# The quadratic product bound also applies to deeper polynomial DAGs

22 September 2026, 06:30 UTC. Correction to the scope caveat in FULL_QUADRATIC_PRODUCT_BOUNDS_V1.md. No new native spectrum or fit was computed.

**Higher-degree intermediates and cancellation do not evade the existing full-quadratic product-count bound for division-free polynomial arithmetic DAGs.** The earlier note excluded that case too broadly. The extension concerns the quadratic coefficient component, not arbitrary finite-sample functional approximation.

Take a finite DAG with scalar inputs, real constants, arbitrary affine combinations, and K binary scalar product gates. Sharing is allowed; each product gate is charged once. For a node u, write its homogeneous components about the origin as u_0, u_1, u_2, etc. At a product gate v=ab,

$$
v_2=a_0b_2+b_0a_2+a_1b_1.
$$

The first two terms linearly combine quadratic computations already present. The last term introduces at most one new product of two linear forms. Induction in topological order therefore puts every node's quadratic component in the linear span of at most K generators, one per product gate. Affine combinations introduce no new quadratic generator. Constants, shared branches, products of intermediate polynomials, and cancellation of higher degrees do not change this argument.

For any number of outputs, there consequently exist vectors c_k, a_k, b_k such that their quadratic coefficient tensor is

$$
T^{(2)}=\sum_{k=1}^{K}c_k\otimes\frac{a_kb_k^{\top}+b_ka_k^{\top}}{2}.
$$

Thus output-mode rank is at most K and input-mode rank at most 2K, just as for a one-layer quadratic product program. Linear sharing may still substantially reduce coefficient storage and additions; the bound charges neither. Arbitrary reuse can be valuable without defeating this necessary product floor.

The construction is the degree-two specialization of homogeneous-component propagation. Standard homogenization is discussed in [Shpilka and Yehudayoff, Section 2.2](https://www.cs.tau.ac.il/~shpilka/publications/SY10.pdf). The gate-count consequence here is derived explicitly above; it is not attributed as a separately verified theorem from that survey. Search indexed the section's general homogenization statement; direct PDF retrieval timed out in this session.

**Consequence for our full last-MLP tensor.** Existing measured unfolding spectra imply the same necessary product counts for these deeper DAGs:

| Coefficient metric | Products necessary for 10% relative error |
| --- | ---: |
| Folded Euclidean |1088|
| Original residual Euclidean |1090|
| Activation-covariance coefficient metric |818|

These remain numerical spectral lower bounds, not attainable constructions or formal interval-certified bounds. Native computation uses 4608 products. Broad sparse/shared programs between the bound and native cost remain possible. Narrow Tucker is only one excluded case; replacing it with a deep polynomial DAG does not erase the output-span requirement.

The statement also covers a DAG whose output contains other polynomial degrees **if the approximation criterion directly constrains its degree-two coefficient tensor**. It does not transfer to empirical text loss or Gaussian functional loss when other degrees are allowed to compensate on those input distributions. It does not cover variable division, RMSNorm, softcap, branching, or other nonpolynomial primitives. Nor does it constrain the separate 16-output quartic target by these particular native quadratic spectra. A bound on its degree-four component would require different accounting: a product's fourth-degree component has several cross-degree terms.

Executable check: check_quadratic_dag_bound.py propagates exact integer quadratic-generator coefficients alongside independently expanded full polynomials. Fifty random shared DAGs exercise constants, affine combinations, repeated branches, degrees through eight, and explicit cancellation. All 1,750 node identities pass. These controls check the recurrence and implementation; the inductive argument supplies the general statement.

[Original native bound and counts](FULL_QUADRATIC_PRODUCT_BOUNDS_V1.md) · [Exact-integer controls](QUADRATIC_DAG_BOUND_CONTROLS_V1.json) · [Implementation](check_quadratic_dag_bound.py).
