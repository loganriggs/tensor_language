**Compare fitting paired states with fitting explicitly interchanged contexts**

The global-ball robustness improvement did not fix native transfer. This experiment instead defines a concrete compositional test on the original calibration data: pair each of448source states with each of448recipient contexts, holding the recipient carry and RMS17denominator fixed. This creates200,704normalizer-clamped interchanges. It does not assume natural source/context independence or recompute the full model after a swap.

The generated component is

$$
\phi_{ij}=\left(\frac{c_i+\tfrac12q_a(z_j)}{s_i}-\alpha\right)
\left(\frac{q_b(z_j)}{s_i}-\beta\right).
$$

With the first read fixed, changing existing second-read coefficients makes the error affine in those coefficients. Its squared loss is exactly quadratic. The Cartesian sum reduces to source-feature Gram matrices weighted by context sensitivity moments; four model-free controls match explicit enumeration. This retains the actual empirical nonlinear source features rather than replacing their moments with Gaussian closure.

Both arms adjust the same32existing correction coefficients. They retain all eight original coefficient, value and derivative safeguards and constrain paired generated error to at most1.10times the parent. Neither adds products, directions or stored coefficients. Only the objective differs: original paired examples versus all source/context pairings. No opened test-panel data are used for fitting.

**Numerical correction retained**

The unscaled Cartesian solve terminated after one iteration without moving, while the paired solve moved. Rescaling each coefficient by its objective curvature and normalizing the objective leaves the mathematical optimization problem unchanged. After this repair, the paired and Cartesian arms converge in52and25iterations, with constraint violations below1.9e-12. Both unscaledV1and rescaledV2results/programs are retained. The no-movementV1Cartesian outcome should not be treated as evidence against the hypothesis.

| Objective optimized | Paired generated error | Cartesian interchange error |
|---|---:|---:|
| Original graph |8.871%|3.285%|
| Paired fit, rescaled |7.320%|3.130%|
| Cartesian fit, rescaled |7.843%|3.046%|

Each column has its own target-variation denominator, so smaller Cartesian percentages do not mean that Cartesian examples are intrinsically easier or better reconstructed in absolute units. The Cartesian arm beats the paired arm on its intended risk, but improves the parent by7.29%, **failing the registered10%gain prediction**. This failure remains explicit.

Both saved forms replay within3.8e-15. These are fitting results, not accepted circuits. A frozen two-candidate screen on the already-opened panels will compare both interfaces and both original323-width baselines. That comparison tests whether this objective distinction affects transfer; it cannot restore fresh status to previously examined data.

[Rescaled fits and controls](INTERCHANGE_READ_V2.json) · [Unscaled results](INTERCHANGE_READ_V1.json) · [Exact conditional metric](interchange_quadratic_metric.py) · [Transfer plan](INTERCHANGE_TRANSFER_PLAN_V1.json).
