# Transfer-interface audit and full-input quartic route — 2026-09-20 16:11 UTC

The previous176-row failure remains a valid failure of the frozen amplitude-coordinate program. It is not evidence that no reusable ambient-residual circuit exists. Reading the native exporter confirms both the1152×5 input port K_b and the4×1152 output reader q_b change with the row. The same global two-layer polynomial is restricted and read out differently.

If square invertible ports were available, coefficients could be transported exactly. Our constructive same-circuit control has177.37% naive frozen-coefficient error and8.96e-16 error after input and output coordinate transport. Native ports are rectangular and can select different subspaces, so this control is not evidence that the real row difference is fully transportable.

A second constructive control proves the missing information problem. In six ambient coordinates, F=(x0²+…+x4²)² and G=F+x5⁴ agree on the port x=(a0,…,a4,0), yet differ on x=(a0,…,a4,a0). A local polynomial alone cannot choose between these ambient extensions. This explains why a pseudoinverse lift would require an additional assumption, rather than recover a uniquely defined global circuit.

## Exact full-input coefficient queries

The next route targets a fixed ambient polynomial directly. Let S_ij be the symmetric residual-vector coefficient of the first bilinear layer, and B the symmetric bilinear map of the second layer including its output readout. The fully symmetric quartic coefficient is

$$
H_{vijkl}^{\mathrm{sym}}=\frac13\left[
B_v(S_{ij},S_{kl})+B_v(S_{ik},S_{jl})+B_v(S_{il},S_{jk})
\right].
$$

An entry batch can be contracted from native factors without storing H. This matches the polynomial on repeated inputs exactly, including symmetries across the first-layer pairing. It is not merely the cheaper restricted tree symmetry.

Independent validation against a dense tensor averaged over all24 input permutations gave:

- Relative entry error4.45e-16.
- Maximum relative parameter-gradient discrepancy2.23e-15.
- Enumerated coefficient Frobenius norm discrepancy2.10e-16.

Uniform ordered-index sampling estimates the squared coefficient Frobenius numerator. It does not estimate Gaussian function error, and a ratio of sample estimates is not generally unbiased. On the dense toy,512 queries had8.28% relative RMSE over100 trials;32,768 queries had0.90%. On a single-coordinate quartic,512 queries returned exactly zero in65/100 trials despite a nonzero tensor. This is a direct falsifier of assuming uniform sampling is automatically adequate for sparse structure discovery.

The queued native pilot queries the pure MLP16→MLP17→unembedding numerator on all1152 possible input coordinates. It retains normalization and omitted residual/attention terms outside the claim. It compares uniform sampling with exact-count weighting of the five input-index collision patterns, checks a complete four-coordinate polynomial restriction, and repeats a small batch in float64. No native stochastic student fit is claimed yet; measured variance will decide whether that is a useful next optimization method.

The full quadratic variable-projection experiment remains queued separately. The shared GPU is occupied by another verified live research job; no duplicate process was started.

Receipts: `TRANSFER_INTERFACE_AUDIT_V1.json`, `IMPLICIT_QUARTIC_CHECK_V1.json`. Code: `audit_transfer_interface.py`, `implicit_quartic.py`, `check_implicit_quartic.py`. Native preregistration: `NATIVE_QUARTIC_QUERY_PLAN_V1.md`.
