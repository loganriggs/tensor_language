# Next hierarchy level: exact root function metric

The shared6 bank uses16totalproducts and24296coefficients, with fresh64error
18.15%. Before further bank-width sweeps, examine its ten root products
jointly. Root input q has dimension4 and consists of quadratic forms of the
original Gaussian x. The bank features are not themselves Gaussian.

Construct exact10x10 covariance G of phi(q)=[q_i q_j] for i<=j. Gaussian input
moments up to degree8 are needed. For x=m+z,z~N(0,I) and q=x^T Qx,
kappa_r = 2^(r-1)*(r-1)!*[tr(Q^r)+r*m^T Q^r m].
The fourth raw moment is k1^4+6k1^2k2+3k2^2+4k1k3+k4. Polarize this expression
in Q to recover mixed fourth moments of the four bank forms. Subtract the
root feature mean outer product. Verify with independent small Gaussian
quadrature before native use and audit PSD/precision.

With a free final constant preserving teacher Gaussian mean, exact root
function mismatch is tr((Z-Zhat)G(Z-Zhat)^T), since fixed W is orthonormal.
This uses a tensor-space metric for a non-Gaussian intermediate distribution.
Compute output-rank retention ceilings before registering optimization widths.
Count added linear combinations as well as removed products. No empirical
output targets needed. Input Gaussian-law and full-composition limitations
remain. The next implementation is the moment oracle, not a fitting sweep.
