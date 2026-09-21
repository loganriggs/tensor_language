**Input covariance, the tensor metric, and the missing mean**

The paper's M acts on lifted tensor coordinates. It can represent a symmetric coefficient metric or a Gaussian function metric; these are distinct. An input covariance Sigma induces a particular lifted metric through higher moments. The paper also states structural restrictions for its efficient deep recursion. Our direct quartic contractions are checked independently rather than assuming that arbitrary DAGs inherit that recursion. [Paper, sections 2.2–2.3](https://arxiv.org/html/2605.15183v1#S2.SS2).

For our homogeneous quartic, a Gaussian function metric involves eighth input moments. Writing Sigma=S S^T, replacing native first-layer factors L,R by L S,R S and student factors p by p S implements N(0,Sigma) exactly. Five native cross, Gram and gradient comparisons against independent quadrature pass within3.5e-14.

This weighting can help approximate behavior in emphasized directions without giving comparable global fidelity. In the explicit redteam f=x0^4+x1^4 versus approximation x0^4, input variances(1,.001) produce relative Gaussian error about1e-6. Under isotropic inputs the same approximation has67.86%error. At variance zero the weighted error is exactly zero although the polynomials differ. Full-rank covariance avoids this exact degeneracy, but approximate equality can still conceal large differences in weak directions.

**The actual inputs are not centered.** The expanded6144-state calibration has mean energy fraction68.62%, covariance trace361.55 and second-moment trace1152. The covariance's minimum eigenvalue is.02290, so no numerical eigenvalue floor is needed for Cholesky. Its top four directions contain11.78%of covariance trace; the second moment's top four contain71.98%. Broad centered variation coexists with a large common mean.

The candidate Gaussian laws are consequently different hypotheses:

|Law|What it matches|What it omits|
|---|---|---|
|N(0,I)|Data-free isotropic reference|Activation statistics|
|N(0,Sigma)|Centered calibration covariance|Actual input mean|
|N(0,Sigma+mu mu^T)|Raw second moment|Actual mean and higher moments|
|N(mu,Sigma)|Calibration mean and covariance|Non-Gaussian higher moments|

Only calibration statistics are exported for fitting. The opened evaluation panel's covariance differs from calibration by73.7%in relative Frobenius norm; this is a statistic comparison, not a distribution-distance estimate or circuit error. It warns against assuming the fitted geometry transfers unchanged. Prefixes are not independent token samples and the evaluation panel is already opened.

An exact noncentral CP Gram uses764 partial pairings of eight affine Gaussian factors. The native teacher cross uses expected derivatives through order four, with a cached quadratic Gaussian projection. Five independent quadrature and gradient controls pass within3.4e-14. This enables a direct comparison of the four reference laws without training-probe interpolation. The student remains the same homogeneous quartic in original coordinates; introducing a shifted reference law does not add constant or lower-degree nodes to its executable program.

[Covariance controls and redteam](COVARIANCE_NATIVE_GAUSSIAN_CROSS_CONTROLS_V1.json) · [Actual input geometry](EXPANDED_INPUT_GEOMETRY_V1.json) · [Noncentral controls](NONCENTRAL_GAUSSIAN_CP_CONTROLS_V1.json).
