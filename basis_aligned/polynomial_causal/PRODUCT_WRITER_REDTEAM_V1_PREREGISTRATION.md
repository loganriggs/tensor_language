# Fixed-product writer red-team and native-interface solve

Source SPARSE_PRODUCT_DICTIONARY_V1 completed12sweeps/562.28fitseconds withAheld,
jointconvergence unmet, centeredcapture1.2205%,lambda.05739531,median3active.
No structural negative. Source products frozen for this diagnostic; no new data.

First calculate exact centered conditional optima: same sparse token supports
with unpenalized coefficients, then unrestricted coefficients on all512products.
For native factor crossC and productGramG, the latter is A=C G^-1. Since C is
Uc times a residual-space map, this unpenalized solution is automatically in
the centered unembedding column space. Same-support solves measure shrinkage;
unrestricted solves additionally measure the support restriction. Both retain
the same input functions, so neither proves joint reader optimization converged.

Second solve the separate full-U target with native legal writers and centered
token concentration. Let U=P R^T, P^TP=I, F=P-mean(P), and Z=R^TW. Fixed products
give GramG and full target crossC in P coordinates. Optimize the convex objective

$$
\frac12\operatorname{tr}(ZGZ^\top)-\langle Z,C\rangle
+\lambda\|FZ\|_1.
$$

The actual full token writer is PZ=UW. Common effects are included in the full
reconstruction objective; only centered effects receive the L1 penalty. This
is distinct from the source run's centered approximation plus exact common
function. No claim of unchanged common function or physical replacement.

Use two penalties: source lambda and0.1times source lambda. Full unpenalized
exact writer baseline is Z=C G^-1. ADMM separates auxiliary A=FZ for soft
thresholding. Its Z update solves a Sylvester equation, accelerated using
F^TF=I-V m m^T withm=mean(P). CPU control agrees with independent epigraph QP
objective within9.6e-12 and Sylvester residual2.9e-16.
Primary algorithm reference:
[Boyd et al., ADMM](https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf).
Convex fixed-function optimum is the target; no theorem of joint input recovery.

A: full targetenergy, whitening, exactOLSstationarity, fixed-support solves and
direct8tokenmatrix/implicitobjective replay<=1e-8.
B: both legal writer arms feasibility, stationarity and L1subgradient residuals
<=1e-5. Each gets180seconds/5000steps withrho1. An unfinished arm is not negative.
C: afterA andlowerpenalty convergence, lowerpenalty arm median top16centered
loading-energy share>=1.25times exactunpenalized baseline while fullcoefficient
capture>=80%of that baseline. Loading energy is not additive quadratic energy.

Literal price: same512products/1179648input numbers,589824residual writer numbers,
nativeU and normalizer/background retained. Auxiliary sparse A is solver state,
not the executable writer. Store actual whitened writers and conversion metadata
in ephemeral shared memory, checksums and complete results durably.

Interpretation: distinguish wrong products, excessive shrinkage, sparse supports,
and an incompatible output parameterization. A small objective alone does not
establish token-group stability, semantic meaning or any of the four properties.
