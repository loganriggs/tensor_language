# Shared cubic source factors: native weight-only pilot

Rank16 source atoms h_r(s)=(a_r.s)(b_r.s)(c_r.s), s=current/base tuple of width2304.
All atom vectors stay on unit spheres. Exact source Gram and query/output
cross-Gram eliminate private quadratic query-dependent writers. Head outputs
are kept separately, through their native O_h, so different private normalization
gates are not assumed equal or canceled. Optimize attention17 query8/source7.
Evaluate fixed source features at source0 by recomputing the algebraic private
query readers, with no new source fitting.

Use fixed weight-only head scales
g0_h=1/(||Q1_h||F ||K1_h||F ||Q2_h||F ||K2_h||F), equal to substituting expected
isotropic squared RMS into the128-squared score divisor. This is a discovery
metric, not actual input-dependent normalization. Actual gates remain required
for a native program. No text, corpus or body forward enters fitting.

Two starts: native-aligned random key1/key2/value combinations, seed73330; fully
random unit source vectors, seed73331. Each uses the existing safeguarded
manifold L-BFGS, <=4000updates/300fit seconds. Objective is negative captured
coefficient energy divided by that start's fixed initial capture; stop at
projected gradient<=1e-6 in those declared units. Finite-source Gram support
uses exact Cholesky; invalid trial factorizations return infinity, not a ridge
change. Preserve time-limit/line-search failures and each arm's history.

- A: existing dense/gradient and four cold-start planted controls pass; native
  finite difference of normalized capture<=1e-4 relative discrepancy; first
  gradient evaluation<=2seconds. Do not start fits if native checks fail.
- B: both arms stop stationary and improve capture by>=20%over their own start.
- C: >=4 distinct source-atom matches across starts have absolute coefficient
  cosine>=0.9; each arm has>=4 literal cubic atoms used by>=2heads, each head
  contributing>=10%of that atom's summed coefficient energy, and summed
  individual-atom energy / total projected energy<=2.

All are structural-screen bars, not OOD/extraction/removal/composition adoption.
Report absolute coverage using1024 independent coefficient probes per position,
seed73332, and held-position capture. Do not infer absent structure from rank16,
the initialization, local convergence, the fixed metric, or a failed sharing bar.
The normalizers and native background remain outside this polynomial objective.

Price: one managed GPU, zero body forwards, FP64;800-second hard limit. Store
atoms and optimization histories, not checkpoint weights or giant tensors.
Literal source readers110592floats; source Gram256. Private query/value maps and
all remaining native weights/gates must also be counted before any cost claim.
