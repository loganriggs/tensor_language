# Three-hour mathematical review

Completed at 2026-09-20T04:22:08.810715+00:00. Next mathematical/literature review due 2026-09-20T07:22:08.810715+00:00.
Previous goal turn: progress (v629 falsified a specific compression baseline).
Full goal remains active; useful reuse and simplicity are not established.

## Native object and high-level judgment

For d=1152, h=4608, V=50304, the folded local quadratic tensor is
T_vij = sum_k C_vk (A_ki B_kj + A_kj B_ki)/2.
C=UD, A=LE, B=RE; E includes each residual contribution once. Input features
z can be intermediate products rather than raw coordinates. Only substitution
raises degree. Two pure bilinear layers give quartic H_vijkl; residuals mix
degrees. Normalizations, attention routing and final softcap remain operations.

The current scalar parent is a 256-square quadratic of normalized h17. Its
recent-path executable takes h16/x0/v1, computes MLP16 and attention17, and
retains background-square, twice-background-product, and product-square terms.
Those terms include cross interactions; none can be silently charged as free
native background. Upstream port generation is still missing from the extraction.

For higher-order folding, compare Sym(H-Hhat), allowing an unsymmetric Hhat.
A coefficient-tensor rank bound is tied to its representative, tree and norm;
it cannot rule out a smaller repeated-input arithmetic circuit. Low rank, few
nonzero interactions and few distinct reusable features are separate objectives.
For a quadratic error E and isotropic standard Gaussian input, direct fourth-moment
expansion gives E[(x^T E x)^2]=2||E||_F^2+(tr E)^2. Thus coefficient Frobenius
error is not even identical to Gaussian functional error; native normalized
inputs introduce a further distribution change. Report both where relevant.

## LITERATURE_SEARCH

Queries actually searched: “Grasedyck hierarchical singular value decomposition
tensors 2010 quasi optimal dimension tree”; “VeST very sparse Tucker factorization
1904.02603”; “bilinear language models interpretable mathematical framework arxiv
2024”. Opened primary sources:

- [Grasedyck, Hierarchical SVD](https://www.mis.mpg.de/publications/preprint-repository/article/2009/issue-27): HT stores order-m, mode-size-n tensors with bounded ranks k in O((m-1)k^3+mnk) values; truncation costs O((m-1)k^4+mnk^2). Mapping: our quartic coefficient tensor with declared output grouping. This is a compression baseline, without semantic uniqueness, automatic branch sharing or repeated-input quotient optimality. Publisher/PDF fetches failed; the institute abstract and author publication list were available.
- [Pearce et al., bilinear MLPs](https://arxiv.org/html/2410.08417v2): contracting an output direction into the bilinear tensor gives a quadratic interaction matrix; eigendecomposition is the direct conventional baseline for our scalar reader. It also discusses HOSVD for shared features. Their language-model findings do not prove our FineWeb component has a sparse reusable causal mechanism.
- [VeST](https://arxiv.org/abs/1904.02603): sparse factors and core via entry pruning and iterative updates. Its partially observed tensor setting is not our repeated-input polynomial quotient. Treat it as a sparsity-method reference, not a guarantee of native circuit recovery.

The inference for our plan: benchmark joint scalar eigendecomposition and joint
Tucker/HT before attributing failure of raw matrix compression to lack of circuits.
A learned sparse method must beat these at matched output, ports and full price.

## BASELINE_COMPARISON

The new [baseline ledger](DECOMPOSITION_BASELINES_2026-09-20.md) fixes the comparison
boundary and lists measured versus missing arms. Native MLP uses 4608 bilinear
products; any fixed scalar quadratic can instead use at most 1152 signed squares.
Neither is a complete semantic circuit. The recent-path reference is about97MB
and depends on three upstream ports. A proposed small parent cannot be compared
against the whole 18-block model while leaving those ports uncharged.

## REDTEAM_POSITIVE

v629's validation gate passes only because the selector returned the dense reference.
No compressed arm passed calibration: this is not evidence of successful compression.
The native component replay remains supported, but parent prediction across prompt
families is not independently demonstrated reuse in another computation. The two
panels are opened, not fresh. Selective semantic preservation and upstream closure
remain absent. Two independent executor tests were rerun successfully.

## REDTEAM_NEGATIVE

A planted CPU counterexample actually executed: f=.01*x0^2+x1^2, with
L=diag(100,1), R=diag(.0001,1). Rank-one stacked-matrix SVD keeps x0 and has
coefficient error nearly100%. Rescaling one channel by a=.001 leaves f unchanged,
but the same algorithm now keeps x1 with about1% error. Joint quadratic spectral
truncation also attains about1%. This proves gauge sensitivity can produce a bad
negative baseline even when the target is simple; it does not prove that gauge
balancing rescues our native result. Independent direct-product versus quadratic
evaluation passes, and a deliberate sign mutation is detected.

Native v629 still rejects only its raw-factor families. Its full-rank matrix audit
and factored-versus-dense executor checks find no obvious transpose/contraction bug;
they do not establish optimizer/global impossibility. A future negative learned
DAG result needs convergence, seed and planted recovery checks.

Executed consequence: check_decomposition_gauge_baseline.py and
DECOMPOSITION_GAUGE_CONTROL_RESULT.json. This changes the next action from more raw
SVD rank sweeps to a gauge-invariant contracted-object baseline (with balanced
factor SVD as a control, not the principal discovery method).

## Organization, efficiency and track handoffs

The cron process was live but /etc/cron.d/bilin18-reviews was absent; prior schedule
documentation was stale. Restored mathematical review scheduling separately, with
five-minute due checks and a three-hour interval, nonblocking overlap protection,
authenticated CLI, and a receipt freshness check. Future receipts must explicitly
include literature, baselines, positive red-team and negative red-team sections.
No historical review is fabricated for the missed hours. This is an organization
repair; schedule operation still depends on a running instance and working login.

Efficiency: v629 native job took under2seconds after waiting in the shared GPU
queue; reuse its executor/scoring rather than author another bespoke framework.
This review's consequence is a small CPU control; no new GPU workload or model fit.
CIRCUIT handoff: require matched preservation/null controls before calling the
scalar writer a semantic circuit. WEIGHT_FOLDING handoff: compare contracted
quadratic/HT objects, preserve gauge freedom and explicit normalization, and seek
reusable intermediate quadratic features rather than merely lower matrix ranks.

Five-property audit: prediction supported on opened/shifted panels with limited
scope; conditional extraction supported; selective semantic removal unproven;
useful reuse untested; simplicity not achieved. No completion claim.
