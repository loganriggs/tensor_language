# Three-hour mathematical, literature and operational review

Actual UTC: 2026-09-20T07:26:17.636993+00:00. Next eligible review: 2026-09-20T10:26:17.636993+00:00. Full goal remains active: simple,
OOD-predictive, extracted, selectively manipulable, reusable circuits. No completed
rung below establishes all five properties. This review is separate from the
hourly alternation; current handoff is CIRCUIT -> WEIGHT_FOLDING.

## Native object and current approximation

Native residual width d=1152, bilinear width h=4608,18blocks,9attention heads of
width128,50304vocabulary outputs. A bilinear numerator is D[(Lx)*(Rx)]. Folding
linear sources E and an output reader U gives C=UD,A=LE,B=RE and the symmetric
quadratic tensor T_vij=sum_k C_vk(A_ki B_kj+A_kj B_ki)/2. Two pure bilinear layers
produce degree4/order5; full attention has two QK factors times V (degree5 without
normalization). RMS/QK RMS, residual scales, cached first-layer values, RoPE and
softcap remain explicit. The normalized model is not a fixed polynomial tensor.

Current executable boundary: post11 response coordinates at every token plus
prepared native baseline contexts. Six stages12–17 combine exact conditional
attention contractions with reduced normalized MLP responses. Each stage has8
coordinates, W_l^T V_l=I, nonorthogonal Gram geometry, and shared unordered
quadratic pairs. Source A=embedding recurrence+early0–3; B=middle4–7+MLP8+MLP10.
Their post11 response superposition differs from a pre11 joint edit: measured
output gap5.03% of the original joint-effect norm. Never silently identify these
interfaces or replace propagation of products by linear readout vectors.

The sparse consumer retains18/36 MLP numerator pairs per stage, sharing each
product across8outputs. It has1718floating values+216integer support indices.
The producer has577536floating values plus six scales carried by the runtime.
Prepared v670 case tensors contain1943258values; artifact15,805,403bytes for8fixed
examples. Native baseline/initial-response generators remain chargeable external
computations. Six dense normalization Gram forms and all attention work remain;
108numerator products is not total quadratic work. Background-dependent attention
storage scales quadratically with sequence length in this implementation.

Gauges: invertible intermediate coordinate changes preserve the function when
all factors/geometry/readouts transform; pair sparsity is basis-dependent. Native
atom-norm ranking is invariant to diagonal rescaling/sign changes, not arbitrary
rotation. The full exported DAG passes a seven-boundary signed-diagonal gauge
audit. Inputs are declared response perturbations on text-derived contexts;
outputs currently four logit-margin contrasts, not the whole vocabulary.
Metrics: per-cell effect-relative L2, modal errors relative to target-effect norm,
absolute/norm-scaled replay, capability, null comparisons, literal tensor prices.

## LITERATURE_SEARCH

Actual queries included:
- discrete gradient chain rule product rule automatic differentiation exact finite differences McLachlan Quispel Robidoux
- site.arxiv.org DeepLIFT rescale rule reveal cancel multipliers difference from reference
- site.siam.org Grasedyck hierarchical singular value decomposition tensors 090764189
- site.arxiv.org balanced truncation quadratic bilinear systems 1705.00160 error bounds
- site.arxiv.org tensor networks arithmetic circuits repeated variables polynomial tensor symmetrization
- site.proceedings.mlr.press weighted automata Hankel rank spectral learning finite rank

Opened primary sources and mappings:

1. [Grasedyck HT](https://epubs.siam.org/doi/10.1137/090764189): fixed-tree hierarchical
low-rank tensor representation and truncation analysis. This maps to expanded
unnormalized numerator tensors, not our complete rational/softcapped network.
For m modes of size n and bounded rank r, a uniform tree stores O(mnr+mr^3)
values; ranks can grow. Repeated input slots and equivalent polynomial coefficient
representatives mean coefficient compression need not minimize computation.
Cross-branch reuse and sparse cores remain additional objectives. No canonical
semantic basis or native behavioral error bound follows. The MPI PDF link failed;
the publisher page was accessible.

2. [Benner/Goyal2017](https://arxiv.org/html/1705.00160v1): reachability and
observability guide quadratic-bilinear control-system reduction. Our X/Y snapshot
balancing borrows the dual-space idea, but empirical finite-reader snapshots are
not the paper's system Gramians. Transformer layers are discrete, nonstationary,
normalized and include explicit context. No nonlinear stability/error guarantee
has been transferred.

3. New search result: [Redmann2026](https://arxiv.org/html/2607.07841v1), tree-based
quadratic-bilinear solution representations and reduction. Its system is
xdot=Ax+Bu+N(u_tensor_x)+H(x_tensor_x),x(0)=0,y=Cx. Theorem3.16 assumes original
and reduced Gramians exist and controls satisfy its finiteness condition. Tree
expansions and exact removal of Gramian kernels are relevant mathematical
analogies, but our finite rational transformer has not been converted into a
system satisfying those assumptions. The theorem is therefore not a certificate
for the current8-coordinate model. It motivates checking actual full-path
observability before choosing coordinates. No uniqueness claim is imported.

4. [McLachlan/Quispel/Robidoux1998](https://arxiv.org/abs/math-ph/9805021) uses
discrete gradients for exact finite changes in dynamical systems. [DeepLIFT2017](https://proceedings.mlr.press/v70/shrikumar17a.html)
propagates differences from a reference through a network. These are relevant
neighbors for our endpoint-dependent readers, not evidence that attribution
closure identifies a causal circuit. Our attention/RMS formulas below are derived
and independently tested, not taken as an applicability theorem from an abstract.

5. [Arithmetic circuit tensor networks](https://arxiv.org/abs/2209.07410) connects
arithmetic circuits and tensor representations. Its abstract supports the
representation connection, not an extraction algorithm or guarantee for our
normalizers; the requested HTML version failed. This reinforces preserving a DAG
and repeated-variable semantics rather than forcing a symmetric dense tensor.
The weighted-automata search is a neighboring terminology check; no finite Hankel
rank assumption has been established for this model, so no realization theorem
is claimed applicable.

## Executed mathematical consequence

For any bilinear contraction M, delta M=M(delta a,mid b)+M(mid a,delta b).
For RMS R(x)=x/sqrt(mean(x^2)+eps), with r_i=sqrt(mean(x_i^2)+eps),m=(x0+x1)/2,
the exact two-endpoint pullback of g is

    q = .5*(1/r0+1/r1)*g - 2*m*(m dot g)/(d*r0*r1*(r0+r1)).

Applying these identities to both QK scores, their product, the value contraction,
RoPE transpose and nested RMS yields q dot delta_x = g dot delta_attention.
Compose with the normalized MLP and final RMS/softcap to obtain exact finite
readers at every suffix boundary. Both endpoints are required: these are calibration
and attribution operators, not predictors. Choices of contraction grouping can
change the allocation while preserving closure, so this is not unique attribution.

CPU tests pass finite closure, causal support and equal-endpoint agreement with
autograd, including a two-block full attention/MLP chain. v675 passes native all-six-
attention closure (norm-scaled max5.22e-7); v676 passes all-four-reader/all-seven-
boundary full-suffix closure (absolute max2.22e-6,norm-scaled max6.99e-7), exact
causal support and native endpoint replay. This directly enables replacing the
conditional-MLP calibration object with full dynamic readers and responses.

A second executed consequence improves efficiency without approximation: QR of
response/readers sample matrices reduces the central balancing SVD to at most
residual width in each dimension. The helper's test matches direct cross-snapshot
singular values, biorthogonality and the reduced cross operator. It avoids
materializing a large sample-by-sample matrix; nonlinear error bounds remain absent.

## BASELINE_COMPARISON

Use DECOMPOSITION_BASELINES_2026-09-20.md. Dense native suffix is the effect
reference; dense reduced cores control sparse pruning; matched random18 supports
and zero quadratic numerators are measured. Top18 passes the regular opened panel
(max7.66%effect error), top9fails28.63%, zero fails22.52%; one of three random18
supports also passes. Irregular nouns: top18max5.09%, but zero/random also pass and
native capability fails onecell83.3%. Do not hide that missing capability gate in
sparse-screen booleans. HT and alternative-tree native comparisons remain missing;
no empirical optimality claim. Same-width multi-reader and source-expanded frames
are distinct measured baselines, with larger calibration cost explicitly charged.

## REDTEAM_POSITIVE

Fresh multi-reader regular syntax passes target/control gates, but is scoped to
four readers and native-generated contexts. Independent CPU artifact replay and
whole-DAG gauge checks pass~1.8e-15. New source reuse reveals the limit: target
responses and nonlinear superposition largely predict, while per-cell modal
preservation predictions fail. This is stricter than the earlier pooled joint gate.
One irregular capability failure and passing random sparse supports limit semantic
and uniqueness claims. No full token-input extraction or broad reuse is asserted.

## REDTEAM_NEGATIVE

Dense-core rescue v672 does not resolve source preservation failure; independent
CPU scoring confirms identical native targets. Broader same-width source calibration
v673/v674 also fails. This is not a theorem against compact circuits. Suspected
mismatch: calibration still used attention-frozen MLP readers while prediction
used dynamic attention. The full finite-reader instrument now passes native tests;
next test changes that object before more ranks or sparsity thresholds. Preserve
all failed receipts and avoid post-selected row exclusions.

## Organization and efficiency

Organization audit found the explanation index and main path/module registries
still pointed predominantly to18September regional work; the subject response
artifacts and latest failures were not discoverable there. Append scoped links to
the subject-response dossier in the main circuit/path/module registries and add a
plain-language update. The generated49-package graph inventory is historical and
is not silently promoted to include this raw conditional .pt artifact.

Efficiency: native v673 inner runtime6.18s, while repeated runner authoring remains
larger. Prefix caching across three source variants retained16prefix calls instead
of redundant repeats. New model-free DAG runtime and batched four-reader pullback
reuse kernels; exact QR balancing is tested. Copying per-rung runners remains a
known cost; the next implementation should reuse traces/readers instead of inventing
another evaluator. Avoid a broad refactor until the calibration boundary settles.

## CIRCUIT / WEIGHT_FOLDING handoffs and ranked next step

CIRCUIT: return to independent source-write reuse, per-cell preservation and fresh
validation after a candidate changes; retain pre11-versus-post11 semantics.
WEIGHT_FOLDING: use the exact full-attention finite pullback to construct full-native
response/reader spaces at all relevant token positions, keeping width8, four readers
and declared costs. This is a new calibration object, not an assumed repair. Kill
it if source/control prediction fails under the existing gates. Do not claim
completion from local exactness. Source-specific hierarchical dictionaries are a
later alternative if a shared space fails; no generic rank sweep is registered.

The best immediate information gain is the full-native calibration comparison.
HT/control-system theory does not yet provide an applicable certificate; it changes
the questions we test rather than replacing those tests. Full goal remains active.
