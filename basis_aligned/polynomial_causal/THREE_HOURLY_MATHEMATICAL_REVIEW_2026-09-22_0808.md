# Mathematical and operational review — 22 September 2026, 08:08 UTC

ACTIVE_TRACK: WEIGHT_FOLDING, under the user's explicit focus through 15:10 UTC. Previous mathematical review: 05:06:30; this review includes a new executed CPU consequence. The full goal remains predictive, selectively manipulable, extracted, reusable and literally simpler circuits covering the full third-order tensor and selected deeper paths. No present reconstruction result proves completion.

## Object and exact metric question

For one bias-free bilinear layer with residual input dimension $d=1152$, channel width $h=4608$ and vocabulary $V=50304$,

$$
F(x)=UD[(Lx)\odot(Rx)],\qquad
T_{vij}=\tfrac12\sum_k(UD)_{vk}(L_{ki}R_{kj}+L_{kj}R_{ki}).
$$

The full object has shape $V\times d\times d$. QR of $UD=QA$, with orthonormal columns in $Q$, preserves Euclidean output errors within its span and permits fitting the smaller output coordinate system. Truncating that span would add approximation; QR alone does not. RMSNorm and final softcap remain explicit operations.

The queued hybrid experiment instead concerns the **selected pure quartic** through MLP16 and MLP17: bias-free MLP16 output, multiplied by its residual transport coefficient, occupies both input slots of MLP17. Sixteen fixed readers observe that contribution. Its order-five coefficients have shape $16\times1152^4$. Attention, other residual sources, biases and mixed terms remain excluded. All four input slots receive the same vector; only the input-symmetric coefficient part determines the polynomial. A compact unsymmetric representative is allowed. Internal scale, permutation and basis gauges do not establish semantic identification.

The correction keeps CP512 fixed and adds 96 quartic atoms, eight for each output 4–15. Each atom is a product of four learned linear forms. It adds 288 variable multiplications and 442,464 coefficients; outputs 0–3 are protected. This is extra discovery capacity, not final simplification. A final arithmetic DAG must charge each distinct reusable product once and also charge nonzero linear coefficients and additions. Native normalization/state-production costs remain outside this local price.

For a frozen eight-feature vector $\phi_g(x)$, define

$$
G_G=\mathbb E_{x\sim\mathcal N(\mu,SS^\top)}[\phi_g(x)\phi_g(x)^\top],\qquad
G_E=\frac1N\sum_n w_{ng}\phi_g(x_n)\phi_g(x_n)^\top.
$$

Here $\mu,S$ are the existing calibration-derived Gaussian parameters. For the sensitivity metric, $w_{ng}$ is the native squared logit-tangent sensitivity, normalized to mean one in each output; the ordinary text metric uses $w=1$. These are eighth-order moments of the original inputs, not just their covariance. The general lifted matrix $M$ therefore carries more information than an input covariance matrix.

If $G_G=LL^\top$ is positive definite, the eigenvalues of

$$
K=L^{-1}G_E L^{-\top}
$$

give the smallest and largest ratios $c^\top G_Ec/(c^\top G_Gc)$ over all nonzero combinations of this frozen dictionary. This is a direct finite-dimensional metric comparison, not a guarantee about residuals outside the dictionary.

For the half-and-half hybrid metric,

$$
G_H=\tfrac12G_G+\tfrac12G_E,\qquad
L^{-1}G_HL^{-\top}=\tfrac12I+\tfrac12K.
$$

Thus its eigenvalues are exactly $(1+\lambda_i(K))/2$, and $G_H\succeq G_G/2$. More generally, for any residual function with finite losses, $E_H=(E_G+E_E)/2\ge E_G/2$. This prevents an empirical null direction from being completely unpenalized in the hybrid **objective**. It does not promise improvement over a Gaussian baseline, native finite-removal fidelity, good optimization, or low error on a new distribution. Ridge is a separate coefficient penalty; its effect depends on feature scaling.

## Literature mapping and limits

[Cohen and Migliorati, Optimal weighted least-squares methods](https://arxiv.org/pdf/1608.00512) studies approximation in a fixed linear function space with independent samples. A space-dependent sampling measure and weights give stability with sample count proportional to dimension up to a logarithm. Here each frozen eight-atom dictionary supplies such a linear space, and its Gram matrix is the corresponding normal-equation matrix. But our text states are document-correlated, our features were learned adaptively, and native sensitivity weights are not their sampling-density correction. Their theorem therefore does not license an $N\gg8$ adequacy claim here. Solving the frozen readout is small; identifying its nonlinear feature space remains unresolved. The executed generalized spectrum checks the actual finite matrices instead of assuming that theorem's hypotheses.

[Grasedyck's HT decomposition](https://epubs.siam.org/doi/abs/10.1137/090764189) organizes coefficients using a dimension tree and hierarchical ranks, with storage $O((n-1)r^3+ndr)$ for order $n$, mode size $d$, uniform rank $r$. Our four repeated input slots can be placed on such a tree. That representation supplies a compression baseline, not a guarantee of a sparse shared arithmetic DAG or a favorable basis. Frobenius coefficient guarantees do not automatically transfer to the non-product empirical lifted metric. No HT refit is warranted solely by the current geometry audit.

The literature search also revisited [spectral learning of weighted automata](https://proceedings.mlr.press/v57/arrivault16.pdf). Its constructive object is a complete finite Hankel block for a weighted language and a low-rank realization. Our current object is a local commutative polynomial at a fixed normalized state, not a demonstrated finite-state linear realization over token strings. No rank/recoverability guarantee transfers without constructing that different object. This is not the next experiment.

The recent [quadratic DAG bound extension](direct_tensor_match/FULL_QUADRATIC_DAG_BOUND_EXTENSION_V1.md) remains more directly useful for the full third-order objective. Each binary product contributes at most one new quadratic product of linear forms; arbitrary reuse and higher-degree cancellation do not evade the resulting output-mode rank bound on quadratic coefficients. Existing numerical necessary floors remain metric-specific; they do not constrain a quartic text-loss replacement or establish attainability. No new universal circuit lower bound is claimed here.

## Executed consequence: native frozen metric geometry

[Code](direct_tensor_match/audit_local_metric_geometry.py) and [receipt](direct_tensor_match/LOCAL_METRIC_GEOMETRY_V1.json) compare all 24 local spans: twelve smaller outputs in each of two frozen Gaussian Adam starts. Each span has eight quartic atoms. Gaussian moments use the existing analytic noncentral moment contraction. Calibration has 6,144 states; evaluation has 16,384 states. No targets or coefficients are fitted.

| Metric ratio | Minimum eigenvalue across spans | Maximum | Spans entirely within [0.5, 2] |
| --- | ---: | ---: | ---: |
| Ordinary calibration / Gaussian | 0.4842 | 2.2171 | 22/24 |
| Sensitivity-weighted calibration / Gaussian | 0.03709 | 0.87345 | 0/24 |
| Ordinary evaluation / ordinary calibration | 0.30447 | 1.42255 | 5/24 |

The factor-two range is a descriptive reference, not a preregistered success gate. The sensitivity weighting substantially changes the norm of some combinations even after normalizing each output's mean weight to one. The evaluation panel also measures some combinations more weakly than calibration. These differences mix sampling variability and distribution geometry; this audit cannot separate them. They do not directly measure target residual error, because the native residual is not asserted to lie in these eight-feature spans.

Controls pass: known non-diagonal generalized spectrum, positive finite spectra, and the exact hybrid eigenvalue identity in all 24 spans. Cholesky succeeded without ridge; adding ridge to make it succeed would have changed the tested metric. Float64 CPU execution only. No semantic or causal identification follows from these eigenvectors. The feature spaces remain fixed; no oracle readout is exported.

## Red-team conclusions and next decision

Positive result to challenge: a lower aggregate value error can hide weak-output failures. The paired whole-document bootstrap now shows only about 7% improvement, with 95% intervals around 6.7–7.5%, on the opened panel. That does not establish fresh OOD behavior. The metric floor above is not an error-to-baseline guarantee.

Negative result to challenge: weak transfer from Gaussian fitting does not prove no simple native circuit exists. Today's calculation directly demonstrates metric differences within existing candidate spaces. Conversely, changing the metric cannot create missing computations in a fixed span, and earlier readout oracles already leave large residuals. The queued learner changes directions jointly and is the appropriate pending discriminator; do not tune its mixture after seeing evaluation outcomes.

If both hybrid starts pass their fixed value/response and Gaussian-retention bars, proceed to finite native intervention and simplicity accounting. If they fail, inspect whether error remains in the same output components and change representation or the source/context geometry. Do not repeat frozen-readout or minor optimizer sweeps. Full-third-order coverage remains a distinct required route, with native 4,608 products and the coefficient spectral floors as explicit comparison points.

## Organization, time and handoffs

Authoritative evidence since 05:06: native Adam/Muon and L-BFGS results; normalization-stage removal failures; token/position attribution nulls; quadratic DAG bound; document-bootstrap analysis; this metric audit. The most recent own completed commit before this review was c27355aad. The preceding Pythia job completed at 07:41:04; SmolLM PID 1147818 is the current managed predecessor, and hybrid is next. Queue latency dominates; no own native fit completed during that wait. No service, queue order or frozen runner helper was changed.

The prior review identifies stale historical index labels. This review links primary receipts directly and does not overwrite another worker's module dossier. The current Gaussian, hybrid and local metric analyses are method evidence for the same selected quartic path, not new semantic circuits. The new files form one evidence chain through this review. A generic runner refactor would risk the hash-bound queue and adds no scientific value now; cached CPU analyses already avoid duplicate native state capture.

Circuit handoff: outputs 4–15 are probe coordinates, not named concepts. Candidate improvements need selective removal, unrelated-reader controls and joint replacement. The protected output 1 failure cannot be repaired by this learner. Folding handoff: preserve the full third-order path and compare sparse/low-rank local forms and arithmetic reuse under the same metric and literal accounting. A mixed-metric improvement alone is not adoption.

Next three-hour review is due 11:08 UTC. The queued two-start hybrid comparison and prepared independent export audit remain the immediate continuation.
