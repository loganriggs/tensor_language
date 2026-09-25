# Review addendum: polynomial coverage and document-level red-team

2026-09-21 10:54:11 UTC. Concurrent-review rule applied: [10:50 mathematical review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_1050.md) appeared after the initial07:44 check. I read it and verified that it contains primary-literature work and a substantive executed structural test. This addendum does not create another three-hour review or reset its clock. Next deadline remains13:50 UTC.

The independent new consequence is an exact polynomial metric-coverage test: a positive coefficient floor can leave an800-fold squared-error transfer factor even for distributions with identical covariance and input radius. A native receipt audit also supports the mixed candidate's absolute result at document grain, while exposing covariance-only document failures. Neither establishes native causal fidelity or repairs the registered relative-baseline failures.

## LITERATURE_SEARCH

Actual queries:
1. “distributionally robust optimization moment uncertainty quadratic loss semidefinite Delage Ye 2010”
2. “sum of squares polynomial moment matrix generalized eigenvalue distribution shift norm bound Lasserre”

Opened primary sources:
- [Delage and Ye, Distributionally Robust Optimization under Moment Uncertainty](https://web.stanford.edu/~yyye/distRobOpt_revision1.pdf), full author-hosted paper, §2. Moment/support ambiguity is a relevant design pattern. Its tractability assumptions do not solve our nonconvex tied-factor search, and input covariance alone does not determine quadratic squared loss.
- [Lasserre and Pauwels, Sorting out typicality with the inverse moment matrix SOS polynomial](https://arxiv.org/pdf/1606.03858), full paper, Assumption1, Lemma1 and Theorem2. Their inverse-moment variational identity requires a positive-definite polynomial moment matrix and finite moments. It gives the closest algorithmic match: measure which polynomial directions a distribution controls. Coordinate invariance requires congruence transformation of the complete metric.
- [When Are Two Networks the Same? Tensor Similarity for Mechanistic Interpretability](https://arxiv.org/html/2605.15183v1), §2 and AppendixA.3. Lifted moments include self-contractions; Gaussian closure is an assumption. Covariance-shaped coefficient loss is not empirical fourth/eighth-moment loss, and explicit native normalization prevents a plain polynomial end-to-end mapping.

All full sources opened successfully. Two PDF text-find attempts on Delage/Ye returned no-match/internal-error placeholders, not successful targeted finds. The paper itself was accessible. Search completed. Plan implication: a frozen low-dimensional coverage diagnostic can distinguish metric coverage from capacity before more empirical refitting; a full native moment SDP is unjustified.

## Mathematical consequence and execution

The current six-read coefficient object is Q in R^(6×1152×1152), symmetric in its input slots. Source reads q_o=z^T Q_o z feed the three conditional products defined in the10:50 review, with native normalized MLP16 input z and pre-normalization MLP17 input h. The shared program has560 mixed atoms plus32 squares private to output5. It retains six affine/mean corrections, three h readers, RMS, centering constants and a common writer. At fixed h, numerator degree is at most4 in z; native attention, normalization and final softcap remain external nonlinear computation.

Source expansion remains essential: r=r0+a+m gives three quadratic self terms and six ordered cross terms, including a^TQm and m^TQa. For attention, each QK factor sums ordered query/key-source pairs; multiplying both score factors and the value gives the full QK1×QK2×V five-slot numerator. No QK2/value omission is justified by a single-factor fold. This local six-read approximation retains all terms of its selected quadratics, but not every model output or the recursively changing later h. Learned residual coefficients, signed first-layer value mixing, native biases, rounded rotary semantics and FP32 RMS epsilon remain part of the native contract.

Our derivation from the inverse-moment identity uses orthonormal symmetric coefficient coordinates psi(z)=svec(zz^T). For p_c(z)=c^Tpsi(z) and positive-definite M:
\[
K_M(x)=\psi(x)^TM^{-1}\psi(x),\qquad
|p_c(x)|^2\le K_M(x)c^TMc.
\]
The extremizer satisfying p(x)=1 is c*=M^(-1)psi(x)/K_M(x), with minimum metric energy1/K_M(x). For a shifted moment matrix N, the sharp squared-loss factor is the largest generalized eigenvalue of (N,M). If ker M is not contained in ker N, the factor is infinite. This is an exact norm comparison, not a theorem that a compressed native circuit is faithful.

Executed [CPU script](direct_tensor_match/review_moment_certificate_20260921_1047.py) with /venv/main/bin/python, CUDA hidden and two CPU threads. Authoritative output: [V2 control receipt](direct_tensor_match/REVIEW_MOMENT_CERTIFICATE_20260921_1047_V2.json).

The planted d=8 example uses36 homogeneous symmetric coordinates. Uniform sign vectors and signed coordinate axes both have covariance I and exactly radius sqrt(8). Their moment ranks are29 and8. q=x0²-x1² has training loss0 and shifted loss16. This repeats the earlier distribution distinction only as the fixture; the new result is its exact coverage certificate and extremizer.

With M=Ms+0.01I, the sharp shifted/training squared-loss factor is800. At an axis point, output1 requires only0.00017854 regularized training energy; leverage is5601. Thus even bounded, equally normalized inputs can escape a weak floor. Five congruent basis changes preserve the certificate; resetting identity ridge after changing basis changes leverage by72%. Random-polynomial inequality checks and extremizer checks pass. The ridge defines a coefficient inner product, not an asserted probability moment matrix.

For native d=1152, homogeneous quadratic dimension is664128: a dense moment matrix needs about441 billion entries. A proposed restricted diagnostic should freeze a small candidate/error span, use O(np²+p³) Gram construction/solve in its p-dimensional coordinates, and charge feature generation. It certifies only that span. No native coverage certificate or optimizer fit was executed here. The metric is unique at a declared basis/measure; the theorem gives no recovery or uniqueness of internal product atoms.

## BASELINE_COMPARISON

Read [DECOMPOSITION_BASELINES_2026-09-20.md](DECOMPOSITION_BASELINES_2026-09-20.md). Keep same six source outputs, three component outputs, z/h ports, precision, affine corrections and error definition.

- Native factorization and dense decoded source matrices remain fidelity references; supplied upstream states and final background are chargeable.
- Wide graph592 source products/1,342,028 floats versus regular pair banks1152/1,340,940. Source scalar multiplications are1,331,088 versus1,330,560; product-node savings do not give a runtime win.
- Literal graph coefficients:1,327,104 input-reader values,3392 output coefficients,6918 affine values,3456 h-reader values,6 centers and1152 writer values. Common FP32 price is5,368,112 bytes; double precision doubles it. Pair programs add3456 int64 indices versus one shared square-output index. Three outer products are common extra work. Batch temporary state and native input/background generation remain outside this local bill.
- Conventional spectral/HOSVD controls:766-span native lower bound27.81%, feasible dense core36.55%, current-span dense core45.32%, fixed-product optimum59.43%. These are costly relaxations, not matched-cost adopted programs.
- Fixed-basis sparse Tucker top592 pairs has88.31% native error. Optimized basis/core recovery is not disproved.
- Prior HT homogeneous quartic result has different outputs/ports; no equal-interface optimized HT or alternative-tree comparison for this six-read normalized target exists.
- Current wide-program constant, original-channel-pruning and matched-cost random batteries are incomplete. Earlier target-specific nulls cannot fill these missing runs.

Native equal-pair Frobenius error, covariance-root-shaped coefficient error, opened scalar error divided by centered target norm, and fresh intervention error divided by reference effect norm remain separate. Original mixed covariance guard and fresh relative-baseline failures stay failed.

## REDTEAM_POSITIVE

Strongest success: all three frozen candidates passed72 aggregate absolute effect gates each on32 new FineWeb documents and16 code files (fit plus native edit; fresh relative to freeze). Candidate/donor hashes predate capture. Pretraining overlap remains unknown; historical1536/448 states and expanded232-prefix calibration are not fresh.

Executed descriptive all-cohort document audit, now opened evidence: group repeated records by document/selection/family/cohort before forming ratios. There are384 FineWeb plus192 code cells, but only48 distinct document/file contexts. These cells are dependent; raw1548 records are not1548 contexts.

Mixed and isotropic graphs exceed none of the15% natural/hybrid or20% change caps at this document grain. Covariance-only exceeds15 cells across13 FineWeb documents, worst21%. The mixed candidate's worst all-cohort document error is14.86% on FineWeb and6.11% on code. This supports its absolute success and challenges covariance-only aggregate reassurance. It does not replace registered cohort or relative gates.

The first diagnostic mislabeled raw records as document cells. [V1 receipt](direct_tensor_match/REVIEW_MOMENT_CERTIFICATE_20260921_1047.json) is preserved; V2 corrects the grouping and supersedes its tail statistics. Independent grouped-sum validation reproduces every original all-cohort aggregate error. No existing research receipt was modified.

Constant/affine corrections, free-native-port assumptions, unrelated damage and reuse remain unresolved confounds. Source execution has no dense quadratic fallback, but native z,h and final background are still supplied. Current matched-cost constant/random controls and three unrelated-reader preservation tests are missing. The mixed candidate still fails10 comparisons against either strong baseline; its absolute pass is not adoption.

## REDTEAM_NEGATIVE

The strongest failure is relative-baseline transfer plus the covariance guard. Existing independent dense/export replay, explicit bias/RMS checks, baseline reproduction, planted loss/gradient checks and equal1500-step parent/random starts support a real failure of these tested programs. Global convergence and full optimizer/rate coverage remain unproved. Earlier full-rank numerical repairs show why failed implementations cannot reject a decomposition family.

Today's equally bounded-input witness provides a non-implementation explanation for covariance-transfer failure. The gauge control also shows that a penalty must transform with coefficient coordinates. Conversely, correct document grouping rescues the mixed candidate from an overbroad claim of document-level absolute failure. Neither observation erases its registered relative misses. No new fit or native intervention was launched.

## Five properties and handoffs

| Property | Current finding |
|---|---|
| Simple |592 vs1152 source products; nearly equal floats and slightly higher total scalar multiplications; no matched-null simplicity or whole-model speed claim |
| Predicts held-out/OOD |Fresh FW/code absolute gates pass; strong matched-baseline relative gates fail; no token-input-only general OOD result |
| Extracted |Conditional executable with two native vector ports z,h plus downstream background/readout; upstream extraction absent |
| Selective |Continuation dossier has localized removal and writer-control evidence; broader unrelated-reader matched-null battery incomplete |
| Composes/reuses |Arithmetic atom/writer sharing and joint-output tests exist; smallest-piece interaction and random-split specificity not established |

The [continuation dossier](direct_tensor_match/circuits/letter_continuation_group2.md) selects the folded writer/constituents by circuit evidence. Folded algebra suggests grouping stable downstream-defined quadratic functions instead of naming unstable atoms. Weight-folding handoff: preserve the current stronger pair baselines and both metrics; a small frozen coverage diagnostic can accompany the primary worker's overlapping-reader reuse direction. Do not mutate its running sweep dependencies.

Later circuit handoff: preserve regional depth, exact port closure, full QK1×QK2×V interactions and forward response census before a suffix; preregister equal-norm and random-split nulls. A closed Möbius identity is not small interaction relative to the smallest piece. TYPED_FACE_EXTRACTION_V1 remains primary-owned and untouched.

## Organization and efficiency

Read the complete skill/startup/NEXT prompt, user authorities/focus, board protocol/tail,08:44/09:44 hourly reviews, current overview/follow-up, registries, MLP16/17 dossiers, continuation dossier, current plans/executors/receipts and relevant connection/backlog tails. Older paths were resolved against this checkout.

The continuation dossier reaches10:37 and provides current receipt links. The stable0104 overview filename was rewritten at10:42. At inspection, explanation “latest” lines and computation-path registry tails remained older; the concurrent10:50 review already identifies navigation repair. Do not race that repair. Generated CIRCUIT_GRAPH_REGISTRY_V1 inventories49 exported packages/44 manifests; its two four-trait labels do not promote this path or imply measured simplicity. Individual MLP dossiers retain earlier scopes. continuation_source_shared16 and letter_continuation_group2 are related package/dossier aliases, not independent discoveries.

Untracked historical addenda and MIDPOINT_SOURCE_SHARED_DICTIONARY artifacts were preserved without inferring ownership or promotion. Current shared/private metrics, source_interface, compact/global executors and regular-pair compiler already reduce duplicated work. Further shared-code refactoring would touch active dependencies without a demonstrated saving; no additional framework is justified. This addendum provides the missing review-control navigation and preserves the V1 correction separately.

Supervisor runners were RUNNING and both queues empty; runner logs showed terminal dual fit10:09:10–10:25:44 and fresh run10:34:02–10:34:14. Measured fits992.65s, pair baselines8.46s, pair enumeration8.83s, native fresh evaluation10.66s. Shared48 captures for five frozen programs save repeated forwards. Exact review/authoring category times are unavailable, so no invented ceremony ratio. Initial workspace had281 dirty entries; secondary checkout had no tracked dirty changes. Recent commits read through a241a3dea. No commits/pushes, queue/timer changes, agents, goals or GPU work by this review.

Source receipt SHA256: DUAL_FRESH_NATIVE_V1.json =3ebe2ceedb7235564865c3a6cb2585da37a1c62636840fcc4c646372454e8b1f. V2 control SHA256 =8eadcdffdab23953c9716b1ebb245e558c751afc015a3db4441d133d48b94926. The control also records its script hash.

This bounded review ends after the executed CPU consequence and this addendum. The concurrent10:50 review owns the three-hour clock.
