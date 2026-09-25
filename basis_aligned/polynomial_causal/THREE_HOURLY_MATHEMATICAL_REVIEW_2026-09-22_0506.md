# Mathematical, organization, and efficiency review

Actual UTC: 2026-09-22 05:06:30. Previous substantive review: [THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-22_0159.md](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-22_0159.md). ACTIVE_TRACK: WEIGHT_FOLDING under the user focus through2026-09-22 15:10 UTC. Next review deadline: 2026-09-22 08:06:30 UTC.

The current direct-weight target is the MLP16→MLP17 pure quartic contribution to 16 fixed output coordinates. New evidence separates an optimization problem from a representation restriction: scale-aware L-BFGS rescues planted fits, but the native residual does not exhibit the toy's rank-four input geometry. The executed consequence below turns Gaussian derivative tails into a two-sided conditional-approximation bound. No native causal fidelity or semantic circuit is established.

## Current native mathematical objects

Zero-based layers: 18 blocks, residual dimension d=1152, MLP width h=4608, nine attention heads of width128, vocabulary V=50304. L_l,R_l∈R^{h×d}, D_l∈R^{d×h}, b_l∈R^d. The MLP is D_l[(L_l x)⊙(R_l x)]+b_l on its actual RMS-normalized input. Residual re-entry coefficients are learned. Native final output is 30 tanh(U RMS(r)/30), U∈R^{V×d}; unembedding is untied. RMS epsilon is the actual float32 epsilon, not zero.

For fixed projected readers w_v, v=1,…,16, define C_vk=w_v^T D17[:,k], B=λ17,0 D16, and

F_v(x)=Σ_k C_vk [Σ_a L17_ka Σ_j B_aj (Σ_i L16_ji x_i)(Σ_i R16_ji x_i)]
                     [Σ_b R17_kb Σ_t B_bt (Σ_i L16_ti x_i)(Σ_i R16_ti x_i)].

Thus the graph is x→two width4608 linear maps→4608 products→width1152 write→two width4608 maps→4608 products→16 outputs. Its coefficient tensor T_{v i1 i2 i3 i4} has shape16×1152^4. Repeated x slots mean only Sym(T) specifies the polynomial; an unsymmetric representative's tensor norm is a different object. Degree is four before normalization. The two downstream branches share B,L16,R16 and the same x; they are not independent draws. CP atom scales/permutations, exchanging input slots, and internal tensor-factor basis transformations preserving adjacent contractions are gauges. Arbitrary rotation of native elementwise channels is not an invariance without transforming the core. Semantic identities do not follow from a factor basis.

Allowed fitting input is x=μ+Sz with z standard Gaussian and frozen calibration-derived μ,S. The polynomial then has degrees0–4 in z. Isotropic coefficients, isotropic Gaussian values, covariance-Gaussian values, empirical text values, and finite logit changes are separate norms. For the current residual screen, R(z)=F(μ+Sz)−Fhat(μ+Sz) restricted to outputs4–15, and ||R||²_W=E[R^T W R], with the existing capped inverse-energy diagonal W. Singular covariance restricts conclusions to its support; no claim extends to missing input directions.

Residual sources matter. Write r=r0+Σ_i a_i+Σ_j m_j, with actual transport coefficients absorbed into each source s_u. Then

D[(Lr)⊙(Rr)]=Σ_{u,v}D[(Ls_u)⊙(Rs_v)].

For r0+a+m this has nine ordered terms: three self terms and both orders of r0/a, r0/m and a/m. The present quartic keeps only m16×m16, using m16's bias-free quadratic part. It omits earlier-residual self, attention self, both ordered attention/MLP crosses, bias contributions and other source combinations. The broader midpoint program retains a different collection of MLP-dependent self/cross terms and remains a separate target.

For attention at target t/source s, score branch q_b(t,s)=Σ_{u,v}s_u(t)^T A_b(t,s)s_v(s), b=1,2; A includes the native Q/K maps, position rotations and specified normalization. A head writes Σ_s O[(q1(t,s)q2(t,s))V r_s], with causal mask/scales retained. Expanding both scores and the value source gives the full QK1×QK2×V interactions. With fixed normalization denominators this is degree five in residual sources; if the score maps are tied their equality is retained, not independently refit. Actual RMS/QK normalization makes the full map nonpolynomial. Native rounded rotary operations and later softcap remain explicit interfaces. No separate-QK analysis licenses dropping their joint cross term.

The current circuit-side object remains the regional source→routing→writer→MLP response graph (not the quartic's arbitrary output basis): earlier attention/MLP sources feed head9.8's two routing branches and values, its write propagates through coupled residual/MLP8 effects and later readers including head17.2. Modules, token positions, recipient/donor context, RMS denominators and external suffix are declared ports. The head17.2 dossier's opened response localization is not a head-removal result. Preserve the full interaction face; a closed Möbius identity does not demonstrate small interactions relative to the smallest singleton or specificity against random splits. TYPED_FACE_EXTRACTION_V1 remains primary-owned and untouched.

### Literal price and five-property score

A native MLP has15,925,248 matrix entries plus1152 biases,4608 variable products. A direct factor executor for this bias-free16-output quartic can store5dh+16h=26,615,808 entries, absorbing λ into B; a dense1152×16 output writer adds18,432. It uses9216 variable products and corresponding dense linear contractions; this is an explicit implementation price, not a minimum. Live buffers have widths1152/4608/16; their peak depends on scheduling. The normalized state producer, normalization/background and full-vocabulary suffix are additional costs for an intervention, not free native ports.

The CP512 parent stores2,385,920 floating coefficients including the common writer and uses1536 products. The lean conditional program stores848,912 with1536 products; at float32 these are9,543,680 and3,395,648 coefficient bytes respectively, excluding serialization metadata and runtime buffers. Exact reader-reuse DAGs reduce1536/1535 fixed-tree products to1527/1525 but inherit parent errors. Logical product savings are not GPU speed measurements.

| Property | Current quartic/conditional-program score |
|---|---|
| Simple | Measured local coefficient savings and CPU timing; full native-port/suffix price still required. Not a complete-program simplicity pass. |
| Predicts held-out/OOD | Some earlier document-held value evidence, but current comparisons reuse opened panels; native small-coordinate responses and shifted-domain strata fail. No general OOD pass. |
| Extracted | Local polynomial executable from declared normalized1152-vector input; token-to-output closure absent. Conditional interface only. |
| Selective | Coordinate1 removal fidelity fails5/16 cells for both conditional variants; semantic target plus at least three unrelated readers and matched nulls remain unestablished. |
| Composes/reuses | Algebraic sharing/replay exists; joint native replacements, small interactions relative to weak pieces and matched random-split specificity are not established. |

The generated CIRCUIT_GRAPH_REGISTRY_V1 inventory reports49 packages,44 manifests,33 declared boundaries and2 four-trait-verified packages. This is a scoped historical inventory, not five-property adoption for this new path; missing fields are unknown. No registry promotion is warranted.

## LITERATURE_SEARCH

Actual searches this turn:
1. `Gaussian polynomial conditional expectation gradient covariance reverse Poincare degree Hermite active subspaces Zahm`
2. `Gaussian polynomials Hermite expansion derivative norm degree reverse Poincare inequality`
3. `Grasedyck hierarchical singular value decomposition hierarchical tensors 2010 approximation error`

Opened primary sources and precise uses:
- [Zahm, Constantine, Prieur, Marzouk](https://arxiv.org/pdf/1801.07922), also its abstract: Gaussian vector-valued ridge approximation using a subspace Poincaré upper bound. Map their codomain metric to W and input Gaussian to whitened z; use H=E[J_R^T W J_R]. The conditional expectation is the optimal square-integrable function of retained coordinates. Population H costs derivative moments plus a d×d eigensolve, generally O(d³); Monte Carlo estimation is not an exact population certificate. An eigenspace does not uniquely identify semantic directions when eigenvalues tie. This informs the new bound below.
- [Diakonikolas, Kane, Kontonis, Liu, Zarifis](https://cseweb.ucsd.edu/~dakane/SuperNonsingularDecompositions.pdf), orthogonal-polynomial section and Fact3.13: degree-k Gaussian polynomials have gradient energy at most k times value energy. Use only this polynomial inequality, extended through W^{1/2}; their robust Boolean threshold learning algorithm does not solve our vector-valued causal problem. This supplies the lower side of the derived sandwich.
- [O'Leary and Rust](https://www.cs.umd.edu/~oleary/software/varpro/varpro.pdf), variable projection and rank discussion: current linear readout can be eliminated inside nonlinear factor fitting. Weighted design/full-rank or minimum-norm conventions must be explicit; ridge is an augmented least-squares problem. Linear uniqueness does not imply global factor recovery. This supports the optimizer diagnosis, without prescribing a new queued run.
- [Grasedyck, hierarchical SVD](https://doi.org/10.1137/090764189), publisher page opened. Its fixed-tree tensor approximation concerns independent coefficient slots and tensor norms, not repeated-input Gaussian polynomial equivalence or arbitrary shared arithmetic DAGs. The MPI preprint PDF fetch failed with `400 Unsupported content-type: application/octet-stream`; no full-PDF reading is claimed. HT remains a required baseline, not a theorem settling native causal fidelity.

Search changes the interpretation: native gradient capture can constrain Gaussian subspace approximation under a degree bound, rather than remaining wholly disconnected from function error. It does not redirect the current native jobs, justify a small rank, or establish a text-distribution bound.

## Executed consequence: degree-four conditional-error sandwich

Let P be any orthogonal projector in whitened coordinates, g(Pz)=E[R(z)|Pz], and E_P=E||R−g||²_W. Set D_P=tr((I−P)H). Rotate to retained/discarded coordinates and expand in orthonormal multivariate Hermites. Conditional averaging deletes exactly coefficients with positive discarded-coordinate degree k_perp. Therefore

E_P=Σ_{k_perp>0}||c_α||²_W,
D_P=Σ_{k_perp>0}k_perp||c_α||²_W,
D_P/4 ≤ E_P ≤ D_P.

This is our derivation from the cited Gaussian facts, not a new literature theorem. Degree≤4 is essential. It includes affine Gaussian shifts and positive semidefinite W. It fails as a direct application to RMS/softcap-composed responses, non-Gaussian text moments, or arbitrary nonlinear feature bottlenecks. For exact population H, minimizing over rank-r P yields a lower bound Σ_{j>r}λ_j(H)/4 on any rank-r linear-input ridge approximation, not on all circuits. Eigenspaces minimize the derivative surrogate; they need not minimize the actual conditional error.

For centered residual variation V_R=E||R−ER||²_W, Gaussian Poincaré gives V_R≤tr H. Thus for population capture c=tr(PH)/tr H,

sqrt(E_P/V_R) ≥ sqrt((1−c)/4).

This normalization is residual variation, not full native output energy; a constant offset cannot evade it. A10% relative residual-variation approximation requires at least96% population derivative capture, a necessary and not sufficient condition.

Executed [CPU control](REVIEW_HERMITE_BOUND_2026-09-22_0502.py) with /venv/main/bin/python; [receipt](REVIEW_HERMITE_BOUND_2026-09-22_0502.json). Four deterministic cases cover a random two-output quartic, sharp linear endpoint, sharp quartic endpoint, and exactly closed retained port. Independent five-point tensor Gaussian quadrature (125 distinct synthetic nodes, exact through degree9 per coordinate) matches Hermite energies; all checks pass. Largest relative discrepancy2.14e-15. This is algebraic control evidence, not text evaluation.

Using existing1024-fit/1024-check native capture estimates only as plug-ins gives rank64 floors35.99%/35.46% and rank512 floors15.04%/14.68%. These are **not certified native lower bounds**: H and captures are sampled, and no concentration interval or exact population contraction was executed. Training eigenvalue tails cannot silently become population-optimal bounds. The control receipt preserves the source SHA256. A future certified decision needs an exact discarded-gradient contraction or defensible uncertainty bounds; no new sampling campaign was launched here.

## BASELINE_COMPARISON

Apply [DECOMPOSITION_BASELINES_2026-09-20.md](DECOMPOSITION_BASELINES_2026-09-20.md). Equal output, ports, precision and norm are prerequisites to numerical ranking.

| Baseline | Evidence and comparison limit |
|---|---|
| Exact native factors | Same pure-quartic16-output target provides reference; actual normalized finite intervention is a separate reference. Algebra is float64; native endpoint replay uses recorded native precision. |
| Constant/low-degree | Prior native spherical truncation fails138% value error; mean-only98%. The single-layer quadratic's93% isotropic mean fraction belongs to a different target. New bound explicitly centers residual variance. |
| Conventional spectral/native-channel | Selected-output quadratic rotated block baseline47.53% centered error at512 products,12.92% at4608; channel pruning68.56% at512. These are single-layer quadratic results, not equal-target quartic winners. |
| Tucker/HOSVD | Joint input-mode/full-rank numerical repairs and restricted spectra exist; no matched-price, same16-output full normalized-path Tucker result is supplied by these receipts. Missing stays missing. |
| HT/fixed and alternative trees | Earlier restricted homogeneous branch rank8 had9.35% coefficient error at656 values, versus exact canonical280; rank2 cost116 and failed36%. It is not the current1152-dimensional16-output native residual comparison. Current matched-interface alternative-tree run remains missing. |
| CP/shared/conditional DAG | Local prices above are measured or explicitly derived; finite-removal failures prevent fidelity adoption. Exact reassociation preserves its fitted parent only. |
| Matched-cost random and gauge controls | Prior scopes contain controls, but a full matched-total-cost random/Tucker/HT comparison for the current residual remains missing. No general architecture superiority claim. |

Keep isotropic and covariance-informed losses side by side. Data-derived μ,S,W and prior output readers are priced/provenanced assumptions; covariance does not supply empirical eighth moments. Error numerators/reference energies and per-output scores remain in primary receipts.

## REDTEAM_POSITIVE

Strongest new constructive success: weight-derived toy subspace recovery10/10 below5%, and scaled L-BFGS8/10. The former receives teacher weight access and intentionally rank-four targets; charge4608 projection coefficients in the4680-value toy program. Its correct matched ambient control is5/10 on densely rotated targets, not the earlier axis-aligned2/10. Initial student functions differ, so it is a pipeline comparison. Native rank64 captures only48–50% on independent checking probes, rejecting transfer of toy compactness. The bound control explicitly tests a constant-insensitive variation metric and an exact closed-port positive.

L-BFGS8/10 uses281–311 evaluations, while only7/10 pass at the first251; do not compare8/10 to Adam's250 updates as equal cost. Profiled output ridge and history memory remain charged. Neither success tests semantic selectivity, unrelated damage, token-only extraction or composition. No dense fallback or free projection is credited as compression. Existing opened-panel numerator fits cannot be promoted through leakage-prone repeated selection.

The conditional program is the strongest recent economical native candidate, but both full and lean variants fail5/16 absolute removal cells; retention fails4/16 and5/16. There are32 distinct document prefixes (16/domain), not independent contexts per seed/strength/stratum. The16 cells are2seeds×2domains×2strengths×2strata for one coordinate. Dependence within documents and reused opened panels preclude fresh-OOD claims.

## REDTEAM_NEGATIVE

Strongest new optimizer failure: wide Adam2/10 and Muon0/10 under the earlier fixed250-step protocol. Positive planted exact witnesses, independent contractions and finite-gradient checks establish available capacity. The later L-BFGS stopping audit identifies tiny directional derivatives tripping an absolute tolerance despite nonconvergence. Fixed initial-loss scaling leaves the minimizer unchanged and removes immediate stops;8/10 recovery rescues much of the failure, while two failures remain. Thus failure is not a theorem against quartic decomposition or Muon generally. Gauge/initial scale, ambient orientation, precision, ridge and actual convergence budgets remain scientific variables.

Strongest native geometric failure: small rank captures little derivative energy. Increasing probes from128 to1024 reduces train/check gaps to about2points at rank64 without rescuing90% capture. Native Euler and finite-difference controls pass. Our exact closed-port example verifies that the new bound recognizes a truly sufficient subspace; the sharp examples prevent an erroneous degree-free equality. No conclusion excludes sparse nonlinear representations spread over many linear directions.

The current CP512 finite-panel oracle is a stronger restricted span failure: independent QR/SVD agree, all four designs have rank512, yet outputs4–15 retain26–49% value and26–46% response errors. It uses evaluation labels and separate optimal readouts, so it is a capacity diagnosis for these frozen dictionaries, not an exported predictor or universal rank bound.

## Organization, efficiency and actionable handoffs

Read the full research skill, current startup/prompt, user authorities, newest board tail, hourly02:17/03:17/04:18 receipts, graph/circuit/path registries, MLP index and MLP16/17/head17.2 dossiers, current decomposition README, relevant preregistrations and primary JSON summaries. Historical absolute /workspace/tensor_language paths resolve here. Startup's workstation systemd text and September13 LATEST/next-action paragraphs are stale; current Supervisor/commits and the explicit focus override them.

Reconciliation: aliases MLP16/layer16 and head17.2/L17H2 map to existing module sections and PATH-SET2-001. The current pure-quartic path is not identical to that regional interaction route or the historical midpoint program. The computation-path registry's04:18 entry correctly links completed removal failure and CP512 oracle; the MLP index's01:59 paragraph still says finite effects queued. The explanation index leads with01:59 and contains older competing “latest” labels. Use the02:41 removal dossier and04:18 path entry as authoritative corrections. No newly identified circuit is missing promotion; new optimizer/subspace controls are method evidence reachable through their interpretation/preregistration files. Their absence from a semantic circuit manifest is appropriate, not a discovery orphan.

No shared-code refactor was performed: frozen queued helpers are hash-bound, and the small audit already reuses analytic native/CP Jacobians and cached inputs. Consolidating the numerous toy runners now would require revalidation and risk the primary worker's comparisons; its expected cost exceeds this bounded review's saving. The worthwhile later consolidation is one declarative optimizer configuration (initial scale, ridge, tolerance, evaluation budget, seed) over the existing shared objective, after the native comparison is stable. Navigation correction is supplied here rather than rewriting concurrent dossiers or historical primary receipts. Existing data capture consumers already avoid a duplicate GPU pass.

Actual recent commits: df8716979 at04:43(subspace toys),9e75185ca at04:47(native screen),43f53c291 at04:49(larger probes),1d3de2f0f at04:58(L-BFGS). Reported CPU kernels: toy subspace12.75s plus ambient14.21s; larger native probe audit25.47s; L-BFGS14.85s plus18.88s. Serial authoring/interpretation exceeds many kernels; no measured ceremony percentage exists. Repeated near-identical narratives and obsolete handoffs cost attention. Stop fixed-bank/radial microvariants; preserve one evidence chain instead of counting every diagnostic as independent progress.

Supervisor bqrunner and bqrunner2 are RUNNING, PIDs8995/8996. Runner log shows Pythia finished04:49:37 and run_hf_all_v763 started then. Queue has two intervening jobs before native local-residual/geometry jobs; their scripts match hashes8952631c… and37674108… exactly. Queue wait is not computation time and is not evidence of an idle GPU. No queue/timer/service changes were made; no agents or GPU jobs launched. Theseus-bench tracked status was clean; live tensor_language untracked artifacts and primary work were preserved.

Circuit→folding handoff: coordinate1 finite-removal failures select a distinct normalized readout/response target; the queued outputs4–15 learner cannot repair it. Preserve the prepared stage-geometry/all-output consumers and use forward response census before choosing a suffix. The regional evidence continues to nominate source interactions and exact port closure once the weights-first focus ends; do not initiate unrelated screens during it.

Folding→circuit handoff: candidate shared readers/products may propose grouping across MLP16/17 or splitting weak from dominant output directions, but their actual native contributions need full source crosses, frozen-background accounting and preregistered equal-strength/equal-cost random nulls. Test joint versus singles relative to the weakest piece, plus three unrelated readers on genuinely fresh distinct context cells. Broad linear gradient geometry alone does not veto such nonlinear grouping.

Next weights-track decision: interpret already-queued new-feature learning and coordinate1 geometry independently; a separately registered optimizer comparison is justified by scale-aware toy recovery, with equal evaluation/time accounting. No further fixed-readout sweep is justified by the current oracle. For subspace proposals, require population-tail evidence sufficient for the desired residual-variation error, then actual approximation and finite-response tests. This review ends after its executed CPU consequence and receipt.

## Receipt provenance and limitations

SHA256 prefixes: native large subspace d02f2ad75cb11797; removal comparison2dbdabffdaacc19b; scaled L-BFGS08eacd3e482cbe2f; planted subspacea00f7f354f502100; current oracle2051f614f9b8cd3b. Exact control hashes and full source hash are in the new JSON. No original receipt was modified. No native text context was newly evaluated by this review;125 Gaussian quadrature nodes are synthetic controls, not125 behavioral contexts. No theorem or replay here proves native causal fidelity. Quantization is excluded.
