# Weight-only structural methods: current receipt index

Updated 11 September 06:54 UTC. Use this index before opening a structural family
or resuming a checkpoint. The25-hypothesis campaign's initial status column is
historical. Current receipts, queues and source hashes override summaries.
Different capacities, penalties and centered/full metrics are not a leaderboard.
“Converged” below means local convergence under the recorded criterion, never
global recovery or semantic circuit identification.

| Family / campaign IDs | Latest authoritative evidence | What remains unresolved |
|---|---|---|
| Free shared products (1,25) | [Penalized128-product fit](PENALIZED_WEIGHT_PRODUCT_V1_CHUNK_00.json) locally converged; [ALS](WEIGHT_PRODUCT_ALS_V1_RESULT.json) and [GN](WEIGHT_PRODUCT_GN_V1_RESULT.json) did not | Changed penalty versus original objective; global optimum and broad capacity coverage unproven |
| Signed squares (2,23) | [Polished256-square fit](WEIGHT_SQUARE_POLISH_V1_RESULT.json) locally converged | Algebraically related to two-reader products; unequal budgets are not independent evidence |
| Shared input reader and compact partner (3,19) | [Joint rank16partner fit](JOINT_SHARED_READER_RANK16_V1_RESULT.json), converged restarts; [upstream interface](SHARED_READER_UPSTREAM_PORTS_V1_AUDIT.json) | One small component with native background, not a whole-model factorization |
| Sparse multiplication graph in a shared frame (4,6) | [RCG](SPARSE_CORE_RCG_V1_RESULT.json), [independent restart](SPARSE_CORE_RCG_RESTART_V1_RESULT.json), [joint-function audit](SPARSE_CORE_TWO_FUNCTION_SPAN_V1_AUDIT.json) | Locally converged functions differ; orthogonal-frame restriction remains |
| Overlapping multi-output blocks (5,10) | [Original fit](WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_RESULT.json) → [custom manifoldCG](MULTIOUTPUT_MANIFOLD_V1_RESULT.json) → [libraryCG](ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_RESULT.json) → [exact-Hessian trust regions](BLOCK_TRUST_REGION_V1_RESULT.json): all unconverged | [Thin trust-region continuation](BLOCK_TRUST_REGION_THIN_V1_RESULT.json) completed 600.49 fit seconds, still unconverged; [exact writer/core gauge audit](BLOCK_WRITER_CORE_GAUGE_V1_AUDIT.json) found negligible balancing benefit; [exact conditional writer/core updates](CONDITIONAL_BLOCK_SVD_V1_RESULT.json) settled with essentially no gain |
| Independent blocks under nonorthogonal congruence (7) | [Verified full spectrum](NATIVE_CONGRUENCE_MIXED_V1_RESULT.json), [centered spectrum](CENTERED_CONGRUENCE_V1_RESULT.json) converged; structural bars missed | Does not rule out overlapping blocks; do not repeat as an untried method |
| Output-rank optimum and bounds (8) | [Full-U spectrum](FULLU_OUTPUT_FUNCTIONS_V1_AUDIT.json) | Bound is on global output rank under coefficient error, not arbitrary shared arithmetic programs |
| Concentrated output loadings (8,9) | [Varimax](OUTPUT_VARIMAX_V1_RESULT.json), [normalized rotation](OUTPUT_VARIMAX_NORMALIZED_V1_RESULT.json) converged | Rotation within a pre-truncated space cannot recover omitted structure |
| Sparse overlapping token-function dictionary (9,10) | [512-function fit](TOKEN_FUNCTION_DICTIONARY_V1_RESULT.json), locally converged; [fixed-support debias](TOKEN_DICTIONARY_DEBIAS_V1_AUDIT.json) | [Input functions remain dense](TOKEN_DICTIONARY_FUNCTIONS_V1_AUDIT.json); sparse usage does not imply simple multiplication |
| Sparse token usage with product atoms (1,9,10) | [Product dictionary](SPARSE_PRODUCT_DICTIONARY_V1_RESULT.json), jointly unconverged | Exact conditional atom updates are not joint convergence; writer-only repairs do not settle reader optimization |
| Native product dictionary selection (1,9) | [Exact greedy OLS curve](NATIVE_DICTIONARY_OLS_V1_RESULT.json), [audit](NATIVE_DICTIONARY_OLS_V1_AUDIT.json) | Restricted dictionary/greedy support;1024products capture31.58%full coefficients, not a generic decomposition bound |
| Unembedding hierarchy before folding (11) | [Token/hierarchy backward fold](UNEMBEDDING_BACKWARD_VIEWS_V1_RESULT.json) | Fixed16-leaf whole-token means failed; large token-specific remainder required. Higher-order or overlapping hierarchy models remain open |
| Shared input span with all mixed interactions (3,6) | [Native family bounds](SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT.json) | Counts every term touching selected inputs; low coefficient coverage is not semantic irrelevance |
| Joint QK source features across positions (19,20) | [Source frame fit](POSITION_SHARED_QK_SOURCE_V1_RESULT.json), [FineWeb frozen validation](FROZEN_QK_FINEWEB_V1_RESULT.json) | Numerator coverage differs from normalized routing effect; the tested physical effect was tiny |
| MLP16 producer / attention QK–OV sharing (19,20) | [Frozen fold](MLP16_PRODUCER_OVERLAP_V1_RESULT.json), [exact key envelope](MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT.json), [joint fit](COUPLED_PRODUCER_NATIVE_V1_RESULT.json), [convergence repair](COUPLED_PRODUCER_CONVERGENCE_REPAIR_V1_RESULT.json) | Original tradeoff miss preserved; [fixed midpoint](COUPLED_PRODUCER_MIDPOINT_V1_AUDIT.json) was a post-result audit on already inspected positions |
| Simple functions inside the coupled spans (19,20) | [Exact scalar-product spectra](PRODUCER_SCALAR_PRODUCTS_V1_AUDIT.json), [optimized mixtures](SIMPLE_PRODUCT_SHARED_SPAN_V1_RESULT.json), [red-team](SIMPLE_PRODUCT_SHARED_SPAN_V1_REDTEAM.json) | All36mixture starts converge but noheadpasses combined screen; one-product ceilings<=20.57%apply only to these fixed spans |

Existing block geometry is also prior art:
[input/output spectra and overlaps](MULTIOUTPUT_BLOCK_GEOMETRY_V1.json),
[shared leading output patterns](MULTIOUTPUT_BLOCK_OUTPUT_MODES_V1.json), and
[common/centered decomposition](MULTIOUTPUT_MANIFOLD_V1_OUTPUT_SPLIT.json).
Do not describe those properties as new without comparing the latest fitted
function. High similarity between leading output vectors is not identity of
the complete block computations.

Campaign IDs12–18and22involve activation distributions, perturbation losses,
regimes, statistics or behavioral weighting. Some historical runs exist, but
new data-guided discovery is deferred by the user's weights-first instruction.
IDs21(stability),23(symmetry),24(output support) and25(cancellation) are also
cross-cutting checks; they do not each constitute a separate factorization.
FineWeb is the training-corpus validation source; Pile is separately labelled
corpus-shift testing. Never adapt on the supposedly untouched validation set.

The longer block run completed; preserve its limit and gradient failures.
The exact same-function penalty minimum is already effectively attained.
Broader adaptive/heterogeneous structure and independent solver coverage remain
open; no automatic identical chunk is queued. Convergence or capture alone does not establish OOD prediction,
extraction, selective removal or composition/reuse.

New follow-up: [conditional output-capacity spectra](BLOCK_CONDITIONAL_CAPACITY_V1_RESULT.json)
show modest rank8 gains in10/16fixed frames; isolated gains are not additive.
The [23x4 full-quadratic frame protocol](FULL_QUADRATIC_FRAME_V1_PREREGISTRATION.md)
changes input/output allocation at370944coefficients and changes the penalty.
[Native result](FULL_QUADRATIC_FRAME_V1_RESULT.json): both starts locallyconverged,
capture5.9467/5.9385%, gain targetmissed, wholefunctioncosine.9315. [Outputsplit](FULL_QUADRATIC_OUTPUT_SPLIT_V1_AUDIT.json)
gives only3.02/2.95%centered capture. [Input-union ceiling](FULL_QUADRATIC_INPUT_CEILING_V1_AUDIT.json)
is18.76%full forany92-dimensional inside-onlyinputspan. No identical fit queued.
The earlier [CG failure](FULL_QUADRATIC_FRAME_V2_RESTART_AUDIT.json) and [TR repair](FULL_QUADRATIC_FRAME_V3_RESTART_AUDIT.json)
remain recorded; memoizedV3solver used natively.

New family under development: mixed-radix full-rank linear stages around a
bilinear product bank. [Construction control](MIXED_RADIX_BILINEAR_V1_CONTROL.json)
passes execution/gradient/full-mode-rank checks, with276480proposedcoefficients
atnativewidth4608. [Small planted fit](MIXED_RADIX_PLANTED_V1_RESULT.json) recovers
2/3starts; [failed-point curvature audit](MIXED_RADIX_CURVATURE_V1_AUDIT.json)
finds no useful negative-curvature escape. [Native V2 jointfit protocol](STRUCTURED_BILINEAR_NATIVE_V2_PREREGISTRATION.md)
completed: both starts time-limited, capture1.4968/0.9388%, unconverged.
[Saved-state continuation](STRUCTURED_BILINEAR_CONTINUE_V1_PREREGISTRATION.md)
is managed-live from06:47:48, exact first-start replay passed. V1stoppedbeforejointfit onFDtruncation;
[step-sizeaudit](STRUCTURED_GRADIENT_STEP_AUDIT_V1_RESULT.json) verifiesrepair.
No converged native verdict or globalrecoverabilityclaim; fixedwiring/initialization
remain limitations. Inspect currentresult/runner before any resubmission.

[Internal stage balancing](STRUCTURED_STAGE_BALANCE_V1_AUDIT.json) is exact,
reducing squared stage norms5.55/5.66x; [actual gradient audit](STRUCTURED_BALANCED_GRADIENT_V1_AUDIT.json)
finds only1.62xrelative-stationarity and1.02xmax-gradient reductions. Both2xtargets
missed. Not a convergence repair or measured speedup; unchanged continuation
remains live. Balanced copies are separate, no optimizer-history reuse.
