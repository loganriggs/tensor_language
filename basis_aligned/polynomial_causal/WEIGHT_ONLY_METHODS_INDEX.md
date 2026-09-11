# Weight-only structural methods: current receipt index

Updated11September05:17UTC. Use this index before opening a structural family
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
| Overlapping multi-output blocks (5,10) | [Original fit](WEIGHT_STRUCTURAL_BASELINE_V1_multioutput_block_RESULT.json) → [custom manifoldCG](MULTIOUTPUT_MANIFOLD_V1_RESULT.json) → [libraryCG](ORTHOGONAL_MULTIOUTPUT_PYMANOPT_V1_RESULT.json) → [exact-Hessian trust regions](BLOCK_TRUST_REGION_V1_RESULT.json): all unconverged | **Current live continuation:** [thin-contraction protocol](BLOCK_TRUST_REGION_THIN_V1_PREREGISTRATION.md), wrapper `ops/run_block_trust_region_thin_v1.py`; inspect its result/runner before any further submission |
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

Next decision after the live block run: inspect all stopping criteria, preserve
any limit failure, then compare block functions/stability using the existing
geometry tools. Convergence or capture alone does not establish OOD prediction,
extraction, selective removal or composition/reuse.
