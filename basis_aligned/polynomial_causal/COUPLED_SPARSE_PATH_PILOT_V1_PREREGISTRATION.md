# Matched full-unembedding sparse path pilot

11 September 2026, before native execution. Implements stage1 of the [user proposal](explanations/for_logan/interaction_path_decomposition_proposal_2026-09-11.md), not a completed circuit search.

Target: exact symmetric quadratic numerator of U composed with MLP17, on two formal source ports: incoming residual r and scaled pre-output-projection attention aggregate z'=s*z, where s=||O17||F/sqrt(1152). The source maps are [I,O17/s]. All rr,ra,aa blocks are included; native RMS squared, bias, direct residual and final RMS/tanh remain external. Full50304-row unembedding metric, no corpus, no semantic task labels. This target does not substitute live QK/value producers yet.

Hypothesis: jointly learned source features support a sparser reusable interaction program than fitting each path with unrelated features. Counter-hypothesis: apparent sharing disappears at equal literal cost or the optimization remains unresolved. Output-only invertible folding cannot lower exact CP rank; this tests constrained sparse execution in declared source coordinates, not that impossible claim. Scalar gain normalization is a coordinate convention, not data whitening.

Arms, for each seed11241/11242:

- Joint: two orthonormal1152x16 bases, shared between all incident path blocks. Up to528 unique quadratic edges; keep96.
- Independent: four orthonormal1152x8 bases: rr, mixed-left, mixed-right, aa. Up to136 unique edges; keep96 globally.
- Both:36864 basis floats+110592 physical output-writer floats=147456 fitted floats,192 edge-coordinate indices plus source/block labels. Same96 scalar feature-pair multiplications. Joint32 versus independent32 linear reads. Actual symmetric-feature constants and source gain adapter are retained. Output vectors need not be sparse; this is output-group/edge sparsity, not output-mode sparse Tucker.

Initialize independent banks from native-reader random sketches; initialize joint bank from the union of the corresponding source spans. This matches starting available information, not the initial sparse-core loss. Report both. No output/task fitting.

Optimization: exact optimal full-output coefficients at every basis evaluation; exact top96 edge support at each outer cycle. Four cycles per arm, at most60seconds standard manifoldCG for each fixed support, then support reselection. Preserve final support gaps and gradients; a time/line-search stop is not convergence. Two consecutive unchanged supports plus absolute tangent norm<=1e-7 and capture-relative tangent norm<=1e-4 are required. If any arm misses, continued optimization or a new solver precedes any absent-structure conclusion. This is a priced pilot, not an artificial final optimization budget.

Predictions registered:

A instrument: bound CPU controls held; native source-coordinate/direct read replay relative<=1e-10; original full-U coefficient norm matches existing99245061353.47293 within1e-8 relative; source norm finite positive; native fixed-support tangent finite-difference error<=1e-4; every basis orthogonality error<=1e-9; support selection nondecreasing capture to1e-10 tolerance; all outputs finite.

B convergence: all four arms meet support and both gradient conditions above.

C structural advantage: final joint captured coefficient energy>=1.10 times independent for EACH seed. Score regardless but interpret substantively only if A/B hold; not a behavioral circuit criterion.

D reusable incidence: EACH joint seed retains>=8 cross-port edges and>=4 source features on EACH side incident to both within-source and cross-source selected edges. This is explicit shared computational incidence, not semantic reuse or causal selectivity.

Record per-block retained coefficient energies, capture, support, stationarity, step timings, maximum GPU memory and literal artifacts. Zero body forwards. Max4x240 fit seconds,1500second alarm allowing setup. Current RTX5090/32GB; native FP64, no TF32. Managed lane1 only. Persist compact final banks/writers and scalar progress, not dense path tensors. Follow result-dependent validation: if structure viable, freeze program before FineWeb and intervention checks; if solver misses, red-team optimization before any structural rejection. No change to earlier suffix controls or claims.
