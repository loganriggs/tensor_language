# Concurrent-review addendum — 2026-09-21 22:58 UTC

The [22:53 substantive review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_2253.md) appeared after this session's initial check (latest then 19:53). Rechecked at 22:58:38 UTC immediately before publication. This addendum preserves a distinct executed CPU consequence; it does **not** reset the clock. Next deadline remains **2026-09-22 01:53 UTC**. The direct weight-decomposition focus remains authoritative through September 22 15:10 UTC. No agents, GPU jobs, queue/timer changes, commits or primary-receipt edits were made.

The new result is a narrow negative rescue: the absolute ridge used in the three-law CP readout comparison cannot substantially change the optimal fixed-feature function **in its own Gaussian norm**. This does not explain or bound error on text or in a different law. Separately, a planted gauge counterexample demonstrates why any future coordinate change must transport the regularizer.

## Executed mathematical consequence

For frozen quartic features phi(x) in R^512, define G=E[phi phi^T], X=E[F phi^T], and a 16-by-512 writer C. Minimizing E||F-C phi||²+lambda||C||F² gives C_lambda=X(G+lambda I)^-1. If G is positive definite, C_0=X G^-1. Diagonalizing G gives the exact uniform inequality

$$
\frac{\|(C_\lambda-C_0)\phi\|_{L^2(P)}}{\|C_0\phi\|_{L^2(P)}}
\leq \frac{\lambda}{\lambda_{\min}(G)+\lambda}.
$$

The denominator is the unregularized projection function, **not the teacher**, and this is not a relative teacher-error certificate. The proof applies modewise: ridge multiplies each fitted Gram eigenmode by g/(g+lambda), and the function norm weights that mode by g. Multiple outputs add their squared norms. There is no optimization assumption beyond fixed features and positive-definite G.

Executed [CPU control](direct_tensor_match/review_ridge_geometry_20260921_2257.py) with `/venv/main/bin/python`; [receipt](direct_tensor_match/REVIEW_RIDGE_GEOMETRY_20260921_2257.json), 1.42 seconds, two CPU threads, float64. It reads immutable saved CP400 factors and calibration statistics, reusing the existing exact noncentral Gram helper. No teacher checkpoint or activation panel is loaded.

| Gaussian law | Smallest Gram eigenvalues, starts 1001/1002 | Function shrinkage upper bounds |
|---|---:|---:|
| Centered covariance | .054876 / .059052 | 1.8223e-5 / 1.6934e-5 |
| Shifted covariance | .112543 / .113857 | 8.8854e-6 / 8.7829e-6 |
| Zero-mean second moment | .148808 / .166971 | 6.7200e-6 / 5.9890e-6 |

All six Grams are numerically positive definite. These are float64 spectral diagnostics, not interval-certified eigenvalue bounds. At lambda=1e-6, removing ridge is therefore not a plausible large rescue **within these reference laws**. Extrapolation to coefficient/text error remains unbounded by this calculation. Original factors are normalized in physical coordinates, which fixes their scale convention; different laws intentionally change feature energy.

Under a diagonal feature change phi'=A phi, G'=A G A^T, X'=X A^T and C'=C A^-1. Transporting the original penalty gives lambda tr(C' A A^T C'^T). Resetting it to lambda||C'||² changes the optimization problem. A two-atom planted quartic with a shifted Gaussian verifies this directly: reset ridge changes the fitted function by **68.8%**, whereas transported ridge replays within 2.2e-16. Independent five-point-per-axis Gauss–Hermite quadrature checks the eighth-degree Gram to 2.4e-15; an unregularized full-capacity solve recovers the planted coefficients to 9.0e-16. These are algebraic controls, not native causal evidence. The scale principle was already known in the 10:54 addendum; the genuinely new result is its quantitative application to all six current native Grams, ruling out this particular rescue.

## LITERATURE_SEARCH

Actual queries issued in this session:

1. `Isserlis 1918 formula product moments normal frequency distribution pdf`
2. `Gaussian polynomial covariance norm equivalence Hermite polynomial degree tensor decomposition Wick theorem`
3. `tensor decomposition CP scaling indeterminacy ridge regularization variable projection Kolda Bader 2009`
4. `"2605.15183" tensor decomposition`

Opened and read:

- [Nissen Gonzalez et al., When Are Two Networks the Same?, §§2.2–2.3](https://arxiv.org/html/2605.15183v1): full symmetrization identifies the repeated-input polynomial; Gaussian matching uses a lifted moment metric, and the cited efficient recursion requires compatible tree structure. Here n=4, outputs=16, and the input metric uses eighth moments. Exact similarity up to positive scale is weaker than our amplitude-sensitive error and does not identify internal edits. This supports continuing the coefficient/Gaussian distinction, not claiming arbitrary-DAG recursion or causal fidelity.
- [McCullagh, Tensor Methods in Statistics, §3.9.1, p.85](https://www.stat.uchicago.edu/~pmcc/tensorbook/DoverEdition.pdf): Gaussian moments are sums of covariance pairings. For our eight affine factors, expansion adds singleton means; the existing helper sums 764 partial matchings. This exactly constructs G for a declared Gaussian law. It assumes Gaussian moments, not merely agreement of first and second moments with text. Computing factor dot products costs O(r²d), the fixed-degree pairing sum O(764r²), and the spectral audit O(r³), with r=512,d=1152. It grants an exact moment algorithm, no feature uniqueness or nonlinear recovery theorem.
- [Kolda and Bader, Tensor Decompositions and Applications, §3.2](https://www.cs.cornell.edu/courses/cs6241/2019sp/readings/Kolda-Bader-2008-survey.pdf): CP scaling/permutation freedom maps to rescaling individual linear factors and compensating the writer. It motivates the transported-penalty control above. General CP uniqueness statements cannot simply be applied to the symmetrized repeated-input quotient or assumed semantic components.

Access failures: opening the [Isserlis original PDF](https://zenodo.org/records/1431593/files/article.pdf), its Zenodo record, and [Mukherjee–Zhu PDF](https://dept.stat.lsa.umich.edu/~jizhu/pubs/Mukherjee-SADM11.pdf) returned internal errors. The Oxford Isserlis page exposed a minimal publisher page, not the full paper. No claim of reading those full texts. The authored McCullagh monograph supplied the checked moment formula. Search completed; the resulting plan change is to demote ridge-removal as a native-law rescue while retaining gauge-aware regularization for future coordinate changes.

## Native object and circuit boundary clarification

For d=1152,h=4608, let L_l,R_l be h-by-d, D_l be d-by-h and b_l in R^d. A native MLP numerator is D_l[(L_l x)⊙(R_l x)]+b_l, with its actual normalized input x and residual scale explicit. Let D16* include the native carry scale into layer 17 and C be the 16-by-h projected layer-17 writer. The active folded target is

$$
u_a=(L_{16}x)_a(R_{16}x)_a,\quad
v_i=\sum_a D^*_{16,ia}u_a,\quad
F_g(x)=\sum_b C_{gb}(L_{17}v)_b(R_{17}v)_b
       =H_{gijkl}x_ix_jx_kx_l.
$$

Indices i,j,k,l span d, a,b span h, g spans16. H is symmetrized over all four input indices; x is tied across those slots. The graph is two linear-reader/product/writer stages. The CP student uses four independent 512-by-d reader matrices, ties their **inputs**, multiplies four readings per atom, and writes through a 16-by-512 matrix plus fixed 1152-by-16 output writer. Input-slot interchange, channel permutation, reciprocal factor scaling and redundant feature representations are gauges; general rotations do not commute with native norms or positional attention maps.

For r=r0+sum_i a_i+sum_j m_j, every MLP numerator expands as sum_st D[(L r_s)⊙(R r_t)]. This includes attention/attention and MLP/MLP self and cross terms plus both ordered attention/MLP terms. Likewise QK_k(t,s)=sum_ab r_{t,a}^T M^k_ts r_{s,b}. A full attention write multiplies QK1, QK2 and V, so its numerator has five residual-source slots and degree five before normalization. Tied QK factors, when present, must be carried into both slots rather than counted as independent parameters.

The current quartic target retains only the bias-free MLP16-produced write in **both** MLP17 slots; it excludes carry/producer cross terms, attention17, biases and other output directions. Restoring a model-level path requires all residual backgrounds, RMS denominators, attention masks/position operations, final normalization and softcap. This is a degree-four branch, not a globally polynomial normalized transformer.

The standing regional circuit candidate is the earlier attention8.2→head9.8 even-key producer route and its regional downstream effects. Both QKs and V remain part of the attention object. The regional MLP16→17 term had the wrong sign as an account of its recursive edit; the recorded forward census instead nominates direct residual transport and attention17. Thus circuit evidence limits the *interpretation* of this quartic target. Current algebra suggests testing shared root-output groupings and same-token contrasts, but cannot promote them to selective components. During the focus window these are handoffs, not new data-based screens.

## BASELINE_COMPARISON

Apply [the baseline authority](DECOMPOSITION_BASELINES_2026-09-20.md) at equal x input, 16 projected outputs, FP32 execution and identical error law. Symmetric coefficient Frobenius error, Gaussian L2 error, opened-state value error, finite responses and sensitivity-weighted error are separate columns, never interchangeable.

| Program | Products | Stored scalar coefficients | Comparison status |
|---|---:|---:|---|
| Original projected native two-MLP branch | 9,216 | 26,634,240 | Exact numerator reference; backgrounds/norms excluded equally |
| CP512 | 1,536 | 2,385,920 | Includes common writer; 9,543,680 bytes at FP32; no sparse indices |
| Shared bank144×4 with512 pairs | 1,088 | 1,353,728 plus1,024 indices | Seeded support, not discovered sparsity |
| Old paired-root graph | 384 | 322,048 plus768 indices | Cheaper but fidelity failures preserved |

CP requires 2,359,296 scalar reader multiplications, 8,192 root-writer multiplications and 18,432 output-writer multiplications per input, beyond its 1,536 nonlinear products. A straightforward execution keeps 2,048 projected readings, 512 atoms and16 root values before the1152-vector output; streamed implementations can reduce this working state. Full vocabulary lifting and native port generators are additional costs at a logits interface. One native input vector is a conditional port, not token-only extraction.

The native factor, conventional spectral, joint Tucker and fixed-tree HT results in the authority concern several different interfaces. They cannot be copied into the current 16-output error table as equal-task wins. A fully matched current spectral/Tucker/alternative-tree HT frontier, equal-cost random support comparison and common-error native-runtime comparison remain missing. The paired CP seeds are restarts, not random-component nulls. Do not infer superiority from lower storage or lower error in a different law.

## REDTEAM_POSITIVE

Strongest new success: shifted-Gaussian readout gives9.6–10.3% opened text value error, using calibration-only moments. Against it, [component transfer](direct_tensor_match/GAUSSIAN_CP_COMPONENT_TRANSFER_V1.json) gives13.1–15.5% same-token response error and20.4–21.8% root1 sensitivity error. The panel contains2048states but the response comparison has **20 distinct unordered pairs**, not30independent directed pairs. Distinct document/template cells for all2048states are not established here. This is neither fresh OOD nor an actual finite removal.

The new ridge audit excludes a large own-law shrinkage artifact but does not test leakage. Remaining checks include frozen calibration/evaluation provenance, constant/current-token baselines on exactly these outputs, unrelated-reader damage, physical no-dense-fallback execution and matched total capacity. The exported homogeneous quartic contains no new explicit constant merely because its fitting law is shifted; nevertheless large mean energy can dominate its value score. Native input/norm/background access remains charged. Sharing512atoms across16outputs is arithmetic reuse, not demonstrated behavioral composition.

Five-property score for current CP candidate: **simple—measured branch savings only**; **predicts OOD—not established**; **extracted—conditional polynomial artifact, model-level independent extraction unestablished**; **selective—not established**; **composes—not established**. Regional evidence remains narrower and is not transferred to CP. Closed Möbius identities would not establish small interactions relative to the smallest piece or random-split specificity.

## REDTEAM_NEGATIVE

Strongest failure: global coefficient/Gaussian fidelity remains poor after larger CP/hierarchy fits. The new positive-definite Gram audit and planted full-capacity recovery reject a large ridge-induced failure of the current three-law fixed-feature solve. Existing independent dense/quadrature/gradient controls support correct axes, means and precision for this instrument. They do not prove that the nonlinear factors reached a global optimum.

The five-family rank-six planted sweep records Muon10/10 versus Adam8/10 below1% at the tested rate; this rescues expressivity from short-budget optimizer failures, not universal optimizer superiority. Full-rank native recovery and a converged global optimum for learned factor directions remain missing. Native CP factor/gauge changes must retain the physical metric and penalty. Preserve all original failed gates; no native failure is relabeled as absence of decomposable structure.

## Organization, efficiency and handoffs

Read the complete local skill, startup and NEXT prompt; user authorities/focus; current board protocol/tail; recent commits; both dirty statuses; hourly20:13/21:13/22:14 receipts; circuit/path registries; graph registry; MLP16/17 dossiers and module/explanation indexes; current plans and primary receipts. Old absolute references were resolved to this checkout. Startup's systemd and September13 next-action text are historical: observed Supervisor bqrunner/bqrunner2 were RUNNING. No service or queue mutation was needed. The last inspected runner entry was three-law readout completion at22:50:01.

Reconciliation: the computation-path registry still ends at September20 20:15; individual MLP dossiers retain older local claims; the broad explanation index starts with an11:32follow-up while for_logan/LATEST identifies22:13. These are navigation lag, not evidence that CP or covariance work is orphaned scientifically. Authoritative chain for this addendum is CP512_NATIVE_V2 → EXACT_GAUSSIAN_CP_READOUT → GAUSSIAN_CP_DATA_READOUT → COMPONENT_TRANSFER, with plans beside each. CP512/CP400 denote architecture/budget variants, not distinct circuit identities. The graph registry's two `four_trait_verified` labels do not certify the current five-property standard or these new decompositions.

No shared registry/code refactor here: another worker is actively maintaining explanations and the new constrained-readout algorithm. Rewriting indexes or consolidating runners concurrently costs more collision risk than it saves. This review's CPU audit reuses `noncentral_gaussian_cp.gram` rather than adding another moment engine, dataset builder or scorer. Native shared hierarchy took883s versus17.79s for the three-law readout; cheaper inference does not imply cheaper fitting. The CPU audit cost1.42s; this session's elapsed time was chiefly reading and literature/interpretation. No unsupported time-allocation percentages or idle-runtime conclusions.

Folding handoff: keep the concurrent review's exact coefficient-budgeted fixed-feature readout test; log regularized and unregularized deterioration, physical factor normalization and smallest Gram eigenvalue. This audit says removing lambda alone is low-information. Circuit handoff after the focus window: retain the regional full-QK1×QK2×V forward-response requirements; for any credible quartic candidate freeze its output grouping, then test fresh same-token/shifted-context predictions and matched-norm selective removals against three unrelated readers before claiming composition. No suffix is justified without its forward census.

Receipt SHA256 prefixes: CP512_NATIVE_V2 `92bda71374b0a0b8`; DATA_READOUT `4bce106eb1e74e8a`; COMPONENT_TRANSFER `28b91f77099c9777`; this CPU receipt `9998c3f4db795693`. Full input hashes are in the CPU receipt. Existing primary receipts and TYPED_FACE_EXTRACTION_V1 were preserved.
