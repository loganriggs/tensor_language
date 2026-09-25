# Concurrent-review addendum

Existing review: [THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_0744.md](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-21_0744.md); clock unchanged.

# Three-hour mathematical, organization and efficiency review

Actual UTC: 2026-09-21 07:45:22.

ACTIVE_TRACK: WEIGHT_FOLDING, under the explicit direct-tensor focus through 22 September 15:10 UTC. Bounded review; no goal, agents, GPU execution, queue/timer changes, commits or primary-receipt edits. Previous review: 21 September 04:35 UTC. Initial clock check: 07:40:10 UTC (185 minutes elapsed).

The current path folds MLP16 source readers into three normalized last-MLP observer components. A graph sharing mixed products between components 1/2 while keeping component 3 private improves fresh relative fidelity at matched coefficient storage. Absolute mode3 continuation fidelity still fails. Product identity is unstable. This review's executed control also rejects uniformly stable four-output coefficient space, while distinguishing three closely aligned directions from a weaker fourth. None of these algebraic results establishes a semantic circuit.

## Native mathematical object and boundary

Native dimensions: 18 blocks; residual d=1152; nine attention heads of width128; MLP width4608; untied output U in R^(50304×1152). For normalized input z, MLP_l(z)=D_l[(L_l z)⊙(R_l z)]+b_l, with L,R in R^(4608×1152), D in R^(1152×4608), b in R^1152. Contracting reader c gives Q_c=sym(Lᵀdiag(Dᵀc)R) and cᵀb. Native residual mixing coefficients, first-layer value sharing, rounded rotary operations and biases must remain as implemented. The final output is 30 tanh(U RMS(h_final)/30); RMS epsilon is float32 epsilon, not a fitted constant.

For positions t,s and head a, attention has two QK contractions S^j_ts=<Rot_t Q^j z_t,Rot_s K^j z_s>, j=1,2, and a causal sum of S^1_ts S^2_ts V_s, followed by its native writer/scalings. There is no softmax. Degree is five in independent normalized residual slots before normalization is substituted. MLP degree is two. Global token computation is not a fixed polynomial because RMS denominators and final tanh remain nonlinear.

Write r_t=sum_i rho_i u_it, where u includes embedding/background, earlier attention writes A_i, and MLP writes M_i. Then r_tᵀB r_s=sum_ij rho_i rho_j u_itᵀB u_js. MLP source quadratics similarly include background/background, A/A, M/M and both ordered A/M and M/A terms. With three source classes at both sites each QK has nine terms: full QK1×QK2×V has 9×9×3=243 source-labelled terms before exact aggregation. Retaining only one QK or dropping cross terms is a different operator. Current source-form approximation keeps an approximation to the entire selected Q_c, not a nominated attention/MLP source subset; it does not close upstream attention ports or identify which of those source terms are dispensable.

The historical circuit object is [letter-continuation group2](direct_tensor_match/circuits/letter_continuation_group2.md): u=aᵀ(h−m0/2)/s(h)−alpha, v=bᵀm0/s(h)−beta, write=wuv. The current original-weight observer modes1/2/3 extend that computation; modes2/3 are not additional established semantic circuits. In exported residual coordinates each mode is

    q_(2j)(z)=c_(2j)+l_(2j)^T z+z^T Q_(2j)z
    q_(2j+1)(z)=c_(2j+1)+l_(2j+1)^T z+z^T Q_(2j+1)z
    phi_j=((a_j^T h−q_(2j)/2)/s(h)−alpha_j)
          *(q_(2j+1)/s(h)−beta_j)
    delta_h=w sum_j phi_j .

Here z,h each have 1152 coordinates: native normalized MLP16 input and native last-MLP input. s(h) is computed explicitly; generation of z,h and the recipient final residual remain external. Six source forms feed three final products and one shared residual writer. Holding RMS as a port leaves degree at most four in z,h. Native downstream RMS/U/tanh evaluate removal, including combined removal. An exact local fold is not a replacement for these backgrounds.

Current shared branch: Q_o=sum_(r=1)^256 W_or sym(l_r r_rᵀ), o=1..4, L,R in R^(1152×256), W in R^(4×256). Private mode3 has a width256 (V1) or width320 (V2) paired-core program for two further forms. Exact centered affine terms and mean corrections remain fixed. Tying consists of the two shared reader banks and product nodes across four outputs, plus shared writer w; the private branch does not share those learned products. Gauges include product permutation, l→a l/r→r/a, l/r exchange, compensating factor/readout scaling and congruence changes of a dense shared-core coordinate frame. Arbitrary mixing of quadratic atoms need not preserve cheap rank-two products.

Weight fitting uses pair-equalized covariance-informed coefficient Frobenius error: M_o=S Q_o S in the fixed calibration metric. It is neither isotropic error nor the actual non-Gaussian fourth/eighth-moment normalized functional loss. Native tests use relative L2 logit-effect error against the same native removal, with natural/hybrid limits15%, change20%, and graph/baseline ratio≤1.10, separately per cohort/component. Replay, prediction and causal selectivity are distinct.

## BASELINE_COMPARISON

Authority: [DECOMPOSITION_BASELINES_2026-09-20.md](DECOMPOSITION_BASELINES_2026-09-20.md). Comparisons below preserve outputs, z/h ports, native normalization/readout, precision and error definition.

| Baseline/candidate | Measured scope and price | Verdict |
|---|---|---|
| Native two-source factors | Earlier exact pair receipt: 10,628,354 floats,4608 products; full spectral2304 products/2,658,818 floats; exact paired compiler1152 products/1,331,714 floats+3456 indices | Exact pair baseline, not same-price three-mode comparison |
| Independent paired-core banks vs partial mixed graph V1 | Both897,804 float coefficients; baseline768 vs graph512 source products | Fresh relative cells72/72 pass; absolute mode3 continuation16.735% fails15% |
| Capacity-matched V2 | Both971,660 float coefficients;832 vs576 source products | Again72/72 relative pass; absolute mode3 continuation16.009% fails |
| Spectral/common-input Tucker | Common320 saves30% coefficients but mode3 opened18.639%; partial sharing also fails relative limits | Measured structural alternatives; not a global lower bound |
| HT / alternate tree / native channel pruning / matched-price random graph | Historical HT homogeneous branch exists; no complete same-output, same-port, same-price comparison for this new normalized three-mode graph | Missing; historical runs do not fill these cells |

V1 stores7,182,432 coefficient bytes at FP64,3,591,216 at FP32; V2 stores7,773,280/3,886,640. These byte conversions do not imply that all deployed artifacts share one precision. Add integer product-index arrays, metadata, native producer weights and native suffix costs. V1 computes768 dense input projections (512 shared mixed readers+256 private readers),1024 shared readout coefficients, plus private-core readout, six affine readers, three h readers, three final products and the common writer. V2 adds64 private projections/products. Linear arithmetic and live buffers remain substantial; no measured wall-time acceleration follows from product savings. Exact full-model or total-port-closure price is not available and is not zero.

## Five-property scorecard

| Property | Current assessment |
|---|---|
| Simple | Partial structural gain:256 fewer source products at matched coefficients; no storage or demonstrated speed win |
| Predicts held-out/OOD | Fresh identified FineWeb and new stdlib-file panels give relative success; absolute subgroup failure retained. OOD code transfer is conditional on native states, not token-only prediction |
| Extracted | Conditional executable z/h→write boundary implemented; two native vector inputs plus native intervention background/readout remain. No upstream closure |
| Selective | Earlier continuation candidate has removal/control evidence, but the new multimode graph lacks the full three-unrelated-reader/equal-norm-random removal battery. Not established |
| Composes/reuses | Actual shared products and joint-removal prediction tested; not independent-task reuse or small Möbius interactions against matched random splits. Not established as a semantic circuit |

Each fresh panel has32 identified FineWeb document contexts and16 source-file contexts, not72 independent contexts:72 are overlapping metric cells. Site counts are nested within those contexts; semantic template/city cells do not exist for this observational panel. Historical calibration32/256 rows are chunks with incomplete document provenance. Prefix deduplication alone does not restore document independence. Evidence tags: learned graph **fit**; exact compiler/replay **fold**; native frozen removals **edit**; no new forward suffix **response** census in this review.

## LITERATURE_SEARCH

Actual queries executed this review:
1. `Wedin 1972 perturbation bounds singular subspaces singular value decomposition`
2. `Kruskal 1977 three way arrays rank uniqueness trilinear decompositions`

Opened primary sources: [Wedin1972](https://link.springer.com/article/10.1007/BF01932678), publisher abstract accessible but full text subscription restricted; [Cai and Zhang, full primary paper](https://arxiv.org/pdf/1605.00353), including Section2 definitions of projectors/principal angles and perturbation framework. Also opened its [abstract](https://arxiv.org/abs/1605.00353). Kruskal's [publisher page](https://www.sciencedirect.com/science/article/pii/0024379577900696) returned an internal access error; [CORE PDF](https://core.ac.uk/download/pdf/82529515.pdf) was inaccessible. No full-text Kruskal theorem verification claimed.

Plan change: use coefficient-output subspaces as a narrower identification object than atom matching. Singular-subspace perturbation applies to a matrix whose columns are vectorized symmetric quadratic forms; it does not require Gaussian data. Fixed metric and separated nonzero singular values are essential. Statistical independent-noise/minimax guarantees are inapplicable to correlated optimizer starts. Mixed products yield a symmetrized tensor, not an unconstrained CP model with independently identifiable columns; CP uniqueness cannot simply be imported.

For this full-column-rank restriction derive an elementary executable bound. Let A=[vec(M_1),...,vec(M_4)], B the corresponding alternate fit, P_A its orthogonal projector. Since (I−P_A)B=(I−P_A)(B−A),

    ||sin Theta(col A,col B)||_2 <= min(1, ||B−A||_2 / sigma_min(B)).

This bounds subspace rotation without asserting unique product recovery. Compute AᵀB from factor contractions:

    <sym(l r^T),sym(a b^T)>
      = ((l·a)(r·b)+(l·b)(r·a))/2.

Thus four-output Grams and cross-Grams require O(d R²+4R²), O(R²+dR) auxiliary storage; only4×4 eigensystems follow. A dense four-matrix contraction is an independent verification, not an exported dense fallback. This precise restriction is cheaper and clearer than another optimizer sweep.

## Executed consequence; REDTEAM_NEGATIVE

New isolated CPU control [script](direct_tensor_match/review_output_span_20260921_0740.py), [receipt](direct_tensor_match/REVIEW_OUTPUT_SPAN_20260921_0740.json), executed with /venv/main/bin/python, CUDA disabled, two threads:0.69seconds computation. Before execution set all-four principal cosines≥.99 across all three alternate fits. Results: worst cosines .95242/.95275/.95495, so FAIL. Other three cosines exceed.9969. Maximum sine .297–.305; analytic upper bounds .685–.727; smallest singular value about.0522. Aggregate coefficient difference4.18–4.43% can coexist with this weak-direction rotation.

Rescue attempt: compare output-form spaces rather than individual product atoms, allowing any invertible output recombination. Positive mixing control cosines≥1−1.6e−13; same-capacity signed-coordinate null maximum cosine .00331; independent dense Gram relative replay1.01e−14. Original input/program/result SHA256 hashes were verified unchanged. This rescues three closely aligned directions descriptively, not the preregistered four-direction claim or a globally unique decomposition. Existing conditioning audit finds product-Gram condition23.7–26.1 and FP32/64 scalar discrepancy<3.3e−6; a precision breakdown does not explain the atom-identity failure. Planted signed profiled-fit failure recovered with another restart; original miss remains optimization evidence, not impossibility.

Concurrent primary receipts now include PROFILED_CONSENSUS_V1 and PROFILED_CONSENSUS_EXECUTION_V1: three directions in the *joint output-weighted atom space* retain99.9937% energy and opened per-mode errors2.86/2.48/11.94%, still512 products. Those are a different object from this review's four source-output matrices. No contradiction, no three-product price, and no fresh confirmation. Their implementation and receipts were not touched.

## REDTEAM_POSITIVE

Strongest success is the partial graph's fresh relative comparison. Both V1/V2 private-mode3 branches are identical to their capacity-matched baselines; passing their relative ratio is inherited, not a learned improvement. Absolute failure remains in the same branch. Calibration means, exact affine terms and RMS/background ports are shared and charged; they can make total error look good without isolated quadratic fidelity. Complete isolated constant/zero/shared-edge removal and matched-cost random controls on this new graph remain missing; old constant controls do not establish their result here. New coordinate-null control tests coefficient stability only, not selectivity or causal reuse.

V1 recipient bootstrap explicitly conditions on frozen donors and does not cover donor-map or training-selection uncertainty. New document IDs and same-token different-document donor constraints reduce leakage, but code files are related source contexts, not independent semantic tasks. No new semantic promotion. A combined low error cannot override mode3 failure or unstable components. Native final replay0 and source replay~1.7e−7 support the instrument, not the causal claim.

## Organization, efficiency and handoffs

Read the full research skill and startup guide, NEXT_CODEX_PROMPT, both user authorities, focus note, latest mathematical/hourly receipts (05:20/06:34), board tail, recent commits, graph/circuit/path registries, module and MLP indexes/dossiers, direct-study README, continuation dossier/package and current primary plans/results. Recent commits2509d9ee2/7a0ec299b/214df4e98 identify the live graph work. tensor_language has286 dirty/untracked paths at observation; theseus-bench is clean. Preserve all concurrent work.

Reconciliation: historical `half0 component2`/`letter_continuation_group2`, native observer modes1/2/3, shared16 source package and current partial graph are distinct versions/scopes, not interchangeable aliases. Continuation dossier carries the source-fold, donor-failure and provenance corrections. The general path registry and explanation README stop at20September20:15 while detailed receipts/board reach21September07:xx. The generated graph registry's49packages/two four-trait entries are historical package-schema assessments, not proof that this graph passes five properties. MLP index summaries still point to earlier route snapshots. This review links live controls without overwriting historical claims; missing current cross-links are documentation debt, not orphaned scientific evidence.

Startup's workstation systemd paragraph and September13 NEXT handoff are stale; current authority is /workspace/tensor_language, Supervisor and the focus note. Resolve historical paths against this checkout for reads. No service changes. bqrunner/bqrunner2 observed RUNNING; queue empty; last two native jobs terminated successfully07:20:00/07:27:28, taking6.23/6.16seconds. Profiled four-fit job took190.4seconds; tiny native confirmations are dominated by preparation/reporting latency. Empty queue alone does not prove waste. Board entries occasionally run ahead of actual wall clock; use execution receipts and actual UTC, not board prose, to date deadlines.

Repeated builders `prepare_partial_graph_fresh*.py` and freeze scripts are plausible declarative-configuration candidates. Existing shared_mixed_source_graph/source executor and scorer already avoid copying core evaluation. No shared-code refactor now: primary worker is actively changing nearby controls, versions preserve frozen provenance, and merging builders requires a provenance regression check larger than this bounded review. The executed mathematical control supplies the requested consequence; a second infrastructure project would cost more than it saves now. No new publisher, duplicate sweep, or registry regeneration.

Folding handoff: retain partial shared1/2/private3 topology and the absolute subgroup miss. Investigate the weak fourth output direction with an explicit metric/covariance-shift or restart hypothesis; do not erase it by aggregate consensus projection. Compare isotropic and covariance-informed coefficient geometry and preserve all optimizer failures. Current consensus execution is opened evidence at unchanged price; a future independent test must freeze its grouping first.

Circuit handoff, deferred until the focus permits: original-observer continuation removal selected these source quadratics as the folding target. The algebra proposes grouping shared products for modes1/2 and keeping mode3 separate; test that grouping with preregistered matched-strength/random controls and at least three unrelated readers. Exact upstream port closure must retain full QK1×QK2×V terms; obtain a forward response census before naming a suffix. Neither closed Möbius identity nor compression establishes small interactions relative to the smallest piece.

Receipt hashes and full numeric control values are in the new JSON; primary receipts remain immutable. Limitations: correlated starts, covariance-shaped metric, no new native forwards, incomplete same-interface HT/random baselines and no five-property adoption. Next deadline remains governed by the concurrent 07:44 review (10:44 UTC); this addendum does not reset it.
