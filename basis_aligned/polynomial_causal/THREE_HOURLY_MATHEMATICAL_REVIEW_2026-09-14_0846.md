# Complete normalized moments: outer integration is still unresolved

Review clock: **2026-09-14T08:46:14Z**. Next mathematical deadline: **2026-09-14T11:46:14Z**. This is one active bounded review and one executed CPU consequence; no offline reviews were backfilled. The previous review, 02:56:03Z, was over 175 minutes old at entry.

The new consequence is an outer-context integration audit and a sign-symmetry obstruction to ordinary antithetic variance reduction. For the frozen setting2 sparse operator, 1024 independent synthetic contexts fail the registered 1% integration agreement bar. Global-sign pairs have correlation 0.999999963 and approximately double variance at equal function-evaluation count. The nine-term identity passes, but its Monte Carlo accuracy remains insufficient for the requested tolerance. No fitting was performed.

Read the complete recovered research-driver skill, current startup and local-start guide, NEXT_CODEX_PROMPT, board protocol/latest tail, newest mathematical review and local CPU receipts. Read SHARED_KEY_VALUE_MOMENT_V1_MATH.md, THREE_GROUP_SHARED_DAG_V1_MATH.md, HEAD17_OUTPUT_BLOCK_FIT_V1_MATH.md and retained_contraction_error_control_v1.py, plus normalized ports/rotary, target builder, native CPU executor and sparse-interface evidence. Inspected current result pointers, ledger/backlog tails, recent commits and both local repositories' dirty state. Historical /workspace paths were resolved against this checkout and the restored local checkpoint for reading. The user’s bounded scope overrides old goal creation, continuation, agents, GPU, scheduling restrictions and commit/push requirements. No goal exists in this thread and none was created. Existing concurrent work and primary receipts were preserved.

The completed Gaussian metric bounds, shared-output fits and coordinate charts remain closed. The prior source-mask Horvitz–Thompson control already established its identity and normalization counterexample; it was read, not rerun. Current receipts preserve the sparse operator’s conditional regional success, broader full-head failure, and lack of CPU speedup. The earlier four-panel native-weight screen is an instrument result under synthetic inputs, not native trajectory fidelity. This review advances its proposed outer-integration audit rather than repeating that screen.

## Exact interface under review

Use residual indices i,j=1,…,1152, head coordinate h=1,…,128, MLP channel m=1,…,4608, selected output o=1,…,12, source s=0,…,t, and final query t. Head17.2 means zero-based layer17/head2. Native head maps Q1,K1,Q2,K2,V have shape 128×1152; W has shape 1152×128. The four raw corners share these maps and inherited first-layer inputs:

\[
X_N=X,\quad X_C=X+\Delta_c,\quad X_R=X+\Delta_r,
\quad X_A=X+\Delta_c+\Delta_r\in\mathbb R^{(t+1)\times1152}.
\]

These are shared raw-state corners, not independently sampled normalized query/key/value ports. With ε=2⁻²³, let r_{u,s}=mean_i X_{u,s,i}²+ε. The additive reconstruction must include

\[
r_A=r_C+r_R-r_N+2\operatorname{mean}_i(\Delta_{c,i}\Delta_{r,i}).
\]

For q=QX or k=KX, the pulled-back residual-RMS followed by head-RMS port is q̃=q/sqrt(mean_h q_h²+εr). The εr term is required. Both scores are retained:

\[
a_{u,s}=128^{-1}\langle R_t\widetilde{Q1X}_{u,t},R_s\widetilde{K1X}_{u,s}\rangle,
\quad b_{u,s}=128^{-1}\langle R_t\widetilde{Q2X}_{u,t},R_s\widetilde{K2X}_{u,s}\rangle.
\]

No softmax is present. Literal R_s is the half-split block matrix [diag(c),diag(sin);−diag(sin),diag(c)], frequencies 10000^(−2j/128). Angles are computed in FP32, cosine/sine tables rounded to BF16 then lifted to the computation dtype. Rounded R_s is not assumed exactly orthogonal; a relative-position replacement is not automatically exact. The query is also source s=t, precluding independent-query/source factorization on that row.

Values are

\[
v_{u,s}=(1-\mu)VX_{u,s}/\sqrt{r_{u,s}}+\mu v^{first}_s,
\qquad \mu=-0.0888671875.
\]

The first-layer head2 value is shared across corners. Its actual upstream generator must be priced and remains part of a full-model implementation. It cannot silently become an independent free value variable.

Write a=a_N and a_c=a_C−a_N, a_r=a_R−a_N, similarly for b,v. Let ā=a+a_c+a_r, b̄=b+b_c+b_r, v̄=v+v_c+v_r, e_a=a_A−ā, e_b=b_A−b̄. The three retained writes are

\[
u_{1,s}=[a_r\bar b+(a+a_c)b_r]_s v_{c,s},\quad
u_{2,s}=[a_c\bar b+(a+a_r)b_c]_s v_{r,s},\quad
u_{3,s}=[e_a\bar b+\bar a e_b]_s\bar v_s,
\quad A_k=\sum_su_{k,s},\quad A=\sum_{k=1}^3A_k.
\]

The output is WA. This is the retained approximation to the larger mixed attention expansion, not the exact full mixed attention response. Six scalar gate products plus three 128-coordinate gate/value products cost 390 multiplications per source; the other expansion terms are still omitted.

For L,R∈R^(4608×1152), D∈R^(1152×4608), selected unembedding U∈R^(12×1152), C=UD,

\[
T_{oih}=\sum_m C_{om}\{L_{mi}(RW)_{mh}+R_{mi}(LW)_{mh}\},
\qquad M_o(z,A)=\sum_{ih}T_{oih}z_iA_h.
\]

T has shape 12×1152×128, 1,769,472 entries. The twelve frozen token IDs are [11383,1430,14148,6907,31049,14733,34614,18388,39012,16521,46729,16924]. Their original selection is inherited; this review reads no text to fit or choose a candidate. z is the declared MLP-only background, not an independent arbitrary Gaussian port in the native circuit. Let H=z+WA and r_H=mean_i H_i²+ε. For fixed E=T̂−T the conditional raw-logit error is

\[
f_{k,o}(\omega)=E_o(z,A_k)/d(\omega),\quad
 d=r_H\rho(h_{final}),\quad
 G_{kl}(\omega)=\langle f_k,f_l\rangle,\quad
 J(E)=\mathbb E_\omega\sum_{k,l=1}^3G_{kl}(\omega).
\]

All nine entries belong to the objective, including negative cross entries. Identical exact denominators are supplied to candidate and reference. Remaining terms, residual/re-entry history, other heads, Down_bias, linear and head/head terms and other output paths remain in the reference background/readout. The native final score is separately 30 tanh(ℓ_o/30) for each original token; this raw-logit squared norm neither replaces that softcap nor estimates CE/Fisher or signed margin damage. A ratio to reference energy and its square root are not unbiased merely because their numerator is.

M is bilinear in free (z,A). The retained expressions are cubic in independent score/value ports. With RMS denominators frozen, the current-value raw numerator has degree five (six with independent linear z); inherited value paths have their corresponding lower raw-current degree. Actual normalization and final softcap make the composition nonpolynomial.

An invertible head-coordinate change A′=GA, W′=WG⁻¹, T′_o=T_oG⁻¹ preserves the interface if all three producers are transported. It does not authorize arbitrary basis changes inside normalized/rotated QK. MLP permutations, L/R exchange and reciprocal channel rescalings preserve the factorization with matching decoder transport. Output rotations require decoding before individual softcaps. Neither this objective nor the sampling theorem guarantees unique factors, a minimum arithmetic circuit, or semantic identification.

## Literature mapping and executable consequence

The applicable variance theorem is given explicitly in [Owen, Monte Carlo theory, methods and examples, §8.2, equations (8.2)–(8.4)](https://artowen.su.domains/mc/Ch-var-basic.pdf). A measure-preserving reflection with finite variance gives an unbiased paired average; at equal evaluation count its variance relative to independent sampling is 1+corr(f(X),f(−X)). Even integrands give the worst case, factor two. Computational reuse must be assessed separately from evaluation count.

Our mapping: the random vector ξ contains four independent standard Gaussian arrays of shape 5×1152, hence 23,040 independent coordinates. Set X=ξ₁, Δ_c=.1ξ₂, Δ_r=.2ξ₃, initial=ξ₄. Let vfirst=V0 initial/RMS(initial), and z=X_t. This is exactly the synthetic law of NATIVE_RETAINED_ERROR_V1, distinct from the prior HT control’s correlated first/background law. The involution ξ↦−ξ preserves this law. Choose f(ξ)=G_kl(ξ) separately for every branch pair, or f(ξ)=Σ_kl G_kl(ξ) for total energy. Ordinary context averages are unbiased entrywise; paired averages also are. Sampling an entire context preserves all its shared nonlinear dependencies.

Finite second moments follow in this restriction without assuming normalized outputs are Gaussian. QK normalized vectors are bounded, rounded rotations have finite operator norm, RMS-normalized current and initial value maps are bounded for fixed weights, and so are A_k. With ε>0, d≥ε^(3/2). Consequently |G_kl|≤constant×||z||², whose square is Gaussian-integrable. This establishes an unbiased finite-variance objective estimator, not a useful worst-case numerical constant or finite-sample confidence guarantee. IID standard-error estimates concern outer contexts; sign partners are dependent and must be counted as pairs.

Our algebraic specialization explains the result. Under the global sign reversal, Q and K both change sign, their norms stay fixed, both score arrays and every gate stay fixed, values and A_k reverse sign, and z reverses sign. Thus each mixed numerator E(z,A_k) is exactly invariant. r_H is also invariant. If final RMS were supplied as RMS(z), the complete Gram would be exactly even and global antithetics could remove no variance.

The actual conditional MLP reference in this control instead computes

\[
h_{final}=H+D[(LH/\sqrt{r_H})\odot(RH/\sqrt{r_H})]+b_D.
\]

Its bilinear MLP term and bias are even, while H is odd. Writing B for their sum, the squared final norms of H+B and −H+B differ by 4 mean_i H_iB_i. Final RMS therefore breaks exact evenness. This offers a real falsifier: antithetic sampling could help only if the resulting energy covariance is sufficiently favorable. It cannot be justified just from symmetric raw Gaussian inputs. The executed covariance resolves this for the declared panel.

We also searched and opened the primary paper [Owen (2008), Local antithetic sampling with scrambled nets](https://arxiv.org/pdf/0811.0528). It treats randomized cubature on a unit cube and obtains improved rates for sufficiently smooth transformed integrands. Our Gaussian transformation is coordinatewise inverse normal; its derivatives can diverge at cube boundaries. The required transformed smoothness and favorable effective dimension are not established at dimension 23,040. Global sign reflection is also different from local box folding. That stronger rate is therefore not claimed here; no QMC implementation was run.

The earlier [Horvitz–Thompson (1952) primary scan](https://www.stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf) was opened. Conditional on the full context and d, let f_{k,s}=E(z,u_{k,s})/d. Its complete source-pair estimator is Σ_st I_sI_t〈f_{k,s},f_{l,t}〉/π_st. For independent Bernoulli inclusion, π_ss=p_s but π_st=p_sp_t off the diagonal. These positive known inclusion probabilities ensure unbiasedness of all nine entries. Recomputing d from sampled writes violates the fixed-population premise; an unbiased numerator does not make reciprocal RMS unbiased. The previous executed receipt demonstrated both failures of naive substitution. Today’s exact source sum avoids this inner variance entirely.

Separable Gaussian contraction concerns a different integrand. [Isserlis (1918), original paper DOI](https://doi.org/10.1093/biomet/12.1-2.134) integrates products of jointly Gaussian linear variables by pairings; the scan failed to open today, and the existing derivation was read locally. Gaussian sharing itself is allowed if every covariance is retained. Here random RMS denominators depend on those same variables, normalized variables are non-Gaussian, additive defects share corners, source/query coincide at t, and inherited values/background need not factor. The completed 1.10560724 restricted Gaussian distortion bound supplies no bound on this normalized J. Exact context sampling changes neither the declared object nor these dependencies; separable Wick contraction does.

## Executed audit and preserved failures

[Control source](outer_context_antithetic_20260914_0844.py) · [Primary receipt](OUTER_CONTEXT_ANTITHETIC_20260914_0844_RESULT.json).

The source and board claim froze 16 seeds 170214840–170214855, 64 contexts per seed, five sources, independent-half and 256→512→1024 checks, 1% agreement threshold and 120-second cap before execution. It reuses the existing normalized ports and retained components, target builder and sparse artifact decoding. Native checkpoint restoration is complete enough for CPU weight reads; no native model trajectories are generated. The restored checkpoint blob was verified by the preceding recovery work; this invocation did not rehash that large file. Candidate SHA256 is recorded in the receipt. The source has no optimization loop and refuses to overwrite its receipt.

Execution used CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=2, MKL_NUM_THREADS=2, PYTHONDONTWRITEBYTECODE=1 and /home/loganriggs/.local/share/bilin18/venv/bin/python. Actual execution started 08:45:42.848748Z, validation 08:45:44.605687Z, receipt 08:45:44.605942Z. Internal runtime was 1.7597 seconds. There were 1024 independent contexts and 1024 dependent sign partners, not 2048 IID samples. Publication boundary was 08:46:14Z; no historical activity-log entries were fabricated or modified.

| Measurement | Result |
|---|---:|
| Full Gram/direct replay | 2.546e−16 relative |
| Mixed numerator and even-denominator sign replay | exactly zero discrepancy |
| Mean complete energy Ĵ | 0.01683111187 |
| IID context estimated relative SE | 3.2717% |
| Independent-panel estimated relative SE | 3.8552% |
| Independent 512-context half disagreement | 2.1309% |
| 256→512 / 512→1024 relative changes | 1.0537% / 1.0655% |
| Diagonal-only / complete mean energy | 1.009727 |
| sqrt(mean error energy / mean reference energy) | 8.2523% |
| Sign-pair energy correlation | 0.99999996335 |
| Equal-evaluation antithetic / independent variance estimate | 1.99999996318 |
| Actual final-RMS sign energy discrepancy | 0.01962% relative L2 |

A passes; B (all integration disagreements ≤1%) fails; C (antithetic variance ratio <1) fails. No sample extension or objective change rescued them. The Gram entry estimates and their SEs are in the receipt: off-diagonal entries are small relative to their uncertainty, which does not authorize dropping them. Their exact cancellation contribution survives in every context.

The approximate normal 95% interval for J is [0.01575182,0.01791040]. It is an asymptotic diagnostic, not a finite-sample bound, simultaneous nine-entry interval, post-selection guarantee or native fidelity interval. Estimated variance scaling suggests roughly 10,961 independent contexts for 1% relative SE under this same law; that is neither a 1% confidence guarantee nor a sample count that will necessarily pass the separate agreement bar. The 8.2523% ratio is descriptive and is not an unbiased relative-error estimator. The older four-panel result is preserved, not contradicted by this different larger random sample.

## Price and decision

The test introduces no new model parameters, edges, masks or intervention generators. Dense T stores 7,077,888 FP32 bytes. The frozen sparse program stores 4,896,936 packed tensor bytes / 4,899,795 serialized bytes, and its CSR representation occupies 9,340,736 bytes before temporary state. All 12 output, 1152 residual and 128 head nodes remain. Existing masks/adapters are included. The audit reconstructs the operator densely in FP64, so it is not a sparse-runtime benchmark.

Five head17 maps, the first-layer value map, W and μ add 1,032,193 scalars / 4,128,772 FP32 bytes. Dense folded plus these producers is 11,206,660 bytes; sparse packed plus producers is 9,025,708 bytes. These are conditional-interface subtotals. If the exact last-MLP/final-RMS generator used in this audit is also explicitly retained, L/R/D and Down_bias add 15,926,400 scalars / 63,705,600 bytes: sparse subtotal **72,731,308 bytes**, dense subtotal **74,912,260 bytes**, still requiring external X, edits, initial and z. This is a literal expanded representation price, not minimal storage: L/R/D may already be shared by other paths, and a native-factor implementation can reuse them instead of storing T. The local L/R/C/LW/RW alternative was previously priced at 47,407,104 bytes. No saving may be charged twice.

A closed model must include the union of upstream trajectory, background, intervention-definition, remaining readout and consumer dependencies. Keeping the native model as that generator retains its 545,902,902 parameters, plus extra artifacts; no smaller closed whole-model total is established. Projected-port state alone includes 1920 projections, four norm/cross scalars and 128 inherited-value coordinates per source across the three raw corners, before routing, decoding and downstream state. Exact source summation costs O(3S×128) after gate formation; each context’s mixed reader preparation costs O(12×1152×128), followed by O(3×12×128+9×12) contractions. Full projections and MLP denominator generation are additional costs and dominate this small source sum. All contexts and sign partners were actually evaluated, with no GPU/body forwards.

The mathematical null is useful: ordinary global antithetics cannot be presumed beneficial for this nearly even normalized objective. Algebraic reuse could compute a partner denominator cheaply, so the equal-evaluation penalty is not a wall-time lower bound. Even free partner evaluation offers negligible variance reduction on the measured panel; no production kernel speed claim was tested.

The next bounded scientific decision is a preregistered higher-accuracy outer integration audit under this unchanged law (or a separately justified law), using independent validation randomness and all nine entries. A substantially larger fixed sample budget is computationally plausible given the measured runtime, but is not launched by this bounded review. Better variance reduction must target the even integrand; a favorable antithetic assumption or another independent Gaussian marginal is not a repair. The current 1024-context estimate does not meet the requested fitting-readiness bar. This is objective/composition infrastructure only: it establishes no native held-out/OOD prediction, semantic grouping, independent extraction, selective manipulation, stable identification, new compression or adoption.
