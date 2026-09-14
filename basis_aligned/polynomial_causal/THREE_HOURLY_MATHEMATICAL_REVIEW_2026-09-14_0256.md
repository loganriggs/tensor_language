# Mathematical review: complete normalized source-pair moments

UTC: 2026-09-14T02:56:03+00:00. Next mathematical deadline: 2026-09-14T05:56:03.000Z.
This is one bounded active review, not a backfill for offline hours. The preceding mathematical review was 2026-09-13 14:29 UTC, more than 175 minutes old. No durable goal exists in this thread; none was created.

The consequence is an unbiased estimator for all nine retained-contraction error cross moments, conditional on the full normalized shared context. Exact enumeration with actual head17.2/MLP17 weights passes at 2.31e-16 relative Gram error. Squaring an unbiased sampled write is biased; recomputing RMS from that sampled write is separately biased. Its measured variance and the cheap exact source sum argue against using source sampling for this short-prefix objective.

## Evidence and boundary

Read completely: the recovered session_recovery/bilin18-research-driver/SKILL.md, current startup guide and NEXT_CODEX_PROMPT.md, plus SHARED_KEY_VALUE_MOMENT_V1_MATH.md, THREE_GROUP_SHARED_DAG_V1_MATH.md, HEAD17_OUTPUT_BLOCK_FIT_V1_MATH.md and retained_contraction_error_control_v1.py. Read the latest board tail, local restoration/hourly receipts, target builder, normalized ports/rotary/package price, sparse executor and shared-write results, latest result pointers and relevant ledger tails. Historical /workspace paths were resolved to this checkout or the restored local checkpoint. Existing dirty board, startup, runner, canary, activity and recovery work was preserved.

The setting2 sparse operator retains its narrow regional result; broader full-head native effects fail. Shared-output and balanced/private fits did not resolve this. The completed independent Gaussian correction has squared-objective distortion 1.10560724 in its restricted measure, not a bound on normalized retained inputs. The earlier synthetic complete-Gram identity already passed; it is reused here as an oracle, not presented as a new result. Coordinate charts, output fits and Gaussian marginal bounds were not rerun. Prior polynomial slot estimators in MIXED_REPEATED_CONTRACTION_V1_MATH.md and COMPOSED_QUARTIC_CONTRACTION_V1_MATH.md were checked: this control instead samples source pairs of a normalized, shared-corner program.

The restored checkpoint is readable. Its blob is 680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3; the concurrent hourly review verified the full hash. This control loads native weights without native trajectories or cached text. Restoration does not turn synthetic corners into model inputs.

At final publication inspection, concurrent [NATIVE_RETAINED_ERROR_V1_RESULT.json](NATIVE_RETAINED_ERROR_V1_RESULT.json) had landed. Its four synthetic 32-row panels evaluate the frozen sparse candidate with actual weights and supplied exact conditional last-MLP/final RMS: relative mixed errors 8.47–9.08%, diagonal/joint energy ratios 1.0116–1.1266, Gram replay at most 1.97e-16. This is separate work, not executed or owned by this review. It already supplies an initial four-seed variability check; the next integration audit should extend that evidence toward a quantified population tolerance, not rerun its screen. Our distinct consequence is exact source-mask expectation, the same-source inclusion correction, and the failure of sampled normalization. No native trajectory or fidelity claim follows from either synthetic measure. The concurrently added local-startup section and ownership claim were read and preserved.

## The actual setting2 object

Indices: residual i,j=1..1152, head coordinate h=1..128, bilinear channel m=1..4608, selected output o=1..12, causal source s=0..t and final query t. A context contains three raw preattention arrays X_N, X_C, X_R in R^{(t+1)×1152}. Define Δ_c=X_C−X_N, Δ_r=X_R−X_N, X_A=X_C+X_R−X_N. Corners share the same five head17.2 matrices Q1,K1,Q2,K2,V in R^{128×1152}, writer W in R^{1152×128}, first-layer values and absolute position tables. These are shared raw-state corners, not independent draws of each normalized port.

For each corner u and source s, r_{u,s}=mean_i X_{u,s,i}²+ε, ε=2^{-23}. Additive reconstruction uses

r_A=r_C+r_R−r_N+2 mean_i(Δ_c,i Δ_r,i).

With raw projection q=Q X and k=K X, the normalized projection is q/sqrt(mean_h q_h²+ε r_u), and likewise k. This is the algebraic pullback of residual RMS followed by head RMS; it retains the ε r_u term. Score a_{u,s}=〈R_t q̃1_{u,t},R_s k̃1_{u,s}〉/128 and b uses Q2,K2 independently in the same formula. Both factors survive; there is no softmax. The final query is also source s=t, so an independent-query/source assumption fails even under independent raw source rows.

R_s uses the native half-split matrix [diag(c),diag(sin);−diag(sin),diag(c)], frequencies 10000^{-2j/128}, FP32 angles followed by BF16-rounded cosine/sine tables lifted to the evaluation dtype. Rounded matrices are not assumed exactly orthogonal; relative positions cannot silently replace these literal matrices.

The value port is v_{u,s}=(1−μ)V X_{u,s}/sqrt(r_{u,s})+μ v^{first}_s, with actual signed μ=−0.0888671875. The inherited first-layer value is shared across corners; in a full execution its generator and upstream dependency remain required. It is not an independent Gaussian replacement for the current source.

Write a,b,v for native ports, a_c=a_C−a_N and similarly for b_c,v_c and remainder changes. Let ā=a+a_c+a_r, b̄=b+b_c+b_r, v̄=v+v_c+v_r, e_a=a_A−ā, e_b=b_A−b̄. The retained three source writes are

u_{1,s}=[a_r b̄+(a+a_c)b_r]_s v_{c,s},
u_{2,s}=[a_c b̄+(a+a_r)b_c]_s v_{r,s},
u_{3,s}=[e_a b̄+ā e_b]_s v̄_s.

A_k=Σ_s u_{k,s}; A=Σ_{k=1}³ A_k. The residual write is WA. This is the frozen three-contraction approximation to the larger finite mixed attention expansion, not equality to the full native mixed response. In particular, omitted score/value-defect products do not become exact through a better error metric. Six scalar gate products plus 3×128 value products cost 390 multiplications per source in the retained core.

For the downstream mixed term, let L,R∈R^{4608×1152}, D∈R^{1152×4608} be MLP17 weights, U_sel∈R^{12×1152}, C=U_sel D, and

T_{oih}=Σ_m C_{om}[L_{mi}(RW)_{mh}+R_{mi}(LW)_{mh}],
M_o(z,A)=Σ_{ih}T_{oih}z_i A_h.

Thus T has shape 12×1152×128 and 1,769,472 entries. Its rebuilt FP64 norm is 27778.377213402393. The twelve token IDs are in the receipt. z is the declared MLP-only background at the pre-MLP17 boundary; it is not assumed independent of the corners. The package's direct final-write interface alone does not authorize propagation through MLP17; setting2 explicitly adds this separate mixed-operator boundary.

For E=That−T, the conditional raw-logit error is

δℓ_o=E_o(z,A)/d,  d=[mean_i(z+WA)_i²+ε] ρ(h_final).

The native conditional validation supplies the same compact state, final RMS and other terms to exact and approximate versions, including their rounding. Linear terms, head/head term, native Down_bias, residual/re-entry history, other heads and output readers remain in that supplied background/readout. The final score is 30 tanh(ℓ_o/30), applied separately to each token. The norm here is J(E)=E_ω Σ_o δℓ_o(ω)² under a declared weights-only context measure, not CE, Fisher, coefficient Frobenius error or a signed margin metric. The reported control is a finite panel version. A ratio to reference energy or its square root is not automatically unbiased.

As a free (z,A) map M is bilinear. In independent score/value ports each retained term is cubic. Freezing every RMS denominator gives raw current-value numerator degree five, and degree six with an independent linear z; actual RMS makes the composed object nonpolynomial and the final tanh adds another nonpolynomial operation. Polynomial coefficient-norm theorems require that different restricted object.

Gauge: an invertible head-coordinate change A′=G A with W′=W G^{-1} and T′_o=T_o G^{-1} preserves this interface if every producer branch is transported. It is not permission for arbitrary changes inside normalized/rotated QK maps. MLP channel permutations, L/R interchange and reciprocal channel rescalings preserve the bilinear factorization with corresponding decoder transport. Output rotations require decoding back to the twelve original readers before separate softcaps. These freedoms make factors nonunique; no basis is an identified semantic circuit merely because it minimizes this loss.

## Literature and theorem mapping

The primary sampling source is [Horvitz and Thompson (1952), A Generalization of Sampling Without Replacement from a Finite Universe](https://www.stat.cmu.edu/~brian/905-2008/papers/Horvitz-Thompson-1952-jasa.pdf), [publisher/DOI](https://doi.org/10.1080/01621459.1952.10483446). Inverse inclusion probabilities yield an unbiased finite-population total when population values are fixed and relevant probabilities are positive and known. The following is our application and direct proof for this operator, not a claim that their paper studied neural normalization.

Condition on the entire context ω, including full denominators, and a fixed E. Define f_{k,s,o}=E_o(z,u_{k,s})/d. The complete conditional error Gram is

G_{kl}=Σ_{s,t}〈f_{k,s},f_{l,t}〉,  J_ω=Σ_{k,l=1}³ G_{kl}.

Map the finite population to ordered source pairs (s,t), with population value 〈f_{k,s},f_{l,t}〉 for each of the nine branch pairs. Sample a common source subset with indicators I_s and pair inclusion π_st. Then

Ghat_{kl}=Σ_{s,t:I_s I_t=1}〈f_{k,s},f_{l,t}〉/π_st

is unbiased entrywise: conditional expectation of I_s I_t/π_st is one. No Gaussian, independent-value, independent-corner or independent-background assumption is needed. The source mask is drawn after fixing those values. Zero pair inclusion destroys this proof for arbitrary nonzero pair contributions.

For independent Bernoulli source inclusion p_s, π_st=p_s p_t for s≠t but π_ss=p_s. Defining Y_k=Σ_s I_s f_{k,s}/p_s gives the executable correction

Ghat_{kl}=〈Y_k,Y_l〉−Σ_s I_s(1−p_s)/p_s² 〈f_{k,s},f_{l,s}〉.

The same-source correction applies to all nine branch pairs, including k≠l. It is different from omitting off-diagonal branch errors. Simply squaring the unbiased Y sum adds the nonnegative bias Σ_s(1−p_s)/p_s ||Σ_k f_{k,s}||². Negative corrected estimates can occur; clipping them introduces bias. Entrywise unbiasedness entails neither positive semidefiniteness per draw nor a unique/minimal recovered factorization.

A second outer average over independent draws of a specified shared-corner law gives an unbiased population J estimate if the relevant moments exist. Merely averaging this fixed eight-context control estimates its conditional mask expectation, not population integration error. A candidate chosen on these same masks needs independent audit randomness; fixed-candidate unbiasedness is not post-selection calibration.

The comparison theorem is [Isserlis (1918), On a Formula for the Product-Moment Coefficient of Any Order of a Normal Frequency Distribution](https://doi.org/10.1093/biomet/12.1-2.134), [primary scan](https://zenodo.org/records/1431593/files/article.pdf). Pairing identities integrate products of jointly Gaussian linear variables using covariances. They do not generally apply after random RMS division. The earlier current-value, unnormalized contraction used degree-four query and degree-six shared-source moments; separable contraction additionally factorized selected dependencies. Shared Gaussian variables themselves are allowed by Isserlis when all covariances are retained. Here normalized projections are non-Gaussian, their random denominators depend on the same raw vectors, the query/source can coincide, inherited values and z can correlate, and nonlinear additive-corner defects share all of these variables. Independent Gaussian contraction therefore changes the objective, not just its estimator variance. Frozen-RMS polynomial controls or a specially justified spherical law are different restrictions, not licenses to discard these dependencies. The older 1.10560724 distortion result supplies no bound between those objectives.

The tensor-network search also opened [Ferris, Unbiased Monte Carlo for the age of tensor networks](https://arxiv.org/abs/1507.00767) and [Huggins et al., Monte Carlo Tensor Network Renormalization](https://arxiv.org/abs/1710.03757). Their stochastic contraction route is relevant to unbiased approximation of a fixed multilinear network. Conditional on ω our finite pair network is already such a contraction, so the elementary pair estimator suffices. Replacing a network total by a noisy estimate and then taking reciprocal RMS or a ratio is outside that multilinear guarantee. No bond-rank, variance or uniqueness guarantee from those works is asserted for the normalized producer. Analytic Gaussian contraction is exact for its restriction; the sampling theorem is exact in expectation for this conditional restriction; neither identifies a smaller native circuit.

## Executed consequence and falsifier

[Control source](normalized_pair_ht_control_20260914_0256.py) and [new primary receipt](NORMALIZED_PAIR_HT_CONTROL_20260914_0256_RESULT.json). The filename suffix is a unique control identifier; the actual execution timestamp is 2026-09-14T02:51:47.477189+00:00.

The control reuses project/from_projections/additive, the retained components oracle, execute and the actual target builder. It loads the restored checkpoint on CPU and changes the builder's old checkpoint path only in process memory. No primary code or receipt was overwritten.

Predictions were in the source before execution: expected full Gram replay ≤1e-10 relative; naive bias agrees with the analytic correction ≤1e-10 and is positive; a fixed scalar normalized witness has >1% bias if RMS is computed from the sampled write. All passed. There was no fitted representation or selected winner.

Eight contexts, five sources, seed 17021456. X and independent perturbation noises are standard Gaussian, Δ_c=.1 noise, Δ_r=.2 noise. The first-layer raw port is .6X+.8 independent noise; its RMS-normalized current value is read by actual layer0 head2 V. Background z=.7X_t+.3X_first,t. Final RMS is supplied as sqrt(mean(z²)+ε), not generated by the native suffix. These choices create explicit shared dependencies without claiming that they model native intervention trajectories. Error direction is fixed E=.01T times alternating ± head-coordinate signs, chosen before outcomes.

All 32 masks are enumerated with p=(.25,.4,.55,.7,.85); expected source count is 2.75. This removes sampling uncertainty from the unbiasedness test for this panel.

| Quantity | Executed value |
|---|---:|
| Retained producer replay | 1.1072e-16 relative |
| Existing component-oracle replay | 8.9989e-17 relative |
| Expected complete Gram replay | 2.3132e-16 relative |
| Maximum absolute entry discrepancy across contexts | 2.8422e-14 |
| Naive bias formula replay | 6.5976e-16 relative |
| Exact mean squared error | 58.9583869463 |
| Naive expected squared error | 86.4134291474 |
| Naive upward bias | 46.5668% |
| Gram bias after sampled-write RMS substitution | 1.2020% relative |
| Fixed scalar RMS-substitution witness bias | 65.6390% |
| Relative mask standard error, one independent mask/context | 26.5456% |
| Internal execution time, two CPU threads | 0.3554 seconds |

The scalar witness uses z=1, two source contributions (1,2), p=(.5,.5), and denominator (1+sum contributions)²+ε. Its pair correction is unbiased with the full denominator but not when that denominator uses the sampled sum. This is an executed counterexample to the tempting normalization shortcut.

The native-weight panel also exhibits negative corrected energy with probabilities up to 1.5% in individual contexts. Approximately 705 independent masks per context would reduce the measured mask-only standard error of this fixed-panel mean to 1%; this is variance scaling, not a confidence guarantee, population sample-complexity theorem or measurement of native fidelity. No outer-context integration/seed stability was established.

Reproduction: CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 /home/loganriggs/.local/share/bilin18/venv/bin/python followed by the control path. The entry point refuses to overwrite its receipt. For read-only rerun, import and call run() in a fresh process.

## Total price and decision

The estimator changes no model nodes, edges, parameters or intervention generators. Exact T costs 7,077,888 bytes at FP32. The fixed sparse alternative has 4,896,936 packed tensor bytes, 4,899,795 serialized bytes and 9,340,736 CSR resident bytes; all 12 output, 1152 residual and 128 head nodes remain. Masks, adapters and decoding are part of those local prices.

For an explicitly expanded raw-port conditional interface, five current projection maps, layer0 value map, writer and μ add 1,032,193 scalar values (4,128,772 FP32 bytes). With dense T the subtotal is 11,206,660 bytes; with the existing sparse packed payload it is 9,025,708 bytes. These subtotals include the writer once and assume the layer0 value generator receives its own raw port. They do not close the upstream trajectories, child/remainder definitions, background, normalization generators, remaining MLP/readout paths or other consumers. Whole-program price is this subtotal plus the union of those required external dependencies, deduplicating shared weights. No smaller closed whole-model total was measured; retaining the native model as generator retains its 545,902,902 parameters, plus any additional stored adapters, rather than granting them for free.

The alternative local native L/R/C/LW/RW representation costs 47,407,104 bytes, but may reuse factors already needed by other paths. Conditional byte savings cannot all be charged as whole-model savings. The exported direct head core alone stores 147,457 scalars and consumes 1920 projection +4 norm/cross +128 first-value scalars per source. Input state, decoding workspace and all three context executions remain charged. No new intervention cost reduction was executed.

Compute matters more than an attractive stochastic formula here. Once ports and the full d are known, form B_{oh}=Σ_i E_{oih}z_i in O(12×1152×128), sum the three head contractions in O(3S×128), then apply B and form the nine Gram entries in O(3×12×128+9×12). This exact route does not materialize a source-pair tensor. The HT source-output implementation adds O(3m×12×128+9m×12) per sampled subset after B, plus full denominator generation or its exact supplied value. General pair enumeration would be O(9m²×12), but the Bernoulli correction avoids it. Storage can stream three output totals and a 3×3 correction after producer/reader preparation.

Consequently, source sampling is a valid diagnostic and a possible tool for an expensive implicit source producer, but it is not an established saving on this already cheap source contraction. It cannot avoid full-context RMS computation by substituting a noisy write. For current short prefixes, exact source summation followed by outer shared-context sampling is the preferable objective implementation. That outer sampling still needs a declared measure, integrability and independent integration-error audits before it selects a fit.

This review advances computation specification and composition/objective correctness. It does not establish held-out/OOD prediction, semantic grouping, autonomous extraction, selective manipulation, stable identification, lower model storage or faster execution. The next bounded research action is to audit outer-context integration stability of the exact-source normalized objective for frozen weight-only candidates at matched total cost. Do not repeat the old separable Gaussian or output-fit probes, and do not start optimization on a biased sampled-RMS objective. No further action, agent, GPU job, timer/queue change, commit, push or contact was launched by this review.
