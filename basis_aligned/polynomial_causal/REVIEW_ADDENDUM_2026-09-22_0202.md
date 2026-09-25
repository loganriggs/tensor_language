# Review addendum: conditional symmetry depends on the reference law

Written 2026-09-22 02:02:12 UTC. This is an addendum to the concurrent [01:59 substantive review](THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-22_0159.md), not a new three-hour review. **The review clock remains 01:59 UTC; next deadline 04:59 UTC.**

The latest review was 21 September22:53 both at the initial check and at the immediate pre-publication check (01:58:30, age185.51minutes). During drafting, another process published the substantive01:59review. The post-write verification detected it at02:01:29. My provisional0158file was withdrawn into this uniquely named addendum before the board append. The concurrent review and all primary receipts remain untouched.

Genuinely new work: an independently executed planted control of symmetry restoration under shifted versus symmetric-mixture Gaussian laws; actual literature searches/opened sources; the boundary, baseline and reconciliation analysis below. The control passes but provides no native causal fidelity. The user-directed direct decomposition focus remains authoritative through15:10UTC. No GPU job, agent, goal, queue/timer change or commit was made.

## Native objects, boundaries and price

Use zero-based layers ℓ=0,…,17, positions s≤t, residual indices i,j=1,…,1152, MLP channels a=1,…,4608, head h=1,…,9 and head coordinates α=1,…,128. Embedding E has shape 50304×1152; untied output U has the same shape. Native parameters total 545,902,902. Define N(x)=x/sqrt(mean(x²)+ε), ε=2^-23 in native FP32. With e_t=N(E[token_t]), a block computes

\[
u_t=\lambda_{\ell0}r_t+\lambda_{\ell1}e_t,\quad
v_t=u_t+A_\ell(N(u))_t,\quad
r^+_t=v_t+D_\ell[(L_\ell N(v_t))\odot(R_\ell N(v_t))]+b_\ell.
\]

L,R∈R^(4608×1152), D∈R^(1152×4608), b∈R^1152. Attention projects two separate Q/K pairs and values, head-normalizes each Q/K, and applies position rotations with native BF16-rounded sine/cosine tables. For each head,

\[
A_{t}=O_h\sum_{s\le t}
\frac{\langle q^1_t,k^1_s\rangle\langle q^2_t,k^2_s\rangle}{128^2}
\big[(1-\gamma_\ell)V_h N(u_s)+\gamma_\ell v^0_{s,h}\big].
\]

O_h∈R^(1152×128); Q,K,V head maps are 128×1152. The layer-0 value producer v⁰ is shared across consumers, with signed learned γ, not a convex mixture. Residual λ are independent learned scalars. Final outputs are 30 tanh(U N(r_final)/30). There is no attention softmax and no MLP SiLU. Norms and final tanh mean the full model is **not a polynomial**. With normalizers exposed as ports, an MLP numerator is quadratic and the current-value attention numerator has five linear input slots; first-value terms have a separate shared producer. Rounded rotations must remain part of replay semantics. See [architecture contract](BILINEAR_RECONSTRUCTION_ARCHITECTURE_CONTRACT.md).

Residual expansion must precede attribution. Write r=∑_u c_u s_u, with named embedding/re-entry, earlier attention and MLP writes. Then a downstream bilinear form is ∑_(u,v)c_uc_v B(s_u,s_v): embedding/self, attention/self, MLP/self, and both ordered attention×MLP terms all occur. Each QK has its own ordered source pair. The full attention contribution sums products B1(s_u,s_v) B2(s_w,s_z) V(s_k), retaining all five source slots and first-value sharing. Separate QK analyses or a closed Möbius identity cannot establish small interactions.

**Active folded object.** Let Bℓ(x)=Dℓ[(Lℓx)⊙(Rℓx)] and let W∈R^(1152×16) be the fixed output writer/reader geometry. After absorbing the actual scalar path gains, the selected homogeneous branch is F(x)=WᵀB17(B16(x)), x∈R^1152. Its symmetric coefficient tensor H has shape 16×1152^4. The contraction graph is x→two linear reads→4608 products→D16→two MLP17 reads→4608 products→WᵀD17. All four tensor slots receive the same x; compare Sym(H), not arbitrary unsymmetrized representatives. The branch retains only the MLP16-producer×same-producer term in MLP17; it omits producer×background, background×producer, background/self, biases, attention17 interactions and explicit normalization. Those remain native dependencies for an installed edit, not free closures.

The CP512 parent is ∑_(a=1)^512 C_ga∏_(k=1)^4 f_kaᵀx. Conditional compression retains rank256 Gaussian input coordinates. Full conditional atoms add six covariance-weighted quadratic terms and a constant. Lean atoms keep only the two pair corrections whose products are already used in the quartic; other four corrections are omitted. Affine factors make the conditional class degree≤4, not a homogeneous quartic. Four factor scalings with compensating writer scale, permutations and invertible internal basis changes produce equivalent functions; recovered coordinates are not identified mechanisms.

Norms are distinct: coefficient Frobenius on Sym(H); isotropic Gaussian E||ΔF(x)||²; shifted/covariance-informed Gaussian; empirical normalized-state relative L2; finite edit and derivative-response error. Gaussian laws use calibration statistics and are data-informed. Small error in one does not bound another without an explicit distribution/metric relation. Native ε means normalized inputs are not an exact fixed-radius sphere.

| Program, same 1152-input/16-output boundary | Stored floats including fixed 1152×16 writer | Variable products | Linear coefficient multiplications, excluding common writer |
|---|---:|---:|---:|
| CP512 parent | 2,385,920 | 1,536 | 2,367,488 |
| Full conditional rank256 | 850,960 | 3,584 | 830,464 |
| Lean conditional rank256 | 848,912 plus one pairing selector | 1,536 | 828,416 |

FP32 payloads are respectively 9,543,680; 3,403,840; 3,395,648 bytes before metadata. Parent evaluation forms 2048 factor values; conditional evaluation also forms 256 projected coordinates; full conditional forms six pairs per atom, lean two. Streaming can reduce live state, but no measured peak-memory claim is inferred. The native two-MLP weight inventory is 31,850,496 matrix values plus 2304 biases before writer/upstream costs; this is a full-module inventory, not an optimized same-output folded baseline. One native MLP has 4608 products. A model-free selected polynomial still consumes a **native normalized 1152-state port**; generating it and applying native denominator/background/suffix are separate costs. No quantization claim.

**Circuit object and five-property score.** The regional depth object remains the extracted city/attention8/MLP8 value-mediator chain and its residual/query/normalization interfaces; it is separate from this quartic fit. The graph registry lists 49 packages, 44 manifests and two records labelled four-trait verified; that schema label does not discharge measured simplicity or stricter matched-null requirements.

| Property | Current conditional quartic | Regional mediator records |
|---|---|---|
| Predicts held-out/OOD | Parent fresh FineWeb test exists; conditional comparisons reuse opened 256 documents/16,384 states. No text OOD result | Scoped fresh/transfer receipts exist; graph records retain prediction or composition gaps depending on boundary |
| Extracted | Exported model-free polynomial at declared x port; full native path not extracted | Explicit regional package boundaries; several still require native arrays/suffix |
| Selective manipulation/removal | Not established; parent selectivity failed and conditional finite-removal jobs are pending | Local evidence exists; three unrelated controls and matched nulls must be checked per record, not inherited |
| Composition/reuse | Arithmetic pair reuse exact; behavioral composition untested | Composition gaps remain; closed faces do not certify small interactions versus smallest piece |
| Simple | Priced smaller linear program; CPU speedup measured, all 512 atoms retained | Port/storage prices must accompany each extraction; no blanket adoption |

Fresh evaluation units were 256 distinct documents, one prefix each, not 16,384 independent contexts. Parity audit used 256 position-31 states. Root-response panels and grouped strata are not additional independent documents. This review's planted CPU control has **zero text context cells**.

## LITERATURE_SEARCH

Actual queries this review:

1. `Isserlis 1918 formula product moments normal frequency distribution Biometrika`
2. `Gaussian conditional expectation active subspaces Poincare bound Constantine 2014 polynomial`
3. `Oseledets tensor train decomposition 2011 theorem 2.2 error unfolding ranks pdf`

Opened primary sources and mappings:

- [Mamis, generalized multivariate Stein lemma, Theorem 1 and Corollary 2](https://arxiv.org/html/2202.00189v3): Gaussian polynomial moments exist; pairing identities apply to the four discarded Gaussian affine-factor noises. They produce the six quadratic corrections and three constant pairings exactly. This validates the conditional parent algorithm, not a Gaussian model of actual text. Construction contracts discarded factor covariances; evaluation uses O(dr+4Kr+6K+oK) arithmetic for d=1152,r=256,K=512,o=16. No uniqueness of factors follows.
- [Constantine, Dow and Wang, section 3.1/Theorem 3.1](https://arxiv.org/pdf/1304.2070), plus its [abstract](https://arxiv.org/abs/1304.2070): conditional expectation is the least-squares function of retained coordinates; the conditional Poincaré bound controls error by discarded gradient energy. Here whiten the frozen Gaussian and sum the 16 output gradient Grams. Standard Gaussian inactive directions have constant one. Computing a dense d×d eigensystem costs O(d³), separate from constructing its Gram. The guarantee is for the fixed parent and law, not the native model or arbitrary non-Gaussian text. Repeated eigenvalues leave retained bases nonunique. This motivates the executed law/symmetry control below.
- [Oseledets, Tensor-Train Decomposition, primary paper PDF](https://users.math.msu.edu/users/iwenmark/Teaching/CMSE890/TENSOR_oseledets2011.pdf): unfolding ranks and sequential SVD supply coefficient-Frobenius tensor approximations. Map to the finite five-mode H, with input symmetrization explicit, not directly to normalized model behavior. Dense H would require 16·1152⁴ entries; no such dense construction is justified here. TT core gauges remain, and this algorithm supplies neither semantic identifiability nor a Gaussian/text-error guarantee. It keeps a conventional same-boundary baseline on the missing-run ledger rather than replacing current work with an infeasible dense SVD.

Access failure: the [original Isserlis publisher page](https://academic.oup.com/biomet/article-abstract/12/1-2/134/193428) redirected to an inaccessible CDN and returned a tool error. Its full text was **not** read; Mamis's opened primary derivation supplies the pairing formula. Search was performed, not marked complete by reference to an older review.

## Executed mathematical consequence

The following is a derivation in this review, not a theorem attributed verbatim to the papers. For an even homogeneous quartic f and X∼N(μ,Σ), g₊(y)=E[f(X)|PX=y] need not be even. Under the balanced mixture of N(μ,Σ) and N(-μ,Σ), let m=Pμ and K=PΣPᵀ be nonsingular. Bayes' rule gives

\[
\pi(y)=\operatorname{sigmoid}(2m^\top K^{-1}y),\qquad
g_{mix}(y)=\pi(y)g_+(y)+(1-\pi(y))g_+(-y).
\]

This is even because π(-y)=1-π(y), but generally **not polynomial**. Uniform even projection is the mixture conditional mean only under additional conditions, e.g. m=0. Singular K requires restriction to its support; no inverse is presumed there. Thus imposing parity by averaging two outputs is not the same as conditioning under a symmetric law. Both lose unrestricted homogeneous-quartic equality in general.

Executed planted example: f(x)=x₁³x₂, μ=(1,2), Σ=I, P=(1,0). Exact shifted conditional is 2y³; uniform even projection is zero; balanced-mixture conditional is 2y³ tanh(y). Independent integration gives absolute squared risks:

| Evaluation law | Shifted conditional | Uniform even projection | Mixture conditional |
|---|---:|---:|---:|
| N((1,2),I) | 76 | 380 | 82.9052259177 |
| Balanced ±μ mixture | 684 | 380 | 82.9052259177 |

The native conditional helper was tested without modification. Full-rank recovery, independent discarded-coordinate Gauss-Hermite integration, reciprocal factor sign/scale gauge, centered positive control and posterior-mixture identity all pass; maximum identity error 2.85e-14. Risk integration at orders128/192 differs by 2.38e-12. CPU process completed in 0.70 seconds. [Control source](REVIEW_SYMMETRY_CONTROL_2026-09-22_0156.py), [receipt](REVIEW_SYMMETRY_CONTROL_2026-09-22_0156.json).

Decision: retain the measured conditional-compression result as law-specific. Do not infer implementation failure from parity loss, and do not adopt a tanh mixture as a free quartic repair: it introduces a new nonlinearity, law assumption and execution price. A future symmetry-constrained comparison needs a declared symmetric evaluation law and matched cost. This cheap falsifier is more informative now than another unregistered native fit; it leaves running experiments intact.

## BASELINE_COMPARISON

Apply [DECOMPOSITION_BASELINES_2026-09-20.md](DECOMPOSITION_BASELINES_2026-09-20.md): equal outputs, ports, precision and error measure before comparing costs.

- Native factorization is the exact selected-branch reference; retain its bias/normalization/background when comparing full edits. The CP512 parent is a fitted baseline, not native truth. Lean versus full versus parent is a valid common 16-output/input-port FP32 execution comparison; CPU timing does not establish GPU speed.
- Spectral input-subspace conditioning is measured on this parent; it is not the coefficient-optimal Tucker decomposition of native H. A direct same-price native HOSVD/Tucker/TT/alternative-tree frontier at this 1152×16 boundary remains missing in the reviewed receipts.
- Existing [HT benchmark](NATIVE_TWO_MLP_QUARTIC_HT_2026-09-20.md) concerns MLP11→12 with five source-amplitude inputs and four outputs, 16 context groups. Rank8 costs656 values with9.35% coefficient error versus280 exact canonical coefficients. It cannot be transferred numerically to this MLP16→17 target. Planted sparse Tucker success likewise is a positive implementation control, not a native baseline win.
- Shared144-producer/512-root model uses1088 products and1,353,728 floats plus1024 indices; 11-step learned fits give12–13% opened error but fail component gates. CP256 parent refit costs768 products/1,202,176 coefficients, <1% parent Gaussian error but about15% native response error. These measured tradeoffs have unequal capacities; they do not establish optimizer superiority at matched cost.
- Calibration constant/isotropic controls, matched-cost random retained subspaces, balanced-gauge factor SVD and native full-rank references must share this precise target. No newly matched runs were executed here; historical different-boundary controls are not substituted. The planted control compares identical ports and scalar output, but adds a priced tanh for mixture conditioning, so it is not a native cost victory.

## REDTEAM_POSITIVE

Strongest new success: lean rank256 preserves much of the parent's opened value accuracy while reducing linear arithmetic; CPU speed is1.48–2.46× across two seeds and batch1/64/2048. Limits: all512 atoms survive; normalization/context and fixed writer are still needed; no dense fallback is present in the inspected conditional evaluator, but the upstream native state is external. Calibration covariance/mean are data-informed; rank selection used opened rows. The originally fresh256-document panel is now opened for conditional selection. Synthetic pairing-selection and held probes are distinct; synthetic probes are not text OOD.

Existing parity audit breaks a global-polynomial reading: lean rank256 error rises from8.4% to13.8–13.9% on negated position31 states. Negation preserves norm but is not demonstrated text reachability. Small-output errors and native response/sensitivity failures rule out treating pooled7.2–7.5% as uniform component fidelity. Constants, lower-degree corrections and background costs are explicit. No unrelated-damage or behavioral-reuse pass is inferred from timing. The executed control further falsifies the claim that an even target forces its optimal shifted-law approximation to be even.

## REDTEAM_NEGATIVE

Strongest relevant negative: conditional parity loss and failed native response retention. The positive planted full-rank and independent quadrature controls rescue **algorithm correctness**, not native fidelity. Sign/gauge and centered controls separate a structural shifted-law effect from an axis or bias bug. Uniform parity restoration can worsen the intended law's loss by5× even with exact arithmetic. Conversely a symmetric-law optimum improves over naive averaging; failure of averaging does not disprove symmetry-aware methods.

For native shared producer failures,11 updates and continuing objective improvement are not convergence or impossibility certificates. CP parent refits selected step100; longer optimization remains unresolved. The CPU toy does not rescue their native finite effects, and no failed original receipt was rewritten. Precision/export controls already pass; current failure should not be renamed a known numerical bug without new evidence.

## Organization, efficiency and actionable handoffs

Read the complete research skill/startup/prompt, user authorities/focus, latest commits and dirty status, board protocol/tail, current hourly reviews, graph/circuit/path registries, module indexes and MLP16/17/readout dossiers. Live checkout paths resolve under /workspace/tensor_language. Startup's workstation/systemd and September13 LATEST/clock instructions are stale; current Supervisor observations take precedence. Theseus-bench status was clean. Main checkout has concurrent tracked/untracked runner and research changes; all were preserved. TYPED_FACE_EXTRACTION_V1 was untouched.

Reconciliation: circuit/module registries contain the older source/edge chain; computation-path registry and direct study README trail newer September22 conditional work; individual MLP16/17 dossiers retain September12-era folding facts. Explanation README still leads with September21 22:13 and labels September20 evidence as latest. These are navigation debts, not orphan experiments or new aliases. CP512→full conditional→lean conditional are successive representations of the same selected branch, **not three new circuits**. Regional typed-face/value mediator and subject-number head3 records remain separate. Canonical current pointers are the conditional, lean, symmetry and parent-refit interpretation/primary receipt pairs below; preregistered full/lean removal plans correctly mark opened selection. No missing native removal result was invented.

No shared-file repair was performed by this reviewer after the executed consequence; the reconciliation here records the inspection snapshot. Full and lean helpers look duplicative but are intentionally frozen for queued jobs, and changing them now risks invalidating source hashes. A later declarative pair-mask evaluator could share core arithmetic only after jobs terminate and masked-full replay is preserved. No new runner/data builder/scorer or publication framework was created. Existing exact helper was reused in the unique small control.

Measured efficiency: shared producer learning1205s; CP parent refit13.94s; fresh capture1.938s and scoring2.655s. Cached larger panels are cheap enough that tiny row counts are not a runtime necessity. At inspection both Supervisor runners were RUNNING; lane1 log showed balanced shared features active since01:30:20. Full/lean removal jobs sit behind six external-model jobs. This is shared-queue latency, not evidence of idle hardware; no queue or timer action taken. Native helpers and pending sweep files remain owned by their primary workers. Precise thinking/ceremony fractions are not reconstructible from receipts; no invented timing allocation.

**Folding handoff:** interpret balanced producer and full/lean finite-removal receipts against their original all-cell gates before any expansion. Preserve parent Gaussian, isotropic/coefficient, weak-feature and actual finite-effect metrics separately. The control recommends law-aware symmetry constraints, not immediate nonlinear mixture deployment. Missing same-boundary conventional/matched-cost controls remain on the frontier ledger.

**Circuit handoff, deferred during direct-decomposition focus:** circuit response evidence selects which fixed reader/self/cross terms deserve compression, but prior regional forward census shows the selected MLP16×MLP17 branch can oppose the total effect. Do not equate the quartic with the regional mechanism. Folded shared pair computations suggest testing groups of reused quadratic features versus equal-cost/random splits after freezing, with full QK1×QK2×V interactions, exact ports and a forward response census before proposing a suffix. Require held-out/OOD effects, at least three unrelated-reader controls, matched-norm nulls and composition error relative to the smallest piece. Algorithmic pair reuse alone does not pass these gates.

### Concurrent organizational correction

The 01:59 primary review reports that its worker repaired the decomposition README, explanation index, computation-path registry and MLP dossier while this addendum was being drafted. The stale-index findings above describe my earlier read snapshot, not an instruction to repeat those repairs. Its receipt owns those changes. Do not overwrite or duplicate them. This reviewer wrote only this addendum, a short board append, and the uniquely named CPU control source/JSON.

## Receipt anchors and limitations

All paths below are in `direct_tensor_match/`; hashes are SHA256 prefixes, recorded after reading primary JSON rather than trusting historical LATEST pointers:

| Primary receipt | Hash prefix | Scope |
|---|---|---|
| [CONDITIONAL_CP_PROGRAMS_V1.json](direct_tensor_match/CONDITIONAL_CP_PROGRAMS_V1.json) | e8c6b41ade1994f1 | full conditional parent-law/output comparison |
| [LEAN_CONDITIONAL_CP_V1.json](direct_tensor_match/LEAN_CONDITIONAL_CP_V1.json) | 4c57b6a477ed1990 | priced two-pair approximation |
| [CONDITIONAL_CP_CPU_BENCHMARK_V1.json](direct_tensor_match/CONDITIONAL_CP_CPU_BENCHMARK_V1.json) | eb5a8e17bddb5420 | CPU only |
| [CONDITIONAL_SYMMETRY_AUDIT_V1.json](direct_tensor_match/CONDITIONAL_SYMMETRY_AUDIT_V1.json) | bf838d2eda6d22d0 | opened algebraic stress |
| [RESIDUAL_FRESH_SCORES_V1.json](direct_tensor_match/RESIDUAL_FRESH_SCORES_V1.json) | 9cb9466dc90483df | original five frozen candidates,256 documents |

Current interpretations: [full conditional](direct_tensor_match/CONDITIONAL_CP_PROGRAMS_INTERPRETATION_V1.md), [lean](direct_tensor_match/LEAN_CONDITIONAL_CP_INTERPRETATION_V1.md), [shared producers](direct_tensor_match/SHARED_MIXED_FEATURES_INTERPRETATION_V1.md), [parent refit](direct_tensor_match/CP_PARENT_REFIT_INTERPRETATION_V1.md). The new control source hash is231227f2426613bc; imported helper054c77a43a43fc2aa. Full hashes and numerical residuals are in the new JSON.

No algebraic replay, law-specific compression, small norm error, model-free packaging or timing result in this review establishes native selective causal fidelity. The bounded review ends after its executed CPU consequence and receipt.
