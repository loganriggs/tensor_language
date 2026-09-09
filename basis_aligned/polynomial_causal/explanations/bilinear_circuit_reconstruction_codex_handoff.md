# Exact compositional reconstruction of bilinear transformers

Standalone research handoff for Codex · Prepared for Logan · 9 September 2026

## Recommendation and initial assignment

Investigate whether a trained bilinear transformer admits a smaller executable organization of shared state updates, nonlinear intermediate computations, and output readers. Begin with exact algebra and a bounded falsification study. Use unlabeled data to propose useful reductions and measure approximate ones. SAEs are optional later candidate generators; they are not the starting point.

The end goal is a circuit program that predicts the model's full output distribution on held-out and OOD inputs, executes from tokens without unexplained cached activations, supports accurately predicted removal and interchange effects, and reuses intermediate computations across consumers. Recovering an is-versus-was contrast alone does not meet this goal. Neither does compressing a tensor without explaining its execution.

The first decision is whether this direction produces a real simplification beyond an exact change of representation. The recurrent attention representation is already known in the linear-attention literature. It is useful machinery, not the novel result. At the current model's short context lengths, materializing its unreduced recurrent state can increase both memory and arithmetic substantially.

Codex should complete the bounded pilot below, preserve negative results, and recommend continue, narrow, or stop. Do not launch a broad training sweep, a new SAE campaign, or whole-model symbolic expansion. Keep code changes small. Use available local code and checkpoints; this handoff does not depend on a GitHub connection.

## Architecture and facts that must be established

The intended family has bilinear MLPs of the form

\[
F(x)=D[(Lx)\odot(Rx)],
\]

and two-QK product attention of the form

\[
o_t=O\sum_{s\le t}(q_{1,t}^{\top}k_{1,s})(q_{2,t}^{\top}k_{2,s})v_s.
\]

The user also specifies repeated access to the original token embedding and a shared first-layer value stream. Write their mixing schematically as

\[
u_{\ell,t}=a_\ell r_{\ell,t}+b_\ell e_t,
\qquad
v_{\ell,t}=\alpha_\ell\widetilde v_{\ell,t}+\beta_\ell v_{0,t}.
\]

These symbols are placeholders for the actual implementation. Do not assume complementary coefficients, scalar rather than vector coefficients, identical conventions at every site, or a particular meaning of lambda.

Before adapting a real model, inspect and record:

- The exact attention formula, including all scaling, masking, denominators, and query/key normalization.
- The ordering and epsilon values of RMSNorm, projections, and RoPE.
- Whether the attention mask is a full causal prefix, a fixed window, or another pattern.
- The exact residual update and original-embedding injection sites and coefficients.
- How the first value is formed, whether it is token-only, its head indexing, and how each later head accesses it.
- Final normalization, logit scaling or softcapping, embedding and unembedding tying, and dtype.
- Whether dropout or other stochastic operations must be disabled for evaluation.

The provided historical notes describe an 18-layer model with 9 heads, residual width 1152, and MLP hidden width 4608. The cost example below additionally assumes both key widths and value width are 128. Verify dimensions from the checkpoint before using those numbers as model-specific facts.

The notes report a useful dictionary over folded QK factors, a separate retracted merge-then-dictionary experiment, and successful use of the shared first-layer values. They do not establish the exact previously attempted three-dictionary to Tucker construction. Do not describe that experiment as replicated or settled without its code and receipts. Previous negative results include whole-layer tensor compression and failure of independently assessed replacements to compose.

## Exact program representation

For one head define an accumulator

\[
M_t=\sum_{s\le t}k_{1,s}\otimes k_{2,s}\otimes v_s
\in\mathbb R^{d_1\times d_2\times d_v}.
\]

Then

\[
M_t=M_{t-1}+k_{1,t}\otimes k_{2,t}\otimes v_t,
\qquad
o_t=O\sum_{a,b}q_{1,t,a}q_{2,t,b}M_t[a,b,:].
\]

The current token's update is included before its query read for a mask permitting self-attention. Update after the read if the actual mask excludes the diagonal. A minimal reference for a single time step, before output projection, is:

```python
# q1, k1: [batch, heads, d1]
# q2, k2: [batch, heads, d2]
# value:  [batch, heads, dv]
# state:  [batch, heads, d1, d2, dv]
state = state + torch.einsum("bhi,bhj,bhv->bhijv", k1, k2, value)
head_out = torch.einsum("bhi,bhj,bhijv->bhv", q1, q2, state)
```

This is a reference identity, not a proposed optimized kernel. It is exact over real arithmetic; reassociation changes floating-point rounding. Query/key normalizers remain in the producer functions. RoPE can be applied when producing queries and keys. A fixed sliding window needs removal of expired updates and sufficient history to do so; constant-memory claims below concern the full causal-prefix rule. If there is a separable score-sum denominator, represent its accumulator explicitly. Arbitrary pairwise normalization or masking requires a fresh derivation.

Processing each token through all layers gives a complete stateful program. Persistent state consists of these attention accumulators and any additional state required by positional or masking rules. MLPs and residual updates are computations within the token step. The shared first value is computed once and passed to its consumers.

Keep this as a DAG or stateful program. Do not expand all polynomial monomials, duplicate shared subexpressions, or materialize the full embedding-to-unembedding tensor. Compilation should also permit the original direct attention contraction as an execution backend when it is cheaper.

## Compute and memory analysis

Let B be batch size, L the number of attention layers, H heads per layer, T sequence length, d1 and d2 the two key widths, and dv value width. Counts below cover attention contractions and persistent attention state. They exclude projections, MLPs, model weights, output vocabulary projection, temporary workspace, and training gradients. One multiply-accumulate is one MAC, approximately two FLOPs under the usual convention.

| Quantity | Direct product attention | Recurrent accumulator |
| --- | --- | --- |
| Persistent inference state | B L H T (d1 + d2 + dv) scalars for two keys and one mixed value | B L H d1 d2 dv scalars |
| One decoded token after T cached tokens | About B L H T (d1 + d2 + dv) MACs | About 2 B L H d1 d2 dv MACs |
| Prefill with ideal causal triangular work | About B L H T(T+1)(d1 + d2 + dv)/2 MACs | About 2 B L H T d1 d2 dv MACs |
| Additional leading implementation concern | Dense or tiled score work; full masked GEMMs can compute the upper triangle too | State bandwidth and an efficient scan; a Python time loop is a poor GPU prefill backend |

Lower-order score products and construction of q1 tensor q2 and k1 tensor k2 are omitted. Direct implementations may reuse or avoid storing parts of values, so record actual allocations as well as this conventional baseline. Sharing v0 does not make all head memories identical: distinct key products weight the same values differently.

For B=1, L=18, H=9, d1=d2=dv=128, T=512:

| Quantity | Value |
| --- | --- |
| Recurrent state per head | 2,097,152 scalars |
| Recurrent state across model | 339,738,624 scalars |
| Recurrent state in FP32 | 1296 MiB, or 1.265625 GiB |
| Recurrent state in BF16 | 648 MiB |
| Conventional product-attention KV cache in FP32 | 121.5 MiB |
| Conventional product-attention KV cache in BF16 | 60.75 MiB |
| Recurrent contraction work per head per new token | About 4,194,304 MACs |
| Direct decode work at 512 cached tokens per head | About 196,608 MACs |

The recurrent contraction uses about 21.3 times the decode MACs at this context, and about 42.6 times the ideal triangular prefill MACs for a complete 512-token sequence. These are attention-contraction ratios, not whole-model runtime ratios. Projections and MLPs can dominate other parts of execution, and kernels determine wall time.

The approximate crossover lengths are:

\[
T_{\rm memory}\approx\frac{d_1d_2d_v}{d_1+d_2+d_v},
\quad
T_{\rm decode}\approx\frac{2d_1d_2d_v}{d_1+d_2+d_v},
\quad
T_{\rm prefill}\approx\frac{4d_1d_2d_v}{d_1+d_2+d_v}.
\]

They are approximately 5.5k, 10.9k, and 21.8k tokens for width 128. A prefill implementation that computes full square GEMMs changes the leading constant. Do not extrapolate the current checkpoint's behavioral validity to these lengths: the historical notes report degradation beyond roughly 512 tokens.

Training or naive autograd through the recurrence can save every M_t and consume O(B L H T d1 d2 dv) memory. Checkpointing, recomputation, or a custom scan/backward changes the implementation tradeoff. The first experiment is inference and algebra only; it requires no end-to-end training through a recurrent state history.

The discovery algorithms have a separate cost. With total state dimension N, a dense Gram or Jacobian uses O(N squared) memory, and a dense SVD is O(N cubed). At N near 340 million these operations are unavailable in practice. An arithmetic expression over n inputs of degree at most d may have binomial(n+d,d) monomials. Neither that representation nor a global Hankel matrix belongs in the pilot.

Use local blocks, matrix-free products, streamed small sketches, and strict expression-size limits. A sketch proposes low rank; it does not certify exact rank. For a candidate feature bank of width m on n samples, forming its full Gram costs O(n m squared) work and O(m squared) memory. Begin with m at most 256.

## What would constitute useful exact reduction

Write the whole step as S_next=T(S,e) and output logits as R(S,e), with time or positional state supplied explicitly when required. A proposed state map tau and smaller executor must satisfy

\[
\tau(T(S,e))=\overline T(\tau(S),e),
\qquad
R(S,e)=\overline R(\tau(S),e)
\]

on the declared domain. Corresponding identities must hold for any interventions the explanation claims to represent. Count the description, stored constants, execution, and state of tau and the reduced functions. Decoding the entire original state and calling the original executor at every step is not an execution simplification.

Search initially for linear shared state coordinates and exact shared update/read functions. Permit a small number of explicitly represented nonlinear intermediates in a second stage. Restrict their grammar and cost; unrestricted nonlinear encodings can hide the original computation. Minimum state dimension alone is not the goal: a larger state with a short reusable update may be a better explanation and a faster implementation.

A useful exact positive control is a token-only accumulator without positional dependence. Stack every head's update into g(u) in R^N for each vocabulary item u. Let U=[g(1),...,g(V)]. If P spans the columns of U and S starts at zero, then S=Pz exactly and

\[
z_{t+1}=z_t+P^{\top}g(u_t)
\]

for orthonormal P. Fold P into the actual readers rather than rebuilding S. Rank(U) is at most V. A still simpler baseline is a vector of token counts followed by the update/read tables.

This control must not be reported as discovery of semantic structure. It is a known consequence of token-only additive updates and will occur for random weights too. Positional rotations and context-dependent later-layer keys change the update family. A worthwhile result must improve on this count/table baseline or identify sharing in a contextual block where it does not apply.

The clean linear weighted-automaton result, that Hankel rank characterizes minimum linear state dimension, motivates the program but is not a theorem about the full nonlinear transformer. Constrained lumping similarly supplies a closure principle; an ODE implementation cannot simply be applied unchanged to layer-dependent normalized transitions.

## Unlabeled data and the permitted equality domain

Use three separate evidence levels:

1. Global algebraic equality of the stated real-arithmetic functions.
2. Equality on a declared set of valid inputs or reachable/intervened states.
3. Approximation under an explicitly stated distribution and intervention family.

Never turn level 3 into level 1 by calling a small singular value zero. Never infer a population identity solely from a finite sample's nullspace.

Valid token indicators satisfy X_tu X_tv = delta_uv X_tu and sum_u X_tu = 1. These identities permit exact folding of repeated functions of the same token. They do not remove interactions among different positions. They also do not hold for arbitrary continuous embedding patches. Keep the intervention domain attached to the rewrite.

Natural state relations can be searched using a small candidate function bank phi. For a scalar discrepancy f-g=c^T phi,

\[
E_{\mathcal D}(f-g)^2=c^T G_{\mathcal D}c,
\qquad G_{\mathcal D}=E_{\mathcal D}[\phi\phi^T].
\]

This is an exact expression for the specified discrepancy. It is not a guarantee on the end-to-end effect of a local replacement. Use teacher full-vocabulary distributions and complete-model recomputation to measure that effect; labels are unnecessary.

Split data by constructions or independent sequences before selecting features or rewrites. For a toy with known independent generative factors, reserve unseen factor combinations and longer sequences. Include a separate family of legal interventions. For the real model, use independent unlabeled corpora/windows and report length limits. Do not use a query-label-free procedure as a reason to skip held-out evaluation: teacher outputs and activation measurements can be overfit.

When the population support is unchanged, probability reweighting does not create new exact equivalences. It changes the approximation tradeoff. A polynomial that vanishes on an open continuous region is identically zero. Finite empirical support can create many spurious vanishing relations, especially with a large candidate bank.

## Red team assessment

**The reformulation may be the only success.** Recurrence is established linear-attention algebra, and the native network is already a compact DAG. Require a reduction in stored constants, arithmetic, or reusable operations beyond a competent exact compiler and the count/table baseline. Merely eliminating an intentionally expanded tensor does not count.

**Exact learned structure may not exist at a smaller size.** Generic trained coefficients may leave matrices full rank and functions algebraically distinct. Tiny numerical perturbations destroy exact shared identities. Include a perturbed version of every planted exact positive. If the algorithm keeps declaring exactness, the instrument is unsound. A negative on an accurately implemented contextual block is useful evidence; do not endlessly adjust rank thresholds.

**State compression may increase computation.** A dense state encoder or decoder can cost more than the eliminated memory, and a small tensor core can have expensive dictionaries. Price the complete executor. Report state bytes, constants, MACs, and measured latency separately.

**Shared payload does not imply shared memory or algorithm.** The same v0 is weighted by different key products. Build a control with identical values and independent routers; it must not be mislabeled as one reusable routing computation. Conversely, equivalent routing functions can act on different values, so track what is shared explicitly.

**The right abstraction may not be linear.** Full linear rank does not preclude a compact nonlinear recurrence. But unconstrained nonlinear maps make the reduction problem circular. First use a fixed low-degree, explicit grammar and a small search budget. If linear reduction fails, nominate one specific nonlinear hypothesis instead of starting unrestricted synthesis.

**Normalization can couple the discarded directions back in.** RMSNorm and query/key norms depend on quantities that a proposed subspace may omit. Frozen normalizer traces certify a conditional computation only. Keep norm evaluation inside the extracted program and recompute it after changes. A positive in a norm-free toy must be rechecked with the real convention before transfer.

**Value mixing or token injection can be folded incorrectly.** A shared value may already include a normalization or a learned map. Lambda may weight the opposite branch from the assumed one. Require modulewise agreement before analyzing structure. Do not reconstruct implementation facts from filenames or memory.

**Mask and positional rules can break the finite-state claim.** The recurrence above assumes a prefix sum. Sliding windows need expiry handling, and an arbitrary pairwise mask may have no small state. Absolute RoPE phases can be computed when keys are stored, but test the actual ordering and convention. A correct asymptotic statement for one mask is not a result for another.

**Dense rotations can be an invalid planted test.** Do not rotate through elementwise products or RoPE while leaving their operators unchanged. If a control uses a changed state basis, transform both update and read maps consistently and count any adapters. Known hidden state coordinates are a test fixture, not evidence that the native architecture has a free gauge at every site.

**Observational agreement can erase causal distinctions.** Two routes that happen to agree on natural inputs can respond differently to removal or interchange. Require the reduction to predict interventions before calling the merged variable a mechanistic circuit. Identical outputs from different algorithms remain a possible behavioral equivalence.

**Removal is not necessarily selective or monotone.** Shared computation has shared consequences; redundant computation can survive deletion; suppression can make adding components reduce fidelity. Recompute all downstream states after an edit. Do not call a single greedy deletion result globally minimum, or assume a small linear-response residual is harmless.

**A compiled edit may not be a native factor ablation.** Deleting a polynomial term is a well-defined edit to that compiled program. It need not equal zeroing any native neuron or weight slice. Report the intervention correspondence actually implemented. Test native and compiled edits only when a valid map is supplied.

**Numerical agreement is not an exact proof.** Reassociation, mixed precision, overflow, and cancellation can change outputs. Separate mathematical real-arithmetic identities, FP64 closure, and deployed-dtype fidelity. Exact rational fixtures can support coefficient certificates. For trained floating-point models, a tolerance check remains numerical unless an exact certificate is provided.

**Search can recreate the exponential expansion.** Distributive rewriting, global polynomial fitting, dense Hankel construction, and unrestricted equality saturation can all explode. Enforce node, degree, state, feature-bank, and wall-time caps before launching each operation. Treewidth controls classes of tensor contraction costs; bounded polynomial degree or a tensor-network label alone does not ensure tractability.

**A convincing toy can be trivial.** Token counts, duplicated heads, unused channels, and planted low-rank factors are necessary instrument controls. Success only on these controls does not establish recovery of learned algorithms. Include at least one contextual two-layer block and compare with a generic matched architecture.

## Bounded pilot and stop conditions

Initial implementation and search budget: at most two CPU-hours and, if already available, twenty minutes of GPU execution. Do not provision hardware. Cap each newly allocated analysis tensor at 256 MiB and the candidate function bank at 256 columns. Start with batch 1 to 8, two layers, two heads, d1=d2=dv=4, residual width 16, bilinear hidden width 32, vocabulary 8, and lengths 8, 16, and 32. These are diagnostic dimensions, not a scaling claim. Record wall time and stop on the cap rather than silently broadening the job.

**Phase A — establish faithful execution.** Implement direct causal product attention and the recurrent reference. Include original-token injection, shared-value mixing, bilinear MLPs, and the output head in a complete tiny executor. Turn positional and normalization features on separately, then together. Compare each module and full logits on deterministic sequences, random inputs where meaningful, nonzero mixed coefficients, and single-site interventions. Reconstruct recurrent state from modified source updates when an intervention changes past keys or values.

For unit-scale, well-conditioned FP64 fixtures, start with atol=1e-9 and rtol=1e-9; report maximum absolute error and relative L2 error. If the fixture is ill-conditioned, investigate and report it instead of changing tolerance until it passes. Rational no-normalization fixtures should additionally permit exact coefficient or identity checks. Do not use teacher accuracy as the compiler closure test.

Stop and repair the implementation if any supported operation fails closure. If a mask or normalization cannot be represented by the stated recurrence, record the unsupported case and derive the correct additional state or keep the original backend.

**Phase B — test whether discovery is sound.** Use four fixed fixtures: a planted shared update/read computation hidden by a valid dense state change of coordinates; shared values with independent routers; a generic contextual two-layer model; and a small nonzero perturbation of the exact positive. Keep the input/output behavior and intervention meanings explicit. Include the token-count accumulator as a separately labeled baseline.

Use exact update/read contractions and a local candidate basis. Search for shared linear update coordinates first, plus a bounded set of common factors across products. For each candidate, emit its proposed reduced update/read equations and validate closure. Recover equivalent subspaces or functions rather than insisting on the original planted coordinate axes. Independent routers may still share the trivial token-count representation; the algorithm must distinguish that fact from equality of the router functions.

Stop if positives fail, known perturbations are called exact, or the algorithm only rediscovers token counts. Fix a concrete instrument defect once if found. Do not start a broad hyperparameter sweep to rescue the result.

**Phase C — test meaningful transfer on one contextual block.** Reuse one existing trained small checkpoint if present. If no suitable checkpoint exists, complete the controls and report the missing real-model test; do not train a large model as an unannounced substitute. Limit analysis to a two-layer motif, retaining the full upstream computations needed for its inputs and all downstream computation used to assess the replacement. Label a motif with a native background as a partial extraction.

Generate candidates on one split of unlabeled sequences. Freeze them before testing independent sequences, unseen combinations where the task generator supplies them, and longer sequences inside the model's valid range. Test full output vectors/distributions, the registered interventions, and joint use by at least two consumers. Do not optimize only an answer-minus-foil margin.

A useful local result should reduce at least one complete cost measure by 25 percent against the best relevant exact baseline, while not increasing either of the other two measures—stored constants, state bytes, arithmetic—by more than 10 percent. These are proposed pilot decision bars, not theoretical thresholds. Always report the raw Pareto comparison even when a candidate misses the bar. Wall time is measured separately; a tiny unfused prototype can establish an arithmetic reduction without establishing a runtime improvement.

Exact candidates must pass the algebraic or declared numerical identity standard without data fitting exceptions. For an explicitly approximate candidate, freeze default diagnostic bars before selection: mean full-distribution teacher KL at most 1e-3 nats/token and 99th-percentile KL at most 1e-2 on both IID and designated OOD test sets; report the maximum as well. For intervention differences, report centered-logit vector relative L2 error, and use an absolute-error gate for near-zero targets. Start with a 1 percent relative error bar for non-negligible intervention effects, with its denominator floor fixed from calibration only. Failing these bars is a failed approximate candidate, not an exact result with a rounding exception.

**Phase D — choose the next action.** Continue if a contextual block shows useful sharing beyond trivial baselines and the proposed reduced program predicts unseen joint interventions. Narrow to distribution-dependent approximation if exact sharing is absent but a prospectively tested approximate program improves cost. Stop the current discovery method if it only changes representation, requires expensive native decoders, fails controls, or cannot transfer beyond the discovery inputs. A successful exact compiler remains useful infrastructure but should be reported separately from the research hypothesis.

## Benchmark protocol

Benchmark direct and recurrent execution at identical weights, dtype, batch, inputs, and outputs. Include both prefill and cached one-token decode. Measure with inference mode, warm up first, synchronize CUDA around timing, and report median and spread across repeated trials. Account for model weights separately from incremental state and workspace. Avoid attributing a Python-loop penalty or an unfused einsum advantage to the mathematical representation itself.

For tiny CPU controls, timing establishes practical feasibility only. On an existing GPU, report the model and software versions and whether each backend uses compiled or fused kernels. Measure at the actual validated context length before adding longer synthetic lengths. Separate compiler time, discovery time, verification time, and per-input execution time. Report timeouts and memory failures as such.

Record these comparisons: native/direct execution; unreduced recurrent execution; elementary exact folding and common-subexpression elimination; token-count or token-table representation when applicable; discovered exact reduced program; and any separately labeled approximate program. This prevents crediting the discovery method with gains from a deliberately weak baseline.

## Requested implementation outputs

Use existing repository conventions and reuse the established model loader. Prefer one small compiler/reference module, one bounded experiment runner, and one report rather than a new framework. Suggested outputs are:

- An architecture contract recording exact forward semantics and supported cases.
- A direct and recurrent reference implementation with a complete tiny token-to-logit execution path.
- An inspectable reduced program and its explicit native correspondence for every proposed intervention.
- A machine-readable result file containing shapes, seeds, hashes, domains, errors, costs, runtimes, controls, and verdicts.
- A short report with the cost table, any algebraic certificates, numerical checks, negative findings, and the continue/narrow/stop recommendation.

Keep any original checkpoint unchanged. A separately compiled or reduced artifact can be created and tested. Do not describe a calibrated or retrained artifact as an exact explanation of the original model. No broad model training, external compute provisioning, or publication is part of this assignment.

## Evidence status at the time of this handoff

This document contains derived operation counts and a proposed experiment; it does not report a trained-model reduction. A small NumPy check with 23 positions, key widths 4, and value width 5 verified the product-attention recurrence against the direct contraction with maximum absolute error 2.13e-14 in FP64. A synthetic token-update dictionary check reduced a 128-dimensional accumulator to an 8-dimensional span with reconstruction error 2.84e-14. The latter is the deliberately trivial count/table positive control, not a mechanistic discovery. No GPU runtime, actual checkpoint execution, or end-to-end OOD result was measured while preparing this handoff.

## Relevant literature

- [Katharopoulos et al — Transformers are RNNs](https://arxiv.org/abs/2006.16236). Kernel-feature attention admits associative and recurrent evaluation. This establishes the background for the accumulator representation; it does not establish interpretability or a smaller learned mechanism.
- [Ovchinnikov et al — CLUE](https://arxiv.org/abs/2004.11961). Exact constrained linear lumping for polynomial ODEs. Borrow its observable-preserving closure principle; derive the corresponding rule for the actual discrete transitions and normalization primitives.
- [Rabusseau et al — Low Rank Approximation of Weighted Tree Automata](https://proceedings.mlr.press/v51/rabusseau16.pdf). Relates multilinear compositional state to Hankel structure. Its minimum-state results apply to the weighted-automaton class, not arbitrary nonlinear transformer state.
- [Willsey et al — egg](https://arxiv.org/abs/2004.03082). Equality saturation supplies infrastructure for searching equivalent expressions. Use bounded local rewrites and an explicit DAG cost; it is not a general-purpose polynomial-time circuit minimizer.
- [Markov and Shi — Simulating Quantum Computation by Contracting Tensor Networks](https://arxiv.org/abs/quant-ph/0511069). Connects contraction cost with graph structure. It motivates controlling contraction width, rather than assuming all tensor-network representations are easy to manipulate.
- [Livni et al — Vanishing Component Analysis](https://proceedings.mlr.press/v28/livni13.html). Finds polynomial relations from unlabeled data. Such relations propose domain-dependent simplifications; finite-sample vanishing is not a population or intervention certificate.
- [Geiger et al — Causal Abstraction](https://jmlr.org/papers/volume26/23-0058/23-0058.pdf). Supplies the framework for requiring the abstract program to preserve specified causal manipulations, in addition to outputs.

## Immediate instruction to Codex

Read this handoff and the repository's applicable instructions. Locate the existing forward implementation and inspect its exact semantics. Run Phases A and B under the stated resource caps, then Phase C only if an existing suitable checkpoint and valid controls permit it. Deliver the executable references, cost and fidelity receipts, and a candid verdict. Prioritize falsifying a proposed reduction over generating many attractive decompositions. The first success must be a smaller, causally testable program beyond the known recurrence and trivial count/table baselines.



Logan: Is MiB or quantizing really the right metric here? How would optimizing for that help with interpretability?

Astra: **No. MiB measures engineering cost, and quantization alone does not explain computation.** I conflated a feasibility check with an interpretability objective. The handoff’s “25% reduction in any cost” success criterion is too weak.

Reducing memory helps us *run the investigation*. It supports interpretability only when the reduction comes from discovering structure—for example, replacing twenty independently specified computations with one shared operation and small, explicit adapters.

Consider three outcomes:

| Change                                                                 | Interpretability progress                              |
| ---------------------------------------------------------------------- | ------------------------------------------------------ |
| FP32 → BF16                                                            | None established                                       |
| Replace a dense tensor with equally opaque low-rank factors            | Establishes a constraint on computation                |
| Discover a shared equality detector used by several routing mechanisms | Explains an operation, its dependencies, and its reuse |

**The objective should be fewer independently specified computations, subject to your behavioral and causal requirements.**

I would assess a candidate through:

* **Explicit operations:** What does each intermediate compute from its inputs?
* **Shared implementation:** Which consumers reuse the same computed quantity or rule?
* **Dependency structure:** How many interactions must be specified to connect these operations?
* **Remaining opaque content:** How many arbitrary coefficients, tables, or native modules remain necessary?

That last item prevents cheating. Calling a giant weight matrix “one operation” does not make its contents explained. Likewise, replacing several heads with a huge lookup table can improve runtime while making the explanation worse.

A description-length objective can encourage the right discoveries **if its language charges for all those ingredients**. A shared subroutine is described once; its calls and adapters cost extra. Arbitrary numerical content remains charged. Quantization savings should be reported separately so they cannot masquerade as structural discoveries.

Even then, compactness is only a search preference. Your actual evidence comes from the program predicting OOD behavior, executing independently, predicting removal effects, and composing correctly with other programs. A short alternative algorithm with matching outputs still needs a causal correspondence to the model.

**I would replace the pilot’s success criterion with:**

> Recover a previously unspecified reusable computation, express its inputs, operation, and consumers explicitly, and verify its extraction and joint intervention behavior on held-out cases. Demonstrate reduced structural description cost against the original implementation, including all adapters and remaining opaque parameters.

Bytes and latency would remain **resource constraints and secondary measurements**, not evidence that we understood the model.
