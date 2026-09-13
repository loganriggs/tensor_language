# Compression of interactions: three settings and what would count as success

Requested direction, 13 September 2026. **Status at 08:37 UTC: campaign underway.** Initial interaction fits, a small exact graph simplification, full-U shared/private fits and sparse-core baselines have completed; see the progress note below. This supersedes the previous restriction against compression as a primary task. Discover structure from weights first, then validate on text. Earlier failed fits remain relevant controls.

The strongest current outcomes are the exact ten-to-six product-bank reduction in setting1 and roughly31% less packed tensor storage in setting2. Neither is a whole-model compression claim. The sparse setting2 executor uses more working memory and runs slower than dense contraction. A newer shared-output-subspace fit converges locally but misses fidelity:12.48% coefficient error, and own-term behavioral errors above10% in two regional groups. It preserves paired common outputs much better than their differences. An executed weights-only balanced-metric control addresses that tradeoff without increasing nominal program size; it has not yet established a converged fit or behavioral improvement. [Shared-write results](../../INTERACTION_SHARED_WRITES_V1_MATH.md) · [Balanced metric derivation and control](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-13_0829.md).

### Initial progress

- **Setting 1:** Unrestricted-input projector/native-product baselines remain weak. Respecting the actual generated response exposes an exact simplification: three response vectors and six downstream symmetric products instead of four and ten. The product bank is40% smaller per context; native validation preserves all160 existing regional/FineWeb outcomes exactly. The shared generator now also generates changed attention, joint normalization and the additive background. Full-panel regional local-effect error is0.116–0.305%; one FineWeb group fails its10% bar. Pristine context, dense weights and native suffix remain. [Exact simplification](../../RESPONSE_PRODUCT_BASIS_V2_MATH.md) · [Background validation](../../COMPOSED_MLP10_BACKGROUND_V1_MATH.md).
- **Setting 2:** An exact shared-expression rewrite preserves all 120 cached writes and removes two scalar multiplications, but saves only about 0.51% of core multiplications. The learned output-block fit improves squared coefficient error by only 0.85% after three promoted starts converge; it also worsens spelling-contrast preservation despite improving total error. [Exact graph rewrite](../../THREE_GROUP_SHARED_DAG_V1_MATH.md) · [Learned fit and counterchecks](../../HEAD17_OUTPUT_BLOCK_FIT_V1_MATH.md).
- **Setting 3:** Both shared/global-private fits have now completed their budgets. They improve on matched-byte global subspaces, but all promoted runs remain unconverged, and both fail FineWeb and regional preservation criteria. This is a useful representation comparison, not an adopted compression. [Fit and limits](../../FULLU_SHARED_LOCAL_FIT_V1_MATH.md) · [Regional validation](../../SHARED_LOCAL_REGIONAL_V1_MATH.md).
- **Setting 2 sparse-core follow-up:** The output/head frame saves30.81% of dense folded-tensor storage at10% coefficient error. Error relative to the mixed term’s own regional effect is2.72–8.12%. Packed storage improves, but CSR resident memory and CPU speed lose to dense contraction, and all reader nodes remain active. [Behavioral checks](../../INTERACTION_SPARSE_REGIONAL_V1_MATH.md) · [Execution and graph costs](../../SPARSE_INTERACTION_EXECUTOR_V1_MATH.md).
- **Setting 1 whole-interaction correction:** The small local-effect errors above do not describe the original two-edit interaction, where current error is11.9–16.1%. The main omission is an inherited mixed response before block10. A shared three-branch extension now retains that response and passes full-panel native validation: original regional target error0.038–0.084%, FineWeb0.45–7.67%. Six tiny FineWeb sign reversals remain. This closes a conditional fidelity gap; no additional compression ratio is claimed. [Coverage audit](../../COMPOSED_ORIGINAL_INTERACTION_COVERAGE_V1_MATH.md) · [Extension and control](../../COMPOSED_JOINT_RESPONSE_V1_MATH.md).

The central comparison is **separate compression versus joint compression of composed terms**. A layer can be complicated while a particular downstream interaction reads only a small part of it. Conversely, two apparently separate interactions may use the same intermediate computation. We should allow both outcomes instead of forcing a separate decomposition for every module.

## Exactly three settings

### 1. The recent layer9–10 regional interaction

Object: the child/remainder interaction passing through MLP9, attention10 and the bilinear multiplication in MLP10. Start with the full direct-product contribution and compare its residual/residual, mixed residual/attention, and attention/attention terms. Keep the mixed terms: residual-only already failed behavioral preservation.

Let the two pre-MLP10 input changes be

$$
c=c_R+c_A,\qquad r=r_R+r_A,
$$

and define

$$
K(x,y)=D_{10}[(L_{10}x)\odot(R_{10}y)+(L_{10}y)\odot(R_{10}x)].
$$

Then the target is

$$
\frac{K(c_R,r_R)+K(c_R,r_A)+K(c_A,r_R)+K(c_A,r_A)}{\rho_{CR}}.
$$

Compare independent compression of these terms with a common reader/product bank serving all of them. Next fold selected readers through the actual MLP9 response and attention10 maps, asking whether total graph cost decreases. Do not assume all of a native head must be retained or removed together.

Why promising: this interaction has an explicit computational specification and behavioral tests. The generated-input CPU baseline preserves regional local effects within0.09–0.21% on96 prefixes. That is a baseline implementation result, not compression. The fixed-writer MLP9 response has now been reduced exactly to three context-dependent vectors and six symmetric downstream products. This **does not imply a global six-factor model**: producing those vectors can remain expensive. The joint response also needs its inherited mixed term; the extension evaluates the same shared generator at a, b and a+b.

Boundary: initially keep normalization and pristine context explicit. Report both local compressed-operator cost and total executable cost, including input/background generators and suffix. Full-panel generated-background validation passes regional local-effect criteria but misses one FineWeb criterion; original-interaction coverage remains a separate, harder target. This setting tests whether upstream folding exposes shared readers/products that survive selective interventions.

### 2. The recent head17.2 → MLP17 → output interaction

Object: the retained three-contraction attention interaction, together with its effect through the final bilinear layer and selected unembedding readers. The three retained contractions involve both QK factors and values; they are not three independent semantic tasks.

Let \(v_1,v_2,v_3\) be their residual writes and \(g_{17}\) the normalized residual-plus-MLP17 map. Preserve

$$
\Phi(z)=g_{17}(z+v_1+v_2+v_3)-g_{17}(z),
$$

or an explicitly declared output projection of it. Do **not** replace this target with the sum of three separately propagated effects: the downstream bilinear layer can multiply the retained writes together.

Compare compressing the three contractions independently, compressing them with shared query/key/value readers, and folding their output readers through MLP17 jointly. Permit private corrections where shared computation alone does not fit. Test whether the retained terms share inputs or output-relevant product combinations beyond their obvious shared native head.

Why promising: the compact retained predictor already has fresh construction/city-context evidence, with roughly2–4% local last-block effect error after MLP17. Compressing this composed target may reveal simplifications missed by compressing the head output alone. Its inherited upstream interaction remains outside this local target, so compression savings cannot be presented as explaining the entire regional circuit.

Boundary: start with the same declared raw input ports/background and selected output readers as the existing predictor; fold backward only when all new dependencies are counted. Keep a residual-write preservation check if the representation is intended for reuse by other downstream readers. Output-only success is insufficient for that broader claim.

### 3. The full unembedding folded into the last bilinear layer

Object: the vocabulary-wide quadratic numerator, without selecting a task first:

$$
T_{vij}=\frac12\sum_{k=1}^{4608}(UD_{17})_{vk}
\left[(L_{17})_{ki}(R_{17})_{kj}+(L_{17})_{kj}(R_{17})_{ki}\right].
$$

For repeated input \(x\), \(T(x,x)\) is the folded bilinear numerator. The bias, residual route, RMS factors and final logit saturation remain explicit in execution and validation; they are not secretly treated as polynomial tensor entries.

Fit a common graph for all output rows, with token-specific connections or corrections. Let token groups emerge from shared functions; do not impose semantic clusters first. Compare separate output-group models against a shared graph, and compare the unfurther-folded object against selected upstream producer paths at the same output scope. This is the broader weights-first control for the two behavioral settings, not a claim that the earlier global attempts succeeded.

The native numerator already has4608 bilinear products, much fewer than50304 tokens. Merely finding fewer factors than vocabulary size is therefore not a meaningful win. Beat the actual factored baseline or expose useful shared structure at matched cost/error. The conceptual tensor has66.76billion entries, about267GB in FP32; evaluate it through implicit contractions, never materialize it. The checked inventory is [here](../../INTERACTION_COMPRESSION_SETTINGS_V1_INVENTORY.json).

## Compression families to compare

| Family | What becomes simpler | Main assumption or failure mode |
|---|---|---|
| CP/product dictionary, plus a spectral baseline | Fewer scalar products and output directions | A small number of separable products may be too restrictive; a small matrix rank need not mean simple multiplication |
| Sparse Tucker-style interaction graph | Shared reader nodes; only selected pairs interact and selected outputs consume them | Sparsity depends on the learned frame; dense adapters can erase savings |
| LL1 / block-term representation | A few output-linked blocks, each allowed several input readers | A task may read a subspace rather than one direction; oversized dense blocks can hide complexity |
| Shared arithmetic DAG | Intermediate linear forms/products feed multiple retained terms or output groups | Topology search is difficult; reuse must reduce actual cost, not merely rename duplicate calculations |

CP and Tucker are established tensor representations; block-term models allow low-multilinear-rank blocks rather than only rank-one terms. See [Kolda and Bader](https://epubs.siam.org/doi/10.1137/07070111X) and [Domanov and De Lathauwer](https://arxiv.org/abs/1808.02423). These references define candidate representation families, not a recovery guarantee for our composed, normalized circuits.

The first executable baselines should prune whole existing terms/edges and refit retained coefficients, then compare learned readers or blocks. A fixed-basis null must not be promoted to absence of sparse structure in another basis. Search DAG sharing during fitting as well as after it: post-hoc merging alone can miss a cheaper joint representation.

## Operational definitions of simplicity and reuse

**Simplicity:** publish a vector of costs, rather than collapse everything into an arbitrary scalar score:

- Independent reader/intermediate nodes, with their widths and operation types.
- Active input/product/output connections and their parameter counts.
- Nonlinear/product operations, including required normalizations.
- Stored scalars or bytes, including adapters, exceptions and opaque weights.
- Executed operations, working state and measured time at stated sequence lengths.

A vector-valued node does not cost the same as a scalar node. A multiplication is a two-input operation—a hyperedge if readers are drawn as vertices—not just one ordinary connection. Count the actual executable graph. Small dense blocks are allowed when cheaper and clearer than many nominally sparse scalar entries. Gauge rotations can change coordinate sparsity, so compare resulting programs and charge their adapters.

**Computational reuse:** one identical intermediate function on compatible inputs is computed once and consumed by at least two distinct interaction branches. It must produce a net cost saving against the best independently compressed alternatives we actually found, at matched error and scope. Similar vectors or coefficients are not enough. Shared weights evaluated at different positions/inputs can save storage while still requiring separate computation; report those separately.

**Circuit reuse:** in addition, the same intermediate supports both consumers under held-out prediction and selective/joint interventions. Cutting its connection to one consumer should leave the other route intact; changing the shared producer should have the predicted consequences at both. Distinguish reuse across terms of one interaction from reuse across different behaviors. The two recent settings are parts of the same regional interaction family, not two independent semantic discoveries.

Readable shared computations with few exceptions remain a legitimate qualitative preference, but should accompany these measurable costs.

## Fair fitting, validation and first sequence

Use the same target, allowed input domain, output scope and fidelity budget for independent/joint comparisons. Fit algebraic operators with exact contractions where practical, otherwise fixed text-independent probes and independent probe validation. Do not mix arbitrary independent QK/value ports with states constrained by the actual folded producer and call their norms equivalent. On a nonlinear producer-constrained target, coefficient error and probe error are different objectives and must be labelled. Preserve normalization functions explicitly initially.

After weight fitting, freeze candidates and validate state/write error, signed effect error, absolute errors near zero, controls, fresh text and joint interventions. Report compression error relative to the retained uncompressed program **and** total error relative to the original model. A heavily approximated retained baseline must not make a further approximation look more faithful than it is.

For nonlinear fits use multiple initializations: ten inexpensive starts spanning structured and randomized initializations, promote the strongest few, and check objective decrease plus meaningful stationarity after balancing redundant scales. Use closed-form coefficient/writer updates when valid. Earlier fits encountered false stopping from huge coordinate scales; raw gradient size alone is not convergence. Budget-matched converged local optima are evidence, not global impossibility proofs.

Start with setting1's joint residual/mixed product graph and setting2's retained three contractions: frozen-term/edge selection and LL1/shared-reader baselines first, then topology changes where residual structure points to them. Use setting3 as the broad control with existing implicit contraction machinery. Compare independent versus shared fits at matched capacity before escalating a large campaign. The normalized-Frobenius/tensor-similarity proposal remains an optional fitting route when it matches the exact target; it is not a prerequisite or a new implementation project now.

The initial residual-fold validation and subsequent generated-background promotion have completed. Their results are recorded above. Compression remains the primary campaign; generating the inherited mixed response is a targeted fidelity check of setting1, not a change back to a separate background-reconstruction campaign.
