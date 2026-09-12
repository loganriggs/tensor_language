# Research update: interaction-path decomposition, shared computations, and the remaining extraction gap

**12 September 2026. Main report through 15:35 UTC; latest findings added through 16:20 UTC. Requested update for Logan.**

This covers the work since the [last major update, 11 September at 21:42](research_update_2026-09-11_2142.md), including implementation of the [interaction-path proposal](interaction_path_decomposition_proposal_2026-09-11.md). It covers Codex's research, without counting Claude's parallel experiments as mine. Layer numbers start at zero: MLP17 and attention17 are the last block.

## High-level overview

**Your interaction-path idea has produced useful mathematics, working decomposition methods, and two concrete shared-component examples. It has not yet produced a good general decomposition from which reliable circuits simply fall out.** The strongest new result is a weight-discovered shared computation inside joint attention, followed by an executable local extraction and controlled behavioral tests. The larger sparse path fits remain much weaker than the full goal.

The main events, in order, were:

1. **We implemented and converged the first joint sparse path comparison.** It jointly decomposes residual/residual, residual/attention, and attention/attention interactions with the entire unembedding included in the coefficient objective. Shared path readers gave more reproducible coefficient structure than independent readers, but only a small reconstruction advantage. The proposed stable pieces did not replicate their native behavioral effects accurately enough.
2. **We folded through two bilinear layers and searched for hierarchy.** This exposed a real mathematical opportunity: different matrices can represent exactly the same quartic polynomial, and searching that freedom revealed much cheaper approximate quadratic-intermediate representations. Unfortunately, the initial small input spaces captured very little of the full composed function, and independently simplifying each intermediate damaged actual execution.
3. **We tried more general mixed products and shared parents.** Some local components became stable and executable, including a grammatical component with two output branches. Fresh constructions and joint interventions supplied partial behavioral evidence. Deeper extraction through MLP15 remained unsuccessful; a good conditional component was not automatically a self-sufficient circuit.
4. **We implemented the full joint QK1 × QK2 × value computation.** This goes beyond folding OV while treating attention scores as unexplained inputs. A source/query factorization exposed two source products that nearly shared a quadratic parent. Rewriting them as a shared parent with two children produced a concrete small arithmetic graph.
5. **We traced that graph into regional-spelling behavior.** Its removal consistently weakens British/American cue effects on several new panels. The cue is carried mainly through the child readings, rather than the shared parent itself. A dominant consumer in head17.2 can be compiled into a portable branch; a complete-vocabulary table closes its first-state input dependency.
6. **Harder tests clarified what that branch actually does.** It responds to distracting regional cues as well as the intended cue. Almost all of its measured regional effect is expressed through reads at positions other than the city tokens themselves. Source changes drive the response, while query changes generally oppose part of it. Those interactions are too large to omit when extracting the computation.

**The encouraging part is actual shared arithmetic and executable conditional components. The disappointing part is how often stable-looking weight structure fails stronger behavioral or upstream-closure tests.** I have preserved those failures rather than treating every successful algebra check as a circuit discovery.

Discovery remained weights-first. These experiments did not use a new million-token training campaign or fit factors to the behavioral panels. Later semantic labels and test choices are supplied interpretations of frozen weight-derived candidates. FineWeb is the training-domain reference; Pile is a separate corpus-shift check, with limitations described below.

**Latest development, through 16:20:** a further upstream fit exposed a recurring shared linear parent, but also showed that a fixed dictionary of source products can badly underrepresent a simple joint query–source computation. Keeping the complete joint QK operation with one folded value direction preserves much more coefficient energy. This is a representation lead awaiting behavioral validation; details are in section 10.

## Where we stand on your four properties

| Property | Evidence gained since the last report | Main unresolved requirement |
|---|---|---|
| **OOD prediction** | Frozen components transfer selected effects to new words, constructions, city/template combinations, and competing-cue arrangements. One grammatical branch also received a FineWeb/Pile comparison. | Broad prediction across natural text and genuine distribution shifts. The Pile capability gate missed; controlled new prompts do not prove pretraining disjointness. |
| **Extraction / sufficiency** | Explicit local programs execute shared products and their interventions. A regional branch now accepts token IDs in place of the original first-state vector. A separate compiler can execute proposed graphs without materializing their original residual states. | Generating the regional branch's query/current inputs independently. A recursive attempt to regenerate routing from selected native updates failed. |
| **Selective removal** | Removing the regional component weakens all 48 target contrasts in the newest 96-context panel; simpler panels passed registered coverage/control bars. A grammatical branch has some selective natural-text evidence. | Robust semantic separation: irrelevant cities also drive the regional branch, one coverage cell misses, and other branches have broader probability-preservation failures. |
| **Composition / reuse** | Shared quadratic parents feed multiple children; exact node/edge intervention accounting works. Regional child effects nearly add on the latest panel. | General reuse across distinct tasks or heads, and faithful producer–consumer composition. Query/source interactions and cancelling consumer combinations remain important failures. |

There is **no new component that I would call a completed four-property circuit**, and no replacement for the complete model. “Extracted” below always states the boundary and the native computation still required.

**Terms used below:** *native* means the original model; *OOD* means out-of-distribution; a *DAG* is a directed acyclic graph of computations; *RMS* is root-mean-square normalization. A *logit* is a score before converting scores to probabilities. A *margin* is the difference between two token scores. A *nat* uses natural-log probability units; increasing cross-entropy loss means worse prediction. *Orthonormal* readers have unit length and mutually zero dot products. A small *primal-dual gap* bounds how far a convex solver's objective is from its optimum; it does not certify that the chosen model family is correct.

## 1. What exactly are we decomposing?

A bilinear MLP computes

$$
B(x)=D\big[(Lx)\odot(Rx)\big]+b.
$$

A **reader** is a row of a matrix such as $L$: its dot product with the residual state produces a scalar feature. The elementwise product $\odot$ multiplies paired readings. A **writer**, a column of $D$, says where the product is added to the residual stream. The residual stream is the model's running vector state.

Let $U$ be the unembedding, which turns the final state into vocabulary scores. Folding $U$ into $D$ gives an output-token-indexed family of quadratic forms:

$$
[UB(x)]_v=x^TS_vx+[Ub]_v,
$$

$$
S_v=\operatorname{sym}\!\left(
L^T\operatorname{diag}((UD)_{v,:})R
\right),
\qquad
\operatorname{sym}(M)=\frac{M+M^T}{2}.
$$

The input appears twice, so only the symmetric matrix matters. This is the third-order tensor view: one output-token index and two input indices.

Your proposal changes the unit of decomposition. Write the last MLP's incoming state as $y=g+p+a$, where $p$ is the preceding MLP's scaled contribution, $a$ is attention's contribution, and $g$ contains the remaining residual/re-entry terms. Then

$$
\begin{aligned}
y^TS_vy={}&g^TS_vg+p^TS_vp+a^TS_va\\
&+2g^TS_vp+2g^TS_va+2p^TS_va.
\end{aligned}
$$

These six source-pair interactions are an exact starting point. A mixed path is already the sum of its two orderings; it is not two separate discoveries. Substituting a bilinear expression for $p$ produces cubic or quartic terms. Keeping the intermediate $p$ or its useful readers avoids constructing an enormous dense high-order tensor.

**Normalization is part of the computation.** The actual last MLP sees $y/\rho(y)$, where

$$
\rho(y)^2=\frac{\|y\|^2}{1152}+\epsilon.
$$

Its quadratic numerator is divided by $\rho(y)^2$. Final normalization, biases, direct residual writing, and the final $30\tanh(\cdot/30)$ score cap also remain. The entire model is therefore not a fixed polynomial in its initial inputs. Most exact polynomial derivations concern specified numerators, with these nonlinear operations retained explicitly.

## 2. The first joint sparse interaction fit really did run

We implemented the first proposal stage: residual/residual, residual/attention and attention/attention, after folding attention's output matrix into the last MLP and using the full unembedding metric.

The learned program has linear source features and selected product edges:

$$
\widehat F(\xi)=\sum_{(i,j)\in E}w_{ij}
(b_i^T\xi)(b_j^T\xi),
$$

with the appropriate symmetric scaling for mixed edges. Here $\xi$ collects the two formal source ports, and $w_{ij}$ is a full output writer. Sharing a feature across several edges is an explicit proposed reuse of that reading.

**This first implementation is sparse in product edges with dense output writers. It is not yet a fully learned entry-sparse Tucker core over input and output modes.** That distinction matters when judging how much of the original proposal has been explored.

The full vocabulary need not be materialized as 50,304 large interaction matrices. For a residual-output error $E$, its full-U squared metric uses $U^TU$:

$$
\|UE\|_F^2=\operatorname{tr}(E^TU^TUE).
$$

For fixed orthonormal readers, output coefficients are solved exactly, and the largest 96 edge energies give the optimal support in that fixed basis. A manifold optimizer then moves the readers while preserving orthonormality; support is reselected between solves. We continued beyond the first pilot until all four arms satisfied their registered local gradient and support conditions.

| Matched arm | Start 1 coefficient capture | Start 2 coefficient capture |
|---|---:|---:|
| Shared readers across path blocks | 2.980% | 2.781% |
| Independent readers for each block | 2.730% | 2.746% |

Each arm uses 147,456 fitted floats and 96 product edges, before native dependencies and adapters. **Coefficient capture** is the fraction of squared tensor coefficients reconstructed. It is not the fraction of tokens predicted correctly or the fraction of a behavior explained. These are small-budget, locally converged fits; they miss the registered 10% shared-versus-independent improvement criterion.

There was nevertheless stronger cross-start structure: the complete joint fitted functions had coefficient cosine 0.944, versus 0.871 for the independent fits. After also removing common vocabulary writing in the comparison metric, 18 edge correspondences passed both screens. One quotation-related edge was an alias of an existing documented square, not a newly discovered circuit.

The crucial native test failed. The 18-edge replicas disagreed by 28.2% in physical writes, and their swap effects disagreed substantially across the grammar/control families. Exact discrepancy accounting implicated output-writing differences more than reader differences. Grouping output directions improved some level/removal comparisons but did not repair all swaps.

**Conclusion:** joint path fitting found reproducible coefficient structure, but that structure was not behaviorally reliable enough to promote. The sparse-interaction hypothesis is not disproved; this specific budget, representation, and objective did not deliver the desired circuits. [Method, convergence, aliases, and native failures](../../COUPLED_SPARSE_PATH_V1_MATH.md).

## 3. Deeper folding exposed hierarchy—and a serious input-space restriction

For two bilinear layers, an important path has the form

$$
P(x)=D_{16}[(L_{16}x)\odot(R_{16}x)],
\qquad
F_v(x)=P(x)^TS_vP(x).
$$

Scaling and biases are handled separately in the actual executor. The displayed numerator is quartic: degree four in $x$.

The first sparse quartic pilot learned 16 input directions and retained 128 fourth-degree interactions. It captured only **0.125–0.130%** of full quartic coefficient energy. A fixed-space ceiling calculation was decisive: even keeping every quartic interaction in those learned spaces would capture only **0.138–0.149%**. The selected edges already retained most of the projected energy. Adding more edges in the same spaces was therefore a poor next move.

A separate scale audit established local stationarity under a meaningful normalized objective; the original convergence predicate remains recorded as failed. Neither statement proves that those are the best possible 16-dimensional spaces.

### A genuine mathematical gain: equivalent polynomial representations

Let $z(x)$ contain quadratic monomials. A quartic can be represented as

$$
f(x)=z(x)^TAz(x).
$$

Different symmetric matrices $A$ can represent exactly the same polynomial because products of the entries of $z$ are algebraically dependent. The canonical matrix can therefore look much more complicated than the actual arithmetic.

A simple example is $x_1^2x_2^2$: it is just the square of the single intermediate $x_1x_2$, even if a canonical pair-matrix representation has several nonzero eigenvalues.

We implemented a convex nuclear-norm search over coefficient-equivalent matrices. The **nuclear norm** is the sum of absolute eigenvalues here; minimizing it favors economical signed-square representations, but is not the same as minimizing rank or arithmetic cost. Eigen-decomposing the resulting matrix gives

$$
f(x)\approx\sum_{j=1}^{k}\lambda_j q_j(x)^2,
$$

where each $q_j$ is itself quadratic and $\lambda_j$ may be negative.

On four fixed projected output functions, the solver converged with primal-dual gaps below $10^{-7}$. At eight retained intermediates, coefficient errors improved from **26–38%** for canonical matrices to **0.83–1.32%** after optimizing the equivalent representation.

That is a real hierarchy result, but **inside already projected scalar functions**, not a decomposition of the whole model. Native validation still exposed the input-space restriction. Lifting back to full inputs helped; independently truncating each inner quadratic then caused roughly **55% native write error**, while an isotropic remainder correction worsened it to about **80%**. [Derivation, converged results, and native controls](../../QUARTIC_GRAM_HIERARCHY_V1_MATH.md).

### Why we moved beyond separate squares

A more general graph allows products between distinct quadratic intermediates:

$$
\widehat F_v(x)=\sum_{(i,j)\in E}A_{vij}q_i(x)q_j(x).
$$

For one learned bank, allowing the full mixed core captured 55.76% more coefficient energy than separate squares; a small greedy mixed graph gained only 3.49%. Those gains did not by themselves pass native-effect fidelity. We implemented joint mixed-graph optimization, exact output elimination, independent starts, node replacements, and planted recovery/curvature checks. Some restricted problems have genuine local traps; merely running the same optimizer longer is not a universal solution.

We also tried a weight-only repeated-input/isotropic objective that includes trace terms absent from the plain coefficient metric. Its exact linear solves improved their own objective but worsened native fidelity, including on learned banks. This was an explicit inductive-bias test, not a failed language-data fit. [Mixed graph and optimization campaign](../../COUPLED_QUARTIC_WRITER_V1_MATH.md) · [Repeated-input metric and failures](../../QUARTIC_REPEATED_INPUT_V1_MATH.md).

## 4. A separate composed-MLP component did become useful

The deeper-MLP route produced a shared quadratic parent with different quadratic partners and output branches. One branch had an `ing`-related behavioral interpretation. Frozen interventions generalized to new verbs and constructions; the joint branch-effect predictions passed within about 0.56%. Independent fits also recovered a very similar complete branch function.

An exact input fold produced a roughly **7.98 MB local executable**, about 5.33 times smaller than its standalone unfused native-factor representation. It still required the native MLP16 input, downstream normalization and background.

On natural text, removing the `ing` branch added about 0.00728 nats of loss on the FineWeb subset and 0.01833 on the Pile subset. The Pile native-capability gate failed, and context-permutation controls limited the semantic interpretation. These results do not establish general OOD success.

Tracing this branch backward was informative. Attention16-dependent terms were small on several active construction families; earlier residual/MLP computations mattered more. Folding through MLP15 exposed substantial mixed interactions. Shared weight-derived input coordinates beat random coordinates, but even converged coupled fits failed the full intervention-fidelity criterion. Larger exact-metric probe checks eventually found noise-dominated search directions rather than a reliable improvement.

This is a **different component** from the regional-attention branch below. Their evidence and failures should not be pooled as if one circuit passed every test. [Full composed-MLP results and scope](../../COMPOSED_WEIGHT_COMPARISON_V1_RESULTS.md).

## 5. The strongest new structural lead came from joint attention

We implemented an operator that keeps both QK factors together with the value/output path. For one query/source position pair, it has the form

$$
a(q,s)=\sum_h g_h(q,s)
(q^TA_hs)(q^TB_hs)M_hs.
$$

Here:

- $q$ is the 1,152-dimensional normalized query state.
- $s$ has 2,304 coordinates, concatenating the current source state and the first-layer value-input state.
- $A_h$ and $B_h$ fold QK1 and QK2, including relative position, into bilinear forms.
- $M_h$ contains value and output writing, including the signed first-value mixture.
- $g_h$ retains the actual input-dependent Q/K normalization factors.

With the gates held explicit, the numerator has query degree two and source degree three. This directly addresses your concern that useful structure can live in the **product of QK1 and QK2**, rather than in a literal division of tasks between them. Exact native folding through the final MLP and full unembedding was checked. However, the subsequent source-feature fit optimized the attention numerator with a weight-derived metric; it was **not** a single end-to-end full-U sparse fit of that whole path. [Joint operator](../../JOINT_ROUTING_VALUE_POLYNOMIAL_V1_MATH.md).

Whole-head similarity failed to find robust reuse. We then allowed shared source computations with private query-dependent consumers:

$$
h_r(s)=(a_r^Ts)(b_r^Ts)(c_r^Ts),
\qquad
\widehat a_h(q,s)=\sum_r h_r(s)\beta_{hr}(q).
$$

This permits two consumers to use the same source feature differently. The source-feature Gram matrix is only 16 by 16; for fixed source readers, the optimal query-dependent writers are solved exactly through that Gram matrix. This is **variable projection**: eliminate the linear subproblem, then optimize the nonlinear source factors.

The broad fit itself was weak: initial reference coefficient capture was about 0.24–0.34%, and the multi-head-use criteria failed. A real optimizer safeguard bug was found and controlled: its descent condition depended on arbitrary objective scale. An angle-based condition repaired that issue. One continuation reached stationarity; the other developed nearly cancelling factors and a line-search failure.

### The cancelling factors exposed a shared parent

Two source products became almost identical in two of their three readers, with opposing large contributions. We first rewrote the pair using exact average/difference coordinates. This preserved the projected computation while reducing its Gram condition number from approximately **320,512 to 27.8**. A condition number measures sensitivity to small numerical perturbations; this improvement made the same computation much easier to execute stably.

The pair could then be approximated by

$$
p(s)=(a^Ts)(b^Ts),
\qquad
h_0(s)=p(s)(c_0^Ts),
\qquad
h_1(s)=p(s)(c_1^Ts).
$$

This is an explicit DAG: compute one quadratic parent once, then feed it into two products. It uses four source readers rather than six. The simplification preserved the small block's native writes within roughly 0.30–0.42% and its full-vocabulary removal effect within 0.37% on the initial FineWeb screen.

This is the clearest new example of the kind of shared intermediate you asked about. But its dominant consumer is head17.2: it does not establish broad reuse across heads or unrelated tasks. A low global capture fraction also remains compatible with a locally useful component. [Source factorization, optimization diagnosis, exact rewrite, and native tests](../../SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md).

## 6. What this attention component does behaviorally

Frozen removal tests found a regional-spelling contribution: British/American cues change preferences between spellings such as `labour/labor` or `theatre/theater`, and removing the component weakens those changes. Several simple and fresh city/template panels passed their registered basic screens, often reducing cue contrasts by roughly 10–18%.

A factorial interchange test split the component into four ports: the shared parent, child readings, query-dependent writers, and normalization gates. It swaps donor values into chosen ports while retaining the recipient's native background. Evaluating all combinations accounts for interactions exactly on that finite intervention grid.

The shared-parent-dominance prediction failed. **The child readings carried most of the regional cue transfer.** The parent appears more like a shared operation used by those readings, rather than the regional variable itself. Independent weight fits supplied further source-span and native-effect agreement.

Upstream producer accounting identified useful contributions from attention8, attention9 and attention13, with both current-value and first-value branches. Their values could be folded into downstream readers. But source-value sufficiency did not close the routing/query dependencies.

One fresh panel had an article confound, “A American.” Its semantic interpretation was withdrawn; corrected rows and a shared validator were used in subsequent tests. Earlier unaffected results are distinguished in the primary note. This was a real evaluation error, not a negative result to explain away.

### The portable extraction now exists

A weight-only cross-consumer Gram calculation selected head17.2 as the dominant private consumer of the shared component. Keeping that consumer reproduced the full shared component's tested removal effects within about 0.4% on one fresh panel. This removes other consumers of the selected feature, not whole native attention heads.

The resulting package contains explicit source readers, joint query/key maps, projected writers and the small dual mixing map. A later token-input version replaces the first-state readings with a complete 50,304-token lookup. That lookup is built from model weights and native initialization, without fitting language examples.

FP32 was accurate for the whole branch but failed a small child-level comparison. Captured-state analysis isolated reader rounding. Retaining higher precision only for the sensitive source readings and dual mix repaired the separate candidate under the unchanged threshold: worst native child error fell from $6.76\times10^{-5}$ to $2.78\times10^{-7}$.

The [portable mixed-precision package](../../extracted_circuits/regional_shared_head2_token_mixed_v1/README.md) stores **863,264 scalars / 4,276,480 tensor bytes**. It requires query/current states, token IDs and relative position, and returns the local branch write. It closes first-state generation but increases storage versus the earlier package. It does **not** generate the remaining native context or final logits by itself. [Extraction and precision details](../../COMPILED_TOKEN_SHARED_HEAD2_V1_MATH.md).

## 7. The newest hard tests: distractors, source positions, and opposing query effects

The latest panel crosses an editor's city with a tourist's city, reverses clause order, and counterbalances which city pair occupies which role. It has 96 contexts and 48 paired editor-city contrasts. These are new arrangements using known spelling concepts and cities.

Removing the branch weakens every target contrast, but one original coverage cell reaches only 9.59%, missing its 10% bar. Both the model and branch respond substantially to the tourist's city. After counterbalancing city assignments, the branch's target/distractor response ratio is **1.64 when the editor clause comes first and 0.94 when it comes second**. Neither child gives a robust role-specific separation.

The two child-removal effects nevertheless nearly add: the within-package nonadditive margin remainder is approximately **0.21%** of the joint effect norm on this panel. That is useful local composition evidence, with limited semantic selectivity. [Competing-cue results](../../REGIONAL_COMPETING_CUES_V1_MATH.md).

### It is not mainly a direct read of the city token

We partitioned the branch's source-position writes into editor city, tourist city, and all other positions. The sum replays accurately. Removing only the two city-position writes fails to reproduce the full regional contrast change, with **95–97% error**. Individual city-site contributions account for only about 3–5% of the corresponding full contrast norm.

This is a statement about where the branch reads and writes, not a claim that the city token is causally unimportant. Earlier layers can propagate its information to later states, and a changed query can modulate reads at unchanged positions. [Position-partition receipt](../../REGIONAL_SOURCE_POSITIONS_V1_RESULT.json).

### Source changes and query changes must be considered together

Let $F(q,s)$ denote the component's total write, and use subscripts 0/1 for recipient/donor query and source inputs. The four evaluations are $F_{00},F_{10},F_{01},F_{11}$. A symmetric exact allocation of the total change is

$$
\Delta_q=\tfrac12[(F_{10}-F_{00})+(F_{11}-F_{01})],
$$

$$
\Delta_s=\tfrac12[(F_{01}-F_{00})+(F_{11}-F_{10})],
\qquad
\Delta_q+\Delta_s=F_{11}-F_{00}.
$$

This is two-port Shapley accounting: average each port's effect over the two possible update orders. It is an accounting convention for the specified intervention, not a unique causal decomposition of the model.

On saved native inputs, the source contribution is larger than the net change and the query contribution partly opposes it. Their write-vector cosine is about −0.57 to −0.60. State changes before the modified cue are exactly zero, as expected from causal attention.

The **native final-suffix test completed while this report was being prepared**. It installs the saved full, source-only and query-only branch replacements in the original recipient background:

- Full and source-only replacements move spelling margins toward the donor on all 192 directed cue swaps.
- Source-only effect prediction still misses the full two-port effect by **14.6–32.9%**, failing the 10% criterion in every cell.
- Query-only replacement is mostly opposed to the donor movement and has **107–117%** effect error.
- The discrepancy between the sum of single-port effects and the joint effect is **12.8–16.1%**. This includes the branch's mixed query/source response and the nonlinear suffix; it must not be attributed solely to final normalization.

This interaction is a different object from the 0.21% child-addition remainder above. The two children can nearly add while the query and source inputs needed to generate them interact strongly. [Write accounting](../../REGIONAL_QUERY_SOURCE_ACCOUNTING_V1_RESULT.json) · [Native effect test](../../REGIONAL_QUERY_SOURCE_EFFECT_V1_RESULT.json).

**The extraction target is now clearer:** generate the contextual source information and the opposing query-dependent correction together. A token lookup or source-only approximation cannot reproduce the full computation to the current accuracy target.

## 8. Did the mathematical cycles help?

Yes, several produced executable changes or ruled out misleading interpretations. They did not produce a theorem guaranteeing recovery of this model's circuits.

| Review / mathematical idea | What was actually done | Practical consequence |
|---|---|---|
| **22:56, output/function grouping** | Grouped complete coefficient functions and tested native replicas. | Stable output subspaces were insufficient for stable context-sensitive swaps; writer-only similarity was too weak. |
| **02:00, mixed quadratic graphs** | Implemented joint optimization of cross-products, exact linear subproblems, and planted/curvature checks. | Moved beyond separate squares; exposed representation restrictions and local optimization traps. |
| **05:00, independent blocks versus shared DAGs** | Ran a matrix-free common-block relaxation and an explicit reusable-star counterexample. | Failure to find independent blocks cannot rule out simple shared arithmetic. Shared parents need not occupy disjoint subspaces. |
| **08:00, source-coordinate redundancy** | Derived and checked the automatic nullspace created by describing residual and attention inputs separately. | A large nullspace after OV folding can be bookkeeping redundancy, not discovered circuit sparsity. Attention16 source-sector tests also redirected tracing upstream. |
| **11:00, factorial interaction algebra** | Implemented exact finite-grid port interaction accounting and applied it to the regional candidate. | Distinguished the cue-carrying child readings from the shared parent; avoided assuming individual swaps add. |
| **14:00, execution in writer coordinates** | Compiled proposed graphs into changing writer-state coordinates, retained exact Gram-based norm accounting, and passed an 18-block synthetic composition test. | Provides machinery for genuinely closed proposed programs. It does not make the present native regional generator closed. |

The quartic equivalent-Gram search and the exact cubic average/difference rewrite were additional particularly useful mathematical results. They show why a dense-looking canonical tensor representation can hide simpler intermediate arithmetic.

Conversely, some mathematical constraints remain real: ordinary tensor rank can be a poor proxy for arithmetic complexity; invertible OV coordinate changes cannot magically lower exact rank; and coefficient error does not bound behavioral error on actual model states without stronger assumptions. The primary reviews contain the literature mappings and their assumptions: [02:00](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_0200.md), [05:00](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_0500.md), [08:00](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_0800.md), [11:00](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_1100.md), [14:00](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-12_1400.md).

## 9. What failed, what remains open, and what should happen next

The main bottleneck is now **faithful generation of interacting inputs**, alongside the broader unsupervised search problem.

An earlier routing attempt retained 13 selected native updates. It worked much better when evaluated on their original states than when allowed to generate its own states recursively: the recursive version incurred roughly 73–103% write error. A dependency audit expanded the selected updates toward most of the original producer. This is direct evidence that conditional preservation is not extraction closure.

Keeping mixed QK terms improved several ordinary value-transfer tests. But some cancelling combinations of consumers still failed, even though ordinary swaps passed. Simplifying the query/key normalizers into constants or 32/64-direction approximations also failed their registered effect tests. I stopped those rank variants rather than treating a sequence of small matrix approximations as circuit discovery.

The next priorities are:

1. **Use the source/query interaction evidence to define the next upstream extraction boundary.** Trace the contextual source readings and their query correction as a coupled computation. Test their joint replacement, not just isolated reconstruction on native states.
2. **Retain the broader interaction-path research objective.** The first sparse-edge fit did not exhaust learned output sharing, block-term structures, shared nonlinear intermediates, or joint DAG search. The successful small shared parent is a concrete design clue, not a reason to declare the regional example the whole project.
3. **Evaluate distinct assumptions with adequate optimization and explicit counterexamples.** Fixed-small-subspace quartics, separate squares, independent blocks, and ordinary product dictionaries failed for different reasons. Future work should change the relevant assumption and test recovery, rather than quietly recycling the same fit with a new label.
4. **Demand producer–consumer composition and broader behavioral controls before promotion.** Local arithmetic identity, new-prompt effect prediction, and portable packaging are useful but individually insufficient. The native remainder and all adapters remain charged.

There has been substantial implementation and analysis since the last report, but also too much repeated experiment authoring and publication overhead. Hourly reviews recorded that workflow failure. The useful progress to preserve is the exact interaction machinery, the converged comparisons and their limitations, the two explicit shared-component examples, and the sharper tests of what their inputs must contain—not the raw number of experiments.

**Reading order if you want the core details:** this report; the [original interaction-path proposal](interaction_path_decomposition_proposal_2026-09-11.md); the [joint sparse path results](../../COUPLED_SPARSE_PATH_V1_MATH.md); the [shared cubic attention derivation](../../SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md); and the [current portable branch interface](../../extracted_circuits/regional_shared_head2_token_mixed_v1/README.md).


## 10. Latest follow-up: folding the regional readers into their upstream attention producers

**This is directly testing your suggestion to decompose a longer interaction path.** We took the regional branch's four current-state readers and folded them backward into attention8, attention9 and attention13, including their actual residual propagation coefficients. These producers had already been implicated by earlier causal tests; their selection is therefore behavior-informed. The subsequent factorization uses weights, not a fit to text examples. This is a component-conditioned upstream experiment, not another full-unembedding fit.

### What was fitted and what failed

The target is the joint QK1 × QK2 × value numerator of all 27 producer heads, writing into those four downstream readings. We tried a common dictionary of 16 cubic source features, with separately solved query-dependent consumers for each head. Equal coordinate names across layers do not mean equal activations: the producers receive different native states.

Two optimization starts completed in about 94 seconds. Both stopped with line-search failures, not convergence. Estimated coefficient capture was only **0.66–0.99% at the fitted position and 0.33–0.54% at the held position**. These are estimates from independent coefficient probes; the held position is not held-out language data. The candidate failed its main fit/stability criteria. [Terminal result](../../FOLDED_PRODUCER_CUBIC_NATIVE_V1_RESULT.json).

The negative-result audit found severe cancellation between nearly identical products. An independent QR solve agreed with the original objective and gradient, so this was not primarily a mistaken Gram-matrix calculation. An exact average/difference rewrite preserved the represented function while reducing condition numbers from about 8.3 million and 2,962 to **2.55 and 1.36**. This repairs representation conditioning, not the failed optimizer or low capture. [Audit](../../FOLDED_PRODUCER_CUBIC_V1_AUDIT.json).

### Another shared intermediate appeared, with a different structure

The earlier downstream regional block used a shared quadratic parent and two linear children. Here the more faithful simplification is a **shared linear parent and two quadratic children**:

$$
h_0(s)=c(s)\,[a(s)b(s)+t^2\delta a(s)\delta b(s)],
$$

$$
h_1(s)=c(s)\,[\delta a(s)b(s)+a(s)\delta b(s)].
$$

Each letter denotes a linear reading of the source state. The small scalar $t$ comes from the separation between the original nearly coincident products. Computing $c(s)$ once and reusing it gives the common parent. The exact rewrite first retains all difference terms; the displayed shared-parent form is a measured approximation that drops the tiny difference in one reader.

This simplification changes the complete fitted function by only **0.024–0.170%** across starts and positions. Forcing a shared quadratic parent instead causes **12.5–21.1%** error. Thus “find shared structure” needs to allow different intermediate degrees, rather than always searching for the same parent pattern. [Comparison](../../FOLDED_PRODUCER_SHARED_PARENT_V1_RESULT.json).

The complete two-child block recurs across starts with coefficient cosine **0.995–0.996**, although relative function disagreement remains about **8.8–10.1%**. Almost all its energy belongs to head13.0. These are coefficient statements; they do not establish semantic identity or native behavioral fidelity. [Recurrence](../../FOLDED_PRODUCER_BLOCK_RECURRENCE_V1_RESULT.json).

### The important interpretation: the dictionary may be the wrong primitive

Checking the existing head dossier and the actual weights revealed that the learned parent is almost exactly head13.0's leading folded value reader: cosine exceeds **0.999995** in both starts. It is not yet evidence of a newly discovered semantic variable. [Weight alias check](../../FOLDED_PRODUCER_PARENT_ALIAS_V1_RESULT.json).

Let $M$ map the concatenated current/first source state into the four downstream readings through this head's value/output weights. Its leading singular component is $\sigma u v^T$. Retaining it while preserving the complete joint QK operation gives

$$
\widehat F(q,s)=g(q,s)
(q^TAs)(q^TBs)\,\sigma u(v^Ts).
$$

This expression has a single value reading, but it retains a rich query-dependent source interaction through both QK forms. A small fixed source-product dictionary may require many terms to express that interaction. Low tensor-dictionary capture can therefore coexist with a relatively simple arithmetic program.

With 2,048 independent coefficient probes per position, this structured baseline retains approximately **63–65% of head13.0's folded coefficient energy**. Its full QK matrices remain stored, so this is **not a matched-cost win** over the cubic dictionary. It also retains normalization and native producer-state dependencies. [Baseline and sampling uncertainty](../../FOLDED_PRODUCER_STRUCTURED_BASELINE_V1_RESULT.json).

The next behavioral comparison is whether the selected value component reproduces the full head's contribution to the downstream regional readings and transfers a meaningful share of the three-producer group's cue effect. **That test is pending.** The explicit-reading executor already replays nine frozen reference cases bit-for-bit; that is an interface control, not a behavioral result. [Executor control](../../COMPILED_READING_HEAD_V1_CONTROL.json).

### A separate extraction clarification and a practical speedup

The intervening source-port test separated changes in four source features from changes in key normalization. Swapping just those features while retaining the recipient's normalization reproduces the full source-swap margin effect within **6.5–9.5%** in every tested cell. Swapping normalization alone fails badly. This helps specify the upstream information to generate, but does not justify deleting normalization or the opposing query correction. [Native effect receipt](../../REGIONAL_SOURCE_READ_NORM_EFFECT_V1_RESULT.json).

We also checked that these cached interventions can run through the last MLP and selected unembedding rows on CPU. Seven 96-context evaluations took about **0.14 seconds**, agreeing with the completed GPU effects to relative error $2.4\times10^{-5}$. This avoids waiting for the shared GPU queue when only selected-token margins are required; full-vocabulary loss is a different computation. Both managed runners were running when checked at 16:20 UTC. [CPU replay](../../REGIONAL_SELECTED_SUFFIX_CPU_V2_RESULT.json).

**What this changes in the proposal:** continue decomposing composed paths, but compare arithmetic representations that preserve joint bilinear operations against dictionaries that flatten them into fixed source factors. The new result supports that comparison; it does not yet show that deeper folding produces a more selective or independently executable circuit.
