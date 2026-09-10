# Unsupervised products and position-corrected QK — 10 September, updated 19:56 UTC

**Both experiments have now run.** The unsupervised fit found two repeatable products with pronoun-related output loadings, but most of the last-layer tensor remains unexplained. Correcting positional coordinates greatly improved the QK transfer between sentence frames; separating the two behaviors still failed. Neither result establishes a circuit with all four requested properties.

**Newest results:** a causal test confirms that the two products affect pronoun predictions, but rejects their sufficiency and selective removal. The 19:49 mathematical review derives and tests a constraint on exact sparse vocabulary connections. Folding the QK readers through their actual weights reveals partial overlap in the input quadratics. Details follow the original results below.

**Scope clarification:** these are two separate experiments. The unsupervised tensor fit composes only MLP17 and the unembedding. The QK work studies the existing two behaviors across their 26 nominated heads; it does not fold the unembedding through MLP17 into final attention. That longer path has not been jointly factorized here.

## What the unsupervised fit actually did

For each output token, the last bilinear layer contributes a quadratic numerator. We fitted all 50,304 model output rows jointly:

$$
S_v\approx\sum_{j=1}^{32} A_{vj}\operatorname{sym}(a_jb_j^\top),
\qquad A=UW.
$$

Each recovered computation is $(a_j^\top x)(b_j^\top x)$. Its column of $A$ describes its signed writes across the vocabulary. The residual writer $W$ lets the same component be executed inside the model, before final normalization. The residual path, final RMS normalization and score saturation remain explicit.

The optimizer adjusted the input readers jointly and solved the output writers by least squares at each step. It used **no token labels and no activations**. One run started from selected native products; another started randomly. Both received 240 fixed optimization steps. Gram contractions evaluated the full tensor error without constructing the vocabulary × input × input tensor.

| Program | Fraction of total squared coefficient norm captured |
|---|---:|
| 32 native products with refitted output writers | 2.76% |
| Learned products, native initialization | 4.66% |
| Learned products, random initialization | 4.28% |

These are coefficient-space measurements, not percentages of model behavior explained. The registered improvement bar was five percentage points for both starts; both failed. Only two products matched across starts at the fixed 0.95 full-term cosine bar, versus the required sixteen. The instrument passed. The run took 5.96 executor seconds and used zero model forwards.

The two repeatable products had term cosines 0.9984 and 0.9585 across starts. After fitting, their largest vocabulary loadings revealed she/her and he/his/him profiles. A factor's sign can reverse together with its output writer without changing the executed term; signs must be aligned before interpreting profiles.

The native remainder remains in the program. This is a small discovered component, not a 32-product replacement for the layer. There is no sparse-loading penalty. A subsequent CPU audit found every product loads nontrivially on all 50,257 valid vocabulary tokens. Large loadings can identify candidate consumer groups, but exact nonzero supports cannot distinguish groups here.

## Comparison with existing MLP17 work

The pronoun/gender readout was already documented in ledger §§1583 and 1589–1591. That work identified a pronoun-class quadratic eigenaxis, a signed response to reflecting it, and a 64-unit committee carrying the measured reflection effect. The current novelty is whether an all-token, unlabeled weight factorization recovers useful pieces of that operation.

A CPU comparison reconstructed the old weight-defined axis in FP64. Its projection length into the two readers of the he-related product is **0.961**; into the she-related product's readers it is **0.257**. The products therefore are not simply two copies of the old axis.

For a unit axis $v$, reflection about zero is $J=I-2vv^\top$. If $H_k$ is a native product's symmetric matrix, define

$$
t_k=H_kv-v(v^\top H_kv),\qquad T=D[t_k^\top]_k.
$$

Then the exact change of the bilinear output is

$$
B(Jx)-B(x)=-4(v^\top x)Tx.
$$

The two recovered products leave **90.35% relative full-vocabulary coefficient error** in this reflection operation. The old 64-unit committee leaves 72.10% under this same coefficient metric. These results do not contradict the committee's earlier behavioral result: coefficient error weights every input direction, whereas natural contexts occupy a restricted domain. The next test asks whether the two products explain the reflection's observed effect on natural pronoun contexts.

There is also a qualification to the old explanation. Its code reflected about an empirical mean $\mu$, not zero. For an eigenaxis with eigenvalue $\lambda$ and coordinate $s=v^\top x$, the class quadratic changes by

$$
\lambda(2\mu-s)^2-\lambda s^2=4\lambda\mu(\mu-s).
$$

Eigenvector status alone therefore does not make that historical intervention exactly invariant. Its measured CE/probability results remain evidence; the claimed exact-even interpretation needs this correction. The new CPU comparison uses zero-centered reflection explicitly and is not a replay of the historical intervention.

## What changed in the QK experiment

The task spaces are now fitted to the joint QK products **before** positional rotation. Both score factors remain involved. We invert the actual rounded position maps and reapply them during attention contraction; we do not assume ideal orthogonal rotations.

| Evaluation | Earlier own-routing error | Position-corrected error | Other-task routing effect after correction |
|---|---:|---:|---:|
| Correlative, original frame | 15.33% | 15.68% | 32.81% |
| Correlative, longer frame | 62.82% | 14.90% | 20.69% |
| Other behavior | 27.17% | 27.23% | 61.96% |

The longer-frame error improved substantially. Its paired finite-panel interval is 11.51–18.46%, compared with 61.91–63.79% before the coordinate change. These are the same previously opened rows, excluded from basis fitting; this is not independent OOD confirmation.

The fixed bars were at most 20% own-routing error, at least 80% projection recovery, and at most 20% other-task effect in every panel. The overall own-routing and cross-task-selectivity hypotheses still fail. This supports a reusable correlative routing space across the two frames, with unresolved sharing or coupling to the other task. It does not establish disjoint native weight subcircuits. Interventions edit joint product features, which need not correspond to a single realizable pair of native QK vectors.

The instrument passed. Managed execution used 28 model forwards over 224 sequences and took 7.28 executor seconds.

Primary receipts: [joint32 fit](../../UNSUPERVISED_JOINT32_V1_RESULT.json), [position-corrected QK](../../CORRELATIVE_JOINT_QK_SUBSPACES_V2_RESULT.json), [prior-art comparison and paired intervals](../../STABLE_JOINT32_PRIOR_V1_AUDIT.json), [exact support audit](../../UNSUPERVISED_JOINT32_V1_SUPPORT_AUDIT.json). Earlier experiments remain in [the preceding follow-up](joint_unembedding_and_qk_results.md).

## Natural-context causal test, 19:48

We used 96 cached corpus sequences, scoring 127 pronoun targets and 18,305 other targets after position 64. The products were not fitted to these activations. The corpus rows had been opened by earlier work, so this is not fresh/OOD confirmation.

The reference changes the complete last-MLP computation from $B(x)$ to $B(Jx)$, keeping the incoming residual fixed. The candidate changes only the recovered two-product component $P$:

$$
B(x)\longmapsto B(x)-P(x)+P(Jx).
$$

This retains the native remainder and tests whether the candidate explains the reflection operation. A separate removal subtracts $P(x)$. Both tests retain final normalization and score saturation.

| Fixed product pair | Full-vocabulary reflection-response error | Target-CE response error | Removal: pronoun absolute CE change | Removal: other absolute CE change |
|---|---:|---:|---:|---:|
| Native-initialized | 85.33% | 30.56% | .2000 nats | .01161 nats |
| Random-initialized | 85.61% | 32.59% | .1702 nats | .01204 nats |

The bars were 20% for both response errors, at least .05 nats for pronoun removal effect and at most .01 elsewhere. **Both scientific hypotheses failed.** The instrument passed: full-model-hook replay agreed within $9.3\times10^{-6}$ logits, and the product-delta identity within $3.6\times10^{-15}$ relative error. Execution took 2.238 seconds and 13 model forwards over 104 sequences.

These components are causally relevant but insufficient. Their other-target effect is close to the limit; it remains a failure. Row-bootstrap intervals for that effect are .01049–.01281 and .01113–.01299, respectively. We do not change the bar or filter contexts to rescue the claim. [Result](../../STABLE_JOINT32_REFLECTION_V1_RESULT.json), [paired intervals](../../UNEMBEDDING_SUPPORT_FEASIBILITY_V1_AUDIT.json).

## The mathematical cycle changed the sparse-factorization proposal

There is a structural restriction behind the dense loadings. A factor implemented as a residual writer $w$ has vocabulary coefficients $Uw$. To affect only a token set $S$, it must satisfy

$$
U_{S^c}w=0.
$$

If the unembedding rows outside $S$ span the residual space, only $w=0$ can satisfy that requirement. The tested pronoun groups all have full numerical complement rank. This rules out nonzero exact token-only writes for those groups within this interface, subject to the numerical rank qualification.

We also computed the greatest possible concentration for **any** residual writer:

$$
\max_{w\ne0}\frac{\|U_Sw\|^2}{\|Uw\|^2}
=\lambda_{\max}\!\left(U_S(U^\top U)^{-1}U_S^\top\right).
$$

The maximum is 10.45% for the six pronoun targets, 20.49% for the she-related product's twelve largest-loading tokens and 22.24% for the he-related product's twelve. Therefore even the best writer must place most of its squared loading elsewhere. This is not merely a failure of our optimizer.

The implication is specific: seek graded signed consumer patterns and shared internal computations; do not assume a sparse token-support graph is achievable through the existing unembedding. A different compiled output interface is possible in principle, but its extra computations and intervention semantics must be explicit. These coefficient bounds do **not** rule out behavioral selectivity—many small writes to unlikely tokens can have little effect on predictions. [Math review, derivation and primary references](../../THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1949.md).

## QK readers folded into the common input space, 19:54

A learned joint-feature reader can be reshaped as a $128\times128$ matrix $W$. Before positional rotation, its query coordinate is

$$
\frac{(Q_1x)^\top W(Q_2x)}{\rho_1(x)\rho_2(x)}
=\frac{x^\top\operatorname{sym}(Q_1^\top WQ_2)x}{\rho_1(x)\rho_2(x)}.
$$

The same construction applies to keys. We pulled back every fixed reader through both actual weight matrices and compared the two task spaces in this common input-quadratic metric. Within each head and role, both task spaces share the same normalization denominator. The quadratic metric still does not weight natural input frequency.

| Mean overlap across the 26 heads | Product-coordinate metric | Folded input-quadratic metric |
|---|---:|---:|
| Query spaces | .0720 | .1635 |
| Key spaces | .1658 | .2852 |

Overlap here is the mean squared cosine between optimally aligned basis directions, normalized by the smaller space's dimension. Zero indicates orthogonal spaces; one indicates containment. Folded query overlaps range .089–.220; keys .191–.366. Both preregistered extremes failed: not every space is nearly disjoint, and no head has mostly shared query **and** key spaces. The largest individual direction cosines are .870 and .945; no exact common direction is established.

This answers more directly whether the two behaviors read from different parts of the input product space: their nominated readers are distinct but partly overlapping, with more overlap after weight folding. It does not by itself show which overlap causes the measured cross-task effects. The CPU calculation used no model forwards, took 1.023 seconds, and matched direct trained-weight contractions within $2.4\times10^{-15}$. [Receipt](../../JOINT_QK_INPUT_PULLBACK_V1_AUDIT.json).
