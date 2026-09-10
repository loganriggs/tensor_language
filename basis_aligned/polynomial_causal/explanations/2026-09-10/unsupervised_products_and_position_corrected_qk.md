# Unsupervised products and position-corrected QK — 10 September, 19:45 UTC

**Both experiments have now run.** The unsupervised fit found two repeatable products with pronoun-related output loadings, but most of the last-layer tensor remains unexplained. Correcting positional coordinates greatly improved the QK transfer between sentence frames; separating the two behaviors still failed. Neither result establishes a circuit with all four requested properties.

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
