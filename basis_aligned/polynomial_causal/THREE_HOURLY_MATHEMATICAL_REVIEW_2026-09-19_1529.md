# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T15:29:59Z**. Next mathematical deadline: **2026-09-19T18:29:59Z**. The prior review was written at `2026-09-19T12:25:52Z`, 184 minutes earlier, so the 175-minute freshness gate required a new review. This is one bounded live review, not an offline backfill. Historical `/workspace/...` paths were resolved against the live checkout `/workspace/tensor_language`.

This review used no agent, GPU/model forward, job, queue, timer, service, commit, push, external contact, primary-receipt mutation, shared-registry mutation, or `TYPED_FACE_EXTRACTION_V1` change. It wrote one uniquely named CPU control and receipt, this review, and a short board append. The newest hourly review remains `ACTIVE_TRACK: CIRCUIT` at 14:39; this review does not change hourly alternation.

## Verdict

Depth on the regional path remains the highest-information plan. The new exact weight-only certificate closes one tempting shortcut: head 9.8's 128-dimensional output image remains rank 128 after composition with each tested immediate block-10 native reader—attention `Q1`, `K1`, `Q2`, `K2`, `V`, and both MLP input maps. Thus the next native block does **not** expose an exact linear consumer quotient of width below 128 at those boundaries.

This sharpens, but does not replace, the live plan. A smaller program must exploit an explicitly declared task/readout quotient, an approximate quotient frozen before fresh tests, dependence on the attainable activation manifold, causal sparsity, or a later boundary. It does not license dropping any of the seven `QK1*QK2*V` finite-difference products. It also does not show that all 128 directions are reachable or behaviorally used.

Evidence type is **fold/weight-only**. It is neither edit, response, nor fit evidence and provides no native causal-fidelity proof. Current five-property verdict: **held-out/OOD prediction PARTIAL; extraction PASS only at intermediate regional boundaries; selective manipulation/removal PARTIAL; composition/reuse FAIL overall; measured simplicity LOCAL ONLY**.

## 1. Current native computation as mathematical objects

### Tensors, indices, shapes, graph, degree, and nonlinearities

Let positions be `t,s in {0,...,T-1}`, `s<=t`; residual/output coordinates `a,o in {1,...,d}`, `d=1152`; heads `h in {0,...,8}`; head coordinates `c in {1,...,d_h}`, `d_h=128`; MLP product coordinates `j in {1,...,m}`, `m=4608`; blocks `ell in {0,...,17}`; and vocabulary rows `v in {1,...,50304}`. The frozen model has 545,902,902 trained parameters and an untied unembedding.

With native `eps=1.1920928955078125e-7`,

```
S(x) = ||x||_2^2 / 1152 + eps,
RMS(x) = x / sqrt(S(x)).
```

For MLP `ell`, with `L,R in R^(4608 x 1152)`, `D in R^(1152 x 4608)`, and `b in R^1152`,

```
M_ell(x) = D[(Lx) odot (Rx)] / S(x) + b.
```

For a named residual decomposition `x=b0+a_i+m_j+d` (background, earlier attention source, earlier MLP source, installed delta), the numerator contains all self and ordered cross terms:

```
B(b0,b0), B(a_i,a_i), B(m_j,m_j), B(d,d),
B(a_i,m_j), B(m_j,a_i),
and both ordered terms coupling b0 or d to every other source.
```

The regional norm-closed MLP8 path retains baseline self, both ordered baseline/edit cross terms, edit self, live normalization, bias, skip, and learned re-entry. It omits only terms unchanged under the scoped intervention.

For attention head `(ell,h)`, after native RMS and rounded rotary maps,

```
p^r[t,s] = <Rot_t Q^r xhat_t, Rot_s K^r xhat_s> / 128, r in {1,2},
v[s,c]   = (V xhat_s)[c],
A[t,o]   = sum_{s<=t,c} O[o,c] p^1[t,s] p^2[t,s] v[s,c].
```

Each `Q^r,K^r,V` is `128 x 1152`; the head slice of `O` is `1152 x 128`. At fixed RMS denominators, a QK score is degree two in named residual sources and attention is degree five; an MLP numerator is degree two. Live RMS makes the maps rational with square roots. The final `30*tanh(U RMS(h)/30)` is non-polynomial, so no finite-degree whole-network claim is made.

If query and key residuals expand as `x_t=sum_i q_i[t]` and `x_s=sum_j k_j[s]`, then

```
p^r[t,s] = sum_{i,j} <Rot_t Q^r q_i[t], Rot_s K^r k_j[s]> / 128.
```

Earlier attention and MLP sources therefore create attention-self, MLP-self, attention-to-MLP, and MLP-to-attention terms in **each** QK factor. Multiplying the two scores and `V` creates every reader-permitted `QK1_term * QK2_term * V_source` interaction. The current path retains that full set; it does not keep only QK1, additive routing/value singles, or self terms.

The live contraction graph is

```
token IDs + native residual6
 -> all attention7 writes
 -> complete MLP7 and head8.2 readers
 -> head8.2 city-write delta
 -> native z8 + delta8
 -> norm-closed MLP8
 -> complete head9.8 QK1 * QK2 * V delta
 -> native block10 readers and blocks10..17
 -> final RMS -> untied unembedding -> softcap
 -> regional target, >=3 unrelated readers, full logits/CE.
```

### Tied parameters, gauges, backgrounds, outputs, norm, and price

Weights are tied across positions/examples, not layers. Learned residual/re-entry coefficients, attention value mixing, MLP bias, and rounded rotary semantics are literal parameters. Gauges include hidden-unit permutation; reciprocal `L_j -> cL_j, R_j -> R_j/c`; `L/R` exchange after symmetrization; compatible Q/K basis changes; `p1 -> c p1, p2 -> p2/c`; and value/output basis changes. Coordinates are not semantic units unless downstream interventions identify them.

Open backgrounds/ports are token IDs, native residual6, unrelated source/query content, RMS denominators, causal mask, rotary phases, learned mixing/re-entry, biases, unchanged attention fields, and the suffix. Outputs to preserve are signed regional effect, at least three unrelated-reader effects, full logits/CE, and named intermediate writes needed for replay.

Algebraic replay uses per-sequence relative `L2`, maximum absolute error, off-support zero, and term closure. Behavioral composition uses the Möbius interaction norm divided by the **smallest live** single effect and a preregistered matched-random-split distribution. A closed face identity is only accounting.

Literal native prices remain: 545,902,902 model parameters; six `128 x 1152` maps or 884,736 FP32 values for a two-QK head plus output slice before biases/mixing; 528 causal cells at `T=32`; 67,584 score-times-value-coordinate products plus 528 score-factor products over all positions (4,096 plus 32 at the final position). The regional norm-closed MLP8 package stores 16,536,963 FP32 values plus token table/cache. Shared projections, state, edges, adapters, and runtime must be deduplicated rather than summing overlapping package manifests. A full residual6-to-suffix price remains missing.

### Current circuit object

The current circuit object is regional city/spelling route `PATH-REGIONAL-CITY-001`, extended in the live plan beyond its registered residual6-to-head8.2 endpoint through norm-closed MLP8, complete head9.8, and the native suffix. Existing extraction replays on 40 opened fixtures and 40 fresh FineWeb sequences from 20 distinct document/context cells. Scoped fresh removals beat same-site nulls. However, packed FineWeb direction transfer is `88/102 < .90`, key/value composition has `interaction/smaller=2.3448`, and prior source/random partitions fail specificity. It is a path with intermediate extraction, not a complete five-property circuit.

The newer number, gender, and person routes remain noncanonical: number transfers on 62 later FineWeb documents but only four cue/label cells; gender copier/reader composition transfers on three sets but lacks smallest-piece matched-split specificity, standalone extraction, and price; person correctly terminates as spread after its seven-head set scores `.549 < .60`. Aspect is parked on syntactically misaligned pairs; temporal retains sign constancy but fails reader/carry/additivity predictions. These opened/edit/fold receipts do not displace the regional depth target.

### Current folded-path object and retained subset

For baseline fields `p1,p2 in R^(T x T)`, `v in R^(T x 128)` and deltas `dp1,dp2,dv`, with `F` the trilinear head contraction, the exact change is

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

The proposed folded path retains **all seven nonempty products**, every live self/ordered-cross source term in both QK factors and MLP8, live denominators, backgrounds, bias, skip, and learned re-entry. Only the unchanged baseline is omitted. The new consumer-rank result does not license a smaller exact subset.

## 2. Primary mathematics and object-to-object mapping

Austrin, Kaski, and Kubjas formulate multilinear maps as structure tensors and derive tensor-network lower bounds using matrix flattenings ([primary paper, Theory of Computing 2022](https://theoryofcomputing.org/articles/v018a016/)). Their framework applies exactly to the independent-port restriction of head9.8 but, as the authors emphasize through the tensor-network model, tensor rank alone need not price every alternate contraction topology.

For the existing head boundary, input modes are independent score fields `p1[e]`, `p2[e]`, values `v[s,c]`, and output `(t,o)`, where `e=(t,s)`. The structure tensor is supported only when both score cells equal `(t,s)`, with coefficient `O[o,c]`. The `(p1,v)|(p2,y)` flattening is block diagonal with one `O^T` block per live cell. The prior exact certificate established `rank(O)=128`, hence exact independent-port CP rank `N_cells*128`.

The present restriction composes the producer image with an immediate native linear reader `W`:

```
z[e,c] -> y[t,o] through O,
then y -> W y,
so the relevant output map is W O in R^(r x 128).
```

By the same block-flattening argument, the consumer-specific independent-port tensor has exact CP rank `N_cells * rank(WO)`. An exact quotient of dimension `k<128` at that boundary would require `rank(WO)<=k`. For each of seven block-10 maps, a `128 x 128` minor of `WO` is nonzero modulo the prime 65,521 after exact uniform float32-to-integer scaling. Therefore `rank_R(WO)=128`; at these boundaries the CP rank remains 4,096 at the final position and 67,584 over `T=32`.

Assumptions met: frozen finite float32 weights; exact real arithmetic; full 128-dimensional head image; independent formal score/value ports; native linear map before its following normalization/product; fixed causal support. Complexity is seven exact `128 x 1152` by `1152 x 128` modular products plus seven `128 x 128` eliminations.

Assumptions violated for any broad impossibility claim: native score/value fields share residual and RMS sources; only the first 128 consumer rows form each witness; attainable activations may occupy a lower nonlinear manifold; task behavior may use a scalar or approximate quotient; later normalization/products/softcap are nonlinear; alternate tensor-network topology may reuse work; and no data, edit, OOD, or causal evidence enters the certificate.

## 3. Executed CPU consequence

Control: [`head9h8_block10_consumer_rank_certificate_2026_09_19_1528.py`](head9h8_block10_consumer_rank_certificate_2026_09_19_1528.py). Receipt: [`HEAD9H8_BLOCK10_CONSUMER_RANK_CERTIFICATE_2026-09-19_1528.json`](HEAD9H8_BLOCK10_CONSUMER_RANK_CERTIFICATE_2026-09-19_1528.json). It used `/venv/main/bin/python`, CPU only, two BLAS/OpenMP threads, and checkpoint SHA256 `680d6c26...d6de3`. Runtime was 1.26 seconds.

| exact composed map | certified real rank | numerical condition number of witness |
|---|---:|---:|
| block10 Q1 first 128 rows times `O_9.8` | 128 | 766 |
| block10 K1 first 128 rows times `O_9.8` | 128 | 299,295 |
| block10 Q2 first 128 rows times `O_9.8` | 128 | 33,210 |
| block10 K2 first 128 rows times `O_9.8` | 128 | 2,373 |
| block10 V first 128 rows times `O_9.8` | 128 | 1,199 |
| block10 MLP Left first 128 rows times `O_9.8` | 128 | 268 |
| block10 MLP Right first 128 rows times `O_9.8` | 128 | 366 |

All three registered predictions pass: all five attention input maps retain rank 128; both MLP input maps retain rank 128; and no exact linear quotient below width 128 exists at a tested immediate boundary. The K1 witness is numerically ill-conditioned but its exact scaled-integer determinant is nonzero modulo the prime, so the real-rank conclusion does not rest on a floating-point threshold.

Executable consequence: do not spend the next fold on a generic exact rank reduction of the head9.8 image inside block10. First exact-replay the full regional delta and run the preregistered forward-response/factorial test. Only then test a **declared** task/readout quotient or approximate quotient frozen before distinct-cell validation.

## 4. Five-property scorecard

| property | current evidence | verdict |
|---|---|---|
| Held-out/OOD prediction | Regional extraction covers 20 distinct FineWeb document/context cells and has Pile tests, but packed FineWeb direction is `88/102`; newer breadth routes do not supply a frozen end-to-end formula on structurally varied cells. | **PARTIAL** |
| Extraction at a declared boundary | Residual6-to-attention8 and norm-closed MLP8 packages execute with declared ports. Complete head9.8 plus suffix remains open. The new rank certificate is not extraction. | **PASS at intermediate regional boundaries; incomplete end-to-end** |
| Selective manipulation/removal | Scoped regional fresh edits beat same-site nulls, but the complete path lacks frozen equal-norm directions, three unrelated readers, matched splits, and full-logit battery at one boundary. | **PARTIAL** |
| Composition/reuse | Regional `interaction/smaller=2.3448`; additive factorial lower bound is `.5862` of the smaller single; matched random-split specificity fails. Gender transfer lacks the full smallest-piece/null test. | **FAIL overall** |
| Measured simplicity | Local storage/state counts, a restricted trilinear rank floor, and now seven immediate-consumer rank witnesses exist. No deduplicated full-path executor price or matched-effect advantage exists. | **LOCAL ONLY** |

Lower replay error, storage, rank, variance, or CE cannot substitute for a missing behavioral property. Claims remain tagged as **fold/edit/response/fit** and **fresh/opened/replay**.

## 5. Organization reconciliation and repair decision

- `COMPUTATION_PATH_REGISTRY.md` contains `PATH-REGIONAL-CITY-001` but still stops at the head8.2 write; the live target includes `city_mlp8_norm_closed_v1`, complete head9.8, the new block10 rank boundary, and a pending suffix factorial. The registry is shared dirty state, so this review did not overwrite it.
- `CIRCUIT_REGISTRY.md` links the regional inherited-city edge and extensive head9.8 evidence. Number/gender/person evidence remains spread across primary receipts and narrative rather than canonical paths. `PATH-SUBJECT-001` is the older L11H3 object and must not be silently reused for the newer number route.
- `CIRCUIT_GRAPH_REGISTRY_V1.md` remains a generated 49-package inventory. It omits `city_mlp8_norm_closed_v1` and the live number/gender/person routes. Its `four_trait_verified` labels do not separately score measured simplicity and therefore are not five-property completion claims.
- `MODULE_DOSSIERS.md` records head9.8 QK source/carry terms and MLP8 evidence, while the MLP dossier index correctly treats MLP7/8 gaps as documentation debt. Dotted head aliases, zero-based tensor indices, unit/site pairs, and operational roles (`reader`, `writer`, `copier`, `hub`) still need canonical per-path alias tables.
- Current v594-v597 receipts preserve predictions and serial runtime but lack uniform top-level `started_utc`, evidence/evaluation type, distinct-context-cell manifest, and literal-price fields. Their row counts cannot stand in for distinct cells.
- `explanations/README.md` and `for_logan/LATEST.md` remain navigation aids rather than authorities for the post-September-18 breadth receipts. Latest commits, primary receipts, current registries, and board claims were used.
- No orphan was copied into a second narrative. The temporal and aspect null/park receipts remain primary records; the new certificate is uniquely named and linked here. Existing primary receipts and the primary-owned typed-face implementation were untouched.

The focused bounded action was the exact CPU consequence, not a shared-code repair. A declarative runner/receipt schema would still save the most repeated authoring, but no scientific candidate or sibling runner has been created since 12:25, the relevant registry is concurrently dirty, and equivalence testing across the many intervention families exceeds a safe micro-refactor. Adding another publisher/checker now would increase duplication. The next safe organizational repair, after ownership clears, is one canonical regional path cross-link plus required cell/evidence/price fields in the existing receipt path—not a parallel framework.

## 6. Efficiency audit

Window: `2026-09-19T12:25:52Z–15:29:59Z`. There were **zero** new scientific runners, result receipts, or commits after the prior mathematical review. The only later managed executions were successful canaries at 12:48, 13:18, 13:49, 14:19, 14:49, and 15:20. Both Supervisor runners were `RUNNING`; both queues were empty. No queue/runtime rescue or service action was justified.

The prior 09:19–12:25 burst created 95 receipts and 95 sibling runners, while successful model execution occupied only minutes. That established one-runner-per-candidate authoring, copied builders/scorers, stale guards, weak cell schemas, and delayed canonicalization as the dominant waste. The following three hours swung to the opposite failure mode: no scientific receipt, with hourly reviews and health checks as the only durable activity. The right correction is not more review ceremony; it is to execute the already frozen regional census/factorial and full fold using shared machinery, then canonicalize once.

Serial scientific latency in this window is undefined because no candidate began or ended. Uninstrumented labor and idle time remain unknown and are not fabricated. Canary runtime is health-check cost, not research evidence. The CPU certificate itself took 1.26 seconds; review/inspection time is not inferred from timestamps.

## 7. Cross-track consequences and actionable handoffs

**Circuit evidence selects the folding target.** The regional route has the deepest declared extraction boundary, genuine distinct-cell fresh evidence, selective same-site tests, material QK2 controls, and failed additive/random-split composition. Those facts select the complete residual6-to-head9.8 delta and later suffix, not another behavior census. They require exact port closure, full QK1-by-QK2-by-V interactions, both ordered MLP cross terms, and live context slots.

**Folded algebra proposes a circuit grouping/split.** The seven-product finite-difference identity defines a five-arm intervention: baseline; routing/key single; value single; both singles installed with explicit product terms suppressed; true joint product. The new rank witnesses say not to split head9.8 by a generic exact linear block10 bottleneck. If the factorial identifies a specific product interaction against matched splits, group those factors at the head consumer boundary. If it fails, split by a declared downstream task/readout quotient only after a forward response census identifies the actual users.

**CIRCUIT handoff:** run the frozen regional forward-response census before proposing a suffix, then the full-suffix five-arm factorial on structurally varied distinct `(template, city pair, endpoint)` cells. Retain all seven products. Score signed regional prediction, full logits/CE, at least three unrelated readers, 16 equal-norm same-site directions, and matched random splits. Normalize interaction by the smallest live single; a closed Möbius face is not a pass.

**WEIGHT_FOLDING handoff:** exact-replay `residual6 -> attention7/MLP7 -> head8.2 -> norm-closed MLP8 -> complete head9.8 vector delta`, retaining every reader-permitted self/ordered-cross term and all seven products. Deduplicate projections, static scalars/bytes, product nodes, graph edges, state, caches, and measured runtime. Do not pursue an exact block10 linear width reduction: the seven tested native readers preserve rank 128. Approximate/task quotients require a declared norm and later causal/fresh validation.

**Organization handoff:** when shared ownership clears, extend the existing regional path record to the norm-closed MLP8/head9.8/rank/factorial boundary and add one alias/cell/evidence/price schema. Then canonicalize number, gender, and person only to the level their primary evidence warrants. Do not duplicate changing narrative across registries.

## Limitations

The new certificate concerns exact weight-only ranks of seven linear maps composed with the head9.8 output image. It does not cover nonlinear RMS/product readers, the attainable native activation manifold, approximate or scalar behavioral quotients, alternate tensor-network topologies, later suffix computations, or causal behavior. The first 128 rows provide witnesses for each full map; they do not claim those rows are the regional users. No new model forward was run. The regional complete executor, suffix factorial, and deduplicated price remain absent. FineWeb signed transfer remains inconsistent and composition fails. No algebraic identity, modular witness, replay, rank, compression, storage reduction, or low error in this review proves native causal fidelity.
