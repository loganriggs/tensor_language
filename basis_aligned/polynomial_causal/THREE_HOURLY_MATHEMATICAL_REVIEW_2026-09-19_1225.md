# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T12:25:52Z**. Next mathematical deadline: **2026-09-19T15:25:52Z**. The preceding review was written at `2026-09-19T09:19:22Z`, about 186.5 minutes earlier, so this review was due under the 175-minute gate. It is one live review, not an offline backfill. Historical `/workspace/...` repository paths were resolved against the live checkout `/workspace/tensor_language`; the checkpoint remains at its validated local cache path.

This bounded review used no agent, GPU/model forward, job, queue, timer, service, commit, push, external contact, or `TYPED_FACE_EXTRACTION_V1` change. It wrote one uniquely named CPU control and receipt, this review, and a short board append. The newest hourly review remains `ACTIVE_TRACK: WEIGHT_FOLDING` at 11:33; this checkpoint does not alter hourly alternation. Concurrent circuit work through v597 was read but not modified.

## Verdict

The authority-designated regional path still dominates another breadth expansion. Its vector endpoint is a genuinely full-width trilinear contraction, not merely seven labels in an expansion. The executed exact certificate proves that the frozen layer-9 head-8 output slice has column rank 128 over the reals. Therefore, when the two QK score fields and value field are treated as independent formal ports, the CP rank of

```
(p1, p2, v) -> y[t,o] = sum_{s<=t,c} O[o,c] p1[t,s] p2[t,s] v[s,c]
```

is exactly `number_of_live_(t,s)_cells * 128`: **4,096** rank-one trilinear terms at the final position and **67,584** over all 528 causal cells for `T=32`. This is both an upper bound from the displayed native sum and a lower bound from the `(p1,v) | (p2,y)` flattening, whose blocks are copies of the rank-128 output map.

Plan consequence: do not seek an exact full-vector saving by merely refactorizing the already exposed `QK1*QK2*V` port tensor. Any useful simplification must instead exploit a restriction that this theorem deliberately excludes: residual-source dependence among the fields, a smaller declared output/readout boundary, causal sparsity in `(t,s)`, or an approximate/freshly validated consumer quotient. Retain all seven finite-difference products until a preregistered forward factorial identifies a smaller consumer-relevant object.

This is **fold/weight-only** evidence and an exact arithmetic lower bound in a restricted grammar. It is not **edit**, **response**, or **fit** evidence; it is not fresh/OOD behavior, extraction, selective manipulation, composition, or native causal fidelity. In particular, a scalar regional readout can have far smaller tensor rank after contracting `O` with that readout, and native `p1,p2,v` are coupled through residuals and RMS. Tensor networks with a different topology may also reuse intermediate products. Algebraic replay and compression remain insufficient for causal adoption.

Current five-property score: **held-out/OOD prediction PARTIAL; extraction PASS only at declared intermediate regional boundaries; selective manipulation/removal PARTIAL; composition/reuse FAIL overall; measured simplicity LOCAL ONLY**.

## 1. Current native computation as mathematical objects

### Tensors, indices, shapes, contraction graph, and degree

Let positions be `t,s in {0,...,T-1}` with `s<=t`; residual/output indices `a,o in {1,...,d}`, `d=1152`; heads `h in {0,...,8}`; head coordinates `c in {1,...,d_h}`, `d_h=128`; MLP product coordinates `j in {1,...,m}`, `m=4608`; blocks `ell in {0,...,17}`; and vocabulary rows `v in {1,...,50304}`. The checkpoint contains 545,902,902 trained parameters and an untied unembedding. Weights are tied across positions/examples, not across layers. Learned residual/re-entry coefficients and attention value mixing are literal parameters.

For residual `x in R^1152`, native epsilon `eps=1.1920928955078125e-7`,

```
S(x) = ||x||_2^2/1152 + eps,
RMS(x) = x/sqrt(S(x)).
```

For MLP `ell`, `L,R in R^(4608 x 1152)`, `D in R^(1152 x 4608)`, and `b in R^1152`,

```
M_ell(x) = D[(L x) odot (R x)]/S(x) + b.
```

If `x = b0 + a_i + m_j + d`, naming background/residual, earlier attention, earlier MLP, and an installed write, its numerator contains every reader-permitted self and ordered cross term:

```
B(x,x) = B(b0,b0) + B(a_i,a_i) + B(m_j,m_j) + B(d,d)
       + B(a_i,m_j) + B(m_j,a_i)
       + all ordered terms involving b0 or d.
```

Thus an MLP input cannot be reduced to `a_i*a_i + m_j*m_j`: the native object separately includes `a_i*m_j` and `m_j*a_i`. The current regional MLP8 fold retains baseline self, both ordered baseline/edit cross terms, edit self, changed normalization, bias, skip, and learned re-entry. It omits only terms proved unchanged under the scoped intervention.

For attention head `(ell,h)`, after native RMS and rounded rotary maps,

```
p^r[t,s] = <Rot_t Q^r xhat_t, Rot_s K^r xhat_s>/128,  r in {1,2},
v[s,c]   = (V xhat_s)[c],
A[t,o]   = sum_{s<=t,c} O[o,c] p^1[t,s] p^2[t,s] v[s,c].
```

Each `Q^r,K^r,V` is `128 x 1152`; the selected head slice of `O` is `1152 x 128`. With fixed RMS denominators, a QK score is degree two in named residual sources and the head is degree five in those sources. A bilinear MLP numerator is degree two. Live RMS makes the maps rational with square roots; final `30*tanh(U RMS(h)/30)` is non-polynomial. No whole-network finite-degree claim is made.

For `x_t=sum_i q_i[t]` and `x_s=sum_j k_j[s]`, each QK factor expands as

```
p^r[t,s] = sum_{i,j} <Rot_t Q^r q_i[t], Rot_s K^r k_j[s]>/128.
```

Earlier attention and MLP sources therefore create attention-self, MLP-self, attention-to-MLP, and MLP-to-attention terms in **each** QK factor. The head then forms every permitted `QK1_term * QK2_term * V_source` interaction. The retained regional path proposes to keep the full set, not a QK1-only or additive routing/value subset.

The live contraction graph is

```
(token IDs, native residual6)
 -> all nine attention7 writes
 -> complete MLP7 plus five head8.2 readers
 -> head8.2 city-write delta
 -> native z8 + delta8
 -> norm-closed MLP8
 -> head9.8 QK1 * QK2 * V delta
 -> blocks9..17, final RMS, unembedding, softcap
 -> regional target, >=3 unrelated readers, full logits/CE.
```

### Tied parameters, gauges, backgrounds, outputs, and norm

Gauges include hidden-unit permutation; reciprocal `L_j -> cL_j, R_j -> R_j/c`; exchange of `L/R` after symmetrization; compatible Q/K basis changes; `p1 -> cp1, p2 -> p2/c`; and value/output basis changes. A coordinate or raw factor magnitude is not a stable semantic identity. Downstream operational equivalence under declared interventions is the relevant quotient.

Explicit backgrounds/open ports are native residual6, unrelated query/source content, RMS denominators, rounded rotary phases, causal mask, learned value mixing and re-entry, MLP biases, unchanged attention fields, and the blocks9-17 suffix. Allowed inputs are token IDs plus every native residual array declared at the chosen boundary. Outputs to preserve are the signed regional target effect, at least three unrelated-reader effects, full logits/CE, and named intermediate writes needed for exact replay.

Replay uses per-sequence relative `L2`, maximum error, off-support zero, and named-term closure. Behavioral composition uses the Möbius interaction norm divided by the smallest **live** single effect and a preregistered matched random-split distribution. A closed face identity is accounting, not proof that the interaction is small, selective, or random-split specific.

### Current circuit object

The primary circuit object is the regional city/spelling route `PATH-REGIONAL-CITY-001`, extended in the live plan from its registered residual6-to-head8.2 boundary through norm-closed MLP8 and the complete head9.8/suffix response. Existing extraction replays on 40 opened fixtures and 40 fresh FineWeb sequences from **20 distinct document/context cells**. Scoped fresh removals beat same-site nulls, but packed FineWeb direction transfer is `88/102 < .90`, key/value composition has `interaction/smaller=2.3448`, and prior source/random-split partitions fail specificity. It is a path with intermediate extraction and selective evidence, not a five-property circuit.

Concurrent circuit work since 09:19 adds useful but noncanonical routes: number v501-v548 closes two readouts and transfers on 62 distinct later FineWeb document contexts but only four cue/label cells; gender v549-v582 closes a five-reader/copier grouping on three sets; person v579-v593 ends as a spread route after the seven-head set misses its `.60` kill bar; aspect v596 is parked because `since/by` pairs are syntactically misaligned; temporal v597 passes sign constancy but fails its reader/carry/additivity predictions. These are primarily **edit/fold on opened or claim-relative fresh rows**, without standalone boundaries, canonical port lists, full matched nulls, or deduplicated prices. They do not displace the regional depth target.

### Current folded-path object and retained subset

For baseline fields `p1,p2 in R^(T x T)` and `v in R^(T x 128)` with deltas `dp1,dp2,dv`, let `F` be the head contraction above. The exact finite change is

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

The current retained subset is **all seven nonempty products**, plus every live self/ordered-cross residual-source term in MLP8 and each QK factor. The only omitted head term is the unchanged baseline. Existing composition failures, material QK2 controls, and the new full-vector rank certificate do not license a smaller exact subset.

### Literal program price

The checkpoint price is 545,902,902 trained parameters. A native attention head exposes six `128 x 1152` maps when two Q/K pairs, V, and its output slice are counted: 884,736 FP32 values before biases/mixing. At `T=32`, there are 528 causal cells. The native trilinear contraction uses 67,584 score-times-value-coordinate products after 528 score-factor products; at the final position it uses 4,096 plus 32. Shared projections and the output projection must be charged once.

The executed certificate establishes a restricted full-vector CP-rank floor of 67,584 trilinear terms over all positions and 4,096 at the final position. It does not lower-bound a scalar readout, an approximate program, or a program exploiting dependent native fields. The norm-closed MLP8 interface still stores 16,536,963 FP32 values plus its token table and derived cache; upstream regional packages overlap weights, so summing package manifests double-prices them. A canonical deduplicated residual6-to-head9.8 count of static scalars/bytes, product nodes, edges, open-state scalars, and measured execution remains missing. There is no whole-path simplicity pass.

## 2. Primary mathematics and precise mapping

The best-matching algorithmic framework is tensor-network/arithmetic complexity of multilinear maps. Austrin, Kaski, and Kubjas define a multilinear map by its structure tensor, price tensor-network executions by contraction dimensions, and use matrix-flattening ranks as execution lower bounds ([primary paper, 2022](https://toc.cs.uchicago.edu/articles/v018a016/v018a016.pdf), especially the introduction's flattening argument). Strassen's tensor-rank program relates rank decompositions to optimal multilinear computation ([primary paper, 1983](https://doi.org/10.1016/0024-3795(83)80041-X)).

Object-to-object mapping:

- Map input mode 1 to independent `p1[t,s]` variables over live causal cells `e=(t,s)`.
- Map input mode 2 to independent `p2[e]` variables.
- Map input mode 3 to value variables `v[s,c]`.
- Map the output mode to `(t,o)` with coefficient tensor `T[e1,e2,(s,c),(t,o)] = 1[e1=e2=(t,s)] O[o,c]`.
- Flatten rows as `(p1-cell, value-index)` and columns as `(p2-cell, output-index)`. After deleting structural zero rows/columns and permuting, the matrix is block diagonal with one `O^T` block per live causal cell. Its rank is `N_cells * rank(O)`.
- The native sum over `(e,c)` is a CP decomposition with `N_cells * 128` terms. If `rank(O)=128`, lower and upper bounds meet exactly.

Assumptions met: frozen finite float32 weights; full vector head output retained; independent formal score/value ports; exact real arithmetic; fixed causal support. The rank certificate is exact because multiplying every float32 entry by `2^149` produces an integer matrix, and the first 128 output rows have nonzero determinant modulo the prime `2^61-1`. A nonzero modular determinant proves the dyadic matrix is nonsingular over the rationals/reals.

Assumptions violated or absent for a broader native claim: `p1,p2,v` share residual and RMS sources; rotary/QK projections constrain attainable fields; the behavioral output may be a scalar quotient; approximation is allowed; tensor-network topology can reuse intermediates; later normalization and softcap are nonlinear; and no causal or distributional evidence enters the theorem. Consequently this is a precise restriction, not an impossibility result for all transparent regional programs.

Computational complexity of the certificate is one 128x128 modular elimination, `O(128^3)` field operations, plus an optional numerical SVD of `1152x128`. It loads weights on CPU and performs no model forward.

## 3. Executed CPU consequence

Control: [`head9h8_trilinear_rank_certificate_2026_09_19_1224.py`](head9h8_trilinear_rank_certificate_2026_09_19_1224.py). Receipt: [`HEAD9H8_TRILINEAR_RANK_CERTIFICATE_2026-09-19_1224.json`](HEAD9H8_TRILINEAR_RANK_CERTIFICATE_2026-09-19_1224.json). It used `/venv/main/bin/python`, CPU only, two BLAS/OpenMP threads, and the checkpoint whose SHA256 is `680d6c26...d6de3`. Runtime was 1.19 seconds.

All three registered predictions pass:

| certificate | result |
|---|---:|
| exact rank of `O_9.8 in R^(1152x128)` | **128** |
| determinant of first 128 rows after exact dyadic scaling, mod `2^61-1` | `1783287831876411080` (nonzero) |
| numerical smallest singular value / condition number | `3.64335 / 3.35355` |
| independent-port CP rank, final position at `T=32` | **4,096** |
| independent-port CP rank, all causal cells at `T=32` | **67,584** |

Executable plan consequence: treat `QK1*QK2*V` as one irreducible full-vector port tensor at the present boundary. Search for savings only by (a) exact source restrictions/port closure, (b) contracting into a declared smaller downstream reader space and then testing that quotient causally, or (c) an approximate program frozen before distinct-cell validation. Do not count a rearrangement of the same 67,584 trilinear terms as simplicity.

## 4. Five-property scorecard

| property | current evidence | verdict |
|---|---|---|
| Held-out/OOD prediction | Regional intermediate extraction replays on 20 distinct FineWeb document/context cells and has Pile tests, but packed FineWeb direction is `88/102`. Number transfers on 62 later FineWeb documents but only four cue/label cells and lacks a frozen end-to-end formula. Gender/person claims have claim-specific transfers, not a declared general executor. | **PARTIAL** |
| Extraction at a declared boundary | Regional residual6-to-attention8 and norm-closed MLP8 packages execute with declared ports. Complete head9.8 and suffix remain open. Number/gender/person routes are not standalone and lack complete port lists. | **PASS at intermediate regional boundaries; incomplete end-to-end** |
| Selective manipulation/removal | Regional scoped fresh edits beat same-site nulls and preserve declared readers, but corpus directions vary. New gender number collateral is nonzero and the broad routes lack equal-norm directions plus a three-reader battery at a complete boundary. | **PARTIAL** |
| Composition/reuse | Regional interaction/smaller is `2.3448` and additive factorial RMS lower bound is `.5862` of the smaller single. Prior random splits fail specificity. Gender has a useful copier+reader composition and person has local composition, but neither has the complete fresh smallest-piece Möbius/matched-split test. | **FAIL overall** |
| Measured simplicity | Regional components have local storage/state counts, an additive lower bound, and now an exact restricted full-vector trilinear rank. No deduplicated full-path executor price or matched-effect comparator exists. Current breadth routes have named heads/units but no static/runtime/port price. | **LOCAL ONLY** |

Lower error, storage, rank, replay, variance, or CE cannot stand in for any missing behavioral property. Evidence types remain separate: **fold**, **edit**, **response**, **fit**; evaluation status remains **fresh**, **opened**, or **replay**.

## 5. Organization reconciliation

- `COMPUTATION_PATH_REGISTRY.md` has a useful uncommitted `PATH-REGIONAL-CITY-001`, but its endpoint still stops at the head8.2 write while the live weight-folding handoff extends through MLP8/head9.8. It has no canonical `PATH-SUBJECT-NUMBER-002`, pronoun-gender path, or person path. `PATH-SUBJECT-001` is an older L11H3 response object and cannot be repurposed silently.
- `CIRCUIT_REGISTRY.md` and the live number narrative contain much of v501-v597, but module facts, aliases, primary receipts, corrections, and missing gates are not represented once in a canonical path record. The newest aspect/temporal screens are terminal nulls/parks and should remain primary receipts rather than seed more duplicated prose.
- `CIRCUIT_GRAPH_REGISTRY_V1.md` remains a generated 49-package inventory. It omits `city_mlp8_norm_closed_v1` and all live number/gender/person routes. Its two `four_trait_verified / next gap: none` rows do not separately price simplicity and therefore are not five-property completion claims.
- `MODULE_DOSSIERS.md`, the module index, and the MLP dossier index preserve regional MLP7/8 prior art but do not cross-link the current number hub `MLP8 829/953/1030`, gender copier heads `8.1/6.1`, person readers/writers, or their claim-specific evidence states.
- Alias debt is active: dotted head labels versus zero-based indices; “reader,” “writer,” “copier,” “hub,” and “detector” as operational roles; pronoun/verb/gender/person axes; unit ID plus intervention site. These need one alias table per canonical path, not module renaming.
- Current receipts v594-v597 preserve predictions and internal prices, but the sampled top-level JSON fields `price`, `scope`, and `cells` are absent/null. Distinct context cells must become a required schema field; row totals are not a substitute. The 62-document number transfer still spans only four cue/label cells.
- `explanations/README.md` and `for_logan/LATEST.md` remain regional-oriented while the live number narrative changes concurrently. Historical `LATEST` pointers are therefore navigation hints, not current authority. Latest commits, primary receipts, registries, and board claims were used instead.
- Primary ownership of `TYPED_FACE_EXTRACTION_V1` was respected. No primary receipt, shared registry, dossier, explanation index, or concurrent implementation was modified.

## 6. Efficiency audit

From 09:19:22 through the v597 receipt at 12:21:34, the follow-up directory records **95 result receipts**, **1,136 declared forwards**, and **96 new `run_*.py` files**. All 95 receipts have a serial runtime; these sum to **190.05 seconds**, median **1.92 seconds**. Git records 99 commits since the cutoff. This is about one receipt per 1.92 wall minutes while successful model execution consumed only 3.17 minutes. Active design, authoring, interpretation, publication, corrections, and idle time are not phase-measured, so commit gaps are not labeled labor or idle time.

The dominant waste remains one-runner-per-candidate authoring and copied schema/data/scoring logic, not model compute. The v595 first execution failed and reran; v596 demonstrates that an in-place lexical swap can produce syntactically nonaligned pairs; v597 immediately opens another behavior while canonical number/gender/person records remain absent. The latest top-level receipts omit a uniform cell manifest and literal price fields. This makes handoffs brittle and encourages row-count claims.

Both Supervisor-managed runners were `RUNNING`; both queues were empty at inspection. The latest lane-1 log completed v597 at 12:21:34. A concurrent process was actively editing the shared aspectual library and preparing its own managed work. No queue/runtime rescue was needed, and changing Supervisor, queues, timers, or another worker's files was prohibited.

The best shared repair remains a declarative intervention runner with one row builder, factor/value/write arm schema, scorer, actual-batch price, semantic slot assertions, cell manifest, and receipt writer. I did **not** alter it: the exact shared library was under active concurrent modification, and equivalence tests across the current intervention families exceed a safe micro-refactor. A second checker would increase duplication. The bounded nonconflicting repair/consequence was instead the exact CPU rank certificate above, which supplies a durable price baseline for the designated fold.

## 7. Cross-track consequences and handoffs

**Circuit evidence selects the folding target.** The regional route has the deepest declared extraction boundary, actual distinct-cell fresh evidence, selective same-site tests, material QK2 controls, and a failed additive/random-split decomposition. Those facts select the complete residual6-to-head9.8 regional delta, not another opened behavior census. They require full QK1-by-QK2-by-V interactions, exact port closure, both ordered MLP cross terms, and the live context slot.

**Folded algebra proposes a circuit grouping and intervention.** The rank certificate says that at the full-vector independent-field boundary the native head is one full-width trilinear object. Combined with the seven-term finite-difference expansion, it proposes a five-arm test: baseline; routing/key single; value single; both singles installed with explicit product terms suppressed; true joint product. If the product-specific arm is fresh, selective, and specific against matched splits, group the factors at their consumer boundary. If it fails, contract first into declared downstream readers and test a split in that quotient rather than calling QK/value labels separate circuits.

**WEIGHT_FOLDING handoff:** complete and exactly replay

```
residual6 + token IDs
 -> all attention7 + complete MLP7/head8.2
 -> norm-closed MLP8
 -> complete head9.8 vector delta.
```

Retain every ordered source term and all seven factor products. Price projections, the 528 score-factor products, the 67,584 score-value products, output projection, open state, edges, caches, and runtime once. Compare against the new exact independent-port rank floor, but seek savings only through closed source ports or a declared consumer quotient. Validate on 40 opened fixtures and at least 20 genuinely distinct fresh document/context cells; preserve exact off-support zero.

**CIRCUIT handoff:** before proposing any suffix, run a forward response census for the frozen regional edit. Then execute the preregistered five-arm full-suffix factorial on structurally varied distinct `(template, city pair, endpoint)` cells with signed predictions, at least three unrelated readers, full logits/CE, 16 equal-norm same-site directions, and matched random splits. Score interaction relative to the smallest live single. A closed Möbius face, the new rank identity, or algebraic replay is not a pass.

**Organization handoff:** after the live breadth sequence freezes, add canonical path records—first `PATH-SUBJECT-NUMBER-002`, then gender/person only if their evidence warrants it—with aliases, distinct-cell manifests, primary receipts, corrections, port lists, evidence tags, missing gates, and one literal price. Do not copy the changing narrative across registries.

## Limitations

The new theorem/certificate prices an exact full-vector independent-port trilinear tensor. It does not cover the dependent native activation manifold, smaller behavioral readouts, approximation, arbitrary tensor-network topology, later nonlinearities, or causal behavior. It uses no new model forward. The regional full head/suffix executor and deduplicated program price remain incomplete. FineWeb signed behavior is inconsistent and composition fails. Concurrent number/gender/person work is rich but post-selected, partly opened, incompletely indexed, and not standalone. No algebraic identity, modular rank witness, compression, storage saving, low error, or low rank in this review proves native causal fidelity.
