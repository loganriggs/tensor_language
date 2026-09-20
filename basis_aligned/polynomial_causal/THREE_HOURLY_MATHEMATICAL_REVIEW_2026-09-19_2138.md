# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T21:38:47Z**. Next mathematical deadline:
**2026-09-20T00:38:47Z**. The preceding review was written at
`2026-09-19T18:34:18Z`, about 184 minutes earlier, so the 175-minute freshness
gate required one live review. This is not an offline backfill. Historical
`/workspace/...` paths were resolved against `/workspace/tensor_language`.

This bounded review used no agent, GPU/model forward, queue or service change,
timer change, commit, push, external contact, existing primary-receipt mutation,
or `TYPED_FACE_EXTRACTION_V1` change. It wrote one uniquely named CPU control,
its receipt, this review, and a short board append. The newest hourly review
remains `ACTIVE_TRACK: CIRCUIT` at 20:55; this checkpoint does not alter hourly
alternation.

## Verdict

The regional route remains the depth target, but the next response census must
not be a first-order or head-only surrogate for the full joint edit. An exact
five-arm decomposition on the existing 20 opened context cells gives

```
total key-by-value interaction
  = transmitted explicit head-product response + native-suffix mixed curvature.
```

For the target readout, suffix curvature is **0.4583 times the total interaction**;
the head-product response is **1.2525 times the total** because the two oppose
with cosine **-0.6784**. Even the best scalar rescaling of the head response
leaves **0.3367 relative L2 error**. Across the 20 actual context cells, the
curvature/total ratio has median **0.4054** and range **0.0440–1.1736**; the
curvature opposes the head term in 12/20 cells. Therefore the pending response
census must decompose the exact joint arm and retain a separate mixed-curvature
account, rather than infer the suffix from the local head product or singles.

This is **response algebra on stored opened edits**, not fresh/OOD evidence,
selective manipulation, random-split specificity, extraction, or proof of native
causal fidelity. Five-property verdict: **held-out/OOD prediction PARTIAL;
extraction PASS only at intermediate declared boundaries; selective
manipulation/removal PARTIAL; composition/reuse FAIL overall; measured
simplicity LOCAL ONLY**.

## 1. Current native computation as mathematical objects

### Tensors, indices, shapes, contractions, and degree

Let positions be `t,s in {0,...,T-1}` with `s<=t`; residual/output coordinates
`a,o in {1,...,d}`, `d=1152`; heads `h in {0,...,8}`; head coordinates
`c in {1,...,d_h}`, `d_h=128`; MLP product coordinates
`j in {1,...,m}`, `m=4608`; blocks `ell in {0,...,17}`; and vocabulary rows
`v in {1,...,50304}`. The model has 545,902,902 trained parameters and an
untied unembedding.

With native `eps=1.1920928955078125e-7`,

```
S(x) = ||x||_2^2 / 1152 + eps,
RMS(x) = x / sqrt(S(x)).
```

For MLP `ell`, `L,R in R^(4608 x 1152)`, `D in R^(1152 x 4608)`, and
`b in R^1152`,

```
M_ell(x) = D[(Lx) odot (Rx)] / S(x) + b.
```

If an MLP input is decomposed as `x=b0+a_i+m_j+d`, where `a_i` is an earlier
attention write and `m_j` an earlier MLP write, its numerator contains the
self terms `B(a_i,a_i)`, `B(m_j,m_j)`, both ordered cross terms
`B(a_i,m_j)`, `B(m_j,a_i)`, the background/edit terms `B(b0,d)`, `B(d,b0)`,
and `B(d,d)`, plus all other reader-permitted source pairs. The current regional
path retains every changed self and ordered-cross term; only terms identical in
baseline and intervention may cancel from a delta, while their background remains
an explicit port.

For attention head `(ell,h)`, after native RMS and rounded rotary maps,

```
p^r[t,s] = <Rot_t Q^r xhat_t, Rot_s K^r xhat_s> / 128,  r in {1,2},
v[s,c]   = (V xhat_s)[c],
A[t,o]   = sum_{s<=t,c} O[o,c] p^1[t,s] p^2[t,s] v[s,c].
```

Each `Q^r,K^r,V` is `128 x 1152`, and `O` is `1152 x 128`. If
`x_t=sum_i q_i[t]` and `x_s=sum_j k_j[s]`, then

```
p^r[t,s] = sum_{i,j} <Rot_t Q^r q_i[t], Rot_s K^r k_j[s]> / 128.
```

Thus earlier attention and MLP sources generate attention-self, MLP-self,
attention-to-MLP, and MLP-to-attention terms in **each** QK factor. Multiplying
the two expanded scores and `V=sum_k V_k` produces all permitted
`QK1_(i,j) * QK2_(i',j') * V_k` interactions. The retained object is the full
product, not separate QK1/QK2 analyses or only additive routing/value singles.

With fixed denominators, an MLP numerator has degree two, each QK score degree
two, and the attention numerator degree five. Live RMS makes the maps rational
with square roots; final `30*tanh(U RMS(h)/30)` is non-polynomial. No finite
polynomial degree is claimed for the whole native network.

The current contraction graph is

```
token IDs + native residual6
 -> attention7 writes + complete MLP7
 -> head8.2 city-write delta
 -> native z8 + delta8 -> norm-closed MLP8
 -> complete head9.8 QK1 * QK2 * V delta
 -> block10 readers -> blocks10..17
 -> final RMS -> untied unembedding -> softcap
 -> regional target + >=3 unrelated readers + full logits/CE.
```

### Ties, gauges, backgrounds, outputs, norms, and literal price

Weights are tied across positions/examples, not across layers. Learned residual
and re-entry coefficients, signed value mixing, MLP bias, causal masks, rounded
rotary phases, and token-only first-value branches are literal program parts.
Gauges include hidden-unit permutation; reciprocal MLP-row scaling
`L_j->cL_j, R_j->R_j/c`; `L/R` exchange after symmetrization; compatible Q/K
basis changes; reciprocal scale transfer between the two QK factors; and
compatible V/O basis changes. Raw coordinates are not semantic units without a
downstream operational test.

Open backgrounds and ports are token IDs, native residual6, unchanged
query/source states, every RMS denominator, rotary phase, causal support,
learned mixing/re-entry, biases, unchanged attention fields, and the native
suffix. Outputs to preserve are the signed regional effect, at least three
unrelated-reader effects, full logits/CE, and intermediate writes required for
replay.

Algebraic checks use relative L2, maximum absolute error, term closure, and
off-support zero. Behavioral composition uses the Mobius interaction norm
relative to the **smallest live** piece and a preregistered matched-random-split
distribution. A closed face identity is accounting, not evidence that an
interaction is small or semantically specific.

Literal prices: a native MLP stores 15,926,400 scalars; one complete two-QK
head interface stores 884,736 FP32 weights. At `T=32`, the head has 528 causal
cells, 67,584 score-times-value-coordinate products, and 528 score-factor
products. The norm-closed MLP8 package stores 16,536,963 FP32 values plus its
token table/cache. Existing residual6 packages are roughly 22–23 million FP32
weights and retain native state/suffix ports. Shared arrays, masks, adapters,
caches, state, edges, products, peak memory, and runtime must be deduplicated;
the complete residual6-to-head9.8 and suffix bill remains absent.

### Current circuit object

The circuit candidate is `PATH-REGIONAL-CITY-001`: residual6/token inputs through
attention7, MLP7, head8.2, norm-closed MLP8, complete head9.8, and eventually the
native suffix. Exact extraction exists at intermediate boundaries on 40 opened
fixtures and 40 fresh FineWeb sequences from 20 distinct context cells. Scoped
fresh removals beat same-site nulls. However, packed FineWeb direction remains
`88/102 < .90`; key/value interaction is `2.3448x` the smaller single; source
and random partitions fail specificity; and the complete suffix is not extracted.
It is a path, not a completed circuit.

### Current folded-path object

For baseline head9.8 fields `p1,p2,v` and deltas `dp1,dp2,dv`, trilinearity gives

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

The proposed fold retains all seven nonempty products, every permitted upstream
self/ordered-cross term, live denominators, bias, skip, learned re-entry, and
unchanged background ports. The opened seven-term census shows that omitting the
four order-at-least-two terms costs `.2853` relative local-head L2. Exact rank
certificates show the 128-dimensional head output image remains rank 128 through
all seven tested immediate block10 readers; they do not establish reachability or
causal use.

## 2. Primary mathematics and precise mapping

Janizek, Sturmfels, and Lee's **Integrated Hessians** constructs pairwise neural
interactions by integrating mixed second derivatives along a path
([primary paper](https://arxiv.org/abs/2002.04138)). The directly relevant
restriction here is simpler and exact. Let `S(z)` be the native suffix from the
installed head9.8 write to the five scored readouts, let `u` and `v` be the
key-only and value-only write deltas used in the summed-mains arm, and let `h`
be the explicit native head-product correction. Then

```
C = S(z+u+v) - S(z+u) - S(z+v) + S(z)
  = integral_0^1 integral_0^1 D^2 S(z+su+tv)[u,v] ds dt,
H = S(z+u+v+h) - S(z+u+v),
I_total = C + H.
```

The double-integral identity follows from the fundamental theorem of calculus.
Its assumptions are aligned arms, a common background, and twice differentiable
`S` along the rectangle. Native RMS has positive epsilon and tanh is smooth, so
the mathematical suffix satisfies them. The stored artifact supplies the four
rectangle corners and the fifth head-product arm, so the finite-difference
version costs `O(5*N*R)` additions for `N=240`, `R=5` and needs no Hessian or
model forward. Unlike a local Hessian approximation, the stored finite difference
includes all orders along the full edit rectangle.

The mapping does **not** inherit a causal-identification theorem from Integrated
Hessians. `u` and `v` share native residual/RMS causes; the cells are opened;
the directions were selected earlier; and no matched random split is present.
It is an exact decomposition of these interventions, not an explanation of why
the model learned them.

Cui et al.'s tensor-network max-flow/min-cut result defines network flow as the
rank of the contracted input-output linear map and the cut as a product of bond
dimensions, but proves equality can fail in general
([primary paper](https://arxiv.org/abs/1508.04644)). A fixed-normalizer linearized
slice of the regional contraction can be viewed as such a map, and every cut is
an upper bound on rank. The actual object violates the clean generic setting:
it has tied tensors, shared residual sources, live RMS, products, and a nonlinear
suffix. Therefore graph width or a cut count is not an exact simplicity
certificate here. The existing modular rank witnesses at the actual frozen
maps, plus explicit execution price, are the appropriate restricted certificates.

## 3. Executed CPU consequence

Control:
[`regional_five_arm_curvature_decomposition_2026_09_19_2137.py`](regional_five_arm_curvature_decomposition_2026_09_19_2137.py).
Receipt:
[`REGIONAL_FIVE_ARM_CURVATURE_DECOMPOSITION_2026-09-19_2137.json`](REGIONAL_FIVE_ARM_CURVATURE_DECOMPOSITION_2026-09-19_2137.json).
It used `/venv/main/bin/python`, CPU only, two BLAS/OpenMP threads, float64 stored
arms, no model load, and no forward. Both input hashes, `[5,240,5]` shape,
20-cell manifest, and zero-error algebraic closure passed.

| scope | suffix curvature / total | head response / total | curvature-head cosine | best rescaled-head error |
|---|---:|---:|---:|---:|
| target | **.4583** | **1.2525** | **-.6784** | **.3367** |
| four controls | .3961 | 1.1679 | -.5629 | .3274 |
| all five | .4569 | 1.2506 | -.6762 | .3366 |

Registered predictions: exact closure passed; curvature `<=.10` of total failed;
rescaled-head replay error `<=.10` failed. This redirects the pending circuit
instrument: decompose the recursively recomputed **joint** response by module and
record the mixed suffix term explicitly. Do not use the earlier QK1-only response
census, a linear response to `h`, or `H` alone to name a suffix.

## 4. Five-property scorecard

| property | current evidence | verdict |
|---|---|---|
| Held-out/OOD prediction | Regional intermediate extraction has 20 fresh FineWeb cells and scoped Pile transfer; packed FineWeb direction is `88/102`; this control is opened replay. | **PARTIAL** |
| Extraction at a declared boundary | Residual6-to-attention8 and norm-closed MLP8 packages execute with declared ports; complete residual6-to-head9.8 plus suffix is absent. | **PASS at intermediate boundaries only** |
| Selective manipulation/removal | Scoped fresh edits beat same-site nulls; complete joint path lacks equal-norm, matched-split, three-reader full-suffix controls. | **PARTIAL** |
| Composition/reuse | Key/value interaction is `2.3448x` the smaller single; suffix curvature is `.4583x` total and varies across cells; prior random/source splits fail. | **FAIL overall** |
| Measured simplicity | Local package prices and exact restricted rank floors exist; no deduplicated end-to-end bill or matched-effect advantage exists. | **LOCAL ONLY** |

Lower error, storage, rank, or CE does not replace a missing property. Evidence
remains tagged **fold/edit/response/fit** and **fresh/opened/replay**, with actual
distinct context cells rather than row counts.

## 5. Organization reconciliation

- `COMPUTATION_PATH_REGISTRY.md` now has `PATH-REGIONAL-CITY-001`, but its endpoint
  is still the head8.2 write. It does not yet join norm-closed MLP8, complete
  head9.8, the two rank certificates, seven-term census, this curvature receipt,
  pending response census, matched-null factorial, or deduplicated price.
- `CIRCUIT_REGISTRY.md` and `MODULE_DOSSIERS.md` contain the regional head8.2/
  head9.8 and MLP7/MLP8 evidence. The MLP7 addendum correctly keeps initial7,
  ordered crosses, normalization, and failed source ownership. The MLP8 dossier
  records exact directional responses and fresh reversal failures, but no one
  canonical residual6-to-head9.8 dossier joins them.
- `CIRCUIT_GRAPH_REGISTRY_V1.md` inventories 49 packages and still omits
  `city_mlp8_norm_closed_v1`. Its two `four_trait_verified` labels are not
  five-property completion: simplicity is unpriced there, and current composition
  requires smallest-piece and matched-split controls.
- The `CITY_INTERCHANGE_COMPOSITION_V1` preregistration, artifact, result, and new
  hash-bound consequence agree on five-arm semantics. The new receipt is a
  derived response-algebra control, not a replacement primary result.
- `explanations/README.md` has repeated “Latest regional result” headings;
  `for_logan/LATEST.md` is a narrative pointer, not authority. Current commits,
  primary receipts, registries, and board claims governed this review.
- Alias debt remains: dotted `head9.8` means zero-based block 9/head 8, operational
  names such as reader/writer/copier are roles, and several records mix package,
  module, and semantic names without an alias field.
- The concurrent temporal lane produced a new aligned `will/had` opened screen
  (`v597`, 43 text pairs, sign constancy `.935` in the board summary), while its
  registered value/reader/additivity predictions failed. It is discoverable by
  receipt and board but not a canonical computation path or dossier. It must not
  displace the user-designated regional depth target.

No registry/dossier repair was safe: the path registry and board are concurrently
dirty, and a live collaborator was editing/rerunning shared circuit code during
this review. The bounded mathematical control had clearer value and avoided
overwriting live work. When ownership clears, the highest-value organizational
repair remains one canonical regional path record with aliases, evidence/evaluation
labels, distinct-cell manifests, ports, and a deduplicated price—not another
publisher.

## 6. Efficiency audit

Window: `2026-09-19T18:34:18Z–21:38:47Z`. No regional model execution or complete
path receipt appeared before this CPU consequence. Six maintenance canaries ran
between 18:53 and 21:25 for about 123 serial seconds by runner start/end marks.
At 21:36 the concurrent lane produced temporal `v597` (15 forwards, 2.13 seconds),
and a subsequent `v598` run was being retried while this review was written;
that work is outside the regional handoff. Both managed runner shell processes
were present, queues were empty when inspected, and no queue/service action was
authorized or needed.

The repository contains thousands of versioned `run_*.py` files; normalized
stems expose families with dozens of near-duplicate runners (for example 53
`run_unit_family_separability_spec` and 39 `run_unit_broad_circuit` variants).
The regional family likewise repeats row builders, installed scorers, artifact
binders, and per-candidate preflight code. This creates stale-guard crashes,
duplicate reruns, brittle absolute paths, and receipt-schema drift. The active
collaborator's live shared-library work makes consolidation unsafe now. A future
repair should be a declarative arm/row/null manifest over one existing executor,
not a new compiler, publisher, or audit layer.

Review/health ceremony again exceeded measured regional science until the final
control. The new consequence itself is linear in a 6,000-value artifact and ran
in under one second wall time including interpreter startup. Serial regional
candidate latency remains undefined because the declared response census and
complete fold never started; uninstrumented labor and idle time are not invented.

## 7. Cross-track consequences and actionable handoffs

**Circuit evidence selects the fold.** The regional route has the deepest
declared boundary, distinct-cell fresh evidence, scoped selective tests, material
QK2/value effects, and failed independent composition. It selects the complete
residual6-to-head9.8 joint delta, with exact port closure, both ordered MLP cross
terms, all seven head products, and live context slots.

**Folded algebra changes the circuit grouping.** The seven-term expansion rejects
a three-single head split; the new five-arm identity further rejects grouping the
observed behavioral interaction as only the transmitted explicit head product.
The appropriate candidate grouping is two-level:

1. the seven-term head9.8 joint-write object; and
2. the downstream mixed-curvature response induced by installing routing and
   value together.

If a recursive module census localizes the second object stably on fresh,
structurally varied cells and beats matched splits, group the head term with those
downstream response modules across native boundaries. If it does not, split by
actual downstream consumers and retain the native suffix as an open background.

**CIRCUIT handoff:** run the frozen complete-city joint response census through
blocks10–17, final RMS, unembedding, and softcap. For every module record the
joint response, key single, value single, summed-mains arm, explicit head-product
increment, and mixed suffix curvature; require full-vector closure. Then score
fresh structurally varied context cells, target effects, full logits/CE, at least
three unrelated readers, equal-norm same-site directions, and preregistered
matched random splits. Only afterward name a suffix or claim grouping.

**WEIGHT_FOLDING handoff:** exact-replay and deduplicately price
`residual6 -> attention7/MLP7 -> head8.2 -> norm-closed MLP8 -> complete head9.8`,
retaining all reader-permitted self/ordered-cross terms and all seven
`QK1*QK2*V` products. The block10 rank-128 certificates rule out an exact linear
width `<128` at those seven immediate readers; any smaller quotient must be
explicitly approximate/task-specific or exploit an identified attainable
manifold.

## Limitations

The new calculation reuses an opened, response-conditioned 20-cell artifact. It
does not rerun the model, establish fresh transfer, identify a causal module,
test random splits, close any port, reduce price, or prove the interaction is
semantically specific. The finite-difference suffix term is exact for these five
arms but does not uniquely allocate curvature among downstream modules; that is
the purpose of the pending response census. Integrated-Hessian terminology does
not convert attribution into mechanism. Tensor-network cut bounds and exact
rank witnesses do not prove causal fidelity. FineWeb signed transfer remains
inconsistent, composition remains failed, and the complete executor/price is
still missing.
