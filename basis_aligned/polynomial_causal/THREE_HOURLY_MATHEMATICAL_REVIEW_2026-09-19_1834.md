# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T18:34:18Z**. Next mathematical deadline:
**2026-09-19T21:34:18Z**. The prior review was written at
`2026-09-19T15:29:59Z`, about 184 minutes earlier, so the 175-minute freshness
gate required this review. This is one bounded live review, not an offline
backfill. Historical `/workspace/...` references were resolved against the live
checkout `/workspace/tensor_language`.

This review used no agent, GPU/model forward, job, queue, timer, service,
commit, push, external contact, primary-receipt mutation, or
`TYPED_FACE_EXTRACTION_V1` change. It wrote one uniquely named CPU control and
receipt, this review, and a short board append. The newest hourly review remains
`ACTIVE_TRACK: WEIGHT_FOLDING` at 17:45; this review does not alter hourly
alternation.

## Verdict

Depth on the regional path remains the highest-information route. The exact
seven-term head9.8 artifact now has a distinct-cell interaction census: on the
target readout, collectively omitting the four order-at-least-two
`QK1*QK2*V` terms incurs **0.2853 relative L2** error and their norm is
**1.8837 times the smallest single-term norm**. The omission exceeds 10% in
**16/20 actual context cells**. The triple term alone is small globally
(`0.2023` of the smallest single), but it is not the whole interaction; the
three pair terms cannot be discarded with it.

This is **fold/opened-replay** evidence at a local head/readout boundary. It is
not an edit, response, fit, fresh/OOD result, matched-random-split specificity
test, suffix-fidelity result, or proof of native causal fidelity. It therefore
supports retaining all seven products for the complete replay and the frozen
five-arm experiment; it does not promote the path to a circuit.

Five-property verdict: **held-out/OOD prediction PARTIAL; extraction PASS only
at declared intermediate boundaries; selective manipulation/removal PARTIAL;
composition/reuse FAIL overall; measured simplicity LOCAL ONLY**.

## 1. Current native computation as mathematical objects

### Tensors, indices, shapes, graph, degree, and nonlinearities

Let positions be `t,s in {0,...,T-1}`, with `s<=t`; residual/output indices
`a,o in {1,...,d}`, `d=1152`; heads `h in {0,...,8}`; head coordinates
`c in {1,...,d_h}`, `d_h=128`; MLP product coordinates
`j in {1,...,m}`, `m=4608`; blocks `ell in {0,...,17}`; and vocabulary rows
`v in {1,...,50304}`. The frozen model has 545,902,902 trained parameters and
an untied unembedding.

With native `eps=1.1920928955078125e-7`,

```
S(x) = ||x||_2^2 / 1152 + eps,
RMS(x) = x / sqrt(S(x)).
```

For MLP `ell`, `L,R in R^(4608 x 1152)`, `D in R^(1152 x 4608)`, and
`b in R^1152`:

```
M_ell(x) = D[(Lx) odot (Rx)] / S(x) + b.
```

If `x=b0+a_i+m_j+d` names background, an earlier attention write, an earlier
MLP write, and an installed delta, its numerator contains every permitted self
and ordered cross term, including

```
B(a_i,a_i), B(m_j,m_j), B(a_i,m_j), B(m_j,a_i),
B(b0,d), B(d,b0), B(d,d),
```

plus the remaining background/source couplings. The regional norm-closed MLP8
object retains all changed self/cross terms, live normalization, bias, skip, and
learned re-entry; it omits only terms unchanged under the scoped intervention.

For attention head `(ell,h)`, after native RMS and rounded rotary maps,

```
p^r[t,s] = <Rot_t Q^r xhat_t, Rot_s K^r xhat_s> / 128, r in {1,2},
v[s,c]   = (V xhat_s)[c],
A[t,o]   = sum_{s<=t,c} O[o,c] p^1[t,s] p^2[t,s] v[s,c].
```

Each `Q^r,K^r,V` is `128 x 1152`; the head output slice `O` is
`1152 x 128`. At fixed RMS denominators each QK score has polynomial degree two
in named residual sources and the attention numerator degree five; the MLP
numerator has degree two. Live RMS makes both maps rational with square roots.
The final `30*tanh(U RMS(h)/30)` is non-polynomial, so no finite-degree claim is
made for the whole network.

For source decompositions `x_t=sum_i q_i[t]`, `x_s=sum_j k_j[s]`,

```
p^r[t,s] = sum_{i,j} <Rot_t Q^r q_i[t], Rot_s K^r k_j[s]> / 128.
```

Earlier attention and MLP residual sources therefore produce attention-self,
MLP-self, attention-to-MLP, and MLP-to-attention terms in each QK factor.
Multiplying both QK expansions and the value expansion produces every
reader-permitted `QK1_term * QK2_term * V_source` interaction. The current path
retains the full set, not only QK1, additive routing/value singles, or self terms.

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

### Ties, gauges, backgrounds, outputs, norm, and literal price

Weights are tied across positions/examples, not layers. Learned residual and
re-entry coefficients, signed attention value mixing, MLP bias, causal masks,
and rounded rotary semantics are literal parts of the program. Gauges include
hidden-unit permutation; reciprocal `L_j -> cL_j, R_j -> R_j/c`; `L/R`
exchange after symmetrization; compatible Q/K basis changes; reciprocal scaling
between the two score factors; and compatible V/O basis changes. A coordinate
is not a semantic component without downstream operational identification.

Open backgrounds/ports are token IDs, native residual6, unrelated
query/source state, every RMS denominator, rotary phase, causal support,
learned mixing/re-entry, biases, unchanged attention fields, and the native
suffix. Outputs to preserve are signed regional effect, at least three
unrelated-reader effects, full logits/CE, and named intermediate writes needed
for replay.

Algebraic replay uses relative L2, maximum absolute error, term closure, and
off-support zero. Behavioral composition uses the Mobius interaction norm
divided by the **smallest live** single effect and a preregistered matched-random
split distribution. A closed face identity is accounting, not evidence that an
interaction is small or specific.

Literal native prices remain 545,902,902 parameters for the model; 884,736 FP32
values for a two-QK head's five input maps plus output slice; 528 causal cells
at `T=32`; and 67,584 score-times-value-coordinate products plus 528 score-factor
products across those cells. The regional norm-closed MLP8 package stores
16,536,963 FP32 values plus its token table/cache. Package manifests overlap;
static weights, state, adapters, masks, edges, caches, and measured runtime must
be deduplicated. A full residual6-to-suffix price remains missing.

### Current circuit object

The current circuit object is regional city/spelling route
`PATH-REGIONAL-CITY-001`, extended in the live plan beyond its registered
residual6-to-head8.2 endpoint through norm-closed MLP8, complete head9.8, and
the native suffix. Existing extraction replays on 40 opened fixtures and 40
fresh FineWeb sequences from 20 distinct document/context cells. Scoped fresh
removals beat same-site nulls. However, packed FineWeb direction transfer is
`88/102 < .90`; key/value composition has `interaction/smaller=2.3448`; source
and random partitions fail specificity; and the full suffix has not been
extracted. This remains a path with intermediate extraction, not a five-property
circuit.

Number and gender have substantial opened edit evidence but no canonical
standalone boundary or deduplicated price. Gender copier/reader composition
transfers on three sets yet lacks the smallest-piece matched-split test. Person
is correctly recorded as spread after the seven-head set scores `.549 < .60`.
Aspect is parked because the pairs are syntactically misaligned. Temporal has a
valid sign-constancy screen but failed reader/carry/additivity predictions.

### Current folded-path object

For baseline head9.8 fields `p1,p2 in R^(T x T)`, `v in R^(T x 128)` and
deltas `dp1,dp2,dv`, let `F` denote the trilinear head contraction. Its exact
finite change is

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

The proposed folded path retains all seven nonempty products, every live
self/ordered-cross source term in MLP8 and both QK factors, live denominators,
backgrounds, bias, skip, and learned re-entry. Only the unchanged baseline is
omitted.

## 2. Primary mathematics and precise mapping

Rota's incidence-algebra theorem gives a unique Mobius inversion on a finite
poset; on a Boolean lattice it is inclusion-exclusion ([primary paper,
1964](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf)). Map the three
binary head ports to `S subseteq {K1,K2,V}` and let `g(S)` be the local head
readout with exactly those deltas installed. The unique coefficient is

```
c_A = sum_{B subseteq A} (-1)^(|A|-|B|) g(B),
g(S) = sum_{A subseteq S} c_A.
```

For the native tri-affine map these seven nonconstant coefficients are exactly
the three singles, three pairs, and triple stored in
`CITY_FINEWEB_HEAD9_FOLD_V1_ARTIFACT.pt`. Thus a degree-one truncation has the
unique omitted function `c_K1K2 + c_K1V + c_K2V + c_K1K2V` at the full corner.
Computing its norm is an executable falsifier of the claim that interaction
terms are locally negligible. The inversion costs `O(2^3)` vector operations;
the census over stored rows is linear in artifact size.

Sobol's functional ANOVA supplies orthogonal variance shares when inputs have a
specified product probability measure ([primary paper,
1993](https://www.andreasaltelli.eu/file/repository/sobol1993.pdf)). That theorem
does **not** turn the present ratios into Sobol indices: native K1, K2, and V
fields share residual/RMS sources; the 20 cells are an empirical opened panel;
and the stored coefficients need not be orthogonal. This violated assumption is
why the receipt reports direct norms and per-cell errors, not variance fractions.

Assumptions met for the Mobius result: aligned ports, an exact tri-affine local
head contraction, common vector outputs, all seven stored coefficients, and a
declared Euclidean norm. Assumptions absent for behavioral adoption: fresh
cells, matched random splits, independent deployability, complete suffix
recomputation, causal intervention, and corpus-level sampling.

## 3. Executed CPU consequence

Control:
[`head9h8_seven_term_interaction_census_2026_09_19_1833.py`](head9h8_seven_term_interaction_census_2026_09_19_1833.py).
Receipt:
[`HEAD9H8_SEVEN_TERM_INTERACTION_CENSUS_2026-09-19_1833.json`](HEAD9H8_SEVEN_TERM_INTERACTION_CENSUS_2026-09-19_1833.json).
It used `/venv/main/bin/python`, CPU only, two BLAS/OpenMP threads, float64
stored contributions, no model load, and no forward. Runtime was 0.0147 seconds.
Source hashes and 20-cell/240-row shape assertions passed; an independent
recomputation and AST validation also passed.

| scope | interaction / smallest single | omission / full delta | triple / smallest single |
|---|---:|---:|---:|
| target readout | **1.8837** | **0.2853** | 0.2023 |
| all five readouts | 1.7110 | 0.2845 | 0.1837 |
| four controls | 0.3984 | 0.2133 | 0.0397 |

Across 20 distinct target context cells, interaction/smallest-single ranges
`0.2074–41.7494` (median `6.4761`; 19/20 above `.35`). The more stable direct
approximation measure, interaction-omission/full-delta, ranges
`.0263–.9176` (median `.3002`; 16/20 above `.10`). Extremely large
smallest-single ratios reflect tiny denominators and are not advertised as
effect sizes.

Executable consequence: do not replace the complete head9.8 change by only the
three singles, and do not infer from the globally small triple that all
interaction is negligible. Preserve the three pair terms and the triple in the
exact replay. The later fresh five-arm test, with matched random splits and the
native suffix, remains the experiment that can justify a consumer-specific
grouping or omission.

## 4. Five-property scorecard

| property | current evidence | verdict |
|---|---|---|
| Held-out/OOD prediction | Regional extraction covers 20 distinct fresh FineWeb cells and scoped Pile tests, but packed FineWeb signed direction is `88/102`; this new census replays opened rows. | **PARTIAL** |
| Extraction at a declared boundary | Residual6-to-attention8 and norm-closed MLP8 packages execute with declared ports. Complete head9.8 plus suffix remains open. | **PASS at intermediate boundaries; incomplete end-to-end** |
| Selective manipulation/removal | Scoped fresh regional edits beat same-site nulls, but the complete path lacks one frozen three-reader, equal-norm, matched-control battery. | **PARTIAL** |
| Composition/reuse | Existing key/value `interaction/smaller=2.3448`; this local seven-term census finds 28.5% omission error; prior matched/source splits fail specificity. | **FAIL overall** |
| Measured simplicity | Local storage/state counts and exact restricted rank bounds exist. No deduplicated complete executor price or matched-effect advantage exists. | **LOCAL ONLY** |

Lower error, storage, rank, or CE cannot replace a missing behavioral property.
Claims remain tagged **fold/edit/response/fit** and **fresh/opened/replay**.

## 5. Organization reconciliation and repair decision

- `COMPUTATION_PATH_REGISTRY.md` contains `PATH-REGIONAL-CITY-001` but still
  ends at the head8.2 write. It lacks norm-closed MLP8, the seven-term head9.8
  endpoint, block10 rank boundary, pending suffix factorial, and deduplicated
  price. The file is concurrently dirty, so this review did not overwrite it.
- `CIRCUIT_REGISTRY.md` and `MODULE_DOSSIERS.md` contain the principal regional
  head8.2/head9.8/MLP7/MLP8 receipts and failures. The path record and module
  entries are not cross-linked to the current full endpoint or the two recent
  rank certificates. No evidence was copied into another narrative.
- `CIRCUIT_GRAPH_REGISTRY_V1.md` remains a generated 49-package/four-trait
  inventory. It omits `city_mlp8_norm_closed_v1`; its two
  `four_trait_verified` labels are not five-property completion claims because
  measured simplicity is separate and the current composition standard is
  stronger.
- Number/gender/person work remains distributed across primary receipts, board
  claims, and mutable explanations. `PATH-SUBJECT-001` is the older L11H3
  object and must not be reused silently. Aspect and temporal failures are not
  orphans: their primary receipts and board verdicts remain discoverable.
- Alias debt remains for dotted head names versus zero-based tensor indices,
  unit/site pairs, and operational labels such as reader, writer, copier, hub,
  and detector. Those labels are roles, not module identities.
- `explanations/README.md` has repeated “Latest regional result” entries and
  stale navigation. `for_logan/LATEST.md` is a narrative pointer, not a primary
  authority. Latest commits, primary receipts, registries, and board claims
  governed this review.
- Recent v594-v597-style receipts still lack a uniform top-level evidence type,
  evaluation status, distinct-context manifest, and literal program price.
  Row count alone is not effective sample size.

The one bounded action was the mathematical CPU consequence. A shared receipt
schema/declarative runner remains the best broad efficiency repair, but safely
retrofitting heterogeneous intervention families and proving equivalence is not
a micro-refactor. The computation-path registry is also concurrently modified.
Creating a second publisher/checker would add duplication. Therefore no shared
file or primary receipt was changed; the next safe repair is one canonical
regional path extension and alias/cell/evidence/price schema when ownership
clears.

## 6. Efficiency audit

Window: `2026-09-19T15:29:59Z–18:34:18Z`. No new scientific runner, primary
result receipt, or commit appeared after the prior mathematical review. The
last commit remains `9e0c03df3` at 12:21 UTC. Managed canaries completed at
15:50, 16:21, 16:51, 17:21, 17:52, and 18:22. Both Supervisor runners were
`RUNNING`; both queues were empty. No queue/service action was justified or
authorized.

The prior burst's one-runner-per-candidate pattern, duplicated builders/scorers,
stale guards, and missing cell/price schema remain the dominant code debt. The
subsequent six-hour absence of scientific receipts is the opposite throughput
failure: review and health ceremony continued while the already specified
regional replay and response census did not. Serial scientific latency is
undefined because no candidate began or ended in this window; uninstrumented
labor and idle time are not fabricated. The new replay control took 0.0147
seconds and reused an existing artifact rather than model compute.

## 7. Cross-track consequences and actionable handoffs

**Circuit evidence selects the folding target.** The regional route has the
deepest declared extraction boundary, distinct-cell fresh evidence, scoped
selective tests, material QK2 effects, and failed additive/random-split
composition. It selects the complete residual6-to-head9.8 delta—not another
breadth route—and requires exact port closure, both ordered MLP cross terms,
all seven head products, and live context slots.

**Folded algebra proposes a circuit grouping/split.** The exact Mobius terms
show that a “three independent singles” split is falsified locally: pair-plus-
triple omission is 28.5%. The five-arm causal design should therefore compare
baseline, routing single, value single, both singles with the explicit product
suppressed, and true joint product. If the product arm is fresh, selective, and
specific against matched splits, group those factors at the head consumer
boundary. If it fails, use the forward response census to split by actual
downstream users rather than native QK/value labels.

**CIRCUIT handoff:** run the frozen regional forward-response census before
naming a suffix, then the full-suffix five-arm factorial on structurally varied
distinct `(template, city pair, endpoint)` cells. Score signed regional effect,
full logits/CE, at least three unrelated readers, 16 equal-norm same-site
directions, and matched random splits. Normalize by the smallest live piece.

**WEIGHT_FOLDING handoff:** exact-replay
`residual6 -> attention7/MLP7 -> head8.2 -> norm-closed MLP8 -> complete
head9.8 vector delta`, retaining every reader-permitted self/ordered-cross term
and all seven products. Deduplicate projections, static values/bytes, product
nodes, graph edges, state, caches, and measured runtime. Do not substitute the
new replay census for causal adoption.

## Limitations

The new census reuses an opened 20-cell artifact and a response-conditioned
local reader. It does not recompute the suffix, test random splits, establish
freshness, or show that large interaction is semantically specific. Its
smallest-single ratio can explode when one single is tiny; the direct 28.5%
omission error is the safer local approximation statistic. The complete native
executor, response census, suffix factorial, and deduplicated price remain
absent. FineWeb signed transfer remains inconsistent and composition fails. No
Mobius identity, replay, rank, storage reduction, or low error proves native
causal fidelity.

