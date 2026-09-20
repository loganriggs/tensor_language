# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T03:06:54Z**. Next mathematical deadline: **2026-09-19T06:06:54Z**. The preceding review was `2026-09-18T03:14:10Z`, so this review is due; no offline reviews are backfilled. The hourly clock remains unchanged (`ACTIVE_TRACK: WEIGHT_FOLDING` at 03:00). Historical `/workspace/...` paths were resolved against the live checkout `/workspace/tensor_language`. `TYPED_FACE_EXTRACTION_V1` was not modified. No GPU job, queue, timer, service, commit, push, or external contact was made.

## Verdict

The regional path's correct next mathematical object is the **complete factor-level head9.8 difference**, not another separately scored QK or value approximation. At a fixed native background it is a tri-affine map in `(QK1,QK2,V)`, so its joint delta has exactly seven nonempty Boolean-Mobius components. This is an exact accounting identity, not evidence that the components are small, independently causal, or reusable. The executed CPU control verifies the identity at `7.11e-15` component error and `3.55e-15` closure error, and supplies a falsifying scalar example with exact closure but interaction/smaller-live-single `=100`.

The current regional object scores: **held-out/OOD prediction PARTIAL**, **extraction PASS only at declared intermediate boundaries**, **selective manipulation PARTIAL**, and **composition/reuse FAIL for the tested splits**. Simplicity is separately measured and is not yet a five-property win.

## 1. Native computation as mathematical objects

### Architecture and indices

Let sequence positions be `t,s in {0,...,T-1}`, causal keys satisfy `s<=t`, residual coordinates `a,o in {1,...,d}` with `d=1152`, heads `h in {1,...,9}`, head coordinates `c in {1,...,d_h}` with `d_h=128`, MLP product coordinates `j in {1,...,m}` with `m=4608`, blocks `ell in {0,...,17}`, and vocabulary rows `v in {1,...,50304}`. The checkpoint has 545,902,902 parameters and an untied unembedding.

For any residual vector `x`,

```
rho(x) = sqrt(||x||_2^2 / 1152 + eps),   eps = 1.1920928955078125e-7,
xhat = x / rho(x).
```

The bilinear MLP at block `ell` has

```
L_ell,R_ell : [4608,1152],  D_ell : [1152,4608],  b_ell : [1152],
M_ell(x) = D_ell ((L_ell x) odot (R_ell x)) / rho(x)^2 + b_ell.
```

Thus its unnormalized numerator is quadratic. For a source sum `x=sum_i x_i`, its exact ordered expansion is

```
M_num(x) = sum_{i,j} D_ell ((L_ell x_i) odot (R_ell x_j)).
```

The `i=j` pieces are self terms; `i!=j` are ordered cross terms. `L/R` interchange, hidden-unit permutation, and reciprocal row/column rescaling are gauges. A left or right factor alone is not a semantic unit; the complete written product is invariant.

For attention head `(ell,h)`, write the two rotary query-key factors and value as

```
p^r_{t,s,h} = <Rot_t Q^r_{ell,h} xhat_t, Rot_s K^r_{ell,h} xhat_s>, r in {1,2},
v_{s,h} = V_{ell,h} xhat_s,
A_{ell,h}(x)_t = O_{ell,h} sum_{s<=t} p^1_{t,s,h} p^2_{t,s,h} v_{s,h}.
```

`Q,K,V : [128,1152]` per head and `O : [1152,128]`. The full block sums nine heads. The same weights are tied across positions and contexts; residual re-entry coefficients are learned and retained. Reciprocal Q/K gauges that preserve rotary pairing, value/output basis changes, and the factor redistribution `p1 -> c p1, p2 -> p2/c` leave the head function unchanged. Current attribution is therefore defined operationally by frozen native factor ports and interventions, not by raw factor norms.

Conditioned on all RMS denominators, an MLP is degree two in residual sources, each QK score is degree two, and `QK1*QK2*V` is degree five in residual-source coordinates but degree three in the three factor tensors. With live RMS denominators the local maps are rational with square roots; the final `30*tanh(score/30)` is non-polynomial. No finite-degree whole-network polynomial is claimed.

### Current circuit object

The depth target is the regional city-spelling intervention. Its current contraction graph is

```
(residual6, token IDs)
  -> all nine attention7 heads
  -> full MLP7 bilinear response and five head8.2 readers
  -> complete head8.2 city write delta8 at the attention8 boundary
  -> (native z8, delta8, token IDs)
  -> norm-closed MLP8 response and generated edited rho9
  -> head9.8 value correction
  -> native later suffix and selected output readers.
```

The exact residual6-to-attention8 generator retains residual6, initial/re-entry, attention7, and MLP7 contributions, including all ordered MLP7 self/cross terms, native RMS, rotary maps, biases, and all nine attention7 heads. The corrected replay error is `1.97e-6` on 40 opened fixtures and `1.61e-6` on 40 fresh outcome-blind FineWeb sequences representing **20 distinct document/context cells** and 240 probes. Probe count is not treated as sample size.

The norm-closed MLP8 object takes `z8[1,T,1152]`, intervention `d[1,T,1152]`, and token IDs. With `S(x)=||x||^2/1152+eps`, it retains

```
(Lz odot Rz)/S(z),
(Ld odot Rz + Lz odot Rd + Ld odot Rd)/S(z+d),
(Lz odot Rz)(1/S(z+d)-1/S(z)),
```

plus `D8`, bias in the edited norm, learned residual mixing, and the initial-state table. This explicitly keeps baseline self, both ordered background/intervention cross terms, intervention self, and normalization change. It stores 16,536,963 FP32 values (66,147,852 bytes), 3,216 token-ID bytes, a separately priced 4,718,592-byte derived `Wv*D8` cache, and uses one native `z8` array plus one intervention array (36,864 scalars each at `T=32`). It is extraction/interface closure, not compression.

The nearest separately priced upstream artifacts are the exact earlier-boundary reader generator at 21,851,526 FP32 values and the packed eight-head approximation at 22,432,390 values on its FineWeb table. The corrected all-nine-head residual6 generator does not yet have a canonical manifest price. Therefore the review does **not** add incomparable artifact counts and does not claim a total folded-program price.

### Current folded-path object

The proposed endpoint is the complete head9.8 output-write difference `Delta o9.8[t,1152]` before the later suffix. Let the baseline factor tensors be `p1,p2 : [T,T]` and `v : [T,128]`, with intervention deltas `dp1,dp2,dv`; causal masking and `O9.8:[1152,128]` are implicit. Define

```
F(p1,p2,v)_t = O9.8 sum_{s<=t} p1[t,s] p2[t,s] v[s].
```

Then the exact joint change is the sum of all seven nonempty terms:

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

No term is currently proposed for omission. In particular, QK1, QK2, and value must not be scored as independent modules. Inside each QK delta, if query/key residuals are `r=b+a7+m7+d8`, the exact score contains every permitted ordered pair

```
<Q b,K b>, <Q a7,K a7>, <Q m7,K m7>,
<Q a7,K m7>, <Q m7,K a7>,
```

and the corresponding background/intervention terms involving `d8`; the exact current path retains all of them through the frozen factor deltas. The same rule applies to MLP8. Only after the full operator replays and a preregistered full-suffix factorial identifies a causally selective grouping may a subset be proposed.

Outputs to preserve are: the native head9.8 write for local replay; then the signed regional reader effects, full target effect, at least three unrelated-reader effects, and later full-logit/CE behavior under a recomputed suffix. Local error is per-sequence relative `L2` plus max off-support magnitude. Behavioral composition is `||joint-single1-single2||_2 / min(||single1||_2,||single2||_2)` on live singles, with matched random splits; it is not mere algebraic closure.

## 2. Literature mapping

The best exact match is Rota's incidence-algebra/Mobius inversion on a locally finite poset: [G.-C. Rota, *On the Foundations of Combinatorial Theory I. Theory of Mobius Functions* (1964)](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf). Map the poset to the Boolean lattice `P=2^{QK1,QK2,V}` and define `f(S)` as the head output with precisely the deltas in `S` installed. The unique factorial component is

```
g(S) = sum_{U subseteq S} (-1)^(|S|-|U|) f(U),
```

and `f(S)=sum_{U subseteq S}g(U)`. Assumptions: all eight arms are evaluations of the same deterministic function with the same background and only the named factor ports changed. Cost: eight head evaluations (or direct contractions). Guarantee: exact recovery of factorial components. It gives no causal sufficiency, distributional generalization, small-interaction bound, or random-split specificity.

Kruskal's primary uniqueness result for three-way arrays is a serious but presently inapplicable compression theorem: [J. B. Kruskal, *Three-way arrays: rank and uniqueness of trilinear decompositions* (1977)](https://doi.org/10.1016/0024-3795(77)90069-6). A coefficient tensor for the factor-level trilinear map could be written as a CP sum and would be essentially unique if its three factor matrices satisfy the required k-rank inequality (in the standard form, `k_A+k_B+k_C >= 2R+2`). The native head instead has tied position/source contractions, causal masks, repeated structured factors, context-dependent normalizers upstream, and no certified CP rank or k-ranks. Its native Q/K and V/O gauges also remain. Thus Kruskal does not identify the current seven effects or justify a smaller path. If a future frozen CP candidate is proposed, the executable identifiability gate is to compute/rigorously lower-bound its three k-ranks before interpreting components; there is no reason to run that fit now.

Tensor trains, Hankel realization, and graph-width contraction can price a sampled or discretized evaluator, but no current assumption makes the normalized transformer a finite-state linear recurrence or gives a distribution-wide approximation certificate. They remain lower-priority than exact port closure plus causal factorial testing.

## 3. Executed consequence

`BOOLEAN_MOBIUS_HEAD_PRODUCT_CONTROL_2026-09-19_0306.json` is a CPU-only float64 control run with `/venv/main/bin/python`, fixed seed 20260919, and head-shaped tensors `[T,S]=[3,5]`, value width 4, output width 6. It evaluated all eight arms, Mobius-inverted them, and compared every component with its direct delta product:

- maximum component error: `7.105427357601002e-15`;
- maximum joint-delta closure error: `3.552713678800501e-15`;
- runtime: `0.01545s`.

The paired scalar falsifier uses baseline `(1,1,1)` and deltas `(.001,100,0)`. The two live singles are `.001` and `100`, while their pair interaction is `.1`, hence interaction/smaller-live-single `=100` despite exact Mobius closure. Consequence: the native full-head implementation should use the seven direct products or an equivalent eight-arm audit, but **a closed identity is never a composition pass**. The later five-arm causal design still needs true joint, both singles, additive-without-product, matched same-site random splits, and a recomputed suffix.

This consequence dominates another rank/reconstruction sweep: it is exact, costs milliseconds, catches omitted product terms, and directly fixes the interpretation of the upcoming regional operator. It does not replace native replay or behavioral tests.

## 4. Five-property scorecard

| Property | Current evidence | Verdict |
|---|---|---|
| Held-out/OOD prediction | Exact attention8 write predicts 20 outcome-blind FineWeb document/context cells; earlier packed removal passed fresh Pile documents. But FineWeb full-behavior direction is `88/102 < .90`, and the norm-closed interface has no separately fresh confirmation. | **PARTIAL** |
| Extraction at a declared boundary | Residual6+tokens to attention8 is exact at about `2e-6`; norm-closed MLP8 mediator generates edited RMS and passes isolated/installed replay (`6.2e-5` max readout error). Native `z8`, upstream intervention, blocks0-6, and later suffix remain open. | **PASS at intermediate boundaries; not end-to-end** |
| Selective manipulation/removal | Fresh Pile removal and fresh supplied-norm mediation beat same-site nulls and collateral controls. FineWeb regional direction reversals and the norm-closed object's opened-only scope prevent a general pass. | **PARTIAL** |
| Composition/reuse | Complete city key/value interchange has interaction/smaller `2.3448`; direct/MLP8 fresh composition fails the reversed subgroup at `.4908 > .35`; random-split specificity also fails there. | **FAIL for tested splits** |
| Measured simplicity | Packed attention7 saves 884,736 FP32 values (`3.8%`) locally; norm closure grows from 11,206,658 to 16,536,963 stored floats to remove one scalar port. The exact full combined path lacks a deduplicated manifest price and matched-effect random-component comparator. | **LOCAL PRICES ONLY; no simplicity adoption** |

Lower error and storage do not substitute for a missing behavioral property. Evidence tags remain separate: exact head expansion is **fold**; installed intervention is **edit**; downstream census is **response**; any learned surrogate is **fit**. Fresh/opened/replay labels stay attached to every panel.

## 5. Organization reconciliation

- `PATH-REGIONAL-CITY-001` correctly links the corrected upstream fold and earlier city packages, but still names `city_mlp8_value_mediator_v1`, not the newer norm-closed package or the proposed complete head9.8 endpoint.
- `CIRCUIT_GRAPH_REGISTRY_V1.md` remains a generated 49-package inventory from 18 September and omits untracked `city_mlp8_norm_closed_v1`. Its two historical `four_trait_verified` rows are not five-property certifications.
- `CIRCUIT_REGISTRY.md`, `MODULE_DOSSIERS.md`, and the MLP dossier index preserve the regional failures through the supplied-norm mediator but do not yet cross-link the norm-closed result.
- `explanations/README.md` points first to the 03:26 extracted mediator and omits the 03:42 norm-closed report. This is a stale navigation pointer, not stale primary evidence.
- MLP1 v287-v311 has primary receipts, a scorecard, shared-component notes, and a Logan narrative, but no concise computation-path registry row, current circuit-registry row, or consolidated MLP1 dossier entry. Aliases `{MLP1, layer-1 MLP, gain head units 3289/624}` are therefore not reconciled.
- The regional aliases (`head8.2`, `head9.8`, `MLP7`, `MLP8`) and their principal failures are present in module records. Primary receipts remain authoritative over historical `LATEST` pointers.

No registry/index repair was made. The computation-path registry is already modified by concurrent work, the MLP1 lane was actively producing v311/v312 during this audit, and the norm-closed package/report tree is untracked primary-owned work. Editing shared navigation now risks collision and would exceed the review's chosen single bounded consequence. The debt is explicit and small enough to repair once those lanes freeze.

## 6. Efficiency audit

Supervisor reports both `bqrunner` and `bqrunner2` running; both queues were empty at inspection, and `nvidia-smi` listed no compute process. The last observed lane receipts v304-v312 generally took 2-4 runner seconds; v310 had one preserved execution failure caused by a non-contiguous view before its successful rerun. There is no idle GPU job to rescue and no queue/timer change is justified.

The main duplication is concrete: `basis_aligned/bilinear_quotient/ops/` contains **25 separate MLP1 runners for v287-v311** (v312 appeared while this review was in progress), commonly adding one counterfactual or scorer at a time. Since midnight there were 28 commits before the review cutoff, while the 03:00 review reports only 44.81 model seconds through v309. Scientific iteration is fast, but repeated runner authoring and per-receipt publication dominate model runtime and create brittle hook/layout errors. A declarative MLP1 source-term/counterfactual executor plus shared result writer is likely worthwhile after the lane freezes; doing it during active v312 work would have higher collision/equivalence-test cost than this review can safely absorb.

The regional lane also has several near-duplicate builders/checkers for supplied-norm, norm-closed, isolated, and installed variants. Here the variants enforce materially different interfaces and bindings, so consolidating them before the full head9.8 operator exists would save less than it risks. The better next shared abstraction is one pure head-product routine accepting `(p1,p2,v,dp1,dp2,dv)` and emitting seven named terms plus the Mobius audit; it should be integrated into the next owned implementation, not introduced as a separate framework now.

Distinct-cell accounting is adequate in the latest upstream receipt (20 documents/context cells, 40 sequences, 240 probes). Future suffix tests must vary boundary structure and endpoint, not inflate row counts through repeated probes of the same context.

## 7. Two-track handoff

**WEIGHT_FOLDING handoff:** implement the exact residual6-to-head9.8 output operator with the corrected all-nine-head upstream generator, norm-closed MLP8 response, native RMS/rotary/re-entry semantics, and all seven nonempty factor products. Validate direct-product versus eight-arm Mobius equality and native head-write replay on opened fixtures plus a separately labeled fresh panel. Publish a deduplicated literal price: independent floats/bytes, derived caches, native state arrays, ports, nodes, edges, and contraction FLOPs. Kill or correct on replay failure; do not call exact replay causal fidelity.

**CIRCUIT handoff:** after freezing that operator, install its generated head9.8 write into the native suffix on preregistered structurally varied, genuinely distinct context cells. Use true joint, both singles, additive-without-product, and matched random-split arms; preserve at least three unrelated readers. The full head product is the initial grouping because prior independent key/value and direct/MLP8 splits failed. A pass would motivate a downstream grouping; a failure should split by consumer response or structural context, not by arbitrary native factor magnitude.

Circuit evidence selected this folding target: the forward response census made head9.8 prominent but showed later attention is necessary, and FineWeb reversals plus failed independent composition require the complete coupled product. Conversely, the folded algebra proposes the decisive circuit intervention: compare the true joint product against additive-without-product at the head boundary, then trace any nonadditive response through later attention rather than claiming it from algebra alone.

Limitations: the CPU control is synthetic and proves only an identity; the complete native head9.8 operator is not yet implemented; the combined exact program has no deduplicated manifest price; norm-closed fresh confirmation remains separate; full-suffix OOD prediction, selective manipulation, composition/reuse, and matched-effect simplicity remain incomplete.
