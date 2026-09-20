# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T06:14:00Z**. Next mathematical deadline: **2026-09-19T09:14:00Z**. The preceding review was written at `2026-09-19T03:06:54Z`; this is one due review, not an offline backfill. Historical `/workspace/...` paths were resolved against the live checkout `/workspace/tensor_language`. The hourly clock is unchanged: the newest Codex review declares `ACTIVE_TRACK: WEIGHT_FOLDING` at 05:07, so the next hourly track remains `CIRCUIT`. No agent, GPU job, queue/timer/service change, commit, push, external contact, or `TYPED_FACE_EXTRACTION_V1` change was made.

## Verdict

The regional depth path remains the best-defined joint object. Exact upstream attention8 and norm-closed MLP8 interfaces exist, but the complete residual6-to-head9.8 `QK1*QK2*V` write and its full-suffix causal test do not. Tested regional splits are non-compositional, so algebraic replay cannot be promoted to native causal fidelity.

The executed actual-weight consequence gives a new simplicity constraint. On a 16-dimensional witness subspace of the opened regional `z8/delta8` fixtures, the symmetric quadratic coefficient tensor of the native MLP8 numerator after the 128-dimensional head9.8 value reader has output flattening rank **128**. Therefore every exact implementation in the standard class “scalar product of two linear forms followed by an arbitrary output writer” needs at least **128 product nodes** even on this restriction. This is a robust numerical witness for the stored FP32 map, not a formal interval certificate, approximation bound, semantic decomposition, or causal result.

Current five-property score: **held-out/OOD prediction PARTIAL; extraction PASS only at declared intermediate boundaries; selective manipulation PARTIAL; composition/reuse FAIL for tested regional splits; measured simplicity LOCAL ONLY**. The concurrent number program now has an OOD-live and edited unit/population path plus a causally separable agreement output axis, but no standalone closed boundary, full preservation battery, or composition certificate.

## 1. Native computation as mathematical objects

### Indices, tensors, shapes, and tied parameters

Let positions be `t,s in {0,...,T-1}` with causal sources `s<=t`; residual coordinates `a,o in {1,...,d}`, `d=1152`; heads `h in {1,...,9}`; head coordinates `c in {1,...,d_h}`, `d_h=128`; MLP product coordinates `j in {1,...,m}`, `m=4608`; blocks `ell in {0,...,17}`; and vocabulary rows `v in {1,...,50304}`. The checkpoint contains 545,902,902 parameters and an untied unembedding. All layer weights are tied across positions and examples, not across layers. Learned residual/re-entry coefficients are retained.

For residual `x`,

```
S(x) = ||x||_2^2 / 1152 + eps,  eps = 1.1920928955078125e-7,
RMS(x) = x / sqrt(S(x)).
```

MLP `ell` has `L_ell,R_ell:[4608,1152]`, `D_ell:[1152,4608]`, and `b_ell:[1152]`:

```
M_ell(x) = D_ell[(L_ell x) odot (R_ell x)] / S(x) + b_ell.
```

For named residual sources `x=sum_i x_i`, its numerator is exactly

```
N_ell(x) = sum_i D[(L x_i) odot (R x_i)]
         + sum_{i!=j} D[(L x_i) odot (R x_j)].
```

The first sum contains self terms and the second contains ordered cross terms. In the regional path the names include the native background/residual, earlier attention writes, earlier MLP writes, the selected city write, initial-state re-entry, and any installed edit. Neither ordered cross term may be silently merged or dropped.

For attention head `(ell,h)`, after native RMS and rounded rotary maps,

```
p^r_{t,s,h} = <Rot_t Q^r_{ell,h} xhat_t,
                 Rot_s K^r_{ell,h} xhat_s> / 128,  r in {1,2},
v_{s,h} = V_{ell,h} xhat_s,
A_{ell,h}(x)_t = O_{ell,h} sum_{s<=t} p^1_{t,s,h} p^2_{t,s,h} v_{s,h}.
```

Each `Q,K,V` is `[128,1152]`; `O` is `[1152,128]`. There is no softmax or SiLU. Attention is the complete `QK1*QK2*V` product with the causal mask and learned mixing.

Conditioned on RMS denominators, an MLP numerator is degree two in residual sources; each QK score is degree two; and a head is degree five in normalized residual-source coordinates and tri-affine in the three factor tensors `(p1,p2,v)`. With live RMS denominators the maps are rational with square roots. The final output `30*tanh(U RMS(h)/30)` is non-polynomial. No finite-degree whole-network polynomial claim is made.

### Gauges and identifiability

Hidden-unit permutation; reciprocal `L_j -> c L_j, R_j -> R_j/c`; `L/R` exchange after symmetrization; reciprocal Q/K transformations compatible with rotary pairing; `p1 -> c p1, p2 -> p2/c`; and value/output basis changes leave native functions unchanged. Raw factor magnitude or a chosen coordinate is therefore not an identified semantic unit. The quadratic coefficient tensor and its flattening ranks are invariant to the MLP permutation/rescaling gauges, but that invariance alone supplies no behavioral identity.

### Current circuit object

The primary circuit candidate is the regional city-spelling route:

```
(token IDs, native residual6)
  -> all nine attention7 heads
  -> complete MLP7 response
  -> five head8.2 readers and full head8.2 city-write delta
  -> native z8 + delta8, MLP8 with live normalization and bias
  -> head9.8 QK1 * QK2 * V write
  -> native later suffix
  -> regional target reader, at least three unrelated readers, full logits/CE.
```

The exact residual6-to-attention8 generator replays at `1.97e-6` on 40 opened fixtures and `1.61e-6` on 40 fresh FineWeb sequences drawn from **20 distinct document/context cells**. The norm-closed MLP8 value object takes `z8[1,T,1152]`, `delta8[1,T,1152]`, and token IDs. It retains baseline self, both ordered `z8/delta8` cross terms, edit self, changed normalization, bias, re-entry, and initial-state lookup. The native suffix remains external.

The concurrent number candidate now traces `MLP1 token lookup -> MLP3 agreement populations -> head4.5 copied noun state -> MLP5/6/7 relays -> MLP8 detector 829 -> heads 9.6/12.4/15.1`, with a second post-noun rebuild. Ten named units cost 20.2% of the panel margin and 11.1% on natural rows. A separate unembedding-derived agreement axis `VP0` is written mainly by late MLPs; five selected late units reduce it 12.3% while changing the pronoun margin only +0.7%, 50 times the matched null. The v392 earlier-layer unit closure is invalid because lambda-chain scaling was omitted; rankings and exact MLP17 terms remain candidates, not a closed fold. This is a useful circuit split, not yet an extracted circuit.

### Current folded-path object and retained terms

The current folding endpoint is the complete head9.8 output-write delta before the later suffix. For baseline factors `p1,p2:[T,T]`, `v:[T,128]` and intervention deltas `dp1,dp2,dv`, define

```
F(p1,p2,v)_t = O_9.8 sum_{s<=t} p1[t,s] p2[t,s] v[s].
```

The exact finite change retains all seven nonempty products:

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

No factor-level term is currently licensed for omission. If a query/key residual is decomposed as `r=b+a7+m7+d8`, each QK factor includes its permitted background self, attention self, MLP self, city-write self, and both ordered terms for every live source pair. In particular it contains `a7*a7`, `m7*m7`, `a7*m7`, and `m7*a7`, plus the corresponding background and `d8` terms. MLP8 likewise retains `z*z`, `delta*z`, `z*delta`, and `delta*delta`. The proposed retained subset is therefore the **complete coupled finite-change operator**; selection happens only after a preregistered suffix intervention identifies a smaller grouping.

Backgrounds kept explicit are native `z8`, query and source states not generated by the path, every RMS denominator, rotary phase, learned mixing/re-entry coefficient, unchanged attention fields, and the later suffix. The delta program may omit only the unchanged baseline head write. Outputs to preserve are local head write, signed regional target effect, three unrelated-reader effects, and eventually full logits/CE with the suffix recomputed.

Replay norm is per-sequence relative `L2`, maximum relative `L2`, exact off-support zero, and named term/source closure. Behavioral composition is `||joint-single1-single2||_2 / min(||single1||_2,||single2||_2)` for live singles, compared with matched random splits. Exact Mobius closure is only accounting.

### Literal price

The norm-closed MLP8 interface stores **16,536,963 FP32 values = 66,147,852 bytes**, 3,216 token-table bytes, and a separately priced 4,718,592-byte derived `value_reader @ down` cache. At `T=32` it consumes one native `z8` array and one intervention array, each 36,864 scalars. The nearby exact upstream reader generator stores 21,851,526 FP32 values; the packed eight-head approximation stores 22,432,390. The full deduplicated residual6-to-head9.8 program still lacks a canonical combined price, so these counts are not added as though independent.

The new lower bound prices one narrower object: the exact MLP8 quadratic numerator after the 128-dimensional head9.8 value reader needs at least 128 scalar-product nodes on the declared 16-dimensional witness restriction. It does not price live normalization, QK products, source generation, suffix state, or approximate implementations.

## 2. Primary literature and exact mapping

The best applicable result is the tensor-rank/one-slab-dimension lower-bound framework in Joseph B. Kruskal, [“Three-way arrays: rank and uniqueness of trilinear decompositions, with application to arithmetic complexity and statistics” (1977)](https://doi.org/10.1016/0024-3795(77)90069-6). Kruskal defines a three-way tensor as a sum of triads, relates tensor rank to the dimensions of its slab spaces, and gives sufficient k-rank conditions for essential CP uniqueness.

Map the native restricted MLP8 value numerator as follows. Let `U:[16,1152]` span the witness inputs, `xi in R^16`, and `A=V_9.8 D_8:[128,4608]`. Then

```
y_o(xi) = sum_j A[o,j] (L_j U^T xi)(R_j U^T xi)
        = sum_{p<=q} C[o,p,q] xi_p xi_q.
```

After indexing the 136 symmetric monomials `(p,q)`, `C` is a `128 x 136` output-by-monomial flattening. Any program with `r` scalar products of two linear forms and arbitrary output writers expresses `C` as a sum of at most `r` rank-one output/monomial slabs, hence `rank(C)<=r`. The computed `rank(C)=128` therefore implies `r>=128` for this exact program class.

Assumptions met: a fixed stored FP32 quadratic numerator, linear restriction, exact product-node program class, arbitrary output writers, and no fitting of `C`. Assumptions not supplied: an interval/exact-rational rank certificate, a distribution-wide approximation norm, CP uniqueness, semantic factor labels, live RMS folding into the numerator, or causal adoption. The witness basis is selected from opened activations; it witnesses algebraic rank but is not OOD evidence.

Kruskal's CP uniqueness theorem would require a candidate rank `R` and factor k-ranks satisfying the relevant sum condition (standard three-factor form `k_A+k_B+k_C >= 2R+2`). The tied/symmetric input factors, repeated native structure, gauges, causal position sharing, and untested k-ranks violate or leave those premises open. Thus the lower bound applies, while uniqueness does not.

The executable algorithm costs the fixture SVD plus projections `L U^T`, `R U^T`, coefficient contraction, and an SVD of the `128 x k(k+1)/2` flattening. The present CPU run took 0.69 seconds. It is more informative than another reconstruction sweep because it rules out exact product counts below 128 for this restriction, but it does not rank approximate or causally selected programs.

## 3. Executed CPU consequence

Receipt: [`MLP8_HEAD98_QUADRATIC_FLATTENING_CONTROL_2026-09-19_0611.json`](MLP8_HEAD98_QUADRATIC_FLATTENING_CONTROL_2026-09-19_0611.json). It used `/venv/main/bin/python`, CPU only, two BLAS/OpenMP threads, the frozen norm-closed program and 40-fixture artifact, with both input hashes recorded.

| witness dimension | symmetric monomials | flattening rank | smallest singular value | relative smallest singular value |
|---:|---:|---:|---:|---:|
| 8 | 36 | 36 | 0.2497 | 7.65e-4 |
| 12 | 78 | 78 | 0.1668 | 4.47e-4 |
| 16 | 136 | **128** | 0.1005 | 2.07e-4 |

For the 16-dimensional case the LAPACK rank tolerance is `1.46e-11`, over nine orders below the smallest singular value. This is a robust numerical witness but is deliberately not called a formal certified exact rank.

Executable plan consequence: keep exact port closure and the complete 128-dimensional MLP8-to-value output while identifying the causal grouping. Do not propose an exact `<128`-product replacement for this restricted numerator. Approximate compression remains possible, but it must be selected by causal effect, priced against this bound, and separately pass fresh prediction, selective edit, and composition gates.

## 4. Five-property scorecard

| Property | Current evidence | Verdict |
|---|---|---|
| Held-out/OOD prediction | Regional attention8 write replays on 20 fresh FineWeb document/context cells; earlier fresh Pile intervention passes, but FineWeb signed direction is `88/102 < .90`. Number units/populations and the six-/ten-unit edits transfer to 128 natural rows, but those are not yet a frozen end-to-end output law on declared distinct context cells. | **PARTIAL** |
| Extraction at a declared boundary | Regional residual6+tokens→attention8 and norm-closed MLP8 value interfaces execute with declared native ports. The later full head9.8 product and suffix remain open. The number route has exact folds and edits but no standalone package/boundary. | **PASS at intermediate regional boundaries; incomplete end-to-end** |
| Selective manipulation/removal | Regional fresh Pile edits beat same-site nulls and collateral readers, but corpus direction is unstable. Number named units/populations beat matched unit nulls; VP0 units change their intended axis 50x more than null and mostly spare the pronoun margin, but a preregistered three-reader preservation battery is missing and k=10 moves the pronoun margin 5.4%. | **PARTIAL** |
| Composition/reuse | Regional key/value interaction/smaller is `2.3448`; direct/MLP8 splits and random-split specificity fail. The ten number units look additive across two sites, but no smallest-live-piece Mobius gate with matched random splits establishes composition; cross-task reuse remains untested. | **FAIL overall** |
| Measured simplicity | Regional artifacts have literal local prices and packed attention7 saves 884,736 FP32 values locally; norm closure increases storage. Number unit descriptions are small but omit surrounding populations/ports. The new exact-product lower bound is 128 for one restricted numerator, not an adopted program. | **LOCAL PRICES/LOWER BOUND ONLY** |

Lower replay error, storage, variance, CE, or rank cannot substitute for a missing behavioral property. Evidence labels remain separate: exact contraction is **fold**; native intervention is **edit**; propagation after an edit is **response**; regression/subspace selection is **fit**. Every result must also remain labeled fresh, opened, or replay.

## 5. Organization reconciliation

- `CIRCUIT_GRAPH_REGISTRY_V1.md` remains a generated inventory of 49 packages, 44 manifests, and 33 declared boundaries. It omits `city_mlp8_norm_closed_v1`, and its two `four_trait_verified` rows are not five-property certificates because simplicity is separate and the current composition/null standard is stronger.
- `COMPUTATION_PATH_REGISTRY.md` contains `PATH-REGIONAL-CITY-001` but links the earlier value mediator rather than the norm-closed package or complete head9.8 endpoint. It has no concise number path for v338-v394.
- `CIRCUIT_REGISTRY.md` and `MODULE_DOSSIERS.md` stop the regional navigation at the supplied-norm MLP8 value mediator. The MLP dossier index has the 17 September coupled addendum but no current norm-closed link. MLP1 remains explicitly unconsolidated despite v287-v328.
- The number receipts v338-v394, their embedded preregistration plans, scorecard, board claims, and `in_depth_circuit_number.md` are primary evidence but are orphaned from the circuit/path registries. v392 must be cross-linked with its lambda-chain closure failure, not summarized as an exact unit fold. v393/v394 support a split between agreement `VP0` and pronoun-output axes, but v394 also records 5.4% pronoun-margin movement at k=10.
- Alias reconciliation is still needed: dotted project names `head8.2/head9.8/head4.5`, zero-based array indices, MLP layer names, unit IDs, and output axes (`they-he`, `VP0`) should appear together in one path record. “Copier,” “reader,” and “agreement axis” are operational roles, not module renamings.
- `explanations/for_logan/LATEST.md` now links the norm-closed/fresh regional reports and the evolving number account, while the general explanation index and generated graph remain behind those primary receipts. Primary JSON and correction files override historical `LATEST` wording.
- No existing registry, dossier, explanation index, or primary receipt was edited. `COMPUTATION_PATH_REGISTRY.md` and the runner/number records are concurrently dirty and changing; primary ownership of the typed-face implementation is explicit. A shared navigation repair now risks collision and would become stale before validation.

## 6. Efficiency audit

From 03:06:54 through 06:12:19, the follow-up directory contains **82 receipts**, **1,165 declared forwards**, **83 new/modified sibling runner files**, and Git records **86 commits**. The 80 receipts with sane `serial_seconds < 100` sum to only **159.05 seconds**. Two primary receipts (`reader_key_split_v381` and `reader_key_split_ten_v388`) contain obviously corrupted `serial_seconds` values near 1.8 million seconds each; they must not be used for throughput accounting. They were not modified.

The scientific cadence was roughly one receipt every 2-3 wall minutes. Model processes in `runner.log` are usually 2-4 seconds. v386 lost about 10 minutes to a bare result-file wait after a gate refusal; the existing fail-fast waiter should be mandatory. Repeated inherited constants caused several gate refusals, and v392 omitted lambda-chain scaling. These are authoring/template faults, not GPU limits.

Both Supervisor-managed runners are running, both queues are empty, and `nvidia-smi` showed no compute process at inspection. There is no queued work to rescue and no queue/timer change is justified.

The highest-value refactor is a declarative unit/factor/census executor with one row builder, scorer, fail-fast gate, receipt writer, and schema-validated duration field. It could replace dozens of AST-derived sibling scripts and prevent inherited `PHRASE/BATCH` constants. However, the number lane is actively changing through v394, the shared path registry is dirty, and equivalence testing would cost 30-60 minutes. In this bounded review that costs more and collides more than it saves. The safe nonconflicting action was the uniquely named 0.69-second actual-weight lower-bound receipt above.

Review/phase accounting remains sparse: `RESEARCH_PHASES_2026-09-19.jsonl` contains only the 04:04 and 05:07 review marks, so design, implementation, interpretation, validation, and publication time cannot be reconstructed honestly. Commit gaps are publication cadence, not labor time.

## 7. Cross-track consequences and handoffs

**How circuit evidence selects the fold.** Fresh regional interventions show the head8.2/MLP8/head9.8 route is live, while FineWeb reversals and interaction/smaller `2.3448` reject independent key/value semantics. That selects the complete coupled residual6→MLP8→head9.8 product, with both ordered MLP cross terms and all seven `QK1*QK2*V` delta products, rather than another rank sweep. The actual-weight rank witness further says exact MLP8 value closure cannot use fewer than 128 product nodes on the tested restriction.

**How folded algebra proposes a circuit test.** The seven-term head expansion defines the source of nonadditivity. The next circuit intervention should compare the true joint product with both singles and an additive-without-product arm at the head boundary, then recompute the full suffix and compare with matched random splits. A pass groups terms by the downstream consumer; a failure should split by response/structural context, not QK1, QK2, value, or native module names alone.

The number algebra provides a second reciprocal lesson: unembedding subspace folding exposed `VP0`, and edits show it is partly separable from the pronoun axis. The next number circuit record should group late MLP units by output axis and test matched removals/composition. The v394 k=10 collateral movement is the falsifier against declaring the axes independent.

**WEIGHT_FOLDING handoff:** implement and replay the complete regional head9.8 write with corrected all-nine-head upstream generation, live norm-closed MLP8, native rotary/RMS/re-entry semantics, all source positions, and all seven factor products. Publish one deduplicated price. The 128-product witness is a lower bound for the restricted MLP8 value numerator, not a target to fit blindly.

**CIRCUIT handoff (next hourly track):** freeze structurally varied, genuinely distinct `(template, city-pair, endpoint)` cells and run the full-suffix five-arm coupled regional factorial with at least three unrelated readers, equal-norm same-site nulls, and matched random-split interaction specificity. Preserve the current response census before proposing any suffix. Do not replace this regional handoff with more number attribution.

## Limitations

The lower bound is numerical and restricted to an opened-fixture-selected 16-dimensional subspace. It does not bound approximation error or the complete normalized head/suffix program. The complete native head9.8 closure remains unimplemented. Regional fresh/OOD signed behavior is inconsistent; independent composition fails; the canonical combined price is absent. The number path is promising but post-selected, not extracted, and not fully selective/compositional. No algebraic replay, rank statement, compression, or storage improvement in this review proves native causal fidelity.
