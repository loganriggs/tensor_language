# Three-hour mathematical, organization, and efficiency review

Actual UTC: **2026-09-19T09:19:22Z**. Next mathematical deadline: **2026-09-19T12:19:22Z**. The preceding review was written at `2026-09-19T06:14:44Z`, about 184.6 minutes earlier, so this review was due under the 175-minute gate. It is one live review, not an offline backfill. Historical `/workspace/...` paths were resolved against the live checkout `/workspace/tensor_language`.

This bounded review used no agent, GPU/model forward, job, queue, timer, service, commit, push, external contact, or `TYPED_FACE_EXTRACTION_V1` change. It wrote one new CPU control receipt, this review, and a short board append. The newest hourly review remains `ACTIVE_TRACK: CIRCUIT` at 08:23; this review does not alter hourly alternation.

## Verdict

The authority-designated regional path remains the best target for depth because it has actual distinct-cell fresh extraction and selective evidence at intermediate boundaries. Its unresolved object is not an additive key path plus value path: on the existing 20-context-cell opened factorial, the unique order-two interaction is 2.3448 times the smaller live single. The executed consequence below strengthens this into an approximation lower bound: under the uniform measure on the four Boolean arms, **every additive two-branch program has best possible target-readout RMS error at least 0.5862 times the smaller live single-effect norm**. Every one of the 20 cells has a nonzero bound (range 0.3420–1.3355, median 0.4522). Thus the next fold must retain the coupled full `QK1*QK2*V` operation; an additive surrogate is mathematically incapable of fitting this observed factorial well even if it is allowed to move the baseline and both single arms.

This is **fold/opened** evidence and a lower bound on an observed arm function. It is not an edit, fresh/OOD confirmation, random-split specificity result, or proof of native causal fidelity. The full regional head9.8 port closure and preregistered full-suffix factorial remain the highest-information handoffs.

Current five-property score: **held-out/OOD prediction PARTIAL; extraction PASS only at declared intermediate regional boundaries; selective manipulation/removal PARTIAL; composition/reuse FAIL overall; measured simplicity LOCAL ONLY**. The number lane has substantially sharpened a two-readout circuit, but it remains opened/natural rather than fresh by selection, is not independently extracted, lacks a complete matched-null preservation/composition certificate, and has no deduplicated literal price.

## 1. Native computation as mathematical objects

### Tensors, indices, shapes, parameters, and nonlinearities

Let sequence positions be `t,s in {0,...,T-1}` with causal source `s<=t`; residual indices `a,o in {1,...,d}`, `d=1152`; heads `h in {1,...,9}`; head coordinates `c in {1,...,d_h}`, `d_h=128`; MLP product coordinates `j in {1,...,m}`, `m=4608`; blocks `ell in {0,...,17}`; and vocabulary rows `v in {1,...,50304}`. The checkpoint has **545,902,902** parameters and an untied unembedding. Weights are tied across positions/examples but not across layers. Learned residual and first-state re-entry coefficients are literal parameters.

For residual `x in R^1152`, with native epsilon `eps=1.1920928955078125e-7`,

```
S(x) = ||x||_2^2/1152 + eps,
RMS(x) = x/sqrt(S(x)).
```

MLP `ell` has `L_ell,R_ell in R^(4608 x 1152)`, `D_ell in R^(1152 x 4608)`, and `b_ell in R^1152`:

```
M_ell(x) = D_ell[(L_ell x) odot (R_ell x)]/S(x) + b_ell.
```

If `x=sum_i x_i` names earlier residual, attention, MLP, re-entry, and installed-edit sources, the numerator is

```
N_ell(x) = sum_i D[(L x_i) odot (R x_i)]
         + sum_{i != j} D[(L x_i) odot (R x_j)].
```

The first sum is self terms; the second is ordered cross terms. For the live regional input `x=b+a7+m7+d8`, the fold therefore includes `a7*a7`, `m7*m7`, `a7*m7`, and `m7*a7`, as well as every permitted term with background `b` and city write `d8`. The two ordered cross terms are distinct native contractions even when their symmetrized sum is later used.

For attention head `(ell,h)`, after native RMS and rounded rotary maps,

```
p^r[t,s,h] = <Rot_t Q^r_{ell,h} xhat_t,
                Rot_s K^r_{ell,h} xhat_s>/128,  r in {1,2},
v[s,h] = V_{ell,h} xhat_s,
A_{ell,h}(x)_t = O_{ell,h} sum_{s<=t} p^1[t,s,h] p^2[t,s,h] v[s,h].
```

Each `Q^r,K^r,V` is `128 x 1152`, and `O` is `1152 x 128`. There is no softmax or SiLU. With fixed RMS denominators, one QK score is bilinear/degree two in named residual sources and a head output is degree five (`QK1*QK2*V`). A bilinear MLP numerator is degree two. With live RMS denominators these maps are rational with square roots. The final map `30*tanh(U RMS(h)/30)` is non-polynomial, so no finite-degree whole-network polynomial claim is made.

For a source expansion `x_t=sum_i x^q_i` and `x_s=sum_j x^k_j`, each score factor is

```
p^r[t,s] = sum_{i,j} B^r_{ij}[t,s],
B^r_{ij}[t,s] = <Rot_t Q^r x^q_i[t], Rot_s K^r x^k_j[s]>/128.
```

The full head contains `B^1_ij * B^2_kl * V x^v_n` for every reader-permitted `(i,j,k,l,n)`. Earlier attention and MLP sources therefore generate attention-self, MLP-self, and both ordered attention/MLP cross terms in **each** QK factor, followed by all QK1-by-QK2-by-value interactions. An analysis of QK1 or QK2 alone is not the native routing computation.

### Contraction graph, ties, gauges, backgrounds, and outputs

The current regional contraction graph is

```
(token IDs, native residual6)
 -> all nine attention7 heads
 -> complete MLP7 response and five head8.2 readers
 -> full head8.2 city-write delta
 -> native z8 + delta8
 -> norm-closed MLP8 (bias, skip/re-entry, z*z, z*d, d*z, d*d)
 -> head9.8 QK1 * QK2 * V with rotary/RMS/causal mask
 -> native blocks9..17 and final RMS/softcap
 -> regional target, >=3 unrelated readers, full logits/CE.
```

Hidden-unit permutations; reciprocal `L_j -> c L_j, R_j -> R_j/c`; `L/R` exchange after symmetrization; compatible Q/K basis transformations; `p1 -> c p1, p2 -> p2/c`; and value/output basis changes are gauges. Raw factor magnitude or one coordinate is not a semantic identity. Operational equivalence under declared downstream readers and interventions is the relevant identification relation.

Explicit backgrounds are native residual6, query/source states not generated by the path, every RMS denominator, rounded rotary phase, causal mask, learned mixing/re-entry, unchanged attention fields, MLP biases, and the complete later suffix. Allowed inputs are token IDs plus the declared native residual arrays at the chosen boundary; hidden native inputs count as open ports. Outputs to preserve are the signed regional target effect, at least three unrelated-reader effects, full logits/CE, and local intermediate writes needed for replay.

Replay uses per-sequence relative `L2`, maximum relative `L2`, off-support zero, and named source/term closure. Behavioral composition uses interaction norm divided by the smallest **live** single effect and a preregistered matched random-split distribution. A closed Möbius identity is only accounting.

### Current circuit object

The primary circuit object is `PATH-REGIONAL-CITY-001` extended toward the complete head9.8 output and suffix. Existing exact residual6-to-attention8 generation reaches `1.97e-6` maximum relative error on 40 opened fixtures and `1.61e-6` on 40 fresh FineWeb sequences from **20 distinct document/context cells**. The norm-closed MLP8 object retains baseline self, both ordered baseline/edit cross terms, edit self, changed normalization, bias, re-entry, and initial-state lookup. Fresh same-boundary prediction/selectivity exists, but FineWeb sign transfer for the packed prefix is `88/102 < .90`, and complete key/value composition fails.

The concurrent number circuit candidate is now more precise: an MLP8 hub (`829/953/1030`) feeds almost disjoint pronoun and verb value-copy readouts. Pronoun heads `9.6/10.1/10.5/12.4/15.1` jointly close `0.803` of the pronoun margin at the answer; verb heads including `11.3/5.3/7.8/9.x/13.1` jointly close `0.557` of distant agreement, with MLP17 units `701/2059` a verb-specific final detector. v487/v488 show asymmetric cross-selectivity (`0.041` verb-head effect on pronouns; `0.096` pronoun-head effect on verbs); v493 shows opposite gender tilts cancel; v496 repairs two invalid replay references; v498 shows the adjacent detector is a singular suppressor rather than the same distant-row vector; v500 attributes only `0.031` to the five verb heads' self-reads versus `0.107` for the five-head verb set and `0.260` for all values. These are chiefly **edits/folds on opened natural rows**, not a fresh extracted circuit.

### Current folded-path object and retained subset

For baseline head9.8 factors `p1,p2 in R^(T x T)`, `v in R^(T x 128)` and intervention deltas `dp1,dp2,dv`, define

```
F(p1,p2,v)_t = O_9.8 sum_{s<=t} p1[t,s] p2[t,s] v[s].
```

The exact finite change is the seven nonempty products

```
F(dp1,p2,v) + F(p1,dp2,v) + F(p1,p2,dv)
+ F(dp1,dp2,v) + F(dp1,p2,dv) + F(p1,dp2,dv)
+ F(dp1,dp2,dv).
```

The **current retained subset is all seven terms**, together with every live self/ordered-cross source term inside each QK score and MLP8. The only omitted term is the unchanged baseline head write. No smaller subset is licensed because the existing key/value factorial is strongly nonadditive, QK2 controls are material, and source/random-split partitions have failed. The next causal factorial may identify a consumer-relevant grouping; algebra alone may not.

### Norm and literal program price

The native checkpoint price is 545,902,902 trained parameters. A single native head stores six `128 x 1152` maps when the two QK pairs, V, and O are counted: **884,736 FP32 values**, before biases/mixing. At length `T`, it evaluates four Q/K projections, one V projection, one O projection, two causal width-128 dot products per live `(t,s)`, one scalar score product and one score-times-value operation per pair.

The norm-closed MLP8 interface stores **16,536,963 FP32 values = 66,147,852 bytes**, a 3,216-byte token table, and a separately priced 4,718,592-byte derived `value_reader @ down` cache. At `T=32` it requires native `z8` and intervention arrays of 36,864 scalars each. The exact upstream reader generator stores 21,851,526 FP32 values; the packed eight-head approximation stores 22,432,390. These overlap, so adding them would double-price shared weights. A canonical deduplicated residual6-to-head9.8 count remains missing. Consequently there is no whole-path simplicity pass.

## 2. Primary mathematics and exact mapping

The relevant exact theorem is Möbius inversion on a locally finite poset. Rota defines the incidence algebra, proves that its zeta function has a unique inverse, and states the inversion formula; on a Boolean algebra, the Möbius coefficient is `(-1)^(|y|-|x|)`. Primary source: Gian-Carlo Rota, [“On the Foundations of Combinatorial Theory I: Theory of Möbius Functions” (1964)](https://webhomes.maths.ed.ac.uk/~v1ranick/papers/rota1.pdf), especially Proposition 2 and the Boolean-algebra corollary.

Map the regional two-factor intervention object to the Boolean lattice `2^{K,V}`. Let `f00,f10,f01,f11` be the vector of target/readout score changes for baseline, key-only, value-only, and true joint native installation on the same rows. Möbius inversion gives the unique interaction

```
D = f11 - f10 - f01 + f00.
```

This exactly identifies the order-two arm coefficient; it does **not** say that `D` is small, selective, fresh, or special relative to random splits.

The executable consequence adds a direct Hilbert-space projection. Under the uniform measure on the four corners, the additive subspace consists of `a + b*K + c*V`. Orthogonal projection gives residuals `(D,-D,-D,D)/4`, so

```
min_additive sqrt((1/4) sum_corner ||f-candidate||_2^2) = ||D||_2/4.
```

This is an exact finite-dimensional identity for vector outputs. Complexity is linear in the stored artifact size: four vector additions plus norms, `O(arms * rows * readouts)`, with no fitting or model load. Assumptions met: all four arms on aligned rows, common vector space/norm, uniform arm measure, and an unconstrained additive candidate. Assumptions violated or absent for broader claims: the rows are opened; the empirical cell measure is not a corpus distribution; interventions need not correspond to independently deployable native modules; matched random-split specificity is absent; and the live suffix makes `f` a nonlinear causal response rather than the local tri-affine head alone.

Kruskal's three-way tensor rank machinery remains relevant to the previous 128-product lower bound, but it does not solve the present normalized full-suffix object or identify semantic factors. No theorem found licenses dropping QK2, value, RMS, background, or suffix terms.

## 3. Executed CPU consequence

Receipt: [`REGIONAL_FACTORIAL_ADDITIVE_LOWER_BOUND_2026-09-19_0918.json`](REGIONAL_FACTORIAL_ADDITIVE_LOWER_BOUND_2026-09-19_0918.json). It used `/venv/main/bin/python`, CPU only, float64, two BLAS/OpenMP threads, and the frozen `CITY_INTERCHANGE_COMPOSITION_V1_ARTIFACT.pt` (SHA256 recorded). Runtime was 0.0023 seconds; no model was loaded.

| scope | interaction / smaller single | best additive factorial RMS / smaller single |
|---|---:|---:|
| target readout | 2.3448 | **0.5862** |
| all five readouts | 2.3032 | 0.5758 |
| four controls | 1.4644 | 0.3661 |

Across the 20 declared target context cells, the lower-bound ratio ranges `0.3420–1.3355` with median `0.4522`; it is not a row-count artifact caused by one cell. The product-specific and suffix interaction vectors have cosine `-0.6762`, confirming partial cancellation rather than independence.

Plan consequence: reject additive key-plus-value compression for the current regional arm function. Complete the exact coupled residual6-to-head9.8 fold with all seven head delta products, then run the preregistered full-suffix factorial on genuinely fresh structural cells with matched random splits. Only that experiment can decide whether a new circuit grouping is specific and causally reusable.

## 4. Five-property scorecard

| property | current evidence | verdict |
|---|---|---|
| Held-out/OOD prediction | Regional extraction replays on 20 fresh FineWeb document/context cells and selective Pile panels, but packed FineWeb direction is `88/102`; the new number route repeatedly reuses opened 122 aligned pairs with no declared distinct-cell manifest or frozen OOD law. | **PARTIAL** |
| Extraction at a declared boundary | Regional residual6-to-attention8 and norm-closed MLP8 interfaces execute with declared ports. Complete head9.8 and native suffix remain open. The number route has folds/edits but no standalone boundary or complete port list. | **PASS at intermediate regional boundaries; incomplete end-to-end** |
| Selective manipulation/removal | Regional fresh edits beat same-site nulls and collateral readers at scoped boundaries, but corpus direction varies. Number head/unit swaps show target separation and some random-unit controls, but lack a fresh preregistered three-reader preservation battery and equal-norm same-site directions at the complete boundary. | **PARTIAL** |
| Composition/reuse | Regional key/value interaction/smaller is `2.3448`; the new additive lower bound is `0.5862` of the smaller single. Prior random source/write splits fail specificity. Number routes show local additivity/reuse across two readouts, but not a smallest-live-piece Möbius gate against matched random splits on fresh cells. | **FAIL overall** |
| Measured simplicity | Regional components have local storage/state prices and one 128-product lower bound; no deduplicated full-path price exists. Number named heads/units omit upstream populations, ports, suffix, and actual executor cost. | **LOCAL ONLY** |

Lower error, storage, rank, variance, CE, or arm replay cannot substitute for a missing behavioral property. Evidence types remain separate: **fold**, **edit**, **response**, **fit**; evaluation status remains **fresh**, **opened**, or **replay**.

## 5. Organization reconciliation

- `COMPUTATION_PATH_REGISTRY.md` now improves `PATH-REGIONAL-CITY-001` with the exact-RMS fresh replay and norm-closed package links. It still ends at the head8.2 city-removal write rather than giving the complete head9.8 endpoint, seven retained factor products, suffix test, and one deduplicated price.
- `CIRCUIT_REGISTRY.md` and `MODULE_DOSSIERS.md` preserve the regional evidence through 18 September, including the 2.3448 composition failure. They do not consolidate v396-v500's current number graph, the five-head pronoun set, corrected five-head verb set, units `829/953/1030`, detector `701/2059`, v494/v495 invalid instruments, or v500 self-read result.
- No `PATH-SUBJECT-NUMBER-002` exists. The current number work is split among 104 new receipts since 06:14, `in_depth_circuit_number.md`, board claims, scorecard rows, hundreds of sibling runners, and commit messages. `PATH-SUBJECT-001` is the older L11H3 response-weighted object and must not be silently repurposed.
- `CIRCUIT_GRAPH_REGISTRY_V1.md` remains a 49-package generated inventory. It omits `city_mlp8_norm_closed_v1`, treats two older rows as `four_trait_verified`, and says `next gap: none` without the separate simplicity/five-property standard. Those labels are inventory metadata, not current certification.
- `explanations/README.md` contains many repeated “Latest regional result” pointers and does not expose the current number account. `for_logan/LATEST.md` points to the correct regional upstream closure but predates the newest number narrative, while `in_depth_circuit_number.md` is changing live.
- Alias debt remains: dotted heads (`9.6`, `11.3`), zero-based array indices, block names, unit IDs/sites, pronoun/verb axes, “copier,” “reader,” “hub,” and “detector” need one canonical path record. Operational roles must not be mistaken for module renamings.
- Preregistration and primary receipt distinctions are locally present, but v499/v500 expose a handoff flaw: a derived runner inherited a two-batch price while the row set required three, and the board reports 15 forwards while the surviving v500 receipt reports 10. The primary receipt was not modified; the inconsistency must be preserved and reconciled in the eventual dossier.
- Primary ownership of `TYPED_FACE_EXTRACTION_V1` was respected. No primary receipt, shared registry, dossier, or explanation index was modified.

## 6. Efficiency audit

From 06:14 through 09:18, the follow-up directory records **104 completed receipts**, **1,907 declared forwards**, and **105 new sibling `run_*.py` files**. All 104 sane `serial_seconds` sum to **229.76 seconds**, median **2.08 seconds**, maximum **4.40 seconds**. Git records 108 commits and the board 113 timestamped claims since the cutoff. This is roughly one completed receipt per 1.77 wall minutes, but only 3.83 minutes of successful compute in 184 minutes; design, derived-script editing, corrections, publication, queue gaps, and idle time are not phase-measured.

The scientific lane is compute-cheap and authoring-heavy. Repeated ceremony/faults are visible in v494-v496 (two incorrect replay references before the clean instrument), duplicate execution of v496, and v499/v500 (inherited batch-price mismatch, failed/ambiguous rerun, receipt/board forward-count inconsistency). The 105 runners for 104 receipts are direct evidence that the dominant throughput opportunity is shared declarative execution, not faster model compute.

Both Supervisor-managed runners were `RUNNING` at inspection; both queues were empty; no GPU compute process was visible. The runner log showed the v500 rerun completed at 09:18. There was no queue/runtime action to rescue and changing Supervisor, queues, or timers was outside this review.

The highest-value refactor remains one declarative number-circuit runner with a shared row builder, factor/value/write intervention schema, scorer, fail-fast price derived from actual batches, receipt writer, and schema checks for `started_utc`, `finished_utc`, distinct context cells, forwards, and evidence/freshness labels. It would remove dozens of near-copy scripts and likely prevent the observed reference and price failures. I did **not** make it now: the number lane and its runner inputs were changing through 09:18, primary files are concurrently owned, and equivalence testing a shared executor would exceed this bounded review. The executed nonconflicting mathematical receipt provided the required bounded consequence with higher immediate information value.

## 7. Cross-track consequences and actionable handoffs

**Circuit evidence selects the fold.** Regional fresh/selective evidence makes head8.2/MLP8/head9.8 the depth target, while sign reversals, material QK2 controls, interaction/smaller `2.3448`, failed random source/write splits, and the new additive lower bound reject an independent key/value decomposition. Therefore retain exact port closure, full QK1-by-QK2-by-V interactions, both ordered MLP cross terms, and the complete context slot.

**Folded algebra proposes a circuit grouping/test.** The seven-term head expansion separates first-order factor changes, three pair products, and the triple product. The factorial projection says the consumer-relevant object cannot be an additive key/value pair on current cells. The next edit should compare baseline, live singles, an additive-without-product head arm, and true joint product through the full suffix, then test the interaction against the smallest live piece and matched random splits. If the interaction is specific, group the terms at the downstream consumer boundary; if not, split by structural/context response rather than native QK/value labels.

The number evidence provides the reciprocal lesson: folds nominate one noun-number state, while edits split it into a shared MLP8 hub, nearly disjoint pronoun/verb reader sets, and a verb-only final detector. A future fold should group by operational readout and position, not by whole block. A future circuit test must freeze this grouping on distinct fresh cells and compare matched same-size head/unit splits.

**WEIGHT_FOLDING handoff:** complete and replay `residual6 + token IDs -> all nine attention7 writes -> complete MLP7/head8.2 -> norm-closed MLP8 -> head9.8 output delta`, retaining every ordered residual-source term and all seven nonempty factor products. Require intermediate replay on 40 opened fixtures and 20 genuinely distinct fresh FineWeb cells, exact off-support zero, and one deduplicated storage/state/node/edge/compute price.

**CIRCUIT handoff:** execute the frozen regional full-suffix factorial on structurally varied, distinct `(template, city pair, endpoint)` cells with target sign predictions, at least three unrelated readers, full logits/CE, 16 equal-norm same-site directions, and matched random splits. Report interaction relative to the smallest live single and the split-null distribution. Exact Möbius closure alone is not a pass.

**Organization handoff:** once the live number sequence freezes, add one canonical `PATH-SUBJECT-NUMBER-002` linking v396-v500, aliases, corrections, primary receipts, current narrative, modules, opened/fresh status, and missing gates. Do not copy the narrative into every registry.

## Limitations

The new lower bound is exact for the stored four-arm opened artifact under a uniform factorial norm; it is not a distributional, random-split, semantic, or causal-identification theorem. It uses no new model forward. The regional full head/suffix executor and deduplicated price remain incomplete. FineWeb signed behavior is inconsistent and independent composition fails. The number route is rich but post-selected, repeatedly opened, unconsolidated, and not standalone. No algebraic replay, Möbius identity, approximation lower bound, compression, storage saving, or low rank in this review proves native causal fidelity.
