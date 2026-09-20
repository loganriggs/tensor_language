# Hourly strategic review

ACTIVE_TRACK: WEIGHT_FOLDING

Actual UTC: **2026-09-19T07:21:27Z**. Next deadline: **2026-09-19T08:21:27Z**. Previous Codex review: [06:18 CIRCUIT](HOURLY_STRATEGIC_REVIEW_2026-09-19_0618.md), 1h03m09s old. This is one current review, not a backfill. Evidence cutoff is the truthful phase mark at 07:21:27. Historical `/workspace` paths were resolved against the live `/workspace/tensor_language` checkout. No GPU model was loaded; no job, queue, timer, service, experiment/result, commit, push, external contact, or `TYPED_FACE_EXTRACTION_V1` implementation was changed.

The prior `CIRCUIT` hour did not execute its registered regional five-arm factorial, but it did produce a substantive subject-number circuit receipt: v398-v426 traced a value-copy route, ran replace-edits, and closed a serial edit loop. Alternation is mandatory despite that unfinished regional action, so this review switches to `WEIGHT_FOLDING`. The regional factorial remains the next circuit handoff.

## Seven circuit targets and full goal

1. **Computational specification:** identify what is read, the operation/composition, what is written, and its downstream consumers.
2. **Cross-boundary grouping and within-module splitting:** group pieces across modules when a reader treats them as one variable, and split native modules when their pieces serve different computations.
3. **Held-out/OOD prediction:** predict activation and behavioral effects on unseen inputs, task variations, and shifted corpora.
4. **Extraction or sufficiency:** execute an isolated circuit at a declared boundary with every retained port and background named.
5. **Selective manipulation:** removal, swap, or edit changes the intended behavior against appropriate matched controls while preserving unrelated behavior and accounting for redundancy/interactions.
6. **Composition and reuse:** shared subcomputations serve multiple tasks/modules and their joint behavior follows a tested composition law.
7. **Stable identification:** claimed units survive document/corpus splits, gauges, and fitting restarts, or are defined by downstream operational equivalence.

Goal: a smaller transparent tensor program that **predicts held-out/OOD behavior, is extracted at a declared boundary, supports selective manipulation/removal, and composes/reuses**. Simplicity is the fifth, separately measured property: stored scalars/bytes, product nodes, readers/writers, native ports and states, graph edges, and execution. Lower error, storage, rank, variance, or CE cannot substitute for a missing behavioral property.

## Progress, failures, and five-property score

Since 06:18, v398-v410 supplied **fold** evidence that heads 9.6/12.4/15.1 copy subject number mainly through their value branches at noun and post-noun sites, with head 4.5 an earlier value copier; the same degree expansion replayed on 122 natural aligned pairs. v416-v419 supplied **edit** evidence: noun-site reader-value swaps close 0.37 of the natural-text margin, two sites close 0.50, head4.5 closes 0.19, and the serial composition closes 0.57. v423-v426 tied the copied value to MLP8 units 829/953/1030 and the broader MLP chain: three-unit swaps close 0.124, while whole MLP1-8 noun plus MLP4-8 post-noun swaps close 0.431. This is real causal progress and demonstrates why additive fold shares can double-count a chain: MLP8 alone is 0.93 of the MLP5-8 edit.

Failures/corrections remain part of the result. v400 found case essentially untouched and only a small third-person axis. v401 falsified the expected small value-only share; value dominates. v403 preserved a sign-wording failure. v407 double-counted `Down_bias`; v408 corrected closure without changing ranks. v411 needed two failed attempts (mixed-length batch and a comment swallowing an index) and falsified constant-gate claims. v414's `1e-6` relative closure bar was below the observed FP32 floor. v420 missed sign-inverted head15.1 until v421 used a sign-aware template. v424 missed two effect-size bars. The 122 aligned pairs are repeatedly reused/opened; the receipts do not serialize a distinct-context-cell count, so 122 rows/pairs must not be reported as 122 independent context cells.

| Property | Current evidence | Verdict |
|---|---|---|
| Held-out/OOD prediction | Regional residual6-to-attention8 replay holds on 20 fresh FineWeb document/context cells, but signed direction is only 88/102. The number value-copy algebra replays on 122 natural aligned pairs, but these are now opened and their distinct context-cell count is undeclared. | **PARTIAL** |
| Extraction at a declared boundary | Regional residual6→attention8 and norm-closed MLP8 interfaces execute with declared ports. The complete head9.8 product/suffix remain open. The number route has folds and edits but no standalone package or declared executable boundary. | **PASS at regional intermediate boundaries; number FAIL/incomplete** |
| Selective manipulation/removal | Number value/key swaps discriminate strongly; three units beat random units on 99% of pairs. Regional Pile edits beat same-site nulls. Neither route yet has the required fresh three-unrelated-reader preservation battery at its complete boundary, and regional direction reverses on FineWeb. | **PARTIAL** |
| Composition/reuse | v419 gives useful serial composition and v426 closes the edit loop site by site, but there is no smallest-live-piece Möbius test against preregistered matched random splits. Regional interaction/smaller remains 2.3448. Cross-task reuse is absent. | **FAIL overall; local composition evidence only** |
| Measured simplicity | Three named MLP8 units are small, but their upstream populations, ports, and readers are unpriced. Regional components have local prices; norm closure increases storage, and the deduplicated full path price is missing. | **LOCAL PRICES ONLY** |

Evidence stays typed: algebraic expansions are **folds**, replace/swaps are **edits**, downstream allocation after an edit is a **response**, and templates/subspaces are **fits**. Natural text is not automatically fresh; replay/opened/fresh labels remain selection-relative.

## Highest-value WEIGHT_FOLDING action

Implement and exactly replay the complete regional **residual6 + token IDs → head9.8 output-write delta** before the native later suffix. This remains the highest-information folding route because the regional path already has explicit intermediate extraction and causal activity, while its unresolved large interaction directly selects a complete coupled fold. The hour's number edits are valuable, but another number attribution would violate the authority-designated regional depth plan and would not close a port.

**Endpoint and shapes.** For sequence length `T`, the endpoint is the head9.8 write `delta_h98[T,1152]`. Inputs begin at `residual6[1,T,1152]` plus token IDs. Each attention head has projected factors `Q1,K1,Q2,K2,V[T,128]`, causal score factors `p1,p2[T,T]`, and output map `O[128,1152]`. MLP7 and MLP8 use `L,R: 1152→4608`, elementwise product `[T,4608]`, `Down: 4608→1152`, and bias `[1152]`.

**Exact native factors and retained terms.** Generate all nine attention7 writes, then the complete MLP7 response, five head8.2 readers and its city-write delta; apply learned residual re-entry and the live norm-closed MLP8 with `z*z`, `delta*z`, `z*delta`, and `delta*delta` numerator terms, changed RMS denominator, bias, and initial-state lookup. At head9.8 retain native RMS, rounded rotary phases, causal masking, both QK score factors, value, and output projection. For baseline `(p1,p2,v)` and deltas `(dp1,dp2,dv)`, retain all seven nonempty finite-change products:

`dp1*p2*v`, `p1*dp2*v`, `p1*p2*dv`, `dp1*dp2*v`, `dp1*p2*dv`, `p1*dp2*dv`, and `dp1*dp2*dv`.

When a residual is expanded as `r=b+a7+m7+d8`, keep every reader-permitted self and ordered cross term separately, including `a7*a7`, `m7*m7`, `a7*m7`, and `m7*a7`, plus background and `d8` terms. **Omitted:** only the unchanged baseline head write; no nonempty delta product or live source term is licensed for omission yet. Quantization is excluded.

**Background/nonlinearities.** Keep explicit any native query/source state not generated by this path, every RMS denominator and epsilon, rotary phase, causal mask, learned mixing/re-entry coefficient, unchanged head field, MLP bias, and the native later suffix. Do not average the context slot. Fold/edit/response/fit remain separate.

**Simplicity measure.** Publish one deduplicated count of independent FP32 scalars/bytes, product nodes, readers/writers, source terms, graph nodes/edges, native ports/state scalars at declared `T`, and multiply-add execution. Do not add overlapping package prices. Existing anchors are 16,536,963 FP32 values for the norm-closed MLP8 interface and 21,851,526 for the exact upstream reader generator; the combined deduplicated price is unknown. The 128-product flattening witness applies only to a restricted exact MLP8 value numerator.

**Exact replay check.** On the frozen 40 opened fixtures and the 40 fresh sequences from 20 distinct FineWeb document/context cells, compare every intermediate factor and final head write to native capture; require named-source closure, exact off-support zero, max relative `L2 <= 1e-4`, and no unexplained precision-sensitive residual. Recompute from source tensors rather than loading the target write. Failure of any intermediate closure kills the fold until corrected.

**Later causal/predictive falsifier.** Freeze structurally varied, genuinely distinct `(template, city-pair, endpoint)` cells and run the full-suffix five-arm factorial: baseline, routing/key single, value single, additive installation without their product, and true joint native product. Score signed regional effect, at least three unrelated readers, full logits/CE, 16 equal-norm same-site nulls, and matched random splits. Kill the retained grouping if the signed effect is not predicted, an unrelated reader breaches its gate, a single arm is dead, or true-joint-minus-additive interaction is no more specific than matched splits. A closed Möbius identity alone does not establish small interaction relative to the smallest live piece.

**Concrete track connection:** v425 shows experimentally that writer-fold shares can double-count a serial chain, while v416-v426 edits identify what the downstream value readers actually use. The regional circuit factorial should likewise decide which of the seven exact head9.8 terms forms a consumer-relevant group; the fold must expose all terms first so the edit can distinguish true interaction from correlated attribution.

## Confounds, alternatives, and ranked decisions

The fold must control baseline subtraction and replace-versus-zero semantics; frame/position/city/lexical mixing; shared token/document difficulty; FineWeb/Pile leakage; live RMS, rotary, re-entry, MLP and suffix nonlinearities; dead arms/tripwires; FP32 floors; and post-selection. Actual distinct context cells, not row counts, govern independence.

1. **Complete regional head9.8 closure** — highest information; changes targets 1, 2, and 4 and creates the executable object for targets 3, 5, and 6. Killed by failed native replay, an undeclared port, or a price that merely duplicates opaque native computation without closing an interface.
2. **Number value-copy standalone boundary** — now plausible because folds and edits meet at both sites. Defer to its next folding hour; kill if exact token/MLP-to-reader replay requires the full native residual or if the three-reader preservation battery fails.
3. **Predictive-state quotient or signed composition basis** — consider only if the regional five-arm interaction varies systematically by structural cell. Kill if a frozen quotient does not predict a genuinely new cell family.
4. **More unembedding/subspace or rank sweeps** — demoted: VP0 already failed natural transfer, and lower rank/replay alone cannot establish a circuit.

## Registry and dossier audit

`CIRCUIT_GRAPH_REGISTRY_V1.md` still reports 49 packages/44 manifests/33 boundaries, omits `city_mlp8_norm_closed_v1`, and uses “four-trait verified” without the separate simplicity/five-property standard. `PATH-REGIONAL-CITY-001` now records exact-RMS regional closure but still does not link the norm-closed package or name the complete head9.8 endpoint/combined price. `CIRCUIT_REGISTRY.md`, `MODULE_DOSSIERS.md`, and the MLP8 dossier navigation stop before the complete norm-closed path.

The largest missing cross-link is still a single `PATH-SUBJECT-NUMBER-002`-style record connecting v338-v426: aliases for head4.5, heads9.6/12.4/15.1, MLP units/sites and output axes; the v392 lambda correction; failed VP0 transfer; v407/v408 bias correction; and the v416-v426 edits/composition. It must state that 122 pairs are not a declared independent-cell count. The shared path registry is concurrently dirty and the active lane was still publishing v426 at cutoff, so this bounded review did not overwrite it.

## PAST_HOUR_TIMING

Window: **2026-09-19T06:18:18Z-07:21:27Z**. Twenty-nine receipts v398-v426 finished 06:18:30-07:19:00 and declare **296 forwards**. Their internal `serial_seconds` sum to **55.659s**. Managed runner start/end lines measure about **68s** for successful science processes and **72s including two failed v411 attempts**. Completion-to-completion candidate latency was usually about 1-2.5 minutes; the longest observed gaps were 9m03s before v401 and 8m48s before successful v411. These gaps are not labeled active labor.

- **Design:** unknown; receipt plans/predictions exist, but no design phase boundaries exist.
- **Implementation/data building:** unknown; 29 sibling runners were committed, but commit gaps are not labor time.
- **Validation:** 72s of rounded managed process time includes two 2s v411 failures; debugging time between attempts is unknown. v407→v408 correction took 1m30s completion-to-completion, not necessarily active validation time.
- **Model execution:** 55.659s internal successful-receipt runtime; about 68s rounded successful process wall. These serial processes loaded no model according to the receipts and used existing managed machinery.
- **Interpretation:** unknown; result-to-board/scorecard updates are visible but uninstrumented.
- **Documentation/review:** Claude reviews were published at 06:21 and 07:07; active authoring duration is unknown. `for_logan/LATEST.md` changed at 06:43; duration unknown.
- **Idle/blocked:** unknown. The 9m03s and 8m48s gaps cannot be assigned to idle, design, debugging, or publication from authoritative records.

`RESEARCH_PHASES_2026-09-19.jsonl` contained only 04:04, 05:07, and 06:18 review boundaries before this checkpoint. It covers none of the intervening scientific phases. The existing helper appended one truthful 07:21:27 review boundary; no phase was backfilled and no interval is treated as uninterrupted work.

## PROCESS_IMPROVEMENT

Ranked by time likely saved versus cost:

1. **Declarative shared number fold/edit runner with schema-validated constants and receipt timing:** likely saves 1-2 minutes per candidate and would have prevented the v407 bias duplication, v411 swallowed-index/mixed-batch failures, and much sibling-script authoring; estimated 30-60 minutes plus equivalence tests. It is valuable but not a safe bounded repair while the same lane and files are changing.
2. **Record `started_utc` plus phase category in managed receipts:** low implementation cost and would separate queue/authoring gaps from execution, but changing the active shared runner exceeds this review's nonconflicting scope.
3. **One number-path registry record:** likely saves 5-10 minutes per handoff for under 10 minutes once the sequence freezes; currently collides with the dirty shared registry and would immediately stale under concurrent publication.

No engineering refactor was justified inside this bounded review: the only safe CPU process action was the existing append helper's truthful phase boundary. Mutating active runner/shared registry code would risk concurrent work and exceed the audit's intended size. The review itself records the bounded novelty repair: no further number candidate should be opened without first consolidating v338-v426 and its failures into the computation-path/module navigation.

## Explicit verdicts and handoffs

- `TRACK_ALTERNATION: PASS` — previous `CIRCUIT`; current `WEIGHT_FOLDING`.
- `TRACK_PROGRESS: PASS WITH SCOPE DEVIATION` — the prior hour produced substantive causal circuit edits, matched unit nulls, and local serial composition (v416-v426), but displaced the registered regional factorial.
- `CEREMONY_BUDGET: PASS WITH TIMING CAVEAT` — 29 decision-bearing receipts and 296 forwards dominate two short reviews; however, only ~72s of process time and sparse phase marks are measured, so authoring/review dominance cannot be ruled out.
- `NOVELTY_LESSON_GATE: FAIL` — the work applied “folds nominate, edits decide” and preserved falsifications, but repeated bespoke runners, two v411 authoring failures, an FP32-impossible bar, and the still-orphaned v338-v426 path show the registry/dedup lesson is not closed. The bounded repair is the explicit consolidation gate above; no conflicting registry edit was made.

**Highest-value action now:** complete the exact regional residual6→norm-closed MLP8→head9.8 write replay with all nine upstream heads, both ordered MLP cross terms, and all seven `QK1*QK2*V` delta products, then publish one deduplicated price.

**CIRCUIT handoff:** next circuit hour, run the preserved frozen distinct-cell regional full-suffix five-arm factorial with three-reader preservation, equal-norm same-site nulls, and matched random-split interaction specificity; do not substitute another number census.
