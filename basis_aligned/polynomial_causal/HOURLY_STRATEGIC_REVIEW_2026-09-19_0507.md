# Hourly strategic review

ACTIVE_TRACK: WEIGHT_FOLDING

Actual UTC: **2026-09-19T05:07:07Z**. Next deadline: **2026-09-19T06:07:07Z**. Previous Codex review: [2026-09-19 04:04 CIRCUIT](HOURLY_STRATEGIC_REVIEW_2026-09-19_0404.md), 1h02m54s old at this review's clock. This is one current review, not a backfill. Historical `/workspace` paths were resolved against this live `/workspace/tensor_language` checkout. No GPU model was loaded; no job, timer, service, commit, push, external contact, experiment/result, or `TYPED_FACE_EXTRACTION_V1` implementation was changed.

The circuit hour produced 30 receipts, v338-v367, rather than the proposed regional factorial. It nevertheless made substantive circuit progress: exact folds traced noun/verb agreement factors through MLP writers; edits rejected unit 3465 alone as behaviorally important, identified a causal MLP3 mismatch population, and connected an MLP3 trio through head4.5 to the pronoun-number reader. Alternation now proceeds to `WEIGHT_FOLDING`; unfinished circuit work is preserved rather than allowed to repeat the track.

## Seven circuit targets and full goal

1. **Computational specification:** what is read, what operation/composition occurs, what is written, and which downstream computations use it.
2. **Cross-boundary grouping and within-module splitting:** group pieces across modules when a reader treats them as one variable, and split a native module when pieces serve different computations.
3. **Held-out and OOD prediction:** predict activations and behavioral effects on unseen inputs, task variants, and shifted corpora.
4. **Extraction or sufficiency:** execute at a declared boundary with every retained input and background named.
5. **Selective manipulation:** removal, swap, or edit changes the intended behavior against matched controls while preserving unrelated behaviors and accounting for redundancy/interactions.
6. **Composition and reuse:** shared subcomputations serve multiple tasks/modules and their joint behavior follows a tested composition law.
7. **Stable identification:** units survive document/corpus splits, gauges, and restarts, or are defined by downstream operational equivalence.

Goal: a smaller transparent tensor program that **predicts held-out/OOD behavior, is extracted at a declared boundary, supports selective manipulation/removal, and composes/reuses**. Simplicity is a fifth, separately priced property: stored scalars/bytes, native ports/state, readers, products, writers, nodes, edges, and execution. Lower error, storage, or CE cannot stand in for a missing behavioral trait.

## Progress, failures, and five-property score

Evidence types remain separate: v338-v361 and v363-v367 are folds/responses; v350-v351, v354 and v362 are edits; none is a fitted mechanism. The vocabulary panels and 128 natural sentences are not automatically independent context cells, and selection of the named units/populations makes later tests opened unless their row manifest was frozen before that selection.

| Property | Current evidence | Verdict |
|---|---|---|
| Held-out/OOD prediction | The regional residual6-to-attention8 write still exactly replays on 20 distinct fresh FineWeb documents, but regional behavior reverses on FineWeb (`88/102 < .90`). In the number path, MLP1-fed agreement units transfer from designed rows to 128 natural sentences; head4.5 and its MLP3 writers also remain live there (v364-v366). These are useful frozen-unit transfers, not yet a fresh end-to-end behavioral prediction on declared distinct context cells. | **PARTIAL** |
| Extraction at a declared boundary | Regional intermediate packages remain executable with explicit native ports. The number route now has exact folds `MLP1 lookup -> MLP3 factor/unit population -> head4.5 value -> reader`, but no standalone package, exact port closure, or closed suffix. | **PASS only at regional intermediate boundaries; number extraction incomplete** |
| Selective manipulation/removal | Regional Pile removal and fresh mediator edits beat same-site nulls, while FineWeb direction remains unstable. For number, zeroing MLP3 `{3465,114,493}` changes the target margin `-3.6%`, `16x` the largest random-trio null; a top-10 mismatch population changes continuation log-odds by `-0.73`, while 3465 alone is near-inert (`KL=.0008`). The leaders are 30-58x quieter on grammatical natural verbs, but a multi-reader preservation battery is absent. | **PARTIAL** |
| Composition/reuse | The MLP3 trio edit is locally additive, but no preregistered Mobius/smallest-live-piece and matched-random-split test establishes composability. Regional key/value interchange remains strongly nonadditive (`interaction/smaller=2.3448`). Cross-task reuse is untested. | **FAIL overall; local additivity only** |
| Measured simplicity | Named units and a trio are small descriptions, but the top ten explain only about 21% of the relevant MLP3 contrast and no executable matched-effect price exists. Regional packages have literal local prices, but the norm-closed mediator is interface closure rather than compression. | **LOCAL PRICES ONLY** |

Important corrections and constraints:

- **fold, opened/design-to-natural:** v338's clean frame labeling of the two MLP factors does not transfer to sentence rows (v339). The factors are mixed; retain the exact bilinear product rather than naming one side “noun” and the other “licensing.”
- **fold, natural:** v341-v344 show that MLP1's context-free lookup supplies 58-88% of its contribution to both factors of several agreement units, with MLP2 adding conditioned signal. v367 shows MLP8 unit829 instead receives roughly 69-71% from MLPs5-7 collectively; attention4 is only 1.8-2.7%, falsifying the proposed prominent copier input.
- **edit, opened:** v350's relative-to-near-zero-null gates misleadingly pass despite a negligible absolute effect. v351 repairs the inference with a population edit; future gates need an absolute floor as well as null ratios.
- **fold/edit, opened then natural transfer:** v359 rejects “head4.5 copies the determiner”: 99% of its target write is in the noun's own value. MLP3 supplies about half; units 3465/114/493 lead but lie in a broad population. The trio's causal effect is real and small, and v364-v366 transfer the folded writer ordering—not the behavioral edit—to natural sentences.

## Highest-value WEIGHT_FOLDING action

Build the exact **regional residual6-to-head9.8 intervention-write closure**, selected by the regional circuit's strongest existing causal boundary and failed key/value composition.

- **Endpoint:** the induced head9.8 (zero-based head 8 in block 9) output write at the edited destination, first as its 128-dimensional value result and then through `c_proj[:,1024:1152]` into residual width 1152. The downstream logit reader remains a later validation, not part of the exact local identity.
- **Native factors and shapes:** `residual6 [B,T,1152]` plus token IDs generate the selected attention7/MLP7/head8.2 city write; MLP8 uses `Left,Right [4608,1152]`, product features `[B,T,4608]`, `Down [1152,4608]`, and `Down_bias [1152]`; head9.8 uses two rotary QK factors `q,k1,q2,k2 [...,128]`, scalar source scores `a=(q*k1).sum/128`, `b=(q2*k2).sum/128`, values `v [B,T,128]`, and a projected write `[B,1152]`.
- **Included source terms:** at MLP8 retain residual skip `delta`, both ordered background/edit cross terms `L(delta)⊙R(z8)` and `L(z8)⊙R(delta)`, and edit self term `L(delta)⊙R(delta)`, with native RMS and bias. At head9.8 retain every source position and all seven nonempty finite-change products: `da*b*v`, `a*db*v`, `a*b*dv`, `da*db*v`, `da*b*dv`, `a*db*dv`, `da*db*dv`. This is the full `QK1*QK2*V` interaction, not separate QK summaries.
- **Omitted terms/background:** retain the unedited native `z8`, query state, all RMS denominators, rotary phases, mixing coefficients, and baseline head fields as declared background. Omit only the unchanged baseline head write from the *delta* program, other attention9 heads, and the downstream suffix. Do not omit any live ordered MLP cross term or head-product delta term. Upstream residual sources outside the selected city intervention remain explicit background rather than silently treated as zero.
- **Nonlinearities/gauge:** native RMS with epsilon `1.1920928955078125e-7`, rounded rotary semantics, bilinear MLP product plus `Down_bias`, the two attention score products and learned mixing; no SiLU or softmax. Use native coordinates—no learned gauge or fit.
- **Simplicity measure:** compare the closed delta program with the full native residual6-through-head9.8 subgraph by independently stored FP32 scalars/bytes, supplied native arrays/scalars, executable product nodes, ordered terms, graph edges, state width, and measured CPU operations/time. Existing package counts are baselines, not presumed savings; exact closure with a larger price is not adoption.
- **Exact replay check:** on the frozen existing fixtures, require the sum of the declared MLP8 and seven head terms, after projection, to match the installed native head9.8 intervention write at every source/destination with maximum relative L2 error `<=1e-5`, zero off-support write, finite values, and separate per-term/source-position closure. This is replay evidence only.
- **Later falsifier:** freeze structurally varied, genuinely new `(template, city-pair)` cells and install the folded head write into the native suffix. Require signed held-out/OOD effect prediction, at least three unrelated readers, 16 equal-norm same-site nulls, and true-joint versus additive-without-product Mobius interaction normalized by the smallest live single and compared with matched random splits. Failure of prediction, preservation, or interaction specificity kills the proposed folded component even if exact replay passes.

This action advances targets 1, 2 and 4 immediately and creates a falsifiable route to 3, 5 and 6. It outranks another rank/reconstruction sweep because it closes a concrete native port on the designated depth path. **Concrete track connection:** the circuit evidence says the regional key and value branches are causally live but nonadditive; therefore the fold must preserve both ordered MLP cross terms and all seven head-product delta terms. Conversely, the resulting term-resolved operator will tell the next circuit hour which interaction—not a native module label—must be installed or removed in the five-arm factorial.

**CIRCUIT handoff:** on the next circuit hour, run the frozen regional full-suffix five-arm factorial on structurally varied distinct context cells with three-reader preservation, equal-norm same-site nulls, and matched-split interaction specificity; do not substitute more number-path attribution for this handoff.

## Confounds, alternatives, and registries

Audit baseline subtraction and replacement-versus-zero counterfactuals; frame/position and lexical identity mixing; RMS, MLP, full-head and suffix nonlinearity; shared token/document difficulty; Pile/FineWeb leakage; dead arms and insufficient tripwires; FP32 floors; and post-selection. A closed Mobius identity is accounting, not evidence that interactions are small relative to the smallest live piece or specific against random splits. Counts of tokens/rows are not counts of distinct context cells.

Alternatives were (2) fold the now-named MLP1-to-MLP3-to-head4.5 number route, killed as the immediate choice because its exact terms were selected on the same evolving panels and its downstream behavioral suffix/preservation battery is not frozen; and (3) head17.2 source-coordinate folding, demoted because multiple frozen preservation failures show that its context slot/source support is still unidentified. The regional residual6 route has the cleanest declared boundary, exact upstream replay, fresh interventions, and a specific open interaction.

Registry/dossier audit and missing cross-links:

- `CIRCUIT_GRAPH_REGISTRY_V1.md` still reports 49 packages/44 manifests/33 declared boundaries and omits `city_mlp8_norm_closed_v1`; its four-trait count is not the five-property certificate required by `better_circuits.md`.
- `COMPUTATION_PATH_REGISTRY.md` has `PATH-REGIONAL-CITY-001` and links the corrected residual6 fold, but still lacks the norm-closed MLP8 package and this complete head9.8 closure. It also lacks a number-path record linking v338-v367 to the pronoun-number circuit.
- `MODULE_DOSSIERS.md` contains current regional and number narrative, but `MLP_MODULE_DOSSIER_INDEX.md` still calls MLP1 unconsolidated and has no compact cross-link for v287-v367; the norm-closed regional interface is likewise absent from the relevant MLP8 navigation.
- These are navigation gaps, not absent evidence. No registry/dossier edit was made because the path registry and number dossiers are concurrently changing.

## PAST_HOUR_TIMING

Window: **2026-09-19T04:04:13Z-05:07:07Z**. Authoritative receipts show 30 successes, v338-v367, finishing 04:04:26-05:05:01. Their internal serial runtimes sum to **55.883s**. `runner.log` records **67s** of successful process wall time and **4s** in two failed attempts (v340 writer-tracking error and v367 execution failure before successful rerun). Do not sum parallel work as wall time.

Serial candidate latency was roughly 1.7-2.7 minutes between successful receipt finishes, but only the 1.6-4.7s model/process portions are directly measured per candidate. Commit timestamps corroborate publication cadence, not labor duration.

- **Design:** unknown per candidate. The overlapping lane review self-reports design/script time of about 22m for 03:27-04:19 and 7m for 04:19-04:39; these are not extrapolated to the full window.
- **Implementation:** included with scripting in those self-reports; otherwise unknown. Repeated AST-derived sibling runners and data recapture remain visible.
- **Validation:** 4s of failed runner execution measured; pre-GPU gate-refusal time and dry-run fixes are unknown.
- **Model execution:** 67s successful runner wall, 4s failed; receipt-internal successful total 55.883s.
- **Interpretation:** overlapping self-reports give about 20m for analysis/recording through 04:19 and 7m for 04:19-04:39; later time unknown.
- **Documentation/review:** review36 reports 4m; review37 reports 1m, plus a separately reported 2m external status message. Commit gaps are not used to infer the rest.
- **Idle/blocked:** self-reports say 0 idle and 9m waiting through 04:39; 04:39-05:07 is unknown outside measured processes.

`RESEARCH_PHASES_2026-09-19.jsonl` contained only the 04:04:38 review boundary before this checkpoint: it covered no science phases and cannot support uninterrupted-labor inference. The existing helper appended one truthful 05:07:07 review boundary. Design, implementation, validation, execution, interpretation, and publication boundaries for the intervening science remain unmarked.

## PROCESS_IMPROVEMENT

Ranked by likely saved time and cost:

1. A declarative shared executor plus cached per-cell activation table for the sibling number runners could remove repeated AST-derived scripts and recaptures. At the observed cadence it plausibly saves **1-2 minutes per candidate**; cost is roughly **30-60 minutes** plus equivalence testing and it collides with actively changing runner code, so it is not safe in this bounded review.
2. Add one number-path registry record and the two regional norm-closed/head9.8 cross-links. Likely handoff saving is **5-10 minutes per review** for under ten minutes of editing, but the target registry/dossiers are concurrently dirty.
3. Restore truthful phase coverage at future boundaries. Cost is seconds and it prevents later timing ambiguity. **Executed bounded repair:** appended only the current review boundary with `research_phase_clock_v1.py`; no earlier phase was inferred.

No runner refactor was made: 67 seconds of successful process time across 30 candidates shows computation is already cheap; the high-value repair is shared authoring/data machinery, whose collision and validation cost exceed this review's bounded nonconflicting budget.

## Explicit verdicts

- `TRACK_ALTERNATION: PASS` — previous `CIRCUIT`; current `WEIGHT_FOLDING`.
- `TRACK_PROGRESS: PASS` — the circuit hour produced 30 substantive fold/edit/response receipts, including causal population and trio edits plus natural-text transfer. It did not execute the selected regional factorial, so that handoff remains mandatory next circuit hour.
- `CEREMONY_BUDGET: PASS WITH TIMING CAVEAT` — the overlapping self-reports place science above review, and 30 decision-bearing receipts dominate two short reviews. Exact full-window labor buckets are incomplete because phase coverage is sparse.
- `NOVELTY_LESSON_GATE: PASS` — authorities, current commits/receipts, circuit graph, computation-path registry, module/MLP dossiers, and prior failures were checked. The new fold applies depth on the regional path, exact port closure, the full `QK1*QK2*V` product, both ordered MLP cross terms, response-before-suffix discipline, matched nulls, distinct context cells, and the rule that exact closure or lower storage alone does not establish a circuit.
