# Hourly strategic review

ACTIVE_TRACK: WEIGHT_FOLDING

Actual UTC: 2026-09-19T03:00:18Z. Next deadline: 2026-09-19T04:00:18Z.
Previous Codex review: [2026-09-19 01:55 CIRCUIT](HOURLY_STRATEGIC_REVIEW_2026-09-19_0155.md), 1h04m52s old at this review's clock. This is one current review, not a backfill. Historical `/workspace` paths were resolved against `/workspace/tensor_language`. No GPU model was loaded, no job was enqueued, and `TYPED_FACE_EXTRACTION_V1` was not modified.

The intervening circuit lane converted the open MLP1 port into a concrete mechanism: the context-free token table is nearly full-rank; attention context induces ordered token/context cancellation; units 3289 and 624 are the leading net cancellers; and replacing their contextual activations with their single-token activations raises lookup gain by `.152` versus null changes below `.00042`. Natural-text v309 retains their ranks 1/2 and `22.38%` share over 2,944 positions. This is valuable circuit work, but the user-designated depth target remains the regional path. The new folding hour should close its complete downstream head9.8 product rather than open another fitted or rank-based route.

## Seven circuit targets and full goal

1. **Computational specification:** state what is read, what operation/composition is performed, what is written, and which later computations consume it.
2. **Cross-boundary grouping and within-module splitting:** group pieces across native heads/MLPs when a reader treats them as one variable, and split a native module when its pieces have different roles.
3. **Held-out and OOD prediction:** predict activations and behavioral effects on unseen inputs, task variants, and shifted corpora.
4. **Extraction or sufficiency:** execute the circuit at a declared boundary with every retained background and native port visible.
5. **Selective manipulation:** removal, swap, or edit changes the target against matched controls while preserving unrelated behavior and accounting for redundancy/interactions.
6. **Composition and reuse:** shared pieces serve multiple tasks/modules and their joint behavior follows a tested composition law.
7. **Stable identification:** units survive document/corpus splits, gauges, and restarts, or are defined by downstream operational equivalence.

Goal: a smaller transparent tensor program that predicts held-out and OOD behavior, is extracted at a declared boundary, supports selective manipulation/removal, and composes/reuses. Simplicity is priced separately in stored scalars/bytes, native ports and state, readers, products, writers, nodes, edges, and execution. Lower reconstruction error or storage alone cannot replace any behavioral trait; quantization is excluded. This is the five-property standard: simple, predicts OOD, extracted, selective, and composes.

## Progress and five-property score

Evidence types remain distinct: v287–v304/v306/v308/v309 are chiefly **fold** or response/census evidence; v305/v307 are **edit** evidence; v297/v302/v309 use natural rows but are not fresh relative to every preceding choice; none is a fitted mechanism. The v308 dose-response equality is linear accounting; its matched random-unit nulls, not the identity, carry evidence.

| Property | Current evidence | Verdict |
|---|---|---|
| Held-out/OOD prediction | Regional packed removal passed on 20 fresh Pile documents, but FineWeb behavioral direction failed `88/102 < .90`. The exact residual6→attention8 write replays on 20 fresh FineWeb documents (`40` sequences, `240` probes; 20 distinct context documents). MLP1 self-share and cancellation transfer to 2,944 natural positions, but the unit choice was opened before v309. | **PARTIAL; regional whole-behavior cross-corpus gate failed** |
| Extraction at a declared boundary | Regional residual6+tokens→attention8 write is exact to about `2e-6`; the norm-closed MLP8 value mediator generates edited RMS9 and replays installed readouts to `6.2e-5`, but still takes native `z8`, upstream delta, token IDs, and an external suffix. MLP1 has no standalone package. | **PASS at intermediate regional boundaries; not end-to-end** |
| Selective manipulation/removal | Fresh Pile packed removal and fresh mediated correction beat same-site nulls with collateral controls. FineWeb full-city direction reversals remain; MLP1 v307 changes lookup gain by `.152` (about `380x` its null) but also changes the number margin by `-.7%`, so unrelated-behavior selectivity is not established. | **PARTIAL** |
| Composition/reuse | Regional key/value interchange fails (`interaction/smaller=2.3448`); direct/value composition fails its reversed subgroup (`.4908 > .35`). MLP1 top-unit replacement has a dose response but no cross-task joint composition test. A closed Möbius identity is not small-interaction evidence. | **FAIL for tested regional splits; MLP1 untested** |
| Measured simplicity | Packed regional attention7 saves 884,736 FP32 values (`3.8%`) locally. The exact norm-closed mediator stores 16,536,963 FP32 values and removes one scalar norm port while retaining native arrays; the exact upstream route retains all nine attention7 heads and full MLP7. MLP1 top 200 units are `4.34%` of its 4,608 product units and carry `46.37%` of natural-text net cancellation, but no full executable price or matched-effect random component exists. | **LOCAL MEASUREMENTS ONLY; no five-property simplicity result** |

Failures/corrections since the last review are informative. MLP1's fixed register direction, context-only remainder, reinjection-floor, input-identity-loss, per-head-additivity, and naive zeroing interpretations failed. v305's apparent contradiction was resolved by changing the counterfactual from zeroing to restoring the context-free activation; v307 passed that registered replacement test. Do not silently rewrite v305. The regional direct/value subgroup failure and FineWeb reversals remain unrepaired.

## Highest-value WEIGHT_FOLDING action

Build one exact **residual6→head9.8 city-response operator**, stopping at the native head9.8 output write before the suffix. This composes the corrected residual6→attention8 generator with the existing norm-closed MLP8 value response and the complete attention product; it is port closure, not a new suffix proposal (the forward response census already selected head9.8 plus later attention and rejected early/head9-only sufficiency).

- **Endpoint:** head9.8 output write `o[t,1152]` attributable to the frozen city intervention, at each registered query position.
- **Native factors and shapes:** residual/token inputs `r6[T,1152]`; attention7 and head8.2 native Q/K/V factors (`9` heads, head width `128`) producing `delta8[T,1152]`; MLP8 `L8,R8:[4608,1152]`, `D8:[1152,4608]`, bias `[1152]`; head9.8 QK1/QK2/V projections `[128,1152]`, rotary position maps, and output map `[1152,128]`; residual re-entry coefficients; FP32 RMS denominators with epsilon `1.1920928955078125e-7`.
- **MLP8 terms included:** with native background `z8` and intervention `d=delta8`, retain the normalization-rescaled `z8 x z8` baseline term, both ordered `z8 x d` and `d x z8` terms, and `d x d`; propagate their full `D8` write and bias through the edited residual/RMS. Omit none for the exact replay. Cache `Wv9.8 D8:[128,4608]` only as a derived contraction, not independent parameters.
- **Head9.8 terms included:** expand the difference of `QK1*QK2*V` into all seven nonempty delta triples: each of the three single-delta terms, all three ordered pair terms, and the triple term. Keep complete query/key/value coupling and rotary/RMS semantics. Omit only terms proven identically zero from support; do not omit terms merely small on the opened panel.
- **Background/nonlinearities:** native unedited `z8`, token-derived initial-state table, all residual coefficients, RMS8/RMS9, bilinear MLP product, multiplicative QK1×QK2×V attention, and native output projection stay explicit. Final suffix, final RMS/unembedding/softcap are outside this endpoint and remain declared background for later causal testing.
- **Simplicity measure:** count independent FP32 scalars and bytes, derived-cache bytes separately, source-term triples, product nodes, contraction FLOPs, native input arrays/scalars, and open ports. Compare with the union of the current exact-upstream and norm-closed programs at identical effect, not with a weaker supplied-state interface. A port reduction with larger storage is reported as such, not called compression.
- **Exact replay check:** on the frozen opened fixtures and a separately labelled fresh FineWeb panel, compare the folded head9.8 write against native intervention-minus-baseline in FP64 offline contraction/FP32 native execution; require per-sequence relative L2 `<=1e-5`, zero off-support write, and exact agreement between direct seven-term sum and the composed operator. Report distinct document/context cells separately from probes.
- **Later falsifier:** after freezing, install only this generated head9.8 write and recompute the full native suffix on structurally varied new-endpoint and boundary-changing contexts. Require registered effect prediction, at least three unrelated-reader preservation gates, 16 equal-norm same-site random directions, and a five-arm single/single/additive-without-product/true-joint design. Kill the proposed grouping if prediction or collateral fails, or if its Möbius interaction is not small relative to the smaller live single and matched random splits.

This action advances targets 1, 2, and 4 immediately; only the later frozen suffix test can advance targets 3, 5, and 6. Exact replay alone is not adoption.

Alternatives ranked by information/cost:

1. **Regional complete head9.8 product closure** above: highest value because it joins two exact but currently separate boundary objects and preserves the known nonadditivity. Kill on replay/port-price failure or later causal controls.
2. **Fold MLP1 units 3289/624 backward to embedding/attention0/MLP0 sources:** well motivated by v307/v309 and likely cheap, but it opens a second depth target before the regional path is complete. Resume after its module/path records are consolidated; kill if the frozen source formula does not predict unit replacement on fresh contexts.
3. **Backward unembedding fold for one number-logit contrast:** directly aligned with the ultimate output, but current regional/MLP1 readers provide better localized endpoints. Kill if native factors cannot reduce ports without importing an opaque late residual.

Concrete track connection: circuit edits v305/v307 showed that the meaningful MLP1 object is a replacement-defined net unit, not an isolated cross term; folding must therefore preserve each unit's self and both ordered cross contributions. Conversely, the regional folding failures show why the next circuit hour must test the complete head product rather than independently edited key/value singles.

**CIRCUIT handoff:** preserve the MLP1 replacement counterfactual and v309 natural-text ranks; on the next circuit hour, test fresh selective downstream behavior for the frozen top-unit set or return to the preregistered regional full-suffix factorial—do not substitute zeroing for restoration.

## Confounds, organization, and direction check

The regional path remains the highest-information route because it is the only depth target with fresh causal evidence, an exact earlier boundary, a forward response census, and a concrete remaining product closure. Audit baseline subtraction and counterfactual equivalence; punctuation/frame and position mixing; RMS, bilinear, suffix-loss, and softcap nonlinear composition; shared token/document difficulty; FineWeb/Pile and capability-conditioning leakage; dead knobs and inactive tripwires; BF16/FP32 noise floors; and post-selection. For MLP1 specifically, do not treat 2,944 positions as 2,944 independent documents, the self-share correlation as a fit-free causal law, or v309 as fresh to the selected units.

Registry/dossier audit:

- `CIRCUIT_GRAPH_REGISTRY_V1.md` still inventories 49 packages/44 manifests/33 declared boundaries and only two historical four-trait records; it is a generated inventory, not a five-property certificate.
- `PATH-REGIONAL-CITY-001` now cross-links the corrected upstream fold, regional circuit packages, failures, and causal falsifier. `CIRCUIT_REGISTRY.md`, MLP7/MLP8 records, and the MLP dossier index preserve the regional coupling and failed composition.
- Missing regional cross-link: the untracked `city_mlp8_norm_closed_v1` package is absent from the graph inventory, and `explanations/README.md` stops at the 03:26 mediator rather than the 03:42 norm-closed report.
- Missing MLP1 cross-link: v287–v309 are in the scorecard/shared-components narrative, but the computation-path registry has no MLP1 record, `CIRCUIT_REGISTRY.md` has no concise current entry, and the MLP dossier index still says MLP1 is unconsolidated. These are documentation gaps, not evidence gaps.
- No cross-link was edited: the MLP1 lane was actively writing v309 while this review ran, and the regional package/result tree is concurrent untracked primary work. Touching either would violate ownership/preservation for a low-risk navigation repair.

## PAST_HOUR_TIMING

Window: 2026-09-19T02:00:18Z–03:00:18Z. Only authoritative runlogs, receipts, the lane's review, process times, and Git/file timestamps are used.

- Receipts v287–v309 finish from 02:07:06 through 02:58:46. Twenty-three successful scientific receipts plus one preserved v298 void run report `44.81s` total serial model time, `314` forwards, and `1.79s` median receipt runtime. These durations must not be summed with parallel work as wall time; this lane was serial.
- Runner start/exit pairs are generally 2–3 seconds. v291 records two pre-success implementation/instrument failures (wrong hook name and signed averaging); v298 preserves a void hook-layout run. Exact wasted execution is only a few seconds, while the lane review attributes about four minutes to the v291 repair.
- Review 34 self-reports, for its broader 01:32–02:26 bucket, orientation 4m, scripts/preregistration 22m, waiting 3m, analysis/recording 18m, and review 7m, plus idle until the ~02:00 user redirect. Those categories sum to 54m while the same note also mentions 28m earlier idle, so they cannot be treated as a complete non-overlapping wall-time partition. In this review's window, about five minutes of the declared idle interval overlap (02:00 redirect time is approximate); exact overlap is **unknown**.
- Serial candidate latency from prior-art check through scored dossier is **unknown per candidate**: only model start/end and result/commit times are authoritative, and commit gaps are not active-work measures. Aggregate scientific design is **unknown**; implementation/preregistration is self-reported as 22m for v288–v294 in the broader bucket; validation is **unknown** apart from the explicit failed-run notes; model execution is `44.81s` across the observed receipt set; interpretation/recording is self-reported as 18m and documentation/review as 7m through 02:26; later category splits are **unknown**; idle/blocked overlap is approximately 5m but not precisely timestamped.
- The newest matching phase log for the actual work date remains `RESEARCH_PHASES_2026-09-18.jsonl`, with nine sparse marks ending at 03:00:38 on that date. There is no 2026-09-19 phase file. Coverage is stale for this hour; sparse boundary absence cannot be interpreted as uninterrupted labor or inactivity.

## PROCESS_IMPROVEMENT

Ranked by likely saving versus cost:

1. Replace the sequence of bespoke MLP1 runner files with one declarative source-term/counterfactual executor and shared receipt writer. The lane's own audit attributes 22m to seven scripts/preregistrations before v294; a shared spec could plausibly save 1–2m per follow-up. Cost is likely 30–60m plus equivalence tests, too large for this bounded review and risky during active v309+ work.
2. Add a canonical MLP1 module dossier/path record after the active lane freezes: likely saves 5–10m of repeated novelty search per future handoff; cost under 10m, but editing now would collide with concurrent ownership.
3. Resume sparse phase marks with the existing helper: improves future timing audits but does not save scientific runtime and cannot reconstruct this hour retrospectively.

No CPU repair was justified. Existing helpers already handle self-share, enqueue/wait, derivation, and lint; a new helper would be ceremony, while the two useful documentation repairs overlap active uncommitted work. The review itself creates only this receipt and the append-only board link.

## Explicit verdicts

- `TRACK_ALTERNATION: PASS` — the previous Codex review declared `CIRCUIT`; this review declares `WEIGHT_FOLDING`.
- `TRACK_PROGRESS: PASS` — the CIRCUIT interval produced causal and natural-text receipts, most importantly the v307 replacement edit and v309 natural-text census, while preserving failed hypotheses and the v305 correction.
- `CEREMONY_BUDGET: PASS WITH TIMING CAVEAT` — the lane self-reports 40m of scripts/analysis versus 7m review and 3m waiting through 02:26; exact validation and later buckets are unknown, and its broader-bucket arithmetic is not a valid wall-time partition. Scientific receipts, not review ceremony, dominated observable output.
- `NOVELTY_LESSON_GATE: PASS` — the circuit/path/graph registries, scorecard, MLP index, regional MLP7/8 records, current authorities, and failures were checked. The plan preserves full QK1×QK2×V, ordered MLP cross terms, native normalizers, the forward-response result, and matched nulls; it does not reopen rank/reconstruction or arbitrary source partitions.
