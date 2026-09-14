# Module dossiers

These records collect stable facts about native components independently of any one behavior
circuit. They complement the task-defined records in `DOSSIER.md`. A native module boundary is an
index for retrieving evidence, not an assumption that the module is one semantic unit.

## `module.attention.5`

Aliases: attention block 5, `attn5`, L5 attention, induction gate, copy gate, content gatherer,
pooler. Related but narrower object: head 5.7, also called the sink or constant-write head.

### Established before 2026-09-04

- Sections 877 and 882: removing attention block 5 removes nearly all measured induction; the
  front circuit was summarized as attention 0 supplying the key and attention 5 performing copy.
- Sections 998, 1006, and 1007: the content-gathering effect is concentrated in layers 3–5 and is
  dominated at head level by layer-5 head 7, with a jointly important distributed remainder.
- Sections 1039, 1043, 1044, and 1047: attention 5 operates on the value residual and belongs to
  the broad/local-residual routing part of the attention map; simple token and embedding-bag
  stand-ins do not reproduce it.
- Legacy registry entry `Sink (5.7)`: head 5.7 can be replaced by one constant vector without the
  measured loss cost of deletion.

### 2026-09-04 extension

The whole block, not only head 5.7, has a nearly one-dimensional output geometry on the measured
corpora: 98.1% of write energy lies in one direction; independently fitted natural-text directions
have absolute cosine 1.000; the code direction has cosine 0.997; and a fixed-vector replacement
recovers about 95% of the whole-block value. This is a held-out geometric refinement of an existing
module account, not a new functional localization.

### Still unknown

- which input-dependent computation produces the nearly fixed whole-block write;
- how the induction/copy and content-routing functions split below the native head basis;
- whether one intervention-defined subspace can selectively change either function while preserving
  the other; and
- which downstream computations treat parts of this write as the same variable.

Any future attention-5 experiment must cite this dossier and state which unknown it resolves.

## Saved correlative interface: 26 heads across layers 3–16

10 September update: [weight pullback and input-reader overlap](../../polynomial_causal/explanations/2026-09-10/attention_ov_input_reader_overlap.md). The fixed block projector and complement use `O_P,h=(Oq)q_h^T` and `O_R,h=O_h-O_P,h`, respectively. Their within-head QK1/QK2 routing is identical by construction. At relative rank tolerance1e-6, 19 per-head remainder writers retain all128 coordinates and can read the saved scalar’s full value-input function. Seven single-head blocks retain127 coordinates with partial input overlap. Full-block cross-head cancellations prevent interpreting these per-head overlaps as globally duplicated computation.

The earlier donor double dissociation supports a swap interface, while selective mean removal and endpoint-addition tests failed. Neither the broad complement nor an OV row-space overlap is a named semantic circuit. Check specialist-heads.md, attn-middle-pooling.md, channels.md and the linked native/CPU receipts before claiming new per-head functionality.

### 15:17 UTC: QK1/QK2/value dependencies

[The complete factor lattice](../../polynomial_causal/explanations/2026-09-10/attention_qk1_qk2_value_dependencies.md) extends the earlier combined-routing test to both branches. Full swaps and prior P route/value cells replay. Opposite global stored-score-half dependence failed: QK1-minus-QK2 conditional recovery losses .00637/.00118/.01895 on P-A1/P-A2/R-C, below.20 and not opposite. Conditional value losses .81049/.78021/.89233 pass. Current values contain earlier routing computation, so this does not establish routing irrelevance.

R-C QK1+value has9.51% full-vocabulary error versus19.47% for QK2+value, a descriptive candidate only; no independent promotion or threshold rescue. Higher-order endpoint interaction is .3956/.3343/.3638. Stored QK labels are independently exchangeable per head. The broad complement remains unexplained and original selective-removal failure unchanged. Canonical subroutine.correlative.score_half_task_split is rejected with valid instrument/value-dependence evidence retained.
## Gerund readout scalar, 10 September 2026

The [MLP17 dossier](../../polynomial_causal/explanations/MLP17_CURRENT_UNDERSTANDING.md) now includes `subroutine.readout.gerund_shared_scalar`: a direction from eight unembedding bare/-ing contrasts, with sixteen distinct test verbs. Final-state scalar cue transfer .586/.621 and cross-verb transfer .584/.620 are partial and fail the registered .80 sufficiency bar. Zero removal target CE+.565/+.686 versus unrelated C meanabsCE .0232 passes the narrow registered removal test. MLP-only scalar contributes .071/.100 recovery; most scalar change arrives from earlier state. No independently extracted producer or broad OOD/composition evidence. See [the computation and controls](../../polynomial_causal/explanations/2026-09-10/shared_gerund_component_from_unembedding.md); do not repeat it as discovery of generic quadratic folding or promote it to a sufficient grammar circuit.

The [distributed output-port follow-up](../../polynomial_causal/explanations/2026-09-10/gerund_scalar_writes_and_live_feedback.md) at16:06 held joint-swap sufficiency (.936/1.017) and narrow zero-removal selectivity (C absCE.0433), while rejecting scalar-only edit prediction (full-vocabulary errors .654/.670). Both claims are separately recorded under `subroutine.readout.gerund_scalar_write_network`. All36 output e-coordinates prescribed to donor guarantee the final scalar, but native complementary state changes materially. Weighted native MLP contributions84–87% are an attribution identity, not an independent producer or individual-module circuit. All native weights remain.


### Fresh grammatical transfer and agreement failure, 10 September 16:31 UTC

The fixed all36-port interface transfers to new16 verbs and might/were or would/was cues: recovery .95175/.86341, paired intervals [.92557,.97813]/[.83145,.89820]. All96 native pairs capable; old R replay and instrument pass. Broader preservation/removal fail on closer he/they runs/run agreement G: swap recovery .13342 and meanabsCE .10793, zero meanabsCE .54788 (CI .51396–.58346), versus .10 limits. G is one agreement contrast over16 contexts, not16 agreement readouts; it remains a failed control. Original C-only selectivity and failed scalar prediction are preserved. No independent producer, single-module promotion, pretraining-OOD or saving. The updated gerund_scalar_writes_and_live_feedback.md explains both individual-token and shared-structure-plus-remainder folds. Canonical distributed-port revision records positive fresh transfer and failed broader selectivity separately.


### Grammatical scalar consumers, 10 September 16:42 UTC

Fixed all18 post-RMS MLP e reads are held at natural base values before the original all36 output e edits. Exact weight-folded cross/square correction, no refit. Target swap recovery falls .95175->.66036 and .86341->.57286: losses .29139/.29055 with paired intervals [.25625,.32386]/[.26203,.31871]. Base-only G zero CE falls .38682->.11766 (69.6% reduction), not directly comparable with the previous two-endpoint .54788. Native scalar prediction errors .64876/.64467 become .15287/.08449 on the altered blocked computation; joint <=.10 closure still fails. A/B/C/E held D failed. All64 native pairs capable, exact replay, folded relative bridge <=1.60e-6,28forwards448seq in1.7175seconds. Folded context readers k_v=2Q(v)e for e and runs/run have median cosine .2053 across layers; descriptive immediate-output maps, no site selection or semantic separation proof. Original v184/v185 algebra reused; new causal consumer evidence under subroutine.readout.gerund_mlp_scalar_consumers. No independent context producer, OOD or selective repair. See gerund_scalar_writes_and_live_feedback.md and GERUND_MLP_CONSUMER_V1_RESULT.json.


### Selected-reader response state, mathematical review16:49

The exact MLP17 finite-difference contraction compiles two context rows K and squared coefficients a for e and runs/run. With q=Ku, response=delta*q+delta^2*a and q_next=q+2delta*a. Trained FP64 selected-response error1.01e-13 and sequential-composition error1.59e-13; saved2306-coefficient program reload also passes without native checkpoint. This is local conditional response extraction, not absolute outputs or a text-to-answer producer. All16 synthetic equal-e/equal-K/equal-norm reflection pairs still have different full-output responses; any common prediction has minimum relative error .1087–.2416. No natural-text reachability/OOD claim. Canonical gerund_selected_response is specified for selected local operation, rejected for full-output closure; zero native forwards. Review THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_1649.md proves the kernel criterion and state update, compares exact reduction methods, and keeps initial contextual producers/downstream readers as the missing work. LATEST16:54 and existing gerund explanation carry this scoped result.


### Native context transfer rejects the two-reader replacement, 10 September17:04

The frozen local response program is valid on native MLP17 scalar-input edits: selected-reader errors4.57e-7/3.74e-7 and all64 endpoint pairs capable. However reference-context errors .26068/.24985 and cyclic next-verb errors .19779/.08618 fail their joint.10 criteria. Even the exact-context minimum-norm two-reader output writer leaves .99009/1.00935 centered full-vocabulary effect error and CE prediction MAE .77457/.46752. Native local CE damage+.71521/+.41998, cue recovery .09755/.07401. Managed13forwards193seq,1.33336sec; A instrument held B/C/D failed. Paired exact-writer errorCIs .98670–.99386/1.00793–1.01076. Canonical gerund_native_response_gate rejected; original local selected equation/composition remains valid but insufficient for behavioral replacement. No new gate/rank/donor choice. Next distinguish omitted token-specific outputs and RMS effects with existing calibration/two-reader dossiers as precedent. See NATIVE_RESPONSE_GATE_V1_RESULT.json and updated gerund_scalar_writes_and_live_feedback.md.


### Grammatical token numerators and RMS,10 September17:17

Existing calibration two-consumer algebra reused on the current grammatical MLP17 scalar-input response. Neither norm-only (.93533/.97952 error), numerator-only (.47626/.58483), nor old two-reader numerator with true norm (.87871/.91634) meets the joint.10 full-effect bound. Missing normalization alone therefore does not repair omitted token readers. Endpoint interaction .04486/.06194 is descriptive, not independent-circuit composition. A conditional program folds actual answer/foil numerators and shared quartic RMS into11 context coefficients; max native tokenerror8.03e-6, margin8.73e-6, sequential composition5.33e-15. Initial h/u-derived coefficients and full-vocabulary probabilities remain native-dependent. A/E held B/C/D failed. V1 publication st_size() bug preserved; V2 same science12forwards192seq in1.37302sec, V1/V2 saved states exactly equal. Paired/saved-state CPU audit actually performed with no model load; cache3,010,885bytes retained for future CPU reuse. Canonical gerund_token_norm_program specified conditional token scores, rejected isolated simplifications. See updated gerund_scalar_writes_and_live_feedback.md. Next identify upstream producers of required token/context readers and norm moments; no fitted output-rank rescue.


### Token-context source census, 10 September 17:37 UTC

Actual answer-minus-foil reader v is folded to k_perp=2Q17(v)e-2(e^T Q17(v)e)e.
Tau=k_perp·u17 supplies the context part of the local scalar-input response.
Same-frame cyclic-verb donors with the recipient reader fixed tested all36
single attention/MLP output sites. A/C held, B failed: no shared-source nominee,
maximum target transfers .3591/.3701, minimum errors .6566/.7107. All48 base
endpoints correct; all-chain/no-op/MLP17-output causal-zero controls exactly0.
117forwards1872seq,3.51036sec. No single module promoted. CPU paired intervals
for all sites and saved mean-plus-token-remainder accounting actually executed;
mean-only target gate errors .64924/.90508 on already opened rows. No independent
producer, new OOD or structural saving. Both user-directed unembedding paths
remain explicit; cluster/hierarchy remainder cannot be silently dropped.
Primary receipts: TOKEN_CONTEXT_SOURCE_V1_RESULT.json and
TOKEN_CONTEXT_SOURCE_AUDIT_V1_RESULT.json in polynomial_causal; explanation
2026-09-10/gerund_scalar_writes_and_live_feedback.md. Do not repeat singleton
whole-module localization as a new circuit discovery or select a head subset
from this null without a different registered hypothesis and dossier lookup.


### Shared first-value lexical/context test, 10 September 17:48 UTC

Known token-only first-value producer (channels/v173/v174) was tested as a
cross-layer source for current actual-token MLP17 context gates. Cyclic primed
verb at position3; ordinary token stream versus attention0-returned value cache
factorial, preserving attention0 own output. All later native consumers live.
A instrument held, B context source/C joint lexical selectivity failed.
Value-only lexical recovery .84174/.80855 with all32pairs both-endpoint capable,
but gate transfer .48551/.60670/error .55717/.56737; remaining-stream gate
magnitude .53379/.59949. G absCE .70470 (CI .40194–1.07834) fails preservation.
Seven harmed/nine improved; signedCE -.08536 does not rescue the absCE failure.
Edited foil logits were not saved; do not attribute this CE change specifically
to agreement-margin damage. CPU synthetic equal-margin/different-CE witness
and paired audit executed. Full-vocabulary interaction .36498/.32789.
Token-to-V0/noop/saved-u checks exact0;15forwards240seq,1.42524seconds. No smaller
consumer program, OOD, independent extraction or selective circuit promotion.
Primary receipts TOKEN_CONTEXT_BROADCAST_V1_RESULT.json,
TOKEN_CONTEXT_BROADCAST_AUDIT_V1_RESULT.json and
TOKEN_CONTEXT_BROADCAST_CONTROL_CE_V1.json in polynomial_causal; updated dated
gerund_scalar_writes_and_live_feedback.md. Next lexical/form readout separation,
not a head-index slice or rank rescue of the failed whole-source test.


### Lexical/form command reuse and coupling, 10 September18:03 UTC

Fixed lexical command: first-value position3 cyclic donor. Fixed form command:
all36 output e scalars from original-lemma ing cue, reused without adapters for
the cyclic lemma. Native2lemma x2cue grid, both singles in both contexts andjoint.
A instrument held; B/C/D/E failed. A1native16/16everycorner; A2Y/Z15/16, shop
and laugh contexts prefer traveling in their respective four-token comparisons.
L recovery .84174/.84124 and .80855/.78531; form drift .0462–.0976. F recovery
.95175/.95041 and .86341/.86437; lexical drift .1149–.1688 fails. Joint correct
14/16,13/16; four-token error .17660/.19757; full additive error .12298/.08285.
G agreement-margin meanabs .23328(CI .16004–.30963)fails .10 and old CE .70470
replays. This resolves the previous missing-foil limitation; no threshold rescue.
Exact four-reader Hadamard decomposition and MLP17 product fold held1.16e-14.
Actual CPU score-coordinate accounting: interaction contributes58–81% of F's
squared lexical drift, descriptive not causal mediation. Selected-score addition
error .04706/.04069 versus ACTUAL joint command, not native joint target; saved
state addition+exact norm/softcap .04604/.06495 does not fix joint choice failure.
23forwards368seq,1.55786sec. Saved7,476,061bytes reader/product coefficients and
finalstates for CPU reuse; no independent producers or structural saving.
Primary LEXICAL_FORM_INTERCHANGE_V1_RESULT.json, its AUDIT_V1 and
LEXICAL_FORM_DRIFT_COMPONENTS_V1_RESULT.json in polynomial_causal; current dated
gerund_scalar_writes_and_live_feedback.md. Next investigate explicit coupled
lexical/form operations and their producers, not independent-axis relabeling.


### 18:13 coupled-readout null

`LEXICAL_FORM_READOUT_FACTORS_V1_RESULT.json` rejects physical scalar-only, scalar+actual-norm and complement+actual-norm explanations of F lexical drift (all .20 bars fail). Full saved-state readout bridges hold. Required complementary changes must be explained upstream; this does not reopen generic MLP17 rank approximations. Details and exact cancellation limits are in the current dated `gerund_scalar_writes_and_live_feedback.md`.


### MLP17 terminal complementary sources, 18:50 UTC
V2 instrument held; MLP-only complement (.710–.833 error) and carried-only (.342–.373) both fail .20 lexical-drift sufficiency. Actual last-MLP complement clamp matches the physical readout counterfactual. V1 precision-check failure preserved, shared V1/V2 states bitwise equal. See polynomial_causal/explanations/2026-09-10/terminal_complement_sources.md and TERMINAL_COMPLEMENT_SOURCE_V2_RESULT.json. No independent producer or circuit promotion.

## Head9.8 conditional even/odd routing and selective scalar — 14 September, 12:17 UTC

The full-value even-key extension retains both QK factors, 128 current/inherited
value channels, native mixture and output map. Native instrument replay passes,
but regional selectivity and unrelated preservation fail. Its cue coverage
41.84–44.51% is close to the earlier scalar, with substantially larger collateral.
Physical removal of the complement R=even−scalar changes paired cue contrast by
only 3.91–6.36% of the scalar change, while causing 4.23–7.09 times its unrelated
readout effect. Regional composition passes; one newline half fails at 5.458%.
The executed CE-stage discriminator leaves substantial raw-score/final-RMS
nonlinearity, so this is not dismissed as loss curvature.

A weight-derived shared S/R/O graph reuses native V/output coordinates and passes
36 signed-write controls. All evidence remains conditional on native contexts and
suffix and uses development panels; no newly named semantic circuit or full-model
compression is established. The next registered S+O/R+O interventions resolve the
remaining pair/triple ambiguity. [Primary method and receipts](../../polynomial_causal/EVEN_KEY_VALUE_BANK_V1_MATH.md).


### 2026-09-14 — head9.8 S/R/O composed with MLP9
The [primary record](../../polynomial_causal/EVEN_KEY_VALUE_BANK_V1_MATH.md)
now includes complete native removal factorials (312 additional forwards), exact
weight-derived rational MLP9 composition (110 native comparisons), and charged
CPU query amortization. Pair-only full-removal error is <=0.276% on the old panel.
The selected S/O/SO cue reduction has <=4.38% paired-cue error across all removal
corners but loses most unrelated effects; full-value selectivity remains failed.
Conditional execution/extraction and composition advance; fresh/OOD cue reduction,
semantic identification of O and static model compression remain unproven.


Fresh update: [72 new-construction rows](../../polynomial_causal/SRO_FRESH_CUE_V1_RESULT.json)
falsify the S/O/SO reduction at its 10% gate (8.20/31.96/10.43%); the middle
family also misses native capability. Post-failure accounting attributes the
main omission to direct R, while pair-only composition remains within 0.455%.
R cannot be classified as generally unrelated background. This failure supersedes
any generalization from the earlier development-panel reduction.


### 2026-09-14 — remainder source and confirmation corrections
[Source removal](../../polynomial_causal/REMAINDER_SOURCES_NATIVE_V1_RESULT.json)
finds current-only cue errors22.18/18.88/9.82%; inherited values are not uniformly
negligible. Source-effect composition passes <=4.49%. Family2 of both fresh and
four-term panels had an indefinite-article/city confound; retain their original
results but supersede clean-template interpretations with the
[uniform correction](../../polynomial_causal/SRO_ARTICLE_CORRECTION_V1_RESULT.json).
Corrected four-term cue errors4.20/2.54% pass; three-term10.049/8.231% fails its
unchanged10%bar. Native capability passes corrected families, but the unaffected
opposed publisher/author family has only1/12 positive native contrasts. No clean
writer-role claim, general control preservation or static parameter saving.


### 2026-09-14 — current-remainder routing/value interaction retained
The [primary record](../../polynomial_causal/EVEN_KEY_VALUE_BANK_V1_MATH.md)
now rejects city-only inherited-value sufficiency (90–113%cue error) and demotes
the small inherited branch. Full current remainder interchange is consequential
(~11%full-head cue norm); value-only sufficiency fails, and dropping its mixed
routing/value term causes up to48.65%control error. Three-term native composition
passes<=3.47%. Exact coupled-strength MLP9 compilation merges equal monomials,
retains the mixed interaction, and saves7.11%prepared-context payload across100
CPUcases. Native prefix/weights and independent-strength restrictions remain
explicit; no new semantic-role identification or static whole-model saving.


### 2026-09-14 — coupled MLP9 to complete attention10
[Combined native V2](../../polynomial_causal/COUPLED_ATTENTION10_NATIVE_V2_RESULT.json)
passes100coupled signed cases: post10 error<=1.53e-7, full scores<=6.73e-7,
registered own-effect floors hold. Runtime programs include the source residual
path and attention10 projected polynomial ports; measured payload saving is
36.41–38.71%for18–19token contexts. Native model/context generators remain external
preparation dependencies. V1 shape failure and layout-only recovery are preserved.
No static whole-model reduction or additional semantic role is claimed.


### 2026-09-14 — longer contexts and shared-weight accounting
The [primary interaction record](../../polynomial_causal/EVEN_KEY_VALUE_BANK_V1_MATH.md)
adds200native short/long cases for the exact12-vector consumer: state/score and
own-effect bars pass, but long-context storage fails (1.469–1.624x).
Across eight contexts, shared native weights require88.4MB versus304.6MB for
projected programs. This rejects shared-service weight compression from these
context exports. Prior combined prices omitted8bytes of re-entry scalars;
corrected prices preserve every verdict. No new semantic circuit is identified.

The [shared-bank extraction](../../polynomial_causal/COUPLED_SHARED_EXECUTOR_V1_RESULT.json)
then passes64 isolated CPU cases across18- and84-token contexts, including seven
off-grid strength pairs. One31.86MB native bank plus caller contexts replaces
duplicate projected consumers (45.04MB package versus75.01MB projected pair).
Context-program generation and the native suffix are still external, so this is
conditional extraction/reuse rather than a complete or semantically identified
circuit.

The [native shared-interface validation](../../polynomial_causal/COUPLED_SHARED_EXECUTOR_NATIVE_V1_RESULT.json)
passes128 native-versus-exported comparisons on eight contexts and seven off-grid
coupled interventions plus zero. Full-score error is at most7.88e-7 and every
effect/fidelity/price gate passes. This closes conditional suffix fidelity and
reuse for the tested interface. All545.9M native parameters still generate the
contexts and suffix; no unseen-text, semantic-role or whole-model claim follows.

[Odd-branch source localization](../../polynomial_causal/ODD_SOURCE_POSITIONS_NATIVE_V1_RESULT.json)
finds a material/selective O cue effect (13.10–15.86% of full-head cue norm),
but direct changed-city sources recover only3.5–8.0% of its effect: city-only
relative errors are92.02/96.46%. Exact source recomposition passes. The branch
therefore carries cue information mainly through contextual source states on
this panel; its specific source positions and semantic operation remain open.

[The follow-up](../../polynomial_causal/ODD_CONTEXTUAL_POSITIONS_NATIVE_V1_RESULT.json)
localizes that effect to post-city nonfinal sources: cue errors versus all-O are
8.73/3.18%, compared with99.92/100.58% for the final self-source and >100% for
pre-city sources. Selective controls pass. This supports a contextual relay at
head9.8 on the corrected panel, with instruction versus quoted-clause carriers
still unresolved.

[Framing versus clause removal](../../polynomial_causal/ODD_SEMANTIC_POSITIONS_NATIVE_V1_RESULT.json)
passes the framing prediction: framing errors are30.68/8.09% versus78.65/95.85%
for copied-clause tokens, with selective control ratios. O therefore reads the
regional cue after it has propagated into post-city prompt framing on these two
templates. Fresh templates and finer within-framing localization remain required
before identification.

[Fresh authored templates](../../polynomial_causal/ODD_FRAMING_FRESH_V1_RESULT.json)
confirm native capability, O materiality and framing-over-clause localization,
but only one of two templates passes the work/jobs selectivity control (.863
versus.298). The computational read transfers; robust selective identification
does not yet. Control-family sensitivity is being tested explicitly.

The [frozen multi-control diagnostic](../../polynomial_causal/ODD_FRAMING_CONTROL_FAMILIES_V1_RESULT.json)
passes all four unrelated control pairs in both templates. Median control/target
ratios are.226 and.101; the maximum is.374. The prior work/jobs failure remains
part of the record and appears readout-family sensitive rather than broad
collateral. Within-framing source roles are still unresolved.

The [within-framing role test](../../polynomial_causal/ODD_FRAMING_ROLE_SPLIT_NATIVE_V2_RESULT.json)
finds no template-invariant role. Description sources dominate local-history
(31.89% framing-effect error) but instruction sources dominate radio (28.72%).
The opposite roles miss by70.04% and71.89%. Exact anchors and source partition
pass in corrected V2; V1's mislabeled all-O/full-head arm is retained as an
instrument failure. Source count or position remains a live confound.

Equal-count early/late removal does not resolve that confound: local-history is
early dominated, while radio has a distributed effect with the late half closer
but insufficient. Paired source-state interchange nevertheless transports the
target direction selectively (cosine.989/.999; all new-control ratios<.1), while
its twice-removal magnitude law fails local-history at56.08% and passes radio at
12.00%. Routing and value change together, so their interaction remains open.

The [exact interaction split](../../polynomial_causal/ODD_SOURCE_SWAP_INTERACTION_NATIVE_V2_RESULT.json)
finds value-only interchange within2.04% of full swap in both templates, while
routing-only error exceeds102%. The routing×value term is algebraically live but
behaviorally omissible here (<=2.29% target error). This identifies value
transport at the all-query-fixed head9.8 boundary; current versus inherited
value provenance remains open.

An isolated layer-onset trace keeps block9 queries, keys and residual path fixed.
Layers0--6 explain <=6.56% of the transported current-value effect, layer7
9.74--19.42%, layer8 13.42--27.03%, and direct layer9 input interchange gives
100%. Both frozen onset hypotheses fail. The unresolved writer lies mainly in
block8's transformation into the block9 residual, motivating an attention/MLP/
carry split rather than another semantic source mask.

Block8 carry/attention/MLP hybrids compose the isolated effect, but the writer
class is template dependent: attention suffices for radio (14.62% error) and no
single class suffices for local-history (best attention at53.33%). Registered A
remains failed because separately rounded FP32 writes miss its `1e-10` residual
identity bar at9.67e-7. Attention8 head writes are the next screen.

[Attention8 head decomposition](../../polynomial_causal/ODD_ATTENTION8_HEAD_WRITERS_V1_RESULT.json)
finds one stable upstream writer: head8.2 reproduces full attention8 transport
within2.78%/.91%; all other heads have approximately100% error. Head effects
compose and controls remain small. This supports a cross-layer head8.2 write to
head9.8-O current-value read. A subsequent
[source-edge intervention](../../polynomial_causal/ODD_ATTENTION8H2_SOURCE_EDGE_V1_RESULT.json)
finds that changed-city source alone reproduces the head8.2 effect within
3.04%/2.71%, while all other sources miss by97.08%/102.63%. Effects compose and
all four unrelated controls remain below.087. The source arms share paired
donor framing queries, so this supports a donor-query-conditioned city source
-> head8.2 framing write -> head9.8-O current-value edge. The
registered exact-partition gate remains failed at5.33e-8 under FP32 arithmetic;
the separate numerical control only classifies it within native tolerance.
With recipient queries fixed, the city key/value factorial remains strongly
value dominated: value-only cue errors8.12%/2.49%, versus routing-only errors
110.52%/104.84%. The fixed-query cue closely matches the donor-query cue in
paired direction, but its exact score anchor fails at.01794; this preserves the
query condition on the earlier claim. The routing-value interaction is needed
for one template under the registered10% bar.

The city-source value itself is inherited-first dominated. With recipient
routing fixed, inherited-only errors versus the full value effect are
15.07%/16.44%; current-only errors are85.89%/84.03%. Both source deltas are live
and separate behavioral effects compose within.514%. The supported chain is a
static changed-city token value read by head8.2, a contextual framing write,
and a head9.8-O current-value read. This is still conditional on two authored
templates and the native prefix/suffix.

The frozen inherited-city chain transfers to two new templates, two new city
pairs and six unseen spelling endpoints. Value-only predicts the full city cue
within2.54%/5.53%, and inherited-only predicts the full cross-layer cue within
20.65%/23.08%; routing and current alternatives miss by75--107%. Native
capability passes and all unrelated controls are below.087. This adds held-out
authored-task prediction, while corpus OOD and independent prefix/suffix
extraction remain open.

Destination semantics do not define a stable submodule across the four
constructions. Description/instruction errors versus the full inherited write
are56.96/43.81%,79.89/20.56%,69.70/30.69%, and55.88/44.61%. Instruction alone
suffices in two families; the other two are distributed. Effects compose within
.70% and controls remain below.088. The invariant is instead the exact
destination scalar times a shared projected inherited-city writer direction.

That invariant now has a standalone conditional executor. Native factor-first
replay is within1.73e-7 and block0 token-value reconstruction within2.64e-7 on
all96 rows; four standalone template representatives replay within2.92e-7.
The294,913-scalar package saves94.91--95.75% of the conditional dense-write
interface and94.22--95.05% of projection/scale multiplies at observed lengths.
Head8.2 routing, block8 state generation, head9.8 O and suffix remain external.

An extended884,737-scalar executor closes head8.2 city routing from normalized
block8 state. Routing/write replay is1.62e-7/2.31e-7 over all96 rows and the
standalone control is<=5.49e-7. If that state port is dedicated to this edge,
the closed input is8.3--10.0% larger than the dense write; the large output
saving applies only when block8 state is shared. Head9.8 O and suffix remain
external.

The head8.2 program is now composed through block9 reentry/RMS and the exact
head9.8-O current-value graph. Target scores replay within3.34e-6, controls
within2.39e-6, and the standalone four-template delta control within6.20e-7.
The1,933,572-scalar package remains conditional on native state ports and suffix;
dedicated ports cost about4.09 dense deltas. Donor interchange attenuates all48
paired regional contrasts, supplying a concrete direction for a midpoint edit.

That midpoint intervention now passes on all four frozen families. Its native
behavioral effect is within.17--.38% of half the full donor effect and attenuates
all12/12 paired regional contrasts per family. Mean attenuation is2.08--4.64%
of the native cue, while all unrelated-control ratios stay below.088. The edge
therefore supports selective, graded manipulation in addition to held-out
prediction, causal transfer, extraction, and cross-boundary composition. The
claim remains conditional on native state generators and suffix; corpus OOD,
whole-head removal, and favorable dedicated serialization remain untested or
failed as stated above.

Current-value provenance is exact on the paired framing sources: current-only
and full-value swaps are behaviorally identical, while inherited-first swap is
zero. The registered both-arms-live gate fails and remains visible. Architecture
explains the zero because block0 `v1` is a tokenwise projection and paired
framing tokens match. The cue therefore reaches O through contextual current
states; its upstream layer onset is not yet localized.

Equal-count early/late removal does not resolve that confound: local-history is
early dominated, while radio has a distributed effect with the late half closer
but insufficient. Stable semantic/position identification is therefore rejected
on this panel. Paired source-state interchange is the next extraction test.
