# Module dossiers

These records collect stable facts about native components independently of any one behavior
circuit. They complement the task-defined records in `DOSSIER.md`. A native module boundary is an
index for retrieving evidence, not an assumption that the module is one semantic unit.
Folded cross-module and within-module routes are indexed in
[`../COMPUTATION_PATH_REGISTRY.md`](../COMPUTATION_PATH_REGISTRY.md); path dossiers
should link back to the relevant module sections here.

## `module.attention.11.head3`

Aliases: L11H3, layer-11 head 3, subject-number write head. Related path:
[`PATH-SUBJECT-001`](../COMPUTATION_PATH_REGISTRY.md).

### 2026-09-15 subject-number write factorization

- A frozen four-scalar direction-by-cardinality coefficient law reproduces the prior
  rank-one causal effects on 512 fresh interventions with cosine `.999816`, relative
  L2 `.023954`, and sign agreement `1.0`.
- Replacing the fitted 1,152-dimensional write axis with the top left singular vector
  of this head's native output-projection slice preserves the held-law causal effects
  with cosine `.999447`, relative L2 `.079550`, and sign agreement `1.0`.
- An outcome-blind affine readout from the first two native output-weight singular
  coordinates fails cross-construction amplitude prediction at relative L2 `.58757`;
  widening through rank eight does not repair it.

Primary receipts: [coefficient law](fast_screens/subject_number_coefficient_bilinear_law_v1_result.json),
[native axis](fast_screens/subject_number_native_weight_axis_v1_result.json), and
[native scalar-readout null](fast_screens/subject_number_native_scalar_feature_discovery_v1_result.json).
These establish a reusable causal output axis and compact coefficient law. They do
not identify how earlier state computes direction/cardinality. Adding the exact
MLP8-input E/A/U/W secant coordinate $g$ and its interaction with the native axis
coordinate $z$ is also a valid null: leave-one-construction-out relative L2 moves
from `.616996` for `[1,z]` to `.553126` for `[1,z,g,zg]`, only `.063871`
improvement versus the frozen `.10` gate and still above the `.40` ceiling. The
quadratic-context extension reaches only `.549892`. See the
[upstream-coordinate null](fast_screens/subject_number_upstream_context_coordinate_discovery_v1_result.json).
Future discovery should change coordinate family rather than widen either readout.

The next frozen assay changes the coordinate family to the head's own response to
an upstream MLP6/7 number-source switch. For background $b$, it measures
$s_b=u^\top(H(x_{b,YZ})-H(x_b))$ along the same native output axis and compares
cross-construction scalar programs. Response-only and additive forms miss the
`.40` relative-L2 gate at `.44493` and `.45149`; the multiplicative form
`[1,z,s,zs]` passes at `.37689`, versus `.60931` for `[1,z]`, with cosine
`.92650`. The $z$ and $s$ collections have cosine `-.87042`, yet their product
still supplies the necessary non-additive term. See the
[head-response result](fast_screens/subject_number_native_head_response_coordinate_discovery_v2_result.json).
This is donor-dependent discovery on opened rows. The next test must predict $s$
from recipient-side upstream state and transfer to fresh rows before this becomes
a native generation mechanism.

A leave-one-construction-out donor-free test then replaces each row-specific
MLP6/7 displacement with one training-construction mean vector per answer direction.
Those two vectors predict the exact response scalar strongly (cosine `.96590`,
relative L2 `.28804`), but the necessary $zs$ multiplication amplifies the residual:
the complete coefficient error is `.51585`, improving only `.09346` over the native
baseline and degrading `.13896` from the exact-response oracle. See the
[donor-free proxy null](fast_screens/subject_number_donor_free_head_response_proxy_v2_result.json).
This closes scalar direction prototypes as a complete program, while preserving
their response-level transfer as a constraint for a vector-valued predictive state.

A fixed rank-2 predictor then used the top two training-construction recipient-state
PCs to predict the top two grouped MLP6/7 displacement PCs. It retained roughly
`.37–.43` of recipient-state energy and `.49–.54` of displacement energy, but changed
response error only `.28804→.28722` and coefficient error only `.51585→.51072`.
See the [rank-2 null](fast_screens/subject_number_rank2_recipient_state_response_proxy_v2_result.json).
This rejects activation/displacement energy as the next basis-selection criterion;
future coordinates should be oriented by the frozen head-response operator.

The response-oriented assay does improve the executable proxy. It contracts the
rank-16 training displacement span with the exact input gradient of the native-axis
head response and collapses the fit back to one fixed vector per answer direction.
On held-out constructions, response error is `.19194` and complete coefficient
error is `.44250`, versus `.28804` and `.51585` for the mean prototype. Eight
matched response-permutation controls have median coefficient error `.49038`.
The valid V2 prototype norms are `.87–1.08×` their mean-vector norms. The weaker
ridge V1 remains explicitly invalid because its vectors were `2.44–5.43×` the
mean norm despite better apparent errors. See the
[valid response-weighted result](fast_screens/subject_number_response_weighted_prototype_v2_result.json)
and [invalid weak-ridge receipt](fast_screens/subject_number_response_weighted_prototype_v1_result.json).
This supports causal-response weighting over activation PCA on opened authority;
it does not yet establish fresh-text OOD prediction or downstream removal.

The two all-row vectors were then frozen before a fourth corpus existed. On 512
novel background cells, their scalar coefficients transfer at cosine `.94757`,
relative L2 `.47024`, and sign agreement `.99609`, beating matched equal-norm
input prototypes by `.22118` error. The resulting rank-one writes predict native
causal effects at cosine `.95639`, relative L2 `.33462`, and sign agreement
`.98633`, with both new templates below `.361` error. The overall registered test
is mixed-null because replay of the older symbolic-law effect misses its `.50`
ceiling at `.56658`. Post-result diagnosis finds that the law itself is weaker
against fresh native effects (`.53796` error) than the extracted response program
(`.33462`); preserve the law-replay failure while treating native effect prediction
as a preregistered positive component result. The program's ports are native MLP8
state, frozen head weights, and a categorical edit-direction selector. See the
[fresh causal result](fast_screens/subject_number_response_weighted_fresh_causal_v1_result.json).

The matching removal assay subtracts the predicted rank-one write from the exact
opposite-number head. Residual-effect ratio is `.32420`; the removed amount matches
the native effect at cosine `.95968` and sign agreement `.98633`. Symbolic-law
removal leaves `.52788`, while four equal-norm same-site random directions leave
median ratio `1.00956`. V1's `.250333` work/jobs collateral remains a valid null,
but that contrast encodes grammatical number. The prospectively corrected set of
five non-number readers passes with maximum ratio `.18679`. See the
[V1 control-design null](fast_screens/subject_number_response_weighted_removal_v1_result.json)
and [V2 selective-removal result](fast_screens/subject_number_response_weighted_removal_v2_result.json).

Operational graph: `(native MLP8 state, requested direction) -> fixed prototype
response through H -> [1,z,s,zs] beta -> alpha times native L11H3 axis`. This is a
standalone one-writer graph at its declared boundary. The direction selector and
dense prototype storage remain external ports/costs rather than hidden claims.

The complete generator has now been tested at both subject sites of the opened
two-clause composition panel. Instrumentation is valid (`1.34e-5` maximum native
head replay error, exact zero-write replay, seven forwards/112 sequences). Given
the two independently generated writes, behavioral composition is essentially
exact: overall cosine `.99999991`, relative error `.000431`, sign agreement `1.0`,
and norm ratio `.999920`; the later site has zero effect on the earlier answer.

Do not promote this to complete-generator OOD composition. The registered result
is a null because the generated single-site writes move in the requested direction
on only `.625/.6875` of rows and site 1 RMS is `.00958` against a `.01` floor. A
post-hoc implementation audit exactly reproduces the coefficients and localizes
the failure to the response open slot: all 32 two-clause $s$ values lie outside
the fresh single-clause cardinality-four range `[-61.65,-11.06]`, clustering near
zero instead. The scalar gate therefore emits coefficients about `.91-.96` relative
error from the effective fixed coefficient. The writer node composes; its current
state-to-coefficient gate is not topology-invariant. Per the DCT briefing, future
work should keep context/position slots open or normalize/derive them rather than
average them away.

Opened-authority red-teaming closes two simple upstream repairs. A first apparent
site-conditioned improvement was invalid because whole MLP8-state interchange did
not carry changed MLP4--7 downstream slots; the endpoint error was `61.74`. The
corrected grouped-`YZ` intervention is exact but weak: exact response RMS is `3.73`,
the response-weighted prototype has `1.050` leave-template-out response error, and
its fixed-coefficient error is `.962`, worse than the old prototype and permutation
controls.

The complete `E/A/U/W/Y/Z` input factorial is also a valid null. Reciprocal
template selection yields `.785` held-out coefficient error, two folds with no
qualifying subset, and incompatible four-/five-port selections in the other two.
Möbius mass is low-order but spread across `YZ`, `EY`, `WY`, `AY`, `EZ`, `Z`, `WZ`,
`Y`, and further pair terms. Therefore retain L11H3's fixed rank-one writer as the
sparse reusable node, but do not represent its two-clause native generator as a
site prototype or small subset of these six MLP8-input ports. A future generator
must move earlier or expose a richer context graph explicitly.

## `module.attention.17.head2`

Aliases: head17.2, L17H2, regional late attention head. Related folded route:
[`PATH-SET2-001`](../COMPUTATION_PATH_REGISTRY.md).

### 2026-09-15 task-matched MLP17 interaction screen

On 96 controlled British/American prefixes, exact source-term folding through each
row's UK-minus-US unembedding reader finds that terms containing head17.2 jointly
have normalized cue-change norm ratio `.3153`. The largest such terms are earlier
residual × head17.2 (`.2655`) and MLP16 × head17.2 (`.1272`); head17.2 self is
`.03236`. The same terms were below `.017` on unrelated grammatical endpoints.

This establishes circuit-conditioned interaction magnitude, not causal necessity
or sufficiency. The exact instrument used 14 forwards/96 sequences, no fits or
behavioral logits, and closed at relative error $6.61\times10^{-16}$. See the
[primary screen](fast_screens/setting2_regional_four_source_term_census_v1_result.json).
The leading overall term is earlier residual × MLP16 (`.6184`), so future head17.2
work should remain a secondary branch until a selective intervention distinguishes it.

The later QK1 routing-removal census finds a full attention17 induced response of
`.24952` of the final pre-RMS numerator-change norm, aligned at `.24518`. The
frozen nine-head fold now localizes almost all of that induced response to
head17.2: response-norm ratio `1.05565`, cosine `.99901`, and family ratios
`1.07240/1.03835`. The best three heads plus the explicit BF16 projection
residual replay attention17 within `.02198`; exact response partition error is
$9.42\times10^{-10}$. All preregistered gates pass. This is response attribution
on an opened panel, not a selective head-level causal intervention. See the
[result](fast_screens/setting2_regional_attention17_head_response_fold_v1_result.json)
and [preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17_HEAD_RESPONSE_FOLD_V1_PREREGISTRATION.md).

An exact eight-corner `QK1 × QK2 × value` factorial then decomposes the induced
head17.2 write into seven Möbius terms. Exhaustive unit-gain support selection
on the opened panel chooses `QK2`, `value`, and `QK2×value`; equivalently, keep
native QK1 while replacing QK2 and value with their upstream-edited states.
This three-term program replays the paired head response at `.03528` relative
L2 and `.99977` cosine. Causal installation into native layer-17 background and
removal from the upstream-edited background pass at `.019–.045` familywise
relative L2 with perfect signs; five unrelated readers do not worsen and eight
equal-norm random writes in the same head output subspace have `.934–1.083`
error. Exact audits are below `1.90e-7`. This was an opened-panel candidate with
three native factor ports, not donor-free extraction. Its frozen corner was then
tested without reselection on 48 wholly fresh rows spanning two new templates,
two new city pairs, and six endpoint pairs. Response replay is `.02136` relative
L2 at `.999976` cosine; causal installation/removal errors are `.02098/.02090`,
both families have perfect signs, unrelated controls pass, and eight equal-norm
same-head random writes have `.898–1.045` error. The first fresh receipt narrowly
failed only the frozen exact-expansion audit (`2.285e-6` against `2e-6`) because
separate BF16 corner sums accumulated differently. A pre-documented correction
promoted only offline corner contractions to FP64, reducing that error to
`2.97e-9`; a result-switching guard confirmed every substantive metric moved by
less than `4.1e-6`. Thus the corner is a fresh-confirmed response module, while
native QK1 and upstream-edited QK2/value remain explicit live ports rather than
a donor-free extraction. Split or predict those ports next.
[Preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_FACTOR_INTERACTION_FOLD_V1_PREREGISTRATION.md),
[discovery result](fast_screens/setting2_regional_attention17h2_factor_interaction_fold_v1_result.json),
[fresh preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V1_PREREGISTRATION.md),
[precision correction](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_FACTOR_CORNER_FRESH_V2_CORRECTION.md),
[fresh result](fast_screens/setting2_regional_attention17h2_factor_corner_fresh_v2_result.json).

The first port-closing attempt used the exact propagated attention/MLP write
differences from layers 9–16 as candidate graph edges. Exhaustive unit-gain
search over every support of at most three terms selected `attn9 + attn15 +
mlp16`. This support does carry the target response (`.24991` relative L2,
`.99566` cosine) and predicts installation/removal (`.19416/.20324`, perfect
family signs), far ahead of equal-cardinality random supports (median `.94174`).
It nevertheless fails unrelated-reader controls in both families, sometimes by
roughly 3–4×, showing that omitted module changes provide important collateral
cancellation. This is a valid selectivity null: do not fresh-test or promote the
three modules. Whole writes are too coarse; the next split should use
response-oriented within-module interaction directions. The receipt explicitly
accounts for a non-selectable `4.83e-6` BF16 recurrence-rounding residual and
otherwise replays V1 scientific fields exactly. [Result](fast_screens/setting2_regional_attention17h2_port_source_fold_v2_result.json),
[preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V1_PREREGISTRATION.md),
[correction](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_PORT_SOURCE_FOLD_V2_CORRECTION.md).

The proposed one-layer causal-Hessian bridge has now been checked on the actual
raw bilinear MLP17 weights with the input normalizer explicitly outside the
slice. Weight contraction and FP64 autodiff agree at `2.90e-15` relative error;
background invariance, finite cross-differences, symmetry, analytic/autodiff
factor agreement, and a planted solver control all pass. This licenses the
identity and implementation for a response-oriented split. It does not license
low-rank compression by itself: rank 4 leaves `.89983` relative error in the
frozen random native subspace. Any next DCT use must choose and charge a causal
metric/subspace, and compare its truncation against that null. [Certificate](../../polynomial_causal/MLP17_CAUSAL_HESSIAN_IDENTITY_V1_RESULT.json),
[preregistration](../../polynomial_causal/MLP17_CAUSAL_HESSIAN_IDENTITY_V1_PREREGISTRATION.md).

A preservation-aware exhaustive search then tested whether the earlier
three-edge failure was merely target-only selection. Searching every one of
6,885 unit-gain supports through width five found
`attn9 + mlp9 + attn10 + attn11 + mlp15`: on discovery it reaches `.20763`
response error, `.19764/.19854` installation/removal, perfect signs, and a worst
control ratio `.83426`. Frozen fresh testing rejects promotion. Causal errors
remain good (`.23774/.23903`, perfect signs), but aggregate response is `.25068`
against `.25` and fresh apple/orange and work/jobs controls reach a worst ratio
`1.14877`. Preselected width-five nulls are much worse (median `3.42496`), so the
program is structured but panel-specific. This closes unit-gain whole-module
supports through five edges; do not widen the graph as a rescue. Split inside
the implicated modules using consumer/Hessian geometry. [Discovery](fast_screens/setting2_regional_attention17h2_port_preservation_search_v3_result.json),
[fresh null](fast_screens/setting2_regional_attention17h2_port_preservation_fresh_v1_result.json),
[fresh preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_PORT_PRESERVATION_FRESH_V1_PREREGISTRATION.md).

An exact native-head split tests the next finer obvious boundary. `attn10` and
`attn11` close as nine projected head writes each (recorded closure corrections
`1.38e-7/1.55e-7` relative), while the explicit layer-9 edit is already wholly
inside head 9.8. A deterministic width-eight beam selects
`attn9h8 + attn10h2 + attn10h5 + attn11h1 + attn11h6 + attn11h8 + mlp9 + attn15`.
It passes every discovery gate (`.20725` response, `.19918/.19991` causal,
`.83113` worst normalized control) against a same-width null median `3.7480`.
Frozen cue-role crossover testing preserves and improves target fidelity
(`.12782` response, `.12335/.12303` causal, perfect signs), but unrelated
readers reach `1.31635` of their allowed effect and the sparse program is less
selective than the complete-head expansion (`1.14505`). This is not a target
prediction failure; it is evidence that native head identity does not preserve
the context-dependent cancellation. Do not search more head subsets. Derive a
multi-reader consumer-response/Hessian basis inside these heads and the two MLP
ports. [Discovery](fast_screens/setting2_regional_attention17h2_head_source_beam_v1_result.json),
[crossover null](fast_screens/setting2_regional_attention17h2_head_source_fresh_v1_result.json),
[preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_HEAD_SOURCE_FRESH_V1_PREREGISTRATION.md).

The briefing's multi-reader response-coordinate proposal was then tested
directly. Installation/removal VJPs for the target and five controls define a
shared 1,152-dimensional Gram. Its rank-eight projector passes every discovery
gate (`.2021` response, `.1964/.1962` causal, `.8869` minimax) while retaining
only `20.00%` of source-write norm. It was exported as an explicit `1152×8`
orthonormal artifact. On zero-overlap competing-cues rows, target prediction and
causality improve (`.1525`, `.1635/.1606`), but preservation fails at `1.1881×`.
The predeclared source-PCA rank-eight comparator instead passes that panel at
`.97084×`, with `.11845` response and `.12512/.12328` causal errors while
retaining `59.01%` of source norm. Because that was a comparator result, it was
frozen on a second short-context panel with unseen spelling endpoints. It again
predicts target and causal effects (`.19797`, `.18977/.19002`) but fails two
family-0 controls at `1.11847×`. Thus neither a fixed first-order consumer basis
nor a fixed source-variance basis supplies transferable preservation. The next
coordinate must expose context as a computed gate or be identified across
multiple environments; another fixed global rank/support sweep is closed.
[frontier](fast_screens/setting2_regional_attention17h2_consumer_response_basis_v1_result.json),
[artifact receipt](fast_screens/setting2_regional_attention17h2_response_basis_export_v1_result.json),
[response fresh null](fast_screens/setting2_regional_attention17h2_response_basis_fresh_v1_result.json),
[source-PCA second-panel null](fast_screens/setting2_regional_attention17h2_source_pca_fresh_v1_result.json).

The first explicit context-slot test used those two frozen rank-eight source
bases as alternatives.  A one-bit, label-free gate selected whichever basis
retained more source-addition energy for each sequence, then ran the same
head17.2 corner and recursive suffix on a third zero-text-overlap panel.  It
retains `63.54%` of source norm and still predicts response (`.24643`) and
bidirectional causal effects (`.24076/.24153`, perfect family signs), while
four matched random rank-eight bases lose essentially the whole response
(`.9940--.9973` error).  But it fails preservation at `1.45896x` and does not
beat either fixed basis; the gate chooses the competing-cues basis on 37/48
rows.  The rank-16 union also fails (`1.44139x`).  Most decisively, the
unprojected eight-edge source support fails the same fresh controls at
`1.40077x`, so the null cannot be explained solely by a mistaken low-rank
projection.  Energy is therefore not an adequate context gate, and this
particular source support is not a reusable unit on the third environment.
This is a cheap activation-energy proxy for the briefing's open-context idea,
not a DCT factorization or a weight-derived polynomial gate.  Do not add gate
complexity around these two bases.  The next attempt must
jointly identify the source coordinate and its state/weight-derived gate across
multiple environments, or change the source support, then freeze both on an
untouched panel.  A post-run audit found one redundant deterministic removal
suffix call; it changes the literal suffix price from 960 to 1,008 sequences
but cannot change any metric or verdict.
[Preregistration](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_CONTEXT_GATED_SOURCE_BASIS_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_attention17h2_context_gated_source_basis_v1_result.json),
[execution note](../../polynomial_causal/SETTING2_REGIONAL_ATTENTION17H2_CONTEXT_GATED_SOURCE_BASIS_V1_EXECUTION_NOTE.md).

## `module.attention.9`

Aliases: attention block 9, attn9. Related route:
[`PATH-SET2-001`](../COMPUTATION_PATH_REGISTRY.md).

### 2026-09-15 propagated interaction with MLP16

An exact learned-residual expansion of the task-matched regional `earlier residual
× MLP16` MLP17 term ranks the propagated attention9 block write first among 34
embedding/module sources. Its change-norm ratio is `.26094` globally and
`.19032–.32205` across the four construction families. Exact source reconstruction
error is $1.45\times10^{-7}$ and folded cross-term closure is
$3.21\times10^{-16}$.

The top five sources together still leave `.64073` relative error, so attention9
is an organizing handle rather than a sufficient path. The subsequent exact
head split assigns nearly all aligned attention9 × MLP16 change to head9.8:
its change-norm ratio is `1.01222`, its aligned fraction is `.97762`, and its
family ratios are `.99497–1.05071`. Head9.7 is the only substantial secondary
term (`.20124`); keeping both gives `.08243` relative replay error. The nine-head
sum closes at $1.97\times10^{-16}$, while the separately rounded native attention
output bridge is $1.43\times10^{-7}$. This establishes a computation-path overlap
with the setting1 head9.8 circuit, without yet showing that this specific folded
term is causally sufficient. See the [upstream fold](fast_screens/setting2_regional_mlp16_upstream_source_fold_v1_result.json)
and [head split](fast_screens/setting2_regional_attn9_head_mlp16_fold_v2_result.json).

### 2026-09-15 head9.8 QK source interactions

Holding native denominators and values fixed, both QK factors of head9.8 admit
exact ordered query-source × key-source expansions over block-8 carry, attention,
and MLP writes. QK1 is led by carry8 × carry8 at `.85882` of the complete
head9.8 folded change norm, stable at `.78111–1.00441` across four families.
MLP8-query × carry-key is `.20754`, and carry-query × MLP8-key is `.10556`;
these top three replay with `.19641` relative error.

QK2 is more distributed: carry8 × carry8 `.33203`, MLP8-query × carry-key
`.23088`, carry-query × attention8-key `.16233`, carry-query × MLP8-key
`.14406`, and attention8-query × carry-key `.14170`. Its top-three replay error
is `.33529`. All attention8-containing interactions aggregate to `.20479` in
QK1 and `.36420` in QK2. Exact score and downstream-fold closure are below
$6.89\times10^{-16}$; the recomputed native head bridge is
$1.97\times10^{-7}$. This localizes routing computations along the folded path,
while holding values and downstream context fixed. See the
[QK source fold](fast_screens/setting2_regional_head9_8_qk_source_fold_v3_result.json).

The dominant QK1 carry×carry term further expands into 289 ordered pairs among
embedding and attention/MLP writes from layers 0–7. The exact score and downstream
closures are $1.50\times10^{-16}$ and $1.32\times10^{-16}$, and the recomputed
parent ratio is `.85881922`. Its top ten pairs leave `.66738` replay error, so the
fine-grained path is distributed. Cross-source terms aggregate to `.87347` of the
parent change; attention-containing and MLP-containing aggregates are `.62900`
and `.81750`. Nine top-ten pairs lie among attention5, MLP5, MLP6, and MLP7,
led by MLP6-query × attention5-key at `.04839`. See the
[carry-source fold](fast_screens/setting2_regional_head9_8_qk1_carry_source_fold_v1_result.json).
This licenses a frozen late-group block test, not an individual-pair causal claim.

The frozen late group $D=\{\mathrm{attention5},\mathrm{MLP5},\mathrm{MLP6},
\mathrm{MLP7}\}$ gives an exact four-block expansion against the other 13 carry
sources. $D\times D$ leads at `.45480` of the parent change norm and remains
`.42180–.48700` across the four prompt families. The ordered cross-boundary sum
$D\times R+R\times D$ is `.44088`; retaining all three terms that touch $D$
leaves only `.11186` relative replay error. The grouped score and downstream sum
close at $1.51\times10^{-16}$ and $1.07\times10^{-16}$. This is selected-row
attribution; causal use awaits the preregistered fresh routing edit. See the
[late-group fold](fast_screens/setting2_regional_head9_8_qk1_late_group_fold_v1_result.json).

On 48 rows unused to select the group, the three late-touching blocks retain
`.13562/.14175` folded replay error across the two authored templates. Recursive
removal is material: `.52843/.59816` of full-head9.8 removal and
`6.99/5.76` times the $R\times R$ removal effect. But the frozen
MLP16×MLP17 reader predicts the opposite signed change (cosine `−.92536`, zero
sign agreement), and unrelated-reader ratio `.53452` misses the `.50` gate in
one template. Thus the QK1 grouping is causally active inside head9.8, while the
registered backward suffix is not its behavioral explanation. QK2 controls are
`.695/.796` of the selected edit; late current-value controls are only
`.0419/.1219`. See the [fresh routing receipt](fast_screens/setting2_regional_head9_8_qk1_late_group_fresh_routing_v3_result.json).

The exact downstream response census explains the sign failure. The directly
propagated edited attention9 write is the largest final pre-RMS numerator term,
with change-norm ratio `.54223` and aligned fraction `.53301`. Attention17 is
second (`.24952`, aligned `.24518`). MLP17 is third by norm (`.15921`) but
anti-aligned (`−.13785`); MLP9 is only `.04892`, so immediate MLP compensation
does not explain the mismatch. The top five terms replay at `.24342` error, and
the complete numerator predicts the final logit edit at cosine `.99903` with all
24 signs correct. The next fold should follow direct propagation and attention17,
not deepen the rejected MLP16×MLP17 branch. See the
[response census](fast_screens/setting2_regional_qk1_edit_downstream_response_census_v2_result.json).

The interaction-decomposition briefing's per-layer causal-Hessian allocation
has also been applied to the earlier CrossFirst child/remainder composition
failure at this head9 boundary.  Nested JVPs split the complete suffix mixed
derivative into MLP9, separate attention/MLP stages through block17, and the
final readout while retaining every prompt.  The 18 terms sum to the complete
mixed derivative within `3.10e-7` for the real writer and `7.69e-7` across four
equal-norm random writers.  The full Hessian predicts the finite interaction at
`.090--.328` relative error and leaves only `.009--.042` child-relative error.
The preregistered three-stage energy-sparsity criterion fails in two families,
but an opened-panel audit identifies the same two stages in every
leave-one-family-out fold: MLP10 and attention17.

Frozen on 48 zero-full-prefix-overlap rows, that two-stage correction reduces
aggregate additive composition error by `27.1--55.5%` and leaves all four
families at `.0175--.0944` of the child effect, beating four fixed two-stage
nulls (`.1632--.2090` minimax).  Exact allocation closure remains below
`3.40e-7`.  The stronger finite-interaction-vector gate fails: two family
cosines are `.8148/.8544`, with five row-level sign misses, mostly on very small
interactions.  Preserve both conclusions.  This is a fresh, compact correction
to aggregate behavioral composition, but not a faithful per-row decomposition
or standalone circuit: exact child/remainder fields, context derivatives, and
the native suffix remain live ports.
[Allocation preregistration](../../polynomial_causal/CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1_PREREGISTRATION.md),
[allocation result](../../polynomial_causal/CROSSFIRST_PER_LAYER_CAUSAL_HESSIAN_V1_RESULT.json),
[fresh preregistration](../../polynomial_causal/CROSSFIRST_HESSIAN_TOP2_FRESH_V1_PREREGISTRATION.md),
[fresh result](../../polynomial_causal/CROSSFIRST_HESSIAN_TOP2_FRESH_V1_RESULT.json),
[fresh scale/sign audit](../../polynomial_causal/CROSSFIRST_HESSIAN_TOP2_FRESH_V1_AUDIT.json).

A second, wholly new endpoint panel rejects the retrospectively selected
four-stage Hessian support `MLP9 + MLP10 + attention17 + readout`.  Its four
family child-relative errors are `.325/.098/.083/.151`; removing readout changes
the minimax only `.325→.326`, so the support is neither sufficient nor
stage-necessary.  The complete 18-stage Hessian also misses the live
`licence/license` interaction at `.403` child-relative error, excluding stage
selection as the explanation for that endpoint.  A frozen scale curve then
shows the complete Hessian is locally valid (all quarter-scale relative errors
at most `.106`) while its `licence/license` error grows monotonically with edit
scale.  A quarter/half-scale cubic extrapolation reduces the native-scale error
by `97.7%`, establishing real higher-order curvature rather than a dead reader
or Hessian implementation bug.

The exact direct third derivative sharpens that conclusion.  After an initial
unscored PyTorch cache-lifetime failure was corrected by evaluating the same
rotary formula statelessly, symmetry-equivalent triple-JVP orderings agree to
`5.22e-7`.  `H+C3` improves the hard endpoint's finite-interaction relative L2
from `.3741` to `.1209` at `.9982` cosine and passes the `.10` child-relative
bar in every construction family (worst `.09979`) and five of six endpoint
concepts.  It is nevertheless a valid null: `licence/license` remains `.13025`
child-relative, the direct cubic differs from the frozen finite-estimated cubic
by `.320` there and `.333--1.066` elsewhere, two already-easy concepts worsen,
and one family's improvement is only `22.4%`.  Thus third order is a useful
diagnostic but not a frozen correction, much less an extracted circuit.  Do not
spend the next circuit test merely adding Taylor order to the live deep suffix;
test the briefing's consumer-defined most-additive split at the first
downstream bilinear layer instead.
[Four-stage preregistration](../../polynomial_causal/CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_PREREGISTRATION.md),
[four-stage result](../../polynomial_causal/CROSSFIRST_HESSIAN_FOUR_STAGE_FRESH_V1_RESULT.json),
[scale-curve result](../../polynomial_causal/CROSSFIRST_HESSIAN_SCALE_CURVE_V1_RESULT.json),
[direct-cubic preregistration](../../polynomial_causal/CROSSFIRST_DIRECT_CUBIC_V1_PREREGISTRATION.md),
[direct-cubic correction](../../polynomial_causal/CROSSFIRST_DIRECT_CUBIC_V1_CORRECTION.md),
[direct-cubic result](../../polynomial_causal/CROSSFIRST_DIRECT_CUBIC_V1_RESULT.json).

The briefing's full-width weight-only DCT reader proposal has now also been
tested directly at raw MLP9.  An implicit symmetric orthogonalized ALS used only
`Left`, `Right`, and `Down`; eight analytic tensor contractions replay nested
JVPs to `3.92e-15` relative error.  A first execution serialized some symmetric
factor signs incorrectly and is explicitly retained as invalid; the corrected
run leaves the scientific outcome unchanged.  Rank eight removes only
`.65--.77%` of fixed random-probe tensor norm, is not identifiable across two
seeds (minimum input/output principal cosines `.0116/.0145`), and captures only
`.0711` of the exact opened-panel suffix-reader norm, below the equal-rank
random median `.0794`.  A positive post-hoc audit finds four individually
reproducible factors after permutation (input cosines `.9992--.9998`, output
cosines above `.99999`), but their reader coverage `.05757` is likewise
indistinguishable from the rank-four random median `.05811`.  Thus real stable
weight-tensor structure exists, but unconditioned raw-weight DCT does not supply
the CrossFirst reader.  Do not enlarge this same energy-ranked raw tensor basis;
the remaining DCT route must keep native context/RMS slots open or be
prompt-conditioned, and must still face causal extraction/removal tests.
[DCT preregistration](../../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_V1_PREREGISTRATION.md),
[implementation correction](../../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_V1_CORRECTION.md),
[corrected DCT result](../../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_V1_RESULT.json),
[rank-stability audit](../../polynomial_causal/MLP9_DCT_UNSUPERVISED_READER_RANK_STABILITY_V1_AUDIT.json).

The briefing's most-additive objective was then tested in the narrowest
hierarchy-preserving family: move one frozen scalar fraction of the existing
remainder into the child while keeping their sum and writer direction exact.
Selection used only finite nonadditivity of the first consumer
`z + MLP9(RMS(z))`, with both pieces required to remain substantial.  The
selected `lambda=.525` is stable under leave-one-family-out (`.500--.525`),
balances the two local effects at roughly `.47--.54` of the parent, and lowers
worst-family interaction over the smaller single effect from `.13697` to
`.07902` (`42.3%`).  This is not a circuit-specific decomposition: four
equal-norm orthogonal writer controls optimize to much smaller
`.03266--.03561` interactions.  The registered specificity prediction fails,
so no new behavioral panel was opened.  Scalar rebalancing can manufacture a
locally additive partition of any direction; the next most-additive attempt
must split consumer-relevant vector/tensor modes across multiple readers, not
relabel fractions of the same one-dimensional field.
[Most-additive preregistration](../../polynomial_causal/CROSSFIRST_MOST_ADDITIVE_SPLIT_DISCOVERY_V1_PREREGISTRATION.md),
[most-additive discovery null](../../polynomial_causal/CROSSFIRST_MOST_ADDITIVE_SPLIT_DISCOVERY_V1_RESULT.json).

Keeping the native state and RMS denominator open rescues compression but not
reader discovery.  A behavior-blind contextual Hessian range finder evaluates
20 fixed input directions through the exact local map
`z + MLP9(RMS(z))`, freezes an eight-dimensional output basis on one panel,
and transfers it without refitting to the disjoint new-endpoint panel.  Mixed
JVP symmetry closes at `1.58e-7`.  The basis retains `.9978553/.9978550` of
contextual response norm on discovery/confirmation, and the four independently
stable raw-DCT inputs carry `148.34x` the response energy of 16 random inputs;
their own contextual response family is `99.998%` rank-eight by energy.  This is
a real, cross-panel high-curvature MLP9 response core.  It is not the
CrossFirst reader: coverage is only `.07187/.08819`, comparable to random
output subspaces and to the raw-weight DCT basis, with confirmation-family
coverage `.08366--.09158`.  Preserve the compressed response core as a generic
model primitive, but do not promote it into this behavior's circuit.  The next
useful step is to export its exact weight/state-derived RMS coefficient law as
a standalone node and test response prediction on fresh contexts; behavioral
removal should wait until a downstream circuit actually reads the node.
[Contextual-range preregistration](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_RANGE_V1_PREREGISTRATION.md),
[contextual-range result](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_RANGE_V1_RESULT.json).

That standalone export has now been tested prospectively.  V1 froze the four
stable raw-DCT inputs and an energy-selected rank-eight contextual output
subspace, with the exact RMS coefficient law implemented from weights and the
live pre-MLP9 state.  Its aggregate fresh-context prediction looked strong
(`.02629` relative L2, `.999654` cosine), but the preregistered result was a
valid null: low-energy off-diagonal ordered pairs reached `.9092` relative
error and a cancellation-sensitive mixture reached `.09957`.  This was not an
implementation failure—the analytic formula replayed nested JVPs to about
`7e-7`, and isolated package replay was exact.  The failure exposed a bad
selection proxy: those off-diagonal pairs carried only `.00366` of total
response energy but became important after diagonal cancellation.

A prospective rank-16 correction, selected before opening a second disjoint
64-prefix panel, resolves that failure.  It retains `.9999911` of discovery
response norm and predicts the second panel at `.004928` relative L2 and
`.999988` cosine; every family is below `.00520`, every ordered pair below
`.01047`, and all eight frozen mixtures below `.00801`.  The matched random
rank-16 median error is `.99281`.  The package reproduces its fixture within
`7.3e-9` relative L2 in an independent isolated replay,
uses only the native pre-MLP9 `z9` activation port, and stores `.6751` as many
tensor parameters as the native MLP9 bilinear matrices.  This is the first
extracted, cross-panel predictive, compositionally reusable generic MLP9
response node from this DCT line.  It still has no behavioral reader,
selective-removal, or complete-circuit evidence, and the native activation port
means it is not an autonomous input-to-output circuit.
[V1 node preregistration](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1_PREREGISTRATION.md),
[V1 diagnostic null](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_FRESH_V1_RESULT.json),
[rank-16 correction preregistration](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_PREREGISTRATION.md),
[rank-16 result](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_NODE_RANK16_FRESH_V2_RESULT.json),
[standalone package](../../polynomial_causal/extracted_circuits/mlp9_contextual_dct_node_v1/README_RANK16_V2.md).

The node now also has a bounded causal use, with an important scope limit.  An
opened-panel suffix-reader search predicts exact local-Hessian logit responses
at `.00521` relative L2, but its absolute-effect gate fails because unit-norm
DCT directions induce only a `3.60e-6` logit contrast; retain that receipt as a
valid discovery null rather than silently changing its bar.  A subsequent
opened-panel scale curve freezes direction 2, its diagonal interaction, scale
32, and reader `token 21215 - token 6165`.  On 64 new score-blind contexts the
packaged node predicts the finite local state interaction at `.00567` relative
L2 and the exact-suffix all-logit installation effect at `.00430`.  Removing
the packaged response leaves `.03696` of the native all-logit mixed effect and
`.01397` of the frozen-reader mixed effect; worst family ratios are `.04039`
and `.01770`.  Eight matched random removals reach only `.120--.137` of the
target removal, and three frozen collateral contrasts are `.152/.158/.459`.

A post-outcome adversarial audit strengthens but also bounds specificity.
Matching response norm separately at every token gives target-effect ratios
`.103--.149` for isotropic controls, `.195--.236` for controls inside the same
rank-16 output subspace, and `.110--.319` for the other three diagonal DCT
responses.  However, alternate diagonal responses still cause
`.418--.964` as much total all-logit change.  Therefore this is a four-trait
*synthetic local-interaction primitive*: it has cross-context prediction,
standalone execution, reader-selective install/removal, and bilinear reuse.
It is not a semantically identified behavior, naturally occurring feature
removal, globally unique residual direction, suffix extraction, or complete
sparse circuit.
[Reader-discovery null](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_DOWNSTREAM_READER_DISCOVERY_V1_RESULT.json),
[finite-scale discovery](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_FINITE_SCALE_DISCOVERY_V1_RESULT.json),
[fresh causal preregistration](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_PREREGISTRATION.md),
[fresh causal result](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_RESULT.json),
[specificity audit](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_SPECIFICITY_AUDIT_RESULT.json).

A natural-behavior bridge was tested against the established equality/copy
reader and closes this local DCT route for that circuit.  The rung-500 score
restoration remains healthy (`1.121/1.156` copy-token NLL recovery across the
two opened splits), so these are not dead-behavior nulls.  The frozen generic
DCT basis captures `.1764/.1823` of native/restored copy-positive MLP9 response
norm, modestly above rank-16 random controls around `.12`, but it captures
noncopy responses slightly better (`.1846--.1880`) and therefore fails semantic
selectivity.  A copy-conditioned response PCA is not stable either: rank 64
captures `.8099` on its discovery documents but only `.3649` on the next
documents; rank 16 confirms at only `.2465/.2591` native/restored coverage.

Most importantly, the exact causal ceiling fails.  Under the equality-score
removal, replacing the complete MLP9 write with its exact native write recovers
only `.03929` of the copy-token NLL effect.  DCT rank 16 recovers `.01988`, while
copy-conditioned ranks 16 and 64 are slightly harmful (`−.00939/−.01246`).
Every installed write matches its prescribed tensor exactly, both 32-document
halves agree on the exact ceiling (`.0411/.0364`), and noncopy mean changes stay
below `.0011` nat.  MLP9's equality response is therefore a reproducible reader
or diagnostic of the upstream score action, not a sufficient causal site.  Do
not search for a better fixed MLP9 output basis for this behavior.  A future
equality decomposition must retain the altered upstream residual/action path
and test multi-module state restoration; the local DCT node remains generic.

That boundary test now succeeds.  Factoring the complete post-MLP9 state as
the pre-MLP residual plus MLP9's write shows that restoring only the native
pre-state recovers `1.01032` of the copy-token effect, whereas restoring only
the native MLP9 write recovers `.02863`.  Restoring both components or
recomputing MLP9 from the restored pre-state reproduces native behavior exactly
(`1.0` recovery and `0` mean noncopy change).  The causal recovery interaction
is small (`-.03895`), and the pre-state result is positive in both document
halves and all four distance/multiplicity cells.  Native and absent trajectories,
writes, BF16 boundary construction, and both native-state installation routes
all replay exactly.  This establishes a two-edge *oracle boundary factor* with
the upstream residual as the dominant edge; it does not extract that upstream
state, establish fresh/OOD transfer, or turn MLP9's generic DCT node into the
equality circuit.  The next useful decomposition target is therefore the
pre-MLP9 residual delta generated by the known equality score action.
[Natural DCT bridge null](../../polynomial_causal/MLP9_CONTEXTUAL_DCT_EQUALITY_RESPONSE_DISCOVERY_V1_RESULT.json),
[consumer-response basis null](../../polynomial_causal/MLP9_EQUALITY_CONSUMER_RESPONSE_BASIS_DISCOVERY_V1_RESULT.json),
[projected-write causal null](../../polynomial_causal/MLP9_EQUALITY_PROJECTED_WRITE_CAUSAL_DISCOVERY_V1_RESULT.json),
[projected-write execution corrections](../../polynomial_causal/MLP9_EQUALITY_PROJECTED_WRITE_CAUSAL_DISCOVERY_V1_EXECUTION_NOTE.md),
[post-MLP9 state factorial result](../../polynomial_causal/EQUALITY_POST_MLP9_STATE_FACTORIAL_DISCOVERY_V1_RESULT.json),
[factorial execution note](../../polynomial_causal/EQUALITY_POST_MLP9_STATE_FACTORIAL_DISCOVERY_V1_EXECUTION_NOTE.md).

The dominant upstream residual has now been split one boundary earlier into
attention8, MLP8, and attention9 writes.  The full `2^3` oracle factorial is
strongly sparse: attention8 alone recovers `.92683`, versus `.14238` for MLP8
and `.02339` for attention9; all pair and third-order recovery interactions are
below `.029`.  The selected singleton is positive in every cell and half with
zero replay error.  On the separately frozen code-OOD corpus the same edge
remains dominant and stable but over-recovers at `1.17756`, failing the frozen
natural-to-code calibration-drift threshold despite only `.00660` nat noncopy
damage.  Edge identity transfers; its behavioral scalar does not.

The edge executor is the zero-new-parameter equality contraction at L8H4.  A
projected-payload form preserves behavior but fails tensor replay because it
reorders BF16 operations.  The corrected raw-payload-then-output-projection
form has exactly zero term and removal replay error and a `2.87e-7` bilinear
composition error.  Exact reinstallation still fails because BF16 subtraction
and addition are not inverses; behavioral recovery remains close (`1.17494`
versus oracle `1.17756`) while logit relative error is `.01582`.  Retain this as
an exact extracted remover and close behavioral installer, not a bit-exact
bidirectional replacement.  The next graph-level task is to choose a canonical
arithmetic boundary or explicit rounding/remainder port, then extract the
native score and raw-payload producers rather than widening downstream bases.

That arithmetic boundary is now implemented and verified.  A separately typed
FP32 roundoff port makes semantic split/removal and merge/reinsertion exactly
reversible: term, split, merge, MLP9, and logit replay errors are all zero on
the complete code-OOD role, and canonical recovery exactly equals the oracle
at `1.17756`.  The correction has only `.001215` of full-write norm and `.002712`
of term norm, although it is nonzero in `9.43%` of elements.  It is compiler
bookkeeping, not another semantic edge.  The resulting node has zero new
learned parameters, exact removal and installation, cross-corpus causal use,
and an explicit bilinear composition law.  The remaining scientific gaps are
the failed natural-to-code scalar calibration and extraction of the native
score/raw-payload producers feeding this node.

One of those producer ports is now reusable across heads.  The frozen
natural-text L5H5 score adapter, inserted into the exact-order L8H4 node, recovers
`.97287` of the code-OOD copy effect with `.00240` nat noncopy damage and
`.99883/.94979` half recoveries.  The frozen L7H3 wrong-score donor gives
`-1.84448`, and the target score cosine on equality edges is `.82849`.  The
exported adapter charges one stored calibration scalar and zero new learned
parameters.  This identifies a sparse `L5H5 score -> L8H4 equality node` edge;
the next extraction boundary is residual-to-Q/K score production within L5H5,
while L8H4's raw payload remains a separate native port.

That L5H5 score-production boundary is now partially extracted.  A
zero-parameter four-port node independently replays the exact causally masked
bilinear score from post-projection, post-RMS, post-rotary `q1/k1/q2/k2`.  Across
the complete 192-document code-OOD role, score and downstream donor-logit errors
are exactly zero, recovery remains `.9728678`, and the explicit `2x2`
half-coordinate product expansion has `1.07e-7` relative error.  A first receipt
reported `1.0932` score error because its diagnostic node omitted the triangular
causal mask; its simultaneous zero logit error exposed the miscoding, and the
bound correction passes all six gates.  This establishes prediction, causal
installation, zero-parameter extraction, and algebraic reuse for the score
node.  It does not yet extract the residual-to-Q/K linear projections or their
source support, and it leaves L8H4 raw payload native.

The residual-to-Q/K operation and a sparse source support are now explicit.
The extracted zero-learned-parameter executor consumes one pre-L5 residual and
the four frozen native L5H5 head-weight slices; full score and donor logits match
factor capture exactly.  With RMS retained as a shared nonlinear context slot,
an exhaustive natural-only source screen chose `L2+L3+L4` from the exact
`E,L0,...,L4` provenance groups.  Natural score error/cosine is
`.14462/.99197`; frozen code error/cosine is `.15268/.99233`; downstream
recovery is `.90458` with `.00172` nat noncopy damage.  The complete three-port
Möbius graph closes at `3.94e-9`.  A prospective red-team removes the explicit
roundoff port and repeats selection: the same support survives, code error
slightly improves to `.15255`, recovery changes by only `-.00020`, and an
equal-norm position roll moves recovery by `.00287`.  This is therefore a
correction-free sparse boundary graph with OOD prediction, causal removal/
installation, extraction, and composition.  The layer-2/3/4 writes are still
native frozen ports, so recursive producer extraction—not another score-basis
search—is the live handoff.

The prospective module-write refinement first exposed an arithmetic bug: a
flattened sum changed BF16 association and missed exact parent replay by
`.0004195`.  The corrected canonical-order V2 has zero all-six replay error and
therefore makes its negative result interpretable.  Natural-only selection
keeps `M2,A3,M3,A4,M4`, omitting only `A2`; the selected graph transfers to code
at `.03875` parent-score error and `.99934` cosine, recovers `.88553`, causes
`.00132` nat noncopy damage, and closes composition at `9.60e-9`.  Five writes
exceed the preregistered ceiling of four, so module-level sparsity is a valid
null.  The zero-parameter five-write executor is retained as the next exact
boundary, with all five native producers still external.

M4 is now executable behind a finer weight-only boundary.  Exact native product
selection is distributed (4,096/4,608 atoms), whereas SVD of the M4 writer
contracted with all four frozen L5H5 Q/K readers reaches the frozen natural gate
at rank 256.  The corrected float64 bridge error is `2.06e-13`; code score
error/cosine is `.03818/.99927`, recovery `.89159`, noncopy damage `.00155` nat,
and an equal-norm position roll falls to `.71098` recovery.  This fails the
registered rank-64 compactness bar and retains all native product evaluations.
Most-additive natural selection also remains noncompositional: `.13323` code
score error and `.42108` behavioral error.  Retain the rank-256 executor and
represent its child/remainder interaction explicitly in the next graph.

That interaction is now a validated first-class node.  The baseline, rank-128
child effect, rank-128 remainder effect, and their Möbius cross-difference close
with zero measured natural/code score and downstream replay error.  Removing
the interaction changes copy-token NLL by `.30096` of the joint-vs-baseline
effect (`.26961–.32079` across every subtype and half) with only `-.000345` nat
noncopy mean change.  Its one-query equal-norm roll has `.52158` effect cosine
to true removal.  Direct and composed recovery are both `.89159`.  The graph
therefore has frozen OOD prediction, extraction, selective removal, and exact
composition/reuse at this boundary.  Its four score evaluations remain an
explicit execution-cost caveat and the next folding target.

The first direct-kernel fold is a useful two-sided red-team null.  Sharing
baseline/child/remainder Q/K projections cuts 16 projections to 12, but misses
the code interaction gate at `.10063`; all four equivalent arithmetic gauges
agree.  A natural-selected K2 correction gives a 13-projection score-space
positive (`.09956`) and nearly unchanged aggregate recovery (`.89071` versus
`.89159`).  Behavioral comparison overturns that apparent success: tokenwise
composed replay error is `.258`, and the approximate interaction-removal vector
has only `.294` cosine and `1.198` relative error to the exact removal.  This
rules out both a coding-gauge explanation for the 12-projection null and a
score-only acceptance of the corrected surrogate.  The exact 16-projection
graph remains the reusable behavioral node.

Behavior-aware native-order selection then tested every proper subset of child
`Q1,K1,Q2,K2` projections.  The full four-map implementation control reproduces
the authority removal vector exactly on both panels, but no proper subset
passes even on natural data.  Best diagnostic `Q1+Q2` has worst natural-cell
removal error `.915`; frozen code error/cosine is `.846/.650`.  Its composed
replay and aggregate recovery are exactly authoritative because the graph
algebra closes to the native joint corner, demonstrating that those two checks
are blind to an incorrect interaction-removal counterfactual.  All four child
ports are jointly necessary in this representation; the next decomposition
target is their exact Boolean/Möbius interaction lattice.

That lattice is now exact and behaviorally audited.  All 15 nonconstant terms
reconstruct the native child score at zero measured error.  Cumulative order
three has only `.144/.132` natural/code score error, but still incurs `.863`
overall natural removal-vector error and `.937` in its worst cell.  The omitted
four-way term has small norm and near-zero global cosine, while the full
order-four control is behaviorally exact.  Thus neither score norm nor
polynomial order identifies removable edges.  The compact exact handoff is the
algebraic two-factor form: first-QK correction, second-QK correction, and their
cross-product as three macro-nodes.

That handoff required a precision correction.  The first real-arithmetic
three-node implementation is invalid (`.339/.353` natural/code closure error)
because native BF16 dot/product association contributes at the scale of the
small correction.  Making that arithmetic residual an explicit fourth node
restores zero score and behavioral closure.  On code, first-factor,
second-factor, algebraic-cross, and arithmetic-node removals have `.834`,
`.859`, `.688`, and `.848` effect norm relative to exact interaction removal;
all are selective with at most `.000507` nat noncopy mean change.  The cross is
only `.00122` of correction-score norm yet causally large.  The exported
four-node graph is the current exact reusable boundary; its 16 native Q/K
projections and upstream factor ports remain the next extraction cost.

The preferred export now removes the native-child-score oracle from that
interface.  It accepts eight derived/native raw Q/K ports plus rotary context
and performs normalization, rotation, native BF16 multiplication, and arithmetic
residual construction internally.  Natural/code closure is zero and every
node-removal, selectivity, score-norm, and rolled-control statistic matches the
validated parent exactly.  The remaining external boundary is residual-to-port
projection plus the rank-256 M4 producer, not a behavior or score oracle.

The current preferred export internalizes those frozen projections too.  It
accepts four residual corners plus rotary context and reuses the native Q/K
weights.  Across 1,811,939,328 values, projected ports are bitwise identical;
all score and behavioral statistics also match exactly.  The live upstream
boundary is therefore construction of baseline/child/remainder/joint residuals
and the rank-256 M4 mode producer.
[three-edge factorial](../../polynomial_causal/EQUALITY_PRE_MLP9_THREE_EDGE_FACTORIAL_DISCOVERY_V1_RESULT.json),
[code-OOD calibration null](../../polynomial_causal/EQUALITY_A8_EDGE_CODE_OOD_CONFIRMATION_V1_RESULT.json),
[projected-payload extraction null](../../polynomial_causal/EQUALITY_L8H4_EXTRACTED_NODE_CODE_OOD_V1_RESULT.json),
[exact-order extraction result](../../polynomial_causal/EQUALITY_L8H4_EXACT_ORDER_NODE_CODE_OOD_V2_RESULT.json),
[reversible-edge result](../../polynomial_causal/EQUALITY_L8H4_REVERSIBLE_EDGE_CODE_OOD_V3_RESULT.json),
[reusable score-port result](../../polynomial_causal/EQUALITY_REUSABLE_SCORE_PORT_CODE_OOD_V1_RESULT.json),
[score-adapter export](../../polynomial_causal/EQUALITY_L5H5_SCORE_ADAPTER_EXPORT_V1_RESULT.json),
[bilinear score-node red-team null](../../polynomial_causal/EQUALITY_L5H5_BILINEAR_SCORE_NODE_CODE_OOD_V1_RESULT.json),
[causal bilinear score-node result](../../polynomial_causal/EQUALITY_L5H5_BILINEAR_SCORE_NODE_CODE_OOD_V2_RESULT.json),
[sparse residual-source graph](../../polynomial_causal/EQUALITY_L5H5_RESIDUAL_SOURCE_GRAPH_V1_RESULT.json),
[correction-port red-team](../../polynomial_causal/EQUALITY_L5H5_RESIDUAL_CORRECTION_REDTEAM_V1_RESULT.json),
[L2+L3+L4 graph export](../../polynomial_causal/EQUALITY_L5H5_L234_SCORE_GRAPH_EXPORT_V1_RESULT.json),
[invalid flattened-write refinement](../../polynomial_causal/EQUALITY_L5H5_L234_MODULE_WRITE_GRAPH_V1_RESULT.json),
[canonical module-write sparsity null](../../polynomial_causal/EQUALITY_L5H5_L234_MODULE_WRITE_GRAPH_V2_RESULT.json),
[five-write graph export](../../polynomial_causal/EQUALITY_L5H5_FIVE_WRITE_SCORE_GRAPH_EXPORT_V1_RESULT.json),
[native M4 product sparsity null](../../polynomial_causal/EQUALITY_L5H5_M4_PRODUCT_CHANNEL_GRAPH_V1_RESULT.json),
[invalid float32 contracted-mode receipt](../../polynomial_causal/EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V1_RESULT.json),
[float64 contracted-mode compactness/composition null](../../polynomial_causal/EQUALITY_L5H5_M4_CONTRACTED_MODE_GRAPH_V2_RESULT.json),
[most-additive mode-split null](../../polynomial_causal/EQUALITY_L5H5_M4_MOST_ADDITIVE_MODE_SPLIT_V1_RESULT.json),
[rank-256 mode-boundary export](../../polynomial_causal/EQUALITY_L5H5_M4_RANK256_MODE_GRAPH_EXPORT_V1_RESULT.json),
[explicit M4 interaction graph](../../polynomial_causal/EQUALITY_L5H5_M4_EXPLICIT_INTERACTION_GRAPH_V1_RESULT.json),
[explicit interaction-graph export](../../polynomial_causal/EQUALITY_L5H5_M4_EXPLICIT_INTERACTION_GRAPH_EXPORT_V1_RESULT.json),
[shared-projection kernel null](../../polynomial_causal/EQUALITY_L5H5_M4_SHARED_PROJECTION_KERNEL_V1_RESULT.json),
[arithmetic-gauge red-team null](../../polynomial_causal/EQUALITY_L5H5_M4_SHARED_PROJECTION_GAUGE_V1_RESULT.json),
[one-map score-space correction](../../polynomial_causal/EQUALITY_L5H5_M4_SHARED_PROJECTION_CORRECTION_V1_RESULT.json),
[corrected shared-kernel behavioral null](../../polynomial_causal/EQUALITY_L5H5_M4_SHARED_KERNEL_BEHAVIOR_V1_RESULT.json),
[behavior-selected native-projection null](../../polynomial_causal/EQUALITY_L5H5_M4_BEHAVIORAL_PROJECTION_SELECTION_V1_RESULT.json),
[four-port Möbius low-order null](../../polynomial_causal/EQUALITY_L5H5_M4_PORT_MOBIUS_ORDER_V1_RESULT.json),
[invalid three-node real-arithmetic factor graph](../../polynomial_causal/EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V1_RESULT.json),
[precision-corrected four-node factor graph](../../polynomial_causal/EQUALITY_L5H5_M4_TWO_FACTOR_CORRECTION_GRAPH_V2_RESULT.json),
[precision-factor graph export](../../polynomial_causal/EQUALITY_L5H5_M4_PRECISION_FACTOR_GRAPH_EXPORT_V1_RESULT.json),
[raw-port factor graph](../../polynomial_causal/EQUALITY_L5H5_M4_RAW_PORT_FACTOR_GRAPH_V1_RESULT.json),
[raw-port factor graph export](../../polynomial_causal/EQUALITY_L5H5_M4_RAW_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json),
[residual-port factor graph](../../polynomial_causal/EQUALITY_L5H5_M4_RESIDUAL_PORT_FACTOR_GRAPH_V1_RESULT.json),
[residual-port factor graph export](../../polynomial_causal/EQUALITY_L5H5_M4_RESIDUAL_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json),
[invalid product-port storage-rank receipt](../../polynomial_causal/EQUALITY_L5H5_M4_PRODUCT_PORT_FACTOR_GRAPH_V1_INVALID_STORAGE_RANK_RESULT.json),
[product-port factor graph](../../polynomial_causal/EQUALITY_L5H5_M4_PRODUCT_PORT_FACTOR_GRAPH_V1_RESULT.json),
[product-port factor graph export](../../polynomial_causal/EQUALITY_L5H5_M4_PRODUCT_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json),
[normalized-input factor graph](../../polynomial_causal/EQUALITY_L5H5_M4_INPUT_PORT_FACTOR_GRAPH_V1_RESULT.json),
[normalized-input factor graph export](../../polynomial_causal/EQUALITY_L5H5_M4_INPUT_PORT_FACTOR_GRAPH_EXPORT_V1_RESULT.json),
[reader-weighted CP-rank obstruction](../../polynomial_causal/EQUALITY_L5H5_M4_READER_WEIGHTED_CP_RANK_V1_RESULT.json),
[moment-weighted CP-rank context null](../../polynomial_causal/EQUALITY_L5H5_M4_MOMENT_WEIGHTED_CP_RANK_V1_RESULT.json),
[typed query/key moment null](../../polynomial_causal/EQUALITY_L5H5_M4_TYPED_MOMENT_CP_RANK_V1_RESULT.json),
[invalid quadratic-gate capture receipt](../../polynomial_causal/EQUALITY_L5H5_M4_QUADRATIC_CONTEXT_GATE_V1_INVALID_CAPTURE_RESULT.json),
[label-free quadratic context-gate null](../../polynomial_causal/EQUALITY_L5H5_M4_QUADRATIC_CONTEXT_GATE_V1_RESULT.json),
[red-team note](../../polynomial_causal/EQUALITY_A8_SPARSE_EDGE_EXECUTION_NOTE.md).

The retained three-block score also has a smaller exact executor:

$$
B(D,D)+B(D,R)+B(R,D)=B(D+R,D+R)-B(R,R).
$$

Its $2\times2$ group coefficient matrix has rank two, proving that 256 scalar
multiplications per query-key cell are sufficient and necessary under independent
group inputs, versus 384 for three separate width-128 dots. When the native parent
score is already live, only the $R\times R$ dot is incremental. This is an exact
score-level saving with supplied denominators; it does not simplify QK2, values,
source generation, or the unresolved suffix. See the
[rank control](../../polynomial_causal/THREE_BLOCK_BILINEAR_RANK_CONTROL_20260915_0549_RESULT.json).

## `module.attention.8`

Aliases: attention block 8, attn8. Related route:
[`PATH-SET2-001`](../COMPUTATION_PATH_REGISTRY.md).

Attention8 enters the task-matched head9.8 route through QK source interactions
as well as the separately established head8.2 value-transport circuit. Terms
containing the full attention8 write aggregate to `.20479` of the QK1 folded
change and `.36420` of QK2. In QK2, carry-query × attention8-key is `.16233`,
while attention8-query × carry-key is `.14170`. These magnitudes do not identify
the responsible attention8 head or equate its routing and value roles.

## `module.mlp.8`

Aliases: MLP8, layer-8 MLP. Related route:
[`PATH-SET2-001`](../COMPUTATION_PATH_REGISTRY.md).

MLP8-query × carry-key contributes `.20754` in QK1 and `.23088` in QK2.
Carry-query × MLP8-key contributes `.10556` and `.14406`, respectively, while
MLP8 self-interactions are only `.02357/.03635`. The task-matched path therefore
uses MLP8 mainly in cross terms with the incoming carry.

## `module.mlp.16`

Aliases: MLP16, layer-16 MLP. Related route:
[`PATH-SET2-001`](../COMPUTATION_PATH_REGISTRY.md).

### 2026-09-15 downstream cross-term role

On task-matched regional prefixes, the interaction between MLP16's propagated
write and all earlier residual sources is the largest MLP17/unembedding term
(`.6184` normalized cue-change norm ratio), with at least `.531` in every family.
Within that interaction, attention9 is the largest individual upstream source
(`.26094`), but many later MLP and attention sources remain material. This is
exact path magnitude without selective intervention, so MLP16 is a path endpoint
candidate rather than an identified semantic module.

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

### 2026-09-15 regional QK1 path overlap

An independently backward-folded UK/US reader path places attention5 inside the
dominant head9.8 QK1 carry×carry branch. Attention5-query × attention5-key is
`.03830` of that parent change; MLP6-query × attention5-key leads all 289 pairs
at `.04839`, and attention5 participates in seven of the top seventeen. Summed
aligned fractions are `.22039` on the query side and `.19886` on the key side.
This is exact path attribution with fixed denominators and value, not evidence
that the attention5 induction/copy function itself implements regional spelling.
The next grouped test must preserve this distinction.

That fixed grouping test assigns `.45480` to late×late, `.25335` to late-query ×
remainder-key, `.19041` to remainder-query × late-key, and `.11186` to
remainder×remainder. Late×late leads in every family. Because these are norms of
correlated change vectors, they are interaction magnitudes rather than additive
percentages or causal recovery scores.

### 2026-09-14 — input-dependent mean deviation is not the gross-value head pair

The disjoint-document [head split](../../polynomial_causal/ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V2_RESULT.json)
preserves the earlier fixed-write account while separating its valuable variable
correction. Replacing attention5 by its fit-set mean costs.12845 nats. Restoring
only heads5/7's deviations recovers42.51% of that damage, while restoring the
other seven recovers84.56%; the latter ordering holds in both fixed evaluation
halves. Singleton recoveries rank heads6,7,3 first (.3510/.3451/.2895), while
head5 ranks seventh (.0964). Thus the heads carrying most gross attention5 value
are not the complete source of the input-dependent correction. Registered C/D/E
fail. Registered A also remains failed because the head-sum tensor check is
.001953 against a too-tight.001 absolute bar, although all-head restoration
matches native CE within6.96e-8 and the matched manual/model CE check is exact.
The frozen6/7/3 candidate must transfer to untouched natural and code corpora
before any within-head routing/value/source factorization.

That [prospective transfer](../../polynomial_causal/ATTENTION5_DEVIATION_LEADERS_TRANSFER_V1_RESULT.json)
passes closure, live-correction, and leader sufficiency, but rejects an exclusive
three-head circuit. Heads6/7/3 recover71.63% of mean-write damage on untouched
natural documents and58.92% on code. The complementary six also recover63.20%
and64.81%, failing their50% insufficiency bar; on code the complement wins in
both fixed halves. Each leader remains individually positive on both corpora,
with head3 falling from23.56% natural recovery to3.60% code. The correction is
therefore overlapping and corpus-sensitive at native-head grain. No within-head
factorization is licensed; the next test must ask whether all nine interventions
share a lower-dimensional downstream response quotient.

The [all-nine response quotient](../../polynomial_causal/ATTENTION5_DEVIATION_RESPONSE_QUOTIENT_V1_RESULT.json)
is local to natural text and fails code transfer. Three natural-text response
coordinates explain87.41% of the nine corpus-mean final-vocabulary responses,
but that frozen subspace reconstructs code with86.75% relative error. Median
same-head natural/code response cosine is.221. Separately measured singleton
responses miss the joint all-head response by73.42% on natural text and49.28%
on code, despite all nine singleton deviations improving CE on both corpora.
The mean-deviation correction is real, distributed, nonlinear downstream, and
corpus-conditioned at this broad endpoint. Registered C/D/E fail. Further
factorization now requires a task-defined causal endpoint; another document-CE,
rank, or head-set search would not identify a stable circuit.

The first such [task-defined split](../../polynomial_causal/ATTENTION5_COPY_CONTENT_DEVIATION_SPLIT_V1_RESULT.json)
also rejects the proposed head roles. On final-natural copy-positive cells the
mean-write damage is only.0185 nat, below the registered.05 live-effect gate;
heads5/7 have negative recovery while heads6/7/3 over-recover. On code positives
the mean damage is.411 nat and heads5/7 recover67.60%, but code matched negatives
have negative mean damage and reverse the comparison. All-head restoration and
manual/model checks pass. The frozen terminal-copy endpoint therefore does not
support a corpus-stable copy/content split of these mean deviations. Preserve
this null; do not tune masks, groups, ranks, or denominators around it.

## `module.mlp.5_to_7`

Aliases: late pre-block-8 MLP carry group. Related folded route:
[`PATH-SET2-001`](../COMPUTATION_PATH_REGISTRY.md).

### 2026-09-15 interaction with attention5 in regional QK1

MLP5/6/7 participate heavily in the 289-term expansion of head9.8 QK1's
carry×carry path. Query-side aligned fractions are `.12010/.20707/.15755`; key-side
fractions are `.12960/.16501/.14781`. The largest individual terms include
MLP6×attention5 `.04839`, MLP7×attention5 `.03788`, attention5×MLP6 `.03734`,
MLP6 self `.03252`, and MLP6×MLP7 `.03094`. Individual terms are small and the
top-ten replay is poor, so these modules remain a candidate interaction group
rather than three independent semantic components.

Together with attention5 as a frozen late group, these sources account for the
leading self-block (`.45480`) and participate in two substantial ordered boundary
blocks (`.25335/.19041`). The three late-touching blocks replay the exact parent
to `.11186` relative error. The next test should edit this grouped QK1 routing on
fresh rows rather than subdivide the selected rows again.

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

### Cumulative depth-band producer, 14 September 20:04 UTC

The fixed blocks7–16 plus attention17 band reproduces the actual-token MLP17
context gate on A1/A2 with transfer .9751/1.0030 and relative error
.0412/.0542. Its final pre-RMS second-moment transfer is .9460/1.0548 with
error .0795/.1081. The cumulative target-gate transfer rises from .4047/.2181
through block7 to .8445/.7735 through block11 and .9294/.9304 through block13.
This passes the registered distributed-band, onset and norm-state tests while
using21 complete output-vector ports rather than35 for the full causal prefix.

Selectivity fails: the same band reproduces the G runs/run reader's gate but
changes correct-token CE by mean absolute .7465 nats, above .15. Every arm
cyclically changes the preceding lexical verb while holding the grammatical
frame fixed; it does not swap G's he/they cue. Treat the band as a conditional
broad lexical-context interface, not a gerund-specific producer or a semantic
grammar state. Native weights, donor generation, other positions, MLP17 and suffix
remain external, so the40% port reduction is not static model compression.
V1's gate verdict is withdrawn because it rolled the row-specific reader with
the state; V2 recomputes the preregistered fixed-recipient-reader metric from
the immutable native states with zero new forwards. Primary receipts:
`GERUND_DEPTH_INTERVAL_V1_RESULT.json`,
`GERUND_DEPTH_INTERVAL_V2_AMENDMENT.md`, and
`GERUND_DEPTH_INTERVAL_V2_RESULT.json`.

The held-out-operator R test uses16 different bare/-ing pairs in the can/are
frame. The21-site band7 interface transfers again: gate transfer.9702 with
error.0688 and norm-moment transfer.9456 with error.1431. The registered onset
predicate fails because prefix7 reaches.5115, just above its fixed.50 upper
bar; prefix11/13 reach.8306/.8869. The prospectively promoted13-site band11
also fails: its gate transfer.7241 is below.75, although gate error.3268 and
norm transfer/error.8212/.4082 pass their individual bars. Preserve both
failures and retain21 sites as the smallest validated fixed band. This is
operator/lexicon transfer on previously used rows, not new-text or pretraining
OOD evidence. Receipt: `GERUND_DEPTH_INTERVAL_LEXICON_TRANSFER_V1_RESULT.json`.

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

A new score-blind 48-row panel directly composes that head8.2-induced
current-value input with the independently folded late-touching head9.8 QK1
route. Both singles are live in both templates. The exact head-space
routing×value term is 59.13%/61.94% of the smaller single, and the recursive
UK/US target difference-of-differences is 56.93%/61.03% of the smaller single.
The joint signed effects are −.798/−.923 logits, joint/additive cosines exceed
.9996, and five unrelated-reader ratios are all below .288. This establishes a
material overlap at the bilinear head input while preserving the distinction
between the within-head routing component and the cross-layer value producer.
The tested value change is allowed through the full native head9.8 routing at
framing sources; the earlier O-only result identifies its origin but is a
narrower projected subcomponent. [Preregistration](../../polynomial_causal/SETTING2_REGIONAL_QK1_VALUE_COMPOSITION_FRESH_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_qk1_value_composition_fresh_v1_result.json).

The five-arm attribution separates this explicit head product from nonlinear
suffix curvature. Joint minus an additive-without-cross head intervention
matches the ordinary recursive factorial interaction with cosine
`.99930/.99891` and relative error `.16533/.09669`. The explicit head-cross
effects have RMS `.28072/.37577` logits versus factorial interaction RMS
`.24192/.34678`. Thus the head9.8 product determines the interaction direction
and most of its magnitude; the native suffix attenuates it modestly. This panel
was already opened by the four-arm result, so it strengthens mechanism
attribution rather than adding fresh transfer evidence. [Preregistration](../../polynomial_causal/SETTING2_REGIONAL_QK1_VALUE_INTERACTION_ATTRIBUTION_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_qk1_value_interaction_attribution_v1_result.json).

The exact cross term also separates over framing source tokens. On the opened
panel, four relative-to-quote offsets replay its final-query vector with
`.21902/.10950` error and transfer to the other template with
`.14374/.28944` error. The common leading offsets are the quote, colon, and
offset −7; the fourth is −9/−10 by family. Because −7 and the fourth offset map
to different words across templates, this is positional/boundary compression,
not a stable lexical role. Instruction-side sums alone reach `.19058/.14979`
error, while description-side sums miss at `.81713/.85442`. These rankings are
discovery on already opened rows and need a new recursive subset intervention.
[V2 result](fast_screens/setting2_regional_head_cross_source_census_v2_result.json),
[V2 correction](../../polynomial_causal/SETTING2_REGIONAL_HEAD_CROSS_SOURCE_CENSUS_V2_CORRECTION.md).

Fresh recursive testing preserves the instruction-source compression but rejects
selective adoption. On two new templates and four new cities, instruction-only
cross vectors replay the full cross at `.18909/.23507` error and recursive
paired-logit cross effects at `.24669/.32261`, with cosines
`.99782/.99803`. Native capability and cross liveness pass. The second template
fails the unrelated-reader gate: work/jobs changes at `1.15068×` the full-joint
target effect, above the frozen `.75` bar; the first template's maximum is
`.19368`. Record this as a transferring causal interaction subset and a valid
selectivity null, not an adopted circuit. [Preregistration](../../polynomial_causal/SETTING2_REGIONAL_INSTRUCTION_CROSS_FRESH_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_instruction_cross_fresh_v1_result.json).

Subtraction against the additive-without-cross arm localizes that failure away
from the interaction component. All instruction-cross-specific control ratios
are below `.09658`, and all full-cross-specific ratios are below `.09212`. In
the failed family, additive routing/value branches have `1.00364×` the
full-joint work/jobs RMS, so they account for essentially all observed spillover.
The full combined package remains failed, while the instruction-source cross is
a fresh, causally transferring, cross-specifically selective component.
[Attribution preregistration](../../polynomial_causal/SETTING2_REGIONAL_INSTRUCTION_CROSS_COLLATERAL_ATTRIBUTION_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_instruction_cross_collateral_attribution_v1_result.json).

The additive split assigns that work/jobs effect to the routing singleton. In
the failed family, routing/additive/value work/jobs RMS is
`.52600/.53042/.03269`; routing reproduces additive at `.07353` relative error.
Both singles remain useful for the target (`.29078` routing and `.36474` value
RMS), and their sum reproduces the additive target within `.09700`. Across all
readers and families, recursive single-effect sums replay additive within
`.11131`. Preserve the value and cross branches; the next circuit split should
separate late-late, late-remainder, and remainder-late QK1 routing blocks for
target versus work/jobs. [Preregistration](../../polynomial_causal/SETTING2_REGIONAL_ADDITIVE_BRANCH_COLLATERAL_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_additive_branch_collateral_v1_result.json).

The exact DD/DR/RD routing split rejects a singleton explanation of that
collateral. In failed family 1, target/work-jobs RMS is `.16460/.23036` for DD,
`.06502/.19958` for DR, and `.01019/.13165` for RD. No block reaches the frozen
full-control replay or 30% selectivity-improvement gates; the ordered cross
blocks are especially control-heavy. Separate effects still sum to full routing
within `.20641` target and `.08101` work/jobs error. Keep the distributed
three-block representation and close one-block pruning on this panel.
[Preregistration](../../polynomial_causal/SETTING2_REGIONAL_QK1_BLOCK_COLLATERAL_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_qk1_block_collateral_v1_result.json).

A prospective 48-row screen closes the three natural structured unions as
selective reusable QK1 units. Native capability passed on all 24 cue pairs, the
complete route remained live, and exact query-late/key-late route replay errors
were below `2.38e-7`. In family 0, query-late, key-late, and cross-only retained
`.8041`, `.6711`, and `.5210` of complete target RMS, but changed the
work/jobs-to-target ratio by only `+3.7%`, `-6.9%`, and `-8.9%`, versus the
required 30% improvement. In family 1, query-late improved that ratio by
`49.7%`, but red/blue and cat/dog effects were `9.66x` and `6.16x` its target
RMS. No union passed both families. Retain the complete distributed
`DD+DR+RD` group; stop query-side, key-side, and cross-only pruning on this
route. [Preregistration](../../polynomial_causal/SETTING2_REGIONAL_QK1_STRUCTURED_UNION_FRESH_V1_PREREGISTRATION.md),
[result](fast_screens/setting2_regional_qk1_structured_union_fresh_v1_result.json).

The same edge transfers to a score-blind cached corpus panel. On FineWeb and
Pile/reference, inherited-edge/full-city errors are19.48% and16.91%, effect
cosines are.9963 and.9948, and all63 native-capable pairs move toward the donor
spelling convention. Mean edge effects are6.15% and9.16% of native paired cue;
the largest unrelated-control ratio is.176. This adds natural-fragment transfer
and stable operational identification, while retaining the explicit caveat that
the cached rows are not proven disjoint from pretraining or near duplicates.

Pair-centering the edge's city-value contrast on the same corpus panel removes
3.07%/4.58% of native FineWeb/Pile-reference paired-cue magnitude and attenuates
all63 capable pairs. Behavioral half scaling is accurate within.054%/.028%, self
is an exact no-op, and controls stay below.176. This supports selective necessity
of the narrow edge contrast. It leaves all other city paths and head services
native and does not overturn the broader producer-pair newline collateral.

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

The failed semantic destination grouping now has a stable operational quotient.
Across the two discovery families, individual destination interventions produce
a six-endpoint response matrix with99.94% rank-one energy. Its frozen response
direction transfers to both held-out families at cosine.972/.970 and24% relative
reconstruction error. Individual destination effects compose to the complete
edge within1.29%, with all unrelated-control ratios below.179. Thus the invariant
is a shared causal response direction scaled by destination-specific gains, not
a description/instruction label. Native state generators and suffix remain
conditional dependencies, and the response direction has no semantic name.

### head8.2 → head9.8-O typed face — 17 September

The fixed complete routing/inherited face now has an isolated two-state-input
head8.2 write executor and a four-context fresh midpoint screen against16same-head
equal-norm nulls. Target/median null12.09; all16beaten; collateral limits pass.
Independent-piece composition remains failed; full-readout strict replay remains
failed. Native upstream state and downstream odd-value/suffix remain external.
See [path registry](../COMPUTATION_PATH_REGISTRY.md#17-september-regional-typed-face-masks-145)
and [scoped report](../../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2110_regional_native_face.md).

### 17 September, 21:40: regional source transfer and composition specificity

[Fresh source transfer](../../polynomial_causal/TYPED_FACE_KEY_SOURCE_FRESH_V1_RESULT.json) rejects carry-only (cue errors .411–.763, gate .35); carry plus MLP7 passes (.104–.183, gate .20) on eight unused contexts with shifted city position. Four new city token IDs occur; endpoints unchanged. Both frozen head-key norms pass the .10 behavioral gate. No source-selectivity claim or port-count reduction. [Physical-write composition](../../polynomial_causal/TYPED_FACE_WRITE_COMPOSITION_V1_RESULT.json) is nearly additive but fails specificity: worst interaction is 1.91 times random-split median versus gate .50. This resolves the previously missing random-split comparison with a failure. The earlier two-input additive failure remains. [Current Logan report](../../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2140_regional_source_transfer.md) tracks all four properties and simplicity. Next donor MLP7 generator has [CPU algebra evidence](../../polynomial_causal/TYPED_FACE_MLP7_DONOR_V1_CPU_RESULT.json), but native replay and full weight pricing remain required.

**21:46 donor-boundary update:** [MLP7 donor package](../../polynomial_causal/extracted_circuits/odd_attention8h2_mlp7_donor_v1/README.md) passes native and isolated CPU replay on 16 opened sequences. Donor input now precedes MLP7; two native-state arrays remain. Storage rises to 16,836,739 floats. Exact extraction boundary progress, no compression or new fresh/selectivity/composition certification.

**21:52 source handoff:** [MLP7 input-source intervention](../../polynomial_causal/MLP7_INPUT_SOURCE_V1_RESULT.json) fails both residual6-only and residual6+attention7 omission hypotheses on eight opened contexts. Initial7 stays explicit; posthoc source5 is unconfirmed. [Local exact response](../../polynomial_causal/MLP7_EXACT_RESPONSE_V1_RESULT.json) preserves normalization, cross/quadratic terms and residual skip. Next circuit work tests structural transfer of the whole retained path.

**22:00 structural transfer:** [Full retained face](../../polynomial_causal/TYPED_FACE_STRUCTURE_V1_RESULT.json) passes all six registered gates across base/no-colon/no-quote/neither/short-frame. Four modified conditions fresh at freeze; four context/city cells per condition. Prediction17–20%, attenuation2.1–3.9%, beats16/16directions each, four collateral ratios<=.219. [Current report](../../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2200_regional_structure.md). Extraction still two native ports; composition specificity fails; source omissions remain rejected ([source failure](../../polynomial_causal/MLP7_INPUT_SOURCE_V1_RESULT.json), [local exact response](../../polynomial_causal/MLP7_EXACT_RESPONSE_V1_RESULT.json)). No broad OOD upgrade from posthoc predictor-null audit.

**22:07 prospective qualification:** [Fixed-baseline transfer](../../polynomial_causal/TYPED_FACE_PROSPECTIVE_V1_RESULT.json) passes prediction improvement on all five fresh construction families; constants and text-OLS remain frozen. Minimum attenuation fails for line breaks (.0191105<.02). All paired directions remain positive, matched-null target and collateral gates pass. This qualifies the preceding all-pass punctuation screen, without retracting it. [Magnitude audit](../../polynomial_causal/PROSPECTIVE_ATTENUATION_V1_AUDIT.json) distinguishes absolute damage and native gap; no downstream causal attribution yet.

**22:15 fixed-edit census:** [Response census](../../polynomial_causal/TYPED_FACE_RESPONSE_CENSUS_V1_RESULT.json) passes reconstruction and family-stability gates but rejects direct-only and fixed early-MLP sufficiency. Attention17 has13–20%aligned response; this is not new head17.2 causal evidence and prior source-support failures remain. [Exact head9 value-difference cancellation](../../polynomial_causal/ODD_VALUE_DELTA_RAW_V1_CPU_RESULT.json) removes inherited-first input algebraically and combines reentry states; native verification is pending. [Current report](../../polynomial_causal/explanations/for_logan/research_update_2026-09-17_2215_regional_response.md).

**22:18 reduced head9 boundary:** [Export](../../polynomial_causal/extracted_circuits/odd_value_delta_raw_v1/README.md) passes native and isolated CPU replay on40opened fixtures,901,121stored scalars. No first-value or separate initial-state input; raw9 and upstream delta remain. Wider head8/head9 boundary would require three native-state arrays. No fresh scientific gate, composition or whole-model simplicity promotion.

**22:25 combined boundary verified:** [Complete conditional write](../../polynomial_causal/extracted_circuits/typed_face_composed_raw_v1/README.md) passes native and isolated40fixture replay. Three native-state inputs,1,788,419floats; no external first-values, initial-state or delta8 input. Native suffix remains; implementation composition is not causal composition certification. Earlier removal and random-split failures remain.

**22:33 destination partition null:** [Framing/clause split](../../polynomial_causal/DESTINATION_PARTITION_V1_RESULT.json) has two live, nearly additive pieces but fails specificity1.0969xrandommedian,beats1/9. Stop arbitrary repartition search. Next [native8 scope registration](../../polynomial_causal/TYPED_FACE_NATIVE8_SCOPE_V1_PREREGISTRATION.md) tests the same write at actual attention8 output against the conditional odd-value operator; implementation begun, no result yet.

### 17 September 23:41 — Regional MLP8 normalization screen

Both registered full-context normalization approximations pass opened causal gates (200 forwards, 3.45s). Frozen norm effect error .00031–.00072; no fresh/null upgrade. [Receipt](../../polynomial_causal/REDUCED_MLP8_NORMALIZATION_V1_RESULT.json). Next one-head normalization candidate has 2.1–6.4% CPU local error, not a causal result: [diagnostic](../../polynomial_causal/SINGLE_HEAD_NORMALIZATION_V1_CPU_RESULT.json), [registered screen](../../polynomial_causal/SINGLE_HEAD_NORMALIZATION_V1_PREREGISTRATION.md). Exact package and composition failures preserved.

### 17 September23:48 — Single-head normalization fresh confirmation

[Fresh result](../../polynomial_causal/SINGLE_HEAD_FRESH_V1_RESULT.json) passes all6gates:1.1–15%prediction error,16–22%attenuation,16/16nulls,collateral<=.163. Same endpoints and2native inputs; composition unchanged. [Pruned package CPU check](../../polynomial_causal/SINGLE_HEAD_PRUNED_V1_CPU_RESULT.json):16,899,587floats,70tokens,40fixture replay pass; native/layout certification pending. This is a normalization approximation, not exact source elimination.

**23:54 certification and next source:** Pruned single-head package passes [native](../../polynomial_causal/SINGLE_HEAD_PRUNED_V1_RESULT.json) and [isolated](../../polynomial_causal/SINGLE_HEAD_PRUNED_V1_STANDALONE_RESULT.json) replay. Next [donor-free inherited-city removal protocol](../../polynomial_causal/CITY_INHERITED_REMOVAL_V1_PREREGISTRATION.md) changes the counterfactual; one native input and7.2–13%local CPU response error, causal effect untested. No transferred swap/selectivity claims.

### 18 September00:07 — Regional inherited-only sign failure and full-city factorial

[New-endpoint removal](../../polynomial_causal/CITY_REMOVAL_ENDPOINT_FRESH_V1_RESULT.json) predicts effects andpassescontrols/nulls butfails direction75%<90%plain-note. Exactnative removal sharesreversals; [cell diagnostic](../../polynomial_causal/CITY_REMOVAL_ENDPOINT_V1_SIGN_DIAGNOSTIC.json). [Native current/inherited factorial](../../polynomial_causal/CITY_VALUE_BRANCH_FACTORIAL_V1_RESULT.json) restores120/120positivefullcityremovals,8.4–14%attenuation; interaction.369>.35copyfamily fails. Keep fullcitycoupled; do notpromote independentpieces. [Nextscreen](../../polynomial_causal/CITY_FULL_VALUE_REMOVAL_V1_PREREGISTRATION.md), [latestreport](../../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0003_removal_sign_failure.md). Paired extractioncertified; removalone-inputpackage/freshfullcity/compositionremainopen.

### 18 September00:18 — One-input full-city removal extraction

[Fresh full-city result](../../polynomial_causal/CITY_FULL_FRESH_V1_RESULT.json) passes all6gates:2.2–13%predictionerror,10–13%attenuation,120/120positive,16/16nulls,collateral<=.167. [Native export](../../polynomial_causal/CITY_FULL_EXTRACTED_V1_RESULT.json) and [isolated export](../../polynomial_causal/CITY_FULL_EXTRACTED_V1_STANDALONE_RESULT.json) pass;16,877,827floats,53tokens,1native residual7 array. Inherited-onlyfreshsignfailure andbranchinteraction.369>.35remain. Naturalcorpusconfirmation next; no corpus/token-only/composition completeness claim.

### 2026-09-18 — Regional full-city removal: filtered Pile transfer
`CITY_FULL_PILE_V3_RESULT.json` passes all seven registered gates on 20 source documents, with unchanged candidate/null outputs relative to V2. V2 analytic-reference precision failure remains in `CITY_FULL_PILE_V2_RESULT.json`; V3 is an opened FP32 reference repair. Effect error .02364 aggregate, .06596 untouched natural arms, .02114 city substitutions. CPU document diagnostic finds six shared native/candidate reversals and two additional approximation sign errors. One native residual7 array and external suffix remain; inherited/current independent composition still fails. Canonical report: `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0033_pile_transfer.md`. No complete-circuit or matched-effect simplicity promotion.

### 2026-09-18 — Full-strength regional removal and block7 city-source fold
`CITY_FULL_STRENGTH_V1` passes all8gates (2.35%effecterror,30.46%attenuation,16nulls); opened matched-suffix `CITY_FULL_QUADRATIC_V1` supports retaining the quadratic term for accuracy (full2.35%,secant7.30%,linear13.60%; all broad screens pass). `CITY_SOURCE7_V1` captures four upstream city-state sources; 80ordered K1*K2*V terms replay nativewrite within1.03e-6relative. `CITY_SOURCE7_CONTRAST_V1_CPU_RESULT.json`: MLP7-present paired aligned.3068/norm.3228; attention7-present aligned.1432/norm.1644, overlapping groups. Native query/normalization dependencies remain. Source attribution nominates a source-specific edit, not module-removal equivalence or port closure. for_logan/0033 remains canonical; extraction/composition limitations unchanged.

### 2026-09-18 — MLP7 reader input expansion (CPU; native checks pending)
`CITY_MLP7_INPUT_TERMS_V1_RESULT.json`: nine ordered residual6/initial7/attention7 products reproduce the384cityreaders; maximum nativewriteerror1.06e-6. Self-only omission causes47–48%pairedreadererror and10.90%aggregatecitywriteerror. AllfourCPU predictionspass. Residual6 source recoveredby subtraction, queries/keyRMS/mixed8RMS native; no causal omission or portclosure claim. New city_mlp7_integrated_v2 shared reader-value helper matches frozenV1 exactly; V1 andbindings preserved. Native reader andinstalled checks remain queued behind liveatlas. Canonical report for_logan/research_update_2026-09-18_0058_mlp7_readers.md.

### 2026-09-18 — Normalization information limit of three-reader MLP7 fold
`CITY_MLP7_NORM_INFORMATION_V1_MATH.md` andCPUresult: a hidden direction in ker(WD) has ||Dv||²=7.8413, so complete output norm does not universally factor through the384readings alone. Scope arbitraryhiddenh, reachability by native z/text unproved. Do not treat12.39Mreader storage as full-executor savings: retainingD/b for an exact localnorm costs17.70Mtotal versus16.37Munfolded. DirectGram21.23M alone. This is a priced simple alternative, not universal algorithmic lowerbound. Native queries/RMS/context stay explicit; GPUcertificates stillqueued.

### 2026-09-18 — Earlier residual6 city-reader export, CPU certified
`extracted_circuits/city_reader_residual6_v1` generates all9attention7heads and foldedMLP7cityreaders from residual6prefix+tokens, retaining nativehead8queries andmixed8cityRMS. `CITY_EARLIER_BOUNDARY_CPU_V1_RESULT.json` all3pass,80sequence-equivalent fullforwards21.027s; targeteffecterror5.37e-6,controls<=7.33e-5 versusGPUreference. `CITY_READER_RESIDUAL6_V1_ISOLATED_RESULT.json` 40fixturestandalonepasses.21,851,526FP32values;23,169nativeinputscalars. Earlierdependency, higherweight/stateprice; not totalcompression, token-only, newfresh or independentcomposition. NativeMLP8/suffix external; olderGPUcertificates remainqueued. Latestfor_logan index points to newboundaryreport.

### 2026-09-18 — CITY_RESIDUAL6_SINGLE_INPUT_V1: one-array approximate extraction
Full attention7 at every position plus five folded MLP7 readers now generate complete head8.2 city removal from residual6 + tokens alone. No supplied query or RMS; mixed8 RMS is approximated with other-source norm, all numerator terms retained. Opened CPU full-suffix screen passes a–e: effect error.012430, controls<=.079730,102/111positive,16/16nulls beaten (38.42×median). Isolated40fixture generator replay passes.23,326,342FP32weights,36,864native floats atT32: larger than previous interface. Extraction and opened selectivity advance; fresh/OOD pending, independent composition remains failed0/16random splits, matched-effect simplicity untested. Native blocks0–6 and MLP8/suffix remain external.
Report: ../../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0143_single_residual6.md. Program: ../../polynomial_causal/extracted_circuits/city_residual6_single_input_v1/manifest.json. Installed receipt: ../../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_CPU_V1_RESULT.json. Fresh continuation: CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1 rows/tables frozen outcome-blind; no fresh behavioral result yet. Older queued GPU checks retain their original boundary/scope.

### 2026-09-18 — Fresh residual6 conditional prediction/selectivity passes
CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1: all5registered gates pass on20new Pile documents/40sequences/240fixed probes; native capture+screen800CPU sequence-equivalents in50.068s. Effecterror.019047, untouched.011895/substituted.019853; controlratios<=.181133;114/120positive, mean.273865;16/16same-site nulls36.04xmedian. One document(context10) reverses all6contrasts in both native and generated interventions; retained. Fresh isolated40fixture execution exact, using the386-token weight-derived attention7 table; all other programweights/executable unchanged. Declared native residual6 input and MLP8/suffix remain external. Fresh same-corpus evidence, not domain-general OOD or token-only execution. Independent composition failed0/16random splits; matched-effect simplicity absent. Fresh price23,303,302FP32values and36,864nativeinputfloats.
Report: ../../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0152_fresh_residual6.md; receipt: ../../polynomial_causal/CITY_RESIDUAL6_SINGLE_INPUT_FRESH_V1_RESULT.json. Older CITY_MLP7_READERS_V1, CITY_MLP7_INTEGRATED_V1 and CITY_MLP7_GENERATED_NORM_EDIT_V1 GPU gates now all pass at their original boundaries; no scope borrowing. Next registered CITY_ATTENTION7_OMISSION_V1 local preflight passes; full/no/leave-one-head writes ready, no causal subset yet selected.

### 2026-09-18 — MLP7 city-source direction failure
Fresh native and generated MLP7-present terms agree on fourreversingdocuments; directiongatefails80%vs90%. Mixedproducts reproducesigns, butselfterms65%positive alsofail. Do not infer MLP7 ownership or a stable sourcecomponent from exactfolds. Canonical failure and receipts: ../CIRCUIT_REGISTRY.md and ../../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0208_source_direction_failure.md. Independentcomposition remainsfailed; nextstrengthdiagnosticusesopenedrows.

### 2026-09-18 — MLP7 source-role hypotheses rejected
Smallstrength signs persist. Key/value factorial fails cleanrole predictions:context11valuecounteractsrouting, valuepositive55.83%. Preserve module/source conditionality; no new identified component. Canonical record: ../CIRCUIT_REGISTRY.md; report ../../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0220_conditional_source_roles.md. Completecityinterface interchange preflightpasses, freshbehaviorpending; two residual6contexts explicitlypriced.

### 2026-09-18 — Head8.2 coupled city interchange
Freshkeys/fullvalueinterchangepasses whileMLP7source-directionfailuresremain. Prefixexporttwo-native-context boundary validated; notindependentcomposition. Canonical evidence ../CIRCUIT_REGISTRY.md and ../../polynomial_causal/explanations/for_logan/research_update_2026-09-18_0235_city_interchange.md. NativeRMS/querysuppliedclaim in installedreceipt is a copiedscopeerror; use linkedexplicitcorrection. Currentcompositionpreflight shows nontrivialheadcross, no behavioral verdict.

### 2026-09-18 — CITY_INTERCHANGE_COMPOSITION_V1: coupled operator retained
Opened complete-city key/value interchange screen passes replay but fails both composition gates: joint interaction/smaller=2.3448235>.35; headcross omission/joint=.4186297>.10; suffix interaction/smaller=1.0746265>.35. Both singles live. All20 document joint-interaction ratios fail. Fresh full-swap prediction/selectivity and two-prefix extraction remain correctly scoped; no independent composition or matched-effect simplicity promotion. Primary receipt: `basis_aligned/polynomial_causal/CITY_INTERCHANGE_COMPOSITION_V1_RESULT.json`; explanation: `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0245_city_composition.md`. Subsequent opened signed accounting finds partial cancellation (cosine−.67844); no new partition selected.

### 2026-09-18 — Packed attention7 drop3 fresh city-removal confirmation
CITY_ATTENTION7_DROP3_FRESH_V1 all5gatesPASS on20freshPile documents/40sequences/240probes: error.0188112<=.05;114/114capable positive;controls<=.123827;16/16nulls,30.891xmedian. Physically packed eight-head prefix, fiveMLP7readers and head8 factors:22,418,566FP32 values on386tokens, saving884,736 versus equal-vocabulary nine-head generator. Isolated40fixture replay exact. Native residual6 plus native suffix remain external; composition and matched-effect simplicity unresolved. Package `basis_aligned/polynomial_causal/extracted_circuits/city_attention7_drop3_v1`; primary result `CITY_ATTENTION7_DROP3_FRESH_V1_RESULT.json`; Logan update `research_update_2026-09-18_0252_packed_city_removal.md`. FineWeb transfer registered and row construction started; no outcome assumed.

### 2026-09-18 — FineWeb direction failure and receiving-value fold
CITY_ATTENTION7_DROP3_FINEWEB_V1 passesprediction(.018626),controls/nulls/isolation butFAILSdirection88/102<.90;nativeandgenerated14reversalsagree. CITY_DROP3_FINEWEB_INTERCHANGE_V1 opened swap likewiseFAILSdirection84/102;error.016716passes. Retain both failures, no cross-corpus selective promotion. CITY_FINEWEB_RESPONSE_V1:early/head9sufficiencyFAIL;head9.8 prominent but laterattentionnecessary. CITY_FINEWEB_HEAD9_FOLD_V1 exact7terms: routing-onlyFAIL(.8122/1.1343),deltaV aligned.7588/1.1075;triple small, no independent composition. Next registered MLP8→value-reader six-piece fold implemented with actual-weight algebra control, nativecapturepending. Full report/primary links: `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0304_fineweb_reversals.md`.

### 2026-09-18 — MLP8→head9.8 value reader: native fold and opened intervention
CITY_FINEWEB_MLP8_VALUE_FOLD_V1 native replayPASS(max2.64e-5), omitMLP8FAIL(2.007/.418),quadratic-negligibilityFAIL(.367/.349). CITY_FINEWEB_VALUE_MEDIATION_V1 all4basicgatesPASS: subtractMLP8 changes reversed target.8815×swap;quadratic.2586/.2656;direct.9649/.5683. This changes onlyhead9.8value, not nativeMLP8. Opened positive94/102 versus84/102, but reversedgroup control/target<=.707 warrants no uniform selectivity claim. Six-term helper11,354,114FP32values; native z8/edit/mixed9norm contexts explicit. Same-boundary16nulls registered; fresh/composition/simplicity unresolved. Report `basis_aligned/polynomial_causal/explanations/for_logan/research_update_2026-09-18_0317_value_mediation.md`; primary receipts have matching stems.

### 2026-09-18 — Fresh-confirmed reduced-interface MLP8 value mediator
CITY_FINEWEB_VALUE_NULL_V1 all3PASS,22.235xnullmedian. CITY_VALUE_MEDIATION_FRESH_V1 all5PASS:prediction7.14e-6;97/100positive;controls<=.139;16nulls21.547xmedian. Parent nativecityswap91/100positive,mean.5299→corrected.4326: not uniformly stronger. No original packed-prefix or wholeMLP8ablation claim. CITY_MLP8_VALUE_EXTRACTED_V1 eliminates unusedmixed9vectors exactly:11,206,658FP32values;nativez8+editedRMS36896scalarsT32,delta36864separate.40local/isolated and80installed-equivalent replaypass. Package `basis_aligned/polynomial_causal/extracted_circuits/city_mlp8_value_mediator_v1`; Logan `research_update_2026-09-18_0326_extracted_mediator.md`. Composition of direct/MLP8value paths registered, no independence or matched-effect simplicity promotion.

### 2026-09-20 — Subject-number conditional suffix, blocks11–17

Four-reader width8 response dictionary; full attention12–17 and sparse18/36 MLP pair cores. Fresh regular target/control transfer and CPU prepared-context extraction pass; native contexts/initial response remain open. Source-write per-cell preservation fails even with dense cores and source-expanded calibration; no five-property promotion. Full dynamic finite readers now pass native closure for calibration. Raw conditional artifact is not a new verified package in the generated graph inventory. [Dossier](../../polynomial_causal/JOINT_READER_SPARSE_RESPONSE_2026-09-20.md), [reuse audit](../../polynomial_causal/SOURCE_REUSE_CPU_AUDIT_2026-09-20.json), [current update](../../polynomial_causal/explanations/for_logan/research_update_2026-09-20_0726_sparse_subject_response.md), [math review](../../polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-20_0726.md).

- Subject source-reuse v680: final native-state oracle modal error<=0.326% for A/B; final output basis is sufficient on these opened cells, propagation remains failed. See [source diagnostic](../../polynomial_causal/SOURCE_REUSE_AND_FULL_READERS_2026-09-20.md).

- Subject sourceA: v685 native readout after fixed initial reconstruction fails modal5% (6.69%). v683 direct attention-to-MLP write also fails; see [full diagnostic](../../polynomial_causal/SOURCE_REUSE_AND_FULL_READERS_2026-09-20.md). Current decomposition remains unadopted for source reuse.

- Subject v686–688 [hybrid output predictor](../../polynomial_causal/SUBJECT_HYBRID_RESPONSE_2026-09-20.md): sparse nonlinear number plus baseline-gradient modal branches passes fresh48-row A/B prediction/capability/collateral gates. IndependentCPU replay2.08e-15 on24sourcecases. Native gradient/context/fullsource inputs remain external; no residual-state replacement or fullmodel adoption. Prior eight-state source-reuse failures remain valid.

- Subject v689–691 [selected modal derivative path](../../polynomial_causal/SUBJECT_MODAL_DERIVATIVE_PATH_2026-09-20.md): retain attention12 and all six token-local MLP derivatives. Prospective48-row longer-combination prediction/capability/collateral gates pass with uncorrected sparse number branch. Earlier simple path cuts fail; later attention derivatives omitted only in modal predictor. Number branch/native context generators still charged; no primal ablation/adoption claim.

- Subject v692–695 [signed and matched controls](../../polynomial_causal/SUBJECT_SIGNED_AND_MATCHED_CONTROLS_2026-09-20.md): negative-B/difference number prediction fails despite dense pairs and paired-block native composition. Post11 matched-random native specificity19.6/20.2x passes; prediction bars use source-effect denominator, random-relative errors remain55–60%. Semantic attractor-control rows prepared, native execution pending.

- Subject semantic evidence: [v696–699 dossier](../../polynomial_causal/SUBJECT_SEMANTIC_COMPOSITION_2026-09-20.md). Congruence changes the sign of attractor evidence; no subject-exclusive circuit claim. Reinforcing composition fidelity fails, chiefly through primitive prediction errors.

- [Semantic source ports and finite pair graph](../../polynomial_causal/SEMANTIC_SOURCE_PORT_CENSUS_2026-09-20.md): direct recurrence closes numerically but fails A sufficiency; early/middle writes pass signed-role screen. B all-pair effect reconstruction1.90% opened/1.99% new constructions; frozen two-pair prediction fails18.68%, and native capability failures prevent full transfer promotion. Full native computation charged; no compressed execution claim.

- [Semantic pair origin across block11](../../polynomial_causal/SEMANTIC_PAIR_BOUNDARY_2026-09-20.md): replay passes; both boundary-only and suffix-only dominance fail. Output reconstruction needs both contributions (13.32%/27.98% errors separately versus1.98% together). Next folded observable must retain transported local response and downstream curvature; no native capability repair or standalone extraction.

- [Conditional three-source quadratic observable](../../polynomial_causal/SEMANTIC_SOURCE_QUADRATIC_JET_2026-09-20.md): validated native derivatives predict unit/negative/mixed effects below10%, doubled edits fail37.16%. Signed rank-one baseline also passes first three arms but directions vary across contexts.192-context CPU coefficient extraction passes; native coefficient generation remains required. v1 precision-invalid and v2 missing-bias-invalid preserved.

- [Shared quadratic and coverage correction](../../polynomial_causal/SHARED_SOURCE_QUADRATIC_DICTIONARY_2026-09-20.md): original amplitude designrank2/6; six independent native settings validate fullquadratic4.72% but reject per-context rank1 15.74%. Sharedrank2 dictionaries also fail. Preserve restricted-arm successes; no shared semantic circuit identification.

- [Four-output source observable and selectivity](../../polynomial_causal/SEMANTIC_SOURCE_MULTIOBSERVABLE_2026-09-20.md): native replay/prediction pass (number9.10%,modal2.82%), native selectivity fails129/288cells. Local three-source leakage bound and clipped candidate expose a restricted-interface limitation; no global impossibility or circuit adoption.

- [Five-source null edit](../../polynomial_causal/FIVE_SOURCE_MODAL_NULL_2026-09-20.md): native collateral below4.39%, but target-strength/selection conjunction passes only1/32cells. [Exact normalized MLP source core](../../polynomial_causal/NORMALIZED_MLP_SOURCE_CORE_2026-09-20.md): weight-contracted numerator/shared RMS denominator and analytic local Hessian pass planted CPU replay/gauge/bias controls; captured native-context validation pending.

- [Native normalized MLP source-core validation](../../polynomial_causal/NORMALIZED_MLP_SOURCE_CORE_2026-09-20.md): exact local output/derivative andCPUexport replay pass; localMLP11curvature alone fails number53.11%/modal8.16%. Full-path source-Jacobian/adjoint capture implemented; planted chain-rule proof passes, native all-node closure pending.

- [Native source-Hessian decomposition](../../polynomial_causal/NATIVE_SOURCE_CURVATURE_DECOMPOSITION_2026-09-20.md): fifteen local terms close1.25e-15 relative. Broad curvature omissions fail; full fifteen-direction finite test passes4.70%number/1.39%modal. Early-layer-only A/B cross-curvature candidate passes6.82/2.00% on openeddata; material interaction errors canreach41.69%. Fresh validation and native-generator reduction remain pending.
