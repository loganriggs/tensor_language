# Specialist heads & the three-goal circuit loop (§1280-1328 arc)

The named single-head (and pair) specialists found by the concentration-screen method, and
where each stands on the user's three goals: **G1 extract** (pull the circuit out, run it
standalone), **G2 remove** (delete the important part, lose only the capability),
**G3 weights-read** (predict behavior from the weights, certify on natural text).
READ THIS BEFORE screening a new behavior or ablating any of these heads.

## The named inventory

| Circuit | Heads | Capability | Key numbers | §refs |
|---|---|---|---|---|
| Question | **10.5** | "?" after WH-openers; ALSO top comparative-refine carrier (§1345, drop +0.343 in the "than" kit — §1310 off-diagonal confirmed) | dmg 0.726 at "?", 5% at ".", 0.4% elsewhere; "question-specific" true at terminal grain only | §1284, §1313, §1345 |
| Comparative | **8.1** | "than" after non-adjacent comparative | 100.7% of L8's 1.64-nat damage; conc 276; criterion STREAM-computed (weights ratio 1.01); mark written by a02 band | §1303-1306 |
| Exclamation | **17.2 + 17.3** | "!" continuation | pair carries 91% (48.5/42.6); perfectly ADDITIVE half-heads (2.6%); criterion stream-computed | §1315-1320 |
| Stem matcher | **1.1 + 1.8** | inflection-variant induction | weights say STEM matcher; 78% of identical-support damage at 1569 natural variant positions (inside registered 30-80% band); control flat | §1307-1308 |
| Fetchers | 2.5, 3.8 (+8.3, 8.4 copy stations) | copy/induction fetch | see induction.md + §1207-1218 | — |
| Sink | 5.7 | position-0 constant | see attn-sink-5-7.md | — |

Circuit structural forms found so far: single OWNER (10.5, 8.1), CROWD (induction band),
additive PAIR (17.2/17.3), half-head pair (matchers). Criterion taxonomy (§1320, final):
all specialist criteria are STREAM-computed, not embedding-native — the front a02 band
writes the class marks the specialists fetch (§1286, §1306; the §1286 content/key
dissociation did NOT repeat for comparative, §1306 pred_c).

## Selectivity (G2 status, matrix-certified)

selectivity_matrix2.py (§1309-1310, disjoint masks, 1920 rows): the remove-circuit x
measure-behavior matrix is STRONGLY diagonal (diagonals 0.13-2.21, off-diag max 0.19) but
not absolute — replicated off-diagonal structure: matchers->successor 0.080,
question->than 0.171, fetchers->successor 0.037. Removal is surgical to first order;
quote the residue when claiming clean removal.

## Extraction ladder (G1 status)

| Rung | Grain | Ident recovery | Lesson | §ref |
|---|---|---|---|---|
| 1 | 7 circuit heads, mean rest | 19% | circuits = heads + upstream closure | §1311 |
| 2 | 33-head dependency closure | 30% | better, far from bar | §1312 |
| 3 | closure + **lambda-v1 route** through removed heads | **79%** | route grain is the lever; description length, not head count (user correction §1314) | §1316 |
| 4 | + printable stand-in code | axis transfers 38% | payload needs rms calibration | §1321 |
| 5 | rms-calibrated additive payload | negative | additive payload dead, 2 strikes | §1322 |

Route grain = keep each removed head's lambda*v1 term (block-0 broadcast, ~free in bits)
+ live patterns (window-foldable weights code, §1161-66), mean-replace fresh values only.
Shared-variable leak: the same broadcast carries the content pool, so elsewhere recovers
68% too (§1316 — expected, registered).

## Standing traps for this arc

- Small-n: comparative targets are rare (n=54 @ 960 rows; §1303-05 all flagged). Run 1920+.
- Additive payload injection into the stream is POISONED without per-position rms
  calibration (§1287 rule; §1321-22 two strikes).
- Concentration screens need BOTH controls (jitter + random) clean before a verdict
  (§1302 withheld; §1315 withheld; both later certified at more data).
- Label bugs: §1304's winner printed "10.1" for 8.1 (hardcoded L); check data rows.

## Comparative: G1+G2 measured (§1329)

comparative_extraction.py, route grain, n=110 targets: route alone recovers 43.7% of the
7.0-nat target gap (the shared-variable leak reaches specialist capabilities); route +
a02 band + 8.1 reaches 69.7% — but elsewhere-recovery 0.679 ~ target 0.697, so the kept
a02 band is the UNIVERSAL class-marker, not a comparative part. The two clean facts:
**8.1's within-extraction increment is 26x target-selective** (0.361 nats target vs 0.014
elsewhere — G2 verified surgically), and **8.1 without the band is worthless** (+0.025
over bare route) — the specialist is 100% conditional on its annotator, both directions
measured. G1 verdict: PARTIAL — 70% reached, kept description not capability-specific.

## Comparative rung 2: the annotator's service is QUERY-side (§1331)

Position-gating a02 at comparative (key-side) positions keeps only 27% of the band's
increment (+0.070 of +0.263 over route); a rank-1 mark direction carries ~55% of that
key-side piece (83-sample estimate, flagged). The dominant ~73% arrives query-side —
front-band processing at the prediction position that 8.1's q reads. BUT both gated
slices are PERFECTLY selective (elsewhere = route +-0.002): [route + comp-position
outputs + 8.1] is an honestly capability-specific 50%-extraction at zero elsewhere cost.
Next: comparative_query_side.py (key vs qry vs both gates; queued).

## Comparative circuit: CLOSED (§1333)

Final description: **[v1-route + a02 live inside two zero-bit token-defined windows
(comparative positions; positions with a comparative 2-20 back) + head 8.1]** = 0.659
target recovery — 85% of the whole-band arm's contribution — at elsewhere route+0.014.
Query-side dominates key-side 2.7:1 (+0.183 vs +0.068, mildly sub-additive at +0.222
joint). With §1329's removal results this circuit has: annotator windows, carrier head,
criterion sidedness, removal selectivity (26x), and description cost. First all-preds-TRUE
run of the thread. The band-arm anchor replicated 3x (0.694-0.697). Unchased tail: 15%
of band service from positions outside both windows.

**THE TEMPLATE** (for the next specialist): extraction at route grain + capability-window
gates on a02 + specialist head; arms full/ymean/route/band/key/qry/both; bars = band>=0.60,
qry-vs-key +-0.05, selectivity +-0.05 of route elsewhere.

## Post-template state (§1334-1345) — read CIRCUIT_REGISTRY.md for stages

- QUESTION closed at head grain (§1342): 16-head clause-gated kit + 10.5 = 0.641; the L4
  "crowd" is a TRIO by drop-cost (4.0/4.1/4.7) — add-grain masked it. Drop-grain is the
  kit instrument; add-grain only screens.
- COMPARATIVE closed 4-STAGE (§1344-45): annotate(a02, two zero-bit gates) -> fetch(8.1)
  -> refine({10.5, 12.8} + 11.7/11.6) -> readout = 0.76-0.78. Missing rest = generic
  mid-pool (L3-5, NOT kit material — elsewhere > target) + the UNPRICED MLP side
  (ATTENTION-ONLY convention: all 18 MLPs live in every extraction arm).
- ENTROPY-SETTER REFUTED for comparative (§1343, three ways: rank collapse = content
  missing; gain-freeze hurts; temperature s*=0.95). First gain-frozen leg of the template.
- CROSS-CIRCUIT MOONLIGHTING is standard, both directions: 0.3 + 1.1 serve the question
  kit (§1342); 10.5 is the TOP comparative-refine carrier (§1345 — §1310's off-diagonal
  seen causally). Kit costs across circuits are NOT additive; price shared heads once.
  Labels of the form "X-specific" hold at behavior grain, not service grain.
- 13.8 = **THE CLOSER** (§1356 rename): delimiter-general close-what's-open — owns every
  bracket subtype (§1341, shares 0.95-1.04) AND quote-closing (107.9% solo, neighbors
  net-negative). Kit: [route + depth>0 gate on a02 + 13.8] = 0.657 (§1346); a14 assist
  does not port (§1347); refine = L14-17 by construction (§1348). Union-kit test §1356+.

## Open

- Exclamation circuit (17.2+17.3): template not yet applied; L17 position — expect the
  route to carry much more.
- Capitalized: late-band shared (a15-17, no owner, §1339) — parked; band-gate extraction.
- The MLP side of every kit: unpriced. Bridge to modules/benchmark.md ladder (mlp1 table,
  top-MLP 16x16 context tables) is the exact-extraction frontier.
- Double-close "))" unevaluable on natural rows (n=4) — needs targeted corpus (§1341).

## CAPITALIZED COMMITTEE (S1397, extended S1411) — the anti-closer
THIRTEEN heads across a13-a17: {13.0, 13.5, 14.4, 14.6, 14.7, 15.3, 16.0, 16.3, 16.4,
16.5, 17.0, 17.1, 17.2} (16.3 confirmed S1418 at NR=1920), solo dCE .016-.035 each,
NO owner anywhere (top shares .38-.55).
Sharp edge: next head below the .015 bar sits at .004. a14 is the band's biggest layer
(.089), missed by the S1339 screen (looked only at a15-17). 17.2 (expressive-broad)
leads = its 3rd moonlight. The committee is the named unit. REMOVAL HANDLE RE-PRICED (S1412): all-12 = .603
target damage (super-additive 1.5x parts; a13-14 and a15-17 crews are redundant
implementations — either alone ~.20); else spillover 6.7%. S1398's .21 undercounted 3x. Unit pricing (S1398): committee-only = 71% of band, else 2% (surgical); rest-20 unselective generic. REMOVAL GOAL DONE (7/162 heads). Grain (S1399): a16 trio conjunctive,
a17 trio payload-dominant. Kit (S1400-01): committee is a REMOVAL HANDLE ONLY — construction
marginal .008-.029 (redundant with commons+upstream); ungated commons serves capitalized .813.

## Two inventories, closed form (S1404)
Removal handles != construction components, now certified at the flagship: 13.8 (97%
bracket owner by ablation) has NEGATIVE drop cost in the unified 28-head kit — closer
targets are served without it under co-residence. Only 10.5 (question) is a load-bearing
specialist in construction (.195); 8.1 marginal .048. Individual drop costs do not
compose (redundancy = shared resource): 14 individually-cheap heads collapse the kit
when dropped together.

## NEWLINE-SETTER CREW: {7.2, 8.2, 10.2, 11.0, 12.6} (S1415/S1418) — joint handle .617
8.2: newline-target damage +.1457, controls <= .0025 (cleanest first screen since the
closer). 11.0: +.0894, clean. Distinct from the old front-attention newline ROUTING
circuit (item 7) — these predict where newlines GO. Quote service (S1420): owner 13.8
(closes ~105%) + helper crew {10.6, 11.5} (joint .133 close / .105 open, 1.79x solo).
OPEN-quote = crew-served WITHOUT owner (3rd service pattern; "unserved" was an
owner-hunting artifact).
Crew structure (S1419): NO crisp sub-band split — near-additive ensemble (cross-split
synergy 1.15x vs capitalized's 1.5x). TWO-CREW MOTIF DEMOTED to "observed sometimes":
redundancy structure varies by service (capitalized = 2 strong crews; newline = 1 loose
crew; quotes = owner + helpers). Corpus emits newlines only as bare '\n'.
Committee-13 joint = .654 (S1419; 16.3's joint marginal +.077 = 5x its solo).


## 11 September: joint routing signature scope

The existing attention17.2/17.3 pair does not have globally proportional normalized
routing on the tested formal continuous inputs (query8, distinct sources7/0).
This is not a negative about task-conditioned sharing or partial input subspaces.
The self-position signature was separately corrected for tied query/key inputs.
See [scope-corrected audit](../../polynomial_causal/JOINT_ROUTER_SIGNATURE_NATIVE_V1_AUDIT.json)
and [method explanation](../../polynomial_causal/explanations/2026-09-11/explanation_2026-09-11_0022.md).


## 11 September 02:43 — joint routing/value source ports

The frozen compact MLP17 component's 17 OV source readers touch 7.06% of joint
QK1×QK2 numerator coefficient energy across nine heads and two distinct source
distances, versus 5.55% for matched random downstream frames. The registered
alignment and 17.2/17.3 bars failed. Separating current/base streams barely
changes this (7.10% versus 5.58%). This is a fixed-component numerator-space
result, not a negative about task-specific sharing.

An executed source-influence spectral calculation instead finds rank-17 source
spaces with 59.63% mean touch (45.7–70.2%), demonstrating that the choice of source
space matters. Mean inside energy is 14.49%, mixed energy 45.14%; outside readers
and full normalizers remain required. Within-head overlap across distances is
0.746–0.896, while the maximum across heads is 0.075 (mean squared principal
cosines). No new behavior or identified circuit; head17.4 enrichment was post hoc.

Primary receipts in polynomial_causal: JOINT_QK_VALUE_PORTS_V1_RESULT.json,
JOINT_QK_VALUE_STREAM_CLOSURE_V1_AUDIT.json, JOINT_QK_SOURCE_BOUND_V1_RESULT.json,
JOINT_QK_SOURCE_SPACE_COMPARISON_V1_AUDIT.json. Explanation0243 defines the metrics.


## 11 September 02:55 — exact conditional normalized source edits

For the frozen QK spectral source spaces, numerator-only source-removal accounting
has10.8–33.3% routing-effect error on formal probes. Recomputing key normalizers
is essential to the exact local formula. A compiled predictor now uses only
baseline score/norm/value ports plus source/query/norm projections; no edited
endpoint enters prediction. Six native-weight edit cases across all9heads and
both distances, including order and gauge checks, pass at6.42e-15. Price per
head/position:92 dynamic numbers,595 small-core numbers,97,920 projection-map
numbers plus native background. This is a conditional intervention tool, not a
semantic circuit or full-model simplification. The next position-shared source
influence method is implemented and CPU-controlled; native selection pending.
Receipts: NORMALIZED_QK_SOURCE_EDIT_V1_RESULT.json,
COMPILED_QK_SOURCE_EDIT_V1_RESULT.json,
JOINT_QK_POSITION_INFLUENCE_V1_CONTROL.json in polynomial_causal.


## 11 September 03:10 — source variables shared across positions

A weight-only common rank17 source space per attention17 head is highly stable
across odd/even distance splits (mean squared principal cosine>0.9998). Mean
validation numerator touch is41.64%:5.95%inside and35.69%mixed. Per-head coverage
34.9–49.7%; common/separate spectral retention77.6–92.0%. Registered all-head
retention/coverage targets failed; numerical and split-stability clauses passed.
Discovery-space bounds below45% for heads1/4/8 are scoped to discovery positions.
No claim that QK source variables form closed or semantically identified circuits.

A fixed adapter using the native common frames and raw query/key vectors passes
local normalized source-edit replay across five position pairs at4.69e-15. It
uses24,531additional stored numbers per head, plus native background, and92dynamic
prepared ports. This is not a whole-model saving. Next: validate frozen features
on FineWeb without refitting; no new data-guided discovery.
Receipts: POSITION_SHARED_QK_SOURCE_V1_RESULT.json,
POSITION_SHARED_QK_SOURCE_V1_REDTEAM.json, SHARED_POSITION_QK_EDIT_V1_RESULT.json.
Explanation0310 defines the metrics and retained interfaces.


## 11 September 03:38 — frozen source read-edge validation

Attention17 rank17 source frames, frozen from weights, predict source32 key/current-
value removal on32historical FineWeb rows; query127 endpoint. Learned logit-change
relativeerror0.0384% aggregated, CEpredictionMAE6.26e-7nats. MeanCEeffect+7.45e-5
is tiny. No-change passes absoluteCEbar but is753xworse inCEerror. Notsemantic,
selective, fresh/OOD or whole-model extraction evidence. Nativebackground retained.
All-source summation CPUcontrol5.71e-16 withqueryrolesfixed, including selfkey/value.
No nativeall-source result. See explanation0338 and FROZEN_QK_FINEWEB_V1_RESULT,
FROZEN_QK_FINEWEB_EFFECT_SIZE_V1_AUDIT, ALL_SOURCE_QK_EDIT_V1_CONTROL.


## 11 September04:06 — MLP16 producer functions behind QK/OV reads

Weight-only frozenfeaturefold AheldB/Cmissed: meanfunctionoverlap5.81%vsraw4.75%
androtatedcontrols5.33%. Noheadcos>=.95. Exactlocalbias/residual/x0/RMSreplay3.2e-15.
Fulljointkeyspace envelope raisesoptimal17featureoverlap33.63%, butmaximumcos
perhead.672–.843; noidentityat.95. The8positionroutingauditfinds QKtouch42.03%old
vs16.63%produceraligned. This is a selectiontradeoff, not semantic identification.
MLP16quadraticoutputcoefficientrank90=848; doesnotcontradictnaturalstate/CEdossier.
Newjointobjectivecontrolled,nativenotfit. Seeexplanation0406 and
MLP16_PRODUCER_OVERLAP_V1_RESULT, MLP16_PRODUCER_KEY_ENVELOPE_V1_AUDIT,
PRODUCER_COUPLED_QK_TOUCH_V1_AUDIT, COUPLED_PRODUCER_ROUTING_OBJECTIVE_V1_CONTROL.


## 12 September — weight-discovered quadratic source block in attention17

A cubic-source factorization exposes an approximately shared quadratic parent
with two linear-product children, predominantly written through head17.2.
On64cached FineWeb prefixes, simplifying this parent preserves its fitted block
writes within0.30–0.42%and its full-suffix logit removal effect within0.37%.
This is a frozen component preservation screen, not semantic identification;
its relation to this dossier's exclamation/capitalization roles is untested.
[Primary math and scope](../../polynomial_causal/SHARED_CUBIC_SOURCE_PROJECTION_V1_MATH.md),
[native receipt](../../polynomial_causal/COMMON_QUADRATIC_NATIVE_V1_RESULT.json).


The subsequent [regional-spelling screen](../../polynomial_causal/REGIONAL_SOURCE_BLOCK_V2_RESULT.json)
passes its registered basic bars when the frozen quadratic block is evaluated at
all causal source positions: removal reduces British/American cue effects by
13.1/18.2%across two templates; all16spelling contrasts move in the predicted
direction. Two-source coverage still misses. This adds a candidate within-head
role beyond the old exclamation/capitalization labels; it is not yet an identified
four-property circuit. New location cues and spelling pairs are frozen for testing.


The [unseen spelling/city-cue screen](../../polynomial_causal/REGIONAL_SOURCE_BLOCK_OOD_V1_RESULT.json)
also passes all4families:10.2–13.9%regional cue reduction,24/24positive contrasts,
with the frozen all-source block. This supports limited lexical/template transfer.
Which of the parent, children, query writers or gates carries the regional cue
remains unresolved; a four-port interchange test is registered.


[Factor interchange](../../polynomial_causal/REGIONAL_FACTOR_INTERCHANGE_OOD_V1_RESULT.json)
locates most paired regional-cue transfer in the block's two child-linear readings,
not its shared quadratic parent; this role passes all4held-out families. The
parent-dominance prediction on the original32prompts failed and remains recorded.
[Cross-start graph matching](../../polynomial_causal/REGIONAL_SOURCE_CROSS_START_GRAPH_V1_AUDIT.json)
finds the same parent/two-dimensional source span in the other weight fit; complete
coefficient-function errors are2.7%. Native cross-start intervention agreement and
upstream producers remain untested.


[Native cross-start correspondence](../../polynomial_causal/REGIONAL_CROSS_START_NATIVE_V1_RESULT.json)
passes all4held-out families:1.067%write error,0.658–0.970%signed removal-effect
error with each fit's own query writers. This strengthens grouped identification
beyond input-angle matching. The two payload readers have now been folded exactly
through MLP16; native producer mediation is registered, not yet measured.


The frozen head17.2 regional branch now has a portable mixed-precision token-input
executor; native query/current generators remain external. In96 competing-city
contexts, removal affects all48editor-city contrasts positively, but one original
coverage cell misses10% and neither native model nor either child has a robust
twofold editor-over-tourist advantage after counterbalancing city assignments.
This limits a role-specific regional interpretation. Within-package child
removal effects have0.210–0.215%relative nonlinear nonadditivity on this panel.
[Results, controls and scope](../../polynomial_causal/REGIONAL_COMPETING_CUES_V1_MATH.md).


## 2026-09-12 — head13.0 regional consumer-specific split, not a new whole-head unit

A coefficient-recurring shared linear source parent aliases the leading value reader after folding head13.0 into four readings of the head17.2 regional branch. With full native routing, that component has expected signed transfer on95/96 controlled swaps but only1.3–2.5%of the attention8/9/13 group transfer. Wholehead contribution changes sign between city assignments; component and remainder oppose for original cities. Consumer-weighted gauge-invariant selection also misses registered fidelity/transfer bars. This is a conditional downstream-reading intervention, not recursive module removal; it does not supersede the capitalized-committee dossier. [Primary math and receipts](../../polynomial_causal/FOLDED_PRODUCER_NATIVE_V1_MATH.md).


## 2026-09-12 — head8.2/head9.8 regional edge screens

A frozen27-head consumer-weighted value bank has two passers:8.2and9.8. Components reproduce corresponding wholehead effects at the four-reader regional interface within1.7–3.8%on96reusedcontexts, and carry28–32%/60–72%of selected producer-group transfer. Writers nearly align(Hcos.99949), source readers differ. This is consumer-specific edge evidence, not wholehead semantic renaming;8.2newline-setter record above still governs. Fresh confirmation/common-output grouping pending. [Primary results and all-candidate receipt](../../polynomial_causal/FOLDED_PRODUCER_NATIVE_V1_MATH.md).


### 12September17:03 — Regional producer confirmation and execution

The8.2/9.8regional edge components pass48freshcity/spelling/templateexamples:2.3–3.0%head-effecterror,89–93%grouptransferjointly. Commonoutputeffecterror.23–.25%;5.54MBscalarproducerpackageexecutes withnative8/9currentinputs+completefirst-tokenlookup. This doesnotclosetheinputgeneratorsorcertifynewlinepreservation. Dedicated16newlinecontextsandactualwholehead8.2positivecontrolregistered; nativecontrolpending. [Primary evidence](../../polynomial_causal/FOLDED_PRODUCER_NATIVE_V1_MATH.md).

### 12September17:40 — Head8.2 newline service and regional edge separation

Natural FineWeb32-prefix V2 control passes: actual wholehead8.2mean replacement adds.0333/.0248newlineCE acrosshalves, while jointregional-edge removal changes CE by.000260/.000347meanabs. Zerohead also damagesnaturalmeanCE; originalauthoredprompts fail under bothinterventiontypes. This supports the existingnewline-service dossier and conditionalregional-edge preservation, not wholeproducer-component selectivity. [V2receipt](../../polynomial_causal/SCALAR_PRODUCERS_NEWLINE_NATURAL_V2_RESULT.json).

### 12September17:46 — Physical producer removal and serial dependence

Original8.2/9.8source components removed recursively reduce67–70%regionalcuecontrast (all24pairsdirection), versus7.6–8.5%underconditionaledge removal. Wholeheadfidelitymisses andoneFineWebrownewlinejointCEdamage.1665failmax.1selectivity; nofullcircuitpromotion. Frozen-second-write diagnostic explains muchofjointnonadditivity, butMLP8/normsintervene. [Results and exactMLP8bridge](../../polynomial_causal/FOLDED_PRODUCER_NATIVE_V1_MATH.md).

### 12September18:10 — Regional serial path confirmed; collateral retained

Head8.2->MLP8->head9.8 direct+mixed generator predicts signed serial effect within.38–1.14%onreused48rows. JointQKroutingneeded: value-onlyscalartransfermisses56.6%inonefamily, with19.3%routing/valueinteraction. Sixfixedcountryvariants ofthepreservednewlineoutlier remaincapable;5/6jointdamages>.1. No globallyselectivecircuitpromotion. [Primary evidence](../../polynomial_causal/FOLDED_PRODUCER_NATIVE_V1_MATH.md).
