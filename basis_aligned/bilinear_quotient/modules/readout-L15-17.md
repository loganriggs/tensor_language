# Readout (L15-17) — near-linear read, the merge, and the frequency calibrator

**Current evidence:** read the [updated MLP17 dossier](../../polynomial_causal/explanations/MLP17_CURRENT_UNDERSTANDING.md), including failed whole-layer transfer, normalized response experiments and September 10 calibration tests. The historical “inert,” “fully isolated” and “closed” statements below apply to their original measurements; they do not establish universal task irrelevance or a complete circuit under the current four-property criterion.

**One line:** attention inert; MLPs are near-linear reads that rotate the stream into the token
basis; block 17 holds the rank-1 frequency calibration; the grammar/content merge into logits is
additive-linear.

## Established facts
- **Attention 15-17: inert** (per-head costs ≤0.006, §1083; §1047).
- **MLPs near-linear reads:** mlp17 0.85, mlp16 0.78/0.94-bilinear (§1046); readout does the bulk
  of output formation (logit-lens CE 5.8@L15→3.26; §944) via ~95% linear rotation (§945);
  readout reads class 13× + position 6× (§851). mlp16 bilinear loss-rank ~64 (§1040).
- **mlp16 structure (§1090, held-out):** output is 82% token-variance (structural fact) but the
  4%-variance dev part carries ~87% of its CE value; dev-only substitution (context without the
  token baseline) costs 0.795 vs mean-abl 0.138 — the big token component is REDUNDANT for CE
  (stream already carries identity, §690) yet load-bearing as a baseline if you keep the dev part.
  §1084's "readout = token calibration lookup" was WITHDRAWN then reinstated only as this refined
  structural claim (§1088→§1090).
- **The merge (§1082/§1086):** final logits ≈ additive-linear in the content component (cosine
  0.77 between logit-delta of content removal and −W_lm·c). Head/tail division is about magnitude:
  content removal hits rare-target log-prob 16× an unscaled random control — but only 1.7× a
  NORM-MATCHED control; argmax fragility is magnitude-generic (norm-matched random flips 41% vs
  content 38%; §1086 retracted §1082's "content picks the winner"). Final-residual deviation is
  much lower-rank than mid-stack content (top-64 = 76% var; §1082).
- **Block 17 = dominant frequency calibrator (§624-662):** rank-1 w_freq (removal kills it 103%,
  random 0-2%), ~40% aligned with unembedding log-freq axis; calibration components in 5 layers,
  block 17 dominating 5-10×; w_freq lives 88% in the massive dims (§676).
- **Errors/calibration (§972-980):** top-1 class-correct 2/3; hedges to function words when
  unsure (calibrated deferral, ECE 0.009); content misses land in-topic 40% vs 15% random.

## Benchmark status
Readout band ~0.56-0.9 (§906/§940; mlp17 0.85, mlp16 0.78 §1046). The merge is understood
(additive-linear); block-17 calibration fully isolated (the model's ONE clean rank-1 knob).

## Gotchas
- Variance-rank ≠ functional rank (mlp17 §615/§660); low-variance tail carries loss.
- At the final residual, most content-specificity is already merged — deletions there act like
  generic damage (§1086); test content claims mid-stack, not at the readout.

**CEILING CERTIFIED (§1131-1133):** held-out — mlp17 0.842 linear / 0.821 top-256 own-neurons;
mlp16 0.813 / 0.807. Two instrument families converge at both MLPs: ~0.81-0.84 capturable, the
final ~0.2 neither sparse nor linear (§660 law certified band-wide). Token augmentation redundant
(−0.009). mlp15 0.40 at near-zero stakes. Do not chase 0.9 here with sparsity/linearity instruments. THIRD FAMILY (§1139): fitted rank-64 quadratic gains +0.01 (0.822/0.857; train R² rises to 0.975 but held-out doesn't move) — the ceiling is a THREE-FAMILY LAW; module closed.

## Open
- Nothing pressing beyond the standing §1069 middle-attention remainder upstream.

## September10 terminal weight-fold update
The entire centered unembedding has now been pulled through MLP17 and attention17 c_proj. Exact checks held, strong output-sharing/head-concentration bars failed: top128coefficient capture30.54%vs29.00%before O; within-head blocks11.86%vs11.25%with scrambled head coordinates. This coefficient-space result does not establish task inertness or close the layer. A further source-coherence decomposition is controlled but not yet measured on native weights. [Current MLP17 record](../../polynomial_causal/explanations/MLP17_CURRENT_UNDERSTANDING.md), [source-constraint math](../../polynomial_causal/SHARED_SOURCE_ATTENTION_QUADRATIC_V1_MATH.md).

The September11 full source/value contraction includes all output rows:43.01%of formal coefficient energy is in a component vanishing for one source but live across sources, versus43.08%source-permutation control. The centered routing-wedge spectrum needs31of36modes for90%; the leading mode largely follows the already-known17.2/17.3pair. This does not assign a new behavior to that pair. [Latest explanation](../../polynomial_causal/explanations/2026-09-11/explanation_2026-09-11_0005.md).


## 11 September03:52 — mixed-interaction shared-reader family

Weight-only full-U/centered native calculation completed2.66s,A/B/Cheld. Allowing
all interactions touching rank11inputspan captures6.09%centeredcoefficientenergy,
with6.80%universalspan ceiling. Rank128capture38.56%, insideonly6.41%; at least147
readers necessaryfor50%capture. Not semantic irrelevance or arbitraryprogrambound.
Numerical/nonorthogonalbasisredteam passed. NextproducerfunctionGram tool controlled;
nativeMLP16fold comparison notrun. See explanation0352,
SHARED_INPUT_SUBSPACE_NATIVE_V1_RESULT/REDTEAM and PRODUCER_FUNCTION_OVERLAP_V1_CONTROL.


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


## 11 September04:33 — coupled producer/routing selection completed

Joint weight-only fit improves producer-function overlap4.77x but retains73.58%
of prior QK touch, missing80%. All36 starts have converged continuations after
same-objective repair; original miss preserved. Fixed midpoint gives2.94x sharing
with90.95% retention on already inspected positions: post-result audit only.
Leading common MLP16 scalar functions remain dense under exact product spectra:
mean16-product capture18.20%, median256products for90%, versus259private. These
fixed-function results do not rule out simpler feature combinations or shared
multioutput programs. No behavioral identification. [Combined explanation](../../polynomial_causal/explanations/2026-09-11/explanation_2026-09-11_0433.md).


11September04:48: selecting combinations within the frozen17D common producer spans improves meanone-product capture3.01%to5.38%; all36starts converge, noheadpasses joint gain/sharing screen. Whole-span one-product upperbounds<=20.57%, not a limit on larger/different programs. Returning to unconverged overlapping full-U blocks; standard manifold adapter and saved-state bridge pass, native continuation pending. [Details](../../polynomial_causal/explanations/2026-09-11/explanation_2026-09-11_0433.md).


11September05:03 correction: custom manifold block fit had already run240s (MULTIOUTPUT_MANIFOLD_V1_RESULT), capture8.62847%, unconverged. LibraryCG continuation from that newer state gives8.62854%, still fails stationarity after a line-search stop. Same16x16x4full-U model. Correct reduced-Hessian trust-region solver prepared; native status in current runner. [Latest account](../../polynomial_causal/explanations/2026-09-11/explanation_2026-09-11_0503.md).

2026-09-11 06:10 Codex: MLP17 full-U23x4quadraticframe fits locallyconverge at5.94%coefficientcapture; centeredcaptureonly2.95–3.02%, so notnewidentifiedreadoutcircuits. [Current evidence](../../polynomial_causal/explanations/2026-09-11/explanation_2026-09-11_0608.md). Newcompactfull-ranktransform controls are syntheticonly.

11September21:42: the new validated suffix component's48readers fold exactly throughattention17 O/current/first-value maps. Its taskconditioned attention-mediated fraction is0.4–8.5%; incomingresidualdominates. Native-reference MLP16/background interactions are38%verbs/19%nouns, so thiscomponent doesnot support anadditive upstreamchain. Scope differsfrom historicalglobalinertness andearlier17readerfolds. [Report andprimaryreceipts](../../polynomial_causal/explanations/for_logan/research_update_2026-09-11_2142.md#6-newest-result-tracing-the-input-upstream).


## 12 September08:12 — mixed MLP15 sources and a common conditional interface

Frozen composed branches3/8 read three inputquadratics sharing oneparent. On128freshendpoints, attention16 residual-only branch8 keeps active swaps within3.2–8.0%; MLP15 source/background isolation bothfail. Exact sourcegain differentiation shows opposing sharedparent/partner contributions, so historical late-attention/near-linearity summaries do not determine this path's mechanism. A common128-coordinate MLP15 producer interface chosen onlyfromweights gives branchwriteerrors2.87/7.72%versusrandom72.55/60.38%, but joint swapfidelity fails. Predeclared512diagnostic preserves branch8 acrossallfourfamilies; branch3adverb13.51%missremains. Normalization, nativeproducerreadweights andbackgroundremainexplicit. No global layer closure or four-property circuit promotion. [Primary source math and results](../../polynomial_causal/COMPOSED_WEIGHT_COMPARISON_V1_RESULTS.md), [interface receipt](../../polynomial_causal/SHARED_MLP15_INTERFACE_V1_RESULT.json).
