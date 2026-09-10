# Codex research-session startup and continuation

## Start this program with Astra

The installed Codex CLI is `0.153.4`. On 2026-09-08, `codex update` resolved and successfully
installed that same latest version. At 14:16 UTC, `codex debug models` reconfirmed that this
account's live catalog lists `GPT-6-Astra` under the exact slug `gpt-6-astra` with
`visibility = "list"`. Astra is already the default in `/root/.codex/config.toml`.

In a running Codex terminal session, enter `/model` and select **GPT-6-Astra**. If Astra is absent
from that picker, exit the session and explicitly override the model. To preserve the current
conversation history, use:

```bash
codex resume --last -m gpt-6-astra -C /workspace/tensor_language -a never -s danger-full-access
```

To start a clean session instead, use:

```bash
codex -m gpt-6-astra -C /workspace/tensor_language -a never -s danger-full-access
```

In the ChatGPT desktop app, first use **Menu > Check for Updates**, then create a new Codex chat
and open its model picker. Astra is rolling out gradually, so it may still be absent from the
desktop picker even when the CLI catalog already exposes it. The explicit CLI command above is
the verified path for this account. If the command ever reports an access error, that is an
account/workspace rollout issue rather than a repository or GPU-instance problem.

For a clean session, paste the prompt in **Suggested first prompt in a new session** at the end of
this document. A resumed session keeps its history, but should still read this file because it is
the durable authority for the periodic clocks and managed runners. To diagnose a stale picker,
query the actual catalog:

```bash
codex debug models | jq '.models[] | select(.slug == "gpt-6-astra")'
```

## Purpose

This document is the restart handoff for the bilin18/Theseus mechanistic-interpretability
program.  A new Codex session should use it to resume the actual research loop rather than
merely summarize the previous session.

The durable objective is to produce a smaller transparent tensor program that is jointly:

- predictive on fresh and out-of-distribution text;
- composable when task programs or replacements are installed together;
- selectively manipulable under removals, swaps, and edits;
- simpler under literal storage, compute, edge, state, and program pricing.

The current circuit-scale priority is to identify high-quality causal circuits and reusable
circuit-finding machinery.  Low rank, activation reconstruction, variance preservation, or
compression alone is not circuit evidence.

## Directional coupling is internal; additive-model bound executed — 2026-09-10 09:00 UTC

Preregistration/result unit 834f09bb1 follows directional preregistration 4176f6a83.
MLP8_COUPLING_DIRECTIONS_V1_RESULT.json is valid A; both one-way hypotheses B/C
and small interaction D fail all 32 worlds. All bridges exact. Mixed vocabulary
interaction / directional coupling norm = 0.420819–0.671962. 512 forwards / 8192
sequences, 9.05565 executor seconds. Runner SHA
c50d7a2bdcba42520bd06ebe20cfcfcea55c9164d13a98679d95ec7884a3a231.

MLP8_COUPLING_READOUT_V1_RESULT.json valid A; final-reader B fails 32/32, internal C
passes 32/32. Additive endpoint a=x10+x01-x00, actual final RMS/unembedding/softcap
recomputed. I_reader=Z10+Z01-Z00-D(a); I_internal=D(a)-Z11; I=sum. Reader remainder
<=0.006006 of interaction across all readouts; mixed vocabulary <=0.002885.
Decoder bridges maxabs2.002716e-5; parent cube replay exact. 512 forwards / 8192
sequences plus 192 decoder batches / 6144 states, 19.02963 seconds. Runner SHA
956bf14bb370bc1b3beb17d6bea5cc955204e7680df12640b6eba5d39e8720dd.

Post-result CPU additive_switch_lower_bound_v1 executed with append-only board claim.
Any additive output model B+F(a)+G(b) has worst-corner error >=||I||/4, attained by
the alternating quarter-interaction correction. 32 random controls pass <=8.89e-16.
Saved directional mixed-vocabulary measurements imply 0.105205–0.167991 minimum
worst-corner error / coupling norm, excluding uniform <=.10 in every world. This is
a bound on the fixed intervention tables, not arbitrary text or all nonlinear
programs. Do not reopen additive architectural split/rank/head/dose rescues.

Original handoff/pilot revisited. Next object: explicit INTERNAL joint computation,
its input products and consumer uses; not another whole-module grouping or final
reader-only explanation. No new native job registered. Actual CPU continuation
completed; all native weights and counterfactual dependencies retained, saving0.
Explanation: explanations/coupled_response_and_additive_limits_2026-09-10.md.
Full goal active. Continuous clock phase markers before work and turn_boundary
before final. Next hourly09:14 UTC; mathematical10:49 UTC (no duplicate review).

## Later response groups coupled; native recurrence repair complete — 2026-09-10 08:40:40 UTC

PreviousgoalturnPROGRESS96166ca07; firstprereg9d83c3e30, repairfbca0e18c.
MLP8_LATE_GROUPS_V1_RESULT.json INVALID: closed-form rawstateabs.013671875 failed
while relative~8e-8 andoutputbridgespassed. Preservedunchanged. V2 replacesonlyrawstate
validation withindependentnativeorderedrecurrence; BITWISE32/32. No tolerance/science/
row/group/interventionchange. frozen_write_recurrence_v1.py capturesnativeoperationorder.
MLP8_LATE_GROUPS_V2_RESULT.json validA; Battention/CMLP/Dinteractionfailall32.
Mixedvocabularyinteraction.247207–.386367 oflate-response norm. Attentiononlyvocaberror
.962–1.022; MLPonly.300–.452; neitherpasses. Native/bypassidentityexact; closed-form
carryoutputmaxabs2.57492e-5,rawclosedformdiagretained. Corrected384forwards6144seq+
64decoderbatches1024states,9.68820s; invalidv1extra9.78265s separatelycharged.
RunnerV2SHA2c2f94c08f99e8cca9a3233f8bad47cfc6702bb47fa4c2310cbcadd914098fe3.

Post-resultCPUlinear_cross_group_paths_v1 executed. Forlinearresidualmaps, group
interaction=M*A*delta (firstorderin source), randommatrixerror1.222e-15. Plantedreader
seesonlyMApath: total1,Aalone0,Malone0,interaction1; reversedAM0. Signoddnessalone
cannotprove linearity(cubiccounterexample). No fittedJacobianresponse/dose-rescueadopted.
Retaincoupledresponse; nativegroupinteractiondoesnotprove nonlinearsourcecomputation.

Nextcandidate directionalcut comparison (NOTREGISTERED): sourceM8mixedremoved/A9native.
Capture bankA fromA10..17live/M9..17nativefrozen, bankM fromMLPlive/Anativefrozen.
Fourarms: bothbanksclamped (independent source responses); bankAinstalled/MLPlive
(allowsA-to-M whileblockingM-to-A); bankMinstalled/Alive (allowsM-to-A whileblocking
A-to-M); bothlive(parentbypass). Banksarecounterfactualcut definitions, notfullyedited
writesreplayedunderanotherbackground. CapturesreplaypriorgroupZ10/Z01; bothliveZ11.
Test directionalresponse relativebothbanksbaseline, task/fullvocab andinteraction.
No automatichead/rank/adjacentlayer/doserescan. Nonlinearsuffix meansinteractionisnot
literalalternatingpathcount. Allnativeweights/inputcounterfactuals remain,saving0.
CPUcontinuationexecuted; explanation26updated; fullgoalactive/no successorjobregistered.
Continuousclock RESEARCH_ACTIVITY_2026-09-10_0814.jsonl: phasebeforeactions, separate
markercall beforecomposinglongedits; turn_boundarybeforefinal. Next09:14hourly/10:49math.

## MLP9 mediator rejected; broader group control executed — 2026-09-10 08:25:13 UTC

PreviousgoalturnPROGRESS8ff0131cd; preregistrationebdc63b1d.
MLP8_MLP9_BYPASS_FACTORIAL_V1_RESULT.json validA/D; Bmediator/Cremainingfailall32.
MLP8sourceS0/S1 crossed withMLP9writeM0/M1, A9nativeheldinEVERYarm. Fullvocabulary
metrics included. All5bridgesexact;448forwards7168seq,7.91374s. RunnerSHA
6b9a9f270e02b31236176173bc7c81bd17c0187270d0f010f4394d0705dfaeea.
Taskremainingpasses8/16original,0/16fronted; mediator0/32. Remainingvocabularyerrors
.366–.480; MLP9dominanceerrors .807–.895vocabulary. Interaction32/32pass, max
.00076521mixed/.00583117fullincludingvocabulary. Do notpromotetask-onlypreservation.

Post-resultCPUmlp9_consumer_scope_bounds_v1 executed. FromT=M+R+I, mediatornorm/T
liesbetween remainingerror +/-interactionnorm; envelope.36545–.48010vocabulary.
This isnot signedexplainedfraction. Sequentialgroupcontrol outputcube[[5,2],[4,0]]
forattentionlive×MLPlive undereditedsource; lateeffects total5,A1,M3,I1; directcarry1.
ReplayingfulleditMLPwrite underAclamp gives1 insteadactual2, so livegroups mustrecompute.

STOP automaticadjacent-layerwalking. Nextbroadercandidate: sourceMLP8fullmixedremoved,
A9nativealways; groupsA10..17 andM9..17 eachlive vsclampednativewrites. Fourarms
(frozen/frozen,live/frozen,frozen/live,live/live) withnativeprefix andlivegroupsreally
recomputed. Define late-response effects relativeallfrozen, whose finalstate should
bridge the prior directcarry prediction; alllivebridgespriorbypass. Nativeidentity
clamps required. KeepfirstVtuplesnative; groupsdiagnosticnotsemanticunits. No new
nativegroupjobregistered. CPUcontinuationexecuted; explanation25updated.
Fullgoalactive, all545902902weights/nativecounterfactualinputs retained,saving0.
Continuousclock RESEARCH_ACTIVITY_2026-09-10_0814.jsonl: markrestore beforefirstreads,
then scientificdesign, implementation, validation, science, publication BEFOREactions
(prefer a separate marker tool call before composing a long edit). Markturn_boundary
beforefinal; excludeinter-turngap. Next hourly09:14 UTC; math10:49 UTC.

## Direct carry rejected; MLP9 response control and hourly review executed — 2026-09-10 08:16:50 UTC

PreviousgoalturnPROGRESS83b6d5fa3; preregistration9a15028b8.
MLP8_BYPASS_CARRY_V1_RESULT.json validA; Btask/Cvocabularyfail all32worlds.
Directcarryscaleprod(lambda9..17)=1.4817735440233635. FinalnativeRMS/fullvocabhead/
softcap retained. Errororiginalmargin.393–.588,fronted.175–.757; vocabulary.829–.916.
Native/source/bypassparentreplayexact; fullnativedecodermaxabs2.67029e-5.
192forwards3072seq+128decoderbatches2048states,4.25978s. RunnerSHA
 a97161051e936c3b1b89612e926c781da921de331f19f35560292bc755f3d4a0.
No directcarry promotion or gain/rank/approximation ladder.

Post-resultCPUbilinear_normalized_source_response_v1 executed. Exact finite response
with native-normalizedinputs: Down[(Lu)(Rd)+(Ld)(Ru)]+Down[(Ld)(Rd)]. Both terms kept;
closure<=3.886e-15, planted quadratic omission errors.352–.456. This is notnative
MLP9dominance evidence. Next boundednative question sourceS0/S1 × MLP9writeM0/M1
insideA9-clampedbypass. A9nativeinEVERYarm; sourceS1=fullmixedMLP8outputremoved.
Bothdiagonalsreplaynative andpriorbypass; identifyMLP9response vsremainingroutes before
promoting its exactweightbranches. Ifweak, stopautomaticadjacent-layerwalking and
comparebroaderlate-responsepartition. No successorjobregistered; CPUcontinuationdone.
Allweights/nativecounterfactualinputs retained,saving0; fullgoalactive.

Hourly08:14reviewcomplete:7validscreens,median451s,max891s,executor38.300s. CIRCUIT_FOCUS/
NOVELTYpass; CEREMONY_BUDGETunestablished/fail. Phaseaudit49.9%coverage. Actualrepair:
research_phase_audit_v1.py pluscontinuous RESEARCH_ACTIVITY_2026-09-10_0814.jsonl.
At NEXT TURN START, before research reads, append restore/design event(categoryreview)
with research_phase_clock_v1.py to that continuous file. Mark implementation/validation/
science/publication BEFORE their actions. Mark turn_boundary(categorypublication) before
final; auditor excludes that inter-turngap. Do not combinecontinuous/per-jobintervals.
Next hourly09:14 UTC; mathematical10:49 UTC. Explanationsection24updated.

## MLP8 source / attention9 factorial completed — 2026-09-10 08:07:49 UTC

PreviousgoalturnPROGRESS6d1ff2cd6; preregistratione5ba410a0.
MLP8_ATTENTION9_FACTORIAL_V1_RESULT.json validA/D; Battention/Cbypass fail everyworld.
SourceS0/S1=fullmixedMLP8removal; crossed withwholeA9writeA0/A1 atallpositions.
Bothdiagonalidentityclamps andnative/source-editedreplayexact; firstVbitwiseunchanged.
384forwards6144seq,5.60494s. RunnerSHA
17823e91aba1599876c0e0a9b417774cac40bbf776accc5f0342a86209c7f98d.
Neitherroutealone<=.10fidelity; interaction32/32pass, max.00064146mixed/.00335469full.
The bypass is sourcechange withA9nativeheld; includesresidual/latercomputations.

Post-resultCPUsource_attention_route_accounting_v1 executed. Correct-margin signed
attentionprojection .17196–.37422original,.11468–.43358fronted; bypass .62567–.82783/
.56658–.88530. Small signedinteractionclosesaccounting, notindependentsemanticcircuits.
Synthetic frozen-write residualcarry verified: deltafinal=prod(lambda9..17,0)*deltaMLP8,
fixedlaterwrites/embeddinginjections; nonlinearfinalRMS/readersoftcap retained.
Maxstate1.777e-15/reader2.665e-15; live-nonlinear-writefalsifiererror1.30492.

Nextnativequestion: directresidualcarry prediction for the measuredbypass (F00-F10),
comparedwith actualsource/A9factorial outputs. Captureactualnativefinalresidual and
MLP8fullmixedwrite; subtract exacttransport atsemanticreadout, usefullnativefinal
RMS/unembedding/softcap. Preserve all32worlds andparentbypasseffects; no gain/rank
rescue. Old all-MLP finalcarry andvaluepathdirectcarry aredifferenttargets. Passing
would support an explicit final reader route; failingrequireslaterconsumerresponse.
No successorjobregistered; post-resultCPUcontinuationactuallyexecuted. Explanation23.
All545902902weights/nativecounterfactual inputs charged,saving0; fullgoalactive.
Next hourly08:14 UTC; mathematical10:49 UTC. ReadlivequeuebeforeGPUwork.

## MLP8 all-consumer comparison completed — 2026-09-10 08:01:25 UTC

PreviousgoalturnPROGRESSe3e20feef; currentpreregistration587efef7f.
MATURE_VALUE_MLP8_CONSUMERS_V1_RESULT.json validA/E; Bvaluepath/Cnew/Dinherited fail.
All32worlds failB/C/D, composition32/32. ActualMLP8output full/new/inherited mixed
writes removed, allnativeconsumers live. Fullsource value-path errors .526–.791
original,.566–.898fronted acrossbothreadouts; fulltablecompositionmax.013074.
Nativebaselineexact;320forwards5120seq,5.42232s. RunnerSHA
15847a86c4eaae66c56f473443506763ecb280f186b91f2c2a904cf247f343af.
Shared module_output_delta_v1.py checksincomingtensorbitwise and restoresonerror.

Post-resultCPUmlp8_consumer_effect_geometry_v1 executed. Value-path margin signed
projectionontotal .2120–.4722original,.1410–.4542fronted; positivepartialalignment,
not full mediation. Synthetic F(x,a)=x+a+2xa gives total4,isolatedattention3,bypass3,
interaction-2: totalminusisolatedpath isnotnative-backgroundbypass. This prevents
mislabeling residual effect as another circuit without joint interventions.

Next native candidate: sourceS0/S1 (fullmixedMLP8removal) crossed with wholeattention9
outputA0/A1, allfour F_s(A_t) withsource-consistentnative prefix andlivesuffix.
Native A captured ineachsource state; diagonalreplaysall-consumerparent. Offdiagonal
clampswholeattentionoutput preservingfirstV tuplefield (unchangedbyMLP8). Testattention
vsbypassdominance andinteraction explicitly; no chosenhead/position/signedgain. The
background includes residual andlatercomputations; do notcallitonesemanticmodule.
No successorGPUjobregistered; actualpost-resultCPUcontinuationcomplete. Explanation22.
Fullgoalactive, all545902902weights andnativecounterfactualinputs retained, saving0.
Next hourly08:14 UTC; math10:49 UTC. Readlivequeue beforestartingnativework.

## MLP8 value-path source partition completed — 2026-09-10 07:55:46 UTC

PreviousgoalturnPROGRESS8c929f718; preregistration/mathreview fc415323c.
MATURE_VALUE_MLP8_ORIGIN_V1_RESULT.json validA/E; Bsource/Cnew/Dinherited fail.
No world passes B/C/D; all32compositionpass. MLP8mixed source effect norm/fullC
.440–.557original,.464–1.012fronted; these are norm ratios, not explained fractions.
MLP8new=Down(Lo*Rh+Lh*Ro), inherited=Down(L0*Roh+Loh*R0). Subtract branch times
native block9lambda0 from raw L9input, recompute native RMS and partial valueCprime,
install B+Cprime keepingnativeP0/background, fullsuffix. Other MLP8consumers untouched.
MLPcomponentbridge<=1.439e-5; exactnativebaseline;320forwards5120seq,5.66846s.
RunnerSHA68dfc323ae3467b8399500ecdd3d3210ff482c72cc57ada23045816c0e9f7bd5.
Fulltablecompositionmax.00245455. Capturev3exposes MLP8input/output, oldversionsimmutable.

Post-resultCPUmlp8_value_branch_accounting_v1 executed. New/inherited margin signed
projection relativeMLP8effect .2559–.4850/.5149–.7440 original; -.1732–1.2513/
-.2516–1.1732fronted. Cancellationexplicit, neitherfailedbranchpromoted.
Review THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-10_0749.md searchedprimaryliterature:
Gramorthogonalinvariants are generic; CP uniqueness condition cannotcertify supplied
4608termfactorization; WFAlineartransitionassumptions fail. NativeGram-onlyreplay
demoted in favor of actual upstream source test. Fullgoalactive/all545902902weights
and nativecounterfactualinputs retained, saving0. Explanationsection21.

Next meaningfulconsumerquestion: actualMLP8branchremoval with alldownstream consumers
live vs currentisolatedvalue-pathremoval. Agreement <=.10 onbothreadoutscouldsupport
valuepathdominance; disagreementrequiresotherconsumers ratherthan namingwholeMLPa
singlecircuit. This isnot yetregistered; preserveall32worlds and failedglobal-source
verdict. Post-resultCPUcontinuationactuallyexecuted. Clocks math10:49/hourly08:14 UTC.

## Raw mixed removal completed; exact Gram continuation executed — 2026-09-10 07:47:38 UTC

Previous goal turn PROGRESS e73097a53; current preregistration 1b24746d6.
MATURE_VALUE_RAW_REMOVAL_V1_RESULT.json valid A; B necessity and C regeneration fail.
xprime=x-Qoh x, native RMS recomputed inside partial value producer, native P0/background retained.
Raw-removal fidelity16/16 original,5/16 fronted; no regeneration-only pass. Maximum errors
.680047 margin/.625274 centered readers fronted, <=.038022 original. Native/full replay
exact; identity maxabs8.58307e-6.320forwards5120seq,5.0131s. RunnerSHA
6600a9501f1d8dee3776e04c0489b87a717122df94ed4c611cc79555e9c8d77c.
Registered spill/projection diagnostics saved separately; no revised verdicts.

Post-result CPU continuation executed: rms_regeneration_gram_v1.py and
RMS_REGENERATION_GRAM_V1_CONTROLS.json. xprime=a+ob+hc; six Gram entries determine
four RMS scales, normalized mixed coefficient=g_oh*a+g_h*b+g_o*c. Arbitrary linear
readers fold into these three vectors exactly; 32fixtures plus planted controls,
max errors1.166e-15 mixed/4.330e-15 folded. This is not fixed semantic coordinates,
native identification, or free input generation. All545902902weights remain.
Next native instrument can reconstruct the regenerated common write through this explicit
Gram interface before tracing which upstream writes produce the context-dependent vectors.
No successor native job registered. Explanation section20. Full goal active.
Next mathematical review07:49 UTC; hourly08:14 UTC. Existing phase clock begins late
and merges implementation/validation/native execution; do not infer a ceremony-budget pass.

## Normalization-origin partition is structure-dependent — 2026-09-10 07:34:37 UTC

PreviousgoalturnPROGRESS4136ff3f0; currentpreregistration/implementationb66e647df.
`MATURE_VALUE_NORMALIZATION_V1_RESULT.json` validA/D; Braw/ Cnormalization fail.
Actualu=g*x, fullconditional u_oh=g0*x_oh+g_oh*x0+go*xh+gh*xo. InheritedR=g0*x_oh
versusN=otherthree, propagatedthrough unchangednativeP0/W_V/W_O tocommonwrite,
fullnative suffixrecomputed. Rfidelity16/16original,6/16fronted; N0/32; composition32/32.
Rawerrorsoriginal<=.0369; frontedup to.6211margin/.5701readers. NativeRMSrecurrence
bitwise; FP64ubridge5.544e-8,componentbridge1.907e-5,outputbridges6.676e-6abs/
2.920e-7rel.320forwards5120seq,5.147s. RunnerSHA
`d40266b140c3da893c5071ea6b22fcd4fdc0f4f4c39832ba5cff5a4c9d070927`.
Completedv1sourcespreserved; `mature_value_route_native_v2.py` exposesrawresidual
andnormalizedinput; `rms_mixed_input_partition_v1.py` exactsyntheticcontrols pass.

ExecutedCPUcontinuation: `normalized_value_branch_accounting_v1.py` /
`NORMALIZED_VALUE_BRANCH_ACCOUNTING_V1_RESULT.json`. Nmargin-effectnorm/fullcomponent
.0119–.0366original,.0225–.6212fronted; Nsignedprojection-.0235–.0240original,
-.1053–.2319fronted. Cancellation/orthogonalityexplain whyNcanmatterwithoutbeing
sufficientalone. Fullgoalstillincomplete; normalizationcannotbedroppeduniversally.

Next producer question requires a genuine raw-input perturbation, not interpreting
R=g0*x_oh removal asrawstateablation. KeepfullRMS: candidate x'=x-Qoh x, recompute
u'=RMS(x'), then commonC'=L(P0,W_V Qoh u') withnative meanrouting fixed; install
B+C' atcommonwrite. Compare withnative/fullCremoval to testwhetherrawmixedstate
isnecessary for thispartialproducer despite regeneratednormalizationinteractions.
This is a value-producer path test, notglobalrawresidualablation changingQK.
No successor nativejobregistered yet; do not infer circuitimportance fromrawnorms,
fitgains or chooseonlypassingworlds. All545902902weights/fournativecounterfactual
inputs/native suffixremain. Explanationsection19updated. Post-resultCPUcontinuation
actuallyexecuted. Phaseclockin MATURE_VALUE_NORMALIZATION_V1_PHASES.jsonl. Next
hourly08:14 UTC; mathematical07:49 UTC. Fullgoalactive; verifylivequeueonresume.

## Native value/routing contraction valid; separate-factor rules fail — 2026-09-10 07:24:53 UTC

Previous goal turn PROGRESS3ce1462d1; current preregistration/implementation8b19c2005.
`MATURE_VALUE_ROUTE_V1_RESULT.json` validA; Bvalue/Crouting/Dinteraction fail.
All16pairs tested2value×2routing×2recipient. No pair passes either one-factor
rule acrossbothrecipients/readouts; smallinteractionpassesonlydad:defend (1/16).
Correct-marginvaluefollowingerrors.0438–.6516; routing.6323–9.0971; interaction
up to.6467margin/.5310centeredthree-readers. No value-onlypromotionfromsmallererrors.
Native compiledcomponentbridge<=1.902e-5relative; alloutputbridges<=8.584e-6absolute
and2.874e-7relative.448forwards7168seq,7.184s. RunnerSHA
`d2fb56603d1e3be2a4317c241baa04261b44088f2eebb979e0f66182cc132b8d`.
Reusable `mature_value_route_native_v1.py` captures/compiles the2query×3source
nativebilinearproducer; normalizedinputs, nativeRMS/RoPE/P0 and W_V/W_O remain.

Executed CPU continuation: `mature_factor_path_attribution_v1.py` /
`MATURE_FACTOR_PATH_ATTRIBUTION_V1_RESULT.json`. Sixchangeorders telescope to
allfronted-minus-alloriginal effect; margin signedprojections value.8133–1.2420,
routing-.2202–.1468,recipient-.0319–.0956. Signedcancellation/interactiondistribution
explicit. This doesnotrescuefailedfactorinvariance. Keepjointvalue-routingunit.
Next useful producer question: does mixednormalizedvalue preexist in residual
input, or arise fromRMSnormalization? Preserve routing, useexactscalar×vector
factorpartition ratherthanhead/rank/gain orfrontedtextrepair. No successor native
experimentregistered yet. Check prior normalization lessons before implementing.
All545902902weights/fourcounterfactualinputs/native suffix remain; independent
semanticextraction andfullgoal incomplete. Existingexplanationsection18 updated.

Hourly0714completed with5priorreceipts, median8m10.5 butleading21m45; executor17.393s.
CIRCUIT_FOCUS/NOVELTYpass; prior CEREMONY_BUDGETunestablished/fail. Boundedrepair
`research_phase_clock_v1.py` implemented/executed; currentphase boundaries in
`MATURE_VALUE_ROUTE_V1_PHASES.jsonl`, excludesinitialdesign/hourreview andfinalgit
fromanyprecommit summary. No retrospectivetimingclaim. Nextclocks08:14hourly,
07:49mathematical. Fullgoalactive; post-resultCPUcontinuationactuallyexecuted.

## True interchange: both simple rules fail; producer split prepared — 2026-09-10 07:12:13 UTC

Previous goal turn PROGRESS0c8b2fbf2; current preregistration/implementationdf3ba4abc.
`THIRD_NOUN_COMMON_INTERCHANGE_V1_RESULT.json` valid A/D; Bproducer/Creader fail.
True replacement uses B_r=A_r-C_r and installs B_r+T C_p; E_pr compares that
with removed-C background. All16pairedworlds, bothdirections, fullnative suffix.
Producer rule passes3/16pairs jointly onbothreadouts; reader0/16; smallinteraction
16/16. Correct-marginproducer errors.0367–.2435; reader.6199–9.0943. Interaction
<=.09086margin/.09841centeredthree-readers. Native/removedparent grids exact.
256forwards4096seq,4.274s; runnerSHA
`7323959c4f4ca11d8924439fc78bc3e45d9eb61c5f98bbe5a4cb9cade29951c4`.
Do not promote producer-only because its errors are smaller.

Executed CPU continuation: `common_interchange_change_attribution_v1.py` /
`COMMON_INTERCHANGE_CHANGE_ATTRIBUTION_V1_RESULT.json`. Exact symmetricproducer
change projects .9062–1.0325 onto totalstructuraleffectchange; reader -.0325–.0938,
readerchangenorm .0194–.1105total. Signedcancellationallowed; this describes the
opened2x2contrast, notzero-reader dependence. `mature_value_route_contract_v1.py`
plus `MATURE_VALUE_ROUTE_CONTRACT_V1_CONTROLS.json` derive/test next boundedobject:
C_q=(1-lambda)sum_hs P0_hqs W_Oh W_Vh(u_s)_oh. Commonqueries to/action, sources
first-joint-information/to/action; causal mask [[1,1,0],[1,1,1]] inbothlayouts.
Matrixidentityerror1.33e-15,32rowgeometrieschecked. Completionnounrolesdiffer;
alignment isbyinformationstage, notnounidentity. Actualnative own-factorbridge
must pass before interpreting cross-factor swaps. Localcoefficient1.65625.

Next highest-information experiment: separate native contextual value production
from conditionally averaged attention routing using a2x2factorial, retainingboth
recipientcontexts (2x2x2 effects) because reader-only/producer-onlyarebothfalse.
No nativefactorial isregistered/executedyet. No gain/head/rank/rowrescue; all545902902
weights/fourcounterfactualinputs/native suffixremain. Existingexplanationsection17
updated. Fullgoalactive, actualpost-resultCPUcontinuationcomplete. Hourlyreview
next07:14 UTC; mathematicalreview07:49 UTC. Verifyliveboards/queuesonresume.

## Common mature interface localized; cross-context accounting tested — 2026-09-10 07:03:05 UTC

Previous goal turn PROGRESS7ff6ac79c; current preregistration/implementationd2abbb742.
`THIRD_NOUN_COMMON_SUFFIX_V1_RESULT.json` valid A/B/D; C/E/F fail. Same16original
and16fronted worlds. Full mixed attention-write role swap could leak future cue
information: original attractor7 maps to fronted attractor2 beforeobject8, reverse
fronted object8 maps to originalobject4 beforecategory7. Causal role audit executed.
Only to/action are common mature roles. Their mixed-write branch C preserves the
full mixed component's effect with .002071–.039322relative error acrossall32;
remaining noun branch N alone failsall32 (.9853–1.0190error). Composition errors
<=.000937correctmargin and<=.001053centeredthree-reader. Parentbaseline/fullmixed
replays exact;320forwards5120seq,4.874s. RunnerSHA
`446a65c6a82c7e597316a2549c1cb26fd04a17a63bf4edce64bc8f082ff02d44`.
Commonfactorselectivity16/16original,10/16fronted; gender14/16original,3/16fronted.
Do not promote a semantic circuit or erase fronting/gender failures.

Post-result CPU continuation executed: `cross_context_effect_attribution_v1.py`
controls and `COMMON_SUFFIX_CONTINUATION_V1.json`. For E[producer,recipient],
producer-following predicts E10~E11 andE01~E00; reader-following predicts E10~E00
andE01~E11, eachdirection relativeerror<=.10. Exact symmetricchange/interaction
accounting distinguishes planted producer-only,reader-only,interacting fixtures.
No nativecross-contextoutput yet. Next highest-information action is preregister
and implement the now-licensed common to/action write interchange across original
andfronted layouts, fullnative recipient suffix live, no fittedgain/rolemap.
This distinguishes the wholewrite producer from recipientcontext/laterreaders,
not valueproduction from attentionrouting inside the producer. Both remaincharged.
No successor nativejob registered yet. Explanationsection16 updated, fullgoalactive;
all545902902nativeparameters/fourcounterfactualinputs remain. ActualCPUcontinuation
complete. Hourly review07:14 UTC; mathematicalreview07:49 UTC. Verify livequeues.

## Structural transfer: longer phrase passes, fronting fails — 2026-09-10 06:55:03 UTC

Previous goal turn PROGRESS5c618f7e6; current preregistration/implementationfa15a4af8.
`THIRD_NOUN_WRITE_STRUCTURE_V1_RESULT.json` valid A/C/E; B structural transfer and
D gender fail. Same16 lexical worlds in each of two new layouts,1024 new prefixes.
Longer intervening phrase13tokens (object4/category10):16/16 transfer,13/16gender.
Fronted phrase11tokens (category2/object8):7/16transfer,3/16gender. All32 have native
mixed-marginRMS>=.05 and all32 pass mixed-effect fidelity<=.10 and composition.
Fronting has8 materiality failures and5 spill failures with overlap; do not filter
or retune. Native/reference full/mixed/spill grids replay exactly. Shared dynamic
executor controls pass; installed mixed impurity<=3.75e-6.266forwards4256seq,4.382s.
Runner SHA `681ed6fc110b29625c75e27758ad016bf78c41aa9662c84eb4c08786a37d09ef`.

CPU continuation executed: `third_noun_structure_geometry_v1.py` and
`THIRD_NOUN_STRUCTURE_GEOMETRY_V1_RESULT.json`. Signed projection decomposes as
(effect/native norm ratio)*cosine. Longer effect/native cosine .9677–.9953;
fronted -.3495–.9815, with4 negative. Eight fronted effect vectors oppose their
original-layout counterparts, while native-interaction cosine remains positive
in all16. This rejects mere uniform positive-gain repair; it does not localize
whether producer, averaged routing, or later readers change the semantic role.

Next question: localize where structural rearrangement changes the role of the
selected value-write operation. Algebraic fidelity/composition alone do not imply
shared semantic computation. Do not retry fronted text/threshold/gain selection.
Use existing partial operation as a hypothesis, not an identified universal number
circuit. New helper `attention_write_factorial_executor_v1.py` and geometry-aware
`live_value_factorial_executor_v2.py` support other equal-length signed cubes;
completed v1 sources are preserved. No successor native job registered yet.
Explanation section15 updated. All545902902 parameters/native counterfactual inputs
and suffix remain; fullgoalactive. CPU audit is completed continuation receipt.
Hourly review due07:14 UTC; mathematical review07:49 UTC. Both runners healthy;
Claude v483completed and v485queued/running independently—inspect live state.

## Live mixed-write split passes fidelity/selectivity/composition — 2026-09-10 06:46:07 UTC

Previous goal turn PROGRESS6cc3d8bbe; current preregistration/implementation5545c4470.
`THIRD_NOUN_VALUE_WRITE_FACTORIAL_V1_RESULT.json` valid A/B/C/E; D gender fails.
At layer9 attention output, split the original local-value intervention's all-position
write difference D into Dm=Qoh D and Ds=D-Dm. Recompute the native suffix after
removing each part or both. Across all16 frozen fresh lexical worlds:
- Mixed-only target-effect error .00653–.09228 versus the original local-value edit.
- Factor spill .06386–.22375; all16 pass <=.25, including six parent failures.
- Natural mixed-effect projection .1391–.2712: still a partial component.
- Composition error <=.01458 correct margin and <=.01343 centered three-reader vector.
- Gender still fails monk/introduce (.3961) and woman/introduce (.2943).
All native/edited parent and full-write replays exact. Installed mixed-write impurity
<=2.85e-6; early delta zero. 160 forwards/2560 sequences,2.535s. Managed runner SHA
`ce8ef4120e372f0d1e8ffb1fe88b6296f8475a24a23f650b9135af4a276e9e38`.

Executed CPU continuation: `third_noun_write_factor_attribution_v1.py` and
`THIRD_NOUN_WRITE_FACTOR_ATTRIBUTION_V1_RESULT.json`. For a pure mixed value
correction d=oh*d_oh, Qoh(P*d)=mean_oh(P)*d. Synthetic matrix identity error4.44e-16.
This interprets Dm as the value interaction read through conditionally averaged
routing; the other routing factors produce Ds. Saved nuisance attribution shows
cancellation; do not call signed projections positive fractions of a mechanism.

Next highest-information step: freeze the mixed-write operation and test structural
OOD transfer before further extraction. This bank was already open when Dm was
registered, so its repaired selectivity is not new OOD evidence. Do not tune gender
thresholds or discard failures. Full independent producer/suffix extraction remains
missing; all545902902 native weights charged. No successor GPU job registered yet.
Explanation section14 updated. Full goal active; CPU audit is completed continuation.
Hourly review next07:14 UTC; mathematical review07:49 UTC. Both runners healthy.

## Fresh value transfer: material effect persists, selectivity fails — 2026-09-10 06:38:38 UTC

Latest preregistration/implementation commit: `48080763f`.
`THIRD_NOUN_VALUE_TRANSFER_V1_RESULT.json` is valid; A passes, B/C/D fail.
The frozen layer 9 local-value operator transfers a material partial effect to all
16 fresh noun/action worlds (projection .1286–.2689), but six fail factor spill;
both male and female panels pass only 5/8. Gender control fails monk/introduce
(.3985) and woman/introduce (.3003), against unchanged .25. Reference native and
edited replay exact. Managed runner SHA
`56f1e00018740b0f505be23f17655432ebe5bb8f9298d05f69fcaa6c64371ed8`;
70 forwards/1120 sequences, 1.328s. No clean reusable number circuit promoted.

Post-result CPU continuation is executed: `third_noun_reader_geometry_v1.py` /
`THIRD_NOUN_READER_GEOMETRY_V1_RESULT.json` and
`CONTEXTUAL_READER_FACTOR_SPILL_V1_CONTROL.json`. Orthogonal reader coordinates
N=z_t-(z_m+z_f)/2 and G=z_m-z_f verify answer-pair and centered-energy identities.
Number energy 89.36–99.11% and answer-effect cosine .9817–.9983 do NOT override
failed selectivity. Exact toy d=oh, J=1+.3h gives Jd=oh+.3o: input interaction
purity does not guarantee output factor selectivity under contextual readers.

Next highest-information question: distinguish contextual downstream reading from
an upstream value producer bundling different variables; design a producer/reader
causal test with explicit opposing predictions. No successor GPU job registered.
Do not tune thresholds, remove failed groups, or declare independent extraction:
four native counterfactual input states and all 545902902 parameters remain.
Existing explanation section 13 holds the result and derivation. Hourly review
next due 07:14 UTC; mathematical review 07:49 UTC. Both bqrunners healthy at startup.
Full durable goal remains active; CPU audit/control are actual continuation receipts.

## Live value partial causality/selectivity; shared scorer executed — 2026-09-10 06:19:51 UTC

Previous goalturnPROGRESS68242ceaa; prereg/runnereca250445.
THIRD_NOUN_L9_VALUE_LIVE_V1 validA/B/D; Cdirectcarryfails. CompileW_V Poh(native
normalizedL9input), removeatlocalc_v allheads/positions, native1.65625mixingand
fullsuffixrecompute. Natural mixedanswer signedprojection .1570–.2533;
nonmixed/mixed answerchange .1373–.2332 passesfactorselectivity. This iswithin-
bank factorselectivity, notunrelatedtaskcontrols. Remainingnaturalinteraction
magnitude .7489–.8440: partialcomponent, notsufficiency.
Directcarryerror margin.0950–.3744/fullmixedvocabulary.8583–.9204; preserveCnull.
Q/K/Q2/K2andsharedfirstVbitwiseunchanged, identityexact; weightcommute4.74e-6,
localwriteoracle<=1.66e-5; parent0/nativefinaldecoder2.67e-5abs/4.62e-7rel.
48forwards/768seq +32decoderbatches/512states,2.015s,0fits. RunnerSHA
 a71a17c063568039c5e41a1807c5e3d7335516e461d1df624e275cac5ef9ad89.
Completed artifactsimmutable. Correction stillusesfourcapturednativeinputs;
all545902902parametersremain/saving0, noindependentproducer orOODclaim.

Hourly0614complete:5validscreens through06:14:16; consecutivegapmedian9m36,
executor5.916s. CIRCUIT_FOCUS/NOVELTYpass, CEREMONY_BUDGETunmeasured/fail.
Required boundedreuse repair CLAIM+EXECUTED factorial_effect_metrics_v1.py and
 audit_live_value_factor_metrics_v1.py: reuseWalshkernel, native/direct projection,
spillratio andParseval spectrumreplay<=2.78e-17; target/spillcontrolpasses.
Timed CPUread/analysis/validation only, notearlieragentthinking/reportingtime.
LIVE_VALUE_FACTOR_METRICS_V1_RESULT.json storescoefficientsand remainingratios;
largestspilltermvariesworld(co,oa,h,o), notoneuniformrepair.

Nextpriority freshlexical/different-reflexive-reader transfer ofSAME frozen
weightcompiledvalueoperator beforemoreproducerdecomposition. No nextnative
rows/protocol/jobselectedyet. Keeporiginalhandoff/pilotauthority, notstalegoaltext.
Existingexplanationsection12updated. Nextclocks07:14hourly/07:49math. Bothmanaged
runnershealthy; preserveClaudework. Fullgoalactive; post-resultCPUauditexecuted.

## Layer9 routing/value partition and causal-source support — 2026-09-10 06:06:40 UTC

Previous turnPROGRESS c46e16885; prereg/runnerc1b46d894.
THIRD_NOUN_L9_ROUTE_VALUE_V1 validA/Bmateriality, C/D/Eallfail. Localmarginerrors:
routing .919–1.127,value .372–.760,cross .340–1.055; fullvectoralsofails.
No branchwinner, head/source/rankselection orbarrescue. Exactconditionalpartition
Poh(PV)=PohV0+P0Voh+PoVh+PhVo; actualpostRMS/RoPE/nativevaluemix/all9heads/all10sources.
Factorclosure1.40e-15,nativemixedwritebridge3.51e-6.16forwards/256seq+
80decoderbatches/1280states,.943s. NativeL9naturalprojection.1821–.3806 remains
partialmateriality, notwholecircuitsufficiency. RunnerSHA
9d30e9c2aa68314be0d8e0616271d22e756ca877e679329cdb1c4f771f0bf9d4.
Completedartifactsimmutable. Nativeprefix/firstV/pattern/currentnormallretained.

Post-resultclaim+EXECUTED third_noun_causal_source_split_v1.py:
metadataall256tokensverifyoat4/hat7; valuebranchbefore7 andcrossbefore4zeroexact.
Sources4–6 cannotknowh: crossisPhVo only, categoryinformationentersquerynotearlierkey.
Sources7–9 canknowbothandcarrymixedcontextualvalues. Earlier-numbercrosssigned
projection -.1547–.5486; later-contextvalue .2940–1.0458 under fixednativeRMS,
PRE-softcaplinearreader accounting. Notphysicaledgeablation orfull-outputeffect.
Head6largest7worlds/head7one underthisdiagnostic; notuniversalhead6circuit.

CPUcheckpointmmap nativeL9attn.lamb=-.65625,localcoefficient1.65625.
SharedfirstVistoken-local, PohV1=0 onthisdisjointdomain; hencePohVlocal=
1.65625 W_V Poh(normalizedcontextinput). W_O/W_Vfoldforthisbranchlegalonlywith
actualcontextualnorminputandP0routingretained. SharedVcanstillserveothercrosspieces.
CoefficientreceiptTHIRD_NOUN_L9_VALUE_MIX_COEFFICIENT_V1.json. No newnativeforward.

Nextusefulobject contextualmixed-valueproducer orproducer-level earlier-number
routingtest; neither nextnativeprotocol norjobselectedyet. Existingexplanation
section11updated. All545902902weightsremain,saving0;fullgoalactive.
Clocks06:14hourly/07:49math. Bothmanagedrunnershealthy; preserveClaudework.
ExecutedCPUsourceaudit+scalarread ispost-resultcontinuationreceipt.

## Attention direct-write factorial; reader-weighted layer9 localization — 2026-09-10 05:54:14 UTC

Previous goalturnPROGRESS9c34f680c. New prereg/runner842ae2f51.
THIRD_NOUN_ATTENTION_READER_V1 validA, BfailsALL8worlds, Cpasses: Aonlyremoval
remainingmixedmarginratio .2688–.4328; signedprojection .5786–.9755. AMremoval
remaining.0708–.1112. Mixeddecoderjointinteraction ratio6.20e-5–1.40e-4;
fullvocabulary2.91e-5–2.54e-4. NeitherA norMsufficientalone; do notpromoteB.
Compositionhereis syntheticfinalstate transported-native-write edits, not
upstreammoduleinterventions orindependentlyextractedcircuits.
16nativeforwards/256seq+64decoderbatches/1024states,.961s,nofit.
RunnerSHA b122c0bc6020136f12fb1df8efa5cb8b9beec66599bdcf977eb96d97cddcaf96.
Stats-vs-GPUdecoder max3.10e-6/rel1.09e-7, parent0; Mreplay0,AMratio1.18e-6.
19fixed-sourceexactper-rowreader/normstatistics savedinresult (~2.2MB), enabling
CPUcounterfactuals withoutanothernativeforward. Completedartifactsimmutable.

Post-resultclaim+EXECUTED third_noun_reader_route_audit_v1.py (zero nativecalls):
A-only numeratorvsnormcounterfactuals; norm-onlyeffect .000566–.001550 ofnatural
mixedmarginmagnitude. Mainattentioneffect viareader numerator, notnormscale.
Layer9attention largestindividualsignedsourceall8worlds .1821–.3806;
10/12oftennext, orderlexicaldependent. Layeradditivityerror<=5.65e-5ofnatural
mixedanswer. Descriptiveopened-data finalwrite edits, not upstreamcausality.
No layer9-sufficiency orsharedsemanticunitclaim; nohead/token/rankchosen.

Nextnativequestion: formationoftheseattentionwrites viaQ/Krouting,valuecontent,
andupstreamproducers; layer9iscandidatewithin-moduleinterface. No nextGPUprotocol
orjobselectedyet. Existingexplanationupdatedsections9/10. All545902902weights
charged,saving0; fullgoalactive. Clocks06:14hourly/07:49math. PreserveClaude
v479/v481; v481wasrunningafterownedresult, bothmanagedrunnershealthy.
CPUaudit is actualpost-resultcontinuationreceipt.

## MLP factor partition fails behavior; exact decoder edit statistics executed — 2026-09-10 05:45:40 UTC

Previous goal turnPROGRESS c8536531f. New prereg/runner bddb67047.
THIRD_NOUN_MLP_FACTOR_PARTITION_V1 validA, B/C/Dallfail: removingallMLPmixed
leaves naturalmixedmarginRMSratio .681–1.103; newonly .765–1.030; inheritedonly
.693–1.191. No new/inheritedwinner orMLP17rescue. This rejects MLPbehaviorcarrier
inference from85–95% RAWVECTORprojection. Wholemixedstateremovalpositive remains.
Exact symmetric factor algebra new=LoRh+LhRo, inherited=L0Roh+LohR0;
all18nativeLeft/Right/Down, inputnormalizationretained. Factorclosure<=2.01e-17,
nativewritebridge<=1.542e-5rel; parentreplay0,decoder1.05e-5abs/1.58e-6rel.
16forwards/256seq +64decoderbatches/1024states,1.078s. RunnerSHA
7642a21007674fbaab6651a5145e93cfd744a0deb05c31b27ad5ccdbf29c4309.
Preflight ABS-VS-REL warning heuristic: code/protocol enforce BOTHabsANDrelative.
Completedrunner/helper/rows/protocol/binding/results immutable.

Concretepost-resultclaim+executedCPU rms_softcap_edit_statistics_v1.py:
Wx'=Wx+sumalpha Wd; norm²'=norm²+2alpha·base-source+alpha^TGramalpha.
SharednormGram plusindividualreader numerators exactly predict fixed-sourcejoint
edits through nativeRMS/softcap. Directerror<=8.89e-16,gauge<=3.22e-15;
zero-stateepsiloncontrolpass; omittedsourcecross-termerror.149. Notnewsemantic
sharing orindependentactivationproducer. Counterexample largevector99.99% yet
smalldirectioneliminatesreader. ControlsreceiptRMS_SOFTCAP_EDIT_STATISTICS_V1_CONTROLS.json.

Conditional nativeattention remainder from totalstateeffect minusMLPeffect has
signedprojection .5787–.9756 afterMLPmixedalreadyremoved (numericalclosureaudited).
NOTattention-onlyremovalinnativebackground. Nextnative should capture exactreader/
normstatistics and testattention inbothbackgroundsbeforeheadselection. No nextGPU
protocol/job chosenyet. Do notrepeat normrankings orresurrect failedMLP/iswas routes.
Existingexplanation updatedsections7/8. All545902902weightsremain,saving0;
fullgoalactive. Clocks06:14hourly/07:49math; bothmanagedrunnershealthy.

## Native mixed-state dependence passes; source Gram audit executed — 2026-09-10 05:35:40 UTC

Previous goal turn PROGRESS (9d0b6d524). New prereg/runner commit138aac63f.
THIRD_NOUN_MIXED_STATE_V1 completed validA/B/C on all256 frozen prefixes.
Remove P_oh from final rawstate per fixed c/s/a square, decode unchanged model:
remaining mixed-margin RMSratio .0708–.1112, signed removedprojection .8983–.9381;
all8worlds pass nativefloor.05 and ratio<=.25. This is synthetic four-state
interaction removal, not an independently generated/selective circuit.
16nativeforwards/256seq +32decoderbatches/512states, .919s, nofit/backward.
RunnerSHA cc08ca2bf95ebee640e55d33b0b250da3e5b2a553f4f9e891232240298de1a2c.
Parentmarginbridge0; finaldecodermax1.05e-5/rel1.58e-6; rawtelescope rel8.56e-8,
mixedtelescope2.22e-6. Firstattentionquerymixed zero controlpasses (roundoff);
following normalization/MLP firstcreatesmixed. Nativecompleted artifactsimmutable.

Post-result continuation CLAIM+EXECUTED third_noun_source_gram_audit_v1.py:
36transportednativewrite Grams, signedprojections sum1, cancellationcontrolpasses,
rawsquarednormclosure<=9.40e-7. AllMLPwrites .8535–.9522 signedprojection;
attention .0478–.1465; layers12–17 .8053–.9164. LastMLP17largestallworlds
(.4099–.6428) but alonevectorerror .497–.714. Sumindividualnorms/sumnorm2.28–3.06.
These are contextual nativewrite contributions, not independent causal sources;
attention canremain necessary toproduceMLPinputs. Do notclaim MLP17is thecircuit.

Nextinformationtarget: inherited versus newlyformed interaction in normalized
bilinearMLP inputs, with attention producers retained. No nextnativeprotocol orGPUjob
chosen/queued yet. Avoid firstattention/iswas/phase resurrection or selectorrescue.
Updated existing explanation noun_number_selection_and_cross_token_interactions_2026-09-10.md.
Clocks06:14hourly/07:49math. All545902902nativeweights charged,saving0;fullgoalactive.
Bothmanagedrunnershealthy. CPUsourceaudit is actualpost-result continuationreceipt.

## Controller and third-noun screens; cross-token support audit — 2026-09-10 05:27 UTC

Original handoff/pilot remain the authority, not stale better_math goal wording.
SUBJECT_OBJECT_CONTROLLER_V1 validA, Enearest passes, B/C/D/Ffail: all128 paired
reflexive preferences follow second/object noun; all32 promised opposite-number
cases fail grammatical-controller labels. 8forwards/128seq/.505s, nofit. Controller
additivity audit (published ef310a3b1) drops interactions yet preservesall128 signs;
interaction/nonconstantRMS .062–.158. This is opened per-world diagnosis, not OOD.

THIRD_NOUN_ANIMACY_V1 now completed validA; allfour controller/object/nearestnoun/
nearesthuman gates FAIL. Rawcorrect173/211/173/219 of256 is not promotion. Both
animacy-conditional gates alsofail. 16forwards/256seq/.558s,nofit. Runner SHA
b7b5ad733bb5eaa4254a0bccd695da4318e625906033e3cf9c3917596bf98954.
Completed helper/rows/protocol/binding/runner/resultimmutable; no phrase/rule/bar rescue.
All545902902nativeweights remain, saving0. No ownedGPUjob pending.

Concrete post-result continuation: factorial_semantic_support_v1.py reuses existing
dealiased_boolean_spectrum Walsh kernel with explicit signedfactor/bit ordering;
audit_factorial_semantic_support_v1.py executedCPU and saved
FACTORIAL_SEMANTIC_SUPPORT_V1_RESULT.json. Both native margin tables replay
within1.78e-15, allthirdnoun rule predicatesexact. Every64 object-number x human-
category (oh) square cancels for ANY token-local lookup, editspositions4/7;
noneof64 attractor-number x human-category (ah) squaresdoes, bothposition7.
Native oh/o RMSratio.2318626; outputmixed canstill arise solelyinfinalRMS:
additivestatecounterexample rawmixed0,normalizedmixed.0207664. Thus do not call
ah learnedmultiplication oroh attentiongating without localization.

Highest-information next native question: where disjoint-token oh interaction
forms in residual versus normalization/readout, then producer/consumer math only
ifmaterial internaloperationexists. No protocolfor thisnextnativejobselectedyet.
Do not resurrect iswas/phase/firstattention orpromote failedsimple selectors.
New explanation noun_number_selection_and_cross_token_interactions_2026-09-10.md.
Hourly0514 complete, next06:14; mathematical0449 complete,next07:49. Hourmedian
receiptgap17m19>10m; ceremonybudgetunestablished/fail. Required bounded reuse repair
is the executed shared transform/scorer/supportaudit, notanother nativeframework.
Bothboardsupdated; bqrunner/bqrunner2healthy. Fullgoal remainsactive.

## Query-phase portability fails; joint-consumer metric executed — 2026-09-10 04:55 UTC

Previous turn PROGRESS bd33ed367. New branch uses Claude v473 about/for distance
failure, not closed first-attention/iswas queries. ABOUT_FOR_QUERY_PHASE_V1 is
valid A/D, B/Cfail: native24-head fit across13layers replays within.000460;
fw axisfraction .688135 -> .378006 under fixed query-phase cap. Native complete
recovery .937137 vs phase-complete .526582. Native-cache/short-phase bridges0.
1405forwards / 107792seq /120backwardsteps,58.719s executor. All545902902weights
remain, saving0. Completed runner SHA849d54c438d6262757f70970b7022b400cfa962ffa1a98dde745baf8364d92c6.
Completed runner/helpers/protocol/binding/rows/results immutable. No shift,
key-phase, individualhead orrank rescue. This null does not exclude other
positional mechanisms. Parent v473 longer text also changes contextual content.

Exact CPU transport uses R(new)R(old)^-1 and native BF16 cosine/sine. Native
c²+s² ranges .994644–1.005508; transpose-as-inverse wrong (.01738 control).
FP64transport<=1.34e-15, FP32<=9.54e-7; actualtinycapture nativeoutput unchanged,
rerotationoracle5.56e-17. Read-only alternate heads retain actual native prefix
at every layer; endpoint is native_base+virtual_donor-virtual_base, then original
BLOCK-LIVE fitted axes/native suffix. No independent token execution.

Metadata audit: all256 A1pairs equal-length/singlechangedtoken; fit indexgaps3/4,
fu4,fw9,atto4. Frozen row file contains80fit A1+80fitC+128heldA1. Old prose
five/ten tokens includescue; phase shift is precisely-5 (9->4).

Concrete post-result continuation executed:
output_reader_error_partition_v1.py / OUTPUT_READER_ERROR_PARTITION_V1_RESULT.json
finds>=99.9912% of full-vector squarederror outside one answercontrast; short
panels>=99.9990%. Fullhead-reconstruction failure does not alone refute a
TASK-SPECIFIC circuit, while one-margin success does not explain all consumers.
Next object must account for multiple actual consumers and their background.
Do not relax the failed phase fidelity bar or relabel the result a positive.
consumer_quotient_gram_v1.py / CONSUMER_QUOTIENT_GRAM_V1_CONTROLS.json executed
with duplicate/mixed/overlapping readers and dependentcycle. Exact observed
norm=(We)^T(WW^T)^+(We); maxfixtureerror3.56e-15, inconsistentcycle rejected.
Shared answer token alone gives readercosine.5, not hidden computation sharing.
This is metric machinery, not identifiedcircuits or dynamicclosure.

Three-hour0449 review complete, next07:49; hourly remains05:14. Explanation
position_and_content_in_shared_attention_2026-09-10.md includesnative outcome.
No ownedGPU successor pending. Next native experiment not preregistered; CPU
consumer audit/metric is executed continuation. Future reuse of fitted axes
should SAVE them rather than repeat this59s selection/fit each time; this
completed runner did not save them and must not be modified. Preserve Claude
v475 and both boards: recent Claude messages are in BQ/AGENT_BOARD.md as well
as the root coordination board. Both managedrunners healthy. Fullgoalactive.

## First-attention carrier fails; mathematical progress audit executed — 2026-09-10 04:33 UTC

FIRST_ATTENTION_CUE_MESSAGE_V1 completed validly: instrument and compiled fidelity
pass, across-panel cue-carrier fails. 40 forwards / 720 sequences / eight independent
token-factor productions in 1.626s. All nine heads/all valid positions; receiving
firstV preserved. Signed margin projection has .003–.014, is A2 .290–.298,
is held-out .062–.071. Close this first-attention WRITE branch; no head, token,
gain or firstV rescue. Runner SHA b906ebea79c73430a19905b32d69e67d4f7b8e74a78e88a02570199b1d1fc883.

User requested mathematical reconsideration using original handoff/pilot.
Executed causal_effect_geometry_audit_v1.py (zero model forwards). For natural
paired target t and native component effect d, r=||d||/||t|| and
p=<d,t>/||t||² give natural-target error E=sqrt(1+r²-2p). Compiler relative
error e only changes E by at most e*r. Native answer-margin E is .704–.712
for is A2 and .933–.941 for is held-out; has .989–.997. Maximum compiler
uncertainty 4.722e-6. This is post-result diagnosis, not changed bars or a
new candidate. Identity and triangle-bound controls pass. Saved receipt:
CAUSAL_EFFECT_GEOMETRY_AUDIT_V1_RESULT.json.

Append-only board claim plus executed CPU audit is the continuation receipt.
Broader ledger inspected: v23 direct-carry .833 is conditional-joint-relative
and failed selectivity; it is not natural-effect sufficiency. Smaller-model
endpoint-field result has later broader-layout/field failures. Do not resurrect
either as an established solution. Original handoff/pilot's shared read–route–write
objective controls, not stale better_math goal wording. Weight folding and joint
reader cross-block algebra already exist; repeating them is not new discovery.
Next candidate must identify an operation with actual producers and consumers,
carry substantial behavior, and then face extraction/joint-intervention tests;
no particular new native interface is preregistered or queued yet.

Current explanation is updated in place:
explanations/first_attention_support_after_raw_origin_null_2026-09-10.md.
All 545902902 native weights remain, zero structural savings. Hourly0414 is
complete; next hourly05:14, mathematical review04:49. Both managed runners
healthy. Preserve Claude fbd82f8ec/v471 and newer shared work. Goal remains active.

## Raw origin fails; first-attention support rule executed — 2026-09-10 04:08 UTC

Previous turnPROGRESS (cdf442f60). CurrentRAW_INTERACTION_ORIGIN_V1 completed
03:56:50 in2.289s,112forwards/980seq/64localMLPsyntheses; Avalid,B/C/Dfail.
Attention4-only mixedinputcorrectionfull-vectorerrors.771–.856;earlier-only
.441–.570. These arefourthcornerinput-edgeedits withproducersretained, not
independentattention4execution. MLP/identity/directsource/parentreplay0;
rawrecurrencebitwise,directembeddingmixedzero. RawFP64sourcefoldmax.000411,
rel5.755e-8 passesregisteredrelativebar. RunnerSHA
8117626bc1a84c9f91c343ae4bacbe29869b9e80b16e1c3a8d71263eb0929d16.
All545902902weightsremain,saving0. Completedrunner/protocol/resultimmutable.

Materialitymiss matters: is_a2mixedcorrectiononly.052–.067of fullsourcecue
logitnorm and.009–.010ofmargin; has.436–.597logit/.123–.290margin. Do not
continueweakMLP4mixed-inputdecompositionasthemainiswascircuitsearch. No raw
normalizationthreecornerrescue orsource-subsetwinnerpromotion.

Concretecontinuation: token_local_attention_support.py andexecutedcontrols.
FIRSTattentionlayerfactors q/k/v/q2/k2 aretokenlocal evenwithnativeRMS/RoPE.
No softmax meansreadisaSUM ofquery-sourcepairfunctions. Singlecue delta can
useonlycausaledges touchingchangedcuepositions. Mixedcue/context delta can
useonlyedges fromquerychangedbyonevariable tosourcechangedbytheother.
Unchangedqueries havezero mixedread BEFORE followingnorm/MLP; notvalidfor
latercontextuallayers. Actualtinyfactororacle0,single-edgeerror9.715e-17,
mixededgeerror5.552e-17, livemixednorm.4954. Softmaxcounterexamplemixed.03918
atunchangedqueryconfirmsassumptions. Hooks/methodsrestored.

Executed audit_token_local_attention_support_v1.py writes
TOKEN_LOCAL_ATTENTION_SUPPORT_V1_ROW_AUDIT.json. All46squares:1412causaledges,
326singlecueedges,82mixededges PERhead. Eval35squares:1247/271/63. All46
semanticquerypositionshavepossiblemixedread, socounts donotexplainisweakness.
Countsarecounterfactualsupport,nativecausalimportanceandwholesavingsunproven.

Nextnativeobject shouldtestMAINfirst-layercue-messagecausalrelevance using
exacttoken-derivededgeprogram BEFORE focusingitsmixedpart. No trained
first-layerprotocol/jobregisteredyet; CPUtool+audit isexecutedcontinuation.
This islocalization/executiontool, notdiscoveredtensealgorithm orsmaller
independentmodel. Existinglookupendpointcircuitisdifferentmodel; sharedfirst-
valuepayloadnullstillbinding. Originalhandoff/pilotauthority, notstalebetter
mathgoaltext. Newexplanationfirst_attention_support_after_raw_origin_null_
2026-09-10.md. Goalactive;clocks04:14hourly/04:49mathunchanged. PreserveClaude
v465/v467andsharedrunnerchanges. NoownedGPU successor pending.

## Native semantic square fails; raw-input lineage prepared — 2026-09-10 03:50 UTC

Current turn PROGRESS: SEMANTIC_SQUARE_V1 INVALID preserved; V2 validnative
null. V1 128forwards/1120seq/2.987s:48localFP64/FP32absoluteoracles and2small
legacycross-batchmarginrelativechecksfailed. NativeMLPFP64roundoffmax.004272,
rel2.024e-7; full-logitoraclesalready<=2.146e-5. Do noteraseorrelabelV1.
V2 numericallycorrectscontrols ONLY: FP32weightreplay; explicitr_i native-minus-
FP64roundoff inknowncornerindependentoracle andfourthdependentexactsumcontrol;
originalparentbatchANDsemanticreadoutreplay; full-logitreadoutbridge. Candidate
local/transport/inherited/full/identityunchanged; V1margin drift0.

V2 completed03:42:24,144forwards/1408seq+16localMLPoracles,3.544s; A/Dpass,
Btotal andCinteraction fail. Totalfull-vectorerror has.432–.620,is.054–.150;
5of16cellspassbothframes. Interactioncorrection error.598–.852;0of16pass.
is_a2totalcuepassesbutinteractiondoesnot; do notpromoteweakcontextvariation
intoexplainedcomposition. Inherited-onlydiagnosticalsofails. FP32weight,
exactsum/localindependentlogits andoriginalparentmarginreplay0. Readoutbridge
max3.434e-5; FP64identitymax2.297e-11. Sciencebarsunchanged,noanchor/norm/gain
rescue. RunnerSHA8387c7528a578d8a0c464f8522bc667541d4524f2eeb1850bf56b4c7be12f0df.
Allnewcompletedrunners/protocols/resultsnowimmutable; all545902902paramsretained.

Concrete continuation: four_corner_residual_lineage.py andcontrols pass on
actualtinybackend; rawsource/mixedsourceclosures<=4.441e-16, normalizedinput
recurrencebitwise. RawpreMLP4=alpha*e + attention0..4 + MLP0..3 transported
bynative residuallambdas (ten terms). Coefficient18.11378644forembedding;
source0:.00666975,1:.52537125,2:.26580048,3:.462890625,attention4:1.
These aretransportweights,notimportance. Actualtrainedsourcebanknotcapturedyet.

CPU audit_four_corner_embedding_lineage_v1.py executed, saves
FOUR_CORNER_EMBEDDING_LINEAGE_V1_AUDIT.json: all326validtokenpositions in46
squares havematchingdiagonaltokenmultisets, soANYper-tokenlookup/RMS haszero
mixed difference. No trainedforward;fivecheckpointlambda tensors readviammap,
BF16bytehash viauint8view. Pairedequal-edge subtraction makesembeddingzero
bitwise; naivefourtermassociation initiallylefttinyroundoff incontrol andwas
correctedbeforeanynativecapture. Overlapping-tokeneditcounterexample live.
Embeddings stillfeedcontextualwrites andfullnormbackground; noablation/extraction
claim fromzero DIRECT embeddingmixedterm.

Nextscientificobject: originofinheritedcue/contextinteraction inrawsourcewrites,
particularlyincomingresidualversusnewattention4, withnormalizationexplicit.
Thisissource-contributionlocalization, NOT renormalizedthreecornerprediction
rescue. Distinguishcapturededgeeditsfromglobalproducerremovals. No nativeorigin
protocol/jobregisteredyet; CPUcapture+tokenidentityauditareexecutedcontinuation.
Newexplanationsemantic_square_native_result_2026-09-10.md. Goalactive;original
handoff/pilotauthority. Nextclocks04:14hourly/04:49mathunchanged. PreserveClaude.

## Shared products fail; semantic-square math executed — 2026-09-10 03:31 UTC

Previous goal turn PROGRESS (d9597e5da). Current SHARED_PRODUCTS_V1 valid
managed result03:24:31,80forwards/1392seq/4backwards in2.071s. A/C/Dpass Bfail:
shared128full-vectorerror.700–.820, own128.566–.731, cross.720–.924,
random.973–.990. Shared beatsrandom byregistered.05 inbothframes/all8cells,
but allsharedsufficiencycellsfail. Sharedstaticremoval retains.828–.894 signed
nativecuecontrast. Maxauditerror2.167e-5, allpass. RunnerSHA
178acfd7538249fadccaaaa951d12eeb611d96edb91814e9ad1f1936f16109f1.
No budget, subset-family, score or gain rescue. All545902902weights remain,
saving0. Fitting used24originalpairs ONLY, evaluation72openedpairs. Fixed
sharedminnormalizedsaliency128; tasksets overlap26, notsemanticidentification.

source_margin_gradient.py enables reverseAD atdetachedsourceMLP4output inside
UNCHANGED nativebackend/no_grad; freezes/restores parameterflags andgradmode,
saves lm_head outputwithoutdetach, autograd.grad onlysource. Never use
inference_mode aroundthishelper. Eighttinychecks pass; centraldiff2.03e-12.
Initialtinyheadzero causeddead-gradient checkfail; plantedheadinitialized
before native. Callbackexception cleanup tested, noparametergrads accumulated.
Completedhelper/runner/protocol/result hashes immutable.

Next changedobject: semantic_square_bilinear.py plusexecutedcontrols and
SEMANTIC_SQUARE_ROWS_V1_AUDIT.json (builderaudit_semantic_square_rows_v1.py).
For fourobservednormalizedinputs, u=n10-n00,v=n01-n00,
w=n11-n10-n01+n00,a=n00+u+v. Exact mixedMLPoutput = B(u,v)+B(v,u)
plusB(a,w)+B(w,a)+B(w,w). Firstpart constructsinteractioninsideMLP;
secondresponds toinputnonadditivity, which includesupstreamANDnormalization.
MaxCPUidentityerror1.457e-13; local-only/inherited-only/RMScontrols pass.
This is not the oldone-pairLeft/Rightresponsefactorial. No native squaredomain
or componentfidelity protocol is registeredyet, noownedGPUjobpending.

Manifest creates46disjointsquares from92of96pairs:11fit,35eval; twois_fit
andtwois_a2pairsunused forcompatiblepartneravailability. Fixedcuepositions,
cuetokens/answers/length; contextpositionsdisjoint. DeterministicrowIDs and
nooutcomes used. Context islexical-background BUNDLE, not necessarilya noun.
Allrowsopened; nofreshOOD. Threecornerinputa maynotbenativelynormalized or
reachable; donotclaimrawtext extraction. Nativefourthinputneededforinherited
term cannotbeusedasheldoutprediction. Nexttestshould askwhereconjunctionis
formed usingthese objects, notrevive sparsebudgetselection.

New explanation behavior_guided_products_and_semantic_squares_2026-09-10.md.
CPUalgebra/rowaudit isactualcontinuation afterthenativenull. Goalactive,
originalhandoff/pilot authority, clocks04:14hourly/04:49math unchanged.

## Product-subset math after normalization-response null — 2026-09-10 03:14 UTC

Current goal remains active; original handoff/pilot and appended structural
criterion override the stale better_math_ideas wording. NORM_PRESERVING_RESPONSE_V1
completed03:00:25, A/Cpass Bfail,48forwards/864seq in2.821s. Full-vector effect
errors.332–.510 inall8cells; nativefactororacle<=3.741e-5. Actual changed RMS,
projections andRoPE retained. Close response-approximation ladder; no degree,
module-subset or gain rescue. All545902902nativeparams remain, saving0.

Concrete continuation executed: bilinear_product_subsets.py and
BILINEAR_PRODUCT_SUBSETS_V1_CONTROLS.json. Ten algebra checks pass,max2.843e-14:
exact subset/full interchange, disjoint union, weight removal, signed score
conservation, factor-rescaling/permutation, and factorized symmetric tensorGram
against a dense oracle. Planted100,-100,1 cancellation shows BOTH energy-only
and signed local-write ranking fail to identify the simplest sufficient subset.
Do not turn the diagnostic score into a circuit-identification claim.

Next object: MLP4's native product subsets shared/private acrosshas/had andis/was,
with full downstream native recomputation. Native neurons are candidate terms,
not semantic units. Prior rank16hidden compiler/groups coverMLP0/1/2/3/6 output
interfaces; Tier4v118 captures downstreamcross/self terms. No newnative subset,
budget or protocol fixed yet; no ownedGPU successor queued. This turn's CPU
implementation/control receipt fulfills continuation after the last native null.
Do not drift into another global rank/energy scan. Subset selection must face
held-out causal fidelity and matched controls; bothscore rules alone are killed.

New explanation bilinear_products_after_response_nulls_2026-09-10.md explains
math and limits. Hourly0314 performed late from0306: five valid receipts since
0206, medianinterreceipt12m07s, no identifiedcircuit. CEREMONY_BUDGET not established
forwholehour; reuse sourcecapture/subset machinery, no newforwardframework.
Next clocks04:14hourly,04:49math. Bothmanagedrunnershealthy. Native row families
alreadyopened; no freshOOD or adoption claim. Preserve Claude/livequeue files.

## Mixed chain and receiving tangent fail; nonlinear norm test prepared — 2026-09-10 02:52 UTC

Previous goal turn PROGRESS: two managed native screens, CPU direction bound,
factor-response algebra and prospective next protocol. INTERVENING_CHAIN_V1
A/Dpass B/Cfail;80forwards/1440seq,3.884s. MLP-onlyfullvectorerrors .346–.456;
attention-only .725–.801. Identity/parentreplay0, residualonlyrawV<=1.40e-5.
RunnerSHAf7fe0e33bc0755fc68b14752fb3d0ff0c83e197d55a0683cd9c8fb5bd85b937a.
Actual unfrozen modules recompute; this is not postcomputed-source deletion.

VALUE_TANGENT_V1 A/Cpass Bfail;48forwards/864seq including8JVP,3.391s.
RunnerSHA879dbb834d13ae551268306b87c0a57de6031e178774a561e906111480001c01.
Receiver-only derivative uses unchanged native/no_grad backend viaforwardAD;
do not wrap in inference_mode or detach observedc_v9output. Tinybackend
centraldifferror1.81e-11; nativeprimal/identity/parentreplay0. Full-logiteffect
errors .375–.480, margin .161–.440, localV .472–.549. No scalar/dose rescue.
Both screens use unchanged72openedpairs andexistingpartialvaluepath. The
prior hasfull-logitcarrier miss remains failed; no newtextOOD/adoption.

Concrete continuation: audit_value_tangent_direction_v1.py executed,
VALUE_TANGENT_DIRECTION_V1_AUDIT.json. Best possible scalar perpanel/direction
still gives.276–.464 relativefull-vectorerror, identityclosure3.89e-16.
This bounds fixed scalar corrections on these vectors only. No gaininstalled.
polynomial_factor_response.py executed5controls; attentioncentraldiff4.43e-11,
singlefactor4.15e-16, MLPidentity4.44e-16. Mixed module-enable paths canexist
in a strictly linear source map; JVPnull, not clampingalone, tests nonlinearity.

Next BILIN18_MLP4_NORM_PRESERVING_RESPONSE_V1_PREREGISTRATION.md: retainACTUAL
changed RMS/projections/headRMS/RoPE, simplifyonlyfirst-order changes ofthe
multilinear factors insideattention/MLPs5..8. ForMLP keepL0deltaR+deltaLR0,
dropdeltaLdeltaR; attentionanalogousfivefactorfirst-orderresponse withmask.
Both types/allfourlayers use the SAME rule, no posthocsubset/higherdegree
rescue.48forwards/864seq, compare inducedlocalV9 viareceivingnativeH1/H4
readerbackground; relativeeffect<=.10 full-logit/margin everycell, identity
and parentreplay. This differs from the failed raw-input tangent by keeping
normalization nonlinear. Native hookintegration/execution stillpending;
not queued. Allweights/receivingfactors/sourcegeneration remaincharged.

New explanation mixed_chain_and_nonlinear_response_2026-09-10.md. Goalactive;
all545902902 nativeparams retained, saving0. LatestClaude v461ranbeforeJVP;
preserveitschanges. Runnershealthy/noownjobpending. Clocks03:06hourly/04:49math.
Originalbilinearhandoff/pilot controls, not stalegoalbetter_math_ideas.

## Direct value fold fails; intervening chain registered — 2026-09-10 02:33 UTC

Previous turn PROGRESS: managed nativefold48forwards/864seq in2.864s,
exact finite-change CPU operator, tested intermediate-output clamps and
next native protocol. BILIN18_MLP4_VALUE_LINEAGE_NATIVE_V1_RESULT.json
A/Dpass B/Cfail. RunnerSHA7919d9bbc758d037cf212027175d88aa4e942fb773f90c8ca1fcfeea03331f24.
Gamma4 .03257659278460778. Direct/fullvalue centered-effect projections
.125–.205, relativeerrors .810–.880; marginerrors .708–.872. Foldedcontent
FP64error9.88e-15, nativevalue1.595e-5, directfold/rawdirect logits1.335e-5,
identity0; recurrencebitwise andhooksrestored. Laterwritesmatter. Fullvalue
carrierB misseshasfull-logit .057–.078 versus .10 (margin .110–.167), while
is passes. AllvalidMLP4 tokenpositions are swapped here; earlier selected
bank taskscope/results are different and their nulls remain preserved.

Concrete continuation: normalized_bilinear_secant.py executed5controls,
errors<=3.56e-14. RMS exactpair operator is diagonal minusrankone, then
bilinear midpoint response; finite-step receivingtangent failsplantedfixture.
It conditions onbothendpoints: not independentprediction or fixedsharedgate.
intervening_write_clamp.py alsoimplemented/tested5controls; sourceperturbation
withbothinterveningmoduletypesfixed recoversdirectlineage2.64e-16, sharedfirst
valuepayloadpreserved. No newforwardframework.

BILIN18_MLP4_INTERVENING_CHAIN_V1_PREREGISTRATION.md is the next80forward/
1440seq test. DuringMLP4swap, freeze receiving-nativeattention5..8, MLP5..8,
both, orneither; every unfrozenmodule recomputes on CURRENT input. Theninsert
eachinduced localV9 inreceiverH1/H4 whileQ/K/otherbackgroundnative. This tests
real dependencies rather than droppingafter-computation source terms.
B MLP-only andC attention-only independentlyrequire<=.10 relativeeffecterror
againstfullvalue inbothcentered-logit/marginframes/allpanels/directions.
Residualonly isoracle, notfallbackpromotion. Identityclamps andparentreplay
required. Explainexistingpartialpath only; failedhascarrierB isnotrelaxed.
Native runner integration and execution stillpending; do not claim queued.

New explanation mlp4_value_folding_and_intervening_computation_2026-09-10.md.
All545902902 nativeparameters and1,179,648foldedcoefficients charged; saving0.
Goalactive, originalbilinearhandoffcontrols. Noownjobpending; clocks03:06/
04:49. PreserveClaude/sharedwork. Source/rank/dose/thresholdrescuesclosed.

## Query carrier closes; direct MLP4-to-value fold executed — 2026-09-10 02:21 UTC

Previous goal turn PROGRESS. Native query partition/interchange24forwards,
432seq,2.548s, A/Dpass B/Cfail. BILIN18_L9_QUERY_PARTITION_NATIVE_V1_RESULT.json.
RunnerSHA2305f1f4f45daea71f8238ef7e43693fb529259ca47dae76efd384041e4142ab.
Full contextual query swaps atL9H1/H4 have abs signed projections<.012 on
every panel/direction in both full-logit and margin frames. Stop per-source
tense-carrier search attheseQports; contextual query computation still matters
as background. Do not infer irrelevance or exclude untested joint Q/K edits.

Local source/rest numerator-cross explanation fails: errors1.236–1.339
in residual-write frame; normalization remainder projects1.215–1.312 with
opposing cross contribution. These are cancelling vectors, not variance shares.
Actualbank947936entries/4034332compressedbytes saved as
BILIN18_L9_QUERY_PARTITION_NATIVE_V1_BANK.npz, hash inresult, numpy no-pickle.
CPU QUERY_UNIFORM_GAIN_V1_AUDIT.json confirms scales.5/2 leave write unchanged
atFP64 precision; zero kills it; exact uniformderivative<=1.251e-20 relative.
This qualifies prior zero-cut interaction as normalization geometry, not
proof of semantic cooperation. Native comparison identitymax1.53e-5.

Concrete continuation: mlp_value_lineage_fold.py executed five CPU controls,
allpass max5.69e-14. For direct MLP4 change gamma*delta_m at V9 input,
deltaV=Wv*gamma*delta_m/s1 + Wv*u*(1/s1-1/s0). FoldWvDown into the exact
midpoint bilinear difference. Norm term and source dependencies retained;
Downbias cancels in difference but remains in nativebackground. Native
fullMLP4 intervention also changes later writes, which this fold DOES NOT
include. Next hypothesis is DIRECT residual lineage versus these intervening
nonlinear writes. No nativevalue-path run registered yet. Prior restricted
MLP4-to-localV9 fractions.870/.954has and1.156/.982is supportthisbranch;
originaltotal-mediationnull remains. No newsource/rank/prototype rescue.

New explanation query_background_and_value_path_2026-09-10.md gives outcomes,
formulas and limitations. All545902902 nativeparams charged, saving0,
goalactive. Bothrunnershealthy,noownedjobpending. Clocks03:06hourly/04:49math.
Originalbilinearhandoff controls, not stalegoalbetter_math_ideas. PreserveClaude.

## Query-source tensor and trained atlas complete — 2026-09-10 02:05 UTC

Previous goal turn PROGRESS: exact normalized source-gain attention tool,
managed19-source causal atlas, and CPU source/rest interaction analysis.
Follow the original bilinear handoff/pilot, not stale goal wording. New
user-facing explanation: explanations/query_source_interaction_math_2026-09-10.md.

BILIN18_L9_QUERY_SOURCE_ATLAS_V1_RESULT.json is valid: A/D pass, B/C fail.
Managed02:00:05–02:00:21,13.179s executor,336forwards/6048seq on unchanged
72 opened has/had and is/was evalpairs. No singleton passes any whole panel.
Robust >=.10 omission sets are empty under BOTH-panel rule; this does not
prove no shared dependence or different task mechanisms. All local tensor
oracles pass4.97e-14; native headread1.03e-5, normalized lineage1.60e-6,
unityfull-logits1.53e-5, independent raw MLP8 query omission1.34e-5.
RunnerSHA0b0b925ff5cfbc66a2c851d53d70a8441edf3cbf326ff43c94e9013df2108c31.
Prereg02:04 header is a timestamp typo; immutable protocol bytes were
hash-bound before actual02:00 execution. Do not edit completed authorities.

source_gain_attention.py uses19 fixed native query-source edges and their
actual keys/mixed values. Each read is a quadratic numerator over gains
divided by sqrt(two quadratic RMS factors).190 monomials; rounded RoPE
and Gram cross terms retained. Synthetic and tiny CPU capture/replay pass.
This is a conditional local intervention tensor, not a global polynomial
model or token-to-logit extraction. Sources held fixed means these are
consumer-edge cuts, NOT global upstream-module removal. All545902902
nativeparameters remain charged; weight saving0.

Concrete continuation receipt: audit_source_rest_query_interactions_v1.py
executed on saved four-cell arms and wrote SOURCE_REST_QUERY_INTERACTIONS_V1_AUDIT.json.
For each source/rest partition I=F(all)-F(only)-F(omit)+F(zero), where F is
paired answer-margin contrast. Exact identity closes0. Naive sum of19
leave-one-out effects misses joint query effect by.733–.808 relative RMS.
MLP6/rest interactions.609–.891 of total query margin-effect norm. This is
post-atlas description on opened rows, not19-player Shapley, a full-logit
bound, or semantic factor identification. Do not lower thresholds or
promote a posthoc single source. Next useful discrimination is where the
source interaction enters: bilinear routing numerator, query normalization,
or downstream suffix, followed by task-specific joint edits. No such new
native run is yet registered. Keep the shared computational object explicit.

Additional concrete continuation: query_partition_norm_math_v1.py executed
and saved QUERY_PARTITION_NORM_MATH_V1_CONTROLS.json. A planted valid local
query bank has zero quadratic numerator cross term but source/rest read
interaction -.74235 generated solely by normalization. The tool splits
interaction into cross numerator at full-gain norm and a norm remainder,
both in the SAME head-read frame. No attribution of native final-logit
interaction is inferred. Next native discrimination should preserve this
frame and separate numerator versus norm; no such run registered yet.

Mathematical review0149 complete, next0449. Hourly0206 complete, next0306.
Bothrunnershealthy; no owned job pending. Preserve all Claude work. Goalactive.

## Two query simplifications fail; projected source edits executed — 2026-09-10 01:45 UTC

Previous goal turn PROGRESS: two valid managed native screens, weight-fold oracle,
unfiltered fit/eval manifest, reusable scorer, and executed query-source algebra.
BILIN18_L9_SHARED_QUERY_ROUTER_V1_RESULT.json: A/D pass,B/C fail;36forwards/624seq,
2.284s. One equal-task mean Q/Q2 prototype for L9H1/H4; task means diagnostic.
HasA1/A2 sharedKL.011315/.007206; isA1/A2 .000876/.003011. Only isA1 passesall
basicfidelity. Own-taskmeans fail3/4distributionpanels. Foldedpatternsmax2.69e-7,
relative2.49e-7. RunnerSHA e6adb596e63e44b7b84f911304e4dcd56cec547d0bc1609da78bdca8d86f2edd.

BILIN18_L9_EMBEDDING_QUERY_V1_RESULT.json: A/D pass,B/C fail;32forwards/576seq,
2.180s. Query-only source RMS(alpha*e), e=RMS(Wte[token]); direct skip/reentry
alpha34.5760605212. No contextual query norm retained. KL.0255-.0461,
paired-effecterrors.2705-.3043; allpanels change predictions. Identityquery
full-logitmax1.53e-5, projection6.42e-6, nativecontrast/removalmetricsreplay0.
RunnerSHA b9f36bf756b0f50bbcf68c08efe6e1640e78b92a29b5df06ae54798d3cabc78e.
ResultSHA2d7dabb074e1ad2994490eae54c869da0ff4700c71161bd8417474c925027893.
This rejects the specific raw directembedding query replacement, not every possible
nonlinear token-only query program. No prototype/head/rank/offset/dose rescue.

These screens use the original UNFILTERED has/had and is/was shared-path96pairs,
not the will/had MLP1 cohort. Fit24/eval72, zero token-sequence overlap acrossfit
and each of4evalpanels. Texts opened historically, allnativeerrorsretained. Manifest
BILIN18_L9_SHARED_QUERY_ROUTER_V1_ROWS.json records sole legacy hashdrift inshared
producer; both taskbuildersvalidate andcurrentbackend/checkpointare separatelybound.
Do not claim the old whole-executor authority replayed or refilter tocapable rows.

Concrete continuation: projected_query_source_edits.py executed;5CPUcontrols
pass5.55e-16. Pre-attention9 has19sources: directembedding plusattention/MLPwrites
at0..8, each multiplied by transportedlambda coefficient. Store their four-head
queryprojections p_s and sourceGram G. Gainsz givep=sum z_s p_s,rho=z^T Gz and
headquery=p/sqrt(mean_head(p²)+eps_head*(rho/1152+eps_residual)). This supports
source-edge removals/jointedits withrecomputednorm, not global upstreammodulecuts.
Native coefficientauditmatchesalpha. Initialtoy crossGram liveness.007922missed
.01bar; plantedreader scaled.1 beforeanynativebankrun, giving.1804, barunchanged.
No native19-sourcebank captured yet. Gramnorm cancellation must beaudited; actual
denseeditstate10089scalars, hypotheticalpacked9918, bothlargerthanone1152residual.
All nativeinitialization/weights/background remaincharged. No identified/adoptedcircuit.

Next scientific object is contextualqueryproducer attribution usingthese explicit
sourceedges; anotherfixedprototype or globalrank/square-bankobstructionisdemoted.
The new explanation shared_attention_query_rules_2026-09-10.md starts with the
native outcomes and gives the formulas, data, prices and precise limitations.
No ownGPUjob pending; bothrunnershealthy. Latesthourly0106(next02:06); regular
mathematicalreviewstilldue01:49. Goalactive. Originalbilinearhandoff controls,
not stalegoalwordingbetter_math_ideas. Preserve allClaude work.

## Native operand-domain null and information bound — 2026-09-10 01:09 UTC

Previous goal turn PROGRESS: one managed causal-equivalence screen and one executed
CPU information bound. BILIN18_MLP1_OPERAND_DOMAIN_V1_RESULT.json is valid, terminal
`symmetric_function_loses_native_operand_interventions`; A passes, B/C fail.
Runner SHA24ebff62eb300c8937bea70ee8a0ee630ff68d878cf65c7d23bd4f071bd20f23.
Result SHA f83913271aea0b32569458512a817332900ed27d612843f5e2a7580d64738ac7.
Managed01:06:14→18,2.083s,16forwards/512sequences, same48+16openedpairs, no fits.
Symmetric mixed output averages native Left-donor/Right-base and reverse. Actual
independent branch hooks yield distinct effects: full-logit causal errors .234-.353
on targets, versus .01. Both-operand/direct-donor final replay0; local oracle max
.000578. Hooks restored. Independent-operand interventions expose the antisymmetric
part that vanishes on ordinary tied inputs. No existing symmetric rewrite restricted
to tied-input edits is invalidated; no universal native-neuron API is imposed on
future circuits. Translation can retain operand identity, with its cost charged.

Concrete continuation: audit_operand_identity_information_v1.py executed on saved
margin effects. Any common prediction for Left/Right has exact finite-cohort least-
squares floor13.97%temporal/14.54%iswas/20.85%P relative joint RMS. It explains
98.6%/98.9% of target squared error of the actual symmetric candidate. Identity
closure0. This is not a KL/full-logit/population bound or a bound for typed edits.
See OPERAND_IDENTITY_INFORMATION_V1_AUDIT.json and the updated explanation
shared_nonlinear_factors_after_removal_2026-09-10.md. No new trained circuit adopted.

Hourly review0106 is complete; next02:06, regular mathematical review still01:49.
Three prior native screens averaged over10min receipt gaps; current claim-to-result
is about3min through shared capture/row/scoring reuse. No new forward framework.
No own GPU job pending; Claude v455 began01:06:18, do not interfere or duplicate.
Continue original bilinear handoff's explicit shared producers and consumer branches;
command-mode path parked. A further global square-bank/rank obstruction was demoted
as low circuit information. Goal remains active; all native weights remain charged.

## Weight removal fails; shared nonlinear factor math executed — 2026-09-10 00:59 UTC

The previous goal turn is PROGRESS. The managed removal runner completed at 00:49:18,
with a valid instrument, restored original weights/bias, 16 forwards/512 sequences,
and 4.084 seconds executor time. Its immutable result is
`BILIN18_MLP1_WEIGHT_DEFINED_REMOVAL_V1_RESULT.json`, terminal
`task_reader_quadratics_not_independently_removable`. Own signed contrast retention
is .738652 temporal and .405691 iswas; P controls change 3/2 endpoint predictions,
joint 7. B/C fail. Close this homogeneous reader split; no rank/site/dose/offset rescue.
Runner SHA: 4149ae5e98693e0e360e0270e19bf16b9e1f405c6abc6319ac73c6c7ad188e15.

The concrete continuation is CPU math already executed in
`shared_quadratic_factor_math_v1.py`: eight exact symbolic controls distinguish a
shared nonorthogonal square-feature library from the excluded smaller linear-plus-norm
state. A planted pair has full rank and invertible raw commutator yet shares two squares;
a third consumer fails the necessary inverse-reference commutator criterion. A raw-dot
attention gauge fails normalized routing. No trained factor discovery is claimed.
The user-facing derivation and removal summary are in
`explanations/shared_nonlinear_factors_after_removal_2026-09-10.md`.

Continue the user's original bilinear handoff/pilot joint read–route–write direction,
not better_math_ideas or the parked is/was command-mode loop. Shared producers and
consumer-specific branches need separately specified interventions; all normalization,
adapters and opaque weights remain charged. A native square-bank test is NOT yet
registered or enqueued. Do not turn this toy into a native positive or launch a rank
sweep. Both runners are healthy, with no owned GPU job pending. The program goal
remains active. Review clocks remain 01:05 hourly and 01:49 mathematical.

## Cross-program null; fixed-weight removal is the next requirement — 2026-09-10 00:43 UTC

Previous goal turn is PROGRESS. BILIN18_MLP1_ATTENTION_CROSS_PROGRAM_V1
finishes validly in2.237s with19forwards/528sequences; pre-runaddendum kept
controlbatchgeometry16, replacingplanned14forwards/32-rowduplication. A/Cpass,
Bfails. Cross .04111/.11447 versusfull .52002/.57473 signednativeeffects,
effectvectorerrors .91927/.79564. P KL4.638e-5/6.605e-5 withzero flips passes
the ORIGINAL ordinaryparent KL+1e-6 bars. Do not promote another descriptive
component. Parent replay0; componentalgebra1.55e-11, norm6.18e-7,
reader6.35e-4, fullcompiledlogits1.53e-5. RunnerSHA
f132c485bc2b953b7324775cdc485eb99537fedb966f7fd661ba7171cc0053f8.
ResultSHA6131b05a703f96c7b93bfd15a1ae5ba94005c32ec14a6cf86a2aa06571c80a4c.
CPU singletoneffectsum relativeerror .3894temporal/.1045iswas: downstream
nonlinearity matters, not proof of failed nonlinearcomposition. Allweightspriced.

Next BILIN18_MLP1_WEIGHT_DEFINED_REMOVAL_V1_PREREGISTRATION.md addresses
actualremoval. Freeze sameC/dualwriter, natural homogeneousquadratic component
withphi=(Left n)*(Right n), originalDown_biasretained. StaticDown edit
W_without_t=W-dual_t(C_tW), singles/joint, preserves opposite localreader.
Interchanges alone leave constant reallocation unidentified; this fixes a
weight-definedzero and tests it, without offsetfitting or newdata. Seven CPU
controls inquadratic_reader_weight_removal.py pass, identity3.55e-15, including
restoration/commutation and affine-origin counterexample. Native runnerpending.

Use same48target+16Ppairs, bothbase/donor, native/A/B/AB =16forwards and512
sequenceevaluations. Biasunchanged, checkpointneverwritten, in-memoryweights
scopedandrestored aftereacharm. A:weight/output FP64identity1e-8/1e-9,
nativeMLPoracle1e-3/1e-5, identicalinputs andparentnativecontrast/top1replay.
B:owncuecontrastabsprojection<=.1 andRMSretention<=.25; othertask contrast
relativeerror<=.1 andendpointtop1unchanged. C:bothremoved suppressbothtasks,
and everyPsingle/joint KLmean<=.001,p99<=.01,zero endpointflips. No rank,
dose,site,offset,biasrepair. This is a newremoval test, not relaxation of the
failedinterchangebar. Reuseexistingnativecapture andsavedrowmanifest/scorer.
Bothrunnershealthy; noownjobpending. Clocks01:05hourly/01:49math. Goalactive.

## Dual-reader behavioral screen fails P criterion; actual bilinear cross program next — 2026-09-10 00:24 UTC

Previous goal turn is PROGRESS. Managed BILIN18_MLP1_DUAL_READER_NATIVE_V1
finished validly in2.430s,18forwards/576sequences. A/B pass,C fails. Nativezero
andjoint replay0; ownsigned effect .5200/.5747 versusordinary.5229/.5831,
cross/ownRMS .00454/.02760. No Ptop1flips, but temporaldual KL.000167102
exceedsordinary.000157899+1e-6. Preserve this miss; no promotion/thresholdrepair.
RunnerSHA12f00093193f40dbc5c141f5208fe71189e0e911ccb6c397c241a39e7038001f.
Rows saved in BILIN18_MLP1_DUAL_READER_NATIVE_V1_ROWS.json with exact prior
hashes; onebuildpermodule reducedpreparation to8.174s. No refit or datachange.

Next BILIN18_MLP1_ATTENTION_CROSS_PROGRAM_V1_PREREGISTRATION.md fixes
pre-attentionstream u and full newattention1write a. CDown folds outputreader;
three numerator terms pre/cross/attention share FULL norm denominator. This
is a computational output-node edit, not sourceknockout or independent norm.
Old rungs487–491, failed readerpath492, September2normalizationcorrection and
MLP10source-pair507 checked. Native implementation must retain that distinction.

mlp1_attention_cross_program.py implements components and read-only earlystream
capture by wte/attention0/MLP0/attention1/MLP1hooks with nativelambda reentry.
Seven CPU controls pass, algebra8.53e-14. Native runner pending: reuse saved
48target+16Pcontrol rows, previous capture/scorer, source frames during4native
forwards. Five patcharms percohort: fullnative dual, pre, cross, attention,
sumcompiled. Controlpatchcohort duplicates16rows with A/B labels/uniqueIDs,
using cached16rowbase/donor references:14forwards/528sequence evaluations.
B requirescross>=90%fullsigned effect AND<=.15effectvectorerror bothtasks;
C keeps the ORIGINAL ordinary P KL+1e-6 andzero-flip bars, no fallbackterm
promotion. No models/weights fitted, native suffix recomputed, allweightspriced.
Bind newprotocol/primitive/parent sources before managedenqueue. No own job
pending; bothrunnershealthy. Goal active; clocks01:05hourly/01:49math unchanged.

## Exact quotient obstruction complete; two-reader native selectivity next — 2026-09-10 00:05 UTC

Previous turn is PROGRESS: managed native weight audit, exact coefficient
certificate, new mathematical intervention tool and seven native-hook controls.
BILIN18_MLP1_NORM_OBSERVABLE_OBSTRUCTION_V1_RESULT.json is valid:
2.857s, zero forwards, determinant12024 modulo65521; independentCPU replay
12024 in1.279s. Saved matrix SHA bdca9a6f0ca638482cefb7fdf08168745df0cb8eb10cb7b663c120f177724729.
This rules out every proper linear-coordinate-plus-norm representation of the
fixed dyadic-real reader functions over all real inputs, not nonlinear circuits
or domain-specific representations. No additional input-rank/module/pair scans.

Next output-side two-reader constraint: D=C^T(CC^T)^-1, CD=I. Its task blocks
give minimum-norm local writes that preserve the other reader. Saved MLP1
reader analysis BILIN18_MLP1_DUAL_READER_EDIT_V1_RESULT.json passes instrument
and prospective2xwrite-gain bar; actualworstgain1.056784,condition1.3985,
maxediterror8.22e-15. This is local edit-tool feasibility, not semantic discovery
or final behavior invariance. No new native weight savings.
Native row audit now confirms all64base/donor alignments, zero target token
sequence overlap with reader-fit data, and exact restoredfitrowhashes. Existing
texts remain opened. BILIN18_MLP1_DUAL_READER_NATIVE_V1_ROW_AUDIT.json binds
expected row hashes. Legacy repeatedbuild_rows cost dominates CPU preparation;
build each unchanged builder once per process or freeze row JSON for native
preflight. Do not refit/readjust selection to speed up the audit.

Active protocol BILIN18_MLP1_DUAL_READER_NATIVE_V1_PREREGISTRATION.md fixes
48opened temporal-v12/iswas-v11targets (24each) and16temporalPcontrols,
18nativeforwards, no fits. Both counts restored CPU. Compare ordinary/dual
single-task writes, dualjoint versus orthogonaljoint, zero replay. Hook primitive
dual_reader_output_intervention.py passes7controls. Native runner integration
pending; reuse existing v1.cap, backend.native, atlasrun.states, comp.margins,
das.head_logits and exact old row builders; no new executor stack. Scalar
joint final effects need not add. Other-task-specific controls and freshOOD
remain untested. Bind sources and prereg before managedenqueue.

Updated explanation weight_tensor_two_circuit_math_2026-09-09.md has proof,
exactresult and within-module split derivation. Hourly0005complete; next01:05.
Regular math remains01:49. Both runners healthy, no owned GPU job pending.
Goal active; concrete continuation is implemented/controlled native edit
primitive plus registered test. Preserve all concurrent Claude files.

## MLP1 full-input audit valid; exact norm-quotient certificate queued — 2026-09-09 23:58 UTC

BILIN18_MLP1_JOINT_READER_WEIGHT_V1_RESULT.json is valid:72.514s,8forwards,
300sequences, FP64 max1.32e-11 and nativeFP32 max4.48e-4/relative3.90e-7.
Full input support1152/1152,min/max.03115. Function principal cosines
.4251/.1944/.0914/.0506. Reader JSON saved; no refit needed. All4608native
products active under both task readers; no semantic sharing or adoption.
Different from old top2reader overlap; rank4 noisy behavioral null unchanged.

Logical scope refinement: full support excludes a smaller linear-coordinate
numerator representation only. A retained norm permits alpha_k I terms, so
Q=I is a counterexample to a blanket linear-plus-norm impossibility claim.
New protocol BILIN18_MLP1_NORM_OBSERVABLE_OBSTRUCTION_V1 fixes first temporal
and first iswas reader. A nonzero exact commutator determinant modulo65521
certifies no proper linear-plus-norm state for all real inputs, even with
arbitrary decoder. Zero is inconclusive; no pair/module/rank scan. Dyadic
weights mapped exactly, bounded integer FP64 GEMMs, GPU elimination plus
independent CPU integer determinant and saved matrix. Six CPU controls pass.
Managed runner enqueued at45a2b36dbdc5ab6039cce46366f4a2cebc57ffdccaecbe2739641f15ae836ee0,
ops/run_bilin18_mlp1_norm_observable_obstruction_v1.py under bilinear_quotient.
Zero native forwards; no refit, training or weight changes. Inspect live result
and queue before any action. Interpret it before choosing nonlinear factor
work. Do not change or duplicate queued source. Goal active; preserve Claude
files. Previous turn/continuation is PROGRESS. Clocks hourly0003/math0149.

## Joint-reader native audit queued; edit-response math executed — 2026-09-09 23:49 UTC

MLP1 runner now exists and is managed-queued behind Claude v449:
ops/run_bilin18_mlp1_joint_reader_weight_v1.py under bilinear_quotient,
SHA cd9b4efd8ffe9fa8d4b76d2c1113e5de496c695dc20bb9313ac58dc3680abdf2.
Frozen27-source manifest and runner published in de10f5189. Check queue/log
and BILIN18_MLP1_JOINT_READER_WEIGHT_V1_RESULT.json before any action;
never edit/duplicate queued source. Eight forwards restore pooled/task readers,
all8 full1152-dimensional quadratic forms, normalization/native replay,
function principal angles, full input support spectrum, literal native co-use.
No trained result at this checkpoint. No new rank/candidate scan authorized.

Additional CPU derivation/control is complete in
bilinear_joint_edit_observable_reference.py: fixed additive writers W admit
closed nonlinear edit state p=xQx, g=W^TQx, h=W^Tx, rho=||x||². For8readers,
2writers,96D random full-rank forms,27scalars reproduce edits with1.71e-13
output error;7checks pass. This is an edit-response engine with native
initialization and weights retained, not an independently extracted circuit.
Explanation weight_tensor_two_circuit_math_2026-09-09.md includes proof,
explicit cross-writer/norm coupling, price and limits. Mathematical progress
continues during GPU serialization. Interpret native result next; choose
writer interfaces only from existing task evidence. Goal remains active.
Regular hourly0003/math0149 Sep10 clocks unchanged. Preserve concurrent files.

## User redirects to weight-based two-circuit mathematics — 2026-09-09 23:37 UTC

User reports insufficient progress in current iswas direction and explicitly
requests mathematics from the original handoff/pilot, bilinear MLP/attention
folding, and constraints from two circuits using the same module. Pause the
command-mode/matcher sequence; TWO_MATCHER_COMMAND_PRODUCT is parked unrun.
Follow bilinear_circuit_reconstruction_codex_handoff.md and appended structural
criterion, not stale better_math_ideas goal wording. Goal active; prior turn
progress includes a valid numerical repair, a scoped null and new mathematical
reference plus saved trained-weight audit.

Newest explanation weight_tensor_two_circuit_math_2026-09-09.md is first in
README. Re-read its exact normalized-observable and additive-edit construction:
Q_k=sym[L^T diag((CD)_k)R]; U contains both reader-form ranges and writer maps;
state=(U^T x,||x||²). Explicit norm update preserves independent/joint writes.
Eight CPU controls pass on a rotated64D planted module using6linear features
plus1norm scalar, error4.44e-15. This is no trained reduction claim.
Saved v17 MLP11 factor audit: antisymmetric expanded-tensor norm40.4% is
functionally zero; fixedtop32 symmetric residual16.4% versus old18.9%.
Both saved attention writer maps folded through the same numerator tensor;
joint cross identity2.00e-11, direct/symmetric2.38e-13. Scope is savedU8 only;
v18 broader input-interface failure remains binding. No native weight saving.

Next BILIN18_MLP1_JOINT_READER_WEIGHT_V1_PREREGISTRATION.md restores the existing
pooled rank8 projector and task rank4 readers via the COMPLETE rank-ladder
recipe, not the older top2 default. FixedMLP1, bothtasks, full1152-dimensional
input quadratic forms withbias/nativeepsilon, eight native capture forwards
expected. Reuse subspace_weight_atlas, pooled_response_projector and ladder;
serialize reader maps and inspect joint function/input/factor sharing separately.
No native runner authored/enqueued yet. Mathematical implementation and controls
exist; the registered restoration/weight-object test is the next active work.

Upstream mixed-state V1 remains invalid (.000122 FP32 roundoff exceeds absolute
bars); V2 per-coordinate/command-mode half-ULP audit passes with identical
outputs. Read mixed retention1.03675/1.00586, outputmixed .79251/.82384,
separate-mode changes1.65–8.50%; common upstream-variable B/C fail. Both immutable
results and rounding controls retained. No boundary scan or candidate rescue.
The user-requested daily update remains research_update_2026-09-09.md, cutoff2306.
This extra math session is user-requested; regular clocks remain hourly0003 and
math0149 Sep10. Both managed runners healthy, no own GPU job pending; preserve
concurrent work. Publish only owned files. Full structural goal remains unmet.

## Native shared read needs joint routing; daily report — 2026-09-09 23:14 UTC

Previous turn PROGRESS, and this continuation produces a managed native read
screen plus the user's requested high-level-first daily explanation. Follow
bilinear_circuit_reconstruction_codex_handoff.md and its appended structural
criterion; stale better_math_ideas goal wording does not control. Goal active.

BILIN18_ROUTED_SHARED_COMMAND_INTERACTION_V1 completes23:06:28 UTC in1.880s:
32forwards/128sequences,192extra local contractions, actual545902902params.
A/B pass,C false. Native maxabs2.14e-5/relative1.33e-7, convolution1.42e-14,
future mixedread0, payloadmixed7.11e-15,14controls pass. Latequery mixed/shared
RMS .03247/.03579; crossed-only mixedwrite error .77813/.75007 FIT/HOLDOUT.
Joint routing is material, no final-effect sufficiency or structural reduction.
Runner SHA c37eb7776bc41d7c3d08d2acb4bed8dbedf02c208e6e324f6f9aaf4767ca963a.
Result and helper/runner are immutable; do not edit hash-bound primitives.

User-requested explanations/research_update_2026-09-09.md covers the early
pilot through23:06, main afternoon work from1354, explicit terms/computations
and positive endpoint evidence with its limits. Linked first in README.
Canonical shared_first_value_payload.md updated to completed read result.

Next BILIN18_UPSTREAM_MIXED_COMMAND_STATE_V1_PREREGISTRATION.md: remove only
the two-command mixed residual mode after zero-based block8, preserving
mean/single modes, original embedding and first-value memory. Test whether
one common upstream component supplies the four shared readers' interaction
and selective output coupling. Fixed boundary/heads/cohort, no scan. New
mixed_command_state_intervention.py implements scoped tuple-preserving hook
and mode removal; six algebra controls pass. Native hook control/integration
and64forward managed screen pending. All four-cell prefix/weights priced,
no standalone semantic-variable or reduction claim. Prior-art check separated
this from old MLP11 occupied-factor rescue.

Both runners healthy; no own GPU job pending. Preserve concurrent files.
Latest clocks2303hourly/2249math; next0003/0149 Sep10. Concrete continuation
receipt is registered upstream-state test with implementation underway.

## Shared producer is additive; downstream response is coupled — 2026-09-09 23:03 UTC

The bilinear reconstruction handoff and its appended criterion remain the
controlling direction, overriding stale better_math_ideas goal wording.
Three new scored receipts: BILIN18_VALUE_COMPONENT_FACTORIAL_V1 (GPU3.238s),
BILIN18_SHARED_RESPONSE_COMMAND_MODES_V1 (GPU1.765s), and
BILIN18_SHARED_VALUE_PRODUCER_INCIDENCE_V1 (CPU1.061s). First two are valid
nulls; incidence is an exact positive producer certificate on795 positions.

Shared/context effects are not opposing (cosines .381–.943); independent
contributions fail (all-token interaction10.4–12.0%). Shared-only deletion
also fails KLmean .001504/.001407, so no contextual-only fallback.
Best command-independent full response has relative error .374/.373;
best additive-command response has an M11 floor .0728/.0721. At the later
iswas query these bounds are .478/.514 and .137/.154. L2 bounds are not KL
bounds. Native/parent replay and future-command zero pass. All native errors
and545902902 parameters remain. These opened worlds are not fresh/OOD proof.

Exact token incidence shows ANY token-only shared-value producer has U11=0.
Interaction in its removal response arises downstream. Next registered
BILIN18_ROUTED_SHARED_COMMAND_INTERACTION_V1 captures native shared reads
at the fixed L9H1/H4,L11H3,L15H5 union and both command queries. Exact
S11=P11U00+P01U10+P10U01 distinguishes joint routing from crossed routing
and payload. Primitive shared_read_command_convolution.py passes eight toy
controls; trained capture/scorer is still pending. Reuse parent native
forward and command_response_modes.py, do not mutate hash-bound old helpers.
Managed32forward run only after native-capture implementation and preflight.
The primitive's crossed-only fixture was strengthened before trained work
because norm1 did not exceed its strict>1 negative-control bar; bar unchanged.
Non-additive effects do not disprove composition with an explicit joint term.

Canonical dossier: explanations/shared_first_value_payload.md.
New mathematical review2249 and hourly review2303; next clocks01:49 Sep10
and00:03 Sep10. Both runners healthy, concurrent Claude v447 completed22:53;
preserve its files. No own GPU job remains. Concrete continuation is the
registered reader-convolution protocol with implemented/controlled algebra;
next integrate scoped native capture and streaming statistics. Goal active.

## Shared-only payload fails on full bilin18; component factorial next — 2026-09-09 22:35 UTC

The bilinear reconstruction handoff and appended structural criterion control,
overriding the durable goal's stale better_math_ideas wording. This turn is
PROGRESS: trained query-gate certificate, managed full-model screen, and a
new implemented causal intervention primitive. Overall goal remains active.

QUERY_SHARED_LINEAR_GATE_V1 saved CPU .00436s: first exact3x3 determinant
nonzero, so no shared linear gate for all58 centered route forms in the fixed
small-model query interface. Eleven controls pass. Close these simple query
block/squared/single-gate grammars; no factor-count/output-selection rescue.

BILIN18_SHARED_FIRST_VALUE_ONLY_V1 managedGPU3.228s tests the actual545.9M
bilin18 and fixed licensed union L9H1/H4,L11H3,L15H5. All128 already-opened
dual-command sequences; full token count is3180 (1572FIT+1608HOLDOUT),
all50304 outputs,128forwards/512sequence evaluations. Nine controls and A
pass; fidelityB and compositionC fail. All-four contextual-value deletion:
KLmean .048704/.042429, p99 .760107/.611856; joint interaction/effect
.18940/.17981. Paired command-vector errors .828/.853 temporal and .634/.605
iswas. Native full-vocabulary accuracy is lower than old answer/foil capability;
retain native errors. No head/group/lambda fallback, no589824-weight saving.
Result SHA447e72dbe5b032976c6aaac902fd8aea587eb3a4c2cc9fa91ca382975401014c.
Runner SHAba85983fc445adb1f858f857af9a71827f1fcff52a956ed9e82edc1677506b63,
managed log completed22:29:19 UTC. The initial enqueue failed only model-free
__file__ lint; module-scope path resolution repaired it, no bars changed.

Next BILIN18_VALUE_COMPONENT_FACTORIAL_V1_PREREGISTRATION.md: consumer-local
context/shared/both deletions on the same fixed union and opened cohort,
with all upstream/downstream states recomputed. Test signed opposing command
effects and additive joint contribution; no new replacement nomination.
shared_value_component_intervention.py passes seven controls, including
parent contextual-cut replay, direct whole-head-zero, live arms, method/hook
restoration and exception cleanup. Trained factorial not yet run. Integrate
with parent's streaming scorer and native forward, bind hashes and enqueue
129forward/516sequence audit through managed runner. No new executor stack.
All545902902 native parameters remain priced. The smaller model's387968
export count must not be confused with this full-model parameter count.

Dossier explanations/shared_first_value_payload.md is first in README;
query history join_write_response_curve.md updated. Both Supervisor runners
healthy; no own live job remains. Preserve Claude files and queue ownership.
Clocks: next math22:49 UTC, hourly22:58; latest reviews19:49/21:58. Do not
emit early duplicates. Stage/push only owned artifacts. Continue from evidence.

## Coupled query block certified; shared multiplication test next — 2026-09-09 22:17 UTC

The user's bilinear reconstruction handoff and appended structural criterion
control, overriding the durable goal's stale better_math_ideas wording.
This turn made progress through one native coefficient audit and one exact
saved-artifact bound. No extracted structurally smaller circuit; goal active.

QUERY_COMMON_SQUARED_CHANNELS_V1 CPU .087s: fixed first IID world/order,
forward hop3, all58 centered forms and3x3 norm Gram. Native8.88e-15,
compiler1.78e-15, norm5.55e-17, lineage8.88e-16. Exact positive Gram;
first lexicographic pair I:0/I:1 obstructs simultaneous congruence. No
three independent squared channels under any invertible coordinate change.
Coefficient artifact SHA3f195fe72cc86a62b96ba08733c89e4768a6462608b97a0cbf4483fe1b4e3151.

QUERY_COUPLED_BLOCK_BOUND_V1 CPU .011s: exact commutant rank8/nullity1
in11 equations, so no proper common1+2 block split either. Six controls pass.
Exact trace-rational commutator bound exceeds1%: numerical lower bound
.0153125 per-form relative Frobenius error after whitening the fixed Gram.
This is not a route-logit or KL error bound. Both exact certificates concern
stored FP64 coefficients; native agreement is not an ideal-real interval proof.
All387968 opaque export coefficients and root/context production remain priced.

Next QUERY_SHARED_LINEAR_GATE_V1_PREREGISTRATION.md tests a distinct circuit
operation: shared multiplication f_k(z)=g(z)*h_k(z) across the contracted
I/J readers. Block inseparability does not exclude such reuse: the planted
(z0²,z0*z1,z0*z2) example is irreducible as a block family yet shares z0.
shared_query_gate_reference.py implements its controls and exact necessary
condition: one nonsingular3x3 form rejects a common linear factor; all
singular forms are inconclusive. Trained determinant outcomes remain unopened.
Next implement/execute the small hash-bound saved-coefficient audit, then
interpret without increasing gate count or selecting outputs to rescue it.
Prior raw MLP-factor scans remain closed. No new GPU job required.

Canonical dossier explanations/join_write_response_curve.md updated. Both
Supervisor bqrunners healthy at22:08; preserve concurrent Claude work.
Review clocks unchanged: latest hourly21:58, next22:58; mathematical19:49,
next22:49. Use actual elapsed time before any new review. Push owned files.

## Query-root separation bounded; exact common-channel test next — 2026-09-09 22:02 UTC

The user's bilinear reconstruction handoff and appended structural criterion
control. This goal turn made progress: two native CPU screens, one saved-row
function-class bound, an implemented exact rational primitive and the2158
hourly review. The overall structural circuit goal remains active and unfinished.

ENDPOINT_ROUTE_QUERY_LINEAGE_V1 CPU4.847s,512 opened requests: native1.42e-13,
saved9.24e-14,lineage2.13e-14,zero-query routes0,10 controls. Fixed I-local/
J-complement assignment fails; forward-hop3 task ratios .0284/.00161 for I
and .0963/.0357 for J. Selected-vector errors .922–.945 I,.700–.734 J; mixed
query terms remain large. Payload origin does not prove an entity identity.

ENDPOINT_MIXED_QUERY_INPUTS_V1 CPU5.048s,512 requests: native7.11e-14,
local replay0,root closure5.11e-15,mixed partition4.62e-14. Entity x complement
splits exactly into entity x task-token (LH) and entity x document (LD).
Both fixed dominance hypotheses fail; neither a pure route/root nor a selected
population is adopted. Gold probabilities of additive logits need not add.

QUERY_ROOT_ADDITIVITY_BOUND_V1 CPU .0257s: four-corner triangle inequality
bounds arbitrary additive programs over root blocks. All forward-hop3 I/J
bipartitions have fixed-native-route relative lower bounds7.8–9.4%;84/96
partition groups exceed1%, no floored references. Exact stored-table replay
4.62e-14. The bound uses a fixed native-route RMS denominator on binary Q-root
gates, not the prior per-intervention denominator or a whole-model impossibility.
FP64 RHS is not interval-certified;4e-9 contrast slack does not change targets.
Stop further root-dominance probes. Existing I nomination remains FAILED
(.48058 IID below .5), and the16/63 backup census is not a repaired nomination.

Next action: QUERY_COMMON_SQUARED_CHANNELS_V1_PREREGISTRATION.md. Test whether
any invertible coordinate change separates the norm and all centered I/J
readers into three shared squared channels. First fixed context only:
N.populations()['iid'][0],token row3 (forward hop3). Compile its3x3 Gram and
all58 symmetric route/output forms, validate held-out amplitude vectors using
existing Q/interaction/native oracles, then run exact dyadic congruence scan.
simultaneous_congruence_reference.py is implemented; nonorthogonal positive,
noncommuting negative,singular/indefinite controls pass. Trained matrices are
not yet compiled or opened. Any exact obstruction rules out this grammar even
locally; no jitter,new context,rank/feature count or output subset rescue.
Jiang/Li Theorems3.2/3.3 mapped in the protocol. This does not rule out general
coupled bilinear features or prove any native structural reduction.

Hourly2158 complete:11 receipts this hour,median5.41min all/5.98min native
screen intervals; focus/novelty PASS,ceremony PASS on latency proxy only.
Authoring/publication still dominates seconds of compute; reuse machinery.
All387968 opaque native constants,gates and remainder remain priced. Dossier:
explanations/join_write_response_curve.md. Next mathematical22:49 UTC, hourly
22:58 UTC. Both managed runners healthy; concurrent Claude v441/v443 work
must be preserved. No own GPU job pending. Stage/push only owned artifacts.

## Exact endpoint routes show backup; query-lineage test next — 2026-09-09 21:40 UTC

The bilinear reconstruction handoff and appended structural criterion control;
do not revert to better_math_ideas. This goal turn made progress through three
native CPU screens and one saved-output census. The complete structural circuit
objective remains active and unfinished; no failed gate was relaxed.

OVERLAPPING_JOIN_KEY_GAIN_V1 CPU1.647s,512 requests: exact native1.14e-13,
saved replay/source reuse0,14 controls. Direction-dominance nomination fails:
forward direction/full task ratios1.690/1.410, backward1.059/1.059. Gain partly
offsets direction damage; full-vector direction errors .220/.234 forward and
.125/.136 backward. Carry both direction and normalization in the explanation.

OVERLAPPING_JOIN_ADDRESS_PRODUCERS_V1 CPU3.540s,512 requests: native1.56e-13,
saved1.42e-14,reuse0; exact quadratic three-way closure2.84e-13. Fixed E-forward/
O-backward address split fails. E contributes almost none of the forward task
effect; existing previous-key path O accounts for both query effects. E+O
full-vector errors .119–.165 reject a sufficient two-path replacement. No
producer expansion or renewed origin-field interchange claim follows.

ORIGIN_ENDPOINT_INTERACTION_V1 CPU1.859s,512 requests: exact additive node
I = native - O_key_cut - F_value_cut + both_cuts. O through L2H2 keys interacts
with direct endpoint F through L2H1 values at the shared source. All final
heads retained, native gains fixed. Seven controls/native correspondence1.14e-13.
The node knockout is an expanded-read intervention, not a physical L2 source
removal. Circuit nomination FAILS: IID target loss .48058 misses .5; OOD .66782
and all other query/hop selectivity panels pass (max mean abs goldchange .01444).
Do not rescale, reselect cases or promote the near miss to a passed screen.

ENDPOINT_ROUTE_PARTITION_V1 CPU .0246s uses saved arms to split total endpoint
read T into I and remaining read J. Exact knockout/partition replay1.14e-13.
Among63 initially correct target cases,16 tolerate either single-node knockout
but fail the joint:10 IID/6 OOD. IID32 census:13 only I single fails,7 only J,
1 both singles fail,10 joint-only,1 native error. OOD32:21 only I,4 only J,
6 joint-only,1 survives all. Full endpoint-read removal gold losses .933/.951.
This is backup evidence, not a new passing nomination or identified algorithm
for J. Full all-hop/order/correctness census preserved.

Next claimed action: ENDPOINT_ROUTE_QUERY_LINEAGE_V1_PREREGISTRATION.md. Use
existing frozen payload lineage to split final query into original query-entity
root L (position49) and complement C. At native query gain, keep all source
features/skip fixed and evaluate Q from L+C,L,C,0. Decompose each exact I/J
route into local, complement and mixed-query terms. Fixed task nomination tests
I-local versus J-complement without assuming payload origin proves entity
identity. Small native Q-port oracle and bounded CPU screen next; derived
outcomes not opened. No GPU job pending for this claim.

All387968 opaque export constants/native gates/remainders remain priced.
Canonical dossier: explanations/join_write_response_curve.md. Next hourly21:58,
mathematical22:49 UTC. Preserve prior metadata overlays and concurrent work;
stage/push only owned artifacts. Overall goal is unfinished.

## Overlapping joins couple through keys and values; direction/gain audit next — 2026-09-09 21:19 UTC

The user's bilinear reconstruction handoff and appended structural criterion
control. Previous goal turn made progress: three completed bounded CPU receipts,
shared two-write response primitive, canonical explanations and next protocol.
The full structural circuit objective remains active and unfinished.

ENDPOINT_RENAMING_CAUSAL_BALANCE_V1 .062s, accounting3.55e-15; invariant
physical-removal hypothesis fails. Hop3 removal changes .360/.309 IID/OOD;
native output change is1.95/2.48 times removal change, not the proposed smaller
compensated total. Paired native query JS floors .02533/.05155 exceed .001.
Do not pursue another constant/invariant gate or endpoint amplitude.

OVERLAPPING_JOIN_NORMALIZER_V1 5.224s CPU: two adjacent
joins arrive at the same late binding;512 opened requests/32worlds/two orders/
two query origins/four hops. Fixed L2H1/H2 writes reuse exactly across queries.
Native/axis correspondence1.42e-13/1.03e-13. Native capability passes, but
selectivity and shared-normalizer-only composition fail. Backward join removal
changes forward-query goldP mean absolute .176/.351. Both-cut interaction
prediction errors4.784/4.693 forward and2.322/2.178 backward. Mixed numerator
terms matter; no head/scale/field rescue. Complete univariate degrees retained.

OVERLAPPING_JOIN_PORTS_V1 2.622s CPU, same512 requests, full8-subset native
K1/K2/V factorial. Oracle1.28e-13, saved full physical-cut replay1.42e-13,
query reuse0. Fixed key-pair candidate overshoots forward-query gold effect
by ratios1.351/1.176; backward-query ratios1.054/1.043. V-only partly offsets
key damage. Overall shared-address nomination and sufficient-port hypotheses
fail. Full-vector errors .187/.187 forward, .092/.111 backward. Do not adopt
a backward-only subset or call source-port hybrids residual-space edits.
All387968 opaque export constants/background remain priced.

Next concrete action: OVERLAPPING_JOIN_KEY_GAIN_V1_PREREGISTRATION.md. Before
interpreting keys as address information, split K(x-v) direction from shared
RMS gain g(x-v), retaining both query origins and all heads/hops. Exact2x2
native/direction/gain/full-key factorial; use existing source-port executor,
cohort and scorer with small native projection hooks. Protocol fixes direction-
dominance task bars and zero/equal-norm/radial controls. No derived outcomes
opened and no job running for this claim yet; implement bounded CPU audit next.

Dossiers: explanations/join_matcher_factors.md and join_write_response_curve.md.
Next hourly21:58, mathematical22:49 UTC; prior2058 ceremony failure repaired
with shared helper/executor reuse. This block produced three CPU receipts
within roughly12 minutes of research before publication; no GPU work needed.
Preserve immutable result sources, metadata overlays and concurrent Claude work.

## Degree and grouped-path nulls; physical causal balance next — 2026-09-09 21:03 UTC

The bilinear reconstruction handoff and its appended structural criterion
control. The durable goal remains active; no structurally simpler sufficient
circuit was identified in this turn. Do not revert to better_math_ideas.

JOIN_WRITE_READ_DEGREE_V1 completes4.384s on768 requests. Exact response and
removal partition pass1.28e-13/1.33e-14; fixed forward-C1/backward-C2 hypotheses
fail. Hop3 native-scale errors are7% forward and53–65% backward. No degree,
scale, head or task rescue. All387968 opaque export constants remain.

CROSS_LAYER_ENDPOINT_TRANSPORT_V1 then executes1.859s on CPU,384 paired cases
across32 opened worlds/three forward orders/four hops. Whole write/read oracles
close7.11e-15. At hop3 the writer has27/48 IID and30/48 OOD negative-cosine pairs;
including all final readers removes those negative cosines, with means .998735
and .999340. Nevertheless grouped-path changes .22525/.21721 and live-removal
prediction errors .36223/.36127 fail both fixed1% bars. Every other hop group
fails too. Frozen gates are conditional attribution, not physical removal.
Dossiers: explanations/join_write_response_curve.md and join_matcher_factors.md.

Hourly2058: focus/novelty PASS, ceremony FAIL for main-screen median14.8min
versus seconds of compute. Bounded workflow repair completed: shared
query_hop_fork_reference.py with five controls; executed immutable metadata
correction QUERY_HOP_METADATA_CORRECTION_V1.json. Degree rows need576 answer
and expected-answer fixes; earlier pair rows need576 expected-answer fixes
(their actual answers were correct). No registered scores consumed the stale
fields. Apply the overlays before later answer-based analyses. Completed
runner/results/rows unchanged. New cross-layer rows already use the helper.

Next action is claimed and its exact accounting derived in
ENDPOINT_RENAMING_CAUSAL_BALANCE_V1_PREREGISTRATION.md. Use saved physical
removals and native query logits to partition matched-context output change
into removal-effect change and remaining-computation change. Align by the
known middle/foil transposition, including hop2 answer correspondence; no
fitted rotation. Measure causal-field portability and paired query JS floor
on every opened population/hop group. No GPU or model execution needed.
The derived metrics have not been opened; execute this bounded CPU step next.

Both managed runners healthy; last observed GPU v439 completed20:54:14.
Inspect current queue before GPU work. Next hourly21:58 and math22:49 UTC.
Preserve concurrent Claude work and all closed nulls; commit/push owned files.

## Pair transport rejected; exact join-read degree audit active — 2026-09-09 20:43 UTC

The bilinear reconstruction handoff and appended structural criterion control
the work. Previous turn progressed through the managed pair-transport screen,
a complete CPU pointwise audit, and an implemented response-curve primitive.
The overall structural circuit goal remains active and unfinished.

MATCHER_PAIR_TRANSPORT_V1 completes6.79s, A/C true and B false. Native/export
full outputs close1.42e-13. Hop3 joint gate transfer queryKL9.994/12.019 forward
IID/OOD and3.145/2.697 backward; querychange/removal ratios1.227/1.292 forward,
.921/.788 backward. No single-factor, scalar/entity, role or hop rescue.
MATCHER_PAIR_POINTWISE_V1 retains192 matched source pairs:108 value-to-value
sign flips, full joint-score relative changes1.432–1.610 and cosines -.349–.005.
Near-equal RMS had hidden pointwise differences. Sign-flip cases dominate KL,
but same-sign IID pairs still give .087/.537 meanKL. No fitted sign correction
or causal sign-only conclusion is licensed. Dossier: join_matcher_factors.md.

Active JOIN_WRITE_READ_DEGREE_V1_PREREGISTRATION.md and
join_write_read_degree_reference.py,12 CPU controls pass. A source-only L2 write
leaves the final query fixed, so its final read has a cubic numerator C0..C3
times the live RMS gain cubed. The norm is represented with a stable completed
square; no normalization clamping. Exact curve scales -1,0,.5,1,2 must match
physical source edits and native L2 attention-cell scaling. Removal partitions
into normalization, linear, quadratic and cubic contributions. These are
algebraic attributions, not independently editable circuit variables.

Prior independent port tests motivate fixed forward-linear/backward-quadratic
read hypotheses: retain C0 and the normalizer plus C1 forward or C2 backward.
Require1% query-effect prediction at every nonzero registered scale and every
population/orientation/hop group, floor1e-6. No degree/head/scale-range fit or
selection after failure. All32 opened worlds/six orders/four hops,768 requests.
No trained degree-audit outcomes opened. All native weights/background and
per-query response caches remain charged; exact curve closure alone is not
structural reduction. Derivation: explanations/join_write_response_curve.md.

Next implement managed ops/run_join_write_read_degree_v1.py. Extract native
selected writes with join_contribution_context_reference.contributions; use
the existing matcher pair native score-cell oracle and source-edit executor.
The new job is not yet queued. Both runners healthy; inspect live shared queue.
Next hourly20:51 and mathematical22:49 UTC. Preserve all prior nulls.

## Coupled matcher pair nominated; native transport next — 2026-09-09 20:28 UTC

The controlling direction is the bilinear circuit reconstruction handoff and
its appended structural criterion. The previous turn progressed through one
managed factor screen and two executed CPU diagnostics, then implemented the
next native intervention primitive. Overall structural circuit goal is unfinished.

MATCHER_FACTOR_TRUTH_TABLE_V1 completes .836s: A/D true, B/C false. Native and
saved joint scores replay exactly0 on all768 opened cases. Neither individual
factor passes both mismatches in any full population/orientation group;
factor1 ratios .399–.800, factor2 .087–.333, joint .0265–.0686. No best-cell
or near-threshold individual-factor success is claimed. Coupled product remains.

MATCHER_FACTOR_ROLE_OVERLAP_V1 passes exact replay and24-permutation energy
control. Forward native mismatch ratios .059–.069 versus .116–.124 under the
norm-preserving role-permutation energy baseline show role co-location helps
selectivity; backward results are not uniform. This is not a shuffled native
intervention. MATCHER_FACTOR_INTERCHANGE_SCORE_V1 then tests factors through
their native multiplicative consumer: neither individual swap suppresses both
mismatches; both do. Matched backward single-factor hybrids can reach2.37–2.53
times base score RMS, while joint transfer remains .966–.979. Those aggregate
norms do not establish pointwise or output invariance. Joint replay0, mixed1.78e-15.

Active MATCHER_PAIR_TRANSPORT_V1_PREREGISTRATION.md and
matcher_pair_transport_reference.py,13 CPU controls pass. Matched base/both-label
donor pairs from the opened32 worlds/six orders, fork recipients into all four
query hops,768 recipient requests. Transfer factor1/factor2/joint product only
at the original four pair cells with native H1/H2 orientations; keep recipient
values and all other computation fixed. Reuse the binding-only factors across
query hops. Include native and zero-score route-removal controls. Exact compiled
L2 score-write delta uses the final source-edit executor; independent native
oracle replaces attention cells. The joint pair must preserve full outputs at
meanKL<=.001,p99<=.01 and change query vectors by<=1% of native route-removal
effect in every population/orientation/hop group. Native mixed effects must
replay. Single-factor arms remain diagnostics, not fallback candidates.

Next implement managed ops/run_matcher_pair_transport_v1.py. It is not queued
yet. All387968 native export constants/background remain charged; no reduced
program is adopted. Canonical dossier: explanations/join_matcher_factors.md.
Initializer reduction and complete single-QK copying routes stay closed.
Both managed runners healthy; inspect shared queue before enqueue. Next hourly
20:51 and mathematical22:49 UTC. No new trained pair-transport outcomes opened.

## Initializer route closed; contextual matcher factors active — 2026-09-09 20:15 UTC

Follow the bilinear reconstruction handoff and appended structural criterion.
The durable goal's older better_math_ideas wording is superseded. The previous
turn made progress through a managed causal screen and three CPU consequences.

SUMMARY_LAYOUT_MEDIATION_V1 completes8.48s: A/C true, B false, full-output
oracle2.13e-13, mixed3.55e-13. Summary-only transfer leaves hop3 order-effect
errors1.0001–1.0008. Prefix-only diagnostic errors .00655–.01640; both cyclic
shift groups miss1%, so no alternative arm is adopted. SUMMARY_VARIATION_VS_REMOVAL_V1
shows summary variation is small relative to removal at hop3 but reaches .412
of removal magnitude at other hops. SUMMARY_INVARIANT_INTERCHANGE_BOUND_V1
retains all3072 native summary swaps: every effect exceeds the1e-6 floor, so
an invariant summary encoding's identity/zero prediction fails all cases at
the1% relative-effect bar. This is not a factual constant-bias replacement test.
Stop initializer reduction at exact partial extraction/CSE infrastructure;
do not rescue it by adding a fitted bias, field, head or rank. No structural
model reduction was adopted. Dossier: explanations/summary_record_transport.md.

Return to the already identified contextual L2H1 forward/L2H2 backward join.
JOIN_MATCHER_COMMON_FACTOR_V1 CPU audit checks its complete QK bilinear forms
at fixed query3/source1 positions. Native product oracle1.39e-16; exact nonzero
stored coefficient minors show nonproportionality, best scalar residuals
.999990/.999960. This rejects copying one complete QK function into the other,
not semantic sharing on the actual restricted input/intervention domain.

Active MATCHER_FACTOR_TRUTH_TABLE_V1_PREREGISTRATION.md and
matcher_factor_truth_table_reference.py, six CPU controls pass. Use the opened
SUFFIX_JOIN_MIDDLE_MATCH_V1 cohort:32worlds,all6orders,all4 label-intervention
cases, original pair masks and H1/H2 orientations. Test each QK factor's middle
matching truth table separately at fixed .10 mismatch/matched RMS bars, all
four pair cells retained, joint-score replay as positive control. No new fit,
head/order selection, causal identification or fresh-data claim. All native
weights stay charged. Dossier: explanations/join_matcher_factors.md.

Next implement managed ops/run_matcher_factor_truth_table_v1.py, reusing the
old middle-match generator, native score receipt and current factor primitive.
No such job is queued yet. Positive nomination requires native factor interchange
and fresh confirmation; a null preserves the coupled product. The two CPU
audits and common-factor audit were actually performed after the main screen,
and the next primitive is implemented. Both runners healthy; inspect shared
queue before enqueue. Next hourly20:51; next mathematical22:49. Goal active.

## Record order affects outputs; summary mediation active — 2026-09-09 19:59 UTC

The bilinear circuit reconstruction handoff and appended structural criterion
remain controlling. Previous turn progressed by executing the registered
RECORD_ORDER_OUTPUT_INVARIANCE_V1:2.49s managedGPU, A/C true, B false,
native/export1.42e-13. Across3072 opened pairs, hop3 paired-query KL floors
are .005156/.009056 IID and .011555/.019661 OOD for record swap/cyclic shift.
These reject a function-only predictor without record-order inputs at the
registered .001 per-group paired-query bar. They are not all-token bounds.
No head/task/adapter rescue and no newly held-out-data claim.

RECORD_ORDER_FAILURE_CENSUS_V1 CPU audit retains every pair, exact saved-radius
replay0. Both-correct pairs alone contribute .003292–.006520 to all four hop3
group means using the original denominators, so rare errors do not explain
the entire obstruction.19 argmax disagreements across3072 pairs. Full proof,
table and scope: explanations/summary_record_transport.md.

Active successor SUMMARY_LAYOUT_MEDIATION_V1_PREREGISTRATION.md and
summary_layout_mediation_reference.py,14 CPU controls pass. Cross original/donor
binding prefix with original/donor first-layer document summary, then execute
the live suffix. Local query inputs are identical. The independent native hook
uses the donor's native first-layer final-query state, matching a summary-only
swap because local terms agree. Allfour cells, mixed effects, live paths and
restored hooks validate. Test whether summary-only transfer predicts full
record-order query effect to1% and donorquery KL<=.001,p99<=.01 in every group.
Prefix-only remains diagnostic. All387968 native coefficients stay charged.

Next implement managed ops/run_summary_layout_mediation_v1.py on all3072 opened
pairs using the existing initializer executor/scorer and saved native endpoints.
It is not yet queued. The last output-test enqueue initially failed the required
three-prediction-label gate; existing saved-native replay was exposed as C before
model execution, with no scientific threshold change. Preserve that receipt.
Both runners healthy; inspect current shared queue before enqueue. Clocks remain
hourly20:51 and mathematical22:49. Overall structural circuit goal is unfinished.

## Controlling handoff and current continuation — 2026-09-09 19:53 UTC

Follow `basis_aligned/polynomial_causal/explanations/bilinear_circuit_reconstruction_codex_handoff.md`
and its appended structural success criterion. The older durable goal's reference
to better_math_ideas is superseded by the user's explicit correction. Bytes,
quantization, ordinary recurrence and CSE alone do not satisfy this criterion.
GPU use is authorized where useful; use the managed runner for every GPU job.

QUERY_INITIALIZER_FACTORIZATION_V1 completes4.03s, all four predictions pass on
1536 fresh requests. Full native outputs, independent removals and mixed effects
close <=2.99e-13; document/local cache reuse closes4.44e-15. This is exact partial
extraction with all387968 native export coefficients retained, not structural
model simplification. The literal summary/local source partition remains fixed.

SUMMARY_RECORD_TRANSPORT_BOUND_V1 completes .23s CPU: native-token summary
correspondence3.11e-15. At affine transport norm cap1e6, relative operator error
is at least .11069 for the fixed record swap and .11749 for the cyclic shift.
SUMMARY_PERMUTATION_CLOSURE_V1 completes .059s: exact nonzero dyadic minor modulo
2147483647 certifies full46-dimensional slice rank for the stored FP64 operator,
hence minimal linear permutation closure1058, the full incidence domain.
These concern all-bijections affine summary transport, not nonlinear encodings,
unrounded interval certificates, native output fidelity or full-model impossibility.
Proof and prices: explanations/summary_record_transport.md.

The next action is RECORD_ORDER_OUTPUT_INVARIANCE_V1_PREREGISTRATION.md. On the
opened16 initializer worlds, test native full-query output under the same two
answer-preserving reorderings. Paired JS>.001 in any population/hop/permutation
group rejects a function-only abstraction at that paired mean query-KL bar;
smaller JS means not ruled out, not confirmed. Reuse existing exported/native
execution and entity-symmetry paired-bound machinery with identity label mapping.
No new fitting, task filtering, adapter rescue or fresh-data claim.

record_order_reference.py is implemented. RECORD_ORDER_TOKEN_CONTRACT_V1.json
passes3072 reordered requests, identity/inverse/function/suffix preservation and
malformed-permutation rejection. Managed full-output integration remains unfinished;
do not claim it queued or complete. Unrelated v435 occupies the shared GPU queue.
Both Supervisor runners are healthy. Latest reviews hourly1951 and math1949;
next20:51 and22:49 UTC. Preserve the active overall goal and live shared Git work.

## Instruction already computed; exact initializer factorization active — 2026-09-09 19:29 UTC

LATE_HOP_INSTRUCTION_V1_RESULT.json completes5.16s managedGPU,A true,B/C/D false.
Fulloracle1.81e-13, earlieroutputsexact0. Directembedding-swap queryKL2.728–39.576,
effecterrors.620–1.012; instructionerasure ratios.648–.987. LiteralE(hop)/8 isnot
a sufficient lateinstruction. No computed-only or expandedfield rescue.

Immediate CPU LOCAL_QUERY_INITIALIZER_V1_RESULT.json onopened first4worlds/pop,
768states,6controls/nativeoracle1.56e-13. Physically local firstlayer queryinit
using onlylast3tokens removes192of204 sourceedges butfailsfullnativefidelity.
Hop3queryKL.143/.118,hop2 .0053/.0061; allhopvectorerrors.027–.348 exceed1%.
Tinyhop0/1 KL doesnot implyvectorfidelity orselectivity. Thislocal-onlycandidate
staysclosed; allnativeweights wouldremainretained, noadoption.

Active QUERY_INITIALIZER_FACTORIZATION_V1_PREREGISTRATION.md and
query_initializer_factorization_reference.py,12CPUcontrols pass. Exactfirstlayer
finalquery state = S(bindingdocument,hop)+L(queryentity,hop), fixedpositions/length.
S isindependent ofqueryentity; L useslast3tokens only andisindependent ofdocument.
The sourcepartition definesremovals, no fittedadditivegauge. Bothcomputations
are extracted directly fromoriginaltoken/position projections, no fullquery
attention computedthen discarded. Keepboth,summary-only,local-only,neither have
independentnative mask/residualoracle. Zerojointquery isabsorbing inthisbiasfree
network; zero/uniformoutput isexpectedcontrol.

Next managed ops/run_query_initializer_factorization_v1.py: fresh34909/34910
first8worlds/pop,1536requests. Reuse4 summaries/document across24queryconsumers,
localstates acrossdocuments, validatefullnative outputs andallremoval/joint
effects. Reportperhop probabilityandvectorbehavior withoutinventedselectivity.
All387968 nativeconstants,caches/adapters charged; no novelstructuralreduction
claimed fromsource-additivity/CSE. This ispartialextraction machinery, notgoal
completion. No freshtrainedoutcomesopened. Canonicalhopdossier current.

Hourly1851 next19:51; mathematical1648 next19:48. Bothmanagedrunnershealthy;
inspect livequeues. Bilinearhandoff/appendedcriterion remainscontrolling.

## Query-factor and common-address nulls; late instruction active — 2026-09-09 19:15 UTC

HOP_QUERY_FACTOR_TRANSFER_V1_RESULT.json completes3.45s managedGPU,A/D true,
B/C false on4608fresh ordered transitions. Nativecapability>=.9635,oracle1.56e-13,
explicitmixedcomposition1.63e-13. Q1errors.594–1.659,Q2 .826–1.970; neither
nativefactor supplies independenthopcontrol. Mixedterm/targetRMS.911–2.562.

Immediate CPU HOP_COMMON_ADDRESS_BOUND_V1.json: bestpossible scalar-modulated
shared binding-message pattern pernativehead fails, residual.435–.611; all1536
world/query/head cases exceed1%,savedreadoracle4.55e-13. This is a fixed causal-
response lowerbound, not rankselection or arbitrary-grouping impossibility.
Next saved-output READ_AFTER_ADVANCE_AUDIT_V1.json rejects nativeonehop read at
F^(h-1)(q) as higherhop reader: h2errors.823/.820,h3 1.401/1.404 IID/OOD.
All768 comparisons retained, identityexact0, no newnativeexecution or fitting.

Active LATE_HOP_INSTRUCTION_V1_PREREGISTRATION.md and
hop_instruction_field_reference.py,18 CPUcontrols pass. Fresh33909/33910 first8
worlds/pop,all24queries/all12ordered hopchanges. Test literal E(hop)/8 atfinalquery
as late instruction: swap onlyembeddingdifference, recompute finalRMS/QKV/residual,
predict fullnewhop distributions/effects. Removingliteralinstruction musterase
hopdependence. Computed-only isdiagnostic; fullstatejoint isdonorpositivecontrol,
not an alternatecandidate. No scale/producer/head/rank rescue. All387968 weights
staycharged. Next integrate managed ops/run_late_hop_instruction_v1.py using
exact_source_edit_reference.py andsharedmetrics; no trainedfreshoutcomesopened.
Canonical circuit record: explanations/hop_instruction_circuits.md.

Hourly1851 next19:51; math1648 next19:48. Both runnershealthy, sharedlaneactive;
inspect livequeues beforeenqueue. Bilinear handoff/appendedcriterion controls.

## Fresh content confirmation fails; query-factor study active — 2026-09-09 19:00 UTC

FRESH_ERROR_CONTENT_V1_RESULT.json completes2.60s managedGPU,A/B/D true,C false.
Every3072 fresh native state retained: IID9errors,8supported,5repaired;
OOD11errors,10supported,5repaired, below the registered50% OODbar. Field-level
raw/join mediation passes, correct peers18/18 remaincorrect, mean absolute
goldP change.00917 and worstloss.00364. Numericaloracle1.56e-13,composition9.24e-14.
Do not relabel the overall confirmation as a pass or enlarge the failed rule.

WRONG_SOURCE_ROOT_CONTENT_AUDIT_V1.json follows immediately onall20 openederrors,
all51 inputroots: source/nativehookclosure1.17e-13. Strongest rootmatcheswrong
identity19/20. OODworld5 query3 renamed has strongestroot correctvalue7 despite
predicting19. SIGNED_RAW_READER_AUDIT_V1.json resolves this as suppression of7:
positive-P H0 contributes-4.125 togold versus+.163 towrong. Negative attention
alone is not the explanation; semantic sign combines gate and decoder. Static
H1/H2 decoders favor sourceentity24/24, H3 suppresses24/24, H0 suppresses19/24.

SIGNED_COPY_SUFFICIENCY_V1.json tests a fixed96-coefficient shared centeredonehot
decoder derived by orthogonal projection. Every40 full-vector raw-removal case
fails1%relativeRMS, max.990; residualdictionaryFrobenius.328. Preserve this null:
sign/copy descriptions are insufficient for full distributions. All387968 opaque
constants staycharged; no decoder/rank/shared-table rescue is selected.

Active HOP_QUERY_FACTOR_TRANSFER_V1_PREREGISTRATION.md and
query_factor_transfer_reference.py,13 CPUcontrols pass. Fresh31909/31910 first8
worlds/pop,all24entities,all12ordered hop transitions,4608transitions. Only final
hop token changes, so initialbinding states/K1/K2/V remain identical bycausality.
Interchange finalquery Q1,Q2,both: target is entire29-vector initialbinding read,
not the querysuffix orresidual/fullmodel. Both-Q transfer must equal donorread;
test Q1-only andQ2-only separately at1%centered effecterror in everypop/hoppair.
Exact bilinear interaction isreported, not assumedzero. No trained outcomeopened.
Next integrate managed ops/run_hop_query_factor_transfer_v1.py using existing
export/native helper/sharedmetrics; all pairedcontext/gates/weights charged.
This changes queryinstruction factorization, not a fit or repair of valuefields.

Hourly1851 completed with circuit/throughput/novelty audit; next19:51.
Math1648 next19:48. Both runners healthy; inspect currentqueues beforeenqueue.
Bilinear handoff/appended structural criterion remains controlling; goalactive.

## Wrong-source content paths pass an opened-case screen — 2026-09-09 18:43 UTC

Follow the bilinear circuit reconstruction handoff and its appended structural
success criterion; stale durable-goal wording mentioning better_math_ideas is
superseded by the user. Full predictive/extraction/removal/composition objective
and reduced structural description remain unachieved.

FINAL_SOURCE_FAILURE_ATLAS_V1.json: all20 answer-disagreement pairs,40 states,
960 binding cuts. Of21 wrong states,16 can be repaired by an off-chain read cut;
20 strongest wrong-support sources are off-chain. Every strongest source carries
the wrong answer as its raw value (15) or an earlier forward-join endpoint (6).
Export/native correspondence5.68e-14. Initial native-baseline delta check is
preserved separately; canonical candidate uses independently exported baseline.

BAD_SOURCE_CONTENT_MEDIATION_V1_RESULT.json: all42 predeclared primary/paired
states, six arms, all four gates pass in0.90s CPU. Removing nominated E/8 or L2H1
joined-endpoint content through final V, under native fixed P/RMS/Q/K/residual,
repairs12/15 raw and5/6 joined errors. Each field explains about92.5% of the mean
whole-source wrong-minus-gold margin; wrongP loss is.689/.688. All19 correct peers
stay correct. Native all-output oracle5.68e-14; additive path composition3.55e-14.
This is a defined content-path intervention, not physical state deletion with
recomputed RMS, and does not revive rejected global field interchanges.

Immediate successor BAD_SOURCE_CONTENT_SPECIFICITY_AUDIT_V1.json rules out a
large cancellation hidden by the pooled mean on these19 peers: mean absolute
goldP change.00471, max absolute.02182, worst loss.01147. Opposite joined path is
absent for4/15 raw primaries; among11 present controls only1 repairs. Opposite raw
path repairs1/6 joined primaries. Four remaining primary failures change to other
wrong answers, all retained. This successor CPU analysis actually completed.

Next research decision: prospectively validate the fixed raw/join error-content
rule on a fresh complete cohort, counting uncovered errors and unsupported source
types rather than filtering them out. Freeze source nomination and all gates
before opening outputs. Discovery selection used known errors/gold labels, so
current17/21 is not a deployable error detector or fresh identification. Use
shared intervention/scoring helpers; keep independent native correspondence and
all387968 coefficients/background charged. No structural saving is established.

Both managed runners healthy; last own GPU job completed. These small audits
were CPU-only. Hourly1751 next18:51; math1648 next19:48; inspect live state.

## Native entity-symmetry obstruction and failure corpus — 2026-09-09 18:24 UTC

GLOBAL_ENTITY_GATE_REUSE_V1_RESULT.json completes2.70s,A true,B/C/D false.
Nativeoracle1.99e-13/factorial2.34e-13, native capability licensed. Payloadonly
allmeanKL8.835/10.389,queryKL1.084/.848,queryeffecterrors.410–.820. Hop3 payload
accuracy.760/.776 andgatesonly.766/.734 fail. Global gate reuse CLOSED; no selected
layer/head/normalizer repair.387968 coefficients,334560byte gatecache/context,
439008 totalcache/context, pairedcontexts2, no weights removed.

Immediate CPU ENTITY_EQUIVARIANCE_LOWER_BOUND_V1.json: six controls and correct
vocabulary alignment pass. Any exactly entity-equivariant surrogate has paired
mean queryKL>=.014390 IID/.006391 OOD; hop3 floors.050015/.025294. This exceeds
.001 querybar, but query-only all-token floor.000282/.000125 does not rule out
all-token meanbar alone. Proof/primary citation in explanations/entity_symmetry_fidelity_bound.md.
Small-bound label corrected to not_ruled_out_by_pairwise_bound; initial misleading
can_meet-labelled bytes retained in _INITIAL_LABELS.json, numerical values unchanged.
Not a bound against identity-dependent smaller models; full goal remains active.

Immediate CPU ENTITY_SYMMETRY_COUNTEREXAMPLES_V1.json preserves all1536pairs.
20 answer-disagreements total: IID16 correctnesschanges+1 bothwrong/different,
OOD3 correctnesschanges. Both-correct pairs still contribute.003881/.004801 to
populationmeanJS, so this is not solely rare task errors. Complete native paired
query logits/tokens remain in GLOBAL_ENTITY_GATE_REUSE_V1_ROWS.pt.
Next highest-information action: causal final-reader source-contribution audit
on all20 disagreement pairs in both contexts, with corresponding binding positions
matched by recordordinal, no cross-context hidden-state patching. Determine whether
wrong outputs depend on an irrelevant binding or wrong content at the expected
carrier. This is opened-case discovery, not fresh promotion or a restored symmetry
claim. Claim before implementing; use exported token-derived state, independent
native single-source removals, shared metrics, and retain everyselected case.
Do not discard confidence-only symmetry deviations or reduce fidelity to accuracy.

Hourly1751 next18:51; math1648 next19:48. Both managed runners healthy; inspect
currentqueues. Bilinear handoff/appended criterion overrides stale goal wording.

## Active whole-model entity/gate reuse — 2026-09-09 18:15 UTC

SOURCE_PORT_FIELD_INTERCHANGE_V1_RESULT.json completes12.58s,A/C true,B/D false.
Native port correspondence1.71e-13, full-port finalquery/physicalstate1.42e-13.
V-only raw retarget.615–.750, removal.569–.790; rawV collateraljoinedconsumer
loss.168–.203 and jointjoinedaccuracy.742–.755 fail. Key-only cells barely retarget;
addingkeys does not consistently help. Fixed V-only candidate and physical raw/
join field candidates stayclosed; no alternate subset or expanded-field adoption.
Full endpoint dossier explains evidence and limitations.

CPU FROZEN_PAYLOAD_LINEAGE_V1_AUDIT.json completes.73s on eight openedworlds,
onequery/world. Complete key/localvalue/transferredvalue/query-root partition
replays nativeprefix2.84e-14. Diagonal local lineage vs individual-root propagation
exactly0; local versus E/8 cosine.629–.654. This diagnoses omitted local payload
processing, not an adopted larger rawfield. Gatecache250920bytes/input (prefix3),
387968 constants retained, zero removed. frozen_payload_lineage_reference.py
propagates arbitrary payloads under native P/RMS gates; not true token Jacobian.

Active GLOBAL_ENTITY_GATE_REUSE_V1_PREREGISTRATION.md and
 gated_payload_model_reference.py,9 CPU controls pass. Full four-attention model
is linear in embeddingpayload E conditional on allnative gates G: logits=T(G)E.
Fresh27909/27910 randomworlds,first8/pop,all96queries/world;16worlds1536pairs.
Rename allentitytokens byfixed24-cycle sigma: keys,values,queryentity; graph F
becomes sigma F sigma^-1 withsameorder/topology, answers renamebysigma.
Fourarms native=T(Gold)Eold, payloadonly=T(Gold)Enew, gatesonly=T(Gnew)Eold,
renamednative=T(Gnew)Enew. Test whole-distribution KL and signed queryeffect,
plus semantic separation; no fitting or layer/head selection afterfailure.
Paired-context gate interchange is explicitly priced, not recipient-only gate
extraction. All387968 coefficients and allfourlayer gatecache retained.
Next integrate managed ops/run_global_entity_gate_reuse_v1.py with shared metrics,
independent native checkpoint oracle, graph/label checks and fixed bars inprereg.
No trained renamed outcome opened. Do not conflate linearity in E with linearity
in each gate: T(G) contains composed multiplications acrosslayers.

Hourly1751 next18:51; math1648 next19:48. Both managed runners healthy; inspect
livequeue before enqueue. Durable goal follows bilinear handoff/appended criterion.

## Active source-port field test — 2026-09-09 18:03 UTC

DUAL_VALUE_FIELD_INTERCHANGE_V1_RESULT.json completes15.05s,A/C true,B/D false.
Native correspondence1.99e-13, prefixidentity3.55e-15. Raw E/8 removal isstrong
(.611–.768 goldP loss), but raw retarget.620–.711 and collateraljoinedconsumer
loss.210–.240 fail; jointforwardhop3 targetaccuracy.725–.728. Joined-only gate
passes independently. Physical overlapping raw/join field abstraction CLOSED
as specified; no scale/producer enlargement. Shared field_intervention_metrics.py
is used by the completed runner, implementing the hourly scorer correction.

CPU DUAL_VALUE_NORMALIZATION_AUDIT_V1.json completed24.89s on first4openedworlds/
pop,768queries; saved live replay3.27e-13. Fixed final source gains reduce
interactionRMS2.64/2.52→1.84/1.77 but do not restore raw/joint semantics. Fixedgain
is a diagnostic different intervention, not an adopted native repair.
Canonical dossier explanations/forward_endpoint_field_circuit.md updated.

Active SOURCE_PORT_FIELD_INTERCHANGE_V1_PREREGISTRATION.md and
source_port_interchange_reference.py.19 CPU checks pass all8 native K1/K2/V
projection-hook comparisons, live subsets, emptyidentity, hooksrestored,
full-port finalquery=physicalsource edit and binding-output inequality control.
Fresh26909/26910 dual-field worlds, first8worlds/pop, all96queries/world retained:
16independentworlds1536variants. All8 source-port cells for raw/join/joint mapping,
plus V-only raw/join/both removals and native (28namedarms). V-only is the fixed
semantic candidate; no best-subset/head/dose selection. Queries and residualskip
staynative everywhere, even atchangedbindings. All-output native port oracle
required; physical source-edit correspondence applies only to unchangedfinalquery.
Next integrate managed ops/run_source_port_field_interchange_v1.py using shared
field_intervention_metrics.py and exported387968-constant program. No trained
port-factorial outcome opened. Generator now accepts optional fresh seeds with
prior defaults unchanged. All native background and derived caches stay charged.

Hourly1751 next18:51; math1648 next19:48. Both managed runners healthy; recheck
livequeues/processes. Goal remains the full bilinear handoff/appended criterion.

## Active dual-value-field experiment — 2026-09-09 17:50 UTC

FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_RESULT.json completes12.50s: numerical controls
pass1.42e-13, path1.11e-15, identity0, rawquery composition<=2.14e-13. Broad
hop2+hop3 family FAILS: hop2 endpoint-only removal goldP loss.0086–.0119 vs.25,
retarget accuracy.055–.062. Receipt instrument_invalid means the failed causal
route requirement inside A, not code/numerical failure. Separate immutable
FORWARD_ENDPOINT_RANDOM_LAYOUT_GATE_AUDIT_V1.json classifies existing bars;
original outcome unchanged. Hop3 raw bars pass (retarget.863–.881, removal.883–.888),
all controls pass, but do not drop hop2 and relabel the combined outcome a success.

Immediate CPU FINAL_ENDPOINT_READER_AUDIT_V1.json completes19.24s on first4 opened
worlds/pop,768queries, replay2.77e-13. Final H0/H3 carry most hop3response; each
alone retargets.33–.45, no singleton adopted. Hop2 centered effectRMS3.99–4.74 is
large despite weak answer damage, so it is not numerically unread. Full circuit
dossier explanations/forward_endpoint_field_circuit.md explains evidence limits.

Active DUAL_VALUE_FIELD_INTERCHANGE_V1_PREREGISTRATION.md and
 dual_value_field_reference.py (7 CPU controls pass). Fresh25909/25910 random
worlds,32worlds3072queries. Compare original binding-value residual E/8 with
computed forward endpoint messages at overlapping postL2 record positions.
Distinct maps sigmaRaw and sigmaJoin=sigmaRaw², no sameentity outputs. Raw consumer
hypothesis: hop1/2 and backwardhop3; joined consumer: forwardhop3; hop0control.
Arms native,identity,mapRaw,mapJoin,mapBoth,removeRaw,removeJoin,removeBoth.
Mechanical A explicitly separate from semantic B/C/D; exact RMS interactions
retained, no raw-additivity bar. Do not silently enlarge or rescale failed raw E/8.
Next integrate managed ops/run_dual_value_field_interchange_v1.py with exported
program and independent native endpoint-message oracle plus captured prefix
producer identity. No trained dual-field outcome opened. Default random-layout
seeds remain unchanged; generator now accepts optional seeds for this fresh test.
All387968 native constants and derived caches/background remain charged.

Hourly1751 review complete; next18:51. Math1648 remains due19:48.
The hourly ceremony/latency gate failed: before opening unrelated science, integrate
the dual-field runner using shared field_intervention_metrics.py (5 CPU controls
pass) for signed panel metrics and mechanical-only correspondence. This bounded
scorer repair is implemented; avoid another duplicated perexperiment scorer. Both runners healthy;
inspect live queue. Goal stays active under bilinear handoff and appended criterion.

## Active arbitrary-layout endpoint family — 2026-09-09 17:39 UTC

JOIN_ENDPOINT_FIELD_SWAP_V1_RESULT.json passesA/B/C/D in3.49s: all single/joint
hop3 intended answers correct across32IID/OOD worlds, targetP gains.956–.998.
Independent full-logit correspondence1.21e-13; otherquery collateral<=.00697,
lowerhop<=.01845; finalquery raw composition<=9.60e-14. This identifies endpoint
interchange on fixed layouts, not the full structural-simplicity goal.
Canonical circuit dossier explanations/forward_endpoint_field_circuit.md.
Complete-route removal is causal; endpoint-only removal remains to be tested.

Immediate CPU FORWARD_ENDPOINT_MESSAGE_COMPILER_V1_AUDIT.json completed1.60s:
162 exhaustive tiny function/orders plus7 controls; exact native path5.55e-16
and all token-parsed join messages vs native hooks7.77e-16 on32 opened worlds.
forward_endpoint_program_reference.py defines joins/dictionary/messages. Shared
phi(e)=O_L2H1 V_L2H1 Emb(e)/8, contextual scalar=P_L2H1(t,s)*sourceRMSgain.
Derived dictionary3072coefficients/24576bytes, zero independent additions and
zero native coefficients removed;387968 constants remain. All native prefix,
routing/gains, complement/readers and arbitrary phi content stay charged.

Active FORWARD_ENDPOINT_RANDOM_LAYOUT_V1_PREREGISTRATION.md and
forward_endpoint_random_layout_reference.py: fresh24909/24910 IID24-cycle and
OODthree8-cycle worlds, uniformly random record orders, all24queries×hops0..3.
32worlds/3072variants; parser selects all forwardjoins, partitions bytargetrecord
parity, fixed random24-cycle endpoint map. Arms native,identity,mapEven,mapOdd,
mapAll,removeAll(endpoint terms only). Structural eligibility/desiredanswers
fixed fromlast-two-edge orientation; no native-error/layout filtering.
Generator's10CPU checks pass; everyeligible hop/parity/population has88–104 rows.
Next integrate managed ops/run_forward_endpoint_random_layout_v1.py using exported
program plus independent native-message oracle. No trained outcome opened.
All quantitative gates, controls and price in prereg; no routing refit orhead sweep.

Both runners healthy, latest GPU endpoint job completed. Clocks hourly1651 due17:51,
math1648 due19:48. Durable goal follows bilinear reconstruction handoff plus
appended criterion, overriding stale better_math_ideas wording. Goal remains active.

## Active endpoint-field interchange — 2026-09-09 17:30 UTC

JOIN_ORIGIN_FIELD_SWAP_V1_RESULT.json is a valid null: A/D true,B/C false.
Native accuracy.9375 eachgroup; joint intended-answer accuracy.50–.6875, below.80.
Independent full-output oracle max1.42e-13, path2.11e-15, identity0; lowerhop
mean goldP changes<=2.95e-5 and rawquery composition<=1.28e-13. Exact execution
is established; reliable semantic origin interchange and structural reduction are not.

Immediate saved-output CPU JOIN_ORIGIN_FIELD_SWAP_FAILURE_AUDIT_V1.json performed:
38/64 joint hop3 cases retarget; failures10original/5otherchain/11unrelated.
Joint/single correctness differs on only2/64, one rescue and one loss. No filtering,
model access or fitting. Keep the fixed backward-origin interchange closed; no
head/donor/gain tuning. Full report cold_query_composition_reconstruction.md.

Active JOIN_ENDPOINT_FIELD_SWAP_V1_PREREGISTRATION.md and
join_endpoint_field_swap_reference.py,7 CPU controls pass including independent
native L2 pattern/value-input hooks. Fresh23909/23910 both-forward worlds: swap
complete B bindingpairs13/21 in arrangement0; native masks unchanged, headsH1.
Exchange only E(source-value)/4 through recipient gain,V,H1 score,O; keep source
key and contextual producers intact. Earlier forward E task recovery.969–.999
motivates this distinct semantic test; value-position-only interchange is untested.
Singles should retarget ownquery and preserve otherquery; bothSwap retargetsboth.
Arms native,identity,swapA,swapB,bothSwap,cutA,cutB. Adapt the prior managed field
swap executor, correcting the single-arm scoring to this endpoint semantics.
No trained outcome opened. Use exported387968-constant program, independent
checkpoint oracle, managed GPU only; all native weights/background remain charged.

Current clocks hourly1651(next17:51), math1648(next19:48). Both runners healthy;
check live queue before enqueue. Durable goal follows the bilinear reconstruction
handoff and appended success criterion, overriding stale better_math_ideas text.

## Active semantic origin-field swap — 2026-09-09 17:19 UTC

EXACT_SOURCE_EDIT_V1_RESULT.json passesA/B/C/D: all17 fresh IID/OOD edit arms
maxerror1.14e-13, CPU export replay136example-arms2.06e-13 with original checkpoint
reads blocked. Package EXACT_SOURCE_EDIT_V1_PROGRAM.pt, usage EXACT_SOURCE_EDIT_V1_README.md.
Program387968 constants (onlygeneric12672 fold saving), cache1510008bytes.
Sparse editor~.98ms is slower than nativecachedsuffix~.51ms; no speed orsemantic
parameter reduction claim. Fullnative1.84ms, prep1.88ms. Allweights stillcharged.

Active JOIN_ORIGIN_FIELD_SWAP_V1_PREREGISTRATION.md and
join_origin_field_swap_reference.py. Seven CPU controls pass direct receiver-path
contraction vs independent native L1 pattern/value hooks. Fresh22909/22910 data
use arrangement0 transformed to two backward joins (bindingpairs11/19 swapped);
all functions/answers preserved. Swap copied origin value reads while keeping
receiver L1/L2 scores,gains and exact complement. Joint exchange should swap
hop3 answers; singles with the other join removed test reuse, lowerhops specificity.
No fit/head/source/donor selection. Exported program supplies candidateweights;
original model is only an independent intervention oracle. New managed
ops/run_join_origin_field_swap_v1.py remains to be integrated. Do not confuse
path edits with whole-token swaps or whole-upstream-layer ablation.

Current clocks hourly1651(next17:51), math1648(next19:48). Goal remains active
under bilinear reconstruction handoff; all prior quantitative simplification
failures remain in cold_query_composition_reconstruction.md.

## Active exact source-edit executor — 2026-09-09 17:01 UTC

SOURCE_SUPPORT_INTERACTION_V1_RESULT.json passes all4 gates. Fresh IID/three8cycle
withheld triple removal maxerror1.85e-13/1.56e-13; forbidden pair support<=2.14e-13;
overlap thirdRMS.00583/.01235 and logprob pairmax7.44/6.45. Exact grouping rule for
one final attention suffix, not learned parameter reduction. Full report and
1648 mathematical proof state scope restrictions (no extra suffix norm/softmax).

Active EXACT_SOURCE_EDIT_V1_PREREGISTRATION.md and exact_source_edit_reference.py:
SourceEditProgram.prepare(tokens) computes native-prefix/QKV cache; edit(context,
delta) recomputes RMS/RoPE/projections only at affected source/query positions and
all affected source sums. Supports overlapping edits exactly. Physically removes
final O by known generic readout fold; expected387968 constants.15 CPU controls
pass, nonzero editreplay<=1.11e-16. No native or learned opaque coefficient hidden.

Next integrate ops/run_exact_source_edit_v1.py on fresh21909/21910 three-chain
worlds (extend generator's seeds argument without changing default behavior),
all8 disjoint plus8 overlapping producer edits andhalf-dose. Candidate must form
its own writes from retained prefix weights. Compare native/full andcachedsuffix,
export state/config and fresh-process CPU replay without original checkpoint,
then price preparation/repeated edits/cache. No trained outcome opened yet.

Latest hourly1651 (next17:51) andmath1648 (next19:48), both completed. Goal active
under the bilinear reconstruction handoff, overriding stale better_math_ideas
text. Origin path and payload-field task gates pass but quantitative sufficiency
remains rejected; do not relax the original full-distribution/intervention bars.

## Active exact value-producer split — 2026-09-09 16:35 UTC

JOIN_CONTRIBUTION_CONTEXT_V1_RESULT.json passesA/B and failsC/D. Two disjoint
native joins are selectively causal (own goldP loss.423–.987, controls pass),
joint own-write replay4.97e-14 and reuse across8 queries exactly0. Shuffling other
facts while keeping selected facts/positions/fullmap fixed breaks quantitative
cross-context transfer: joint query-effect errors.167–.263. All native weights
and donor prefix retained, not a smaller extracted program.

Immediate CPU JOIN_CONTEXT_FACTOR_AUDIT_V1.json performed. Native-score/value
factor interchange on opened cases locates most transfer variation in payloads;
donor scores still have3–11% query-effect errors, not1%. No fit or promotion.

Active successor: JOIN_VALUE_PRODUCERS_V1_PREREGISTRATION.md and
join_value_producer_reference.py. Five CPU controls pass exact x2=E/4+Y0/4+Y1/2,
three V terms under shared live RMS/routing, and joint reconstruction. Test fixed
forward E/current-value vs backward Y0+Y1/origin-key hypothesis; all8 subsets
for causal interpretation, no best-subset search. Fresh18909/18910 generator and
ops/run_join_value_producers_v1.py remain to be integrated. Full native background
and coefficients charged. Inspect live files/queue before continuing.

Latest strategic1550 (due16:50), math1348 (due16:48). Active durable goal follows
bilinear reconstruction handoff, overriding stale better_math_ideas goal text.

## Shared kernel rejected; causal contribution separation — 2026-09-09 16:23 UTC

SHARED_JOIN_KERNEL_V2_COVERAGE_RESULT.json is valid: A/D true,B/C false.590/590 cells,
exact native/folded replay1.14e-13; physical361776 constants. Full all-token meanKL
36.27/82.94/80.37. No smaller extracted circuit established. Both-heads-removed
candidate/native replay closes, so retained executor is correct. Fixed local-record
kernel CLOSED: no role/head/lag/feature/rank/iteration or regularization rescue.

Immediate CPU successor already performed: SHARED_JOIN_KERNEL_FAILURE_AUDIT_V1.json.
Reusing opened cases, exact-native-score match support still fails (queryKL.130/
.475/5.814). Sparse calibration entity combinations also explain huge learned
coefficients/outliers; ordinary RMS gains rule out tiny-norm explanation. No new
fit chosen. Read cold_query_composition_reconstruction.md for full diagnosis and
native causal findings that survive. Next object is the causal join contribution
with explicit contextual inputs and consumers, not equating entire heads with a
local token equality rule. This diagnostic is not new held-out identification.

Bqrunner healthy; inspect live queue/processes. Latest reviews1550/1348, due16:50/
16:48. Durable goal active under bilinear reconstruction handoff and appended
success criterion, overriding the stale goal text's better_math_ideas reference.

## Active shared-kernel coverage repair — 2026-09-09 16:18 UTC

SHARED_JOIN_KERNEL_V1_RESULT.json is instrument-invalid:555 calibrated cells leave
14/16/9 unseen on held panels. No held model outputs were evaluated; exact error0
is an empty maximum. The physical executor has361776 constants and22 CPU controls
pass, but fidelity/removal are untested. Preserve this invalid receipt.

Active V2_COVERAGE makes one concrete instrument repair: original256 calibration
inputs plus96 complete entity×hop query forks of one new world. Synthetic support
audit passes590/590 possible cells. Same grammar/6576 coefficients/8ALS passes,
fresh16910/11/12 held seeds, unchanged full-distribution/removal thresholds.
Implementation and CPU controls complete; inspect managed queue/result next.
Valid failure closes this fixed kernel, no posthoc feature/head/lag/rank sweep.
Latest reviews1550/1348, next16:50/16:48. Goal remains active under the bilinear
reconstruction handoff, not better_math_ideas.md. Current dossier remains
explanations/cold_query_composition_reconstruction.md.

## Active shared-kernel compilation — 2026-09-09 16:00 UTC

SUFFIX_JOIN_MIDDLE_MATCH_V1_RESULT.json passes all4 gates: valid permutation
single-column label mismatch reduces joint score to2.7–6.9% and full-query causal
effect to0.9–4.3%; both-column renaming restores a live matched route. The frozen
L2H1/H2 therefore have evidence for middle-entity matching, directional write use,
and a reusable final key/value interface. This is still not a compact replacement.

Active next: SHARED_JOIN_KERNEL_V1_PREREGISTRATION.md and
shared_join_kernel_reference.py. The causal record parser and8-pass weighted ALS
core are implemented; native weight-removal executor and managed
ops/run_shared_join_kernel_v1.py remain to be integrated. One shared field-equality
rule with swapped adapters, live RMS gains, and role/lag/entity coefficients would
remove32768 Q/K weights and add6576 constants. No fitting outcome opened; fixed
calibration15909 and fresh15910/11/12 panels. Same initial binding pairs are excluded
prospectively; unsupported matched cells must fail closed. All remaining native
weights/values/readout are charged. Fresh full-distribution and single/joint-removal
success is required, not task accuracy alone. Do not add a feature/rank/head sweep.

Latest strategic review1550 (next16:50); mathematical1348 (next16:48). Goal remains
active under the reconstruction handoff. Read the current cold-query report and
live queue/result before acting. Middle-match receipts and row tensors are preserved.

## Active middle-entity matching test — 2026-09-09 15:50 UTC

Latest strategic review HOURLY_STRATEGIC_REVIEW_2026-09-09_1550.md; next after16:50.
Mathematical review remains1348, next after16:48 UTC. Full goal remains active.
SUFFIX_JOIN_HEAD_PORT_V1_RESULT.json passes all4 predicates: L2H1 forward-value,
L2H2 primary backward-key, with H3's smaller contribution retained. All final source
ports together give exactly0 query-logit replay error, reused fromhop0. This is a
native intervention interface, not a smaller standalone semantic join implementation.
Full per-example logits are in SUFFIX_JOIN_HEAD_PORT_V1_QUERY_LOGITS.pt.

Active managed successor: SUFFIX_JOIN_MIDDLE_MATCH_V1_PREREGISTRATION.md,
suffix_join_middle_match_reference.py and ops/run_suffix_join_middle_match_v1.py.
Four valid permutation column-swap cases break/restore the shared middle equality;
frozen L2H1/H2 edges must gate both joint routing score and full-query removal effect.
Fourteen structural plus nine reused head/port controls pass. Inspect live queue/
result before acting. If it passes, move toward joint read-route-write replacement,
not another source-mask census. All native coefficients remain charged; strict
native-distribution failures in earlier tests remain preserved in the report.

## Active join-head/port continuation — 2026-09-09 15:38 UTC

SUFFIX_JOIN_WRITER_V1_RESULT.json completed7.84s: A/B/D true,C false. L2 (zero-based)
and all-prefix joint cuts pass the selective writer gate in both orientations and
both populations. L2 S→J loss .891–.971; lowerhop and matched nonjoining controls
small. Source ports fromhop0 reuse exactly; all-prefix-cut rescue recovers~100%
behavior but three groups fail strict query-distribution gates (p99 up to .0504).

Active next: SUFFIX_JOIN_HEAD_PORT_V1_PREREGISTRATION.md and
suffix_join_head_port_reference.py (nine CPU controls pass). Managed
ops/run_suffix_join_head_port_v1.py still needs integration. Test individual L2
writing heads and all8 final K1/K2/V restore subsets after L2-only cuts, on fresh
seeds13909/13910. Direct final-query all-port closure should be exact; do not claim
restored logits at the changed J positions. Test forward-value/backward-key port
semantics rather than assuming names identify fields. All weights remain charged;
no compact native join implementation yet. Latest report is
explanations/cold_query_composition_reconstruction.md. Reviews due1549/1648, goalactive.

## Active join-writer test — 2026-09-09 15:33 UTC

CAUSAL_SUFFIX_JOIN_V1_RESULT.json completed: A/B/D true,C false. All12 order/topology
groups support J=the later B2/B3 binding: removeJ loss .856–.978, other-chain |loss|
<=.019. keepJ+Q answers all384 positive variants correctly but fails native full-
distribution fidelity (KL .0043–.526). All32 fixed-point controls fail as predicted
(mean goldP.00082). This nominates a causal source; it does not identify join writers
or provide a compact full-native replacement. Report: cold_query_composition_reconstruction.md.

Active successor: SUFFIX_JOIN_WRITER_V1_PREREGISTRATION.md,
suffix_join_writer_reference.py and managed ops/run_suffix_join_writer_v1.py.
Ten synthetic controls pass. Cut earlier→later suffix-pair edges in each prefix
layer/all3, with source/hop controls, then rescue final native K1/K2/V atJ using
ports from the same world's hop0. Fresh32worlds×6orders×4hops. These are causal
reference ports, never claimed as independent extracted runtime. Inspect queue/result
before acting. Review clocks1449/1348 remain due1549/1648; full goal remains active.

## Latest cold-composition continuation — 2026-09-09 15:20 UTC

COLD_COMPOSITION_SOURCE_V1_RESULT.json passes A/D but rejects direct-answer-binding
sufficiency/selectivity. Larger-panel native hop3 .906/.813; earlier-chain removal
loss .381/.309. COLD_COMPOSITION_ORDER_V1_RESULT.json is an opened-panel CPU diagnostic:
earlier-source loss is nearly zero when the target fact lies in its causal future,
but large when visible. Six order groups nominate reading the later B2/B3 binding,
not simply the latest of all3. Preserve this as nomination, not confirmed algorithm.

Active managed test: CAUSAL_SUFFIX_JOIN_V1_PREREGISTRATION.md,
causal_suffix_join_reference.py and ops/run_causal_suffix_join_v1.py. Fresh64 worlds
counterbalanced through6 binding orders, plus32 fixed-point controls; tests exact
source extraction, selective later-suffix removal and strict full-output sufficiency.
All opaque background charged; fixed-point native failures explicitly tested.
Implementation/shared preflight ready. Inspect live queue/result before action.
Review clocks1449/1348 remain due1549/1648; full goal active under the handoff.
While the suffix-source job waits behind live v389, causal_edge_join_reference.py
implements the candidate operation J_i=E_i C_prev+C_prev E_i. The total is exactly
F² minus same-edge squares. All1296 exhaustive three-node fact-subset/order cases
pass, with separate source-write removals; fact deletion must recompute later joins.
CAUSAL_EDGE_JOIN_REFERENCE_CONTROLS.json records zero trained-model calls. This is
an explicit proposed operation, not a trained identification. Await/inspect the
managed source result before choosing its native writer/reader follow-up.


## Active reconstruction target — 2026-09-09 15:09 UTC

Read explanations/cold_query_composition_reconstruction.md under polynomial_causal.
HOP_COMPOSITION_CAPABILITY_V1_RESULT.json verifies the existing attn4-rms-seed0
checkpoint on cold first queries (all64 correct,16 perhop), unique queries(hop3 .962)
and short-cycle OOD(hop3 .920). Same400640 parameters as the weak pilot. No earlier
answers exist in the cold population. Old attention-trace and linear-probe stories
are not causal algorithm certificates. The weak model's MLP term test is a valid
null: A/D true,B/C false; X-only queryKL2.731/2.733. Close that simplification branch.

Active successor: COLD_COMPOSITION_SOURCE_V1_PREREGISTRATION.md and
cold_composition_source_reference.py. Ten CPU parser/fold/removal controls pass.
The managed ops/run_cold_composition_source_v1.py remains to be integrated. Test128
fresh binding worlds, each forked into4 cold queries, with exact final source
partition and answer-binding sufficiency versus earlier-chain sources. Charge all
native background; no model outcome opened. Board claim is recorded. Review clocks
remain1449/1348, due1549/1648 UTC. Full goal stays active under the reconstruction
handoff, not the empty better_math_ideas.md. Completed managed MLP job is terminal;
inspect live shared queue before adding the successor.

## Latest result and continuation — 2026-09-09 15:02 UTC

UPSTREAM_PRODUCER_FACTOR_V1_RESULT.json is a valid null for selective paths (A/D
true,B/C false). The MLP supplies dominant final Q/K/V inputs for both repeated
higher-hop history and fresh hop1 lookup; no one of15 reader paths is selective.
Joint-reader effects are strongly nonadditive. These cuts preserve the native live
RMS gain and are not upstream state removals. Report is contextual_answer_history_circuit.md.
Active successor: HOP_MLP_TOKEN_CONTEXT_V1_PREREGISTRATION.md, hop_mlp_terms_reference.py,
and managed ops/run_hop_mlp_token_context_v1.py. Eleven CPU controls pass. It reuses
known TT/X/CC algebra to test source-defined lookup sufficiency and all8 true term
interventions with downstream normalization recomputed. Inspect queue/result before
action. Review clocks remain1449/1348, due1549/1648. Full goal remains active.

## Latest restart delta — 2026-09-09 14:49 UTC

Strategic review is now HOURLY_STRATEGIC_REVIEW_2026-09-09_1449.md (next15:49 UTC).
Mathematical review remains THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-09_1348.md
(next16:48 UTC). Newer live receipts override the historical active-successor text.
HOP_STATE_COMPILER_V1 is validly rejected: queryKL1.485–2.514, joint errors .927–.961,
despite lower charged constants375424. CPU HOP_STATE_FACTORIAL_V1 uses frozen books
on opened panels and rejects both one-sided Q/K replacements; interaction49–56% of
joint error. Stop semantic codebook guessing. Active continuation is an exact
upstream-producer/final-factor dependency atlas, preserving live RMS gains and all
background and asking which producer actually feeds the answer-history computation.
The source circuit report records all stronger-null results. Full goal remains active.
Next protocol: UPSTREAM_PRODUCER_FACTOR_V1_PREREGISTRATION.md. Its reusable exact
executor upstream_producer_reference.py passes15 CPU controls; managed runner
integration is the immediate outstanding work. No new trained atlas outcome opened.

## New direction and completed bounded pilot — 2026-09-09 04:04 UTC

### Active-goal restart — 2026-09-09 14:07 UTC

The user explicitly activated the durable goal: reverse engineer the model by a simple
decomposition with OOD prediction, extraction, characterized removal, and composition/reuse.
They corrected the reference to the reconstruction handoff below; `better_math_ideas.md`
is an empty file and is not the authority. The goal is active and unbounded. Do not stop
the program because the original bounded pilot or a successor hypothesis finishes.

Latest strategic/mathematical reviews are both `2026-09-09_1348`; next active-work
checkpoints are after14:48 and16:48 UTC respectively. The 04:12–13:40 inactivity gap
was explicitly disclosed; do not describe those hours as completed Codex research.

The first nonlinear follow-up is complete: a fixed37-case entity-equality/type routing
rule shared by four first-layer heads of the existing small hop checkpoint. V1 is
preserved as instrument-invalid because the real-arithmetic lag-only RoPE table missed
its frozen1e-4 logit control. V2 uses exact absolute native cached positions and matches
native patterns/full forward exactly, but the invariant candidate fails full-output KL
and joint removals: query KL6.57–6.85, joint centered effect error1.028–1.033. V2's
separate candidate lag-versus-absolute agreement predicate also fails and is retained.
The immediate CPU norm-gain explanation also fails (zero heads below the0.10 normalized
kernel residual bar). These close the naive invariant router and magnitude-only variant,
not all joint read-route-write decompositions or the full goal.

Report: `explanations/equality_router_reconstruction_report.md`. Primary receipts:
`EQUALITY_ROUTER_V1_RESULT.json`, `EQUALITY_ROUTER_V2_ABSOLUTE_POSITION_RESULT.json`,
and `EQUALITY_ROUTER_GAIN_V1_RESULT.json`. Native-background/opaque constants are
charged; no circuit has been promoted by these tests.

Active successor: `LOCAL_TRANSPORT_V1_PREREGISTRATION.md`, reference
`local_transport_reference.py`, managed runner
`../bilinear_quotient/ops/run_local_transport_v1.py`. It retains the entire coupled
QK1*QK2*V computation and native absolute RoPE, replacing first-layer dense causal
source access with self/previous-token shifts shared by all heads. Fresh frozen
seeds3909/3910/3911, full29-way distributions on IID/metamorphic/topology-OOD text,
and native-corresponding self/previous/joint edge removals. Six synthetic controls
pass. Implementation is complete and managed preflight passes; inspect the current
queue/log/result before enqueuing or interpreting—never duplicate a run.
All400640 parameters remain opaque and charged. Local transport has now landed:
`LOCAL_TRANSPORT_V1_RESULT.json` is a valid null, query KL1.35–1.58 and joint-removal
relative errors0.639–0.665. No radius or threshold sweep is licensed.

The active successor is `CONTEXTUAL_HISTORY_V1_PREREGISTRATION.md` and managed
`ops/run_contextual_history_v1.py` (under bilinear_quotient). It asks whether the
small checkpoint's higher-hop floor comes from repeated-query answer retrieval.
The explicit final readout is partitioned into residual, initial binding, earlier
matching answer and other sources, with matched nonmatching-history controls.
Full upstream native computation is retained and counted; final W_O is physically
folded into the vocabulary readout in `contextual_history_reference.py`. Nine CPU
controls and model-free preflight pass. Inspect its queue/log/receipt before acting.
This contextual source hypothesis has now passed all four registered tests:
`CONTEXTUAL_HISTORY_V1_RESULT.json`. Higher-hop IID repeated queries are100% correct
versus6.28% novel; matching-history removal drops repeats to6.50% versus an almost
inert count/hop-matched control. Short-cycle OOD supports the same dependence;
unique-query OOD remains6.63%. Novel hop1 depends on initial bindings instead.
Native/extracted full29-way logits close to5.12e-13 and joint edge removals6.25e-13.
The contextual routers/prefix remain native and charged. The complete portable
`CONTEXTUAL_HISTORY_V1_PROGRAM.pt` loads without the original checkpoint and passes
fresh8-document CPU replay (`CONTEXTUAL_HISTORY_V1_EXPORT_REPLAY.json`, max2.27e-13).
Constants387968 versus400640 native, with12672 saved by generic readout folding;
do not count that fold as semantic discovery. Report:
`explanations/contextual_answer_history_circuit.md`.

Active successor: `CONTEXTUAL_HISTORY_FIELDS_V1_PREREGISTRATION.md` and managed
`ops/run_contextual_history_fields_v1.py`. Fresh fixed-position counterfactuals
independently change stored entity key, hop key and answer payload, then jointly
retarget the current query. This addresses source-age ambiguity and tests composite
key/value semantics. A separate IID-nominated/OOD-confirmed test asks whether native
QK1/QK2 factors specialize to the fields; failure of that screen must not erase a
passing semantic product-level result. Twenty parser checks plus nine readout controls
pass; implementation/preflight ready. Inspect live queue/result before enqueuing.
Reviews remain1348, due1448/1648 during active work. Full bilin18 goal remains active.
The fields test has now completed validly with A true, B/C/D false: payload/retarget
probabilities miss .8 in several groups; no head splits fields cleanly by factor.
Preserve this stronger null. Active successor is HOP_STATE_COMPILER_V1_PREREGISTRATION.md,
hop_state_reference.py and ops/run_hop_state_compiler_v1.py: replace final heads0/3
q1/k1 using one causal role/last-visible-hop parser and fitted32-vector means. This
removes16384 weights, adds3840 constants, and tests fresh full outputs and joint
removals; implementation and10 CPU controls ready. Inspect live receipts before action.


The user redirected Codex to
`basis_aligned/polynomial_causal/explanations/bilinear_circuit_reconstruction_codex_handoff.md`.
Read its appended success-criterion correction: bytes/quantization/rank are not interpretability
success. The target is a previously unspecified reusable operation with explicit inputs/consumers,
held-out extraction and joint-intervention evidence, and lower structural description cost including
adapters and opaque parameters. The user also authorized GPU use wherever advantageous; the original
two-CPU-hour budget is not a user-imposed hardware restriction.

The requested bounded A/B/C pilot is complete. Its report is
`basis_aligned/polynomial_causal/explanations/bilinear_reconstruction_pilot_report.md`; the immutable
receipt is `basis_aligned/polynomial_causal/BILINEAR_RECONSTRUCTION_PILOT_V1_RESULT.json`.
Execution and controls pass; scientific discovery does not. Tiny FP64 logits close to 3.11e-15;
full existing small-checkpoint logits/edits close to 3.69e-13. The planted shared update has natural
rank 4 but needs rank 8 for independently edited source histories, matching elementary cubic reuse.
The trained contextual update bank is rank 128/128, and all 512 MLP products and 1,024 factors are
exactly distinct under the registered proportionality/product grammar. This is a local sharing null,
not a theorem ruling out compact nonlinear circuits. No new circuit meets the revised criterion.

The bounded pilot recommends stopping this local span/duplicate-factor discovery method and retaining
the reference executor. Do not relaunch it, an SAE campaign or a broad rank sweep automatically.
A possible next hypothesis is a joint nonlinear matching/routing operation shared by multiple
consumers; it is a proposal, not an experiment already started or a discovered mechanism.
The final pilot used two CPU threads and zero GPU execution because the trained small model completed
in 19.12 seconds while the GPU lane was occupied by v297. The owned pending GPU pilot was removed;
there is no duplicate queued reconstruction job. Existing Supervisor-managed runners remain intact.

The older v26 capability has landed as an honest null: reported-source targets and their controls
failed; three other complete structure pairs passed. No v26 causal continuation is eligible.
Newest review clocks: hourly `HOURLY_STRATEGIC_REVIEW_2026-09-09_0340.md` (next after 04:40 UTC during
active research); mathematical `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-09_0226.md` (next after 05:26).
Treat newer files, board, Git and runner state as authoritative over this snapshot.

## Previous authoritative delta — 2026-09-09 03:10 UTC

The pushed research edge is commit `71b0496f0`. Treat Git, immutable results, the append-only
board, and live managed-runner state as newer authority if they have advanced beyond this snapshot.

The strongest current `is`/`was` circuit evidence is now an explicit but still partial computation:

- four complete attention heads, `L8H1`, `L9H1`, `L9H4`, and `L11H3`, pass exact input-factor
  replay on the v23 aligned bank;
- the valid directed source/destination cells are primarily changed-cue-to-matched-suffix and
  matched-suffix-to-matched-suffix writes; `L11H3` is dominated by suffix-to-suffix;
- the exact per-cell contraction is a causal sum of
  `(q1 dot k1 / 128) (q2 dot k2 / 128) u` over the selected source and destination positions;
- at block 11, the rescued correction propagates through layers 12--17 as approximately `.83266`
  direct residual carry plus `.17281` distributed downstream response, with `-.00547` final-decoder
  interaction. V2 closes the state telescope to `7.28e-12` and logits exactly;
- do not call the direct route fully selective: its P-control leak is `.155885` against the frozen
  `.15` bar. Do not call the circuit structure-general yet.

The user's latest red-team is binding: changing profession nouns inside one temporal-prefix,
comma, subject template is lexical variation, not syntactic generalization. V25 therefore tested
eight genuinely different structures at 8--18 tokens. Four complete target/control structure
pairs were natively capable—fronted era, subordinate-clause prefix, reported-source frame, and
postnominal temporal modifier. Four others failed honestly—post-subject time, relative-clause
subject, long coordinated prefix, and embedded-predicate nominal. V25 is a capability null and
opened no causal result.

The active successor is v26, a completely fresh 64-row holdout over the four capable structure
families. It uses 16 new compound professions, no reused row IDs or prompt text, four examples per
direction, and no post-outcome row filtering. The capability runner is:

```text
basis_aligned/bilinear_quotient/ops/run_tense_auxiliary_is_was_structural_holdout_v26_capability_v1.py
SHA-256: 88ac97c2429234fdc5a9a6f35bce2cf87f1fef76fb837438251b72cc1132c6b9
```

At this snapshot it is queued exactly once in managed lane 1 behind the live family-separability
v295 run. Do not enqueue a duplicate or run it directly. Once it lands, score it exactly. Only an
all-construction capability pass licenses a separately preregistered v26 causal transfer of the
four-head/directed-cell program. A capability null must be preserved and should redirect toward
structure-conditional circuit comparison rather than sentence filtering.

The newest periodic clocks are:

- hourly circuit review: `HOURLY_STRATEGIC_REVIEW_2026-09-09_0224.md`; next safe-boundary review
  after 03:24 UTC;
- three-hour mathematical review: `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-09_0226.md`; next
  safe-boundary review after 05:26 UTC.

These clocks are research-loop reminders, not merely prose. During active work, check them at every
safe boundary, write the due review, and immediately begin its selected action. Supervisor keeps
the bqrunners alive independently of a Codex session; Codex must inspect and feed their managed
queues, never replace them with direct GPU commands.

## Previous authoritative delta — 2026-09-08 14:50 UTC

The pushed code edge is commit `0e4de3243`; always inspect newer commits, dirty files, queue,
results, and Supervisor state before acting. The corrected P7 source-conditioned live-module
effect game has now completed valid. Its immutable v2 result is
`circuits/followups/temporal_iswas_identity_p7_live_module_effect_game_ood_v2_result.json`,
SHA-256 `171d60c3aaf2d35b6a400b9ac753c2c0d8936ee6d354ab91e3bcc9da4f813fc2`.
All A--E instrument, exact-accounting, live-route, interaction, and selectivity predictions pass.

The source effect contains a direct/unmasked-background share of `.4499/.4706` on frozen OOD
FIT/HOLDOUT and a mediated P7 share of `.5501/.5294`. Six P7 modules have stable positive exact
Shapley credit: A11, M11, M12, M15, M13, and M10. A11 and M11 lead at about `.175` and `.133`.
A11+M11 is the only stable pair interaction, positive at `.0455/.0461`; its raw second difference
is positive in every one of 32 background contexts on both phases, so the evidence is stable
complementarity rather than redundancy. M16 does not pass the prospective `.03` stable-credit
bar. These are operational mediation nominations, not yet direct edges.

Preserve v1's immutable `invalid_instrument` receipt. V2 repaired only the parent replay reference:
an all-live suffix must replay the selected writer, whereas v1 incorrectly required the fully
native-clamped identity arm. No scientific population, intervention, bar, or candidate was changed.

The evidence-selected next experiment is a sealed-population physical reader-loss/output-rescue
test for A11 and M11. It must first establish native capability on a genuinely new lexical bank;
then source-present normalized-reader replacement by the source-absent tensor must remove the
registered module contribution, restoring the source-present complete module output must rescue
it, and matched controls must remain inert. Include individual and joint A11/M11 arms so the
positive pair excess is tested physically. Do not call exact Shapley credit or checkpoint-weight
alignment a directed edge without this intervention.

The latest periodic clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_1422.md` (next review after 15:22 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_1426.md` (next after 17:26 UTC). Both
Supervisor-managed `bqrunner` services were healthy and the managed queue was empty at this
snapshot. Never create duplicate runners or launch GPU work directly; use `ops/enqueue.sh`.

## Authoritative handoff snapshot — 2026-09-08 13:58 UTC

The pushed science edge immediately before this handoff refresh is commit `4666f2934`; always inspect newer commits, dirty files,
and managed-runner state before acting. The is-was selected writer is `L7H7 + L9H4`. Its complete
downstream singleton atlas showed a distributed response, and the prospectively fixed cumulative
order selected P7=`A11+M11+M12+M15+M13+M16+M10`. P7 recovers `.5276/.5376` on original
FIT/HOLDOUT with cosine `.9937/.9945`, direction `1.0`, and zero temporal collateral.

An independent block-10 residual-input x complete-module-output 2x2 then established two nearly
additive physical streams. Residual identity alone recovers `.5209/.5088`; all module responses
recover `.4896/.5029`; installing both exactly replays the selected writer. A direct final-residual
add/remove experiment verified the residual recurrence to `3.61e-8--3.69e-8` relative state error
and `3.81e-6--5.72e-6` registered-logit error. Preserve its scientific OOD null: identity alone
drops to `.4582/.4803`, so exact transport is not by itself stable circuit identification.

The decisive OOD composition receipt is
`circuits/followups/temporal_iswas_p7_residual_two_stream_ood_composition_v1_result.json`, SHA-256
`55498ec9110fbddc8ae9a86eb4bab63efeb92540322aaa8783cec2d8c7bd55a7`. It passes all predictions:
the original-selected P7 transfers without OOD reselection at `.5685/.5479`; identity contributes
`.4582/.4803`; P7+identity exactly replays the writer at `1.0/1.0`; both streams remain necessary;
finite-vector nonadditivity is only `.0296/.0326`; all directions are `1.0`; temporal collateral
is zero. Treat this as an OOD-composable executable two-stream interface, not as a learned DAS
subspace or a complete semantic decomposition.

The fixed-P7 leave-one-module-out necessity result has now landed valid at SHA-256
`20eff67e079f46beefe0cf6ec14f16af7674c359fb460c8d73357175761154e4`. Its instrument and
selectivity predictions pass, but every individual deletion is exactly null: full and every
`without_A11/M11/M12/M15/M13/M16/M10` arm recover `1.0` with cosine/direction `1.0` on both OOD
phases. The terminal is `nonminimal_sufficient_bundle`, with no stable necessary modules. This is
a sequential redundancy/overwrite result: later selected-writer clamps can absorb an earlier
omission. It is not evidence that the native modules are all inert.

Two outcome-conditional executors remain complete and tested, but the immutable necessity result
made neither eligible. The atomic binder returns an empty eligible set and created no binding
files. Do not force or enqueue them post-outcome. The closed A11 executor is
`ops/run_temporal_iswas_p7_identity_a11_head_endpoint_atlas_ood_v1.py` at reviewed SHA-256
`417aaa017c06116f69141dbaaa683c511b955c5f80f59f7ccda0508cabbbf3e7`. The closed M11 executor is
`ops/run_temporal_iswas_p7_identity_m11_exact_product_factorial_ood_v1.py` at reviewed SHA-256
`6551339eeec03b3121ed23fb274d24adc02a8654cd5dc6555dfe3a5158f98b43`.

The active successor is the exact source-conditioned live-module effect game, preregistered after
the LOO redundancy trigger. Runner
`ops/run_temporal_iswas_identity_p7_live_module_effect_game_ood_v1.py`, reviewed SHA-256
`b53113fe0237d4e5c0267b61f357d1e397c6ccecce670002118c062281649274`, enumerates every one of
128 P7 subsets under source-absent and source-present block-10 identity states. Enabled P7 modules
recompute live; complement modules clamp native outputs; source-absent arms must be exact no-ops.
Exact Mobius/Shapley accounting separates the direct/unmasked background route from live-module
credits and pair redundancy/complementarity without fitting a surrogate. It is queued exactly once
in managed lane 1 behind Claude's verified-live hash-bound `run_unit_family_separability_spec_v260.py`.
Do not reorder, duplicate, or run it directly. A large credit nominates operational dependence,
not a direct edge; passing pieces still require checkpoint-reader interchange/removal.

The current CPU contract for translating passing causal pieces into checkpoint tensors is
`basis_aligned/polynomial_causal/TEMPORAL_ISWAS_P7_WEIGHT_TENSOR_TRANSLATION_PLAN_2026-09-08.md`.
It maps A11 head deltas through the corresponding `W_O` slice and M11 product-factor deltas through
`Down`. Weight overlap only nominates an edge; identification still requires downstream reader
interchange, upstream writer reproduction, composition, selective removal, and sealed transfer
when the component was not prospectively nominated.

The newest clocks are `HOURLY_STRATEGIC_REVIEW_2026-09-08_1322.md` (next safe-boundary review
after 14:22 UTC) and `THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_1126.md` (next after 14:26 UTC).
Both `bqrunner` and `bqrunner2` are Supervisor-managed with autostart/autorestart and were healthy
at this snapshot. Never start duplicate background runners; use `ops/enqueue.sh` only.

## Required startup sequence

1. Work in `/workspace/tensor_language` and read
   `/root/.agents/skills/bilin18-research-driver/SKILL.md` completely.  Follow its
   anti-pause, circuit-focus, queue, review, and continuation rules.
2. Read only the current state slices first:
   `AGENT_BOARD.md`; the first current entry in
   `basis_aligned/polynomial_causal/explanations/README.md`; the tail of
   `BILIN18_CONNECTION.md` and `BENCHMARK_BACKLOG.md`; the newest hourly and three-hour
   reviews; recent Git commits/status; and the managed-runner state.
3. Treat the worktree, result files, queue, process handles, and Git history as authoritative.
   The prose below is a locator, not permission to ignore newer evidence.
4. Inspect the managed services and queue before doing GPU work:

   ```bash
   supervisorctl status bqrunner bqrunner2
   tail -n 30 basis_aligned/bilinear_quotient/runlogs/runner.log
   sed -n '1,30p' basis_aligned/bilinear_quotient/queue.txt
   nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
   ```

   Both services are configured with `autostart=true` and `autorestart=true`.  Do not launch a
   duplicate runner or a direct GPU process.  If an authoritative Supervisor status says a
   runner is stopped, start that Supervisor service; do not replace it with `python ... &`.
5. Queue GPU experiments only with:

   ```bash
   cd /workspace/tensor_language/basis_aligned/bilinear_quotient
   EXPECTED_SHA256=<reviewed-runner-hash> bash ops/enqueue.sh /absolute/path/to/runner.py
   ```

   Use `FORCE=1` only to retry the same scientific experiment after a verified execution-only
   failure and an explicitly recorded instrument repair.
6. Preserve other agents' dirty files.  Stage exact owned paths only, commit each durable unit,
   and push it: `/workspace` is not backed by a persistent Vast volume.
7. Do not ask for routine permissions.  The user has explicitly authorized in-scope research,
   repository writes, managed execution, commits, and pushes.  Ask only if genuinely new
   authority or a materially different external action is required.

## Periodic research clocks

At the first safe boundary after each clock expires, perform the review and immediately take its
chosen action.  A reminder or review without a concrete continuation is not progress.

### Hourly circuit review

Use the timestamp in the newest
`basis_aligned/polynomial_causal/HOURLY_STRATEGIC_REVIEW_*.md`; do not duplicate a review inside
the hour.  Restate the seven circuit interpretation targets: computational specification;
cross-boundary grouping and within-module splitting; held-out/OOD prediction; extraction or
sufficiency; selective manipulation; composition/reuse; and stable identification.  Then audit
the full goal, new evidence and corrections, confounds, alternative approaches, ranked next
moves, serial candidate throughput, and these exact lines:

```text
CIRCUIT_FOCUS: PASS|FAIL
CEREMONY_BUDGET: PASS|FAIL
NOVELTY_LESSON_GATE: PASS|FAIL
```

A failed line forces the next bounded block to repair that workflow before unrelated research.

### Three-hour mathematical review

Use the newest
`basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_*.md` as the clock.  Define the
actual model and circuit target as tensors, dimensions, contractions, polynomial degree,
symmetries/gauges, allowed inputs, preserved outputs, norms, and literal prices.  Map candidate
theorems or algorithms object-to-object, list their assumptions and our violations, and derive at
least one executable circuit consequence.  Begin the best consequence immediately after writing
the review.

## Authoritative restart delta at 2026-09-07 23:00 UTC

### Live continuation at 06:55 UTC on 2026-09-08

The four-head temporal/is-was program has now passed the confirmation stages that were still
pending in the 05:55 handoff:

- `L9H1 + L9H4 + L11H3 + L15H5` passed simultaneous joint composition on the original bank;
- the frozen dual-command OOD authority passed native capability in 31/32 cells at 8/8 and one
  cell at 7/8, then the same four-head program passed the prospective OOD joint-composition bars;
- a registered H4 midpoint clamp selectively removed the intended half-command effect on original
  and OOD rows with unit reduction, direction preservation, low collateral, and additive joint
  removal.  This upgrades the H4 addition from a predictive screen to a manipulable program
  component for this intervention family;
- the reader-factor split showed that `L11H3:v` is the strong stable reader, while the proposed
  `L15H5:q/q2` explanation failed: q is anti-causal, q2 is weakly positive, and their combination
  largely cancels.  Preserve that falsification; do not use the dominant L11 effect to rescue the
  L15 interpretation.

The source-localization receipt,
`basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_l11h3_value_source_region_localization_v1_result.json`,
returned terminal `invalid`.  Its region selections (`bridge` for temporal and `postcue` for
is-was) were initially unclaimable because prediction A failed: the runner compared a parent-relative
full-effect cosine to an older command-gold cosine, producing an apparent replay error.  Preserve
the immutable invalid receipt and do not change A or retro-pass it.  The separately registered
three-forward replay audit has now passed with zero like-for-like error and exactly reproduced the
`.028743869178895265` mixed-target error.  It licenses interpretation of the already-frozen B--E
outcomes: the value-source partition is compositional and stable on OOD text, but task typed;
temporal selects the unchanged bridge and is-was selects postcue.  The follow-up 16-corner
q/k/q2/k2 factorial has now passed A/B/C/E with D false, terminal `stable_routing_invariant`.
Is-was donor routing changes the value-parent effect by only 5.6--8.1%; temporal selects k+q+q2
but the full routing change is only 8.1--12.3% of the value parent.  Recipient-native attention
routing therefore carries most of the stable value effect.  The exact native-routing source-term
extractor has since passed all gates: its formula matches the c_v-patch head delta within `9.54e-6`
and direct L11H3 interface installation matches selected logits within `8.59e-6` on original and
OOD text.  Four checksummed 128x128 query-tensor banks are stored in the result.  The immediate
continuation is exact upstream writer-weight translation through L11H3 W_v followed by a causal
source-position module/head screen.  The exact atlas has now frozen six unique head candidates:
shared L6H7/L7H7/L9H1/L9H4, temporal-only L5H1, and is-was-only L3H4.  Original/OOD top-ten
  rankings are identical and shared enrichments are 6.11x--17.25x.  Treat these as weight
  candidates only until the all-singleton plus task-top-five causal source-position screen lands.
  That screen has now landed valid at SHA-256 `d00fdf1f...`: L9H1 and L9H4 each pass the frozen
  shared cross-task causal-writer singleton gate, both top-five unions pass, and singleton-sum
  composition has maximum relative L2 `.17425` and minimum cosine `.98560`.  The temporal union
  recovers about `.55-.58` of the L11 value effect; the is-was union overshoots at `2.30-2.91`,
  making it causally sufficient but overcomplete.  The preregistered distributive branch therefore
  requires a weight-ordered greedy prefix successor, selected only on original FIT and then frozen
  for original HOLDOUT/OOD validation.  Do not run the 32-subset interaction branch.

At this checkpoint `gpt-6-astra` is the configured Codex default and is visible in the account
catalog.  Both `bqrunner` services are Supervisor-managed and healthy; their queues are currently
drained after the QK factorial.  The latest clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0715.md` (next review after 08:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_0526.md` (next review after 08:26 UTC).
Inspect Git, files, runner logs, and queue state before acting because they remain authoritative
over this prose.

### Live continuation at 05:55 UTC on 2026-09-08

The composition branch has advanced past the 04:42 prerequisite state:

- physical temporal Q8 and both is-was planes form a stable task-typed union rather than one shared
  subspace (`temporal_iswas_common_final_gauge_basis_capture_v1_result.json`, terminal
  `task_typed_direct_sum`);
- a complete exact-weight atlas finds shared physical interfaces despite distinct states: L9H1,
  L11H3, and L15H5 writers plus L11H3:v and L15H5:q/q2 readers;
- the first dual-command bank is a preserved native null, while the prefix-preserved successor
  licenses all 32 same-sequence rows;
- no singleton parent module reaches 0.50 recovery, so that result remains a registered null;
  nevertheless L9H1+L11H3+L15H5 is a licensed distributed program, recovering temporal
  `.706/.752` and is-was `.523/.491` on FIT/HOLDOUT with at most 1.13% cross-command interaction;
- the frozen greedy extension selects only L9H4, raising recovery to temporal `.833/.848` and
  is-was `.696/.669`. L8H1 is not selected because the four-head arm already meets the prospective
  quality bars with fewer additions.

The current decisive job is hash-bound in managed lane 1 behind live unrelated `v252`; do not
enqueue a duplicate:

- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_four_head_union_joint_composition_v1.py`;
- reviewed SHA-256: `aeee380e62576074ba0ad31217506a1dd48b226a2d3720f0dd05580f58c03e18`;
- result: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_four_head_union_joint_composition_v1_result.json`.

It must reproduce the selected four-head single-command metrics, then confirm T/I/TI composition
with <=10% interaction, >=.99 additive cosine, <=.05 simultaneous recovery loss, <=.01 collateral,
and exact later-to-earlier causal zero. Passing promotes L9H1+L9H4+L11H3+L15H5 as the higher-quality
joint program. Failure preserves the already-licensed three-head program and closes L9H4 for
simultaneous use. Do not change the union or bars after the result.

Latest reviews are `HOURLY_STRATEGIC_REVIEW_2026-09-08_0515.md` (next after 06:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_0526.md` (next after 08:26 UTC). Latest branch commits
at this handoff include `ee991bf65` (H4 joint runner), `743b24c52` (greedy result), and
`6914642c6` (licensed H3 joint composition). Both bqrunners remain Supervisor-managed and healthy.

### Live continuation at 04:42 UTC on 2026-09-08

The latest strategic decision is no longer router feature engineering. Three increasingly explicit
objects—tied-embedding routing, contextual Gram moments, and projective response shape—failed
stable selective routing, including catastrophic A1/A2 exchange on the v16 construction. The
04:15 hourly review closes that branch and redirects to simultaneous task composition and physical
weight-readable command coordinates.

Two prerequisite jobs are hash-bound in managed lane 1 behind the unrelated live `v248` run, in
this order; do not enqueue duplicates:

1. `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_common_final_gauge_basis_capture_v1.py`
   at SHA-256 `a2bdc76d61c54e6879d84a0a5b451ae39ed798bcea74fdb77e10579ee08d78cd`.
   It stores the temporal 1152x8 Q8 basis and both cross-fitted is-was 1152x2 bases in the same
   physical final-residual gauge and decides shared state versus a task-typed direct sum.
2. `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_dual_command_native_capability_v1.py`
   at SHA-256 `0ebc5d1bd8cbcde0db8bee0aa09e7a5384ce35f58df0002f87e4fb0809d689f9`.
   It uses one model forward for 128 same-sequence 2x2 endpoints and scores temporal plus is-was
   auxiliaries separately in 32 hard capability cells. No rows or templates may be filtered.

The exact joint-command Möbius/additivity scorer is already implemented in
`basis_aligned/bilinear_quotient/ops/joint_command_composition_contract.py`. A causal joint
factorial is eligible only if the native capability job issues its all-row license and the basis
capture is mechanically valid. Read and publish both immutable outcomes first, then choose the
shared-basis or task-typed intervention exactly as the basis terminal directs. If native capability
fails, preserve the null and do not tune or filter this bank post hoc.

Current commits are `264af0b60` (dual-command native gate), `cb75e4b2d` (dual-command authority),
and `2ccef52ef` (joint composition contract). The latest clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0415.md` (next due after 05:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-08_0226.md` (next due after 05:26 UTC).
Both bqrunners should remain Supervisor-managed and automatic; verify them at startup as specified
above, and immediately restore queue depth if either lane is unexpectedly empty.

### Live continuation at 00:38 UTC on 2026-09-08

The current decisive job is already hash-bound in managed lane 1; do **not** enqueue a duplicate:

- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py`;
- reviewed SHA-256: `4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3`;
- current queue relation at this checkpoint: depth one behind live `v244`;
- result path when it lands: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1_result.json`.

This no-refit factorial asks whether each stable construction-specific oracle retains its own
target transfer when layer-15 attention is live.  Predictions B/C are the admission gate.  If
either fails, close the L15H5 causal-reader hypothesis for this intervention family: the static
weight alignment was not on the exercised native route.  If both pass, execute the already
preregistered full layer-15 head/module mediation atlas, not an immediate Q/K/V claim:

- prior: `basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_attention15_head_module_mediation_atlas_v1.json`;
- pure intervention/accounting contract: `basis_aligned/bilinear_quotient/ops/head_response_mediation_contract.py`;
- contract commit: `223369b3f` (seven focused tests pass).

The atlas crosses upstream off/on with absolute downstream-response source off/on for the whole
layer-15 attention module and each of its nine heads.  It separately measures rescue, reset loss,
bypass, and interaction.  Only a passing singleton-composition test licenses a parity-cross-fit
greedy head union; otherwise use an interaction-aware head-set test.  Only a passing L15H5
head-level result licenses its later Q/K/V split.

The latest review clocks are
`HOURLY_STRATEGIC_REVIEW_2026-09-08_0015.md` (next safe-boundary review after 01:15 UTC) and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2326.md` (next after 02:26 UTC).  Both `bqrunner` and
`bqrunner2` were healthy at this checkpoint.  Re-check files, process state, queues, and newer Git
commits on restart because those facts can advance after this document is written.

### Live continuation at 23:30 UTC

V16 capability has landed.  The immutable first receipt reports `null` only because it asked for
24 jointly capable rows from 16-row A panels; all eight direction-by-side capability cells are
actually `1.0` and both A1/A2 have 16/16 jointly capable rows.  A hash-bound zero-model audit
translated the registered 24/32 ratio to 12/16 and returned `manifest` without causal access.

The fixed-rank multi-construction causal executor is now hash-bound in managed lane 1 at SHA-256
`8c14405674b4773303e197be04668f29ca97d2662dbeccc0f9653c119716c556`, behind live v238 and the
previously queued v247/v249 jobs.  Its prior, joint fit/selection core, sealing tests, and no-model
preflight are committed.  It fits/selects entirely on v15 A1/A2/P/C, then opens v16 A1/A2/P once;
v16 C remains excluded.  The 23:26 mathematical review also implements an exact projective-bisector
falsifier for the failure branch.  The latest strategic and mathematical clocks are now
`HOURLY_STRATEGIC_REVIEW_2026-09-07_2315.md` and
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2326.md`.

The multi-construction run has since completed validly at result SHA-256
`0eba172c8ada3e3cfcaa2826afa7a348480a6b61bde819c1a0daad9ed8dcb1a4` with terminal
`fixed_projector_infeasible_on_observed_constructions`.  No initialization passes both target
constructions in both held parities.  Aggregate v15 A1/A2 projections (`.80699/.81765`) hide this
fold failure; sealed v16 reaches only `.64172/.49549`.  Therefore the attention-15 dependency
factorial is currently ineligible.  Build and execute the separate A1/A2 oracle-axis plus exact
projective-bisector falsifier first.

Its prospective receipt is
`basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_construction_oracle_projective_bisector_v1.json`.
The reusable single-panel fit core is
`basis_aligned/bilinear_quotient/ops/construction_oracle_projector_fit.py`; it must be integrated
into a hash-bound managed runner without changing the completed parent artifact.

The integration is now complete in
`basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_construction_oracle_projective_bisector_v1.py`
at reviewed SHA-256 `f1a1fae56a6c78e4a0b093ab49032649d8d1d66ecee3f2876540417f8f40bc3d`.
After confirming that hash and the current managed queue, enqueue it through `ops/enqueue.sh`; do
not run it directly.

Scope its v16 evidence as `OOD_TEXT_REUSE_NEW_INTERVENTION`: v16 text, native capability, and the
failed joint-projector response were already open, although the new oracle/bisector intervention
is frozen entirely from v15 before its v16 contexts are constructed.  It is intervention-held, not
a new pristine task discovery.

The bisector result has landed validly at SHA-256
`dde8cc793f2ba2d1f0bf7439dd9c62cb53ad979ea4ab3363928015b8cbe7e224` with terminal
`construction_conditioned_coordinate`.  Both construction oracles are stable/effective, their
same-head cosines are only `.409-.671`, and cross-use fails.  The bisector passes v15 target bars
but retains two P flips in one parity and fails v16 A2.  A naïve routed mixture is therefore not
selective because each own expert has the same parity-0 P flips.  The active successor is the
zero-forward CPU weight-convergence diagnostic in
`basis_aligned/bilinear_quotient/ops/run_temporal_iswas_construction_oracle_weight_convergence_v1.py`.

That weight screen has now returned `reader_equivalent_distinct_writes`: the construction axes do
not generally align as W_O writes or W_V pullbacks, but all five L15H5 Q/K/Q2/K2/V responses rank
top-ten for all four sources in both folds.  The next runner is
`basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_construction_oracle_attention15_dependency_factorial_v1.py`
at reviewed SHA-256 `4d435dfa6c6f29a34de2b5ba7679aafb3b622cdf0fbf2ea779c58bea982b1fe3`.
It must be enqueued through the managed GPU lane and interpreted before any L15H5-specific reset.

Do not overstate the exact-weight result.  A hash-bound zero-model path audit proved that every
current DAS outcome clamps the complete attention-15 donor head output after layer-15 Q/K/V has
been computed.  Thus the causal projector effect can travel through the residual skip/MLP15 and
layers 16-17, but cannot validate the weight-ranked `L15H5` Q/K/V reader path.  After the queued
construction result, run the frozen `upstream projector on/off x complete attention-15 on/off`
dependency factorial before any L15H5 reset/rescue claim.

That successor is now prospectively specified in
`basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_selected_projector_attention15_dependency_factorial_v1.json`,
with its derivation in
`basis_aligned/polynomial_causal/explanations/TEMPORAL_ISWAS_SELECTED_PROJECTOR_ATTENTION15_DEPENDENCY_FACTORIAL_DESIGN_2026-09-07.md`
and reusable CPU accounting in
`basis_aligned/bilinear_quotient/ops/two_by_two_dependency_contract.py`.  It is eligible only if
the multi-environment parent is mechanically valid and its registered joint-v15 feasibility
prediction passes.  Otherwise run the already committed projective-bisector falsifier first.

The target-feasible DAS run described below has completed.  It validly beats matched difference
in means (DIM) on cross-fitted A1 target transfer (`.8719` versus `.7250`) with stable rank-one
directions (minimum principal cosine `.8474`), so optimization is not the null.  It does not pass
construction-general identification: sealed A2 is `.6489`, below `.75`, and one P-control fold has
mean KL `.01945`, one flip, versus DIM `.00678`.  A complete 30-configuration red-team found that
the registered noise/Jacobian regularizers change selection scores by less than `.0005` and never
improve flips.  The current diagnosis is missing construction variation, not insufficient local
regularization strength.

The next multi-environment selector is already implemented and tested.  It keeps rank one fixed,
requires every fitted target construction to pass separately before control scoring, and selects
using worst P/C panels.  A history-disjoint third construction (`v16`: Right now/Back then and In
these/those days) is hash-bound in managed GPU lane 1 for a capability-only two-forward gate:

- runner:
  `basis_aligned/bilinear_quotient/ops/run_tense_auxiliary_is_was_fresh_lexicon_v16_capability_v1.py`;
- reviewed SHA-256:
  `333c5398566539d8ba1f8940ab894d2ed8db9f1449caf73d5fd8d0156d40c34f`;
- queue order at this checkpoint: immediately behind live `v236`, before `v238` and `v247`.

Do not inspect a causal v16 outcome before the capability gate lands.  If it passes, score and
publish it, then use v15 A1/A2 as separate fitted environments and keep v16 sealed for the fixed
rank-one multi-construction causal transfer.  If it fails native capability, preserve the null and
do not repair the text post hoc.

An exact zero-forward weight translation of the learned four-head directions is also complete.
It maps each head coordinate through $W_O$ into residual space, ranks downstream Q/K/V and MLP
interfaces, and pulls it backward through $W_V^{\mathsf T}$ to rank earlier writers.  After a
preregistered scale-aware float32 audit, the diagnostic is valid: fold cosines are
`.9368-.9923`; every source ranks all five inspected `L15H5` interfaces in its top ten; and writer
pullbacks recover `L8H1 -> {L9H1,L9H4} -> L11H3` at `.9725-1.0` percentiles.  Treat this as an
explicit weight-compatibility hypothesis, not causal identification, until the sealed
construction transfer succeeds.

Latest Codex commits are `5f4acf140` (published weight-interface audit), `000d94082` (audit gate
repair), and `d1e461377` (audit preregistration).  The canonical dossier remains
`basis_aligned/polynomial_causal/explanations/CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md`.
The latest hourly clock is `HOURLY_STRATEGIC_REVIEW_2026-09-07_2215.md`; the next review is due at
the first safe boundary after 23:15 UTC.  The latest mathematical clock is
`THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2026.md`; the next is due after 23:26 UTC.

## Earlier circuit state at 2026-09-07 21:07 UTC

The canonical dossier is
`basis_aligned/polynomial_causal/explanations/CIRCUIT_temporal_iswas_rank46_task_modes_2026-09-07.md`.

The v15 construction changed the causal graph enough that the old rank-16 five-MLP source program
failed. Complete native-module patches localized the changed behavior primarily to attention
layers 8, 9, and 11. Exact head patching then identified a distributed four-head core: `L8H1`,
`L9H1`, `L9H4`, and `L11H3`. These are real within-module splits, not singleton-sufficiency claims.

Aligned, capability-qualified P/C controls repaired an earlier absolute-token-position defect. The
repaired exhaustive five-piece attention lattice is valid and shows that complete head responses
recover A1/A2 behavior (`.80535/.86829`) but are not selective: P has five flips and C has three. A
cross-fitted linear complement removes P collateral but also collapses A1/A2, rejecting simple
linear task/nuisance separation.

The exact pattern/value/interaction decomposition is complete. Its mandatory shared lesson is that
an absolute downstream clamp differs from adding a donor-minus-base delta to a live head already
changed by upstream interventions. The valid absolute-clamp result localizes most target transfer
to value content; pattern and interaction pieces are comparatively selective but too weak.

The latest completed screen is:

- prior: `basis_aligned/bilinear_quotient/circuits/prior_art/temporal_iswas_v15_head_factor_dual_greedy_v1.json`;
- runner: `basis_aligned/bilinear_quotient/ops/run_temporal_iswas_v15_head_factor_dual_greedy_v1.py`;
- runner SHA-256: `8c02ee2a04faad82c3341667f457c30537f351c9147a26dadd7b19736a5d3a48`;
- result: `basis_aligned/bilinear_quotient/circuits/followups/temporal_iswas_v15_head_factor_dual_greedy_v1_result.json`;
- result SHA-256: `51c705281cc4fa10aa7d5c3c54bc3b1ee1e9d6d3af3e74f3b98cef3a34ad0285`.

It is mechanically valid at 152 observed forwards and returns
`no_selective_dual_greedy_program`. The strongest zero-flip/low-KL visited arm reaches only
`.13763` A1. Target-first combinations reach `.84510` A1 but still flip five P and two C rows. The
full 12-factor arm replays its parent within `4.62e-7`, reaches `.80535/.86829` A1/A2, and flips
five P plus three C rows. This null is path-local, not an impossibility theorem.

## Earlier continuation: target-feasible regularized DAS (completed; see restart delta)

Use the protocol derived in
`basis_aligned/polynomial_causal/THREE_HOURLY_MATHEMATICAL_REVIEW_2026-09-07_2026.md`:

1. Make cross-fitted A1 transfer a hard feasibility constraint rather than optimizing a soft
   target-minus-complement score.
2. Among feasible projectors, minimize worst-group P KL plus Gaussian-noise/Jacobian sensitivity
   and cross-fold projector instability.
3. Select dimension, noise, and regularization on opposite A1/P parity only; keep A2/C sealed.
4. Use orthonormal projectors and a fixed unit intervention dose so a zero-effect complement cannot
   win by construction.
5. If the subspace passes, translate it through native QK/OV and downstream weight tensors to
   identify explicit writers/readers, then run fresh held-out manipulation.
6. If no target-feasible projector exists, preserve the null and change the object to nonlinear or
   input-conditional subspaces rather than another complement-only optimizer.

For every completed result, score the registered predictions exactly, preserve nulls and invalid
instruments, update the dossier and `AGENT_BOARD.md`, commit and push exact owned files, then begin
the evidence-selected successor. Before ending a turn, require a continuation receipt: an
append-only board claim and either completed CPU analysis, a committed preregistration with
implementation underway, or an audited job in the managed queue/runner.

## Suggested first prompt in a new session

```text
Read /workspace/tensor_language/CODEX_RESEARCH_SESSION_STARTUP.md completely, then read the
bilin18-research-driver skill and the current-state slices it specifies. Resume or create the
durable bilin18/Theseus circuit-finding goal and actively continue it from authoritative Git,
result, board, queue, and runner state. Keep the hourly circuit reviews and three-hour mathematical
reviews active using the newest timestamped artifacts as clocks. Keep Supervisor-managed bqrunner
and bqrunner2 running by default, inspect their state at startup, and send all GPU experiments only
through the hash-bound ops/enqueue.sh workflow—never direct GPU Python and never duplicate a queued
job. Interpret every result, preserve nulls and invalid instruments, commit and push durable units,
and begin the evidence-selected successor before stopping. The user has granted full permission
for in-scope research, repository writes, managed executions, commits, and pushes; do not ask for
routine permissions. Do not mark the overall goal complete merely because one rung finishes.
```
