# Version 2: native arithmetic repair, scientific protocol unchanged

V1 is INVALID: its real-arithmetic closed-form raw-state absolute bridge failed
(max .013671875, relative around8e-8), despite output bridges passing. V1 outcome
flags are not scientific evidence. The sole repair is an independent replay of
native FP32 operation order. Capture entry toblock8, originalnormalizedembedding,
A8write, actualsource-editedM8write andfrozen A9..17/M9..17. Evaluate every native
lambda multiply/add and residual addition in order. Require BITWISE agreement
with the actualall-frozen finalstate. This replaces the closed-form raw-state
reference; the old discrepancy remains in closed_form_state_diagnostic.
All output bridges, predictions, bars, rows, groups, interventions, counts and
closed-form decoded-output checks below are unchanged. No threshold relaxation.
The original CPU floating-point fixture demonstrates why algebraic transport and
ordered native arithmetic can differ. V1 is retained and hash-bound explicitly.

# Later attention and MLP response groups

Source is complete Qoh MLP8output removal, allpositions. HoldA9native inEVERY arm.
Groups A=attention10..17 and M=MLP9..17 each live or clamped to nativefullwrites.
Use all32worlds from fixedrolebank. Livegroups really recompute, never replay their
outputs from the fullyedited run underanotherclamp. Nativeprefix andsharedfirstV
remain accounted. Groupsare architectural diagnostic proposals, notsemanticunits.

Cube Z[a,m],0=frozen/nativewrite,1=live. Define total late response=Z00-Z11,
attention=Z00-Z10, MLP=Z00-Z01, interaction=total-attention-MLP. All are relative
allfrozen editedsource, notnativeoutput. Directcarryeffect=native-Z00; fullbypasseffect
=native-Z11. Those add algebraically tolate response; no independentrouteassumption.

A instrument: nativeall-clamp identity, sourceMLP8incomingbitwise, allfrozenwrites
observedafterclampbitwiseequalnative, firstVunchanged in everyarm, finite/cleanup/counts.
Native/fullbypass replay MLP8_BYPASS_CARRY_V1 native/bypass three-logit grids;
allfrozen replay itscarrygrid, maxabs<=1e-3 ANDrel<=1e-5. Captureallfrozenfinalstate
and compare BITWISE to independently replayed native ordered recurrence;
decodepredictedstate and compare fullvocabulary logits withallfrozen at samebars.
Native2forward+nativeidentity2+fourarms8 perworld=384forwards6144seq;
64decoderbatches1024states,0fits,600s cap. Sharedintervening_write_clamp controls
include nativeidentity, payload preservation andfrozenwrite directlineage. Group
sequentialcontrol rejects wrong full-edit-write replay. All545902902weightsretained.

B attentiondominance: attentionmatcheslate total<=.10relative in Qcorrectmargin,
centeredQthree-readers ANDcenteredQfullvocabulary, EVERYworld. Late total norm >=.10
fullbypass norm in coretwo readouts. C MLPdominance: same forMLP.
D smallinteraction: interaction/late total<=.10 in mixed ANDfulltable versions of
allthree readouts, EVERYworld. Zero scientific denominatorsfail; smallnorm alone
doesnotinvalidateinstrument. Reporttask-onlypasses withoutfullvocabularypromotion.
IfB/Cfail, keepbothresponsegroups andinteractions. IfDfails, nextobject is their
coupled update, not selectingthelarger group. No fittedgain/head/rank/rowrescue;
no newOOD orindependentinput claim. Structural saving0.
