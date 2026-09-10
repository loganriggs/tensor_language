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
and compare to nativefinal-gamma*QMLP8readout, maxabs<=1e-3 ANDrel<=1e-5;
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
