# MLP9 mediator within the attention9-clamped bypass

Same32worlds, fullfive-factorcubes. SourceS0=native;S1=subtractactualQohMLP8output at
allpositions. Capture native A9write and unmodified/sourceedited A9writes as clamp
tripwires. Capture M0=nativeMLP9output; M1=MLP9output underS1 withA9heldnative.
Execute allfour F_s(M_t), alwaysclampingA9native andlettingalllaterconsumersrecompute.
Neither source-dependent normalization nor the MLP9 product is linearized.

Definitions: total=F00-F11 (the priorbypass), mediator=F00-F01,
remaining=F00-F10, interaction=total-mediator-remaining. Remaining includes direct
carry andotherlatercomputations; notonesemanticmodule. Save three-reader outputcube
and fullvocabulary effectmetrics; no fullvocabulary success inferredfromthreeanswers.

A instrument: native/sourceedited/bypass captures replay priorSOURCE_ATTENTION
F00/F11/F10 respectively; allfour combinations preserveA9firstV; incomingMLP8,
A9 andMLP9 patches match captured source-specific outputs bitwise. Bothdiagonals
replaycapturednative/bypass; allbridges maxabs<=1e-3 ANDrelative<=1e-5. Finite,cleanup,
6captureforwards+8cube perworld=448forwards7168seq,0fits,600s cap. Shared source,
attention, joint-factorialcontrols reused. All545902902weightsretained,saving0.

B MLP9dominance: mediator effect agreeswithtotalwithin.10relative inQoh correctmargin,
centeredthree-readers ANDcenteredfullvocabulary, EVERYworld. Total/nativemixednorm
>=.10 forcorrectmargin/three-readers. Zero scientific denominators fail.
C remainingdominance: same forremaining. D smallinteraction: interaction/total<=.10
inmixed ANDfulltable versions ofallthree readouts, EVERYworld. Report task-only
passes separatelywithoutpromotingthemtofullvocabularysuccess. If B/Cfail, keepboth
routes; no winnerbysmallererror. IfMLP9notdominant, no automaticadjacentlayerwalk:
compare a broaderlate-responsepartition. Existingexactresponsecontrol isalgebra,
notnativeMLP9dominanceevidence. No newOOD/semanticlabel/rank/dosetuning.
