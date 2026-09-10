# Direction of coupling between later attention and MLP responses

All32worlds, completefive-factor cubes, sameM8mixedsource removal andA9nativeclamp.
BankA is attention10..17 output under editedsource withMLP9..17nativefrozen.
BankM is MLP9..17 output undereditedsource withattention10..17nativefrozen.
Each bank is separately generated; never use fullyeditedwrites asonewaybanks.

Cube Z[am,ma]: am enablesA-to-M (MLPs live), ma enablesM-to-A (attention live).
Whenma=0 installBankA; whenam=0 installBankM. Thus00installsbothindependent banks,
10hasBankA/MLPlive,01hasAlive/BankM,11bothlive. A9alwaysnative; sourceeditunchanged.
Nativeall-clampidentity uses nativebanks andzero sourceedit. Observerscheck every
frozen modulewrite bitwise andfirstVunchanged. Livegroups recompute in currentcontext.

A instrument: allnative/bankA/bankM/fullsourceoutput replays parentLATE_GROUPS_V2
native/Z10/Z01/Z11 respectively (three-readers); nativeidentityfullvocabulary replay.
Allbridges maxabs<=1e-3 ANDrel<=1e-5, bitwisesourceincoming, clampaudit, finite/cleanup.
Native2+bankA2+bankM2+nativeidentity2+fourarms8 perworld=512forwards8192seq,0fits,
600s cap. ReuseWclamps andplantedlinear A-M, M-A, A-M-A controls. No rawstateclosed-
form identity required. All545902902weights, nativecounterfactuals retained,saving0.

Coupling total=Z00-Z11, AM=Z00-Z10, MA=Z00-Z01, I=total-AM-MA.
B AMdominance: AMmatchescouplingtotal<=.10relative for Qcorrectmargin,
centeredQthree-readers ANDcenteredQvocabulary, EVERYworld. Requirecouplingtotal
centeredQvocabularynorm>=.05 offullbypasseffect(native-Z11). C MAdominance:same.
D smallinteraction:I/total<=.10 in mixed ANDfulltable versions ofallthree readouts,
EVERYworld. Zero scientific denominatorsfail. Reporttask-onlypasses separately.
IfB/Cfail, retainbidirectionalcoupling; no winnerbysmallererror. Dfailuremeansboth
switches matterjointly, notproof ofsource nonlinearity orliteralalternatingpathcount.
No source-dose/head/rank/row/barrescue andno freshOOD orindependentcircuitclaim.
