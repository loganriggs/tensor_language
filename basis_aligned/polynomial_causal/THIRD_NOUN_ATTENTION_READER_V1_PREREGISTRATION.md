# Attention versus MLP mixed-state readout factorial v1

The previous allMLP interaction removal failed behavioral dependence despite large rawvectorprojection. This run tests the remaining attention contribution in both backgrounds, using actual reader and normalization statistics. Same256frozenprefixes/8worlds; nativeoutputs alreadyopen, interventionsprospective. Priorwork: wholemixedstatepositive, MLPfactorpartitionnull, exactRMS/softcapsource-edit algebra. No failedselector, MLPbranch, rank, phrase, orhead rescue.

Capture actualnative semantic-position attentionwrites, MLPwrites, andfinalrawstate using sharedbackend. Apply fixed conditionalPoh frommixed_state_projector_v1; transporteachwrite bylaterlambda0 products. Let A=sum18transportedattentionmixedwrites and M=sum18transportedMLPmixedwrites. Decode x, x-A, x-M, x-A-M with unchangedfinalRMS/unembedding/softcap. This is a direct-native-write factorial atfinalstate, not upstreammodule removals with downstream recomputation. It does not independently generate A orM.

Ainstrument: hashes, finite, hookcleanup, exact16nativeforwards/256seq +64decoderbatches/1024states. Parentnative margin bridge BOTHabs<=1e-3 ANDrelFrob<=1e-5. All1024twoanswerstat predictions versus GPUfullnativefinaldecoder BOTHabs<=1e-3 ANDrelFrob<=1e-5. M-arm perworld remainingmixedmarginratio replays MLPpartitionparent within1e-3; AM-arm replays wholemixedstateparent within1e-3 (native telescope rounding allowed). SharedCPUstatscontrols pass.600sexecutorcap.

Bnativeattentiondependence: originalmixedmarginRMS>=.05 and attention-onlyremoval leavesmixedmarginRMSratio<=.25 in EACH8worlds. Report attention removal conditionalonMremoved separately; its signedprojection usesoriginalnativemixedmargin asdenominator. No pooledpromotion.

Cdecoderjointcomposition: mixed Möbius term P(yAM-yA-yM+y0) RMS divided byoriginalmixedmarginRMS<=.10 EACHworld. Report full-vocabulary mixednonadditivity separately asdiagnostic. This tests composition ofsyntheticfinalstate sourceedits, not independentcircuits orunknownOODcontexts. Preserve B/Cfailures withoutchangingbars.

Save exactper-row statistics for19fixedsources:18attentionlayer mixedwrites plusoneaggregateMLPmixed. Each includesWx, Wd_i, ||x||², <x,d_i>, Gram(d_i,d_j), D, nativeFP32epsilon andsoftcap30. This supports laterCPU source-editquestions inthisfrozenbackground withoutanothernativeforward. Source order preregistered, no layer/headselection duringrun. Twoanswerreaders only; nativefullvocabularydecoder validates thefourarms butcompactstats do notpreserveotherreaders. Independentactivationproduction/semanticidentification stillmissing.

Price16nativebatchesof16,64decoderbatches/1024states,256smallCPUstatcompilations,0fits/backwards. Largestfull-vocabFP64array98.25MiB; savedstatsapproximately110kGramentriesplusreaders, noactivationbasisfit. All545902902nativeparameters retained; structural savings0. No unrelatedbehaviorselectivity ornewOODsemanticprogram claim.
