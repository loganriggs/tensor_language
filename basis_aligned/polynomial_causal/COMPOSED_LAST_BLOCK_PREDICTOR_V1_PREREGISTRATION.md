# Composed last-block local-interaction predictor

13September2026.120 reusedregionalprefixes. ExecuteonlyN/C/R nativeconditions,360fullforwards,360extra lastMLP evaluations,480readouts,180seconds. No A/Ptrajectory ordatafit.

Let zBar=zC+zR-zN bethepreMLP17 additiveinput, hBar=hC+hR-hN thefinaladditivebackground, andg(z)=z+MLP17(RMS(z)). Reconstructedhead17.2 mixedwrite v2 iscomputed bythevalidatedthreeprojectedcornerinterface. Candidatefinalstate isg(zBar+v2). Its localmixedstate isg(zBar+v2)-hBar, equivalently[g(zBar)-hBar]+[g(zBar+v2)-g(zBar)]. Thiscomposes theMLP localinteraction withtheattentionpathanditsdownstreamresponse. A compactvariantusesthefixedthree-contraction v2. MLP-onlycontrolg(zBar) isscored.

A: priorN/C/Routcomes andhBarreadout replay<=1e-4relative.
B: fullv2composedlocaleffect predicts nativewholelastblocklocalgeneration effect<=10%relative eachfivegroups.
C: compactv2composedlocaleffect predicts samewholelastblockeffect<=10%relative eachfivegroups.

ReferenceY_A16 andhBar come fromearlierfrozen nativeartifacts and areloadedonlyafterscoring. NativeA16 =lastblockonadditiveblock16input; targetY_A16-f(hBar). This isnotoriginaltotalP-C-R+N. Bothfullandcompact readoutpredictions now includeMLP17propagation; earlierdirectheadwrite errorbarsdo notapplyautomatically.

Null: omittedotherheads, nonlinearMLP propagation orFP32cancellation prevent accuratecomposition. ReportMLP-onlycontrol, perprefixsigns, absoluteerrors andscalingrelativeoriginalnonadditivity inCPUfollow-up. FullMLP17weights andthreezinputs arenative dependencies,notanewtinyfactorization. ExactresponsealgebrawaspreviouslyvalidatedinCROSSFIRST_LAST_ATTENTION_RESPONSE_V1_CONTROL; existingnativeMLPusedhereavoidsunnecessarynewkernel.
