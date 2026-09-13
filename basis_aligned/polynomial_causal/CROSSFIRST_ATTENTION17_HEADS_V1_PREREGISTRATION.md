# Native headwise mixed attention17 write

13 September2026. 160existingprefixes,800fullforwards,1920finalreadouts,180second limit. No fitting. Prior17.2 regional source-block dossier checked; this is a distinct finite mixed input under additive background, not rediscovery of thathead.

Nativehead outputs immediatelybefore c_proj are yN,yC,yR,yA. For eachhead h, delta_h=yA_h-yC_h-yR_h+yN_h; project v_h=W_h delta_h withactual c_proj columns. BothQK factors, RoPE, allsource positions, current/first values andnormalizers are generated natively. Full mixed directwrite fromactualpreMLP states remains independent reference, includingnativeaffinerounding.

Readout arms atfixed finaladditivebackground: base, independentfullv, sumheads, nineindividualheads. A: nativefive/base/direct outcomesreplay<=1e-4relative eachpanel, headsumstate error<=2%eachpanel. B: knownhead17.2 effect approximatesfullvwithin20%relative allfourregionalgroups. C: sumseparateheadeffects predictsjointsumwriteeffect within5%relative allfourregionalgroups. Otherheadrankings andFineWeb descriptive; no threshold retuning or heldoutclaim. Report allheads andabsolutecontrolerrors.

Headsum equality in realarithmetic follows linearoutputprojection. Native rounding maymatter for smallmixedstates; retain measuredreplayandabsoluteeffecterrors. This is a directwrite screen; previouslyvalidated MLPcorrection is omitted here and must be restored/tested for anyextractedfullresponse. No nativeheadboundary presumedsemantic.
