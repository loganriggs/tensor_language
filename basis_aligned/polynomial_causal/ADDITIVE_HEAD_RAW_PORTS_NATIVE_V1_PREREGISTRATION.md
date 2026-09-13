# Three raw-state additive head reconstruction: native validation

Frozen13September2026 before native scoring.120 reusedregionalprefixes, fiveN/C/P/R/A16referencearms,600forwards720readouts,180seconds. No newholdout or datafit.

A: all120priornativefive-arm outcomes andfirstfourreadouts replay<=1e-4relative, independentfullportstate<=2%allfivegroups.
B: rebuilt fullmixed direct-write effect fromthree rawpreattention states predicts nativehead17.2 effect<=2%relative norm eachfivegroups.
C: rebuilt compactthree-group effect predicts previousnativeportcompact effect<=2%relative norm eachfivegroups.

Candidate usesonlyN/C/R preattentionrawstates andsharedfirstvalues. Reconstruct rawA=rawC+rawR-rawN andrhoAfromthatsum. Itreceivesno Aquery/key/value or Arawstate; nativeAexecution isreferenceonly. Actualmodel matrices andnestedRMS epsilonare retained. Finalnativeadditivebackground andreadout stillsupplied. Fivefullforwardarms arevalidation costs, notclaimedreducedforwarddeployment. NativeFP32rounding mayinvalidate exactreal-arithmetic corner relation atsmallmixed-effect precision; preservefailures andreportabsoluteerrors.

Opposingpredictions: exactweightgeometry closesinputcorner sufficiently, ornativearithmeticerrors amplifybeyond2%. AlreadyexecutedFP64control<=2.2e-15 andnullspacefalsifierforcesresidualnormdependency. No normomission, rankfit, semanticfactorclaim or four-propertypromotion.
