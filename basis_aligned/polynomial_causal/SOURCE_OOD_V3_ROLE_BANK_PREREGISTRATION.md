# Prospective frozen reader-bank test

48freshtexts,96nominatedsites,16panel/template/number/rolecells. Newactor/author/owner/reader/soldier/pilot nounpairs andalong_with/behind_subject templates; allsingle GPT2noun/pluraltokens. Inputsdisjoint frompriorv1/v2/control rows. Freeze rowandreaderbankhashes in SOURCE_OOD_V3_ROLE_BANK_BINDING.json beforeanynew modeloutcomes.

Use only RESIDUAL_ROLE_BANK_FROZEN_V1.pt rolebanks, fittedtooppositev2. No per-inputrecipientreaders, no newgradients, no refit, no outcomeamplitudeadjustment. Compute exact23native sourcevectors, contract frozen9readers, restorecanonicalnumberorientation, selectsameLP coefficients. Compare unitB(knownparentmapping) andcandidate native effects.

Primaryall16nativegate: >=80%alignednumberretentionrelative tounitB; eachof8controlL2effects<=10%candidate numbernorm. Separatenumber/controlpredictionsmustbe<=10%/5%actualnumbernorm percell. Nativecapabilitybaselinecorrectnumbermargin>=90%percell; weakcapabilitycellsretainedandflagged. Subjectandattractorresultsreportedseparatelywithoutchangingall-cellgates. This is prospectiveinputOOD,unlikeprioropenedpaneltests.

Instrumentation:oldv2readercontrollerreplaymustmatchpreviousrecipientrolebank andunitB,exact23parentcollapse<=1e-10,gaugecorrection<=1%. Shape/token/semanticpositionchecks. Model-freebindingandhashchecksbeforeexecution. Nativeprefix/suffixcounts tobefrozenwithrunner. Sourcepositionsaredeclaredinterfaceinputs,not claimed discovered bythe readerbank. Reader20,736weights+nativeprefixcounterfactuals+23sourcevectors+LP allcharged.
