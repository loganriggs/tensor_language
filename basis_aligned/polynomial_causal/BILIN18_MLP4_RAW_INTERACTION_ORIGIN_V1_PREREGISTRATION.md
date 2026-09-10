# Origin of mixed raw input before MLP4: attention4 versus earlier residual

Registered before native execution. Original bilinear handoff/pilot authority.
The normalized-input three-corner predictor failed; its criteria and result
remain binding. This new screen is diagnostic localization of a fourth-corner
interaction at two input edges, not a new normalized prediction candidate.

Use all35evaluation squares andallfour missing-corner directions from the
frozen SEMANTIC_SQUARE_ROWS_V1_AUDIT.json. Same token/batch/answer geometry as
SEMANTIC_SQUARE_V2;70paired rows, nofit/filter/anchor selection. Openedfamilies,
lexical-backgroundbundle context, disjointcue/contexttokenchanges. NofreshOOD.

For eachnativecorner capture rawpreMLP4input x, actualnormalizedMLPinput n,
MLPoutput m, ten residualsources, andfull endpointlogits. Sourcecapture follows
native FP32 recurrence; require RMS(x) bitwiseequal n. Let a denote the native
attention4output (unscaled coefficient1 atthisinput). For eachdirection relabel
00=othercontext/receivingcue,01=othercontext/targetcue,
10=receivingcontext/receivingcue,11=receivingcontext/targetcue.

    D = (x11-x10) - (x01-x00)
    A = (a11-a10) - (a01-a00)
    P = D-A.

P includes the incoming residual's mixed contribution and native addition
roundoff. Also record the independently transported mixed attention0..3/MLP0..3
writes andtheir difference fromP. Transported directembedding mixedterm must
beexactzero usingpaired-edge subtraction; it doesnotremoveembeddingdependence
ofcontextualproducers ornormalization. SourcefoldFP64/native comparison uses
relativeFrobenius<=1e-5; reportabsoluteerror withoutassertingFP64==FP32.
The actual rawrecurrencebitwisecheck is the primaryforwardcontract.

Source fields, with synthetic raw arithmetic inFP64 then cast to nativeFP32:
full=x11; no_mixed=x11-D; attention_only=x11-P; earlier_only=x11-A.
Apply actualFP32RMS andnativeMLPweightformula to eachfield. This is an output
replacement atMLP4 in the receiving-native context; it doesnot globally edit
an earlier producer. The fourthcorner attentionwrite itself dependsonearlier
computation evenwhenits directresidual mixededge isremoved. Do notcall the
attention_only arm an independentlyexecutingattention4circuit.

Perdirection fullnativeforwards: identity(actualm10), full(recomputedm11),
no_mixed, attention_only, earlier_only, direct_oracle(actualm11). Fourcorner
nativecaptures plus4directions×6arms perpanel=28forwards perpanel,
112forwards/980sequenceinstances total.16nativecornerMLPformulaoracles reused
forfull,plus48changed-fieldlocalMLPsyntheses=64localcalls. No fits/backwards,
600secondwatchdog, managedGPUonly. No subset/dose/normalization/anchor rescue.

A instrument: frozenrowbuilder/manifest/lineage/backend/checkpoint/parenthashes,
sharedtinylineagecontrols, exactnativeRMSrecurrence; sourcefoldrelative<=1e-5,
bitwisezero embeddingmixedterm; nativeFP32MLPformula replaylocalandidentityfull
logits, full/directoraclelocalandlogits andparentV2fullsource margins/norms at
maxabs<=1e-3 ANDrelFrobenius<=1e-5 (normsummaryrelative1e-5). Exactcounts,
finiteoutputs, untouchedpadding, restoredhooks. Nativefullsourceeffect and
full-minus-no_mixed corrections live>1e-8 inbothframes/everycell.

B attention_only localization: its effect relative to no_mixed must predict
full-minus-no_mixed withrelativeerror<=.10 inBOTHcentered50304-logit and
answer1-minus-answer0margin frames, EVERY16panel/direction cells.
C earlier_only localization: same criterion for earlier_only, independently
registered. These opposing localized-source hypotheses canbothfail; neither
passing wouldbyitself identify a semantic circuit orsource independence.
D materialmixedinput: ||full-no_mixed||/||full-native||>=.10 inBOTHframes,
EVERYcell. This prevents a tiny correction beingdescribedas theprimarycue
carrier. Also reporttotalfullsource predictionerrors diagnostically; no_mixed
is a baseline, not an alternative promoted predictor afterthepreviousnull.

All545902902nativeparameters, fourcornergeneration andwholebackground remain
charged; actualstructuralsaving0. Local algebra/support source norms are not
causalimportanceornewoperation discovery. Anyclaim ofselectiveremoval,
extraction, composition ornewfamilyOOD needs separateevidence afterscreening.
