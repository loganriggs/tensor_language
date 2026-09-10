# Token-derived first-attention cue message: execution and causal carrier

Registered before native execution. Original bilinear handoff/pilot andupdated
structural criterion. Previous support theorem supplies anexact candidate
executor, not a learned semantic mechanism. Test MAINcue relevance before any
mixed-component interpretation. No head/site/edge subset,rank,dose orfit search.

Use unchanged72unfiltered evaluationpairs fromL9_SHARED_QUERY_ROUTER_V1_ROWS,
all4panels,bothdirections,all9heads/allvalidpositions. No fitting, newtextOOD or
nativeerror filtering. Pair lengths equal. Changedcuepositions aredetermined
from paired tokenIDs; this is a registeredintervention interface, not a learned
cue detector. Contextual background and all opaque weights arecharged.

The independent tokenproducer first_attention_token_program.py uses onlytokens
andweights: actual embeddingRMS, firstblocklambda operationorder, inputRMS,
Q/K/Q2/K2/Vlinearweights, headRMS, nativeBF16RoPE usingactualinv_freq semantics,
andfirstVmixing operation (even though bothfirstlayer terms share thelocalV).
No nativeprefix/model/attention.forward call orcachedactivation is used to
producefactors. Its firstlayer readscore is (QK/D)*(Q2K2/D), causal, no softmax.
Outputprojection uses theoriginalweights. The singlecueedge mask contains
causaledgeswhosequery ORsource tokenchanged. Compute masked donor-minus-base
read andprojectit to anattentionwrite delta. The mask removes unchanged terms
exactly inrealarithmetic; nativeFP32agreement is tested numerically.

Perpanel2nativeforwards capturefirst-attentionfactors/read/write/firstV plus
full endpointlogits. Produceindependentfactorbanks forbase/donor (8banks total).
For eachreceivingside run4arms:
- direct_swap: insert the complete otherendpoint attention0WRITE;
- compiled_swap: receivingnativewrite plus theindependentmaskedcuemessage;
- compiled_identity: same-token zero-change message;
- remove_cue_sources: subtract the receivingcompiled read over ALLcausaledges
  whose SOURCE is a changedcueposition. This differs from thequery-or-source
  mask used forcue deltas; do not call it removalofallcue dependence.

Preserve the receivingnativefirstV payload object forlaterlayers inALLarms.
No tuple/donor firstV swap. Native residualembeddinginjection andsuffix execute
normally.40forwards/720sequenceinstances,8tokenfactorproductions,0fits/backwards,
600secondwatchdog, managedenqueue. Sharedtuple andpadding preservation audited.

A instrument: frozenproducer/support/row/backend/checkpoint sources; tiny live
controls; independentfactors andfullwrites versus native, maskedwrite delta
versusnativedonor-minusbase atvalidpositions, compiled/directfull logits,
identity/native logits, all maxabs<=1e-3 ANDrelFrobenius<=1e-5. FP64support
identity separately maxabs<=1e-9 ANDrelFrobenius<=1e-9 usingcomputedfactors.
Counts,finiteness,padding,firstVpreservation,hook/methodrestoration. Native
paired full-vector/margin contrasts and directwrite-swap effects live>1e-8.

B causalcarrier: directwrite-swap effect signedprojection onto natural paired
donor-minusbase change >=.10 inBOTHcentered50304-vocabulary andanswer-minusfoil
margin frames, EVERY8panel/direction cells. Reverse swaps use reversed natural
contrast. This prevents faithful executionofaweakbackgroundcomponent being
promoted intoa taskcarrier. No sign reinterpretation if this fails.

C compiled effectfidelity: compiled_swap-native versusdirect_swap-native,
relativeerror<=.01 inBOTHframes/EVERYcell. Exactcomponentexecution doesnot
establishcircuitidentification, evenif Cpasses. A missesinvalidate; B/C misses
remainnulls. No posthocfirstV, headsubset, tokenposition orgain rescue.

Reportcue-source-edge removal's remainingnaturalpairedmargincontrast norm and
signedprojection. It is diagnostic here; noselectivity/adoptionclaim without
unrelatedcontrolfamilies andfreshtransfer. No new winner chosen from removal.
All545902902nativeparameters andreceivingbackground retained. The exported
changeprogram stillchargesall arbitraryweights itreads. Actualstructural
saving0; no independent smallerwholemodel, semanticruleorjointcompositionclaim.
