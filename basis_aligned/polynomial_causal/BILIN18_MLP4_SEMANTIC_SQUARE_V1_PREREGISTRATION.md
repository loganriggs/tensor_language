# Native cue-by-context squares at MLP4

Registered before native execution. Original bilinear handoff/pilot authority,
with appended structural success criterion. No sparse-product rescue. This
screen tests whether local bilinear multiplication explains context dependence
of a cue's MLP4 write, or whether its input already carries that dependence.

Use all35evaluation squares from SEMANTIC_SQUARE_ROWS_V1_AUDIT.json:16has_a2,
8has_heldout,7is_a2,4is_heldout. Fitting squares are not used. Four paired rows
were unused in the full audit, two in evaluation and two in fitting, solely for
lack of remaining compatible context partners. No behavioral filtering. These
are previously opened rows; context is a lexical-background bundle, not a
claimed single subject variable. Cue/context token changes are disjoint.

Capture allfour native corners' normalized MLP4 inputs n, MLP4 outputs m, and
full endpoint logits. Eachcorner's batch uses its actual text and answer IDs.
For EACH receiving context and cue orientation, relabel:
00=other context/receiving cue,01=other context/target cue,
10=receiving context/receiving cue,11=receiving context/target cue.
This tests allfour missing-corner directions without selecting an anchor.
Allfour inputs have equal length within a square; edit all valid MLP4 tokens.

    B(x,y) = Down[(Left x)*(Right y)]
    u=n10-n00; v=n01-n00; w=n11-n10-n01+n00; a=n00+u+v
    local=B(u,v)+B(v,u)
    inherited=B(a,w)+B(w,a)+B(w,w)
    m11-m10-m01+m00 = local+inherited.

The actual native normalized inputs are used. Input nonadditivity w includes
upstream computation, normalization and numerical rounding; do not assign it
to a named earlier module. The synthetic three-corner input a need not be
reachable or normalized. B(a,a)+bias is a predicted SOURCE WRITE, not a
standalone token-to-logit circuit. Do not change normalization after outcome.

In the fixed receiving-native context, evaluate these source replacements:
identity=m10; full=m11; transport=m10+(m01-m00);
local=transport+local; inherited=transport+inherited;
exactsum=transport+local+inherited;
independent_prediction=B(a,a)+native Down_bias.
Only the local/independent predictions exclude the fourth input. The inherited
arm is an explanatory diagnostic and cannot be promoted as prediction.
Fullnative downstream execution remains active for everyarm. No frozenV9ports.

For each of4panels:4nativecorner captures,4directions×7arms=32forwards.
Total128forwards/1120sequence instances,0fits/backwards,600second watchdog.
Only through managedenqueue. Nativeweights/background/source generationcharged;
actualsaving0. No anchor, dose, rank, subset or response-order sweep.

A instrument: frozen rows/source/backend/checkpoint/primitive hashes; square
builder exact replay; CPU partition controls, including local term independent
of changing fourthcorner; native MLP weight formula replay; FP64partition
closure maxabs<=1e-9 andrelFrob<=1e-9; exactsum versusfull localwrite andfull
logits, local versus independent_prediction localwrite andfull logits,
identitynative, each maxabs<=1e-3 ANDrelFrob<=1e-5. Parent fullsource margin
replay after mapping row/direction/answer orientation atsame1e-3/1e-5bars.
Source masks preserve padding; finiteoutputs, exactcounts andrestoredhooks.
Instrument misses invalidate, not scientificnulls. Absolute denominators use
floor1e-8; live predicates below prevent vacuous prediction passes.

B totalcueeffect: candidate(local)-native versusfull-native, relativeerror<=.10
in BOTH centered50304-vocabulary logits ANDanswer1-minus-answer0margin,
EVERY16panel/direction cells.
C contextinteraction: candidate(local)-transport versusfull-transport, same
relativeerror<=.10 in BOTH frames/EVERYcell. This prevents transferring a mostly
context-independent cue effect from masquerading as predicting its interaction.
D live: full-native ANDfull-transport have norms>1e-8 inbothframes/everycell.

Report local normalized-input and MLP mixed-write partitions with signed
projections, norms and errors onvalidtokens. The inherited and transport arms
are diagnostics. Passing local reconstruction alone is not a causal circuit;
passing B alone withoutC cannot establish the proposed interaction computation.
Preserve allfailures andotherarmoutcomes without alternativewinner promotion.
No removal/composition ornewfamilyOOD claim from this local source screen.
