# Lexical identity, inflection and composed commands

10 September2026. Original handoff/pilot, individual and structured unembedding
paths. Existing all36 gerund e ports and first-value channel are fixed. Dossiers
and v173/v174, gerund scalar network/fresh tests, first-value source factorial
and old four-corner controls checked. This asks semantic reuse and composition,
not rediscovery of a quadratic expansion or selection of a better head slice.

Use opened A1/A2 sixteen base/donor-cue rows; cyclic next row changes only the
primed bare verb at position3. Native grid: B original lemma/bare cue, X cyclic
lemma/bare cue, Y original lemma/ing cue, Z cyclic lemma/ing cue. Four output
readers are [original bare, original ing, cyclic bare, cyclic ing].

Lexical command L changes ONLY position3 of the first-value cache returned after
attention0 to the cyclic lemma's value. Attention0 output and all other cached
positions remain native; downstream routing/values/MLPs recompute. Apply L to B
and Y. Form command F prescribes all36 final-position output e scalars from Y,
using the existing live scalar replacement; apply the SAME Y scalar command to
B and X. This deliberately tests reuse across lemmas without context adapters.
Joint J applies L+F to B, with those same independent donor commands. Own-cache
L+base-scalars F is a no-op. Four native+four singleton+joint+noop=10 bodies/frame.
G uses base/cyclic/value-only to obtain previously missing runs/run foil scores.
Total23forwards368seq, <=900sec. All native weights remain; no fitting.

Readout decomposition: H rows [1,1,1,1],[-1,-1,1,1],[-1,1,-1,1],
[1,-1,-1,1]. For stacked token weights U, components C=H U/4 give grand,
lexical, form and interaction readers; U=H^T C exactly. Fold each to MLP17
products as C Down17, without materializing dense quadratic tensors. Final
RMS and softcap stay explicit: linear weight decomposition is before softcap,
while H applied to final scores describes final-score interaction. Neither
basis change alone identifies a circuit. Save readers, product coefficients,
final native/edited states and selected scores for CPU reuse.

For scores s=[s0bare,s0ing,s1bare,s1ing], lexical margins are
[s1bare-s0bare,s1ing-s0ing]; form margins are [s0ing-s0bare,s1ing-s1bare].
L recovery uses the lexical margin at the grammatical form appropriate to its
cue, relative to B->X or Y->Z. F recovery uses the form margin at the actual
input lemma, relative to B->Y or X->Z. All row denominators must exceed1e-6;
no filtering. Preservation uses aggregate relativeL2 on BOTH margins: L's form
movement relative to native B->Y (for L(B)) or X->Z (for L(Y)); F's lexical
movement relative to native B->X (for F(B)) or Y->Z (for F(X)). Each compared
margin vector has two entries per row. Require reference norms>1e-4.

Predictions:
A instrument: exact23/368; shared engine state/readout/scalar checks; no-op
logits maxabs<=1e-3,rel<=1e-5; H inverse and folded-reader reconstruction
maxabs<=1e-10; prior bare-cue value-only lexical recovery replay maxabs<=1e-5;
G old value-only meanabsCE replay<=1e-6. Finite outputs, single changed token3.
B lexical reuse/selectivity: A, all native four-token choices correct in both
frames, ALL L recovery denominators positive; mean L recovery>=.80 in BOTH
cues of BOTH frames; form-preservation error<=.10 in all four cases.
C form reuse/selectivity: same native capability; ALL F denominators positive;
mean F recovery>=.80 on BOTH lemmas of BOTH frames; lexical-preservation
error<=.10 in all four cases. Same fixed Y command, no lemma-dependent repair.
D joint independent-effect hypothesis: A and native capability, all16 J outputs
prefer cyclic ing among four tokens in BOTH frames; centered four-token effect
error against B->Z<=.20 and centered FULL-vocabulary additive prediction error
||(J-B)-(L(B)-B)-(F(B)-B)||/||J-B||<=.10 in BOTH frames. Failure rejects this
independent-effect approximation, not composition programs with explicit coupling.
E G agreement log-odds preservation: A and mean abs change in runs-minus-run
score under L<=.10 nats. This new margin question does not replace/reclassify
the previously failed .10 correct-token CE preservation test; report both,
including natural cyclic-context effects and shared two-token score shift.

CPU continuation: paired intervals for all registered transfers/preservation
errors, interaction and G margin/CE; list failed native endpoints. Decompose
selected score changes into H coordinates on all rows, no new selection.
A positive would remain a conditional screen requiring independent consumers,
fresh/OOD cases, removal and reduced structural description before promotion.
