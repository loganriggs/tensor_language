# September 16 update: the embedding number coordinate is causally used

## Result

The frozen grammatical-number coordinate in the checkpoint token embeddings is
not merely decodable. Removing that coordinate at a fresh subject token
selectively damages the model's number-agreement margin.

The new authority contains 16 noun pairs absent from both decoder panels: eight
irregular and eight regular. Both forms occur in two new prepositional templates
with an opposite-number attractor, for 64 fixed prompts. Native agreement and
the frozen decoder were correct on all 64 rows.

For normalized embedding `x`, unit decoder direction `a`, and centered threshold
`t`, the intervention replaces `x` with the same-norm point in the plane of `a`
and `x`'s orthogonal component whose decoder score is exactly zero. This is a
label-free removal: the same formula is used for singular and plural subjects.
Each of 32 controls moves the same subject embedding by exactly the same distance
on its constant-norm sphere along a frozen random tangent direction.

All registered predictions passed:

- target damage RMS: `2.94985`, or `.5340` of native-margin RMS;
- target damage was positive on all 64 rows and in every number, template, and
  regularity cell;
- median random-control damage RMS: `.07316`;
- strongest random-control damage RMS: `.20182`;
- target/random-median ratio: `40.32`;
- target/strongest-random ratio: `14.62`;
- unrelated `can`/`will` control RMS: `.0593` of target damage;
- target removal flipped 6 of 64 answers; the remaining 58 generally retained
  the right answer with a smaller margin, which is expected for projection to
  the decision boundary rather than an opposite-number swap.

The implementation audit is unusually strong. A separately coded manual
input-state forward reproduced the observed facade with zero measured logit
error. The target decoder residual was `2.89e-15`; target norm error was
`2.09e-16`; and the worst random distance/norm mismatch was `5.77e-14`.
Therefore the result is not explained by changing embedding norm, unequal random
perturbations, a stale token position, direction-label leakage, or a replay bug.

[Result](../../SUBJECT_NUMBER_EMBEDDING_COORDINATE_REMOVAL_V1_RESULT.json) ·
[preregistration](../../SUBJECT_NUMBER_EMBEDDING_COORDINATE_REMOVAL_V1_PREREGISTRATION.md)

## What this adds to the four-trait graph

The subject-number line now has evidence for each desired property, but not yet
as one fully native end-to-end circuit:

| Trait | Current evidence | Boundary |
|---|---|---|
| OOD prediction | perfect decoder transfer to two successive disjoint vocabularies | checkpoint embedding coordinate |
| Extraction | executable token-to-direction-to-fixed-rank-one-write program | cardinality `4` is still an external context port |
| Removal | fresh same-norm embedding-coordinate removal dominates 32 matched random edits | broad unrelated-task preservation is not yet tested |
| Composition/reuse | two generated fixed-axis writes compose with relative error `1.07e-4` | one weak direction missed the registered absolute liveness floor by 1.1% |

This is substantially closer to a sparse graph than a donor patch. The input
node is a model-weight coordinate, causally necessary on fresh text, and the
output operation is a reusable rank-one write. The missing edge is direct
mediation: we have not shown that removing the embedding coordinate changes the
specific frozen L11H3 scalar and that restoring that scalar rescues the loss.

## DCT audit

The workspace already contains the literal DCT checks proposed in the briefing;
they should not be conflated with this embedding result.

- At MLP17, weight-derived and autodiff Hessians agree to `2.90e-15`, are
  background invariant, and produce identical rank-four factors. Rank-four
  reconstruction error is nevertheless `.8998`. The bridge is correct; low rank
  does not follow. [Result](../../MLP17_CAUSAL_HESSIAN_IDENTITY_V1_RESULT.json)
- Raw weight-only MLP9 DCT is numerically lawful, but its top-eight output span
  covers only `.0711` of the behavior-selected reader, below a matched random
  median `.0794`. Planted recovery and exact replay pass, so this is a valid
  reader-discovery null rather than a solver bug.
  [Result](../../MLP9_DCT_UNSUPERVISED_READER_V1_RESULT.json)
- Keeping native context and RMS open yields a contextual rank-16 MLP9 primitive
  with `.00493` fresh response error and `.00430` suffix-installation error.
  [Result](../../MLP9_CONTEXTUAL_DCT_CAUSAL_FRESH_V1_RESULT.json)
- That generic contextual basis does not recover the semantic equality/copy
  reader: copy-response coverage is only `.1764`, and the preregistered semantic
  gates fail. [Result](../../MLP9_CONTEXTUAL_DCT_EQUALITY_RESPONSE_DISCOVERY_V1_RESULT.json)

The resulting methodological conclusion is narrower and stronger than “DCT
works” or “DCT fails”: the causal-Hessian bridge is exact, and open-context DCT
can produce reusable local primitives, but generic leading factors are not
automatic semantic readers. Behavioral readers, selective removal, and OOD
causal installation remain necessary certification stages.

## Embedding-to-L11H3 mediation: small, rank one, and OOD-stable

We ran that discriminating experiment in two stages.

The opened removal panel first produced a registered null for **substantial**
mediation. Restoring the complete native L11H3 change recovered `.09554` of the
embedding-removal damage, just below the preregistered `.10` floor. The frozen
rank-one axis recovered `.09049`. That null is preserved.

The internal structure was striking: the rank-one rescue reproduced the full
head rescue with cosine `.99908`, relative L2 `.06155`, and norm ratio `.95484`.
Matched random-axis rescues had median effect RMS `.00754`, versus `.30115` for
the frozen axis. This suggested a different, explicitly smaller hypothesis:
L11H3 carries a stable roughly 5--15% path rather than the main number path.

We froze that hypothesis on a fourth disjoint vocabulary panel: 16 new noun
pairs, 64 untouched `Under`/`Above` prompts. Every registered prediction passed:

- embedding-coordinate removal again damaged all 64 rows, RMS `3.37997`;
- full L11H3 restoration recovered `.11844` of that damage, cosine `.92689`,
  with positive rescue on `.9375` of rows;
- the frozen rank-one axis recovered `.11256` of total damage;
- rank-one versus full-head rescue: cosine `.99957`, relative L2 `.05236`, norm
  ratio `.95627`;
- rank-one rescue RMS `.41301`, versus random median `.00887`; every matched
  random rescue had at least `.9786` relative error to the full-head effect;
- decoder and native behavior accuracy were `1.0` in every registered cell,
  and state/capture geometry closed at machine precision.

Thus the native edge is real but quantitatively small:

`embedding number coordinate -> about 11% of agreement damage through the frozen L11H3 rank-one axis`.

The remaining roughly 89% travels through other heads, layers, or distributed
routes. The axis result should not be inflated into a claim that L11H3 is the
whole number circuit.

[Opened discovery](../../SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_DISCOVERY_V1_RESULT.json) ·
[fresh confirmation](../../SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_FRESH_V1_RESULT.json) ·
[fresh preregistration](../../SUBJECT_NUMBER_EMBEDDING_TO_L11H3_MEDIATION_FRESH_V1_PREREGISTRATION.md)

## Donor-free scalar attempt: valid null

The mediation rescue still reads the native base L11H3 value as a donor port.
We tested whether its scalar change could instead be generated by a small
label-free polynomial of the frozen embedding score and same-norm removal
distance.

Four forms were preregistered, from affine score through signed quadratic and a
score/distance interaction. The capture instrument independently reproduced the
previous mean axis-change norm within `3.68e-6`. The selected signed quadratic
was nevertheless inadequate:

- leave-pair-out relative L2 `.56562`, cosine `.82518`, sign `.8125`;
- leave-template-out relative L2 `.55838`, cosine `.83051`, sign `.79688`;
- permutation-median error `.71080`, only `.14518` worse than the candidate,
  below the registered `.20` advantage.

Every candidate remained around `.566--.597` worst-fold error. This is not a
solver or head-slice failure; the target scalar itself was reproduced exactly at
the aggregate audit boundary.

A diagnostic performed only after the null clarifies the missing ports. Giving
an oracle the mean scalar for each subject token reduces error to `.22065` with
cosine `.97535`, so lexical information beyond the one-dimensional number score
matters. Even that token oracle cannot remove template dependence: the two
templates for a fixed token differ by RMS `46.74`, versus target scalar RMS
`105.91`. The DCT briefing's open-context warning applies directly—averaging
this edge into a token-only scalar discards a material context mode.

[Scalar discovery null](../../SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_RESULT.json) ·
[preregistration](../../SUBJECT_NUMBER_EMBEDDING_TO_L11H3_SCALAR_DISCOVERY_V1_PREREGISTRATION.md)

## Updated boundary and next experiment

The honest sparse graph boundary is now:

`causal embedding-number node -> unresolved lexical/context transform -> small rank-one L11H3 edge`.

The endpoints and the edge's OOD causal effect are verified, but the transform
is not extracted. The next useful decomposition should keep subject identity and
prompt context as separate native slots and measure their Möbius interaction on
the L11H3 scalar. A compact context-conditioned function can then be frozen and
tested on a fifth vocabulary/template panel. A larger token-only polynomial or
post-hoc template indicator would merely hide the failed port accounting and
should not be treated as progress.
