# Research update: DCT modes move inside M4, but expose a required interaction

We moved the equality graph boundary inside the largest retained native write,
M4, and applied both sides of the DCT briefing's advice: use a weight/moment
decomposition to propose coordinates, but let downstream interventions decide;
and treat failures as possible instrument bugs before scientific nulls.

## Native product atoms: exact but not sparse

M4 has 4,608 exact atoms `D[:,i](L_i z)(R_i z)`.  We ranked them on natural
text using product RMS contracted with the four already-frozen L5H5 Q/K
readers.  Full native-order replay was exactly zero-error, so the instrument is
lawful.  However, the first support satisfying the frozen score gate retained
4,096 atoms.  It transferred strongly—code parent-score error/cosine
`.02914/.99958`, recovery `.89166`, and `.00108` nat noncopy damage—and a
one-token equal-norm roll reduced recovery to `.71577`.  Thus the M4 contribution
is causal and direction-specific, but sparse native product coordinates are the
wrong representation.

## Contracted DCT modes: real compression, not yet compact

We next formed the weight-only contracted operator `W D`, where `W` stacks the
four L5H5 Q/K readers, and used its right singular vectors as dense bilinear
product modes.  The first float32 SVD receipt was invalid: its operator replay
error was `1.26e-4`, beyond the preregistered bridge tolerance.  A precision-only
correction certified the same construction in float64 at `2.06e-13`.

The corrected basis needs rank 256 rather than 4,096 native atoms.  Natural
parent-score error/cosine is `.04462/.99906`; frozen code is `.03818/.99927`;
recovery is `.89159`; noncopy damage is `.00155` nat; and an equal-norm roll
drops recovery to `.71098`.  This is a useful zero-learned-parameter executable
boundary and a 16x reduction in scalar writer modes relative to the selected
atom basis.  It does not pass the preregistered compact-rank ceiling of 64 and
still evaluates all 4,608 native L/R products.

## Composition is the remaining structural obstacle

An arbitrary alternating 128/128 split had `.49047` component-relative score
composition error.  We did not accept that split as definitive.  A prospective
natural-only most-additive search tested prefix/remainder cuts at
8/248, 16/240, 32/224, 64/192, and 128/128.  Every candidate carried both
components; the best was the contiguous 128/128 split.  It improved frozen code
score composition to `.13323`, stable across halves at `.13302/.13346`, but
missed the `.10` gate.  More importantly, direct NLL composition remained
`.42108` overall, `.378–.461` across copy subtypes, and `.401/.444` across
halves.  The joint node remains causal (`.89159` recovery) and selective
(`.00155` nat noncopy damage), so this is not a dead component—it is a genuine
interaction.

## Updated sparse-graph handoff

The useful boundary is now:

`M4 normalized input -> 256 dense bilinear scalar modes -> 256 writer directions -> L5H5 score`.

The next graph should not force those modes into two additive nodes.  It should
make the Möbius cross-difference between the selected 128-mode child and
128-mode remainder an explicit third node, then test removal and OOD reuse of
that interaction.  Only after that should each dense quadratic reader be
factorized to reduce the still-native 4,608 product evaluations.  This keeps
the overall four-trait goal honest: the current boundary has OOD prediction,
execution, and causal/selective use, but composition requires a named
interaction edge rather than wishful additivity.

## Explicit interaction node: the sparse graph now closes

We implemented that handoff prospectively.  For bias-only baseline `S0`, the
rank-128 child score `Sc`, complementary rank-128 remainder score `Sr`, and
rank-256 joint `Sj`, the graph stores `Sc-S0`, `Sr-S0`, and the cross-difference
`Sj-Sc-Sr+S0` as three typed effects above the baseline.

All registered gates passed.  The graph closes at exactly zero measured error
on both natural and code roles and reproduces direct-joint downstream NLL and
recovery exactly.  Direct code parent-score error/cosine remains
`.03818/.99927`, and direct/composed recovery is `.89159`.

The interaction is causally necessary even though its aggregate recovery partly
cancels across tokens.  Removing only it changes copy-token NLL by `.30096` of
the joint-vs-baseline effect.  Every subtype and half lies between `.26961` and
`.32079`; incremental noncopy mean change is only `-.000345` nat.  A one-query
roll preserves full interaction norm within `1.17e-7` but its behavioral effect
has only `.52158` cosine with true interaction removal.  This rejects the idea
that any equal-norm score perturbation would produce the same result.

At this boundary we now have all four desired traits: the frozen graph predicts
code OOD, runs as an extracted zero-parameter package, supports selective
interaction removal, and composes/replays exactly as a reusable Möbius graph.
The price caveat matters: the interaction currently consumes four score
evaluations (`S0,Sc,Sr,Sj`).  The next compression problem is to fold that
cross-difference into a direct kernel without losing its removal and transfer
certificates, then reduce each dense mode's still-native product cost.

## Direct-kernel compression: score success, behavioral null

We tested that folding target rather than assuming that score fidelity implies
intervention fidelity.  Sharing the projected baseline, child delta, and
remainder delta reduces the four-corner executor from 16 to 12 Q/K projections.
It preserves the joint score well (`.00376` relative code error), but the
interaction error is `.10063`, just outside the frozen `.10` gate.  Reassigning
which of the four corners is derived arithmetically does not rescue it
(`.10069`).  Adding one selected K2 correction costs one projection, only
`.00225` of the corrected child-port norm, and brings code interaction error to
`.09956`; all preregistered score-space gates then pass with 13 projections.

The behavioral red-team nevertheless rejects that 13-projection replacement.
Its aggregate recovery (`.89071`) nearly matches the exact graph (`.89159`),
interaction-removal magnitude remains `.306`, and noncopy change is only
`.000320` nat.  But tokenwise composed replay error is `.258`, while the
approximate and exact interaction-removal vectors have only `.294` cosine and
`1.198` relative error.  This is a genuine behavioral mismatch hidden by good
global score geometry and aggregate recovery, not evidence that the exact
interaction is unnecessary.  The 16-projection graph remains authoritative;
future folding must optimize and certify the downstream removal vector, not
only global score L2.
