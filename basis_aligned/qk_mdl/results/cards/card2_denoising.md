# Circuit card 2: the denoising paradox

**One repeated sequence; the match head attends correctly, removal barely
hurts, cleaning HELPS.** (WW-2/WW-6/WW-7 at single-sequence resolution.)

## The sequence

` lantern crossing violet harbor thistle meadow copper sparrow lantern crossing violet harbor thistle meadow copper sparrow lantern crossing violet`  (8 rare words, repeated; predictions scored on the second pass)

## H5 attends correctly

At position 14 (` harbor` second occurrence), H5's top
attention targets are positions [4, 5, 8] = ["' th'", "'istle'", "' copper'"]
— position 4 (the continuation of the first occurrence) IS among them.

## Mean logP(correct next token) over the repeated half

| arm | mean logP | Δ vs live |
|---|---|---|
| live model | -0.722 | +0.000 |
| H5 zeroed | -0.986 | -0.264 |
| H5 content cleaned (cond-mean identity) | -0.886 | -0.163 |
| H5 output rank-2 filtered | -0.939 | -0.217 |
| H7 zeroed (contrast) | -3.496 | -2.774 |
| random head zeroed (L5.H3) | -0.715 | +0.007 |

## Verdict — the paradox has a boundary (and this card found it)

The pre-registered expectation (from corpus-scale WW-6/7 on UNIFORM-RANDOM repeats)
was that cleaning H5's content would IMPROVE copying. On this natural-word sequence
the opposite holds: cleaning costs −0.163 and rank-2 filtering −0.217, while the
controls behave (H7 zero −2.774 catastrophic; random head +0.007 null; H5 zero −0.264
modest). Resolution: **H5 carries context-mixed identity.** On degenerate random-token
repeats the context component is noise — averaging it away helps. On real text it is
signal — averaging it away hurts. "The model under-cashes its induction head" (WW-7)
is therefore a statement about *degenerate contexts*; on natural text the head's
carriage is better than its cond-mean.

Methodological note: this is what cards are for — a single legible example acted as a
regression test on a corpus-level claim and sharpened it. Caveats: one sequence; the
corpus-scale version of the natural-text arm (clean H5 on pile repeats of real
phrases) is the obvious follow-up.