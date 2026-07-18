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

## Verdict

The paradox on one sequence: zeroing the match head costs little; replacing
its carried content with CLEAN token identity, or low-rank-filtering its
output, IMPROVES copying — the head's selection signal is right and its
carriage is noisy enough that the model under-weights it. H7's zero is the
catastrophic control; a random head is the null control. Caveats: one
synthetic sequence (cherry-picked format by design); statistics for these
effects at corpus scale are in results/12 and h5_undercash.json.