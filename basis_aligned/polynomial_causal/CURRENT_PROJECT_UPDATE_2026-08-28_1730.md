# Plain-language project update — 2026-08-28 17:30 UTC

## UPDATE — what changed since the 17:09 explanation

Two concrete things changed.

### 1. The final MLP0/MLP1 experiment is now much harder to run incorrectly

The experiment has 34 early-program choices, each tested with either simplified or
exact MLP2, for 68 total actions. We can now construct the actual program for every
action, including mixed choices such as “take MLP0 from program R but MLP1 from
program L.” Each final batch is bound to the intended program, tokens, support,
background model, and row order by hashes. If any of those are silently exchanged,
validation fails before the result is scored.

The “new fit mean” control is now frozen too. It is a constant-output program whose
two 64-dimensional outputs are the average MLP0 and MLP1 coordinates measured during
the fitting pass. The mean is reconstructed from frozen sums and counts, stored in
the canonical program bank, and cannot be recomputed after final data are opened.

The complete suffix/observed CPU test suite passes **247/247 tests**. This is
infrastructure progress, not a scientific win: no final 68-action result has been
opened yet.

### 2. We learned why one partial compilation collapses at attention layer 5

An older context-free early replacement causes a severe failure when attention layer
5 is restored to the live model. Attention has nine heads. Head 7 produces by far the
largest absolute bad output, but this turned out not to be a one-head circuit failure.

The compiled-to-live output-size ratios for the nine heads were approximately

\[
(0.47,\;8.68,\;240.76,\;0.97,\;34.02,\;36.86,\;83.93,\;158.91,\;6.61).
\]

Head 7 carries about 85% of the excess squared norm because it was already an unusually
large head in the live model. But seven of the nine heads grow by more than sixfold,
so the underlying mismatch is layer-wide.

We tested a simple diagnostic repair: divide each head's output by its measured
compiled-to-live size ratio.

| Repair in the partial model | Fraction of the layer-5 accuracy cliff removed |
|---|---:|
| Correct only head 7 | 14.7–15.0% |
| Correct all nine heads | 98.8–99.6% |
| Apply the head-7 correction when layer 5 is also replaced | 0.0 percentage points |

The nine ratios were measured on one data role and then reused unchanged on the other
two roles, where they still removed 98.8% and 99.2% of the cliff. So this is not merely
a separate nine-number fit on each evaluation slice.

The important conclusion is:

> The immediate layer-5 cliff is almost entirely a vector of per-head gain errors,
> rather than one special head acting alone. No changed attention pattern is needed
> to remove this measured top-1 failure.

“Gain” here simply means output scale. The direction written by head 7 remains almost
unchanged: its live and compiled mean vectors have cosine similarity 0.9990. It also
remains unusually constant across positions. The same vector is being amplified,
not redirected by a dispersed attention pattern.

This also gives a useful warning about interpretation: head 7 owns about 85% of the
excess output norm but only about 15% of the accuracy damage that can be repaired by
head scaling. “Largest activation” is therefore not the same as “largest causal
contribution to the prediction failure.”

There are two limits on this result. First, the nine repair constants were diagnosed
from the observed size mismatch; this is not yet a learned or principled compiler.
Second, the repair returns the damaged partial configuration to the accuracy of the
fully compiled baseline (roughly 13% top-1 on these slices), not to the 39–42% live
model. It removes this particular interface catastrophe; it does not explain the
remaining model gap. One preregistered endpoint control was specified against a
different full-rank program, so it reported failure even though this rank-64 baseline
reproduces its correct rank-64 parent to four decimal places. That was a control-design
error rather than contradictory data. The gain repair is still discovery evidence,
not an admitted compiler, because its constants were obtained from the measured
mismatch and the experiment scores top-1 rather than the final CE/OOD/edit criteria.

## Current best understanding

### MLP0

MLP0 writes a continuous code of about 64 useful dimensions. It is not best described
as assigning one hard class to every token. Two tokens can share some lexical
properties and still occupy different points in the code; context can continuously
move those points. We can compress and execute this code, but we cannot yet give most
coordinates stable human-readable meanings.

### MLP1 and MLP2

MLP1 reads the state produced by MLP0. MLP2 compensates for some errors left by the
first two layers. Consequently, separately good approximations do not necessarily
compose: replacing MLP0 changes the inputs on which an independently fitted MLP1 was
accurate, and MLP2 may either repair or amplify the joint error.

This is why the main experiment fits and tests the early layers as a coupled program
and explicitly compares simplified MLP2 against exact MLP2.

### Attention interface

The newest result makes the interface problem more concrete. A replacement can
produce states that look acceptable locally but have the wrong scale in several
directions that a later attention layer strongly amplifies. Ordinary output MSE can
miss this. We therefore need both end-to-end cross-entropy and per-consumer scale
checks at every later layer.

## What we can honestly claim today

| Claim | Status |
|---|---:|
| Structural surrogate exists for each module | 36/36 |
| Whole-program storage certified removable for its registered consequence | 5.3481% |
| Older behavior assigned human-readable labels | 32.1% $\pm$ 6.4% |
| Strict named causal CE headroom recovered | 10.923% |
| Current replacement's $+0.8976$ CE gap recovered by a newly admitted package | 0% |
| Final early-MLP action definitions physically constructed and identity-bound | 68/68 |
| Final early-MLP scientific actions evaluated | 0/68 |

The last two rows are the key distinction. The experimental machinery is becoming
trustworthy, but it has not yet promoted a better early program into the model.

## What is blocking the next scientific result

There is no FineWeb, cache, checkpoint, GPU, or `rspd` blocker. The remaining blocker
is final-runner completeness. Before opening the final rows, it still must correctly:

1. run and verify the native, deployed, and exact baseline combinations;
2. aggregate all actions on exactly the same token rows;
3. report nine predeclared token-frequency bins;
4. measure output norms at all 18 later consumers;
5. execute each finite edit together with its unedited control; and
6. pass an independent audit of the complete capability bundle.

This is slower than fitting another small matrix because the experiment is designed
to answer a causal composition question without leaking final rows or accidentally
giving one arm a different background model.

## Immediate plan

1. Finish the observed baseline backend and its call-count/identity receipts.
2. Add the frequency-bin, all-consumer-norm, and paired-edit reductions.
3. Run a fresh audit, then execute all 68 actions once on common final rows.
4. Decide from CE, model agreement, OOD bins, edit transport, and norm stability—not
   local MSE alone—whether any program is actually simpler in a useful sense.
5. If one passes, install it in the current whole-model replacement and measure how
   much of the real $+0.8976$-nat gap it removes.

The attention gain result will inform the consumer-norm diagnostic, but it does not
replace this plan: it explains one failure mode of an older context-free compiler,
whereas the final suffix experiment tests the coupled polynomial early program.
