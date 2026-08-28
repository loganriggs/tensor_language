# Plain-language project update — 2026-08-28 17:30 UTC

## UPDATE AFTER 17:55 — a one-scalar repair is useful but not freely composable

The successful isolated L5/L6 repair was tested in a harder bottom-up setting: compile
layers from the bottom, retain a live suffix, and apply the measured scalar correction
either to the first live attention layer or to every live attention layer.

The result rejects a universal “just fix the first interface scalar” rule:

- with layers 0--3 compiled, correcting only the first live layer makes the result
  slightly worse, while correcting every live attention layer recovers about 10--12%
  of the full compiled-to-live accuracy gap;
- with layers 0--5 compiled, correcting the first live layer recovers about 13%, and
  correcting all live layers recovers about 11--12%;
- with only layer 0 compiled, correcting the first live layer improves recovery from
  roughly 37--40% to 59--63%, but correcting every live layer makes recovery negative.

The corrected depth curve is therefore not monotone. A scalar is computationally
cheap, but its causal usefulness depends on where the compiled/live boundary lies and
which other corrections are installed. This strengthens, rather than removes, the
need for the 68-action composition test and all-consumer measurements. A calibrated
simplicity frontier must fit corrections only on fit/validation data and evaluate the
complete corrected composition on held-out CE, OOD, and edits; it cannot declare a
transformation “free” merely because it has one parameter.

## UPDATE AFTER 17:52 — every action now has the correct physical call ledger

The older final wrapper claimed that original MLP0, MLP1, and MLP2 calls must all be
zero over the complete experiment. That is impossible for legitimate actions: every
`E` action must call exact MLP2, and `O/O` must call exact MLP0 and MLP1.

The replacement is a 68-entry ledger derived from the physical action plan. For each
action it fixes the total deployed, correction, and exact-original calls over all 48
four-row observational batches. For example:

- `RR/N` uses deployed MLP0/1/2, corrects MLP0/1, and calls no original early MLP;
- `RR/E` has the same MLP0/1 path and exactly 48 original MLP2 calls;
- `O/O/N` has exactly 48 original MLP0 and MLP1 calls and 48 deployed MLP2 calls;
- `O/O/E` has exactly 48 calls to each original early MLP.

A missing, extra, swapped, or globally-zero ledger now fails. This closes the 68
ordinary observational paths; the additional edited-response forwards still need
their own ledger when that backend is implemented. The complete suffix/observed suite
now passes **258/258 tests**. No final rows or scientific outcomes were opened.

## UPDATE AFTER 17:30 — scored rows and all baseline paths are now closed

The final-action identity previously bound the 256 input tokens but not the extra
target token used to score next-token CE. That was a real integrity gap: a changed
target could in principle keep the same model input. The identity now hashes the
complete 513-token role row as well as the inputs and rejects a target substitution
before any model forward.

All four early-layer baselines now have observed execution paths with exact call
ledgers:

| Early MLP0/1 | MLP2 | Meaning |
|---|---|---|
| deployed | deployed | `N/N/N` |
| deployed | exact | `N/N/E` |
| exact | deployed | `O/O/N` |
| exact | exact | `O/O/E` |

The two deployed-MLP2 actions report exact OON teacher KL. For O/O/N, the action is
itself the OON teacher, so the implementation computes KL from the actual logits
against themselves rather than inserting a made-up zero. Exact-MLP2 actions remain
CE-only. Hidden native calls on a deployed path fail immediately. The expanded
suffix/observed suite passes **257/257 tests** at that stage.

A concurrent diagnostic also simplified the attention-scale finding below. A single
scalar for the whole attention layer removes 98.4--99.7% of the L5 cliff and
99.5--100.6% of the L6 cliff, essentially matching nine separate head gains. The L5
scalar was fitted on one role and transfers to the other two. This says the immediate
failure is mostly one layer-wide scale error. It supports the runner's inexpensive
plan to measure one norm ratio for each of 18 consumers; per-head ratios are only
needed if a layer scalar fails. As before, this returns performance to the compiled
baseline near 13%, not to the live model near 39--42%.

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
| Native/deployed baseline execution paths implemented | 4/4 |

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

1. Route every materialized program action through the same full-row identity and
   close the per-action original-call ledger.
2. Add the frequency-bin, all-consumer-norm, and paired-edit reductions.
3. Run a fresh audit, then execute all 68 actions once on common final rows.
4. Decide from CE, model agreement, OOD bins, edit transport, and norm stability—not
   local MSE alone—whether any program is actually simpler in a useful sense.
5. If one passes, install it in the current whole-model replacement and measure how
   much of the real $+0.8976$-nat gap it removes.

The attention gain result will inform the consumer-norm diagnostic, but it does not
replace this plan: it explains one failure mode of an older context-free compiler,
whereas the final suffix experiment tests the coupled polynomial early program.

## Clarification: what the rank-1 result is useful for

The rank-1 result measured next-token accuracy, not agreement with bilin18's chosen
token. Its absolute top-1 values are 9.90%, 10.65%, and 10.07% on the three roles,
versus 12.88%, 13.49%, and 12.89% for rank 64 and 39.32%, 42.35%, and 38.88% for the
live model. Thus “retains 77--79%” means it retains that fraction of rank 64's
accuracy; it does not mean 77--79% agreement with bilin18.

Nevertheless, rank 1 can be a useful extraction if the question is “give me a tiny
rule that performs this behavior” rather than “give me a probabilistic twin of the
model.” We should keep separate frontiers for:

- task or decision-rule extraction: task accuracy and agreement on the relevant
  behavior;
- functional faithfulness: agreement with bilin18's top token and full-distribution
  KL/CE;
- causal editability: prediction of finite edits and selective removal with low
  collateral damage; and
- mechanistic faithfulness: preserving the same internal interfaces or circuits.

Success on the first does not imply the others. In particular, the same in-distribution
top-1 can hide different confidence, failure cases, OOD behavior, or edit response.
The rank-1/rank-64 curve still lacks a matched live-model-agreement and OOD/edit sweep;
that is the measurement needed before calling rank 1 an extracted model policy.

## Cheap affine corrections should be part of the simplicity grammar

A global scalar, scalar bias, or foldable affine change can be negligible compared
with a matrix or token table. If it can be absorbed into an adjacent weight or bias
without changing the function, it should have zero incremental executable cost and
be treated as gauge/canonicalization rather than a new circuit. If it changes the
function, it is not mathematically free, but one scalar or one short bias vector is
still a tiny explicit correction whose description and runtime cost should be counted.

The L5/L6 result is a concrete example: one layer scalar repairs the immediate
interface cliff. This motivates a **calibrated simplicity frontier**: for every
compressed program, report both its raw result and its result after a fixed small
correction family fitted only on fit/validation data. Candidate correction families
should be nested and priced explicitly—for example one scalar, scalar plus bias,
diagonal gain, then low-rank affine correction. Final or OOD rows must never choose
the correction. A cheap calibration that transfers on CE, OOD, and edits is useful
functional compression even if its internal implementation differs from bilin18.

## What the 68 actions actually are

They are **68 controlled replacement configurations**, not 68 discovered circuits or
68 independent neural pathways. There are 34 MLP0/MLP1 configurations, each run with
two MLP2 backgrounds:

- `N`: the deployed simplified MLP2;
- `E`: the exact original MLP2.

The 34 configurations include the inherited program, locally fitted and jointly
suffix-fitted programs, MLP0/MLP1 hybrid removals, an explicit cross-layer transport
term, zero and 20 false-parent transport controls, shuffled controls, native/deployed
baselines, and a fit-mean control. Running all of them on identical rows lets us ask
whether a gain came from MLP0, MLP1, their co-adaptation, the explicit transport term,
or compensation by exact MLP2. It gives finer causal control over this early
MLP0--MLP2 interface, but it is not yet a fine-grained decomposition of the entire
model.
