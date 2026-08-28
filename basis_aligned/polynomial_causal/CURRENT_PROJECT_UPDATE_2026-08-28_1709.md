# Current project update — 2026-08-28 17:09 UTC

## UPDATE AFTER 17:09 — final programs are now assembled and identity-bound

The action list described below is no longer only a naming contract. A sealed program
source bank and materializer can now construct the intended physical program for each
arm. In particular, mixed arms such as `R0-L1` take MLP0's affine map from R and
MLP1's affine map from L; they cannot silently fall back to the joint R program. True
transport, zero transport, and all 20 false-parent cross maps remain distinct by both
component hash and complete program hash. Native and deployed baselines are explicitly
program-free.

Each four-row final batch also receives a final-only identity binding the semantic
action, the materialized component hashes, program-bank identity, inherited snapshot,
common support, exact tokens, and canonical row order. Substituting another hybrid,
background, program, or row order invalidates the identity before scoring. The full
suffix/observed CPU suite now passes 247/247 tests.

This work initially found one additional missing artifact: the preregistration requires
a deterministic “new fit mean” program, but the canonical program-bank schema did not
serialize it. That gap is now fixed. The initial fit-label moment pass replays each mean
from its coordinate sum, constructs a zero-weight program with the two 64-dimensional
means as biases, and freezes its moment hash and tensors into the canonical bank. The
bank validator reconstructs the constant program, and the final source bank is minted
only from the validated bank plus inherited Q. The final runner therefore cannot
recompute or invent a mean after final rows are visible.

The attention explosion was also localized more carefully. L5H7 carries about 85% of
L5's excess squared output norm and grows from 6,657.8 to 1,057,986.8, confirming that
the already-known constant-bias/outlier-dimension head dominates the absolute failure.
However, this is not a one-head-only instability: L5H2 has the largest relative growth
(about 241x), and the layer-wide median is about 34x. At L6 the dominant excess head is
H1, not H7. A follow-up refuted the proposed softmax-pattern-dispersal mechanism: H7's
direction stays almost identical (cosine 0.9990) and remains unusually constant across
positions; the same fixed vector is simply amplified about 159x. The gain falsifier
has now completed. Correcting only H7 recovers about 15% of the accuracy cliff, while
correcting the gains of all nine heads recovers 98.8--99.6%. The same H7 correction is
exactly inert when L5 itself is substituted. Thus the immediate cliff is a layer-wide
vector of head-gain errors, not a one-head mechanism. This only restores the fully
compiled baseline, not the live model. A reported endpoint-control failure was a
mis-specified comparison to a different full-rank program; the run reproduces its
correct rank-64 parent to four decimals. The repair remains discovery-only because its
gains were diagnosed from the mismatch and it has not passed final CE/OOD/edit gates.
The final all-consumer norm measurements remain necessary.

## EARLIER UPDATE SINCE THE LAST EXPLANATION

There are four concrete updates.

1. **A partial compiled model becomes measurably closer to bilin18 as more of the
   exact later network is retained.** Its prediction agreement with bilin18 rises
   from about 22.4% to 64.2%, while its probability-distribution error falls from
   about 3.04 to 1.04 nats. This means the compiled program increasingly reproduces
   bilin18's mistakes as well as its correct answers; it is not merely becoming a
   better generic language model.

2. **Cross-entropy is not hiding a separate success.** Across 12 measurements, the
   compiled program's extra cross-entropy relative to bilin18 and its KL divergence
   from bilin18 agree within 0.7%. Cross-entropy measures how much probability the
   model assigns to the observed next token. KL measures how different the complete
   predicted probability distributions are. Their near equality here says that
   reducing the cross-entropy gap is essentially the same as making this program
   functionally closer to bilin18.

3. **The actual observed-model path now supports fitted early programs with either
   simplified or exact MLP2.** MLP0 and MLP1 can be replaced by one of the fitted
   programs while MLP2 is either kept as the deployed simplified component (`N`) or
   restored exactly (`E`). The boundary counts the literal module calls and releases
   only the measurements we registered. It does not release hidden states or logits.

4. **The 34 final experimental arms now have one explicit physical action plan.**
   Some arms are not simply “run program R.” For example, `R0-L1` means use MLP0's
   part from the jointly suffix-fitted R program and MLP1's part from the locally
   fitted L program. `L0-R1` reverses that choice. The new action-plan compiler fixes
   all such combinations, the true/zero/false cross maps, shuffled controls, and
   native/deployed baselines. Combining each arm with simplified or exact MLP2 gives
   68 fixed actions. This prevents the final runner from silently executing the wrong
   composition while reporting the right arm name.

No final scientific rows have been evaluated yet, so items 3 and 4 are execution
progress, not evidence that the new early program works.

## Current best understanding

MLP0 is well described as producing a small continuous code rather than assigning
each token to one hard cluster. The useful output subspace has approximately 64
dimensions. Tokens can share lexical structure while remaining continuously distinct,
and context can refine a token's location in the same code space. We still do not
have a reliable human-readable decomposition of those 64 coordinates.

MLP1 reads the state created by MLP0 and writes another low-dimensional component.
The best evidence says MLP0 and MLP1 must be treated jointly: changing MLP0 changes
the input distribution seen by MLP1. MLP2 then compensates for some upstream errors.
This is why three independently accurate replacements need not compose into an
accurate model.

The first three MLPs are the largest known early compositional problem. They account
for 0.7277 of 0.8727 held-out global replacement-error nats and 1.0776 of 1.176 nats
on novel rare tokens. Interactions account for roughly 43--64% of measured effects.

## How much of the model is explained?

There is no single honest percentage because the available percentages measure
different things.

| Question | Current answer |
|---|---:|
| Do we have executable structural surrogates for every module? | 36/36 modules |
| How much whole-program storage is certified removable under its registered consequence? | 5.3481% |
| How much behavior has older human-readable labels? | 32.1% $\pm$ 6.4% |
| How much strict named causal cross-entropy headroom is recovered? | 10.923% |
| How much of the current replacement's $+0.8976$ CE gap is recovered by a newly admitted package? | 0% |

The last number is deliberately strict. We have promising local and controlled
interfaces, but no new early-MLP package has yet passed the complete final suffix and
whole-model tests.

## Important confusing results

### Extreme compression can look much better under top-1 than under CE

The exact rank theorem makes the cheapest token-table program only 0.485 million
stored values. It retains 77--79% of the rank-64 program's top-1 accuracy, but loses
1.35--1.39 CE nats. CE penalizes this truncation about four times more strongly than
top-1 does when both are measured as lost progress toward bilin18. Therefore “almost
as accurate” does not imply “almost the same computation.”

### Context-free replacements can damage later consumers catastrophically

One context-free early replacement made attention layers 5 and 6 emit outputs 153 and
74 times their normal size. Matching the replacement's output norm did not fix it.
The polynomial program retains context and may avoid this, but the result explains why
we now require a consumer-norm measurement at every layer.

### A live suffix helps, but does not solve the early code

Retaining more exact later layers steadily improves the partial program. That tells us
the early approximation carries useful information which the live suffix can use. It
also tells us that the remaining early error is still large: the best tested partial
program is around one nat away from bilin18 and only agrees with its top prediction on
about 64% of positions.

## What “simpler” currently means

Storage, arithmetic cost, tensor rank, program length, and human interpretability are
candidate simplicity measures. None is accepted solely because its number is small.
A simplification becomes useful only when its consequences improve:

- equal or better CE and model agreement at lower cost;
- stable behavior on held-out and shifted data;
- components that compose when several replacements are installed;
- predictable responses to finite edits;
- selective removal without damaging unrelated behavior; or
- a certificate that a parameter direction cannot affect registered outputs.

The rank-1 result is the clearest example: it is unquestionably simpler in storage,
but not yet simpler in the sense needed for reverse engineering because its CE is too
poor. Pareto frontiers are therefore used instead of an arbitrary “performance per
parameter” ratio: a program is retained only if no other measured program is both
cheaper and better.

## Current plan

1. **Finish the observed 68-action runner.** Materialize the fixed action plans,
   execute baselines and mixed programs correctly, and add frequency, intervention,
   and consumer-norm reductions. This directly tests composition and causal transport.
2. **Independently audit and run the suffix experiment.** Compare local fitting,
   suffix fitting, component removals, false parents, and exact-MLP2 repair on the same
   rows. This is the first step that can admit or reject the current early package.
3. **Test a small polynomial state-transition law.** Ask whether the 64-dimensional
   state and a small set of bilinear terms predict the next early state under both
   ordinary inputs and finite edits. Compare it to a direct map with the same rank and
   cost so composition, rather than local fitting, is being tested.
4. **Put any admitted early package into the current whole-model replacement.** This
   measures recovery of the actual $+0.8976$ CE gap and exposes interactions with
   attention and deeper layers.
5. **Compile MLP2/3 conditional on the changed upstream state.** Compare independent
   and joint replacements so MLP2 compensation cannot masquerade as understanding.

## What is blocking progress?

The blocker is implementation trust, not data or hardware. FineWeb, cached rows,
checkpoint, model, and GPU are available. The remaining final runner must still:

- construct composite programs from the correct site-specific fitted pieces;
- represent inherited, native, deployed, shuffled, and mean baselines;
- execute finite code edits and their unedited pairs;
- aggregate nine frequency bins and all 18 consumer-norm checks; and
- assemble all 68 actions exactly once without exposing raw model tensors.

The new action-plan compiler and final-only batch identity close the first semantic
ambiguity without widening the old fit trace. The largest remaining implementation
work is now the observed baseline/edit/diagnostic execution and complete aggregation,
not action naming or the mean-program artifact.
