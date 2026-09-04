# Hourly strategic review — 2026-09-04 08:37 UTC

## Circuit interpretation targets

The program is trying to recover an executable description of bilin18 in which a circuit has all of the following
properties:

1. **Computational specification:** state what information is read, what operation is performed, what is written,
   and which later computation uses it.
2. **Grouping across modules and splitting within modules:** merge pieces of different heads or MLPs when later
   computation treats them as the same variable, and split a native module when its pieces do different jobs.
3. **Held-out and OOD prediction:** predict activations and behavioral effects on new nouns, templates, documents,
   and shifted constructions.
4. **Extraction or sufficiency:** reproduce the target computation, or its signed causal effect, from an isolated
   circuit plus an explicit background interface.
5. **Selective manipulation:** swaps, edits, and removals change the intended behavior while preserving unrelated
   behavior, with redundancy and multi-component interactions tested rather than assumed away.
6. **Composition and reuse:** shared computations can participate in multiple tasks, and jointly installed pieces
   behave as predicted.
7. **Stable identification:** the unit survives data splits, fitting restarts, and plausible changes of internal
   coordinates, or is defined by downstream operational equivalence.

The program-level goal remains a smaller transparent tensor program that is predictive, composable, manipulable,
and simpler under literal storage, compute, edge, state, and program prices. Compression, rank, variance, or CE by
themselves do not satisfy these circuit targets.

## What changed since 07:36

### Strict task-14 track

The subject–verb-agreement authority was repaired before model use. The first candidate repeated one literal syntax
across phases and repeated 16 coordinated-subject contrasts in reverse. The replacement has 256 unique FIT
row-sides, phase-disjoint literal templates and nouns, and 32 distinct coordinated contrasts per phase. Independent
review approved it.

The FIT compiler and then the model-facing producer were built in separate commits. The producer preserves exactly

$$
8\text{ calls}\times32\text{ prompts}=256\text{ prompt evaluations}
$$

and retains only the registered `is` and `are` logits,

$$
256\times2\times4=2{,}048\text{ raw numeric bytes}.
$$

The exact producer/blocked-adapter build `26d45e8979` passed 47/47 focused tests, 173/173 relevant broad tests, 78/78
independent coherent mutations, and four additional hash/package mutations. Review commit `753afa27e0` approved it
only as input to a separate authorization successor. Real dispatch remains blocked before artifact capture, module
loading, checkpoint access, or CUDA. The authorization successor is now being constructed and must dynamically prove
that the model, facade, fast loader, and fast-loader dependency used at runtime are the exact captured modules.

### Control and selection corrections from the parallel battery track

The old battery's P family had two distinct problems. On 11 of 21 behaviors its base prompts were literally the A1
target prompts. More importantly, the corrected P donor changes filler while preserving the declared causal variable.
It is therefore a **positive control**: a real variable-carrying component should behave similarly on A1 and P. Using
P as a negative control pinned the old selectivity ratio near one and invalidated its interpretation. The prior claim
that zero behaviors had selective writers was withdrawn; selectivity is unestablished rather than established in
either direction.

The replacement metric has useful range and a reproducible component ordering, but selecting the minimum among 36
native components is unstable: FIT and TEST chose the same minimum for 0/7 behaviors. The FIT-selected component was
still more selective on TEST than a random live component, but it did no better than the causally proposed writer.
This is evidence for a broad equivalence class of useful components and a warning against treating a native-module
argmin as a semantic unit.

This correction changes the planned interpretation of task 14. P preserves grammatical subject number while changing
noun identity, so a subject-number state should transfer across P rather than be absent on P. A1 and A2 change subject
number and should produce a signed interchange effect across two syntactic forms. C changes a distractor under a
coordinated plural subject; it tests rejection of nearby-number heuristics, but it is not automatically the same
positive control as P. The localization preregistration must specify these relations separately instead of combining
all answer-preserving rows into one generic control score.

### Mathematical route

Weight-only bilinear eigenvalue ranking remained anti-predictive. Correct moment weighting did not fix it. Contracting
against a causally established output axis improved median rank correlation from $-0.191$ to $+0.383$, but missed the
registered $+0.60$ predictive bar. The useful constraint is that a bilinear weight analysis needs the causal output
quantity first; it does not discover that quantity from weights alone. This supports task-conditioned interchange
before exact translation into quadratic weights, not another eigenvector-width sweep.

## Is task 14 still the highest-information route?

Yes, through the capability gate. Agreement provides a binary semantic variable, two answer-changing syntactic forms,
a noun-identity positive control, an attractor-conflict construction, and phase-disjoint OOD syntax. Those contrasts
can distinguish a genuine grammatical-number state from token identity, nearest-noun number, fixed position, and
surface-template rules. The capability call is cheap and cannot itself identify a circuit, but it decides whether this
well-controlled task is a valid substrate for the higher-value causal experiments.

The route is demoted immediately if the frozen capability gates fail. If it passes, the next object is not a ranking of
heads or residual eigenvectors. It is a predictive-state equivalence class: two internal changes count as carrying the
same variable only when registered downstream tests cannot distinguish their effects.

## Confound audit

- **Phase leakage and repeated support:** repaired and tested at the authority and compiler boundaries; only FIT bytes
  are materialized in the current closure.
- **Shared token difficulty:** balanced `is`/`are` answers and both prompt sides reduce it; later localization must use
  signed donor-minus-recipient changes rather than raw margins alone.
- **Positive versus negative controls:** the battery failure shows that answer preservation does not tell us which kind
  of control a contrast is. Task-14 A1/A2/P/C semantics must remain separate.
- **Multiple mediators and redundancy:** single-component ablation can include interactions and compensation. Later
  tests need interchange, necessity, and sufficiency across candidate sites and combinations.
- **Native module boundaries:** head/MLP labels are candidate physical sites, not the semantic basis. Downstream-response
  equivalence may group pieces across modules or split one module.
- **Post-selection and winner's curse:** any site or subspace chosen on FIT must be evaluated once on unopened SELECT or
  TEST. The minimum over many sites is not evidence of unique identity.
- **Nonlinear loss composition:** logit-difference movement and CE damage answer different questions. The capability
  gate uses logits; later circuit claims need signed task effects and unrelated-task preservation.
- **Runtime and provenance:** the authorized successor must capture every executable dependency, verify import identity,
  re-hash the checkpoint before and after the eight calls, and publish create-only with the receipt last.

## Genuinely different next approaches

1. **Predictive-state interchange.** Identify a minimal internal state by its full registered future effects across
   A1, A2, P, C, several donors, and later readers. This directly targets grouping, splitting, held-out prediction,
   sufficiency, and stable identification. It dies if the same FIT map does not transfer across nouns/templates or if
   equivalent donors produce inconsistent effects.
2. **Task-conditioned DAS followed by exact weight translation.** Learn the smallest rotated residual subspace whose
   swap changes A1/A2 toward the donor while preserving the P relation and the registered C behavior. At a bilinear MLP,
   expand the identified state change exactly into left–right cross terms and a quadratic self term. It dies if the
   subspace is donor-specific, unstable across fits, or cannot be selectively injected and clamped.
3. **Downstream-response grouping across native modules.** Build response vectors describing how later readers use
   candidate source pieces, then quotient sources that are operationally indistinguishable and split sources with
   different readers. This directly addresses the user's desired interaction-determined basis. It dies if clusters do
   not transfer across held-out constructions or have no selective intervention consequence.
4. **Gradient screen plus causal confirmation.** Use task-logit gradients only to narrow source×reader pairs, then test
   preregistered exact interactions. This is cheaper but remains a screen until swaps/removals reproduce held-out signed
   effects. It dies if gradient ranking does not enrich causal hits over matched controls.
5. **Hankel/predictive-state realization over registered continuations.** Define state by all observed continuation and
   intervention responses and test whether a small reusable transition system predicts them. This is basis-independent
   and mathematically clean, but it dies if sampled state rank is unstable under added tests or cannot map back to a
   selective physical implementation.

The eigen/rank route is not selected: its only demonstrated outputs are compression and a weak causal ranking, so it
cannot currently change a circuit-level decision.

## Ranked actions and concrete continuation

1. **Finish and independently review the prospective authorization successor.** This changes instrument validity, not
   circuit evidence. Kill condition: any source-identity, real-mode, phase, checkpoint, namespace, or publication gate
   fails. This construction is active now.
2. **Run exactly one managed task-14 FIT capability screen if and only if the successor receives exact final approval.**
   It changes whether agreement localization is scientifically admissible. Kill condition: any frozen capability bar
   fails, producing the all-null hard abort.
3. **On a pass, freeze a localization factorial before opening another phase.** Opposing predictions distinguish a
   transferable grammatical-number state from noun identity, nearest-noun number, fixed syntax, and distributed
   answer-only effects. Use multiple valid donors and hold out nouns/templates.
4. **Only after causal identification, compile the state into bilinear weight terms and test selective installation.**
   This is where the tensor structure becomes explanatory rather than merely compressive.

No frontier installation or explained-fraction claim changed during this hour. The current work improves the validity
of a prospective circuit-identification instrument; it is not itself a circuit result.
