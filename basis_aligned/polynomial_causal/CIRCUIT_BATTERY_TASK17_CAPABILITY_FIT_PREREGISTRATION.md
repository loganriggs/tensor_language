# Task 17 positional-list capability-only FIT preregistration

**Frozen prospectively:** 2026-09-04 04:34 UTC. **Execution status:** CPU compilation only; no model invocation is
authorized by this document. This file is immutable once committed. Any change requires a new versioned file and may
not reuse these predictions as prospective.

## Question and claim level

Does native bilin18 retrieve the value at a queried list index strongly enough on the strict task-17 FIT authority to
license a later, separately preregistered writer/reader search? This is only a **capability screen**. It cannot identify,
localize, rank, or select a writer, reader, head, MLP, subspace, or circuit. A pass licenses only a new FIT localization
preregistration; it is not identification or adoption.

The task computes `f(v, q) = v[q]`. Every generated FIT situation has linked `A1`, `A2`, `P`, and `C` transforms. `A1`
changes only the query index; `A2` swaps the queried payload with a distractor; `P` changes one unqueried payload; `C`
changes the query between duplicate targets while retaining a distinct foil.

## Frozen authority and phase closure

- Full four-phase task-17 authority SHA-256:
  `16307b8bb9273d56f7c3d09cd629fca78fa1db7f110278e959b6ee301cfb7571`.
- This invocation contains only its 96 FIT rows (24 groups times four transforms), whose canonical-record SHA-256 is
  `efb8c9c7a4f66b4e816a232d3b8160c36f39d4cc10bcd47c1cb8a76b817be067`.
- The canonical FIT authority artifact is exactly 82,880 bytes with file SHA-256
  `b1d33859f15bee8be04719ec532e84057ac70ef150a06e40ae7583ce70a79d6b`.
- SELECT, TEST, and OOD rows are neither generated nor included as artifacts in the managed invocation. There are no
  prior outcomes. A planted future-split row, a changed authority byte, or a changed semantic record is a pre-call
  error, not a scientific result.

## Native answer-versus-foil metric

Each FIT row produces two native evaluations, one on `base_ids` and one on `donor_ids`. For each side, the target is
that side's jointly tokenized single-token answer. Its foil set is reconstructed separately for that side from every
distinct payload token present in the linked base/donor lists, excluding the side's target. This is important because
an answer-changing donor target can legitimately be a foil for the base prompt and therefore cannot reuse a
base-relative foil list.

For row `i`, side `s`, target token `y`, and frozen foil-token set `F`, retain only

`answer_logit[i,s] = logits[i,-1,y]`

and

`max_foil_logit[i,s] = max(logits[i,-1,f] for f in F)`.

The native margin is their difference and strict correctness is `margin > 0`. No full-vocabulary logits, hidden
states, component activations, ablations, patches, gradients, or candidate reader identities are requested or retained.

## Exact physical calls and literal price

The deterministic compiled manifest has SHA-256
`0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf`. It contains exactly these eight calls, in order:

1. `FIT:base:0:native_base` through `FIT:base:3:native_base`;
2. `FIT:donor:0:native_donor` through `FIT:donor:3:native_donor`.

Every call is a physical native forward of 24 unpadded sequences of length 13. The call kind is
`native_answer_foil_logits`, the guard is `capability_only`, and every arm is typed `native/undirected`. Exact FIT price:

- 8 forward calls;
- 192 example evaluations;
- 0 backward calls;
- 0 model updates; and
- 1,536 retained numeric evidence bytes: two float32 scalars per example. This byte price is the raw array payload;
  immutable `.npy`, call-request, result, and receipt framing is evidence metadata, not learned/scientific state.

Any call-ID, order, batch width, row membership, sequence length, array contract, or price mismatch aborts before a
capability decision.

## Frozen opposing predictions and bars

There are eight side-by-transform cells: `{base, donor} x {A1, A2, P, C}`, each with 24 rows.

- **Capability prediction:** both side-wide strict accuracies are at least `0.80`; every one of the eight cell
  accuracies is at least `0.75`; and the mean answer-minus-maximum-foil margin is strictly positive on both sides.
- **Opposing prediction / capability fail:** at least one side-wide accuracy is below `0.80`, at least one cell accuracy
  is below `0.75`, or at least one side-wide mean margin is nonpositive. This is the exact logical complement of the
  capability prediction, so there is no unregistered gray zone.

The bars are fixed before any logits exist. They will not be relaxed, replaced with top-k accuracy, narrowed to easier
transforms, or supplemented by a favorable posthoc subgroup. All 192 primitive measurements and all eight registered
cells must be published on any valid terminal. Resampling uncertainty may be reported descriptively later, but it
cannot change this deterministic FIT gate.

## Stop and continuation rules

Instrument/authority/call/price checks run before the scientific gate. If capability fails, the artifact-package
decision is `hard_abort`; every scientific projection field is null, and no reader-localization or selection namespace
may be produced. If capability passes, the only licensed next action is to write and freeze a separate FIT-only
localization preregistration. SELECT, TEST, and OOD remain closed until their exact preceding receipts exist under the
four-phase integration contract.

No queue entry, GPU use, checkpoint access, model import, Torch import, outcome read, or result publication is part of
this CPU compilation step.
