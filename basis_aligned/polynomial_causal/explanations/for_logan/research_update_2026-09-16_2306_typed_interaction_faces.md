# Research update: typed interaction faces generalize where atom-wise sparsity nearly fails

## Bottom line

The inherited-city head8.2 → head9.8-O relay now provides a successful
prospective test of the sparse-interaction workflow on a third behavioral
family. The important result is not that arbitrary greedy sparsity worked. It
nearly failed. The successful rule was to keep a physically coherent
multiplicative face of the interaction lattice intact.

The frozen typed graph has three terms:

1. routing main effect, mask `1`;
2. inherited-value main effect, mask `4`; and
3. routing × inherited-value interaction, mask `5`.

On a second, unscored natural FineWeb panel, those three separately measured
terms predicted the full routing/current/inherited intervention with `15.89%`
component-relative L2 error, cosine `0.99957`, and zero material sign
reversals. Both cue directions and all six spelling endpoints passed the
preregistered `35%` composition limit. The frozen graph needs four corners
`(0,1,4,5)` instead of all eight, a `50%` evaluation reduction.

This is not yet a new four-trait circuit. OOD prediction and composition now
pass prospectively, but the exact selected face has not yet been bound into its
own standalone package, and its removal evidence does not yet include an
equal-norm same-site directional null. The verified registry therefore remains
at two complete circuits rather than being prematurely incremented.

## Experiment 1: unconstrained atom-wise selection

The intervention lattice had three declared ports at the head8.2 city-source
write:

- donor routing;
- donor current value; and
- donor inherited/first-layer value.

All eight Boolean corners were propagated through the previously identified
head9.8-O response and the native suffix. The unchanged generic graph utility
computed all seven non-baseline Möbius terms. A coefficient-free, three-step
greedy selector saw only 48 authored discovery rows. It selected:

`(inherited value, current value, routing × current value)` = masks `(4,2,3)`.

The frozen selection was then evaluated on 96 outcome-blind natural rows from
eight FineWeb `skip39000` source documents. No natural logits or activations
were available during selection.

Most of the registered predictions passed:

- discovery relative error: `8.62%`;
- aggregate natural OOD error: `24.94%`;
- natural OOD cosine: `0.97640`;
- material sign reversals: `0`;
- every endpoint error: `23.57–25.93%`;
- unrelated-readout ratios: `9.16–15.29%` of target RMS;
- required corners: five of eight.

But the result was correctly recorded as a near miss. The British-row cell had
`35.318%` error against the frozen `35%` limit. The American-row cell was
`21.46%`.

## Positive red-team: the near miss is real

The model-free audit independently reconstructed the Möbius atoms, replayed the
generic selector, and enumerated every one of the 35 possible three-atom
graphs.

It found:

- exact selected masks and selection-curve replay;
- exact reported-metric replay;
- masks `(2,3,4)` were the global discovery optimum among all three-atom sets;
  and
- the British-cell failure remained `35.318%`.

So there is no implementation correction that turns the first result into a
pass. This is the negative side of the requested red-team policy: the failure
survived an active search for miscoding.

The audit also localized the scientific problem. Atom-wise selection spliced a
large inherited-value singleton together with current-value terms. It did not
preserve a complete producer/consumer factorization. Post-hoc, the complete
routing × inherited face `(1,4,5)` had `16.39%` aggregate OOD error, despite a
worse `25.16%` discovery error. That comparison was diagnostic only and did not
replace the failed registered graph.

## The typed-face rule

For this attention edge, routing multiplies a value source, while current and
inherited values add. A coherent two-factor subgraph therefore contains both
main effects and their interaction. The reusable graph utility now has two
small primitives for this:

- `face_masks`, which closes a selected set of ports under all nonempty
  submasks; and
- `select_graph_family`, which compares caller-declared complete graph families
  on discovery rows without fitting coefficients or dropping inconvenient
  interactions.

For the replication, the two preregistered candidates were exactly:

- routing × current: `(1,2,3)`; and
- routing × inherited: `(1,4,5)`.

On the same authored discovery panel, their errors were `82.89%` and `25.16%`,
respectively, so the rule froze `(1,4,5)` before replication scoring. This
method was designed after the atom-wise miss and is not presented as a rescue
of that result.

## Experiment 2: source-disjoint replication

The replication row freezer excluded every source document used by the first
natural panel, then selected eight additional `skip39000` documents by a fixed
SHA256 ordering of eligible city occurrences. It found 29 eligible occurrences
after exclusion and froze 96 rows: six spelling endpoints and both city cues
per context. Source overlap with experiment 1 was exactly zero.

All five registered predictions passed:

| Test | Result |
|---|---:|
| Instrument and exact algebra | pass |
| Replication OOD relative L2 | `15.89%` |
| Replication OOD cosine | `0.99957` |
| Material sign reversals | `0` |
| British composition error | `15.28%` |
| American composition error | `17.55%` |
| Endpoint composition errors | `15.72–16.23%` |
| Unrelated-readout ratios | `7.56–15.10%` |
| Required corners after freeze | `4 / 8` |

All three selected atoms were live on replication. Exact source-channel,
routing/value-factorial, full-corner, and Möbius-closure checks passed; the
maximum native source replay error was `3.05e-7`, and no write leaked outside
the frozen destination mask. The run used 1,152 forwards, zero fit parameters,
zero gradients, and zero model updates.

## What this changes about the decomposition strategy

The evidence now distinguishes two notions of sparsity:

- **Atom sparsity** chooses whichever individual Möbius terms best reduce an
  in-distribution residual. It can assemble a low-error but physically
  incoherent graph whose support shifts across contexts.
- **Typed structural sparsity** chooses among closed factor faces that respect
  how the model actually multiplies and adds signals. Its discovery fit can be
  worse while its OOD graph is substantially more stable.

This is closely aligned with the DCT briefing's tensor-network view. The useful
unit is not necessarily an individually large tensor coefficient. It can be a
small closed contraction—reader, writer, and interaction—whose terms must stay
together. Context remains an open input port; averaging or independently
ranking terms can hide which contraction remains valid under context shift.

## Four-trait ledger at the pause point

| Trait | Current status for typed inherited-city face | Evidence |
|---|---|---|
| OOD prediction | **Pass** | frozen selection, second source-disjoint natural panel, `15.89%` error |
| Composition/reuse | **Pass** | unscaled sum of three separately measured atoms; both cues and all endpoints pass |
| Extraction | **Not yet certified for this exact face** | related conditional executors exist, but none is yet hash-bound to masks `(1,4,5)` as one package |
| Selective removal | **Not yet certified under the current rubric** | pair-centered midpoint removal exists and is selective, but lacks an equal-norm same-site null for the selected face |

The correct current label is therefore **strong three-family decomposition
progress, with a prospectively replicated typed sparse graph, but not a third
four-trait registry entry**.

## Reproducibility and pause state

Both GPU jobs exited successfully. No experiment is running or queued at this
pause point. The first run used 1,152 forwards in `15.51 s`; the replication
used 1,152 forwards in `15.53 s`. Each runner was preregistered, hash-bound,
passed the mandatory parser/test/gate/dry-run sequence, and executed once via
the managed queue.

Primary artifacts:

- `ODD_ATTENTION8H2_SPARSE_GRAPH_V1_RESULT.json`
- `ODD_ATTENTION8H2_SPARSE_GRAPH_V1_AUDIT.json`
- `ODD_ATTENTION8H2_TYPED_FACE_REPLICATION_V1_RESULT.json`
- `sparse_interaction_graph.py`
- `odd_attention8h2_sparse_lattice_runtime.py`

The natural next gates are deliberately left unstarted: export the exact typed
face as a declared-input package, then run an equal-norm same-site removal null.
