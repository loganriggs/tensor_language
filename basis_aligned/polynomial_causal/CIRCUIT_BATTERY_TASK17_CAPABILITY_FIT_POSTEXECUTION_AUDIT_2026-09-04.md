# Task 17 FIT native-capability post-execution audit

**Executed:** 2026-09-04 05:52 UTC. **Terminal:** valid `hard_abort` at the frozen native-capability gate. This is a
capability null, not an implementation failure. It closes localization, SELECT, TEST, and OOD for this positional-list
task version.

## What was computed

The task asks the model to return the payload at a queried index in a short list. There are 24 linked FIT situations
and four transformations per situation:

- `A1` changes the queried index;
- `A2` swaps the queried payload with a distractor;
- `P` changes an unqueried payload; and
- `C` changes the query between two positions that contain the same target while retaining a distinct foil.

Each of the 96 transformed rows was evaluated on its base and donor prompt. For each side the run retained only the
logit of the correct answer and the largest logit among the other payload tokens. A row is correct when

$$
m_i = z_i(\text{answer})-\max_{f\in F_i}z_i(f)>0.
$$

The run made exactly 8 batched forwards, covering 192 row-sides, and retained exactly 1,536 numeric bytes. It made no
backward calls, updates, activation captures, component interventions, localization decisions, or later-split calls.

## Frozen bars and observed result

The preregistered gate required at least 80% accuracy on each side, at least 75% in every side-by-transformation cell,
and positive mean margin on both sides.

| Measurement | Required | Observed |
|---|---:|---:|
| base accuracy | at least 80% | **43.75%** (42/96) |
| donor accuracy | at least 80% | **40.63%** (39/96) |
| worst cell accuracy | at least 75% | **29.17%** (7/24) |
| base mean margin | greater than 0 | **-0.479** |
| donor mean margin | greater than 0 | **-0.593** |

The full cell breakdown is:

| Transformation | Base accuracy | Donor accuracy | Base mean margin | Donor mean margin |
|---|---:|---:|---:|---:|
| `A1` | 37.50% (9/24) | 29.17% (7/24) | -0.768 | -0.876 |
| `A2` | 37.50% (9/24) | 33.33% (8/24) | -0.768 | -0.774 |
| `P` | 37.50% (9/24) | 33.33% (8/24) | -0.768 | -1.052 |
| `C` | 62.50% (15/24) | 66.67% (16/24) | +0.388 | +0.332 |

For `A1`, `A2`, and `P`, the linked construction reuses the same base prompt while changing the donor relationship;
that is why their base measurements are exactly equal. They are still separate registered row-side evaluations and
were not deduplicated.

## Interpretation

The model does not reliably perform indexed retrieval in this strict prompt format. The easier duplicate-target
control `C` performs better and has a positive average margin, but it also misses the 75% cell bar. This pattern is
consistent with the duplicate answer reducing ambiguity; it is not evidence for a particular head, MLP, or subspace.

Because native behavior itself is weak, a writer/reader search would mostly explain model errors or prompt priors. The
preregistered `hard_abort` therefore did useful work: it prevents a clean-looking causal decomposition from being built
on a task the model does not robustly solve. The thresholds and rows will not be relaxed or resampled.

## Integrity and provenance

- Result SHA-256: `251a963997a46397d45530c5308b61b9e45b4c621f2aeb44b6e34cd334029f69`.
- Receipt SHA-256: `c6916977b33ce5584a8d2247c8929ff1e47a18e999888109309ecbc9e98b5ebf`.
- The package validator reconstructs every receipt/result/evidence hash.
- Instrument evidence and native-capability predicates passed; only the scientific capability predicate failed.
- The checkpoint, runtime, and both canaries matched their frozen identities.
- Evidence, result, and receipt were installed create-only with receipt last.
- Only FIT was evaluated; `forbidden_phases_opened` is empty and `later_phase_generation` is false.

## Next adoption-track step

Task 17 is closed. The next strict capability task will use an already-known-capable behavior from the old diagnostic
battery only as a task-selection prior, while generating a new untouched four-phase authority. The preferred first
candidate is verbatim copy/repetition if its `A1`/`A2`/`P`/`C` transformations can be made genuinely causal and not
mere prompt paraphrases. Otherwise the design step will compare it with successor and agreement tasks before freezing
another authority. As before, the first run will be capability-only FIT; no component search is licensed in advance.
