# Task 14 subject–verb agreement replacement-authority review

**Reviewed:** 2026-09-04 UTC. **Verdict: APPROVE exact commit
`e9686bc9bbb40f872d8e8320b30fab4f019e524d` for separate CPU-only capability-FIT preregistration/compiler
construction.** This approval covers only these exact authority-generator bytes. It does not freeze split files,
approve a future compiler, authorize model/checkpoint/GPU access, enqueue a run, open any phase, or select a circuit.

The predecessor rejection in commit `b294167ff2ccc9da4b9ac11de7599feab71020d1` remains immutable. It is the direct
parent of this replacement and supplies review context, not approval. The replacement was audited from its exact Git
objects rather than inferred from the working tree.

This review was CPU-only and did not edit the generator, tests, or design memo. It did not access a model, checkpoint,
GPU, queue, runner, service, task outcome, result, evidence, localization artifact, or later-phase runtime. Its only
writes are this document and append-only board receipts.

## Exact replacement bytes

| Object | Exact identity |
|---|---|
| replacement commit | `e9686bc9bbb40f872d8e8320b30fab4f019e524d` |
| direct rejected-parent commit | `b294167ff2ccc9da4b9ac11de7599feab71020d1` |
| generator source | `33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94` |
| adversarial-test source | `254fe3798efd8a4426f30e054fd8e5646a5bd6635df69815f376311ac2023694` |
| repaired design memo | `3cb4556d1ad2c1564f2708028e5d624c4519fbc4d52a38cac27b9d10d8312f68` |
| target board bytes | `df203966376026e117ed0051bc66bf6d4119494d7bb8bd59892f164f69bc7c99` |

The commit changes only the existing CPU generator, its tests, the design memo, and the append-only board. Its tree
contains no task-14 compiler, materialized split authority, result/evidence/outcome, model adapter, or execution
artifact. The behavior ID remains `subject_verb.number_agreement`; it has no collision with the implemented task 17
or task 21 IDs or with a registered experiment-index/dossier event. The task-14 ordinal remains the unique
subject–verb-agreement entry in the behavior bank.

## Reconstructed authority and determinism

Executing the exact source blob through the CPU integration contract reproduces 512 unique rows: 128 linked
A1/A2/P/C panels, exactly 32 panels in each phase.

| Scope | Rows | Panels | Canonical logical SHA-256 |
|---|---:|---:|---|
| full | 512 | 128 | `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1` |
| FIT | 128 | 32 | `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1` |
| SELECT | 128 | 32 | `d6d8a7e7cae24ac3e25e3bef11bde4b4b235e950a23c2842978e7fd2a91803b6` |
| TEST | 128 | 32 | `d62dae278f66ae5a2e77aadf8b841fe9aecf4bf2fa7bb9378b8d59e9f5829b27` |
| OOD | 128 | 32 | `f2e4a6fc68be3ff8a87efde056780996106b9fb10a532381588d3d47d9da40b6` |

All 512 row IDs are unique and equal the canonical digest of the complete declared identity. Generation is invariant
to Python hash randomization; changing the explicit seed changes the logical digest. The validator regenerates each
row from split, seed, group number, noun roles, numbers, transform, and template rather than accepting self-consistent
metadata.

## Phase isolation, balance, and repaired C census

Each phase uses a disjoint 16-pair noun vocabulary. Noun forms/token IDs, complete prompt strings, group IDs, template
IDs, and literal template format strings are pairwise disjoint across FIT, SELECT, TEST, and OOD. Within every phase,
all 256 row-side prompts are unique.

TEST A1 now uses the unique literal surface
`The {head} in front of the {attractor}`. Direct comparison of actual `_TEMPLATES` values—not merely their labels—
finds four distinct surfaces within each phase and no cross-phase intersection. Replanting the predecessor's literal
FIT/TEST alias fails the validator with `task14 literal template surfaces leak across phases`.

The repaired second coordinated-head index is exactly

```text
(5 * group_number + 7 + (3 if group_number >= 16 else 0)) % 16.
```

Independent enumeration shows five distinct noun roles inside every group. The +3 second-half shift is collision-free
against the other four roles for every group. Each of the five noun roles remains a bijection in each 16-group half,
so every noun pair occurs exactly twice per role and phase.

In every phase, C now has:

| Property | Reconstructed count |
|---|---:|
| unique base prompts | 32 |
| unique donor prompts | 32 |
| base/donor endpoint overlap | 0 |
| unique unordered intervention pairs | 32 |

Replanting the predecessor's unshifted second-head assignment fails closed because the C base/donor endpoint sets
overlap. Thus none of the 64 C row-side evaluations repeats another endpoint in any phase.

For every A1/A2/P side and phase, all four subject-number by attractor-number cells contain eight rows; ` is` and
` are` each occur 16 times as answer and 16 times as foil. The changed C attractor number is singular in 16 and plural
in 16 rows on each side. Both C sides contain 32 plural ` are` answers and 32 singular ` is` foils by their explicit
coordination rule. These are exact counts, not metadata assumptions.

## Grammar, semantic edits, and OOD structure

The predecessor's valid grammar survives byte-for-byte except for TEST A1's PP wording and the lexical role
reassignment:

- A1 changes exactly the subject-head number token in a prepositional construction, holds attractor features fixed,
  and changes the answer;
- A2 changes exactly the subject-head number token in a number-neutral relative clause, holds both attractor features
  fixed, and changes the answer;
- P changes exactly the nearest attractor lexeme at fixed number and fixed answer; and
- C changes exactly the nearest attractor number under two singular definite conjuncts joined by `and`, with the
  unambiguous plural answer ` are` on both sides.

The 64-pair lexicon contains no collective noun. Although some noun combinations are semantically unusual, none
creates a dialect-dependent singular agreement reading for the coordinated C subject.

OOD contains two attractors in every row. A1 fronts both attractors and places the controlling head after them; A2
keeps the controlling head before both. P and C change the second, nearest attractor. All stored semantic coordinates
match those relations, so the OOD authority tests position and distractor-count shortcuts without changing the
grammatical answer rule.

Every singular/plural noun edit and both answer strings are one GPT-2 token with their actual leading-space boundary.
For all 1,024 row sides, encoding prompt plus answer equals the stored prompt IDs followed by exactly one answer ID.
Every base/donor pair has equal length, one changed prompt token, unchanged head/attractor coordinates, and the same
final prediction position.

## Provenance and mutation resistance

The provenance conclusions of the rejected-parent review remain accurate and do not preselect a component:

- the archived 80-row PP assay was 80/80, including 40/40 incongruent, mean margin 3.769;
- the red-team record retracts balanced all-attention cancellation as evidence and limits L11H3 to a redundant margin
  contributor;
- the locus experiment supplies an early-number-feature/late-reader hypothesis plus same-number identity control;
  and
- natural-text removal of the old two-head ensemble changed accuracy only 0.0052, rejecting that ensemble as a
  general agreement circuit.

No subject–verb-agreement event exists in the current strict dossier/index. Those results motivate capability testing
only; no old rows, model values, component selection, or threshold is embedded in this generator.

The checked-in validator explicitly rejects the two old `31b812b` constructions. An independent exact-object battery
also rejected ten further coherent or re-signed attacks: noun-role substitution, group-number substitution, full
surface/ID transplantation, direction reversal, cross-phase noun insertion, derived-semantics mutation, duplicate
row, missing row, template substitution, and coordinate lie. No mutation survived.

## Proposed FIT price and tests

A future compiler may schedule one 32-prompt call for each of four families on each of two sides. This is exactly
8 forwards and `8 * 32 = 256` unique row-side prompt evaluations. Two retained float32 scalars per evaluation cost
`256 * 2 * 4 = 2,048` raw numeric bytes, with zero backwards and updates. Unlike the rejected predecessor, all 256
FIT evaluations are unique. This is a prospective price only; no call manifest or compiler exists yet.

With bytecode and pytest caches disabled:

```text
focused task14 suite: 16 passed in 2.66s
task14 + integration/compiler/spec-adversarial/result-contract suite: 107 passed in 6.55s
independent exact-Git-object old-construction + coherent-mutation checks: 12/12 rejected
```

## Final verdict

**APPROVE exact generator SHA-256
`33d7b62b3a0ffb4c798e75f085b7e96988e09b07be16667c5f9f8871c6339f94` from commit
`e9686bc9bbb40f872d8e8320b30fab4f019e524d` for construction of a separate CPU-only, capability-FIT
preregistration/compiler.** That next unit must capture only a newly materialized FIT authority at newly frozen hashes,
reconstruct exact 8/256/2,048 pricing, and receive its own review. This approval does not itself create or approve
those artifacts and grants no model/GPU/queue/outcome/localization/later-phase authority.
