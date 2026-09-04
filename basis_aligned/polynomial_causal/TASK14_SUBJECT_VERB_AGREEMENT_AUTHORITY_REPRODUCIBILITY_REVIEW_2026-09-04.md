# Task 14 subject–verb agreement authority reproducibility review

**Reviewed:** 2026-09-04 UTC. **Exact target:**
`31b812b751fb5a39b7c7933294ca18213cb52b9f`. **Verdict: REJECT this exact commit for authority freezing or
compiler construction.** The grammatical transforms, tokenization, balance, deterministic regeneration, and
fail-closed validator are otherwise sound, but two prospective claims are false at the stimulus level: one literal
template is shared between FIT and TEST, and the C rows in FIT/SELECT/TEST contain reverse-direction duplicate
contrasts. Both have small, outcome-blind repairs. No model-facing step should begin until a repaired successor is
independently reviewed and all logical hashes are refrozen.

This review was CPU-only. It did not access a model, checkpoint, GPU, queue, runner, service, task outcome, or later
phase at runtime. It did not edit the generator, tests, or design memo and did not create a compiler, authority JSON,
result, evidence, or execution artifact. The only writes are this review and append-only board receipts.

## Exact object and repository identity

Git resolves the target to the full commit above. It adds only the generator, its tests, and the design memo, plus an
append-only board entry; no compiler, frozen split file, model wrapper, queue record, or outcome is present in the
commit.

| Target file | Raw SHA-256 at exact commit |
|---|---|
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task14.py` | `159fcd7c767c16c5a2f30239d8be26a7344964ec4db23c0ed90515b60011f799` |
| `basis_aligned/bilinear_quotient/ops/test_circuit_battery_task14.py` | `327891bd7af3e5bc74af68b5e6741d65bc6ee826b8de5917284c62733c2ad554` |
| `basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK14_SUBJECT_VERB_AGREEMENT_DESIGN_REPAIR_2026-09-04.md` | `f5fe02302b425c6af1ca5abaf7b1a6df650dc913cf7cf5ce3cb78078e22ff6fb` |
| target board blob bytes | `438c5bdcd66822c7a2d218d18a47d0375ed24081c29f042440c2e5878c9a600b` |

The canonical behavior ID `subject_verb.number_agreement` occurs in no other task implementation, registry event,
or dossier entry at the target commit. Implemented strict task IDs are task 17 `retrieval.positional_list`, task 21
`verbatim_repeat.copy`, and this task 14 ID; there is no collision. “Task 14” agrees with the unique subject–verb
agreement ordinal in the behavior-bank design, rather than claiming a new experiment-index rung.

## Provenance audit

The memo accurately separates old evidence from this new authority:

- old `qk_algoverify_sv_agreement.json` (SHA-256
  `b20851e0205061b5a05233f12649e17508e6a15085244d40ed0be9563b66b9b1`) reports 80/80 forced-choice successes,
  including 40/40 incongruent prompts, and mean signed margin 3.769;
- old `qk_svagree_patch.json` (`cc2629df72ef880cccf330cd5ca70af30dace4cc5870e1ea98b450c44a0b40f6`)
  reports incongruent margin 3.057 and a 1.403 margin drop from L11H3 removal with no accuracy flips;
- `redteam_svagree_2026-07-30.md`
  (`1a2a861aec982d37c07339664536c939f60cefaec48f59843b155b909a71c980`) explicitly retracts the balanced
  all-attention mean as a zero-prior argument and limits L11H3 to a contributory, redundant result;
- `qk_svagree_locus.json` (`962b1b8599b9e43f83584db2eeee7871c37d76640e4756f76c1017dddd125a71`)
  supports the stated early-residual/late-reader hypothesis and its same-number identity control, but does not select
  a task-14 circuit; and
- natural-text `circuit_agreement_results.json`
  (`23328acd355237569a422e2db56db5807f6f4a991a05551860ac488518bd8581`) records clean accuracy 0.9568,
  removal accuracy 0.9517, and a 0.0052 drop, correctly treated as a rejection of the old two-head general-circuit
  claim.

The current circuit dossier and experiment index contain no subject–verb-agreement event. The new data authority
therefore does not duplicate a registered strict experiment, while the old capability and component observations
remain priors only.

## What independently holds

Loading the exact generator bytes from the target Git object reproduces:

| Scope | Rows | Linked A1/A2/P/C panels | Logical SHA-256 |
|---|---:|---:|---|
| full | 512 | 128 | `6432f647eb15bae46cfcc22b922f71958edb9d1b092e7bf3e63601df9084c47a` |
| FIT | 128 | 32 | `02caa03dc84c31afab2f4dd0d175a8d119ce7322e16e63427807bbe2a4df1d35` |
| SELECT | 128 | 32 | `4380825410da4e9b40bca432edf28687889080fae898a21945f0ad99d30c8d41` |
| TEST | 128 | 32 | `da98a4cd55fba3106f68c420168ebf6b7556b14594764da65bcca8ef2a787c54` |
| OOD | 128 | 32 | `6a30cd0e155c8211e8ad4b310a0b4cb39a7313893ba9cc717142eac049462cc6` |

All 512 row IDs are unique and equal the canonical hash of their identity fields. Regeneration is stable under
Python hash randomization; changing the explicit seed changes the logical digest. A row-list permutation or alternate
seed can validate as a new candidate, as intended, but cannot retain the frozen logical SHA once a future authority
file binds it.

Each phase has a disjoint 16-pair noun vocabulary. Across 32 panels, every noun pair appears exactly twice in each of
the five lexical roles. The four ordinary subject-number by attractor-number states each occur eight times. In every
A1, A2, and P side, ` is` and ` are` occur 16 times each as answer and foil. Each C side has 32 ` are` answers and
32 ` is` foils by construction. Template IDs, full prompts, noun forms, group IDs, and noun token IDs are pairwise
phase-disjoint.

The row semantics also hold independently of metadata:

- A1 changes exactly the subject-head number token in a PP, holds every attractor feature fixed, and changes the
  answer;
- A2 does the same on the number-neutral `that I placed/noticed/moved ...` relative-clause surface;
- P changes exactly the nearest attractor lexeme at fixed number and fixed answer, including the second/nearest OOD
  attractor; and
- C changes exactly the nearest attractor number while its coordinated subject remains two morphologically singular
  definite noun phrases joined by `and`, requiring unambiguously plural ` are` on both sides.

The 64 noun pairs contain no collective noun. Some sentences are semantically unusual, but none admits dialectal
singular agreement for the coordinated `and` subject. C therefore has no collective/dialect ambiguity.

Every noun edit and both answer candidates are one GPT-2 token at their actual leading-space boundaries. Joint
prompt-plus-answer encoding equals the saved prompt IDs followed by exactly one saved answer ID. Every base/donor pair
has equal prompt length, one changed token, stable head and attractor coordinates, and an aligned final prediction
position. OOD records two attractors; A1 places the head after both, while A2 places the head before both. OOD P/C
change the later, nearest attractor. These checks pass for all 512 rows.

The validator does not merely trust signed metadata. Ten additional coherent or re-signed attacks—noun-role change,
group-number change, full surface/ID transplantation, direction reversal, cross-phase noun insertion, derived
semantic change, duplicate row, missing row, template substitution, and coordinate lie—were all rejected.

## Freeze blocker 1: literal template leakage

The implementation checks template **IDs**, not their actual format strings. Two differently named IDs have the same
literal surface:

```text
FIT P:  fit_pp_behind  = The {head} behind the {attractor}
TEST A1: test_pp_behind = The {head} behind the {attractor}
```

Full prompt strings do not collide only because the noun vocabularies are disjoint. This is still a template-level
FIT-to-TEST reuse and contradicts the requested phase-disjoint templates and the memo's claim of disjoint noun and
template authorities. A model can reuse the exact syntactic/surface template at TEST; only the lexical items are held
out. Renaming the same string is not isolation.

## Freeze blocker 2: reverse-duplicate C contrasts

For FIT, SELECT, and TEST, group `g+16` reuses all five noun-role assignments from group `g`. C forces both subject
heads to singular, while the half-dependent attractor-number schedule reverses the base/donor orientation. Therefore
each C row in the second half is the exact endpoint reversal of one first-half C row.

| Phase | Declared C rows | Unique unordered C contrasts | Unique endpoint prompts | Repeated row-side evaluations |
|---|---:|---:|---:|---:|
| FIT | 32 | 16 | 32 | 32 of 64 |
| SELECT | 32 | 16 | 32 | 32 of 64 |
| TEST | 32 | 16 | 32 | 32 of 64 |
| OOD | 32 | 32 | 64 | 0 of 64 |

There are no conflicting answers and no cross-family prompt duplicates. Within one C side, all 32 prompts are unique;
however, base and donor C contain exactly the same 32-prompt set in different order. Hence the two registered C-side
gates are numerically redundant, and the 32 row IDs represent only 16 independent grammatical edits in each ordinary
phase. Direction balancing does not require this: base and donor attractor numbers are already exactly balanced, and
a different balanced lexical assignment can retain both directions without replaying the same endpoints.

This does not make any individual C sentence ungrammatical. It does make the stated 32-contrast support and proposed
physical census scientifically inefficient and potentially misleading. In FIT, the proposed 256 row-side evaluations
contain only 224 unique prompt strings; 32 C evaluations repeat exact prompts. A later compiler must not freeze those
duplicates as though they were new examples.

## Proposed price

If all four FIT families genuinely contain 32 distinct linked rows, one base call and one donor call per family gives
exactly 8 calls and `8 * 32 = 256` row-side evaluations. Two float32 scalars per row-side cost
`256 * 2 * 4 = 2,048` raw bytes, with zero backwards and updates. The arithmetic is correct. It should not yet be
frozen for this commit because its C calls would spend 32 of those evaluations on exact repeats.

## Minimal repair and required successor checks

The following outcome-blind repair preserves the grammar, number cells, 512-row shape, and 8/256/2,048 proposed
price while making every contrast distinct:

1. In the second half only (`group_number >= 16`), assign the second coordinated head with index
   `(5 * group_number + 7 + 3) % 16`. Offset 3 is collision-free against the other four roles for every group. Each
   role remains a bijection within each half and therefore appears exactly twice per phase, but C no longer reuses
   the first-half lexical triple.
2. Replace TEST A1's literal PP with a surface absent from all other phases, for example
   `The {head} below the {attractor}`; a distinct ID alone is insufficient.
3. Add validator and adversarial-test requirements for literal template-surface disjointness, 32 unique unordered C
   contrasts, 32 unique prompts in each C side, and disjoint base/donor C endpoint sets in every phase.
4. Regenerate and freeze new full/split logical hashes, source/test/memo hashes, and deterministic subprocess checks.
   Do not preserve any digest above as authority for the successor.

An in-memory audit-only application of the two repairs passed the full semantic validator, retained exact twice-per-
role and number/answer balance, produced 32 unique C contrasts and 64 disjoint endpoints in every phase, and made all
literal phase template sets disjoint. This is a feasibility proof, not a replacement generator or normative digest.

## Tests and final verdict

With exact target hashes verified before execution, the checked-in focused suite passed 14/14 and the relevant
task/contract/experiment-spec/result-contract suite passed 105/105. The separate exact-Git-object reconstruction and
targeted mutation audit passed 10/10. These green tests establish that the implemented contract is deterministic and
internally enforced; they do not cure a missing design invariant.

**REJECT `31b812b751fb5a39b7c7933294ca18213cb52b9f` for authority freezing and compiler construction.** Preserve its
grammatical design and strong regeneration validator, apply the two minimal repairs above, then obtain a new exact-
commit independent review. This veto grants no model/GPU access, compiler construction, phase materialization,
localization, queue action, or execution authority.
