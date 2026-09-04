# Task 21 verbatim-copy authority/compiler reproducibility review

**Reviewed:** 2026-09-04 UTC. **Verdict: APPROVE the frozen CPU authority and capability-only FIT compiler.**
This verdict is bound only to commit
`9ebab94615eade27b1eb63e4f2c6239337b71dc9`. It does not authorize a producer, model, checkpoint, GPU,
queue, enqueue, result namespace, localization, or any later phase. A separately frozen and independently reviewed
managed adapter/authorization remains required before execution.

This was an ordinary pre-run reproducibility and semantic-validity review. No task-21 outcome exists in or was opened
through the reviewed closure, and no model-facing action was taken. The canonical behavior-ID check used the legacy
source registry, not a result value.

## Immutable target and file census

`git rev-parse` resolves the requested object to exactly
`9ebab94615eade27b1eb63e4f2c6239337b71dc9`. The task-21 files in the commit have these raw-file SHA-256 digests:

| File | SHA-256 |
|---|---|
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task21.py` | `bb223267e532d6be64f1ffd02708459d914623695dbe6fb68cc87185fd7d4ae2` |
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task21_capability_fit.py` | `43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390` |
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_fit_authority.json` | `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94` |
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_select_authority.json` | `151e50755c9570cf411e614111fe9c5857d5ea13aab7fb7e53d6ce493b8a1f67` |
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_test_authority.json` | `dc3340c18d7c2efaa460fecf1e0134bc07532f939d1b424016977ecab810c155` |
| `basis_aligned/bilinear_quotient/ops/circuit_battery_task21_copy_ood_authority.json` | `bf338c34ff0ffe17a56c6c8cb8f3e7c74fcf4c0549c4f9933065bbe8cca16c38` |
| `basis_aligned/bilinear_quotient/ops/test_circuit_battery_task21.py` | `6893d5ca918bf4b1e77628cb087f589de5b847ad49030646be3e715b377c721f` |
| `basis_aligned/bilinear_quotient/ops/test_circuit_battery_task21_capability_fit.py` | `7fe20987ad38e7f6f238991dc885850b6a19d8555ea6fdeed64fce9f28578196` |
| `basis_aligned/bilinear_quotient/circuit_battery_task21_capability_fit_v1_dryrun.json` | `7f508a6daa6d322672a386316cb72d4adcd2738e001809eceaf4e62656aae408` |
| `basis_aligned/polynomial_causal/CIRCUIT_BATTERY_TASK21_VERBATIM_COPY_CAPABILITY_FIT_PREREGISTRATION.md` | `da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3` |

The target board blob has raw SHA-256
`322d6a5a29fae30306a9cff5b8c7e1bfaceb2e99327e333c137708130c6585c5`. The current copies of all ten task-21 files
above are byte-identical to the target commit (`git diff --exit-code` returned zero), so the executed tests exercised
the reviewed bytes rather than a later variant.

The compiler also freezes the existing boundaries at these exact raw-file digests, which independently match the
target commit and current tree:

| Boundary | SHA-256 |
|---|---|
| `ops/circuit_battery_integration_contract.py` | `b36317f46127dc90d7b8d38c9aca85440c6ff46adb7087fe2c1fd7a2745cfa3e` |
| `ops/circuit_experiment_spec.py` | `64ba9b75d49dbc6129d592573fee454e27e2de661daef30ca35d457dbbbb093c` |
| `ops/circuit_artifact_package.py` | `6c8f81f16e3465b33c27abacd1114bd8ae7ce2fffa358c2a665f906a49f011cc` |
| `ops/circuit_managed_entry.py` | `1c5bfe6dc8435e767e0d05e4ccb415ce04feb3b7a6da50eb342695e6747dda81` |

## Ordinal, identity, and semantic authority

Task 21 is collision-free in the strict per-task namespace. Immediately before the reviewed commit, the only
task-21-named path was its early design review; the numbered strict implementation namespace contained task 17 and no
task 21. The new implementation consistently uses task 21 in source/artifact/test names, schema, experiment ID, and
`rung=21`. Historical experiments that happen to call themselves “rung 21” are a different namespace and are not
referenced by this contract.

The authority correctly reuses the established behavior ID `verbatim_repeat.copy`. The legacy behavior registry
defines that exact key and description (“continue a verbatim repetition of one token”); the reviewed source and every
split envelope use the same key. It does not create the reversed alias `copy.verbatim_repeat`.

Regeneration and validation produce the frozen full authority SHA-256
`191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b`. There are exactly 336 rows: four phases,
21 linked panels per phase, and exactly one A1/A2/P/C row in every panel. Each phase therefore has 84 rows. The split
record digests are:

- FIT: `c4bd6e01561dc89fe702e8e813e53639cbb4ad3eee4e0c0d8b788b13fbd28cc8`;
- SELECT: `c437ebcf8fa4c00e43be26063ee985dacd767e76c41bbf0263ef9bde52638139`;
- TEST: `d780a7e0993422ed0d52aafacb42c7eb3433503d1b01bf1197bffcdd8b8c6d45`; and
- OOD: `2ee14e4547291888608f484c43d4b656f65bc5e709625cafbc5cac4de9ab640b`.

Within each phase, its 21-token vocabulary appears exactly once in every registered semantic role: target,
alternative, novel control, filler position 0, filler position 1, and—in OOD—filler position 2. Phase vocabularies and
prompt sets are pairwise disjoint. Thus token identity, answer exposure, alternative/foil exposure, and surface
position cannot be confounded with A1/A2/P/C membership.

The validator recomputes, rather than trusts, the prompt text, task/schema/effect labels, sequence length, phase
prefix and shape, structured row identity, trailing-run metadata, transform edit, and answer semantics. Checked-in
attacks establish fail-closed rejection of:

- coherent prompt-plus-ID replacement;
- re-signed structured-token corruption;
- wrong expected effect;
- wrong schema across re-signed rows; and
- changing the OOD row to the in-distribution prefix/shape.

The joint-tokenization check requires the prompt IDs to be an exact prefix of the jointly encoded prompt plus answer
and the answer to add exactly one GPT-2 token. Base and donor prompt positions are aligned. FIT/SELECT/TEST prompts
have 8 tokens and continuation position 8; OOD prompts have 13 tokens and continuation position 13. A1 replaces the
entire run, A2 replaces only the newest two-token run while retaining an older target, P changes only a novel leading
filler, and C extends the same-target run by one without changing the answer.

## FIT-only compiler and decision closure

The managed dryrun captures seven frozen roles, of which exactly one has kind `authority`: `fit_authority`. The spec
contains only phase FIT, explicitly forbids SELECT/TEST/OOD there, and names none of the three future authority files.
The compiler loads the already captured FIT bytes by exact raw-file digest; its dryrun test plants forbidden generator
calls, proving that neither full-authority construction nor any phase panel generator is entered. No SELECT, TEST, or
OOD authority bytes or artifact reference occurs in the invocation. `later_split_generation` is `false` and
`later_split_artifacts` is empty.

The checked-in dryrun parses identically to a fresh `run_managed_dryrun()` result and has frozen raw SHA-256
`7f508a6daa6d322672a386316cb72d4adcd2738e001809eceaf4e62656aae408`. It records zero model forwards, backwards,
and updates and `queue_touched=false`. The compiled spec, call manifest, metric manifest, and entire compiled contract
are independently rebound to:

- spec: `834f5b8f45facb7f585c96d7ee15b86e8a7a68f03117af929cf7843def8ab487`;
- call manifest: `ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e`;
- metric manifest: `e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf`; and
- compiled contract: `5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2`.

The physical plan is exactly four base then four donor native calls. Every call is one 21-row FIT batch at padded
length 8. This is exactly 8 calls and 168 explicit row-side evaluations. Retaining two `float32` scalars per row-side
costs exactly `168 * 2 * 4 = 1,344` raw evidence bytes, with zero backward calls and zero updates. The metric is the
side-specific registered-answer logit minus the maximum registered-foil logit; targets and nonempty foil sets are
recomputed from the linked authority. Call, summary, metric, primitive coverage, literal price, and whole-contract
mutations all reject before interpretation.

The prospective capability alternatives are exact complements: pass requires both side-wide strict accuracies at
least 0.90, all eight side-by-transform cells at least 0.85, and both side-wide mean margins strictly positive;
failure is the negation of any clause. At 21 examples, the cell bar is exactly 18 strict successes; at 84 examples,
the side bar requires 76. The checked-in test exercises both branches. A capability failure returns terminal
`hard_abort` with every one of the seven projection fields null. A pass returns only aggregate native capability
fields. Primitive schemas reject extra localization data, projection names contain no reader/site/component field,
the source contains no science runner or enqueue entry point, and no branch can select or localize a circuit or open a
later phase.

## Test evidence and the task-17 namespace failures

From `basis_aligned/bilinear_quotient` with `PYTHONPATH=.:ops`:

```text
pytest -q ops/test_circuit_battery_task21.py \
  ops/test_circuit_battery_task21_capability_fit.py
20 passed in 1.42s
```

The broader model-free boundary collection comprised 109 tests across experiment-spec adversarial checks, the battery
integration contract, task 17 authority/compiler/producer/adapter, and task 21 authority/compiler. Its result was:

```text
104 passed, 5 failed in 4.40s
```

All five failures are one unrelated environmental condition expressed by five task-17 tests. Each fails in
`circuit_battery_task17_capability_fit_producer.require_unused_namespaces()` because the already-published task-17
result, receipt, and evidence paths occupy the namespace that those pre-execution dryrun tests assume is empty. The
failures occur only in `test_circuit_battery_task17_capability_fit_producer.py` and
`test_execute_circuit_battery_task17_capability_fit.py`, before task-21 code is entered. The reviewed commit changes no
task-17 file, task 21 has a distinct namespace, all other 104 broad tests pass, and all 20 task-21 tests pass. The
task-17 failures are therefore truly unrelated to the semantic validity or reproducibility of task 21; they must not
be hidden by deleting or moving published artifacts.

## Approval boundary

The frozen task-21 CPU authority/compiler is reproducible, semantically valid for the stated local previous-token
repetition screen, phase-isolated, and fail-closed. The preregistration is prospective, states both capability
predictions before any task-21 execution, gives C no posthoc exemption, and explicitly warns that a pass would not
establish induction or attention-mediated copying.

**APPROVE commit `9ebab94615eade27b1eb63e4f2c6239337b71dc9` for use as the immutable input to a new,
separately reviewed task-21 managed producer/authorization. Do not execute or enqueue from this review.**
