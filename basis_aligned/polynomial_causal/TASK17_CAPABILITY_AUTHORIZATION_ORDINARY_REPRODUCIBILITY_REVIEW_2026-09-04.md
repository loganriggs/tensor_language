# Ordinary reproducibility review: task-17 FIT authorization successor

**Reviewed:** 2026-09-04 05:50 UTC

**Exact target:** commit `d1e28e9947d7b257862372ba4329c8f31f2bd11d`

**Verdict:** **APPROVE**, subject to the separately reviewed hash-bound queue dependency and its ordinary pre-enqueue
current-byte check.

This is a routine scientific reproducibility/version review. I compared the authorization successor with the approved
blocked adapter, checked its frozen chain and declared execution scope, and ran the checked-in CPU suites. I did not
develop or run new exploit probes, inspect results, invoke enqueue, touch a live queue, access a model/checkpoint/GPU,
or control a service.

## Exact reviewed objects

| Object at `d1e28e994` | SHA-256 |
|---|---|
| authorization-enabled adapter | `4566f24a5a56364f0b840ed0eb297a888fab4d1017e26f3ef6fa0f4fe95abc46` |
| unchanged producer | `3dcf04c0f776c056f3701967a666025ed8b63cab4d7e60a868fd766b00ac98ea` |
| adapter tests | `e8168f43b505d08730764fd58cbb60813c4777e7e079fdc89b1d8445d2ef786a` |
| saved dry run | `cd090e7db82053ed4a7e8eafd08fc0b36573f11428bfb99a1d301918bee94758` |
| authorization amendment | `449c601472790d3f5c02c07cd3eaad3879e8ae865712d999590d6007cb90ce8f` |

These paths remain byte-identical at the current reviewed worktree. The adapter's `FILES` table binds every compiler,
contract, package, task adapter, producer, authority, preregistration, amendment, review, runtime source, facade, and
canary source by exact SHA-256. The checked-in digest census passes for every entry. In particular it retains:

- compiler commit `5da7c8cea` and compiled contract
  `526f292338abb5583942f95241be6aa2485db8421270e395bb9fa64bb34751c9`;
- full task-17 authority `16307b8bb9273d56f7c3d09cd629fca78fa1db7f110278e959b6ee301cfb7571`
  and the sole captured FIT authority artifact
  `b1d33859f15bee8be04719ec532e84057ac70ef150a06e40ae7583ce70a79d6b`;
- exact call manifest `0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf`;
  and
- exact metric manifest `3efad3188a57628ad35466a0585c2e01ff3ff120642a0301f412900aa987f362`.

## Minimal authorization delta

The approved blocked adapter has SHA-256
`15d60e1760581228b69d214ffcebebf5231a15cd5a09d018bda4bd98bae69ca5`. The exact source diff to this successor is
limited to:

1. changing `EXECUTION_AUTHORIZED` from `False` to `True`;
2. adding frozen references to the independent publication-repair approval and new authorization amendment; and
3. changing only dry-run authorization/status metadata and explanatory text.

It does not change capture, verified-module loading, runtime closure, producer dispatch, call construction, metric
decision, checkpoint/canary gates, evidence, or publication functions. The producer digest is identical before and
after authorization. The amendment itself binds the prior compiler approval, execution amendment, preserved publisher
VETO, repaired producer, provenance correction, approved blocked adapter, and publication-repair approval.

## Exact authorized scope

The sole scientific authority is one managed FIT native-capability invocation over 96 rows:

- exactly 8 native forward calls, four base and four donor calls;
- exactly 192 row-side evaluations;
- exactly 1,536 raw numeric evidence bytes, comprising two `float32[24]` arrays per call;
- 0 backward calls, gradients, or model updates; and
- no SELECT, TEST, OOD, localization, reader, writer, component, activation, or full-logit artifact.

The producer requires exactly all eight compiled calls, validates their prefix/hash and physical shapes, counts all
192 primitives and 1,536 numeric bytes, and passes only the declared primitive evidence into the compiler's capability
decision. A capability failure remains `hard_abort` with all scientific projection fields null. A pass licenses only a
separate future FIT localization preregistration; neither terminal opens a later phase.

## Deterministic model-free reproduction

Two consecutive `BQLIB_DRYRUN=1` dispatches were byte-for-byte equal to each other and structurally equal to the saved
dry-run JSON. Their canonical report SHA-256 is
`a5ecd0c263d361230ea54dbebf93906c472589a0ba8209c543e396a24e4d8bfc`. Each reports 8 completed calls, 192 examples,
1,536 numeric bytes, 0 model forwards/backwards/updates, FIT as the only evaluated phase, no forbidden phase, and the
registered capability-fail fixture with all projection fields null. Torch remained absent from `sys.modules`; the
model source, facade, two canary sources, and `jacclust` package were excluded from dry-run capture.

The focused task-17 compiler/producer/adapter suites pass **49/49** in **2.43 s**. With the experiment compiler,
adversarial specification, integration-contract, and task-adapter suites included under the repository's required
`PYTHONPATH=.:basis_aligned/bilinear_quotient/ops`, the checked-in boundary suite passes **135/135** in **5.86 s**.

## Producer and publication invariants

Because producer SHA `3dcf04c0...` is unchanged, the approved publication protections are unchanged. Before execution
it requires unoccupied create-only result/evidence/receipt namespaces, including dangling symlinks. It stages evidence,
then result, then receipt; publishes each with Linux `renameat2(RENAME_NOREPLACE)`; verifies entry identity after every
move; performs identity-checked no-replace rollback on failure; fsyncs directories; and validates the complete
hash-bound package. It refuses a weaker replace operation. The real adapter accepts no arguments and dispatches exactly
once into that captured producer closure.

## Infrastructure dependency and decision

Authorization is expressly conditional on the hash-bound lane-1 protocol. Exact infrastructure commit
`afa628e118c4ca8a48c719328293dc2c25bb6399` is approved separately by the ordinary reproducibility review in commit
`1caadb3f0`. That protocol requires the independently reviewed adapter digest as `EXPECTED_SHA256`, writes exactly
`<sha256><TAB><absolute path>`, and makes the runner verify and execute captured bytes.

**APPROVE** adapter SHA `4566f24a...` for one hash-bound managed invocation under that dependency. This review is not
an enqueue receipt. Because service inspection was explicitly outside scope, the operator must still make the ordinary
pre-enqueue confirmation that current enqueue/runner files and the live managed runner correspond to the approved
infrastructure bytes, that the adapter still hashes to `4566f24a...`, and that its create-only namespaces are unused.
