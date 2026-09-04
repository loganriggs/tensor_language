# Independent postexecution audit: task14 subject–verb agreement capability FIT

**Audited:** 2026-09-04 UTC

**Exact outcome commit:** `90c5b1606f6eb309ea9fca0042414c9146d8c455`

**Verdict:** **VALID CAPABILITY PASS**. The published package is complete, hash-consistent, reproducible from the
frozen FIT authority and evidence, and its exact registered decision is `terminal="ok"`. This is evidence that the
native model clears the preregistered task14 FIT capability bars. It is **not circuit identification**: no circuit
reader, writer, site, head, MLP, subspace, causal state, intervention, held-out phase, or OOD behavior was measured.

This audit read exact Git objects, the current published task14 package, and the task-specific execution log plus its
completion-ledger line. It used CPU-only parsing and NumPy reconstruction. It did not read model or checkpoint bytes,
use CUDA/GPU, access or modify a queue, control a runner/service, rerun the producer, mutate the result, or read/generate
SELECT, TEST, OOD, localization, or other later-phase data.

## Authorization and immutable ancestry

Git ancestry checks pass for the complete reviewed chain:

| Stage | Exact commit |
|---|---|
| repaired task14 authority | `e9686bc9bbb40f872d8e8320b30fab4f019e524d` |
| authority review | `ea7efad782c088ba91a2ce338a9f740563c4e7c1` |
| capability compiler | `fc586c1158ddeee7df8f4b502deec54189609c4c` |
| compiler review | `10afc5d6005d169879b07e92cb5fcb4e3a65f312` |
| producer and blocked adapter | `26d45e89797515240eec368bc313728925d5f48a` |
| producer review | `753afa27e05b594acc39b0c1d84d72272c26e640` |
| prospective authorization successor | `434f11a927669b86525bf6b9bdc050bd64de544b` |
| final preexecution review | `117af1288b42c8928745842154e0248c5fa9da86` |
| outcome package | `90c5b1606f6eb309ea9fca0042414c9146d8c455` |

Both the authorization successor and final review are strict ancestors of the outcome commit. The outcome commit adds
exactly the result, receipt, and 24 evidence files—26 files total—and does not alter any authority, compiler, producer,
adapter, preregistration, review, or runtime source. At the outcome commit, the critical executable/authority bytes are:

| Object | SHA-256 |
|---|---|
| authorization-enabled adapter | `ea6acb2a0382a474bda5e48f3c21d368697ab4a7b56adeae489506eff0a25ecd` |
| exact producer | `9ba9448fcebcd764aa2b91e91333b3bbb2549a899b1f8304f2ce3f83bf741e3e` |
| capability compiler | `98b2d263c5120c1a7b700dc4bb451f65cc9f9b338740d2cfbc7ae25a3ba5aab1` |
| FIT authority file | `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f` |
| final preexecution review document | `44e90142bc7ff0128b0dd2ad1dbb2e4e1dd3039ab52953de3fdfeb0a59895f7a` |

These exact Git objects match the corresponding current worktree bytes. The authorization was one invocation only;
this postexecution audit grants no retry or second execution.

## Complete package and descriptor closure

| Package object | SHA-256 |
|---|---|
| result JSON | `4239a25df47602dc07fce8602328f555a6bebc237f9dd897f34e812cf69dba12` |
| receipt JSON | `4f46c4b2f376fb96b5d71044a3cd6331c8a0784c90105335d49ebddeaf4f8aca` |
| canonical 24-entry evidence descriptor list | `d663354e317eb7f05c9188e37a9533ab9b7d2bad0faabd709d6d9a198d55a617` |

The receipt binds namespace `circuit_battery_task14_capability_fit_v1`, the exact result hash above, and the ordered
24-entry evidence descriptor list. The result embeds the identical list. Every descriptor's path, serialized byte
length, and SHA-256 matches both the exact Git blob and the current file. Git records all 24 as mode `100644`; the
current result, receipt, evidence files, and every evidence directory entry are non-symlinks. The evidence tree has
exactly the 24 declared files and no missing, duplicate, unsafe, or extra entry. The independently invoked generic
package validator returns the exact result object.

The 16 `.npy` files are 256 serialized bytes each. Each independently loads with `allow_pickle=False` as one finite,
C-contiguous `float32[32]` array with 128 numeric bytes. Thus serialized arrays occupy 4,096 bytes while the registered
raw numeric price is exactly `16 × 32 × 4 = 2,048` bytes. The eight saved call JSON files occupy 22,464 bytes, so
the complete evidence tree contains 26,560 serialized bytes; only the numeric payload, correctly, is priced as model
evidence.

### Exact evidence descriptors

| Evidence object | Bytes | SHA-256 |
|---|---:|---|
| `calls/0000_FIT:base:A1:0:native_base_A1/answer_logit.npy` | 256 | `c4192bbb9253434729bc94440992b2b2b8f67cd034143581972d558d12825479` |
| `calls/0000_FIT:base:A1:0:native_base_A1/call.json` | 2808 | `b6ac4019c11195738f93ac3e2b2732bc17fd471e53332d6f0dc424c72b1512d3` |
| `calls/0000_FIT:base:A1:0:native_base_A1/foil_logit.npy` | 256 | `0c24ff4173fc664eebef876f6c959ad6ecd635641cb46f2c0083da28d8f6cc25` |
| `calls/0001_FIT:base:A2:0:native_base_A2/answer_logit.npy` | 256 | `5da2af16dbe7c9a4b3dc55d6f58c4e0a50c235aa77c7a9e45199d8e59fc55d35` |
| `calls/0001_FIT:base:A2:0:native_base_A2/call.json` | 2808 | `678992b84b365d2c06e93c6745f7b9a1ef23292bb31fcd25779d53ba551e98c3` |
| `calls/0001_FIT:base:A2:0:native_base_A2/foil_logit.npy` | 256 | `aa2b4daa4d84df5fe2804a49cf4e6ad02a99563d7fc208c9148afb6b388ff74e` |
| `calls/0002_FIT:base:P:0:native_base_P/answer_logit.npy` | 256 | `98751ffeaf079b40ef16cd9c1daacfd74c140bc906c923965df340f97a301718` |
| `calls/0002_FIT:base:P:0:native_base_P/call.json` | 2804 | `480dd855accab30e873ba8a3fad08eddbcac68e4a44331a51667338a177732e0` |
| `calls/0002_FIT:base:P:0:native_base_P/foil_logit.npy` | 256 | `bc8a4ae25b332b7c5c78b3ac7ab0a5740a431cfe766f0710379460b9d828d6f1` |
| `calls/0003_FIT:base:C:0:native_base_C/answer_logit.npy` | 256 | `9a3f32b6b8296ba405bd15aa4707ae7ec043f20becdda00397a33d4b3046f989` |
| `calls/0003_FIT:base:C:0:native_base_C/call.json` | 2804 | `fed1e86342506a83462cc50658c2f833176cc77c02bfa76a6594e37a91e3977b` |
| `calls/0003_FIT:base:C:0:native_base_C/foil_logit.npy` | 256 | `c4476f8e5826dc6de1782a40eabe46f43be8ae98fea1ef047639d38c91cc3fa6` |
| `calls/0004_FIT:donor:A1:0:native_donor_A1/answer_logit.npy` | 256 | `5f8d3dd8d8fadfd3132962c0806c87694aa28eeab96c7a2baebd6b250cb1c7f6` |
| `calls/0004_FIT:donor:A1:0:native_donor_A1/call.json` | 2812 | `ba19044e1b0017f4bc5c89cc2f35a2390ea079852bec3629fcd983cd9c61fc1b` |
| `calls/0004_FIT:donor:A1:0:native_donor_A1/foil_logit.npy` | 256 | `9598b093a690ddb86cb82fda262caf9133d475847fd93cce716319c14494e12f` |
| `calls/0005_FIT:donor:A2:0:native_donor_A2/answer_logit.npy` | 256 | `97624366824570e5787954f694b411e7f5c4129f3ee25cfc1847f079932e308b` |
| `calls/0005_FIT:donor:A2:0:native_donor_A2/call.json` | 2812 | `27c5f7235e4e60256b648eae27d95c55d981fa940ea26af57eeade7cac7c7cd7` |
| `calls/0005_FIT:donor:A2:0:native_donor_A2/foil_logit.npy` | 256 | `74ee46f5379a5528e5022a967314528a3f5eba62470ebab6738eb9ab541b8947` |
| `calls/0006_FIT:donor:P:0:native_donor_P/answer_logit.npy` | 256 | `84860cf7edae7c56bca3d00550d0e34510ad6dbd39647abfee35de68b658736e` |
| `calls/0006_FIT:donor:P:0:native_donor_P/call.json` | 2808 | `db8a9606e814c421ef9376429ec25f81637886d06f2269bbaa112edb3f8c0ea4` |
| `calls/0006_FIT:donor:P:0:native_donor_P/foil_logit.npy` | 256 | `c8b9b660ade4400b497ec0d0419093a6247c87783e3bd166fc1b047cac2efe89` |
| `calls/0007_FIT:donor:C:0:native_donor_C/answer_logit.npy` | 256 | `4266a1d320d3fd927ec9883a2ecad5b78f5f5ae6dfa8204a5985228b0a54349b` |
| `calls/0007_FIT:donor:C:0:native_donor_C/call.json` | 2808 | `0835c2f33548b543b791ed22d28022b7c08b251ef94e29b60751401f5148e62a` |
| `calls/0007_FIT:donor:C:0:native_donor_C/foil_logit.npy` | 256 | `aee571be3a795a6a8078a3796cc33d3b25ee51683a04dffe09f249b2a30a2064` |

## Independent FIT reconstruction

I loaded the exact FIT authority Git blob by its frozen SHA, without invoking any phase generator, then recompiled it
through the exact compiler. The recompiled digests match the result:

| Contract | SHA-256 |
|---|---|
| complete authority logical digest | `1cf6cf12668c7428719134bbee03ab84f57cc150f2653cc12ffc4a71566c8db1` |
| FIT authority-record digest | `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1` |
| compiled capability contract | `84f8e1cf85323dba94d13c7c716afef448b8621bff6b534c2025715420e86a82` |
| physical call manifest | `4b4da44c5090914f87d52e018bc9a8d18b74a202bdb82667283a9f1564682e0e` |
| native metric manifest | `5da9f66829156e352afe087c75f92a7a6a37f06fe1ec5177efeffd9442609dcc` |

The authority contains 128 rows and 256 distinct base/donor prompt strings. Each saved `call.json` is the exact
canonical compiled call object. Calls appear once each in the literal order base A1, A2, P, C, donor A1, A2, P, C;
each binds 32 exact row IDs. A1/P sequences have length 5, A2/C sequences length 8, every prediction coordinate is
the frozen final token, and target/foil IDs are the authority-bound opposite copulas. Reconstructing one primitive per
saved array element yields 256 unique `(call, row, side, transform, incongruent, answer_changes)` keys. The completed
call-prefix digest equals the complete manifest digest. No Torch module was imported during reconstruction.

The literal price is therefore exactly eight forwards, 256 row-side evaluations, zero backwards, zero updates, and
2,048 raw numeric evidence bytes. `evaluated_phases=["FIT"]`, `forbidden_phases_opened=[]`, later-phase generation is
false, and neither result nor evidence declares any localization surface.

## Independently recomputed scientific metrics

Correctness is the frozen strict comparison `answer_logit > foil_logit`; mean margin is the ordinary arithmetic mean
of `answer_logit - foil_logit` in the exact evidence order.

| Population | Correct / N | Accuracy | Mean margin |
|---|---:|---:|---:|
| all row sides | 249 / 256 | 0.97265625 | 3.6290491363033652 |
| base pooled | 124 / 128 | 0.96875 | 3.618874939158559 |
| donor pooled | 125 / 128 | 0.9765625 | 3.6392233334481716 |
| base A1 | 32 / 32 | 1.0 | 4.643179416656494 |
| base A2 | 32 / 32 | 1.0 | 4.373246029019356 |
| base P | 32 / 32 | 1.0 | 4.297763131558895 |
| base C | 28 / 32 | 0.875 | 1.1613111793994904 |
| donor A1 | 32 / 32 | 1.0 | 4.71800772100687 |
| donor A2 | 32 / 32 | 1.0 | 4.396267853677273 |
| donor P | 32 / 32 | 1.0 | 4.333137556910515 |
| donor C | 29 / 32 | 0.90625 | 1.1094802021980286 |
| base incongruent pooled | 48 / 48 | 1.0 | 3.7305969297885895 |
| donor incongruent pooled | 48 / 48 | 1.0 | 3.897544801235199 |
| all incongruent pooled | 96 / 96 | 1.0 | 3.8140708655118942 |
| base A1 incongruent | 16 / 16 | 1.0 | 3.8297268748283386 |
| base A2 incongruent | 16 / 16 | 1.0 | 3.8836077451705933 |
| base P incongruent | 16 / 16 | 1.0 | 3.4784561693668365 |
| donor A1 incongruent | 16 / 16 | 1.0 | 4.113225966691971 |
| donor A2 incongruent | 16 / 16 | 1.0 | 3.9367117285728455 |
| donor P incongruent | 16 / 16 | 1.0 | 3.6426967084407806 |

All seven incorrect row sides occur in C. Four are base C and three donor C; no A1, A2, P, or incongruent row fails.
Two C row IDs fail on both sides (`744b63c6...`, `971267ec...`); base-only failures are `e6280101...` and
`9ee7deb4...`, and the donor-only failure is `af94a1f2...`. The smallest-magnitude error is the strict base-C miss at
margin `-0.0000362396240234375`; it remains unambiguously incorrect under the frozen strict-greater-than-zero rule and
does not touch any aggregate bar.

## Every frozen gate and terminal

The independent Boolean reconstruction passes every literal condition:

- exact base/donor counts are 128/128; all eight cells contain 32 rows; all six ordinary incongruent cells contain 16;
- pooled base and donor accuracy exceed the `>= 0.85` bar by 0.11875 and 0.1265625;
- all six ordinary A1/A2/P cell accuracies are 1.0, exceeding `>= 0.85`, and all means are strictly positive;
- all six incongruent accuracies are 1.0, exceeding `>= 0.85`, and all means are strictly positive; and
- base/donor C accuracies exceed `>= 0.75` by 0.125 and 0.15625, with positive means. The smallest mean used by any
  gate is donor C's 1.1094802021980286, safely above the strict `> 0.0` boundary.

The metric-evidence and answer-relation predicates both independently pass. Reapplying the frozen decision function
to the reconstructed 256 primitives exactly reproduces all saved floats, all three true predicate results,
`capability_pass=true`, and `terminal="ok"`. No bar was moved or interpreted post hoc.

## Runtime, checkpoint, canary, and operational receipts

Without reading the checkpoint or live canary artifacts, I compared the published receipts to the exact frozen
producer constants:

- checkpoint revision `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`, config SHA
  `428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`, weights SHA
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`, 2,067,738,635 bytes, tokenizer vocab
  50,257, logit vocab 50,304;
- CPython 3.12.14, NumPy 2.5.2, Torch 2.11.0+cu128/CUDA 12.8, tiktoken 0.14.0, einops 0.8.2; and
- both canary pass flags true, composition `v2_layer17_mlp_plus_scalar`, fingerprint
  `6b22b221a811382775e6a64b4198a61f2f9bcc55b826d0d12d0512d1a28be99c`.

The exact producer source can publish only after pre/post full checkpoint receipts compare equal and pre/post canary
receipts compare equal. The single published receipt is not a fresh checkpoint rehash by this audit; it is a validated
record produced behind those frozen control-flow gates. Rehashing checkpoint bytes was explicitly outside scope.

The current task-specific execution log is a regular non-symlink JSON file, SHA-256
`f8710a244690bd12cea3475f6ac9f7f278061433d1a4436e1e65e01672834128`, and reports the exact result/receipt/evidence
paths, `terminal="ok"`, 8 forwards, 256 evaluations, and 2,048 raw bytes. The completion ledger records
`08:58 execute_circuit_battery_task14_capability_fit exit=0`. These operational files are not part of the immutable
package descriptor and were not modified by this audit.

## Decision and scientific scope

The package is a valid, registered native-capability pass, not an invalid instrument. It establishes that the frozen
model can choose the correct singular/plural copula at the preregistered rates on the 256 task14 FIT row sides,
including perfect accuracy on every ordinary incongruent subset. It does not identify why, where, or through which
internal computation the model succeeds. In particular, it contains no activation, intervention, causal swap,
necessity/sufficiency, held-out validation, or component attribution evidence.

Accordingly, this result may serve only as the capability opener for a separately preregistered, independently
reviewed FIT localization experiment. It does not itself license a circuit claim, later-phase access, execution,
resampling, threshold relaxation, or retry. A separate CPU-only task14 localization-design claim is already present on
the append-only board; this audit neither reviews nor approves that future design.

## Reproduction checks

- Independent exact-Git/current package reconstruction and metric script: **PASS**.
- Generic complete-package validator: **PASS**.
- Focused frozen compiler plus result-contract tests: **33 passed** in 1.69 s.
- Exact Git mode census: **24/24** evidence blobs are `100644`; filesystem census: **24/24** regular files, zero
  symlinks, zero extras.

All reconstruction used `PYTHONDONTWRITEBYTECODE=1`, `BQLIB_NO_MODEL=1`, and `CUDA_VISIBLE_DEVICES=''` where applicable.
