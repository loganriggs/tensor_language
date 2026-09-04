# Task14 subject–verb agreement capability-FIT producer implementation preregistration

Frozen prospectively at 2026-09-04 08:13 UTC, before any task14 model execution or outcome access.

## Authority and scope

This build is licensed only by compiler commit
`fc586c1158ddeee7df8f4b502deec54189609c4c` and its independent review commit
`10afc5d6005d169879b07e92cb5fcb4e3a65f312`. The exact compiler-review bytes have SHA-256
`a1707dd88949a9b5beb439b275e665cda1a7a62a6d5eedf076d20d192c852e59`.

The immutable inputs are:

- capability compiler SHA-256 `98b2d263c5120c1a7b700dc4bb451f65cc9f9b338740d2cfbc7ae25a3ba5aab1`;
- FIT authority file SHA-256 `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f`;
- capability preregistration SHA-256 `06a9747b4707999e11637a45cf83588bfd9cb8671d6b3a25790518af62900f8b`;
- compiled-contract SHA-256 `84f8e1cf85323dba94d13c7c716afef448b8621bff6b534c2025715420e86a82`;
- call-manifest SHA-256 `4b4da44c5090914f87d52e018bc9a8d18b74a202bdb82667283a9f1564682e0e`;
- metric-manifest SHA-256 `5da9f66829156e352afe087c75f92a7a6a37f06fe1ec5177efeffd9442609dcc`.

This unit may construct a producer, a model-free dryrun, tests, and an adapter whose real branch is disabled.
It does **not** authorize model/checkpoint/GPU access, queueing, publication, result/outcome access, localization,
or SELECT/TEST/OOD materialization. Any real run requires a later prospective authorization and a fresh
different-agent review of exact committed bytes.

## Exact computation

The producer must execute the frozen calls in this literal order: base A1, A2, P, C, then donor A1, A2, P,
C. Each call contains 32 distinct FIT rows. A1/P have sequence length 5; A2/C have sequence length 8. For
each row it reads logits at the frozen final prompt position and retains only two C-contiguous finite
`float32[32]` arrays: the registered answer-token logit and the opposite-copula foil-token logit. It must
reconstruct all and only the 256 primitive records, including the frozen `incongruent` and `answer_changes`
labels. No full vocabulary logits or hidden states may enter evidence.

Literal price: exactly 8 forward calls, 256 row-side evaluations, 2 arrays × 256 values × 4 bytes = 2,048
raw numeric bytes, zero backward calls, and zero updates. The scientific complement is a valid all-null
`hard_abort`; malformed arrays, coverage, runtime, or provenance are instrument failures and never become a
scientific pass/fail claim.

## Model boundary and publication

The future authorized branch will bind the exact bilin18 model source, observed-model facade, and the shared
`fastload.py` implementation plus its explicit dependencies. It must first verify runtime versions, CUDA
availability, both canaries, checkpoint revision/config/weight bytes and hashes, and facade topology. It then
uses the shared no-random-initialization + memory-mapped CPU loader, validates the resulting production model,
moves it once to float32 CUDA, and evaluates without gradients. The native path is embedding, input RMS norm,
18 native blocks, final RMS norm, language-model head, and the model's `30*tanh(logits/30)` soft cap.

The reserved create-only namespace is `circuit_battery_task14_capability_fit_v1`. Publication must stage one
complete package, atomically install evidence then result then receipt using Linux
`renameat2(RENAME_NOREPLACE)`, count dangling symlinks as occupied, reject late races without overwrite, and
roll back only entries whose inode identity was installed by that invocation. Receipt is always last. There
is no retry loop.

## Required build evidence

Model-free and fake-runtime tests must cover exact call/order/row/position/token/price reconstruction, strict
array shape/dtype/contiguity/finiteness, all-null scientific failure, distinguishable instrument failure,
FIT-only closure, source/authority/compiler/review mutations, import-cache and disk substitution, checkpoint
and runtime gates, forbidden result keys, dangling-symlink and late-race publication attacks, receipt-last
crash cleanup, exception propagation, and rejection of every attempted real dispatch before producer/model
loading. A checked-in dryrun may contain only the plan and synthetic fixtures; it must report zero model,
GPU, backward, update, queue, and publication operations.
