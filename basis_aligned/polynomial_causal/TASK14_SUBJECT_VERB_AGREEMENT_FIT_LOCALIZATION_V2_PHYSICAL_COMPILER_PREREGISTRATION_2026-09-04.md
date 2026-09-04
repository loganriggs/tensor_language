# Task 14 FIT localization v2: physical compiler preregistration

**Prospective freeze:** 2026-09-04 UTC. **Status:** CPU-only, outcome-blind compiler candidate. This object can license
only a fresh different-agent review of the exact committed bytes. It does not authorize a producer, model or
checkpoint access, CUDA/GPU use, activation collection, result publication, queue/enqueue, retry, or opening SELECT,
TEST, or OOD.

## 1. What this compiler is for

The scientific question is whether the model uses a one-dimensional, causally interchangeable representation of
subject number. At the subject head position, H, the claim is only local noun-number transport. At the final prompt
position, Q, the stronger claim is complete grammatical subject number: a coordinated subject such as “the key and
the dog” must be represented as plural even though each head noun is locally singular. After identifying H and Q
subspaces without assuming that native heads or MLPs are the right basis, the experiment tests necessity,
two-site redundancy, and an ordered H-to-Q reset/rescue relation.

This compiler translates the immutable logical design into exact physical model calls. It does not run them. Its
checked-in manifest commits every possible conditional call through an ordered 32-byte SHA-256 index. The compiler
itself is the canonical replay program: `iter_call_descriptors()` regenerates each full descriptor, including prompt
IDs, answer/foil IDs, target and donor token coordinates, optimization roles, A_C alignment slots, seed/rank/step,
cache reads/writes, and array contracts. A future producer must capture and hash-verify the compiler and every source
role, regenerate calls with this iterator, and compare each call ID to the next index entry before model access.

## 2. Exact authority

The logical v2 authority is commit `8f41f51cdf7e073063201cc48760622607ce91b9`. Its independent APPROVE review is
commit `2ffd6cf77998a6c7fb6af0c4e89c742bf1bbb923`, review SHA-256
`2905aeb040fad2d16062a22e3c4d32d9dd6953c468724ff51a80ab9fa849d384`. The compiler binds:

- FIT authority artifact `e88fd860c28c9b369abe4a8ec28372f93bb94b6e841265206c43e6929a25ac2f`, logical rows
  `3cf3315a77b3176418739e7a9357c0dbd9b95724d6b276038f53691b873377d1`;
- v2 partition artifact `1f43b767fb39082d7872629d1a8b700e90e055c9529d9d319fe483f77d91fad3`, records
  `285092178ef25e5aee923a2b02ec791c6b2df83e7c47f185626cd5cfa507d08c`;
- v2 donor artifact `ff702f2936e2445a247c6fca3a55d177e80974b2a5e14fb6de0a5fe2761db50a`, records
  `6e1fc1fef2715e0c87f0e494646057957bad284f7b69b1e52dcc4ec0f3e6f905`, and endpoint table
  `1b0deab978dbd3126ac09b22818609177b1b1da461eaa1812aa2d05bbb9d8438`;
- the exact v2 builder, tests, preregistration, independent review, 09:30 spectral derivation, shared experiment and
  package contracts, model source/facade, fast loader, and its runtime dependency, with every path and digest in the
  call manifest.

Source reads are descriptor-based, regular-file-only, no-follow reads with before/after identity checks. Invalid
JSON, duplicate keys, nonfinite JSON constants, a changed source digest, wrong phase, wrong row census, or altered
logical-record digest aborts before model access. Only the FIT paths appear in the authority closure.

## 3. Exact model-call semantics

There are 19 residual boundaries: boundary -1 is the normalized embedding input before block 0; boundary b in
0…17 is the residual after complete block b, with the remaining blocks beginning at b+1. The residual width is 1152.
At boundary 17 the remaining computation is final RMSNorm, language-model head, and the exact output softcap
`30*tanh(logits/30)`. The answer contrast is always the float32 logit for ` are` minus the float32 logit for ` is`.

The native cache stores only the H and Q residual vectors for all 256 FIT endpoints at all 19 boundaries, plus the
second C head vector, native answer/foil logits, and the 64 ordinary A1/A2 DISCOVERY endpoint gradients used by the
screen. It does not store a full
sequence at every boundary and it does not use a suffix cache. Every intervention call reruns its target token
sequence through the native prefix, applies the exact one- or two-site hook using cached donor H/Q vectors, and
continues the same forward. Equal-length sequences are batched together. Every descriptor binds the full target IDs,
answer/foil IDs, relevant target and donor positions, and item roles.

The forward is exactly:

```text
token embedding -> input RMSNorm -> blocks 0..17 -> final RMSNorm
-> language-model head -> 30*tanh(logits/30) -> float32 answer/foil values
```

The current production facade cannot be reused as the execution path merely because it validates the checkpoint: its
dispatch surface assumes a different fixed batch shape. A future producer therefore needs a separately hash-bound
full/suffix localization implementation and a native-forward equivalence canary. That implementation is not part of
this unit.

## 4. Conditional call DAG

The fixed stage order is:

1. source/runtime/checkpoint/namespace preflight;
2. FIT native residual and answer/foil cache;
3. DISCOVERY gradients;
4. full-state DISCOVERY ceilings at every H/Q site;
5. the DISCOVERY-only spectral diagnostic and rank-one joint fits at every eligible Q and the top three eligible H;
6. DISCOVERY H selection, Q onset selection, and top-two-Q selection;
7. A1-only, A2-only, rank-two, and rank-four fits at the selected H and Q;
8. locked VALIDATION full-state ceilings and projected transfer/alignment/control gates;
9. selected-Q single-site necessity;
10. top-two-Q joint necessity only if singleton necessity fails;
11. ordered H-to-Q reset/rescue only if a necessity route passes and H is earlier than Q;
12. the v2 terminal projection in its exact nine-clause precedence.

The manifest contains templates for every possible site chosen by a prior conditional. A runtime activates only the
registered chunks, recomputes their exact roots, and writes a stage receipt. It cannot shorten a 400-step fit, reduce
the five seeds, omit a Q trajectory, swap the discovery and validation halves, or reuse a validation outcome as a
selector.

For each optimizer step the logical sampler supplies exactly 32 registered relations. Current-projector leakage
normalizers are recomputed over their complete DISCOVERY reference cells on every step and differentiated through;
they are not detached or replaced by a frozen scalar. A_C uses cached C endpoint vectors and contributes its exact
alignment slots even though it does not itself require an output forward. Each physical graph batch binds the full
logical step; exactly the final batch of that step triggers one backward and one Adam update. Thus “400 optimizer
updates” is not “400 model forwards.”

## 5. Physical price

The compact manifest contains `3,821` conditional chunks and an ordered call index with
`743,881` possible call hashes (`23,804,192` bytes). Enumerating every mutually exclusive
template is an audit device, not the price of one run.

The minimum valid prefix through all native, gradient, and 38 discovery-ceiling chunks is:

```json
{"backward_calls":4,"backward_graph_batches":4,"example_evaluations":14304,"forward_calls":145,"optimizer_updates":0,"token_evaluations":91152}
```

The exact conservative branch-complete maximum—19 eligible Q sites, three H sites, every selected fit and validation
arm, singleton necessity, the conditional redundancy arm, and the subsequent reader arm—is:

```json
{"backward_calls":60004,"backward_graph_batches":118004,"example_evaluations":9207984,"forward_calls":119207,"optimizer_updates":60000,"token_evaluations":63782508}
```

For comparison, the complete path where selected-Q singleton necessity passes and proceeds directly to the reader,
so the conditional two-site redundancy arm is not called, is
`{"backward_calls":60004,"backward_graph_batches":118004,"example_evaluations":9207024,"forward_calls":119177,"optimizer_updates":60000,"token_evaluations":63776268}`.

It contains exactly 60,000 logical Adam updates. `forward_calls` counts physical forward batches;
`backward_graph_batches` counts forward graphs accumulated into losses; `backward_calls` counts actual autograd
backward operations; `example_evaluations` counts prompt sequences passed through those forward batches; and
`token_evaluations` is the corresponding sum of batch size times sequence length. The maximum is exact because the
row/batch price differs by H versus Q but not by residual boundary.

The hard wall-clock GPU allowance is 28,800 seconds (eight hours), enforced by an external watchdog and monotonic
checks around every model call. A call may start only if an independently reviewed throughput receipt shows enough
time for its p99 cost. Missing that proof hard-aborts before model load. There is no automatic retry or partial-run
scientific interpretation.

PyTorch graph, allocator, and kernel-workspace peak bytes cannot honestly be made exact before the model-facing
implementation exists. The manifest therefore fails closed instead of inventing a number. Before authorization, the
exact frozen producer must obtain a hash-bound peak-memory receipt on a non-task canary using the largest registered
shape (192 sequences of length 8), rank four, two simultaneous forward graphs, one backward, one QR, and one Adam
update in float32. The receipt binds the producer/model/checkpoint/runtime/device and both allocated and reserved CUDA
peaks. Free memory before model load must be at least the larger of 1.25 times measured reserved peak and measured
reserved peak plus 2 GiB. Retained evidence arrays are offloaded to CPU after each call. An exceeded reviewed peak,
OOM, absent receipt, or wrong implementation/runtime is a hard abort with no scientific terminal or partial package.

## 6. Arithmetic, initialization, and evidence

Model weights, residuals, projectors, Adam state, training objective, answer logits, foil logits, and retained numeric
arrays are float32. TF32 is disabled. Deterministic algorithms are required. Reported metrics, selection, medoids,
quantiles, correlations, and RMSE are float64 CPU computations. The training median is the exact sorted even midpoint
with autograd; reported quantiles use Hyndman–Fan type 7.

Initialization uses the v2 SHA-256 Rademacher rule followed by reduced QR in increasing column order and positive-R
sign fixing. Runtime must replay and compare every initialized projector before fitting. Adam, cosine schedule,
objective coefficients, cell ordering, SHA ordering within cells, five seeds, three ranks, and all 400 steps are
those fixed by v2.

The retained raw numeric array contract has a fixed no-ceiling prefix of `61,694,592` bytes and a
branch-complete maximum of `63,394,944` bytes. Shapes and formulas are literal in the manifest. Scalar metrics
are canonical JSON float64 numbers, not an unspecified extra numeric array. Evidence, result, and receipt occupy a
new task14-v2 namespace and use atomic no-replace installation that refuses files, directories, dangling symlinks,
and late races; the receipt is installed last. Exceptions, deadline expiry, canary failure, source change, checkpoint
change, incomplete arrays, wrong dtype/shape/contiguity, call-index mismatch, or a dead intervention cannot publish a
scientific terminal.

## 7. Spectral diagnostic is not a decision rule

The 09:30 mathematical review motivates a DISCOVERY operator

$$
A v = \operatorname{mean}\frac{\sigma}{2}
\left[g(\Delta^\mathsf{T}v)+\Delta(g^\mathsf{T}v)\right],
$$

using the same unweighted signed affirmative-cell means as the corresponding H/Q joint objective, excluding controls
and A_C. Float64 CPU Lanczos uses 64 iterations with full reorthogonalization. At every retained site it reports the
projector distance to fitted rank one and the Pearson correlation and RMSE between finite intervention effects and
the local gradient prediction. It sees DISCOVERY only. It never selects a site, seed, rank, validation row, terminal,
or success predicate, and it does not replace the registered Rademacher DAS initialization. This tests whether the
causal direction is locally visible without letting a convenient linearization define the answer.

## 8. Frozen gates and terminal meaning

All scientific bars, including the 0.65 full-state ceiling rule, ordinary and coordinated causal transfer,
construction alignment, same-state leakage, rank falsifiers, cellwise necessity, two-site interaction, reset/rescue
denominators and overshoot guards, medoid/median/four-of-five aggregation, site tie-breaking, and terminal precedence,
are copied into the manifest and source-bound to v2 sections 4–12. A finite causal null is a scientific failure, not
instrument invalidity. Nonfinite arithmetic, broken hashes, invalid optimizer health, or missing required calls are
instrument invalidity. Only the two successful ordered-reader terminals can motivate a separate SELECT design; no
terminal in this run opens a later phase or weight translation automatically.

## 9. Frozen compiler artifacts

| Artifact | SHA-256 |
|---|---|
| compiler source | `ffa56273f6fee686e193fa53cb8021f782536e79fbb629d30020a78cce065e6b` |
| call manifest | `f264ef64c03a2053f2c5344588d0adc8eb03ef3a8cb257d7d02c04f3a478568d` |
| ordered binary call index | `ae399e393d03af9b6232b7fc5339dd892b418ec7c88943735f8b72fc064c8ad9` |
| deterministic dry run | `c9c113dcd1b99fcd51a11046b984cde50d29d31be200aa778242eab079ab13a7` |
| focused adversarial tests | `5bf582950bf1d14bef73cd6605839ef1a88856af6c03984968fc02f6fc9fd256` |

The dry run performs no model, checkpoint, CUDA/GPU, queue, outcome, or later-phase access. Any scientific execution
requires a new model-facing producer, managed adapter, prospective authorization amendment, exact-byte independent
review, and separately reviewed queue entry. This compiler commit by itself grants none of those steps.
