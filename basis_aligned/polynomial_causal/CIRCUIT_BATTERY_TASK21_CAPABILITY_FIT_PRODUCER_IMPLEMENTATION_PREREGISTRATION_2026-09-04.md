# Task 21 FIT capability producer implementation preregistration

**Frozen prospectively:** 2026-09-04 06:39 UTC, after final source normalization. **Status:** CPU-only producer and
blocked managed-adapter build.
This note does not authorize model/checkpoint access, GPU use, execution, queue changes, enqueue, publication, or any
task-21 result/evidence namespace. A fresh independent review and a separate prospective authorization are required.

## Approved input and narrow purpose

Independent review commit `ca088ce0906160958a2586cff50b707699b7eb88` approved only the exact task-21
authority/compiler commit `9ebab94615eade27b1eb63e4f2c6239337b71dc9`. The review document SHA-256 is
`3f66075ab775ce27084203999859ea6941efec6d2154a6987994b48e011c7c50`. The producer preserves:

- full authority `191cb52e627f9ddd482e36214fc3486ccb2b08f7b75f7a15ae800dfee9be325b`;
- FIT authority file `69f3250f71904d0d0dc16253d9819c50587e85a3fd01f7776d36bcafad1b4e94`;
- compiler source `43ff54a930338127670f9291bb7bac66e914a11cdd04e919f222a5a13bb89390`;
- call manifest `ac179a95415a7ae906ab887b97a060c217f4a0efc77b7fbefe42c833c9b2f23e`;
- metric manifest `e8cab6e2fb8000bd144f92182abd71c7774d3afcd2dc1b1de50f9c1a9ec79faf`;
- compiled contract `5e926429a995dc0faa18f7c5b2d00a48e47f6876adda82011e7d0e91e35a16c2`;
- capability preregistration `da72c855b70176563244a292973293247bc014b3bbd07779bee635a8a2a973a3`.

The only future scientific question is whether the frozen bilin18 model predicts the exact registered continuation
token on both sides of the 84 FIT rows. This remains a local previous-token repetition screen. It is not an attention,
induction, retrieval, localization, or circuit-identification experiment.

## Exact model-facing computation

The producer source SHA-256 is
`395ded6fbe39d06cb9e30be0553036a39dc1b51bbecd8ae55a29ad1e5581bcaf`. It is a minimal dimension/identity adaptation
of the already reviewed task-17 producer, retaining its fail-closed runtime, model, canary, evidence, and create-only
publication checks.

For each of the exact four base calls followed by four donor calls, the future evaluator would receive one `21 x 8`
integer token array. It would run the model's native path:

$$
x_0=\operatorname{RMSNorm}(\operatorname{Embedding}(t)),\qquad
x_{\ell+1}=\operatorname{Block}_\ell(x_\ell;x_0),\qquad
z=30\tanh\!\left(\frac{W_U\operatorname{RMSNorm}(x_{18})}{30}\right).
$$

The `tanh` is the model's native logit soft-cap and is not optional. From the final sequence position, the evaluator
copies only the registered answer logit and maximum registered-foil logit into contiguous `float32[21]` arrays. It
does not retain full logits, hidden states, activations, gradients, attention patterns, component labels, or
localization information.

The exact price is:

- 8 model forward calls;
- 168 explicit row-side evaluations, including repeated base prompts as distinct registered rows;
- 1,344 raw numeric bytes (`168 x 2 x 4`);
- 0 backward calls and 0 model updates.

The future decision is delegated unchanged to the frozen compiler. A pass requires both side-wide accuracies at least
`.90`, every side-by-transform cell at least `.85`, and both mean answer-minus-maximum-foil margins positive. Its exact
complement is `hard_abort`, with every scientific projection field null. Neither terminal opens localization or a
later phase.

## Frozen model and runtime gates

The dormant real branch binds the existing observed model, without opening it during this build:

- model revision `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`;
- config SHA-256 `428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`;
- weights SHA-256 `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`;
- expected weights size `2,067,738,635` bytes;
- model source SHA-256 `49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2`;
- observed-model facade SHA-256 `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`.

Before any future model load, the producer requires the frozen CPython, NumPy, Torch/CUDA, tiktoken, and einops
versions; a real CUDA device; both existing canaries; exact facade constants; verified checkpoint bytes and size; one
CUDA device for every float32 parameter; and finite native logits of shape `21 x 8 x 50,304`. Any mismatch aborts
before scientific interpretation.

## Managed closure and publication boundary

The blocked adapter must safely capture every dry-run artifact once by file descriptor and SHA-256, preload the exact
captured Python bytes into `sys.modules` in dependency order, and execute the producer against those module objects.
This prevents mutable worktree or import-cache substitution of the compiler and its dependencies. Model/facade/canary
sources are runtime-only and excluded from dryrun capture.

The final namespace, if later authorized, is new and task-specific:

- `circuit_battery_task21_capability_fit_v1_results.json`;
- `circuit_battery_task21_capability_fit_v1_evidence/`; and
- `circuit_battery_task21_capability_fit_v1_receipt.json`.

Publication uses Linux `renameat2(..., RENAME_NOREPLACE)` for evidence, result, and receipt in that order. It treats
dangling symlinks as occupied, never overwrites a late race, rolls back only entries whose inode identity matches this
invocation, and installs the receipt last. A complete `hard_abort` is a valid package; an incomplete package is not a
scientific result.

## Build-only opposing checks

The CPU tests and checked-in dryrun must establish all of the following without Torch/model import or namespace
publication:

1. exact `21 x 8` requests, 8 calls, 168 evaluations, 1,344 numeric bytes, and 24 evidence files;
2. a passing synthetic fixture returns `ok`, while a capability-failing fixture returns all-null `hard_abort`;
3. call/metric row order, float32 array shape/type/contiguity, primitive coverage, checkpoint receipt, and nested result
   surface mutations fail closed;
4. future-phase roles, generators, outcomes, localization fields, backward/update paths, and old diagnostic battery
   code are absent;
5. dangling final symlinks, preexisting namespaces, late destination races, crash recovery, and external inode
   substitution cannot be overwritten or mistaken for this invocation; and
6. the adapter's real branch rejects before bootstrap, safe reads, module loading, runtime import, or model access
   because `EXECUTION_AUTHORIZED = False`.

Passing these checks licenses only a fresh independent review of the producer and blocked adapter. It does not license
changing the authorization flag, executing the model, or enqueueing anything.
