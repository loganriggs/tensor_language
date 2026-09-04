# Task 17 FIT capability model-execution amendment

**Frozen prospectively:** 2026-09-04 04:49 UTC. **Status:** implementation-only and **not authorized for model
execution**. Compiler commit `5da7c8cea` received independent CPU-contract approval in the review named below. The
model-facing producer and adapter still require their own different-agent approval before execution may be authorized
or enqueued. This amendment changes no task rows, predictions, thresholds, phase order, metric, or scientific claim in
the immutable capability preregistration.

## Why this amendment exists

The immutable preregistration compiled an exact model-call and evidence contract but intentionally did not define a
model-facing implementation. This document freezes that implementation prospectively. It is a capability screen only:
a pass may license a separately preregistered FIT localization step, while a fail is a valid `hard_abort` result with
every scientific projection field null. Neither terminal identifies any model component or circuit.

## Exact native computation

For each token batch $X\in\mathbb{N}^{24\times 13}$, the producer evaluates the pinned float32 bilin18 checkpoint by
the exact trained-model path:

$$
h_0=\operatorname{RMSNorm}(E[X]),
$$

$$
(h_{l+1},v_{l+1})=\operatorname{Block}_l(h_l,v_l,h_0),\qquad l=0,\ldots,17,
$$

and

$$
z=30\tanh\!\left(\frac{W_U\operatorname{RMSNorm}(h_{18})}{30}\right).
$$

The stored answer number for row $i$ is $z_{i,-1,y_i}$ and the stored foil number is
$\max_{f\in F_i}z_{i,-1,f}$, where the target $y_i$ and side-specific registered foil set $F_i$ come verbatim from
the frozen metric manifest. Full logits exist only transiently on the GPU and cannot enter the evidence package.

The facade's existing production dispatcher is not used because its shape contract is fixed to $4\times256$, whereas
this invocation is exactly $24\times13$. The minimal explicit path above retains the final logit softcap and is checked
against the pinned model topology before use.

## Calls, evidence, and price

The producer executes every compiled row-side evaluation, even when two rows contain the same token sequence. It does
not cache or deduplicate the 192 evaluations because transform-cell membership and the registered price are defined at
the row-side level. It makes, in order:

1. four base-side native calls of 24 rows each; and
2. four donor-side native calls of 24 rows each.

For each call it saves the exact compiled `call.json` request plus only `answer_logit.npy` and
`max_foil_logit.npy`, both contiguous `float32[24]`. Thus the raw numeric payload is exactly
$8\times 2\times24\times4=1{,}536$ bytes. `.npy` headers, request JSON, the result, and the package receipt are
audit metadata rather than learned state. The literal price remains 8 forwards, 192 example evaluations, 0 backwards,
and 0 model updates.

## Frozen implementation and runtime boundary

- Model-facing producer SHA-256:
  `a46b64410d0090d2034523be5b1eee58250c876131d78f97b3262c25ca637750`.
- Compiler source SHA-256:
  `c3e8cca7268ee17280dab15f5a5399592db5fc3c6319b4ff693f5c7b6ab259b3`.
- Compiled-contract SHA-256:
  `526f292338abb5583942f95241be6aa2485db8421270e395bb9fa64bb34751c9`.
- Call-manifest SHA-256:
  `0edd2541dcddb0d3442b05e6df3f65971a9d973281a676fc9117338435567bdf`.
- Metric-manifest SHA-256:
  `3efad3188a57628ad35466a0585c2e01ff3ff120642a0301f412900aa987f362`.
- FIT-authority file SHA-256:
  `b1d33859f15bee8be04719ec532e84057ac70ef150a06e40ae7583ce70a79d6b`.
- Original capability preregistration SHA-256:
  `0fea3731f59c8b9f9b1d1e898f2b4dbca65f706406b69f1b3e429e85bc621a63`.
- Independent compiler review SHA-256:
  `0494f037748a5e781d038c9960875fbb1e1ee219711c78649246d402e8e6b5c4`.
- Observed-model facade SHA-256:
  `b62947f772c807259890a9d09dfcbe5e91ad339a0bffa867ab99177fde4c728c`.
- `jacclust/tt_model.py` SHA-256:
  `49ecdbd6c060ff5b3e57f3134d87ba32841390c891c42e6ae23b71d8627612b2`.
- Pinned model revision:
  `ed9146549ee6dc8ed8cd75e9d48fcfe4278f4240`.
- Pinned config SHA-256:
  `428042bfd807ba36f8b4326395440fbbebe52cd3d040212e6fef14a4fdf2d83c`.
- Pinned weight SHA-256:
  `680d6c26cf05af2e9b5eaac1d52fa1c9e4ea443f60a7c74ad211740e317d6de3`.

The real path requires CPython 3.12.14, NumPy 2.5.2, Torch 2.11.0+cu128 with its CUDA 12.8 runtime, tiktoken
0.14.0, and einops 0.8.2. Before checkpoint loading it requires both repository canaries to pass, including stable
canary-2 composition `v2_layer17_mlp_plus_scalar` and fingerprint
`6b22b221a811382775e6a64b4198a61f2f9bcc55b826d0d12d0512d1a28be99c`.

## Publication and failure behavior

The final namespace is exactly:

- `circuit_battery_task17_capability_fit_v1_results.json`;
- `circuit_battery_task17_capability_fit_v1_receipt.json`; and
- `circuit_battery_task17_capability_fit_v1_evidence/`.

All three must be absent before the model boundary. Evidence and result are staged with exclusive file creation,
mutually hash-bound, and atomically renamed with the receipt published last. Existing final paths are never replaced.
Every completed call must be the exact full compiled prefix, and every array must pass exact dtype, shape, finiteness,
and byte-price checks before publication.

Authority, source, compiler, runtime, checkpoint, canary, call, array, price, prefix, or package failures abort without
a scientific terminal. A valid capability failure publishes the complete eight-call evidence package, a `hard_abort`
decision, and null projection fields. In either case SELECT, TEST, OOD, gradients, weight updates, component
activations, and localization outputs remain absent.

## Authorization dependency

The checked-in adapter remains deliberately execution-blocked. Its model-free dry run may be reviewed, but its real
branch must raise before checkpoint or GPU access until a different agent publishes an exact producer/adapter review
and a later authorization amendment freezes that review's digest. Only then may the adapter be revised, independently
re-reviewed, and enqueued through the managed runner. The compiler review already on record licenses this build step,
not model execution. This dependency prevents implementation ownership from becoming self-approval.
