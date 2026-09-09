# Bilinear reconstruction pilot: faithful execution, no new circuit discovered

9 September 2026. Verdict: **stop this local sharing search; retain the executable reference**.

The pilot meets its execution and control gates. It does **not** meet the user's updated
interpretability criterion: no previously unspecified reusable computation has been extracted
from the trained model with identified inputs/consumers and held-out joint-intervention evidence.
The successful recurrence is an alternative way to execute the original computation. It does
not make the model's circuits fall out of a decomposition.

## What the plan contributes to the four properties

| Desired property | Established here | Still missing for a discovered circuit |
|---|---|---|
| OOD prediction | Direct/recurrent agreement on independent generated documents and longer prefixes | A discovered operation that predicts genuinely new constructions or factor combinations |
| Extraction | Complete tiny and existing small-model token-to-logit executors; no cached native activation inputs | A smaller explained circuit with its input dependencies accounted for |
| Removal/interchange | Exact-domain planted edits and numerical agreement after live native edits and downstream recomputation | Causal correspondence for a newly discovered trained-model operation |
| Composition/reuse | Two-consumer planted programs and joint-edit checks; correct nonlinear recomputation | Demonstrated reuse of a discovered trained computation across consumers |

The revised scientific criterion controls this report. Memory, parameter precision and latency
are resource measurements. Lower rank is a candidate description, not an identified mechanism.

## A. Complete execution and normalization controls

The tiny reference has two layers, two heads, residual width 16, key/value width 4, MLP width 32,
and vocabulary 8. It implements original-embedding injection, nonzero independent residual
coefficients, shared-first-value mixing, bilinear MLPs with output bias, untied output weights,
normalization and softcap. RMSNorm and RoPE were switched off/on separately and jointly.
The attention projection width is deliberately 8 rather than 16: this is the requested small
diagnostic, not an assertion about the trained architecture.

Across seeds 909/910, lengths 8/16/32, two-row batches, and 456 comparison records:

- Maximum full-logit FP64 discrepancy: **3.11e-15**.
- Maximum captured module discrepancy: **1.24e-14**.
- All declared atol=rtol=1e-9 checks passed; all nontrivial edits had live effects.
- Full layerwise, direct cached, and recurrent tokenwise execution agree.
- Key removal, value interchange, head-output removal, MLP removal and joint edits were checked.
  Past-source changes replay their updated source contributions; old recurrent state is not reused
  as if nothing changed. Continuous embedding inputs were also tested separately.
- A Fraction-arithmetic attention fixture passed exact direct/recurrent identities. This is
  separate from the numerical FP64 checks and does not certify the entire floating-point model.

The recurrence follows the associativity of kernel-feature attention described by
[Katharopoulos et al.](https://arxiv.org/abs/2006.16236). The normalized, position-dependent
producer functions stay in the executor. No polynomial expansion of the whole model is used.

## B. Exact discovery controls and a causal distinction

The bounded coefficient bank has 20 cubic monomials in four continuous variables and 128
physical state coordinates. Two actual quadratic query readers consume those states. A dense
Hadamard state change is applied after the update products, with the inverse folded into the
readers; no rotation is moved illegally through a product or normalization.

For U containing update coefficients, the exact certificate is

    U = C U_rows,       z_next = z + U_rows phi(x),       y = (W C) z.

The selected-row encoder and rational lift C are retained for inspection. The reduced runtime
uses the folded reader W C, not reconstruction of the 128-coordinate state. Independent source
edits use separately registered per-head update maps. Exact rational matrix identities establish
closure over the stated continuous input domain; fresh numerical source/query streams and
two-consumer edits check the executable implementation.

| Fixture | Natural-state rank | Rank including independent source edits | What it establishes |
|---|---:|---:|---|
| Planted shared cubic updates | 4 | 8 | Shared nonlinear work need not mean one shared editable memory |
| Same fixture, one key coefficient perturbed by 2^-20 | 7 | 11 | The original four-dimensional identity is correctly rejected |
| Shared values, independently generated routers | 19 | 38 | Shared payload does not identify equal routing functions |

The planted edit-closed state has reader-observable rank 8, so the extra state is relevant to
these readers. The independent-router fixture has observable rank 37 within its 38-dimensional
reachable span; that certificate is sufficient, not a claim of minimum output-preserving state.
Its routers are compared as symmetrized quadratic functions, accounting for repeated inputs.

The important distinction is the intervention domain. Four coordinates suffice when both heads
receive the same history of shared cubic features. Independently removing or interchanging
source updates at selected past positions permits different histories and expands the required
linear state to eight coordinates. This is not a theorem that globally disabling a head before
execution always doubles memory; that different edit family can have a simpler implementation.

The shared nonlinear library remains the four explicit cubic terms

    x0^3, x0^2*x1, x0*x1^2, x1^3.

They can be computed once and used by two consumers with distinct update histories. However,
elementary polynomial factoring already gives that library and eight editable state coordinates.
The discovered basis does not beat that competent baseline. The inspectable prototype even
retains zero coefficient columns rather than pretending to be an optimized sparse executor.
Maximum planted-program numerical output error was 9.55e-15; across all three fixtures it was
6.48e-14. None of these intentionally constructed fixtures establishes learned semantic structure.

An eight-token count/table control also closes. It is explicitly token-only and natural-input
only, with eight count coordinates and a 128x8 arbitrary update table. A generic complete
contextual tiny model instead gives numerical rank 128/128 in the registered local update bank,
with smallest/largest singular ratio 0.01013. This is a finite-sample diagnostic, not a global
algebraic theorem about the nonlinear model.

## C. Existing trained contextual model

The unchanged checkpoint is `runs_hop/attn-mlp-attn-rms-seed0/model.pt`, SHA-256
`02a3f0e793849899629af95d88be4f60b5ff1b42ee39116cb2a7d8512cb212ff`.
The established `hop_ablate.load` loader restores its attention/MLP/attention program:
D=128, four width-32 heads, MLP width 512, vocabulary 29, context limit 240.
It is an existing trained contextual product-attention model, not a replica of bilin18's
original-embedding and shared-value pathways, which this smaller architecture lacks.

Discovery used eight generated documents at 64 tokens, seed 1909. Before opening seed 1910
at 64 tokens or seed 1911 at 128 tokens, the local feature bank and basis were frozen.
The bank contains the first four coordinates of each key and value factor for the first two
heads in the second attention layer: 2x4x4x4=128 columns. Full upstream and downstream model
computations remain active in every evaluation.

**The bounded discovery result is a null.**

- The local update bank is full numerical rank **128/128**, with smallest/largest singular ratio
  **0.01116** at the preregistered 1e-10 threshold. There is no near-threshold ambiguity to tune.
- Exact rational proportionality checks on stored MLP weights find **1,024 distinct linear
  factors out of 1,024 occurrences**, and **512 distinct unordered products out of 512**.
- No structurally simpler candidate was therefore promoted. Reduced-circuit held-out extraction
  and causal validation are **not reached**, not passed.

These tests rule out only the registered local linear-span and exact common-factor proposals.
Full rank does not rule out a compact nonlinear program, reader-specific simplification, a
cross-boundary factorization outside this bank, or a useful explicitly approximate operation.
There was no broader factorization or rank-threshold sweep.

The unchanged execution reference passes independently of that discovery null:

- Direct adapter versus native model: exact FP64 logit agreement on the discovery batch.
- Full direct/recurrent FP64 logits over held-out populations and five arms: max error
  **3.69e-13**; worst centered intervention-vector relative L2 error **2.35e-13**.
- Maximum full-vocabulary teacher KL across these arms: **6.79e-16 nats/token**.
- Complete cached decode also passes (maximum recurrent error **3.02e-13**).
- Deployed-FP32 comparison is separately numerical: max logit discrepancy **2.44e-4**,
  relative L2 **6.13e-7**, mean teacher KL **6.28e-12**, max KL **1.07e-10**. It passes the
  combined atol/rtol=1e-4 check; its absolute maximum alone exceeds 1e-4. It is not bitwise equality.

The held-out documents are independent of discovery; their membership in the old training set
is unknown. The longer-prefix set is a shift relative to discovery within the model's trained
context. We did not establish unseen task-family or unseen-factor-combination generalization.
These output comparisons validate the compiler, not a newly identified semantic algorithm.

## Secondary costs and benchmarks

CPU, two PyTorch threads, torch 2.11.0+cu128, inference mode. No GPU operations were used in the
completed pilot. The existing GPU lane had a longer live job; this small test completed sooner
on CPU than waiting for it. GPU support remains available through the managed runner. These are
eager, unfused prototypes; a Python-loop penalty is not an algorithmic lower bound.

At B=1 and 64 retained tokens, with identical weights and FP64 outputs:

| Measurement | Direct | Recurrent |
|---|---:|---:|
| Actual persistent attention cache | 384 KiB | 2,048 KiB |
| Learned weight scalars | 400,640 | 400,640 |
| Prefill MACs including projections/MLP/output | 27.00 million | 58.96 million |
| Decoded-token MACs including projections/MLP/output | 446,080 | 921,216 |
| Median full-model prefill, five timed trials | 1.564 ms | 12.003 ms |
| Prefill min–max | 1.552–1.585 ms | 11.715–14.679 ms |
| Median complete cached token decode | 0.460 ms | 0.571 ms |
| Decode min–max | 0.459–0.466 ms | 0.563–0.575 ms |

Both loaded model objects additionally retain 145,920 fixed buffer scalars: 30,720 RoPE table
entries and 115,200 mask entries. These are not learned structure. The complete weight/buffer
accounting is separate from incremental state; temporary workspace and allocator peak were
not measured. The largest explicitly shaped pilot state tensor is 8 MiB, below the 256 MiB cap.
MAC counts omit lower-order norm, rotary, elementwise and lookup operations; no reduction in
the 400,640 opaque learned coefficients has been achieved.

For bilin18 at B=1,T=512, the derived recurrence state is 1,296 MiB FP32 versus 121.5 MiB
for a conventional two-key/one-value cache. Those are verified dimension-based counts, not a
full-size allocation or runtime experiment. The original direct backend remains the sensible
default at these short contexts.

Final pilot execution took **19.12 wall seconds / 26.99 process CPU seconds**: A=6.66 s,
B about 4 s, C=8.42 s. A preliminary symbolic rank calculation was stopped after minutes and
replaced by exact rational DomainMatrix arithmetic on the same matrix. No bars, populations or
scientific predicates changed. The source/architecture inspection and implementation occupied
roughly 25 minutes of session wall time; that is distinct from executor runtime.

## Decision and next scientific hypothesis

Stop treating an accumulator span or exact duplicate-factor scan as the primary discovery
method. Retain the tested executor and exact edit-domain controls. The scientific success
predicate is false, while the three implementation/control predicates are true.

A stronger next hypothesis would target **joint nonlinear read–route–write operations** across
consumers: for example, whether an associative-retrieval model computes one reusable key-matching
rule that can be explicitly factored into multiple routing/value consumers. It should compete
against separate consumer programs, include all arbitrary coefficients/adapters, and predict
held-out factor combinations and independent versus joint source edits. That is a specific
changed object for a future bounded experiment, not a finding from this pilot and not an
unrestricted synthesis campaign already underway.

The handoff's proposed exact-closure principle remains useful. Related observable-preserving
linear reductions appear in [CLUE](https://arxiv.org/abs/2004.11961), but its polynomial-ODE
result is not a theorem that normalized transformers have small semantic circuits.

## Reproduction and receipts

- [Architecture contract](../BILINEAR_RECONSTRUCTION_ARCHITECTURE_CONTRACT.md)
- [Frozen protocol and controlling success update](../BILINEAR_RECONSTRUCTION_PILOT_PREREGISTRATION.md)
- [Reference module](../bilinear_reconstruction_reference.py)
- [Bounded runner](../../bilinear_quotient/ops/run_bilinear_reconstruction_pilot_v1.py)
- [Final machine-readable result, including the rational planted program](../BILINEAR_RECONSTRUCTION_PILOT_V1_RESULT.json)
- [Actual cache/buffer allocation audit](../BILINEAR_RECONSTRUCTION_ALLOCATION_AUDIT.json)

CPU reproduction, with a fresh output location/version if the immutable result already exists:

```bash
PILOT_DEVICE=cpu CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /venv/main/bin/python basis_aligned/bilinear_quotient/ops/run_bilinear_reconstruction_pilot_v1.py
```

The runner refuses to overwrite its result and verifies reference/preregistration hashes before
execution. Any CUDA execution must go through the existing hash-bound `ops/enqueue.sh` workflow.
No checkpoint was modified, no model was trained and no hardware was provisioned.
