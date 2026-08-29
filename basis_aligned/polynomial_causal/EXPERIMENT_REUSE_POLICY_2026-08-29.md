# Reuse policy for future bilin18 experiments

Date: 2026-08-29

This is an engineering rule for **new experiments**. It does not retroactively
change a preregistered or source-frozen runner.

## Short answer

Yes. A new causal experiment should normally provide only four genuinely new
objects:

1. the candidate component or program;
2. the sites at which it is installed;
3. the masks/cells on which it is judged;
4. the preregistered hypotheses and gates.

Checkpoint loading, model sequencing, arm dispatch, document-wise metric
reduction, source/input binding, create-only publication, and receipt generation
should be shared machinery.

## What the audit found

The scientific forward path is already partly centralized. At least 27 Python
files in `polynomial_causal/` call
`bilin18_observed_model_facade.forward_with_dispatch`. That is the correct
explicit boundary: it owns the residual/RMSNorm sequence, while an experiment
supplies attention and MLP dispatchers.

The largest obvious source-code duplication is outside that forward path:

- 78 files define their own `file_sha256`;
- 13 define their own `logical_sha256`;
- 10 define their own JSON or Torch create-only publisher;
- 36 contain their own `log_softmax`-based scoring path;
- 22 contain their own `torch.quantile`-based interval calculation.

Some of these differ for legitimate preregistered reasons, but most are copies of
the same mechanism with experiment-specific constants embedded in them.

The other project lane measured the performance consequence directly. Factoring
its repeated program construction and arm scoring into `ops/bqlib.py`, then
caching a score by exact program/role/arm identity, reduced an identical workload
from 267.7 seconds to 52.1 seconds. Model loading and base-table construction were
only about 8.3% of runtime. The important savings came from avoiding repeated
solves, SVDs, native forwards, and identical arm scoring.

## The call hierarchy we should use

### 1. Every authoritative model forward

Call:

```python
bilin18_observed_model_facade.forward_with_dispatch(
    model, tokens, attention_dispatcher, mlp_dispatcher
)
```

Do not write another manual 18-block residual loop and do not use untracked
forward hooks for an authoritative physical-intervention result. The explicit
dispatch function makes the intervention site, native calls, residual order, and
RMSNorm boundary inspectable.

Hooks remain acceptable for quick observation-only discovery when they do not
make a physical replacement claim. A result promoted to the strict ledger should
be replayed through the explicit dispatcher or a fully owned tensor program.

### 2. Every replacement component

Represent it as an owned callable/module with:

- a constructor from frozen native weights and fitted objects;
- one `forward(state)` computation;
- a storage/operation price receipt;
- an exact replay or identity test;
- no hidden call to the native component it claims to replace.

The present example is `PhysicalRetainedBilinearMLP` in
`mlp2_cmr_v1_physical_program.py`. This pattern should become a generic physical
component protocol for MLP, attention, table, affine, and tensor replacements.

### 3. Every multi-arm causal comparison

Use a shared arm runner whose key is conceptually

```text
(checkpoint hash,
 dataset-role hash,
 source-closure hash,
 component-program hash,
 intervention plan,
 metric/cell specification)
```

The intervention plan is a declarative map from a site to one of:

```text
native | zero | owned replacement | signed edit | ablation-plus-replacement
```

The runner should build or factor each mathematical object once, run the native
baseline once per role, and reuse an arm's sufficient statistics when this exact
key already exists. A cache mismatch must be loud; a partial key must never be
treated as a hit.

`mlp2_cmr_v1_validation_runtime.forward_arm` is the current narrow version of
this idea. The next implementation should generalize the *plan and call ledger*,
not copy its MLP2-specific arm names.

### 4. Every outcome measurement

Use one streaming document reducer. It should consume native and candidate logits
batch by batch and retain only per-document sufficient statistics:

```text
count,
native and candidate NLL sums,
teacher KL sum,
squared-error and native-energy sums,
top-1 counts,
support hash.
```

The experiment supplies masks such as copy-positive, capitalization, numeric,
syntax, or frequency bins. The reducer supplies the arithmetic, validation,
packing, prefix stability, and document-cluster bootstrap. This prevents each
new semantic circuit from reimplementing CE, KL, normalization, and confidence
intervals.

The current reusable ingredients are
`mlp2_cmr_v1_validation_statistics.py` and
`terminal_copy_streaming_statistics.py`; they should be merged only after the
MLP2 validation is preserved, because their cell schemas and frozen bootstrap
contracts currently differ.

### 5. Every authoritative publication

Use one lifecycle library for:

- file and canonical-object hashing;
- committed-and-pushed source closure;
- exact parent-artifact joins;
- exclusive run locks;
- one-use execution capability;
- create-only atomic JSON/Torch publication;
- result/ledger/receipt/failure ordering;
- terminal semantic replay.

This is the most duplicated code, but not the main GPU bottleneck. It is still
worth factoring because mistakes here have invalidated runs. The best existing
implementation pieces are in `tensor_bilin18_tangent_authority.py` and
`early_mlp_context_cross_v1_lifecycle.py`. New runners should import a tested
generic library rather than copy either file.

## What should be cached

Cache when the exact key above is stable:

- native per-document sufficient statistics for a sealed role;
- candidate per-document sufficient statistics for an identical physical arm;
- decompositions that do not depend on the evaluation role, such as one SVD,
  HOSVD basis, ridge solve, or retained-product selector;
- fixed cell masks and support hashes;
- owned-program materializations and price receipts.

Do not cache only by a friendly arm name such as `rank512`. Rank is not enough:
weights, fit role, gauge convention, correction/bias, downstream background, and
metric support can all differ.

Raw full-vocabulary logits should usually not be the persistent cache. They are
large and create privacy/provenance burden. Per-document sufficient statistics
are much smaller and are enough for CE, KL, accuracy, prefix checks, cell effects,
and clustered bootstrap. Cache response vectors only when a later mathematical
analysis genuinely needs vector geometry.

## What not to factor yet

- Do not rewrite the current frozen MLP2 CMR validation around a new lifecycle
  abstraction. Finish and preserve it first.
- Do not force different scientific cells into one fixed ontology. Share the
  reducer; pass the masks as data.
- Do not hide component-specific algebra behind a generic interface so thoroughly
  that we cannot audit the tensor contraction or price.
- Do not import `bilinear_quotient/ops/bqlib.py` as the authoritative physical
  intervention layer. Its cache and arm-grammar ideas are valuable, but its
  hook-based context-free-table workflow answers a different class of questions.

## Ordered implementation plan

1. Finish and preserve the currently frozen MLP2 finite validation.
2. Extract a generic, tested lifecycle core without altering old receipts.
3. Extract a declarative intervention plan plus exact per-site call ledger around
   `forward_with_dispatch`.
4. Extract a mask-parameterized streaming metric reducer and clustered bootstrap.
5. Add exact-key caching of decompositions, native baselines, and arm sufficient
   statistics.
6. Add a sub-five-second CPU test suite that exercises cache-key sensitivity,
   intervention polarity, bias/correction inclusion, nonempty cells, price replay,
   source closure, and receipt-last ordering before any GPU launch.

After this, a normal new circuit experiment should be a short specification and a
candidate implementation, not another several-hundred-line model runner.
