# Packed sparse reflection-even producer pair

A portable **conditional component** for the previously identified head8.2/head9.8
regional producer pair. Head8 is unchanged; head9 uses the frozen weight-only
composed sparse reader. This package needs PyTorch and the three local files
`execute.py`, `program.pt`, and `manifest.json`. It does not load the original
model checkpoint or import project research helpers.

## Interface

Optional shared runtime (14 September): `shared_execute.py:load_shared` reuses
one basis read across both QK consumers and caches adapters. See
`shared_manifest.json` for its three required files and hashes. Load with
`runtime = load_shared("program.pt", device="cpu")`, then
`runtime.scalar(current8, tokens, index=0)`. This preserves the original program;
isolated loading replays exactly. Tested local CPU speed improves1.12–1.72×,
at256KiB additional resident memory, without reducing serialized weights.
The original manifest remains unchanged and covers only the original entry point.

From this directory:

```python
from execute import load_program, scalar

program = load_program("program.pt", device="cuda")
# current8: actual normalized attention8 input, float32 [batch, sequence, 1152]
# tokens: corresponding int64 token IDs [batch, sequence]
amplitude8 = scalar(current8, tokens, program, index=0)
write8 = amplitude8[..., None] * program["writers"][0]
```

`scalar(..., index=1)` similarly takes the actual normalized attention9 input and
returns its scalar amplitude. Both writes have shape `[batch, sequence, 1152]`.
For physical joint removal, subtract `write8` at attention8 in the native output
dtype, execute the intervening native computation, and then compute/remove the
head9 write from that **changed** attention9 input. A pristine attention9 input
would specify a different intervention. `execute(current8, current9, tokens,
program)` returns both writes when the caller already has the corresponding
actual contexts; it does not generate the second context from the first edit.

The first producer uses its stored first-token value lookup; the second uses its
stored current-value reader. Full QK maps, original Q/K normalizers, native
rounded position tables, causal masking and physical writers are retained. This
is reflection of the key numerator, not reflection of the entire source state
or recomputation of its normalization after reflection.

## What is stored and reconstructed

`program.pt` stores the required maps/lookups/writers, exact head8 basis, and the
head9 sparse matrix as an occupancy bitmask plus FP32 nonzero values. It omits
the original dense head9 basis. Cloning retained tensors prevents a sliced tensor
from accidentally serializing an omitted larger backing storage.

The loader reconstructs the corrected orthobasis on CPU:

$$
Q=S\operatorname{chol}(S^TS)^{-T},\qquad QQ^T=S(S^TS)^{-1}S^T.
$$

It then moves all runtime tensors to the requested device. This preserves the
validated rounding path but expands a dense basis in memory. The package has no
runtime-memory or speed advantage over dense execution; the separate CSR trial
was slower and used more resident memory on CPU.

## Evidence and limits

The frozen candidate passes corpus-level unit-parent preservation on40development
and96additional exact-prefix-disjoint documents, signed hierarchy aggregate
checks, and independent child/remainder composition checks. Substituting it into
the original selective pair preserves regional removal/donor and newline control
bars on the earlier72regional/32newline confirmation panel. Individual relative
errors and small control sign reversals remain recorded. None of these panels
is claimed to be historically untouched or pretraining-disjoint.

The manifest records exact CPU package-versus-candidate replay on independent
head8/head9 probes and both cached head9 contexts, plus hashes and byte counts.
Native package execution is recorded separately in
[SPARSE_PAIR_PACKAGE_NATIVE_V1_RESULT.json](../../SPARSE_PAIR_PACKAGE_NATIVE_V1_RESULT.json): all five predicates pass, and regional/newline scores exactly match the prior validated candidate. An isolated Python process outside the repository also loads and executes both heads using only execute.py/program.pt plus PyTorch.

Actual serialized file size is5,972,357bytes versus6,340,557bytes for the original
mixed-precision package. That difference includes precision and serialization
choices. The separately priced common-FP32 head9 parent-interface saving is
1.806%, including a conservative correction charge; it is not the full-model
compression ratio.

Native contextual input generators, intervening blocks, other-head background
and the downstream suffix remain external dependencies. This is not an autonomous
token-to-logit model, a new semantic-circuit discovery, or proof of the complete
project goal. See [the primary derivation and receipts](../../DIRECT_PARENT_KEY_COMPRESSION_V1_MATH.md).
