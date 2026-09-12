# Regional shared-component branch: portable FP32 package

This executes the weight-derived shared-component branch inside bilin18
attention17/head2. It is **not the entire attention head or a token-to-logit model**.
The package contains666,656 float32 scalars (2,669,957 serialized bytes) and a
runtime requiring only PyTorch and Python's standard library.

## Inputs and output

| Argument | Shape | Meaning |
|---|---|---|
| `query` | `[N,1152]` | Native normalized attention17 query input |
| `source` | `[N,2304]` | Concatenation of current normalized attention17 source and normalized block0 attention input at that source position |
| `rotation` | `[128,128]` | Actual relative rotary map for this query/source position, preserving the model's rounded convention |
| `p` | dictionary | Tensors loaded from `program.pt` |
| returned write | `[N,1152]` | This component's residual-stream contribution from one source position |

For a query position, sum calls over its causally allowed source positions.
Causal masking and rotary-map generation are the caller's responsibility. The
first-value input is not a raw token embedding; it includes the actual initial
embedding normalization and block0 residual reentry/normalization.

```python
from pathlib import Path
import torch
from execute import execute_head

folder = Path(__file__).resolve().parent
p = torch.load(folder / "program.pt", weights_only=True, map_location="cpu")
# query, source, rotation must already be supplied at the interface above.
with torch.no_grad():
    component_write = execute_head(query, source, rotation, p)
```

Use float32 inputs for the exported FP32 package. Device movement is explicit:
all inputs and program tensors must be on the same device. Native GPU research
on this shared instance continues to use the managed runner.

## Meaningful local interventions

The two child readers share a quadratic source parent. To remove child0 from
the extracted feature computation, clone `p["children"]`, zero row0, and pass
that modified dictionary to `execute_head`. Other tensors and original input
states stay fixed. This is an internal feature intervention, not a source-token
edit; it deliberately leaves the native query/key normalization unchanged.
The remaining child writes add linearly at this interface. Their effects after
the native final layer need not add, because that layer is nonlinear.

## Evidence and limits

[Native integration](../../COMPILED_SHARED_HEAD2_NATIVE_V1_RESULT.json) tested
48 geographic prompts across four families. FP64 matches the original branch
writes within2.97e-15; FP32 within1.65e-7. FP32 signed-prefix removal-effect
error is at most8.29e-6. FP64 child branches match their original factor
references within1.25e-14 and sum to the full write within1.60e-16. Their measured
final-output interaction is0.112–0.159%of the joint effect.

Separately, keeping this head2 branch approximates the larger nine-consumer
shared component's removal effects within0.404–0.434%on the tested prompts.
The full component's removal reduces regional cue gaps by12.5–14.9%there.
These are distinct comparisons; neither establishes universal behavioral fidelity.

The [manifest](manifest.json) verifies exported tensors equal the FP32 tensors
used by the validated code, the runtime function's AST is unchanged, and a
round-trip execution matches. The initial export hit disk exhaustion; only the
regenerable npm cache and the incomplete owned output were removed before the
successful retry. Model/research files were retained.

Upstream state generation, final MLP/RMS/unembedding, rotary generation, broad
selectivity and interaction with other extracted components remain outside this
package. [Full derivation and price](../../COMPILED_SHARED_HEAD2_V1_MATH.md).

**FP32 child-intervention precision limit:** later source-position testing found
a small child reading for which the old FP32 package has2.51e-4relative write
error. Full-component FP32 results above remain valid. Tight FP64 child results
do not establish the same tolerance for every FP32 child edit. See the
[precision diagnosis](../../COMPILED_TOKEN_SHARED_HEAD2_V1_MATH.md).
