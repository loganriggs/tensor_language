# Exact source-edit program v1

Verified native-background intervention executor for the existing attn4-rms-seed0
checkpoint (four attention layers,D128,H4,V29). This is not a smaller learned
semantic model. Its only constant saving is the previously known final O/head fold.

The exported `EXACT_SOURCE_EDIT_V1_PROGRAM.pt` contains config, program weights
and exact fixed positional/mask buffers. It contains no input-dependent state or
teacher response table. The following CPU example computes its own states and
registered edits from tokens, with no original checkpoint access:

```python
# Run from /workspace/tensor_language with the repo and polynomial_causal on PYTHONPATH.
import torch
from pathlib import Path
from exact_source_edit_reference import load_package, make_edits
from source_support_interaction_reference import populations

torch.set_num_threads(2)
root = Path('/workspace/tensor_language/basis_aligned/polynomial_causal')
package = torch.load(root / 'EXACT_SOURCE_EDIT_V1_PROGRAM.pt',
                     map_location='cpu', weights_only=True)
program = load_package(package)
world = populations(seeds=(21909, 21910))['iid'][0]
tokens = world['tokens'][:4]
context = program.prepare(tokens)
edits = make_edits(program.background, tokens, world['masks'], world['heads'])
logits = program.edit(context, edits['disjoint_7'])
assert logits.shape == (4, 51, 29)
```

`prepare` runs the retained prefix and builds the final-reader cache from tokens.
`edit` accepts a B×T×D residual change at the final-layer input; it recomputes live
RMS and correctly positioned Q/K/V features for affected positions. Source changes
update all query messages; changed queries read the complete edited source set.
Overlapping edits are summed before normalization. Changes to earlier producer
states must first be propagated to this boundary; arbitrary earlier-layer edits
are not interchangeable with a final-source delta. The verified domain is cold
51-token IID24-cycle and OODthree8-cycle inputs. Longer histories are untested.

`EXACT_SOURCE_EDIT_V1_RESULT.json` records all17 disjoint/overlapping/half-dose
arms over384query variants from32independent worlds (6528 input/edit combinations).
Maximum all-logit error is1.14e-13. `EXACT_SOURCE_EDIT_V1_CPU_REPLAY.json` records
fresh-process export replay for136example-arms at2.06e-13, with an audit hook
blocking reads from the original checkpoint directory. The native checkpoint was
not needed by the replay process. Test fixtures are separate from the program.

Measured RTX5090 FP64,B4,T51,20 warmed synchronized trials:

| Operation | Median milliseconds |
|---|---:|
| Full native forward with6-position edit |1.841|
| Program preparation |1.882|
| Native cached suffix,2/6 positions |.511 / .503|
| Program source edit,2/6 positions |.972 / .990|

The sparse editor is slower than the cached native suffix at these sizes. Fewer
projection rows and source contractions do not establish a latency win. No best-
baseline speedup is claimed. This implementation is useful as an exact exported
intervention interface; the native cached suffix remains the faster tested option.

Price:387968 independent constants versus400640 native;12672 saving entirely
from generic readout folding, zero new semantic parameter saving. Input-specific
cache1510008bytes; fixed masks/positional tables2334720bytes; package5452045bytes.
Measured incremental edit allocation peaks1.64/1.85MB. Preparation, cache copies,
dense readout corrections and original native prefix/reader weights remain charged.
