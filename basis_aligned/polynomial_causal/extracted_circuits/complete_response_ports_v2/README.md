# Complete conditional response package

`execute.py` needs only PyTorch and `program.pt`. `load()` regenerates the
derived Gram matrices and adapters from six stored primitive tensors.
`project(z, h, u, program)` converts three supplied 1152-dimensional pristine
states into 1225 numbers per token. `execute(ports, amplitude, program)` computes
the conditional head9.8 scalar and its input squared RMS after the fixed head8.2
writer edit. Positive amplitude subtracts the writer; negative amplitude adds it.

The code retains the pristine bias-free MLP8 compensation, both normalized QK
factors, reflected-key even interaction, causal mask and rounded rotary constants.
The supported execution contract is FP64 on CPU, with the native FP32 RMS epsilon.
Positions start at zero; padded tokens must be trimmed before projection/execution.

This is an extraction of the existing rank-64 directional predictor, not an exact
replacement for the native upstream MLP. Native tokens/prefix, pristine `z`, `h`,
`u`, intervention amplitude generation, downstream writer and suffix remain
external. It does not autonomously predict text or establish new OOD, semantic
identification or whole-model compression. All projectors are included in the
serialized program; the 1225 input numbers are not a model-storage count.

The export receipt records isolated-process replay against the prior compiled
implementation on 72 existing contexts and six signed strengths. Those contexts
are development evidence. The manifest hashes all required runtime files and
records the source artifact hash. Cached test inputs are not runtime dependencies.
