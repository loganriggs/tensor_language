# Complete standalone bilin18 tensor-program identity

Date: 2026-08-28

The exact tensor program now owns the complete model boundary. It stores independent
copies of the embedding, residual lambdas, all attention and MLP tensors, unembedding,
and directly executes every RMSNorm and the output softcap. The checkpoint model is
garbage-collected before the program is evaluated.

The role-free gate passes:

- base and prefix-intervention logits have maximum absolute error 0;
- native/program SHA256 hashes match for both fixtures;
- both covered synthetic CEs equal 12.686808586120605;
- changing token position 32 while holding all downstream current tokens fixed changes
  later native logits by a maximum 3.497422933578491, and the standalone program
  reproduces that contextual effect exactly;
- native and program storage pointers are disjoint;
- the program contains no native checkpoint module references and makes zero native
  calls after construction;
- the checkpoint model object is collected before either program forward.

Complete stored-value accounting is 545,904,054 float32 values:

- shell: 115,900,452;
- attention bank: 143,328,402;
- bilinear MLP bank: 286,675,200.

This closes exact executable ownership at 100%. It does not increase semantic
explanation or strict simplified whole-model recovery: the program is a dense change of
ownership, not a compression. Its value is that every future simplified attention/MLP
candidate can now be inserted into a single contextual program and judged by complete
CE, causal transport, storage, operations, OOD behavior, and editing consequences.

The first already-available compressed point is shared-QK-384 attention with dense MLPs.
It would store 490,165,686 values, saving 55,738,368 values (10.21%) relative to this
exact reference. It must now be evaluated through the standalone program; the previous
99.44% figure is an attention-stake result, not complete-model recovery.
