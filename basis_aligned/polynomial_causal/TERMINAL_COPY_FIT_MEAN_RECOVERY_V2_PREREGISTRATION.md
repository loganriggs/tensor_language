# Terminal-copy fit-mean recovery v2 preregistration

Status: **prospective engineering recovery; scientific protocol unchanged**.

The v1 fit transaction failed before accepting its first document because
`assert_matches_native` compared the owned CUDA copy of the plain Rotary `inv_freq`
attribute with the native CPU attribute using device-sensitive `torch.equal`. The
v1 failure published no bank, result, manifest, or success receipt.

V2 changes only the identity comparison: tensor dtype and shape must match, and exact
values are compared on CPU when the source and owned tensors reside on different
devices. Projection weights already on the same device retain direct exact comparison.
The test reproduces production's unregistered CPU Rotary attribute beside CUDA
projection parameters and must also reject a one-value mutation.

All scientific and numerical choices remain frozen: the same sanitized 192-document
fit input, six heads, 18 native layers, bfloat16 physical writes, documentwise CPU
float64 accumulation, float32 runtime means, L8 pair arithmetic, call census, protected
parents, outcome denial, and receipt-last lifecycle. V1 authority/failure bytes are
permanently bound. V2 needs a new independent audit, source-closed authority, empty
namespace, and may not reuse or overwrite any v1 output.
