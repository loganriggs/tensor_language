# Terminal-copy fit-mean recovery v3 preregistration

Status: **prospective engineering recovery; scientific protocol unchanged**.

V2 passed its independent audit, froze a source-closed authority, and began the
authorized collection only after the GPU was free. It failed before accepting the
first batch because the outcome-blind final-state integrity hash called NumPy on a
CPU `bfloat16` tensor. NumPy does not expose that scalar type. V2 published no bank,
result, manifest, or success receipt; its authority and terminal failure are bound.

V3 changes only how `tensor_sha256` reads the already detached, CPU-contiguous tensor
payload. It views the same storage as `uint8` before converting to NumPy bytes. The
hash continues to include the original tensor dtype and shape, so this is an exact
raw-byte integrity representation, not a cast or numerical transformation. Tests must
show equal hashes for identical CPU/CUDA `bfloat16` values and a different hash after
one representable-value mutation.

All scientific and numerical choices remain frozen: the same sanitized 192-document
fit input, six heads, 18 native layers, bfloat16 physical writes, documentwise CPU
float64 accumulation, float32 runtime means, L8 pair arithmetic, call census,
protected parents, outcome denial, and receipt-last lifecycle. V1 and v2
authority/failure bytes are permanently bound. All v1 and v2 success outputs and
locks must remain absent through every publication gate. V3 requires a new
independent audit, source-closed authority, and fresh empty namespace.
