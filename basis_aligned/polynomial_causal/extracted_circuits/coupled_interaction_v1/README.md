# Fixed-context coupled interaction

`load()` reads the included program; `execute(program,a,b)` returns post-MLP9,
attention10 write, and post-attention10 residual states. Mixed strength equals
`a*b`. Runtime requires only PyTorch and these files; it reads no checkpoint.

This program specializes one recorded recipient/donor context. Original model
weights and native prefix computations were needed to prepare it. It does not
accept arbitrary text. MLP10 and the remaining suffix are external. The explicit
x0 frame is recomputed on CPU from weights; see manifest provenance. No new
semantic circuit, global parameter saving or runtime speedup is claimed.
