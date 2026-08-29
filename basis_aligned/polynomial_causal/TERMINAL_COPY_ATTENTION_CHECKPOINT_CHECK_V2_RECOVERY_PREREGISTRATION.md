# Terminal-copy attention checkpoint check v2 recovery

V1 is spent with authority SHA256
`15c68bfaff6362300fa680a60ce14077bea43142f135ff4a6c809b5341ce2b5c` and
failure SHA256
`fdb947c9c557f23e819050a007bbf6d5cfc19039ce195aedcd47a6aadcbc9138`.
It found two distinct issues before any behavioral rows existed:

1. the adapter requested the transposed value layout directly from einsum, whereas
   the checkpoint forms `[batch,head,query,d_head]` and then transposes/contiguously
   materializes it before `c_proj`;
2. separately projecting and adding nine bfloat16 head writes changes the GEMM
   accumulation order. Its observed relative error was `0.00263--0.00267`.

V2 makes only the corresponding engineering corrections. The adapter preserves the
checkpoint contraction/layout order. The unpartitioned adapter write and value bus
must still be bit-identical to native. The decomposition tolerance is replaced by the
non-data-dependent bfloat16 summation allowance `relative <= 0.01`; maximum absolute
error is descriptive because it scales with arbitrary random-state magnitude. The
0.01 bound is below the simple nine-term worst-case bound and well below the size of
the candidate ablations that a future behavioral screen must establish. V2 must still
report the raw maximum error and may not use this tolerance as behavioral evidence.

All checkpoint, seed, shapes, layers, price accounting, source closure, create-only
publication, and scientific no-claim boundaries remain unchanged. V2 binds the exact
V1 authority and failure as committed source inputs. A V2 pass closes only the owned
checkpoint-formula implementation gate.

