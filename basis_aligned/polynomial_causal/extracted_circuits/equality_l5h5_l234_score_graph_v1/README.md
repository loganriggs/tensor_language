# Equality L5H5 L2+L3+L4 score graph V1

This correction-free package implements the frozen sparse source boundary found
on natural text and validated on code OOD. It merges three named native write
ports, runs the exact L5H5 residual/QK/rotary/bilinear score computation, and
returns a causal score tensor.

It has zero learned parameters and reuses four native `128 x 1152` L5H5 weight
slices. Measured code score error/cosine is `.15255/.99234`; downstream equality
recovery is `.90438`, noncopy damage is `.00157` nat, and the full three-source
Möbius graph closes at `2.73e-9`.

This is a boundary extraction, not a recursive implementation of the L2/L3/L4
writes or compression of the native Q/K weights.
