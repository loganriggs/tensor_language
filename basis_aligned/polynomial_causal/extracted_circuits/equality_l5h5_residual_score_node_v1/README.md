# L5H5 residual-to-score node V1

This node moves the extracted boundary from four normalized/rotary Q/K
activation tensors to one pre-attention residual tensor. It explicitly executes
the residual RMS, four native head-specific linear projections, Q/K RMS,
rotation, bilinear product, and causal mask.

It learns no parameters. The four `128 x 1152` weight slices are frozen model
parameters, so this is an executable extraction and port reduction rather than
weight compression. `merge_sources` supports an additive FP32 residual-
provenance graph with an explicit BF16 roundoff-correction port.
