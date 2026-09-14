# Shared coupled interaction executor

Load the bank with `load()`. Call `execute(bank, context, a, b)` to obtain the
post-MLP9 state, attention10 write, and post-attention10 state. One bank serves
different context lengths. `contexts.pt` contains two examples; callers may
supply other context programs with the declared interface. Mixed strength is a*b.

Runtime needs PyTorch and explicit bank/context data. It reads no checkpoint.
The full original model was needed to generate the context programs; those
preparation dependencies have not been compressed away. MLP10 and the later
suffix remain external. The long example uses CPU-reconstructed x0; see manifest.
This is conditional extraction and reuse, not a complete language model or a
new semantic circuit. Runtime casts matrices to FP64; no speedup is claimed.
