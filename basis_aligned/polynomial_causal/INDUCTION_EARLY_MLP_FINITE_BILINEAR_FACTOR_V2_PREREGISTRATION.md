# Induction early-MLP finite bilinear factor V2 closure correction

V1 executed the frozen 24-forward intervention panel and serialized an invalid
result because its independent finite-bilinear identity check accumulated
FP32 error of `3.94975e-10`, above the frozen `1e-10` bar. V2 changes only that
independent identity check: cast the already captured FP32 normalized states
and immutable MLP weights to FP64, then recompute the three-term algebraic
closure. The causal response vectors installed into model writes remain the
same FP32 tensors. Rows, DISCOVERY/CONFIRM split, arm order, selection rule,
behavioral bars, price 24/768, and the exclusions on fits, gradients, updates,
gains, subgroups, threshold changes, and quantization are unchanged. V1's
invalid outcomes cannot select or modify any V2 choice.
