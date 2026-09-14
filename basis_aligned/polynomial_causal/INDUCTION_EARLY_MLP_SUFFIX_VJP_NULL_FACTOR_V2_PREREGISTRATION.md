# Induction early-MLP suffix-VJP-null response factor V2 cache correction

V1 failed during the first reverse-mode forward before any result or scientific outcome: native capture had populated each attention rotary cache with inference tensors, which PyTorch refuses to save for backward. V2 clears only `seq_len_cached`, `cos_cached`, and `sin_cached` after native capture so the identical rounded rotary tables are regenerated as ordinary tensors on the gradient forward. Rows, targets, MLP8-12 group, CE-VJP axes, arms, bars, price, and no-fit/no-update/no-quantization constraints are unchanged.
