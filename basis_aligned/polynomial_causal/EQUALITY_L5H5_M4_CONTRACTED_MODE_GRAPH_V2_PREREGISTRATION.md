# Equality L5H5 M4 contracted-mode graph V2 precision correction

V1 is an invalid instrument receipt.  Its CUDA float32 SVD reconstructed the
weight-only contracted operator `W D` at `1.259e-4` relative L2, outside the
prospective `2e-5` bridge gate.  Its apparent rank-256 selection, transfer,
causal recovery, and composition failure are not adopted as scientific facts.

V2 changes only SVD construction and bridge certification: form `W D`, compute
the thin SVD, and measure its reconstruction in float64; cast the resulting
frozen right-singular vectors to float32 for the exact same deployed mode
executor.  Candidate ranks, natural/code rows, score and behavior arms,
selection rule, compact rank ceiling, causal/selective controls, composition
split, thresholds, and price remain exactly those preregistered for V1.

If the float64 bridge still exceeds `2e-5`, V2 is invalid.  Otherwise all
remaining failures are interpretable scientific nulls; in particular, rank
above 64 or composition error above `.10` cannot be relabeled positive.
