# Equality post-MLP9 state factorial V1 execution note

Two pre-receipt implementation issues were caught before adoption.  Neither
changed an experimental arm, example, prediction threshold, or scientific
measurement.

The first execution compared the BF16-rounded post-residual delta with an FP32
distributive expansion of its two component deltas.  BF16 addition is not
distributive after rounding, so this reported `.22214` relative non-additivity
and correctly made the instrument terminal invalid even though native/absent
logits, MLP9 writes, and both installed native boundary states replayed exactly.
The final runner instead checks each captured post-state against the actual
BF16 residual addition.  That closure is exactly zero.  The original `.22214`
quantity is retained separately as `maximum_bf16_delta_nonadditivity_relative_l2`
and is not treated as real-valued component failure.

The next receipt passed all scientific predictions but inherited the dry-run
values `model_loaded=false`, `gpu_accessed=false`, and `queue_touched=false` in
its price block.  It was not adopted.  The final hash-bound rerun sets those
provenance fields truthfully.  Its scientific numbers are identical to both
earlier executions: pre-state recovery `1.0103173`, write-only recovery
`.0286312`, both-fixed and pre-recompute recovery `1.0`, and recovery
interaction `-.0389485`.
