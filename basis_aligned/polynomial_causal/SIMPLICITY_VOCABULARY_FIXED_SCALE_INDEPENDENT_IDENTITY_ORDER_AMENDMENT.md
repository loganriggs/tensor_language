# Rung456b amendment: JSON identity order only

The first managed rung456 attempt stopped before model loading or independent outcome access because rung455's
create-only JSON was written with sorted dictionary keys, while the consumer required its `arms` mapping to retain the
producer's insertion order. All23 names were present; only their serialized order differed.

Preserve the first runlog as
`simplicity_vocabulary_fixed_scale_independent_first_identity_order_invalid.log`. Change only the fixed-spec arm
identity check from ordered tuple equality to set equality. Keep the original rung454 result's ordered arm check,
subsequent scoring order `IDS`, every source/input hash, row role, candidate, fixed-scale formula, intervention,
prediction, bar, null, and routing unchanged. Recommit, re-gate, and rerun through the managed GPU queue. This repair
opens no result and has no scientific decision authority.
