# Terminal-copy selection input exposure erratum

Status: **preserved engineering exposure; no E4 model outcome occurred**.

At 2026-08-29 11:34 UTC, before a selection execution authority existed, an
engineering inspection deserialized the frozen `selection_natural.pt` payload to
identify its schema. The process printed only top-level and nested field names,
container lengths, and tensor shapes/dtypes. It did not print or analyze token IDs,
document identities, mask values, pair indices, synthetic token values, candidate
outputs, logits, losses, causal effects, or bootstrap results. No checkpoint was
loaded and no model forward occurred during this access.

The payload contains copy-cell masks, so a claim that the label-bearing selection
container was never opened pre-authority is false. Any later selection authority must
bind this erratum and explicitly state whether the role remains eligible under the
narrow rule “no selection value or model outcome was observed.” It must not silently
claim pristine container secrecy. Prefer a separately audited, source-closed loader or
projection whose schema is fixed without exposing values to the caller, and keep all
selection reductions inside the owned transaction.

This event grants no E4 evidence or ledger credit and does not authorize loading the
final or OOD roles.
