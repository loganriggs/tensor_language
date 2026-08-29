# MLP2 CMR v1 SUFFIX v2 recovery ruling

Frozen after the v1 failure and before any v2 authority, checkpoint, model, target,
logit, or response access.

V1 stopped on the first MLP2 call before completing a model forward. It opened no
categorical targets, suffix responses, selector scores, validation, or replication.
The failure was numerical and diagnostic: `centered_dual_write` used the fused
`F.linear(product, down, bias)` operation, while the checkpoint's exact native code
uses `Down(product) + Down_bias` as two operations. Those expressions are algebraically
equal but need not round identically in bfloat16, so the frozen bit-exact baseline
guard rejected the intervention.

V2 makes exactly one scientific/implementation change: compute the native baseline
as `F.linear(product, down) + bias`, matching the checkpoint's operation order. Probe
seeds, documents, masks, centering, dual leaves, derangement, context balancing,
rank/ridge rule, supports, gates, call budgets, and publication boundaries are
unchanged. V1 authority and failure hashes are protected parents of v2. Reusing
`FIT_SELECTOR` is permitted because v1 failed before logits, targets, responses, or
selector outcomes existed.

