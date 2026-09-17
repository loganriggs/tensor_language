# Exact typed-face native weight closure V1

Phase start recorded in AGENT_BOARD. No new support selection or fitting.
On the already opened 96-row replication panel compare native, original corner5,
and compiled corner5: 288 forwards, at most 300s, one managed GPU job.

Compiled head8.2 program stores both QK pairs, current-value map, output map,
mixture, and a token-only first-layer value table for the eight registered city
tokens. Inputs: normalized recipient block8 state, donor state at the city,
recipient/donor city token IDs, city index, destination mask. Native input
arrays decrease from five routing/value tensors to two state tensors; tokens
and masks remain explicit. No fitted state proxy. Recipient queries stay fixed;
both QK factors change together; recipient current value stays fixed. All
three routing/inherited self/cross contributions remain in the closed face.
Block9 reentry, odd-value propagation and full suffix remain native/external.

pred_a: token table matches captured inherited city values <=1e-5 relative L2;
compiled write vs original corner5 <=1e-5 relative L2 and outside-mask zero.
pred_b: compiled versus original recursive logit-effect relative L2 <=1e-4
for all five readouts; no claim of new target capability or selectivity.
pred_c: native and original corner5 match saved parent within 1e-4 max absolute
logit margin. Failure invalidates parent replay, not evidence against the path.
Null: state/normalization/rotary/rounding mismatch prevents exact port closure.
Report literal scalars and package bytes; do not claim runtime improvement.
No independent-composition, random-split, or equal-norm-removal gate is included.
These remain necessary and unpassed. Eight-city-token vocabulary is explicit;
unknown token IDs must raise, never silently map to a known city.
