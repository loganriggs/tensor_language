# Native extraction of the rank-one head8.2 city edge

On the frozen96-row four-template union, capture native block8 attention inputs
and first values. For every row, compute the inherited donor-minus-recipient
city value and recipient head8.2 city routing. Compare the native direct form
`c_proj8(routing * delta_value)` with the factored form
`routing * c_proj8(delta_value)` at every destination. Independently reconstruct
the inherited value delta from normalized token embeddings and block0 head2
`c_v`. Export only block0 head2 `c_v`, block8 head2 `c_proj`, and block8 mixture
lambda; caller routing and normalized token embeddings remain explicit ports.
Exactly96 native body forwards, under180 seconds, no fitting.

- `pred_a`: captured native head8.2 sources replay within `1e-5`; direct and
  rank-one writes agree within `1e-5`; block0-weight reconstruction of inherited
  deltas agrees within `1e-5`; values are finite and exactly96 forwards execute.
- `pred_b`: the bound parent city-value, fresh-chain, and destination-role
  receipts load and retain their recorded verdicts without modification.
- `pred_c`: for every observed context length, the conditional `T+1152` runtime
  interface saves at least90% versus a dense `T*1152` write, and factor-first
  projection saves at least90% of the corresponding projection/scale multiplies.

The price includes the two exported128x1152/1152x128 maps. It excludes caller
routing and normalized embedding generation, block9 O, and the native suffix.
This is conditional extraction of one edge, not a full model or corpus-OOD claim.

