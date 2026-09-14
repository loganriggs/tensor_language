# Gerund depth interval V2: fixed-reader reference correction

14 September 2026. V1 completed all 36 registered native bodies and saved the
full MLP17 inputs, final states and row-specific context readers. Its state
instrument is exact, and its norm-moment and CE measurements are valid. Its gate
reference implementation is invalid: it used `tau.roll(-1) - tau`, which rolls
both the state and the row-specific reader. The V1 preregistration explicitly
requires the recipient reader to remain fixed on the cyclic donor state.

V2 performs no native forward and changes no row, arm, state, threshold,
prediction or interpretation. It loads the immutable V1 states and computes

`tau = sum(k_perp * u17)`

`donor_tau = sum(k_perp * u17.roll(-1, 0))`

`gate_reference = donor_tau - tau`.

Each arm's change remains `sum(k_perp * arm_u17) - tau`. Recompute every gate
transfer and relative error, then score the original A--E predicates. Require
the V1 runner/result/artifact and this correction source to match bound hashes.
The complete `pre_mlp17` arm must now have gate transfer 1 and error 0, in
addition to V1's exact state bridge. Any V1 gate verdict is withdrawn; its
native states, norm-moment verdict and collateral CE remain evidence. There is
no threshold rescue and no repeated GPU work.
