# Temporal/is-was shared-eight ordered conditional deletion — preregistration

The scorer-corrected pairwise Möbius pilot showed sparse large interactions but falsified an executable
degree-two truncation. This run therefore measures each module in three causal contexts: alone, added
after all earlier shared modules, and removed from the full shared bank. The fixed layer order is
`MLP0, MLP1, MLP2, MLP3, MLP4, MLP6, MLP7, MLP8`.

The unique response arms are empty, eight singletons, all ordered prefixes, eight full-context
leave-one-out sets, and the full set. Every leave-one-out arm is also run on frozen controls. No
parameters, projectors, modes, thresholds, or order are fitted.

## Gates

- A: all authorities match, splits remain disjoint, values are finite, and the full shared bank retains
  response projection at least `.8` with RSE at most `.2` and behavior at least `.75` on both tasks.
- B: at least one leave-one-out arm remains functionally feasible under those same task bars.
- C: at least one leave-one-out arm is also selective: median control KL at most `.02` and zero top-one
  flips.
- D: the early ordered prefix `MLP0-4` is functionally sufficient on both tasks.
- E: at least five of eight modules are context dependent on each task: either their ordered-prefix or
  full-context marginal signed projection differs from their singleton signed projection by at least
  `.10`.

Among selective arms, select lowest median KL, then highest minimum task response projection, then
canonical removed-module order. If none is selective but functional arms exist, select the lowest-KL
functional arm as a diagnostic frontier, explicitly not a released circuit. The next greedy step may
start from that frontier because task function is preserved, but it must continue optimizing controls.

Maximum price: 40 forwards, zero backward passes, zero model or fit updates.
