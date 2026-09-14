# ATTENTION5_MEAN_DEVIATION_HEAD_SPLIT_V1 preregistration

The `module.attention.5` dossier records a nearly fixed whole-block write: its
top direction contains98.1% of observed energy, a constant write costs about
.119 nats against2.20 nats for deletion, and heads5/7 retain most gross value.
The later rank-one test also found that retaining the native per-position gain
improves over a constant by only.0022 nats. The unresolved input-dependent
object is therefore the deviation around the fixed mean, not another fitted
rank or scalar-gain model.

Fit one output-space mean for each native head on the same frozen24-document
fit split used by the old direction-identity experiment. On the disjoint frozen
24-document evaluation split, decompose the exact block write as

`mean_full + sum_h (projected_head_h - mean_head_h)`.

Run native, mean-only, heads5+7 deviations, other-seven deviations, all-nine
deviations, and nine singleton-deviation arms. No head, rank, direction, corpus,
or threshold is selected from new outcomes.

- **A — instrument and closure:** manual native CE matches the model module on
  the same chunk within.01 nat; the sum of nine projected head contributions
  matches native attention5 output within`1e-3` max absolute error; restoring
  all deviations matches native CE within`.005` nat.
- **B — live input dependence:** mean-only damage is at least`.08` nat.
- **C — prospective heads5/7 sufficiency:** restoring only heads5/7 deviations
  recovers at least50% of the mean-only CE damage.
- **D — prospective other-seven insufficiency:** restoring only the other seven
  deviations recovers at most50% of the mean-only CE damage.
- **E — stable head identity:** heads5 and7 are both among the three singleton
  deviations with the greatest CE recovery, and the heads5/7 versus other-seven
  recovery ordering has the same sign in both fixed12-document evaluation
  halves.

Pass C/D/E would localize the valuable input-dependent complement to the same
two heads that carry most gross attention5 value and license an exact routing /
value / source factorial inside those heads. Failure is informative: it would
separate the nearly constant write from the heads responsible for its costly
input-dependent correction and redirect the next factorization to the observed
singleton leaders. This experiment does not rediscover attention5's induction
role, treat energy as causal evidence, install a surrogate, or claim that a
mean-plus-head interface is a complete model circuit.
