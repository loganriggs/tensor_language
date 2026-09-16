# Equality L5H5 M4 shared-projection gauge V1

The first 12-projection kernel is lawful and preserves baseline/joint scores,
but code interaction error `.10063` narrowly misses the frozen `.10` gate.  Its
chosen algebra projects baseline plus child/remainder deltas and derives the
joint numerator, implicitly assigning BF16 residual-rounding mismatch to the
joint cell.  This prospective positive red-team tests whether that coordinate
choice manufactured the null.

For each gauge, project three exact residual numerators and derive the fourth
from `N_joint = N_child + N_remainder - N_baseline`:

- omit/derive baseline;
- omit/derive child;
- omit/derive remainder;
- omit/derive joint.

Each gauge costs exactly 12 Q/K projections and carries all four exact RMS
denominators.  On natural rows, select the gauge with lowest authority-relative
interaction error among gauges whose baseline and joint errors are at most
`.01`; tie-break in the order above.  Freeze it for code.

Gates: lawful inherited graph; selected natural and frozen code interaction
error at most `.10` with cosine at least `.95`; both code-half interaction
errors at most `.15`; baseline/joint errors at most `.01` and cosine at least
`.999`; graph closure at most `2e-6`; 12 versus 16 projections; zero learned
parameters.  Failure is a valid shared-projection gauge null.  Passing
authorizes behavioral replay/removal testing without changing any scientific
graph node.

Price: one checkpoint load, 192 natural and 192 code prefix documents, four
12-projection gauges, no behavior forwards, fits, gradients, updates, or text.
