# Sixth residual source: valid gain, incomplete selectivity

The corrected native experiment passes instrumentation and finite-effect
prediction. Six sources pass joint strength/selectivity in45/48 cells versus
42/48 for the unchanged five-source optimizer. All three remaining failures
are target strength. Both panels are opened; this is not new OOD validation.
The original invalid run remains preserved with its separate numerical audit.

The sixth source closes the omitted late contributions at the nominated site:
attention8, attention9, MLP9 and attention10, plus the declared rounding gauge.
The first five edit directions remain their original float32 values promoted
to float64; the complement is the exact raw-state difference minus their sum.
Native execution sums the edit and baseline in float64, then casts the state
once. Cache, x0 and all other positions retain baseline values. This does not
equal a full embedding edit or close the whole causal effect of that edit.

Old derivative subblocks replay to2.22e-16; native/reference output differences
are at most6.55e-6. All original thresholds are unchanged. The run uses36prefix,
216double suffix and120native suffix evaluations,96gradient and576Hessian-row
reverse calls. Six-source dense quadratic coefficients cost108values per
context versus80 for five sources, excluding native generators and optimizer.
The added capacity is charged; this is not a simpler circuit yet.

## Negative-result redteam: an upper bound on the selector

An actual search for McCormick's factorable-programming work and the opened
[1975 primary report abstract](https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/ADA009223.xhtml)
motivated a convex outer relaxation. Only the abstract was opened; the following
inequalities and their application are derived directly, not attributed as a
newly verified theorem about our model.

For amplitudes a in[-1,1]^6, introduce Xij to represent ai*aj. For i!=j impose
Xij>=ai+aj-1, Xij>=-ai-aj-1, Xij<=1+ai-aj and Xij<=1-ai+aj.
For squares, impose0<=Xii<=1 and Xii>=2t*ai-t² at five fixed t values.
Every true product satisfies these inequalities. The quadratic effects
e_o=-g_o.a-0.5*a^T H_o*a become linear in the lifted variables. Retain the
selector's per-input |e_modal|<=0.08*sign*e_number constraints.

Maximizing each input's contribution to the cell's number-retention statistic
over this larger feasible set yields an upper bound. For any nonnegative dual
multiplier y on Az<=b, c.z<=b.y+max_box(c-A^T y).z; the implementation evaluates
that box support explicitly rather than trusting solver success alone.
These are floating-point bounds, not interval-arithmetic certificates.

All288LPs complete; maximum primal violation6.67e-16 and dual gap3.56e-15.
Known linear/square optima and product-envelope planted controls pass exactly.
The initial linear control incorrectly ignored the sign constraint; correcting
its reference orientation restored the intended analytic optimum before use.

| Remaining cell (all congruent distractor edits) | Selected quadratic retention | Relaxed upper bound | Native retention |
|---|---:|---:|---:|
| earlier_noticed, singular |0.642|1.109|0.641|
| after_speaking, singular |0.768|0.860|0.763|
| after_speaking, plural |0.572|0.680|0.575|

The last cell cannot reach0.8 **within the registered quadratic selector's
feasible set**, up to reported numerical precision. The other bounds are
inconclusive. This is not a native impossibility result: the surrogate's8%
per-input constraints differ from native10% aggregate cell constraints, and
finite Taylor error is not uniformly bounded over the entire amplitude box.

## Circuit interpretation and next discriminating test

Adding the residual complement improves conditional selective control. It has
not established a shared semantic unit, an extracted context generator or
reusable composition. Before attributing the gain to special structure, compare
a fixed five-of-six port interface at matched width. That tests whether a
source substitution can retain the benefit without increasing dictionary size.
Freeze any selection before a new prospective panel; current panels are opened.

Artifacts: `six_source_complement_v1r1_result.json` under the followup directory;
`SIX_SOURCE_UPPER_BOUND_V1_RESULT.json`, `quadratic_edit_upper_bound.py` and
`audit_six_source_upper_bound_v1.py` in this directory.
