# Native two-QK routing signatures with normalization

Preselected attention17heads2/3 dominate the first routing-contrast coefficient
mode; their specialist role is already in the module dossier. This test asks
whether their routing functions are shared. It does not rediscover their task,
attribute it to one QK branch, or infer routing from the OV output alone.

Compare all9heads' joint QK1×QK2 numerator tensors and complete squared query/key
normalizer polynomials. Query position8; source positions8,7,0, using the native
half-split rotary convention and BF16-rounded trigonometric tables. Norms are
before rotary, with epsilon=float32 eps. Fold each branch's numerator and Gram
with the native1/128scale, so the normalizer factors are x^T G x+epsilon.

Compute exact coefficient Gram matrices using the controlled separate-query/key
biquadratic and fully symmetric quartic contractions. Normalizer Gram includes
degree4, degree2and constant terms. Best proportional squared error between
two coefficient functions is1-cosine². Approximate agreement in this norm is
only a nomination, not a bound on normalized scores where denominators are small.
Exact common numerators alone are explicitly insufficient.

Also run32fixed continuous query/source probes, CPU seed712091. RMS-normalize
each Gaussian residual with native epsilon; the same query is used at both
source positions. These are algebraic inputs, not FineWeb/Pile examples or
proof of text reachability. The independent direct projected-head/RMS/rotary
calculation checks folded real-arithmetic scores. Nonproportional routing rows
on these probes can falsify a global identity on this formal domain.

- A: existing dense Gram controls; finite Gram/cosines, no negative Gram
  eigenvalue fraction below-1e-10, and folded/direct normalized score relative
  errors<=1e-10.
- B: head2/3 best proportional coefficient squared error<=0.10for the joint
  numerator at all three registered positions and both denominator polynomials.
- C: head2/3 two-source routing sine<=0.10for all32queries. For row vectors
  a=(score_to7,score_to0), b similarly, sine=|a1b2-a2b1|/(||a||||b||).
  A nonzero determinant rejects universal proportional routing on the formal
  domain, not possible task-specific agreement on natural text.

Remaining head pairs are descriptive, not post-selected promoted circuits.
Preserve native weights, query/key norms, rotary position and value/residual
interfaces. FP64 arithmetic with native table rounding; no assertion of bitwise
float32-forward equality. Managed GPU,900second alarm, zero body forwards and
zero corpus access. Red-team negatives against formal-input versus natural
scope and coefficient metric; do not infer absence of shared subterms.
