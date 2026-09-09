# How does the final reader use a join write?

Matched gate transport failed, with sign flips hidden by aggregate score norms.
Do not fit a sign/gain correction. Change the question to the native downstream
operation on the join write. Earlier independent source-port experiments motivate
linear use of the forward value write and quadratic use of the backward key-pair
write. These are fixed hypotheses, not choices made after inspecting coefficients.

Use all opened32 worlds and six binding orders from the base cases of
suffix_join_middle_match_reference.py, with the original H1-forward/H2-backward
pair masks. Fork each into all four query hops:768 requests. Extract the exact
L2 selected-source write d with join_contribution_context_reference.contributions.
The selected binding positions precede the final query. Their post-L2 states
under write scale alpha are z(alpha)=z_cut+alpha*d; the final query is unchanged.

With native query projections fixed, the final read of each changed source is

    g(alpha)^3 * (C0 + alpha*C1 + alpha²*C2 + alpha³*C3),

where g(alpha)=RMS_gain(z(alpha)). Each coefficient is a full29-output vector,
computed from the actual two key maps, value map and folded output reader.
RMS is retained, not clamped. Store the source norm as
`mean(d²)*(alpha+shift)² + mean((z_cut-shift*d)²) + epsilon`, with the zero-d
case handled explicitly. This avoids cancellation in a naive quadratic form.

A exact response: at alpha=-1,0,.5,1,2, the compiled curve must agree with
both the exact physical source-edit executor and native L2 attention-cell
scaling, <=1e-9 absolute/1e-10 relative RMS. Include all requests, finite/live
controls, reject writes touching the final query, and replay native alpha1.
The target here is the final query distribution, not every changed source's
own output. No approximation claim is made for those other outputs.

B degree hypothesis: retain C0 and the actual normalizer plus C1 for forward
H1, or C2 for backward H2. At every nonzero registered alpha, compare its full
centered query-effect vector relative to alpha0 with the exact native effect.
Require group relative RMS<=.01 with denominator max(native effect RMS,1e-6),
separately for population/orientation/hop/alpha. Also report query KL and all
degree contributions. A failure closes this fixed degree-only read; do not
choose a different degree, scale range, head, order or task subset afterward.

C removal partition: native alpha1-minus-alpha0 decomposes exactly into
`(g1³-g0³)C0 + g1³*C1 + g1³*C2 + g1³*C3`, summed over changed sources. Require
the same1e-9/1e-10 correspondence. These are numerator-degree attributions,
not independently manipulable physical circuit variables or variance fractions.

This is a token-derived exact response program and a bounded read-operation
screen. It retains the full native prefix and387968 export coefficients. Charge
the per-query coefficients, norm scalars, fixed background and compilation cost;
no structural model reduction or independent circuit identification follows
from response closure alone. B4FP64,1800s,256MiB per new tensor; managedGPU only.
