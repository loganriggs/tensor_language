# Locate the preserved CE composition failure

Remainder/scalar regional composition passes, but the first natural half's CE
composition error is 5.458%, above its 5% bar. Preserve this failure. Capture
native pre-softcap logits for all four corners N/S/R/E on the same 32 natural
prefixes, using fixed physical removals and weights: 128 body forwards, 120s,
managed lane1. No candidate fitting or new validation panel.

Let r be raw logits, T(r)=30 tanh(r/30), L(z)=logsumexp(z)-z[newline].
Use r_add=r_S+r_R-r_N and z_add=T(r_S)+T(r_R)-T(r_N). The exact signed identity is

\[
L(T(r_E))-L(T(r_S))-L(T(r_R))+L(T(r_N)) =
[L(T(r_E))-L(T(r_{add}))] +
[L(T(r_{add}))-L(z_{add})] +
[L(z_{add})-L(T(r_S))-L(T(r_R))+L(T(r_N))].
\]

These three terms distinguish raw-score/final-RMS interaction, softcap curvature,
and logsumexp curvature. The split is ordered; contributions can cancel and their
signed projection fractions are not nonnegative shares.

A: native CE for all four corners matches prior fixed receipts <=1e-5 relative.
B: FP64 analytic identity replay <=1e-10 relative. C: the first term's L2 norm
<=20% of total CE interaction norm in each half. This opposing prediction tests
whether the metric stages largely explain the failure; a miss leaves genuine
raw-score interaction. Analyze stored native FP32 raw logits in FP64, separately
from the original native CE gate. No analytical result retroactively passes C.
