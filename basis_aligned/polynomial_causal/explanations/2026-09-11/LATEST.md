# Latest research update

**11 September, 04:06 UTC:** [MLP16 producer folding and the sharing/routing tradeoff](explanation_2026-09-11_0406.md).

Frozen QK/OV features show weak producer sharing:5.81%overlap versus5.33%rotated
controls; strong-sharing predictions missed. Exact joint key-feature selection
raises overlap to33.63%, but jointQKcoverage falls42.03%to16.63%on8fixedpositions.
No text or activation fitting. No semantic circuit identification.

Joint optimization is now preregistered: equal weights after normalizing both
objectives by their attainable maxima, four starts per head, and a1e-7gradient
convergence bar. Standard Pymanopt solver controls passed. Native GPU matrix
preparation is queued; CPU fitting is ready but has not started.
[Fit protocol](../../COUPLED_PRODUCER_NATIVE_V1_PREREGISTRATION.md).

[Previous reader-family bounds](explanation_2026-09-11_0352.md) ·
[Methods and assumptions](explanation_2026-09-11_0022.md).
