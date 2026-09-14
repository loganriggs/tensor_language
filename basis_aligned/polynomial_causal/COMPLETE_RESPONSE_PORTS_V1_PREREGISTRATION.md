# Complete composed response with contracted input ports

Retain the full pristine MLP compensation term, unlike the earlier simplified
response_ports_v1. Project native z,h,u to1225scalars pertoken once, then evaluate
the unchanged rank64-composed response at arbitrary registered amplitudes.
Only declared ports, amplitude and compiled weights enter execute; no ambient
vector or model callback is allowed. Native prefix generation remains external.

Test all72cached contexts at strengths[-2,-1,0,.5,1,2]. A: complete scalar,
baseline-subtracted own amplitude change, and squared norm versus existing
complete features/scalar all<=1e-10relative (zero reference exact within1e-12).
B: inputportcount<=40%of3456ambient scalars pertoken. Charge projection matrices,
runtime matrices/Gram/adapters and regenerated caches separately. Missing u0ports
must fail instead of silently using the old approximation. CPU120seconds/two
threads; no native bodyforwards, fitting, fresh/OOD or autonomous extraction claim.
