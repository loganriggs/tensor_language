# Exact parent reuse through already-required key outputs

14 September2026. This removes a redundant input reader instead of approximating
its coefficients. The original construction in `scalar_joint_key_bank_v1.py`
forms the64-direction basis B inside the256-dimensional joint key row span.
For K=[K1;K2], both full key outputs are already required for native scores and
normalizers. Solve from weights only

$$K^\top C=B,\qquad C\in\mathbb R^{256\times64}.$$

Then compute the same shared input function using existing key outputs:

$$t=xB=[xK_1^\top,xK_2^\top]C,\qquad
A=KB=(KK^\top)C,\qquad \mathrm{inside}_i=tA_i^\top.$$

Store C rather than B. Derive A once at initialization. This is applied to BOTH
original heads, not a projection of the sparse approximation back into the span.
The latter's head9 basis has5.637% residual outside that span; it is not the
exact original basis and is not silently substituted into this identity.

## Price and evidence

| Complete conditional tensor artifact | Serialized bytes |
|---|---:|
| Original exact pair |6,340,557|
| Earlier sparse approximate pair |5,972,357|
| Exact key-coordinate pair |5,414,601|

The exact rewrite is14.60%smaller than the original artifact and9.34%smaller
than the sparse one. At commonFP32, complete tensor payload falls5,523,456→
5,064,704bytes (8.31%). Mixed tensor precision/archive overhead explain why that
percentage differs. Both prepared exact runtimes retain256KiB of derived adapters;
replacing two FP64 B matrices with C saves917,504resident tensor bytes. Other
working allocations, context generators and native suffix are not eliminated.

[Weight and cached-native control](KEY_SPAN_REUSE_V1_RESULT.json) passes:
basis residual<=1.26e-15, scalar error<=2.73e-15. No text fitting occurred.
The candidate was frozen before cached/native scores. It removes two separate
64-wide residual input reads, while retaining64-wide coordinate combinations
of existing key outputs; no coordinate count is advertised as a semantic count.

[Managed native test](KEY_SPAN_NATIVE_V1_RESULT.json) passes A–F in18.28seconds,
632bodyforwards. Across1200live scalar calls, error versus the original is
<=3.67e-15. Regional/newline historical score errors are3.64e-7/9.12e-7, preserving
capability, joint/separate removal, donor, self-donor and newline-control bars.
This reused72regional/32newline panel is not new corpus/OOD evidence. Exactness
refers to the selected original circuit, not a claim that its control effects vanish.

[CPU runtime](KEY_SPAN_RUNTIME_V1_RESULT.json): versus naive original1.13–1.72×;
versus the already-shared/cached exact implementation0.999–1.071×. The registered
>=1.03incremental speedup in ALL eight cells FAILS on two batch1/length16 cells.
The [executed kernel countercheck](KEY_SPAN_KERNEL_V1_RESULT.json) finds the changed
read itself faster in both short cases, but saves only about13–23microseconds,
2.48–4.33%of prior whole-call time. This supports a small-stage/overhead explanation,
not a repaired runtime pass. No whole-model or GPU-speed improvement is established.

The result improves the conditional storage/fidelity comparison: an exact program
is smaller than the prior approximate one. Full-model simplification, autonomous
input generation and broad OOD/selective/reuse evidence remain unfinished.
