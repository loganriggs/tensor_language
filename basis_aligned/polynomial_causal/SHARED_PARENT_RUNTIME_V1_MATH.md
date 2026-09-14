# Shared parent runtime: exact local reuse, not new compression

14 September2026. Apply the established [shared key read](SHARED_KEY_READS_V1_MATH.md)
to the portable sparse reflection-even pair, whose scalar loop still repeated it.
For each head, compute once

$$t=xB,\qquad A_i=K_iB,\qquad \mathrm{inside}_i=tA_i^\top,\quad i=1,2.$$

The input read has64coordinates; both128-coordinate key consumers reuse it.
The adapters are prepared once. Original FP32 key/query normalizers, FP64
numerators, rounded rotary coefficients, values and background remain unchanged.
No weights are fitted. This is known algebra applied to a previously redundant
executable, not a new semantic discovery or smaller serialized program.

[CPU screen](SHARED_PARENT_RUNTIME_V1_RESULT.json): exact replay on two cached
native contexts and independent inputs for both heads; length32 speedup1.57/1.21
at batch1/8. Preparation1.34ms, extra resident adapters262144bytes, no additional
serialized weights. Per head call, sharing saves batch×length×1152×64 MACs;
caching separately moves2×128×1152×64 MACs from every call to initialization.

[Boundary discriminator](SHARED_PARENT_RUNTIME_BOUNDARY_V1_RESULT.json): all
eight head/batch/length cells replay exactly. Total speedup1.117–1.723 at
batch1/8,length16/128. Comparing against cached-adapters-only isolates1.032–1.073
speedup from sharing the input read itself. Seven interleaved trials per cell;
these are local CPU timings, not full-model or GPU speed claims.

[Managed native validation](SHARED_PARENT_NATIVE_V1_RESULT.json) completed632
forwards in18.56seconds; inherited capability, removal, donor, control and
historical replay A–E all pass. [Matched scalar replay](SHARED_PARENT_NATIVE_V1_REPLAY.json)
is exact for all1200actual pristine/edited-context calls (F passes).
This confirms reuse preserves the existing approximate conditional circuit.
It does not add unseen-text/OOD evidence: the72regional/32newline panel is reused.

The [exception audit](SHARED_PARENT_EXCEPTION_V1_RESULT.json) preserves eight
prompt/arm/readout cells with >10%error against the uncompressed pair, including
one donor-control sign reversal at row54 (-0.0001230→+0.0002055). The shared
rewrite is exact against the sparse implementation, not against the uncompressed
model. All context generators, native suffix and original sparse limitations remain.
