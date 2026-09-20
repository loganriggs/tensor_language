# Attention restoration: exact fold, modest causal recovery

The v656 compiler preserves both QK products, causal diagonal, first-layer value
mixing, RoPE and all RMS epsilon terms. For raw residual x=b+Pz,
RMS(Q RMS(x)) = Qx / sqrt(mean((Qx)^2) + eps_head*(mean(x^2)+eps_input)).
This permits exact quadratic norm and score numerators in the shared coordinates;
normalization stays explicit. P and its encoder need not be orthogonal. All four
Q/K slots and the current value receive the same coordinates. No softmax or row
renormalization is introduced. Context-specific score storage grows as T^2 r^2,
so this compiler is not by itself a cheap general sequence program.

Independent small CPU tests use nonorthogonal factors, non-negligible epsilon,
rounded rotary factors, multiple edit amplitudes and causal-prefix checks.
v656 native attention12 replay has maximum relative output error 3.06e-7 and
finite-response error 6.35e-6; causal-prefix difference is zero.

v657 compares full native effects, six frozen attention writes, native attention12
restoration and projected attention12 restoration, with every suffix MLP dense.
On the four postposed cells, native restoration reduces frozen-model error by
17.9–24.3%, **not the preregistered 50%**. The negative gate is preserved.
The folded arm has 2.7–8.6% full-native effect error across all eight cells and
passes its 10% gate. Baseline replay error is zero. CPU recomputation from saved
arrays independently agrees with both gate verdicts.

Positive red-team: these are 48 already opened rows; all MLPs remain dense and
native contexts remain charged. The folded-versus-native12 discrepancy is only
0.13–0.54% of the full effect norm, but that does not prove selective removal,
reuse, fresh OOD transfer, or a smaller closed circuit. In particular this is NOT
a test of combining the fold with the v654 reduced MLP chain.

Negative red-team: exact local replay, exact baseline replay, independent score
recomputation and agreement with the native12 arm argue against a scoring or
local contraction bug causing the missed 50% prediction. They do not establish
an impossibility theorem. Attention12 contributes only part of the omitted
response under this intervention; dependencies among later responses remain.

Next discriminating question: does installing this response into the reduced
chain preserve its effect, or does it leave the chain's calibrated response space?
That composition needs its own paired endpoint comparison before a new claim.

Primary receipts: [v656](../bilinear_quotient/circuits/followups/subject_projected_attention_v656_result.json),
[v657](../bilinear_quotient/circuits/followups/subject_attention_freeze_v657_result.json),
[independent CPU audit](ATTENTION_RESTORATION_CPU_AUDIT_2026-09-20.json).
