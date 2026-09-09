# Does the shared-value response transport unchanged across command settings?

The value factorial shows small paired answer/foil effects from shared-value
removal, but full-distribution KL remains above the replacement bar. Two
different explanations remain: a largely command-independent background
response, or task-dependent changes hidden outside the measured answer pair.
Test full-vocabulary response transport, not a rank/variance compression.

Use the unchanged32-world dual-command-v2 cohort and fixed four-head union.
Parent BILIN18_VALUE_COMPONENT_FACTORIAL_V1_RESULT.json SHA256
bbffa1dd0665455e84958fbf4b26ecf50623d53923f3eaaf26e3ce8e3ab25a54.
All texts and shared-removal interventions are opened, but their full-vector
four-cell decomposition is not. Cell lengths and both query positions have
been verified identical within every world. No padding or row selection.

For each world define E_ab(t)=center(logits_without_S-logits_native) at
the same token position and in the same50304-output frame, a=temporal bit,
b=iswas bit. Compute the balanced modes
M_uv=(1/4) sum_ab (-1)^(u*a+v*b) E_ab, u,v in {0,1}.
Retain all four modes, with explicit bit labels rather than relying on the
different cell orders in old utilities. Reconstruction and Parseval identity
are numerical checks, not semantic evidence. Prior joint_command_composition
uses anchored differences and casts FP32; use a small dtype-preserving
balanced transform, without changing the existing immutable experiment code.

For ANY response B independent of both command bits, separately within each
world and token, sum_ab ||E_ab-B||² = sum_ab ||E_ab-M00||²+4||B-M00||².
Thus the mean is the best possible such predictor, even allowing arbitrary
world/token-dependent B. The pooled minimum relative RMS error is
sqrt(sum ||M10||²+||M01||²+||M11||²)/sqrt(sum_all_modes ||M||²).
Use total effect RMS floor1e-6, retain raw denominators. This exact real
identity is evaluated FP64 on stored native FP32 logits, not an ideal-real
network interval certificate or KL lower bound. It allows a fully general
world-dependent baseline, so failure rules out any stricter command-invariant
response program on these observed worlds. Passing supplies no token-to-B
algorithm and earns no structural saving or extraction claim.

A: authority/source hashes, complete rows, native/shared-cut replay of parent
pooled KL at1e-5, finite outputs, exact cell/position pairing,64 forwards.
Transform controls: constant positive, live command/mixed effects, bit-flip
sign convention, reconstruction, Parseval, and best-constant Pythagorean identity.
In the trained run, reconstruction abs<=1e-9/relative<=1e-10 using live sums.
The future iswas bit must have zero M01/M11 at the earlier temporal query:
absolute<=1e-8. This is a live causality/alignment tripwire, not a semantic pass.

B command-independent response: best relative error<=.01 separately in each
FIT/HOLDOUT phase for all tokens, temporal-query outputs and iswas-query
outputs. Keep all four modes and every template in the receipt. If B fails,
do not remove a command, choose a token/head subset, or fit a new baseline
to rescue it. The control invariant concerns removal effects, not equality
of the original logits across different commands.

GPU only via bqrunner, batch4,length<=27,64forwards/256sequence evaluations,
no fitting/gradients/training, alarm600s, analysis tensors<256MiB. Retain
only per-world mode norms and closure summaries on CPU. Full545902902 native
parameters remain priced. This is a circuit response-transport falsifier,
not a deployable averaged-logit correction or a new replacement nomination.
