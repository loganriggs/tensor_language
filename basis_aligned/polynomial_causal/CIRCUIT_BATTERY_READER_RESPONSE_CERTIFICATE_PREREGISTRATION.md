# CIRCUIT BATTERY — READER RESPONSE CERTIFICATE (preregistration)

Registered Registered 2026-09-04 04:10Z (box clock) (box clock). Claude, LANE 1 CUDA. Rung `circuit_battery_reader_response_certificate`.
Script: `ops/circuit_battery_reader_response_certificate.py`. Input receipt: `circuit_battery_results.json`
(SS2809, sha 6d1eda1cc05adf72c525375a0602bbafbf9b4335653be0e410de3d69da03265c). Move 1 of MATHEMATICAL_REVIEW_2026-09-04_0404.

## The identity being tested

bilin18's block MLP is `Bilinear` with `gated=False` and `squared_mlp=False` (checked in the checkpoint config, not assumed):
`mlp(u) = Down(Left(u) * Right(u)) + b = Q(u) + b`, with Q homogeneous of degree 2. RMSNorm is `u -> sqrt(D) u / ||u||`. Hence for a
writer's final-position write W and a reader whose input residual at the final position is x, the path-patched arm at removal fraction
t is EXACTLY

    mlp(rms_norm(x - tW)) - b  =  D * [ Q(x) - t B(x,W) + t^2 Q(W) ] / [ ||x||^2 - 2t<x,W> + t^2||W||^2 ]

with `B(x,W) = Down(Left(x)*Right(W) + Left(W)*Right(x))`. Three vectors and three scalars determine the entire removal curve. The
numerator splits the reader's use of W into a CROSS term `B(x,W)` (linear in W: the reader reading W against its context) and a SELF
term `Q(W)` (quadratic in W: the reader squaring W alone); the denominator is a pure RMSNorm GAIN channel that attribution methods
ignore. This is a decomposition strictly finer than an MLP block, in closed form, with no fitted parameters.

Scope: SS2809's capable behaviours whose writer localises (REC >= .50), their top-3 MLP readers from the SS2809 ladder, A1 rows on the
frozen SELECT split, t in {.25, .5, .75, 1}.

## Predictions

```
BARS  = {exact_rel: 1e-5, cross_frac: .70, gain_share: .25, linear_err: .25, half_gap: .10, floor: .5}
NULLS = {cross_frac_le: .50, gain_share_ge: .50, linear_err_le: .10, half_gap_le: .05}
```

**pred_a_rational_identity_is_exact** — max over tasks, readers, rows and t of the relative deviation between the closed-form
prediction and the actual `mlp(rms_norm(x - tW))` is <= 1e-5.
*Worked example:* the identity is exact algebra, so the hypothesis reads float round-off, ~1e-7 in fp32; if the block had ANY
non-polynomial nonlinearity on the hidden units (a relu-square MLP, a silu gate) the deviation would be O(1e-1). Both operands are
non-negative magnitudes; the denominator is a max-abs activation floored at 1e-6.

**pred_b_cross_term_carries_the_read** — median over readers of `||B(x,W)|| / (||B(x,W)|| + ||Q(W)||)` is >= .70.
*Worked example:* if the reader reads W against its context (the "bilinear read" picture) the cross term dominates and this reads
.80-.95; if the reader mostly squares W on its own, .10-.30. Null: <= .50. Both operands are norms, strictly non-negative, so the
denominator cannot change sign or pass through zero (floored at 1e-9 anyway).

**pred_c_gain_channel_is_small** — median over readers of
`|| (numerator frozen at t=0, denominator at t=1) - (native) || / || (t=1 actual) - (native) ||` is <= .25, i.e. at most a quarter of
the reader's response to a full removal is the RMSNorm rescaling rather than the direction change.
*Worked example:* `||W||` is a small fraction of `||x||` at the reader (SS2808 measured the write's norm well below the residual's), so
the hypothesis reads .02-.15; if the write were a large share of the residual norm, .5-1.0. Numerator and denominator are both norms
of differences (non-negative); denominator floored at 1e-9.

**pred_d_linear_attribution_is_materially_wrong** — median over rows of
`|4 * d_m(t=.25) - d_m(t=1)| / max(|d_m(t=1)|, .5)` is >= .25: the linear extrapolation from a quarter-removal (what attribution
patching / EAP effectively assumes, arXiv:2310.10348) misses the true full-removal damage by at least a quarter of it.
*Worked example:* a truly linear response reads ~.02; the (2,2)-rational response with a non-negligible `t^2 Q(W)` and a moving
denominator reads .3-.8. Operands are damages in margin units and MAY be negative, so the expression is an absolute difference over a
FLOORED absolute denominator — never a ratio with a sign-changing denominator. Null: <= .10.

**pred_e_half_removal_is_not_half_damage** — median over rows of `|d_m(t=.5) - .5 * d_m(t=1)| / max(|d_m(t=1)|, .5)` is >= .10.
*Worked example:* linear response ~.01; the measured curvature should put this at .1-.4. This is the behaviour-level (logit margin)
consequence of pred_a's block-level identity, and it is the one that constrains how a compiled program may approximate an edge.
Null: <= .05.

## Stated null

The reader response is effectively linear in the removal fraction and the write is consumed as a self-term: cross fraction <= .50,
gain share >= .50, linear extrapolation error <= .10, half-removal gap <= .05. Each null is reported separately.

## Price

<= 6 behaviours x 3 readers x (1 instrumented forward + 5 margin-level sweep forwards) per length-batch, 16 A1/SELECT rows per
behaviour. Literal budget: <= 400 GPU forwards, 0 backwards, 0 fitted parameters, expected < 60 GPU-seconds.

## What this does NOT claim

The identity is exact only for a SINGLE reader's arm, where the rest of the residual is held fixed. Joint removals change each
reader's input and are therefore NOT covered — that non-additivity is precisely SS2808/SS2809's super-additivity, and the Moebius
interaction transform (move 2 of the same review) is the separate rung that measures it.
